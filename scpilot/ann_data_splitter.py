import torch
import anndata as ad
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Union
from math import ceil, floor
from lightning import pytorch as pl
class AnnDataset(Dataset):
    def __init__(
        self,
        adata: ad.AnnData,
    ):
        super().__init__()
        self.adata = adata
    def __len__(self):
        return self.adata.shape[0]
    def __getitem__(
        self,
        indices: Union[int, list[int], slice]
    ) -> dict[str, Union[np.ndarray, torch.Tensor]]:
        if isinstance(indices, int):
            indices = [indices]
        if isinstance(indices, (list, np.ndarray)):
            indices = np.sort(indices)
        return torch.tensor((self.adata.X.toarray() if hasattr(self.adata.X, 'toarray') else self.adata.X)[indices])
def validate_data_split(n_samples: int, train_size: float, validation_size: Optional[float] = None):
    if train_size > 1.0 or train_size <= 0.0:
        raise ValueError('Invalid train_size. Must be: 0 < train_size <= 1')
    n_train = ceil(train_size * n_samples)
    if validation_size is None:
        n_val = n_samples - n_train
    elif validation_size >= 1.0 or validation_size < 0.0:
        raise ValueError('Invalid validation_size. Must be 0 <= validation_size < 1')
    elif (train_size + validation_size) > 1:
        raise ValueError('train_size + validation_size must be between 0 and 1')
    else:
        n_val = floor(n_samples * validation_size)
    if n_train == 0:
        raise ValueError(
            f'With n_samples = {n_samples}, train_size = {train_size} and validation_size = {validation_size}, the '
            'resulting train set will be empty. Adjust any of the aforementioned parameters.'
        )
    return n_train, n_val

def adjust_train_val_counts(
    n_train: int,
    n_val: int,
    batch_size: int,
):
    """Adjust train/validation counts to avoid singleton final mini-batches."""
    if batch_size is None or batch_size <= 1:
        return n_train, n_val

    # delta > 0: move cells from validation to training
    # delta < 0: move cells from training to validation
    for delta in [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5]:
        new_n_train = n_train + delta
        new_n_val = n_val - delta

        if new_n_train <= 0 or new_n_val < 0:
            continue

        train_ok = new_n_train % batch_size != 1
        val_ok = (new_n_val == 0) or (new_n_val % batch_size != 1)

        if train_ok and val_ok:
            return new_n_train, new_n_val

    raise ValueError(
        f"Could not avoid singleton final batches with "
        f"n_train={n_train}, n_val={n_val}, batch_size={batch_size}."
    )

def stratified_train_val_split(
    adata: ad.AnnData,
    n_train: int,
    n_val: int,
    stratify_keys: list[str],
    random_state: Optional[int] = None,
    min_val_per_group: int = 1,
):
    rng = np.random.RandomState(random_state)

    obs = adata.obs.copy()
    group_labels = (
        obs[stratify_keys]
        .astype(str)
        .agg("||".join, axis=1)
        .to_numpy()
    )

    train_idx = []
    val_idx = []

    unique_groups = np.unique(group_labels)

    for group in unique_groups:
        group_idx = np.where(group_labels == group)[0]
        group_idx = rng.permutation(group_idx)

        n_group = len(group_idx)

        if n_group <= 1:
            train_idx.extend(group_idx.tolist())
            continue

        group_val_frac = n_val / (n_train + n_val)
        n_group_val = int(round(n_group * group_val_frac))

        if n_group_val == 0 and n_group >= 2:
            n_group_val = min_val_per_group

        if n_group_val >= n_group:
            n_group_val = n_group - 1

        val_idx.extend(group_idx[:n_group_val].tolist())
        train_idx.extend(group_idx[n_group_val:].tolist())

    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)

    train_idx = rng.permutation(train_idx)
    val_idx = rng.permutation(val_idx)

    return train_idx, val_idx

class AnnDataSplitter(pl.LightningDataModule):
    def __init__(
        self,
        adata: ad.AnnData,
        batch_size: float = 32,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        shuffle_set_split: bool = True,
        pin_memory: bool = False,
        random_state: Optional[int] = None,
        avoid_singleton_batch: bool = True,
        stratify_keys: Optional[list[str]] = None,
        **kwargs,
    ):
        super().__init__()
        self.adata = adata
        self.batch_size = int(batch_size)
        self.anndataset = AnnDataset(adata)
        self.train_size = float(train_size)
        self.validation_size = validation_size
        self.shuffle_set_split = shuffle_set_split
        self.random_state = random_state
        self.avoid_singleton_batch = avoid_singleton_batch
        self.data_loader_kwargs = kwargs
        self.pin_memory = pin_memory
        self.stratify_keys = stratify_keys

        self.n_train, self.n_val = validate_data_split(
            self.adata.n_obs,
            self.train_size,
            self.validation_size,
        )

        if self.avoid_singleton_batch:
            old_n_train, old_n_val = self.n_train, self.n_val

            self.n_train, self.n_val = adjust_train_val_counts(
                n_train=self.n_train,
                n_val=self.n_val,
                batch_size=self.batch_size,
            )

            print(
                f"[AnnDataSplitter] n_obs={self.adata.n_obs} | "
                f"train={self.n_train} | val={self.n_val} | "
                f"train_mod_batch={self.n_train % self.batch_size} | "
                f"val_mod_batch={(self.n_val % self.batch_size) if self.n_val > 0 else 0} | "
                f"original_train={old_n_train} | original_val={old_n_val}",
                flush=True,
            )
    def setup(self, stage: Optional[str] = None):
        n_train = self.n_train
        n_val = self.n_val

        if self.stratify_keys is not None and len(self.stratify_keys) > 0:
            self.train_idx, self.val_idx = stratified_train_val_split(
                self.adata,
                n_train=n_train,
                n_val=n_val,
                stratify_keys=self.stratify_keys,
                random_state=self.random_state,
            )

            used = np.concatenate([self.train_idx, self.val_idx])
            all_idx = np.arange(self.adata.n_obs)
            self.test_idx = np.setdiff1d(all_idx, used)

        else:
            indices = np.arange(self.adata.n_obs)

            if self.shuffle_set_split:
                random_state = np.random.RandomState(self.random_state)
                indices = random_state.permutation(indices)

            self.val_idx = indices[: n_val]
            self.train_idx = indices[n_val: (n_val + n_train)]
            self.test_idx = indices[(n_val + n_train):]
    def train_dataloader(self):
        return DataLoader(
            self.anndataset[self.train_idx],
            batch_size = self.batch_size,
            shuffle = True,
            drop_last = False,
            pin_memory = self.pin_memory,
            **self.data_loader_kwargs,
        )
    def val_dataloader(self):
        if len(self.val_idx) > 0:
            return DataLoader(
                self.anndataset[self.val_idx],
                batch_size = self.batch_size,
                shuffle = False,
                drop_last = False,
                pin_memory = self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass
    def test_dataloader(self):
        if len(self.test_idx) > 0:
            return DataLoader(
                self.anndataset[self.test_idx],
                batch_size = self.batch_size,
                shuffle = False,
                drop_last = False,
                pin_memory = self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass