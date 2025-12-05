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
class AnnDataSplitter(pl.LightningDataModule):
    def __init__(
        self,
        adata: ad.AnnData,
        batch_size: float = 32,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        shuffle_set_split: bool = True,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.adata = adata
        self.batch_size = batch_size
        self.anndataset = AnnDataset(adata)
        self.train_size = float(train_size)
        self.validation_size = validation_size
        self.shuffle_set_split = shuffle_set_split
        self.data_loader_kwargs = kwargs
        self.pin_memory = pin_memory
        self.n_train, self.n_val = validate_data_split(self.adata.n_obs, self.train_size, self.validation_size)
    def setup(self, stage: Optional[str] = None):
        n_train = self.n_train
        n_val = self.n_val
        indices = np.arange(self.adata.n_obs)
        if self.shuffle_set_split:
            random_state = np.random.RandomState()
            indices = random_state.permutation(indices)
        self.val_idx = indices[: n_val]
        self.train_idx = indices[n_val: (n_val + n_train)]
        self.test_idx = indices[(n_val + n_train): ]
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