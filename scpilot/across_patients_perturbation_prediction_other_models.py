import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import random
import argparse

import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import pairwise
from scgen import SCGEN
from biolord import Biolord
from matplotlib import pyplot as plt
from lightning import pytorch as pl
from scpilot.egd_model import EGD_model


parser = argparse.ArgumentParser(description='across_patients_perturbation_prediction_other_models')
parser.add_argument(
    '--model_name',
    type=str,
    default='biolord',
    help='biolord, identity, scGen',
)
parser.add_argument(
    '--query_key',
    type=int,
    default=101,
    help='101, 107, 1015, 1016, 1039, 1244, 1256, 1488',
)
parser.add_argument(
    '--seed',
    type=int,
    default=1327,
    help='Training random seed. Recommended seeds: 1327, 1337, 1347.',
)
parser.add_argument(
    '--split_seed',
    type=int,
    default=0,
    help='Internal train/validation split seed. Keep fixed to match the main scPILOT/CellOT experiment design.',
)
args = parser.parse_args()


def set_seed(seed: int, deterministic: bool = True):
    """Set random seeds for reproducible model initialization, minibatch order, and inference."""
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    pl.seed_everything(seed, workers=True)

    try:
        import scvi
        scvi.settings.seed = seed
    except Exception:
        pass

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def ensure_dirs(*paths: str):
    for path in paths:
        os.makedirs(path, exist_ok=True)


def to_dense_array(x):
    return x.toarray() if hasattr(x, 'toarray') else x


def obs_value_mask(adata, obs_key, value):
    """Robust mask for numeric or string-like sample IDs."""
    return adata.obs[obs_key].astype(str) == str(value)


def mmd_distance(x, y, gamma):
    xx = pairwise.rbf_kernel(x, x, gamma)
    xy = pairwise.rbf_kernel(x, y, gamma)
    yy = pairwise.rbf_kernel(y, y, gamma)
    return xx.mean() + yy.mean() - 2 * xy.mean()


def compute_mmd_loss(lhs, rhs, gammas):
    return np.mean([mmd_distance(lhs, rhs, g) for g in gammas])


def make_prediction_obs_names_unique(adata_pred, model_name: str):
    adata_pred.obs_names = [
        f'{idx}_{model_name}_pred'
        for idx in adata_pred.obs_names.astype(str)
    ]
    return adata_pred


def adjust_train_valid_counts(
    n_total: int,
    train_size: float = 0.9,
    batch_size: int = 64,
):
    """Choose deterministic train/valid counts while avoiding singleton final mini-batches."""
    n_train0 = int(np.floor(train_size * n_total))

    for delta in [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5]:
        n_train = n_train0 + delta
        n_valid = n_total - n_train

        if n_train <= 0 or n_valid <= 0:
            continue

        if (n_train % batch_size != 1) and (n_valid % batch_size != 1):
            return n_train, n_valid

    raise ValueError(
        f"Could not find a stable train/valid split for "
        f"n_total={n_total}, train_size={train_size}, batch_size={batch_size}."
    )


def fixed_shuffle_adata(adata, split_seed: int):
    """Return a copy with deterministic order used by scvi's shuffle_set_split=False."""
    rng = np.random.RandomState(split_seed)
    indices = rng.permutation(adata.n_obs)
    return adata[indices].copy()


def add_fixed_biolord_split_column(
    adata,
    query_key,
    cond_key: str,
    stim_key: str,
    cell_label_key: str,
    split_seed: int = 0,
    train_size: float = 0.9,
    batch_size: int = 64,
    split_key: str = '_biolord_split',
):
    """
    Add a deterministic BioLORD split column.

    Held-out stimulated cells of the query patient are assigned to 'test' and are
    excluded from both training and validation. All remaining cells are split into
    fixed internal train/valid sets using split_seed.
    """
    adata = adata.copy()

    heldout_mask = (
        (adata.obs[cell_label_key].astype(str) == str(query_key))
        & (adata.obs[cond_key].astype(str) == str(stim_key))
    ).to_numpy()

    pool_indices = np.where(~heldout_mask)[0]
    rng = np.random.RandomState(split_seed)
    shuffled_pool_indices = rng.permutation(pool_indices)

    n_train, n_valid = adjust_train_valid_counts(
        n_total=len(shuffled_pool_indices),
        train_size=train_size,
        batch_size=batch_size,
    )

    train_indices = shuffled_pool_indices[:n_train]
    valid_indices = shuffled_pool_indices[n_train:n_train + n_valid]
    test_indices = np.where(heldout_mask)[0]

    split = np.empty(adata.n_obs, dtype=object)
    split[:] = 'valid'
    split[train_indices] = 'train'
    split[valid_indices] = 'valid'
    split[test_indices] = 'test'

    adata.obs[split_key] = split
    adata.obs['_indices'] = np.arange(adata.n_obs)

    print(
        f'[BioLORD split] split_seed={split_seed} | '
        f'train={np.sum(split == "train")} | '
        f'valid={np.sum(split == "valid")} | '
        f'test={np.sum(split == "test")} | '
        f'train_mod_batch={np.sum(split == "train") % batch_size} | '
        f'valid_mod_batch={np.sum(split == "valid") % batch_size}',
        flush=True,
    )
    return adata


def evaluate_and_save(
    adata_query_ctrl,
    adata_query_stim,
    adata_query_pred,
    top50_genes,
    gammas,
    experiment_name,
    data_file,
    file_type,
    model_name,
    query_key,
    seed,
    split_seed,
    cond_key,
    stim_key,
):
    adata_query_pred.obs[cond_key] = 'pred'
    adata_query_pred = make_prediction_obs_names_unique(
        adata_query_pred.copy(),
        model_name=model_name,
    )

    adata_query_eval = ad.concat([adata_query_stim, adata_query_pred])
    adata_query_eval.obs_names_make_unique()

    plt.figure()
    r2mean_all, r2mean_top50 = EGD_model.reg_mean_plot(
        adata_query_eval,
        cond_key=cond_key,
        axis_keys={'x': 'pred', 'y': stim_key},
        labels={'x': 'Prediction', 'y': 'Ground truth'},
        path_to_save=(
            f'../Figures/{experiment_name}/'
            f'{model_name}_{data_file}_reg_mean_{query_key}_seed{seed}.jpg'
        ),
        gene_list=top50_genes[:10],
        top_genes=top50_genes,
        top_gene_label='T50',
        show=False,
        legend=False,
    )
    plt.close('all')

    x = to_dense_array(adata_query_pred[:, top50_genes].X)
    y = to_dense_array(adata_query_stim[:, top50_genes].X)
    mmd = compute_mmd_loss(x, y, gammas=gammas)

    print(f'{model_name}:')
    print(f'{query_key}: r2mean_all = {r2mean_all}')
    print(f'{query_key}: r2mean_top50 = {r2mean_top50}')
    print(f'{query_key}: mmd = {mmd}')

    adata_query_eval_to_save = ad.concat([
        adata_query_ctrl,
        adata_query_stim,
        adata_query_pred,
    ])
    adata_query_eval_to_save.obs_names_make_unique()
    adata_query_eval_to_save.write_h5ad(
        f'../Result_anndata/{experiment_name}/'
        f'{experiment_name}_{model_name}_{data_file}_{query_key}_seed{seed}{file_type}'
    )

    metrics_df = pd.DataFrame([
        {
            'experiment': experiment_name,
            'data_file': data_file,
            'query_key': query_key,
            'seed': seed,
            'split_seed': split_seed,
            'model': model_name,
            'r2mean_all': r2mean_all,
            'r2mean_top50': r2mean_top50,
            'mmd_top50': mmd,
        }
    ])
    metrics_df.to_csv(
        f'../DataFrames/{experiment_name}/'
        f'{experiment_name}_{data_file}_{query_key}_seed{seed}_{model_name}_metrics.csv',
        index=False,
    )


def predict_perturbation(
    experiment_name='across_patients',
    model_name='biolord',
    data_file='pbmc_patients',
    file_type='.h5ad',
    cond_key='condition',
    ctrl_key='ctrl',
    stim_key='stim',
    cell_label_key='sample_id',
    query_key=101,
    seed=1327,
    split_seed=0,
):
    set_seed(seed)
    ensure_dirs(
        f'../Result_anndata/{experiment_name}',
        f'../Figures/{experiment_name}',
        f'../DataFrames/{experiment_name}',
        f'../model_trained/{experiment_name}',
    )

    adata = sc.read_h5ad(f'../Data/{experiment_name}/{data_file}{file_type}')
    adata.obs_names_make_unique()
    adata = adata[
        (adata.obs[cond_key] == ctrl_key)
        | (adata.obs[cond_key] == stim_key)
    ].copy()

    print(
        f'======Predicting {query_key} with {model_name} | '
        f'seed={seed} | split_seed={split_seed}======',
        flush=True,
    )

    query_mask = obs_value_mask(adata, cell_label_key, query_key)

    train = adata[
        ~(
            query_mask
            & (adata.obs[cond_key] == stim_key)
        )
    ].copy()

    print('train:')
    print(train)

    adata_query_ctrl = adata[
        query_mask
        & (adata.obs[cond_key] == ctrl_key)
    ].copy()

    adata_query_stim = adata[
        query_mask
        & (adata.obs[cond_key] == stim_key)
    ].copy()

    adata_query = adata[query_mask].copy()

    n_top_degs = 50
    sc.tl.rank_genes_groups(
        adata_query,
        groupby=cond_key,
        method='wilcoxon',
        n_genes=n_top_degs,
    )
    top50_genes = adata_query.uns['rank_genes_groups']['names'][stim_key].tolist()
    gammas = np.logspace(1, -3, num=50)

    if model_name == 'biolord':
        biolord_batch_size = 64

        set_seed(seed)
        biolord_split_key = '_biolord_split'
        adata_biolord = add_fixed_biolord_split_column(
            adata,
            query_key=query_key,
            cond_key=cond_key,
            stim_key=stim_key,
            cell_label_key=cell_label_key,
            split_seed=split_seed,
            train_size=0.9,
            batch_size=biolord_batch_size,
            split_key=biolord_split_key,
        )

        Biolord.setup_anndata(
            adata_biolord,
            ordered_attributes_keys=None,
            categorical_attributes_keys=[cond_key, cell_label_key],
        )

        set_seed(seed)
        module_params = {
            'seed': seed,
        }

        model = Biolord(
            adata_biolord,
            n_latent=256,
            split_key=biolord_split_key,
            train_split='train',
            valid_split='valid',
            test_split='test',
            module_params=module_params,
        )

        set_seed(seed)
        model.train(
            max_epochs=300,
            batch_size=biolord_batch_size,
            early_stopping=True,
            early_stopping_patience=20,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )

        set_seed(seed)
        idx_source = np.where(
            (adata_biolord.obs[cell_label_key].astype(str) == str(query_key))
            & (adata_biolord.obs[cond_key].astype(str) == str(ctrl_key))
        )[0]
        adata_source = adata_biolord[idx_source].copy()

        adata_query_pred = model.compute_prediction_adata(
            adata_biolord,
            adata_source,
            target_attributes=[cond_key],
            add_attributes=['cell_type'],
        )

        adata_query_pred = adata_query_pred[
            ~adata_query_pred.obs.index.str.endswith(f'_{ctrl_key}')
        ].copy()
        adata_query_pred.obs.index = adata_query_pred.obs.index.str.removesuffix(f'_{stim_key}')

    elif model_name == 'identity':
        set_seed(seed)
        adata_query_pred = adata_query_ctrl.copy()

    elif model_name == 'scGen':
        scgen_batch_size = 32

        set_seed(seed)
        train_scgen = fixed_shuffle_adata(train, split_seed=split_seed)

        scgen_n_train, scgen_n_valid = adjust_train_valid_counts(
            n_total=train_scgen.n_obs,
            train_size=0.9,
            batch_size=scgen_batch_size,
        )

        print(
            f'[scGen split] split_seed={split_seed} | '
            f'train={scgen_n_train} | valid={scgen_n_valid} | '
            f'train_mod_batch={scgen_n_train % scgen_batch_size} | '
            f'valid_mod_batch={scgen_n_valid % scgen_batch_size}',
            flush=True,
        )

        SCGEN.setup_anndata(
            train_scgen,
            batch_key=cond_key,
            labels_key=cell_label_key,
        )

        set_seed(seed)
        model = SCGEN(train_scgen)

        set_seed(seed)
        model.train(
            max_epochs=100,
            batch_size=scgen_batch_size,
            early_stopping=True,
            early_stopping_patience=25,
            enable_progress_bar=False,
            train_size=scgen_n_train / train_scgen.n_obs,
            validation_size=scgen_n_valid / train_scgen.n_obs,
            shuffle_set_split=False,
        )

        set_seed(seed)
        adata_query_pred, _ = model.predict(
            ctrl_key=ctrl_key,
            stim_key=stim_key,
            celltype_to_predict=query_key,
        )

    else:
        raise ValueError(
            f'Unknown model_name: {model_name}. '
            f'Expected one of: biolord, identity, scGen.'
        )

    evaluate_and_save(
        adata_query_ctrl=adata_query_ctrl,
        adata_query_stim=adata_query_stim,
        adata_query_pred=adata_query_pred,
        top50_genes=top50_genes,
        gammas=gammas,
        experiment_name=experiment_name,
        data_file=data_file,
        file_type=file_type,
        model_name=model_name,
        query_key=query_key,
        seed=seed,
        split_seed=split_seed,
        cond_key=cond_key,
        stim_key=stim_key,
    )


if __name__ == '__main__':
    predict_perturbation(
        model_name=args.model_name,
        query_key=args.query_key,
        seed=args.seed,
        split_seed=args.split_seed,
    )
    print('Done')