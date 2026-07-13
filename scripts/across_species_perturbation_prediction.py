import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import wandb
import argparse
import scanpy as sc
import anndata as ad
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback
from sklearn.metrics import pairwise
from matplotlib import pyplot as plt
from lightning import pytorch as pl
from scpilot.egd_model import EGD_model

parser = argparse.ArgumentParser(description = 'across_species_perturbation_prediction')
parser.add_argument(
    '--query_key',
    type = str,
    default = 'mouse',
    help = 'mouse, pig, rabbit, rat',
)
parser.add_argument(
    '--seed',
    type = int,
    default = 1327,
    help = 'Training random seed. Recommended seeds: 1327, 1337, 1347.',
)
parser.add_argument(
    '--split_seed',
    type = int,
    default = 0,
    help = 'Random seed for the internal train/validation split. Keep fixed to preserve the same data split across training seeds.',
)
args = parser.parse_args()


class EpochProgressPrinter(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        max_epochs = trainer.max_epochs

        metrics = trainer.callback_metrics
        msg = f"[Epoch {epoch}/{max_epochs}]"

        for key in ["VAE_loss_train", "elbo_train", "rl_train", "kld_train", "dl_train",
                    "VAE_loss_val", "elbo_validation", "rl_validation", "kld_validation", "dl_validation"]:
            if key in metrics:
                value = metrics[key]
                try:
                    value = value.item()
                except Exception:
                    pass
                msg += f" {key}={value:.4f}"

        print(msg, flush=True)


def set_seed(seed: int, deterministic: bool = True):
    """Set random seeds for reproducible model initialization, minibatch order, and inference."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers = True)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def ensure_dirs(*paths: str):
    for path in paths:
        os.makedirs(path, exist_ok = True)


def to_dense_array(x):
    return x.toarray() if hasattr(x, 'toarray') else x


def mmd_distance(x, y, gamma):
    xx = pairwise.rbf_kernel(x, x, gamma)
    xy = pairwise.rbf_kernel(x, y, gamma)
    yy = pairwise.rbf_kernel(y, y, gamma)
    return xx.mean() + yy.mean() - 2 * xy.mean()


def compute_mmd_loss(lhs, rhs, gammas):
    return np.mean([mmd_distance(lhs, rhs, g) for g in gammas])


def predict_perturbation(
    experiment_name = 'across_species',
    model_name = 'scPILOT',
    data_file = 'species',
    file_type = '.h5ad',
    cond_key = 'condition',
    ctrl_key = 'unst',
    stim_key = 'LPS6',
    cell_label_key = 'species',
    query_key = 'mouse',
    sub_key = 'individual',
    seed = 1327,
    split_seed = 0,
):
    sns.set_theme(style = 'white', font = 'Arial', font_scale = 2)
    ensure_dirs(
        f'../model_trained/{experiment_name}',
        f'../Result_anndata/{experiment_name}',
        f'../Figures/{experiment_name}',
        f'../DataFrames/{experiment_name}',
    )

    adata = sc.read_h5ad(f'../Data/{experiment_name}/{data_file}{file_type}')
    adata = adata[(adata.obs[cond_key] == ctrl_key) | (adata.obs[cond_key] == stim_key)].copy()
    print(f'======Predicting {query_key} | seed={seed} | split_seed={split_seed}======')

    # Outer hold-out setting is unchanged: only the stimulated cells of the query species are excluded.
    train = adata[~((adata.obs[cell_label_key] == query_key) &
                    (adata.obs[cond_key] == stim_key))].copy()
    print('train:')
    print(train)

    set_seed(seed)
    model = EGD_model(train)
    model.train(
        max_epochs = 400,
        batch_size = 32,
        early_stopping = True,
        early_stopping_patience = 25,
        enable_progress_bar = False,
        callbacks = [EpochProgressPrinter()],
        datasplitter_kwargs = {'random_state': split_seed},
        wandb_project = f'{experiment_name}_{model_name}_{query_key}_seed{seed}',
        experiment_name = f'{experiment_name}_{model_name}_{query_key}_seed{seed}',
    )
    wandb.finish()

    model_path = f'../model_trained/{experiment_name}/EGD_model_trained_on_{data_file}_{query_key}_seed{seed}.model'
    model.save(model_path, overwrite = True, save_anndata = True)
    # model = EGD_model.load(model_path)

    adata_query_ctrl = adata[((adata.obs[cell_label_key] == query_key) & (adata.obs[cond_key] == ctrl_key))].copy()
    adata_query_stim = adata[((adata.obs[cell_label_key] == query_key) & (adata.obs[cond_key] == stim_key))].copy()
    adata_query = adata[adata.obs[cell_label_key] == query_key].copy()
    n_top_degs = 50
    sc.tl.rank_genes_groups(
        adata_query,
        groupby=cond_key,
        method='wilcoxon',
        n_genes=n_top_degs,
    )
    top50_genes = adata_query.uns['rank_genes_groups']['names'][stim_key].tolist()
    gammas = np.logspace(1, -3, num = 50)
    metric_records = []

    for ot_flag in range(2):
        # Reset the RNG before inference to make prediction deterministic for this run.
        set_seed(seed)
        if ot_flag == 0:
            eval_model_name = 'VAEGAN'
            adata_query_pred, _ = model.predict(
                cell_label_key = cell_label_key,
                cond_key = cond_key,
                ctrl_key = ctrl_key,
                stim_key = stim_key,
                query_key = query_key,
            )
        else:
            eval_model_name = model_name
            adata_query_pred, _ = model.predict_new(
                cell_label_key = cell_label_key,
                cond_key = cond_key,
                ctrl_key = ctrl_key,
                stim_key = stim_key,
                query_key = query_key,
                sub_key=sub_key,
            )

        adata_query_pred.obs[cond_key] = 'pred'
        adata_query_eval = ad.concat([adata_query_stim, adata_query_pred])
        plt.figure()
        r2mean_all, r2mean_top50 = EGD_model.reg_mean_plot(
            adata_query_eval,
            cond_key=cond_key,
            axis_keys={'x': 'pred', 'y': stim_key},
            labels={'x': 'Prediction', 'y': 'Ground truth'},
            path_to_save=f'../Figures/{experiment_name}/{eval_model_name}_{data_file}_reg_mean_{query_key}_seed{seed}.jpg',
            gene_list=top50_genes[:10],
            show=False,
            top_genes=top50_genes,
            top_gene_label='T50',
            legend=False,
        )

        x = to_dense_array(adata_query_pred[:, top50_genes].X)
        y = to_dense_array(adata_query_stim[:, top50_genes].X)
        mmd = compute_mmd_loss(x, y, gammas = gammas)

        print(f'{eval_model_name}:')
        print(f'{query_key}: r2mean_all = {r2mean_all}')
        print(f'{query_key}: r2mean_top50 = {r2mean_top50}')
        print(f'{query_key}: mmd = {mmd}')

        metric_records.append({
            'experiment': experiment_name,
            'data_file': data_file,
            'query_key': query_key,
            'seed': seed,
            'split_seed': split_seed,
            'model': eval_model_name,
            'r2mean_all': r2mean_all,
            'r2mean_top50': r2mean_top50,
            'mmd_top50': mmd,
        })

        adata_query_eval_to_save = ad.concat([adata_query_ctrl, adata_query_eval])
        adata_query_eval_to_save.write_h5ad(
            f'../Result_anndata/{experiment_name}/{experiment_name}_{eval_model_name}_{data_file}_{query_key}_seed{seed}{file_type}'
        )

    metrics_df = pd.DataFrame(metric_records)
    metrics_df.to_csv(
        f'../DataFrames/{experiment_name}/{experiment_name}_{data_file}_{query_key}_seed{seed}_metrics.csv',
        index = False,
    )


if __name__ == '__main__':
    predict_perturbation(
        query_key = args.query_key,
        seed = args.seed,
        split_seed = args.split_seed,
    )
    print('Done')
