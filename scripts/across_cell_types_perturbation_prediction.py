import wandb
import argparse
import scanpy as sc
import anndata as ad
import seaborn as sns
import numpy as np
from sklearn.metrics import pairwise
from matplotlib import pyplot as plt
from scpilot.egd_model import EGD_model
parser = argparse.ArgumentParser(description = 'across_cell_types_perturbation_prediction')
parser.add_argument(
    '--query_key',
    type = str,
    default = 'B',
    help = 'B, CD14+Mono, CD4T, CD8T, Dendritic, FCGR3A+Mono, NK',
)
args = parser.parse_args()
def mmd_distance(x, y, gamma):
    xx = pairwise.rbf_kernel(x, x, gamma)
    xy = pairwise.rbf_kernel(x, y, gamma)
    yy = pairwise.rbf_kernel(y, y, gamma)
    return xx.mean() + yy.mean() - 2 * xy.mean()
def compute_mmd_loss(lhs, rhs, gammas):
    return np.mean([mmd_distance(lhs, rhs, g) for g in gammas])
def predict_perturbation(
    experiment_name = 'across_cell_types',
    model_name = 'scPILOT',
    data_file = 'pbmc',
    file_type = '.h5ad',
    cond_key = 'condition',
    ctrl_key = 'control',
    stim_key = 'stimulated',
    cell_label_key = 'cell_type',
    query_key = 'B',
):
    sns.set_theme(style = 'white', font = 'Arial', font_scale = 2)
    adata = sc.read_h5ad(f'../Data/{experiment_name}/{data_file}{file_type}')
    adata = adata[(adata.obs[cond_key] == ctrl_key) | (adata.obs[cond_key] == stim_key)].copy()
    print(f'======Predicting {query_key}======')
    train = adata[~((adata.obs[cell_label_key] == query_key) &
                        (adata.obs[cond_key] == stim_key))].copy()
    print('train:')
    print(train)
    model = EGD_model(train)
    model.train(
        max_epochs = 400,
        batch_size = 32,
        early_stopping = True,
        early_stopping_patience = 25,
        enable_progress_bar = True,
        wandb_project = f'{experiment_name}_{model_name}_{query_key}',
    )
    wandb.finish()
    model.save(f'../model_trained/{experiment_name}/EGD_model_trained_on_{data_file}_{query_key}.model', overwrite = True, save_anndata = True)
    # model = EGD_model.load(f'../model_trained/{experiment_name}/EGD_model_trained_on_{data_file}_{query_key}.model')
    adata_query_ctrl = adata[((adata.obs[cell_label_key] == query_key) & (adata.obs[cond_key] == ctrl_key))].copy()
    adata_query_stim = adata[((adata.obs[cell_label_key] == query_key) & (adata.obs[cond_key] == stim_key))].copy()
    adata_query = adata[adata.obs[cell_label_key] == query_key].copy()
    sc.tl.rank_genes_groups(adata_query, groupby = cond_key, method = 'wilcoxon', n_genes = 100)
    diff_genes = adata_query.uns['rank_genes_groups']['names'][stim_key]
    for ot_flag in range(2):
        if ot_flag == 0:
            adata_query_pred, _ = model.predict(
                cell_label_key = cell_label_key,
                cond_key = cond_key,
                ctrl_key = ctrl_key,
                stim_key = stim_key,
                query_key = query_key,
            )
        else:
            adata_query_pred, _ = model.predict_new(
                cell_label_key = cell_label_key,
                cond_key = cond_key,
                ctrl_key = ctrl_key,
                stim_key = stim_key,
                query_key = query_key,
            )
        adata_query_pred.obs[cond_key] = 'pred'
        adata_query_eval = ad.concat([adata_query_stim, adata_query_pred])
        gammas = np.logspace(1, -3, num = 50)
        if ot_flag == 0:
            plt.figure()
            r2mean_all, r2mean_top100 = EGD_model.reg_mean_plot(
                adata_query_eval,
                cond_key = cond_key,
                axis_keys = {'x': 'pred', 'y': stim_key},
                labels = {'x': 'Prediction','y': 'Ground truth'},
                path_to_save = f'../Figures/{experiment_name}/VAEGAN_{data_file}_reg_mean_{query_key}.jpg',
                gene_list = diff_genes[:10],
                show = False,
                top_100_genes = diff_genes,
                legend = False,
            )
            x = adata_query_pred[: , diff_genes.tolist()[: 50]].X.toarray() if hasattr(
                adata_query_pred[: , diff_genes.tolist()[: 50]].X, 'toarray'
            ) else adata_query_pred[: , diff_genes.tolist()[: 50]].X
            y = adata_query_stim[: , diff_genes.tolist()[: 50]].X.toarray() if hasattr(
                adata_query_stim[: , diff_genes.tolist()[: 50]].X, 'toarray'
            ) else adata_query_stim[: , diff_genes.tolist()[: 50]].X
            mmd = compute_mmd_loss(x, y, gammas = gammas)
            print(f'VAEGAN:')
            print(f'{query_key}: r2mean_all = {r2mean_all}')
            print(f'{query_key}: r2mean_top100 = {r2mean_top100}')
            print(f'{query_key}: mmd = {mmd}')
            adata_query_eval = ad.concat([adata_query_ctrl, adata_query_eval])
            adata_query_eval.write_h5ad(f'../Result_anndata/{experiment_name}/{experiment_name}_VAEGAN_{data_file}_{query_key}{file_type}')
        else:
            plt.figure()
            r2mean_all, r2mean_top100 = EGD_model.reg_mean_plot(
                adata_query_eval,
                cond_key = cond_key,
                axis_keys = {'x': 'pred', 'y': stim_key},
                labels = {'x': 'Prediction','y': 'Ground truth'},
                path_to_save = f'../Figures/{experiment_name}/{model_name}_{data_file}_reg_mean_{query_key}.jpg',
                gene_list = diff_genes[:10],
                show = False,
                top_100_genes = diff_genes,
                legend = False,
            )
            x = adata_query_pred[: , diff_genes.tolist()[: 50]].X.toarray() if hasattr(
                adata_query_pred[: , diff_genes.tolist()[: 50]].X, 'toarray'
            ) else adata_query_pred[: , diff_genes.tolist()[: 50]].X
            y = adata_query_stim[: , diff_genes.tolist()[: 50]].X.toarray() if hasattr(
                adata_query_stim[: , diff_genes.tolist()[: 50]].X, 'toarray'
            ) else adata_query_stim[: , diff_genes.tolist()[: 50]].X
            mmd = compute_mmd_loss(x, y, gammas = gammas)
            print(f'{model_name}:')
            print(f'{query_key}: r2mean_all = {r2mean_all}')
            print(f'{query_key}: r2mean_top100 = {r2mean_top100}')
            print(f'{query_key}: mmd = {mmd}')
            adata_query_eval = ad.concat([adata_query_ctrl, adata_query_eval])
            adata_query_eval.write_h5ad(f'../Result_anndata/{experiment_name}/{experiment_name}_{model_name}_{data_file}_{query_key}{file_type}')
if __name__ == '__main__':
    query_key = args.query_key
    predict_perturbation(query_key = query_key)
    print('Done')