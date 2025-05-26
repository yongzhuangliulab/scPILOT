import scanpy as sc
import anndata as ad
import torch
import matplotlib.pyplot as plt
from scipy import sparse
from cellot.models import load_autoencoder_model, load_cellot_model
from cellot.utils import load_config
from sklearn.model_selection import train_test_split
from scpilot.egd_model import EGD_model
def predict_perturbation(
    data_file = 'pbmc',
    file_type = '.h5ad',
    cond_key = 'condition',
    ctrl_key = 'control',
    stim_key = 'stimulated',
    cell_label_key = 'cell_type',
    random_state = 0,
):
    adata = sc.read_h5ad(f"./datasets/{data_file}/{data_file}{file_type}")
    adata = adata[(adata.obs[cond_key] == ctrl_key) | (adata.obs[cond_key] == stim_key)].copy()
    query_keys = sorted(adata.obs[cell_label_key].unique().tolist())
    for mode, experiment_name in [('ood', 'R1R2')]:
        for query_no, query_key in enumerate(query_keys):
            print(f'======Predicting {query_no + 1}: {query_key}======')
            path_ae = f'./results/{data_file}/holdout-{stim_key}_{query_key}/mode-{mode}/model-scgen'
            ae_model, _ = load_autoencoder_model(
                load_config(f'{path_ae}/config.yaml'),
                restore = f'{path_ae}/cache/model.pt',
                input_dim = adata.n_vars
            )
            query_key_ctrl_obs_names = adata.obs_names[(adata.obs[cell_label_key] == query_key) & (adata.obs[cond_key] == ctrl_key)]
            ctrl_reserve_obs_names, ctrl_holdout_obs_names = train_test_split(query_key_ctrl_obs_names, random_state = random_state, test_size = 0.5)
            query_key_stim_obs_names = adata.obs_names[(adata.obs[cell_label_key] == query_key) & (adata.obs[cond_key] == stim_key)]
            stim_reserve_obs_names, stim_holdout_obs_names = train_test_split(query_key_stim_obs_names, random_state = random_state, test_size = 0.5)
            adata_query_ctrl = adata[adata.obs_names.isin(ctrl_holdout_obs_names)].copy()
            adata_query_stim = adata[adata.obs_names.isin(stim_holdout_obs_names)].copy()
            inputs = torch.Tensor(
                adata_query_ctrl.X if not sparse.issparse(adata_query_ctrl.X) else adata_query_ctrl.X.todense()
            )
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            inputs = inputs.to(device)
            latent_query_ctrl = ad.AnnData(
                ae_model.eval().encode(inputs).detach().cpu().numpy(),
                obs = adata_query_ctrl.obs.copy(),
                obsm = adata_query_ctrl.obsm.copy()
            )
            path_cellot = f'./results/{data_file}/holdout-{query_key}/mode-{mode}/model-cellot'
            cellot_model, _ = load_cellot_model(
                load_config(f'{path_cellot}/config.yaml'),
                restore = f'{path_cellot}/cache/model.pt',
                input_dim = latent_query_ctrl.n_vars
            )
            inputs = torch.Tensor(
                latent_query_ctrl.X if not sparse.issparse(latent_query_ctrl.X) else latent_query_ctrl.X.todense()
            )
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            inputs = inputs.to(device)
            inputs.requires_grad_(True)
            latent_query_pred = ad.AnnData(
                cellot_model[1].eval().transport(inputs).detach().cpu().numpy(),
                obs = latent_query_ctrl.obs.copy(),
                obsm = latent_query_ctrl.obsm.copy()
            )
            inputs = torch.Tensor(
                latent_query_pred.X if not sparse.issparse(latent_query_pred.X) else latent_query_pred.X.todense()
            )
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            inputs = inputs.to(device)
            adata_query_pred = ad.AnnData(
                ae_model.eval().decode(inputs).detach().cpu().numpy(),
                obs = latent_query_pred.obs.copy(),
                obsm = latent_query_pred.obsm.copy(),
            )
            adata_query_pred.var_names = adata_query_ctrl.var_names
            adata_query_pred.obs[cond_key] = 'pred'
            adata_query_eval = ad.concat([adata_query_stim, adata_query_pred])
            sc.pp.pca(adata_query_eval)
            sc.pp.neighbors(adata_query_eval)
            sc.tl.umap(adata_query_eval)
            sc.pl.umap(adata_query_eval, color=cond_key, frameon=False, show = False)
            plt.savefig(f'./Figures/{data_file}_prediction_{query_key}_{mode}.png', dpi = 200, bbox_inches = 'tight')
            adata_query = adata[adata.obs[cell_label_key] == query_key]
            sc.tl.rank_genes_groups(adata_query, groupby = cond_key, method = 'wilcoxon', n_genes = 100)
            diff_genes = adata_query.uns['rank_genes_groups']['names'][stim_key]
            EGD_model.setup_anndata(adata_query_eval, batch_key = cond_key, labels_key = cell_label_key)
            plt.figure()
            _ = EGD_model(adata_query_eval).reg_mean_plot(
                adata_query_eval,
                axis_keys = {'x': 'pred', 'y': stim_key},
                gene_list = diff_genes[: 10],
                top_100_genes = diff_genes,
                labels = {'x': 'Prediction', 'y': 'Ground truth'},
                path_to_save = f'./Figures/{data_file}_reg_mean_{query_key}_{mode}.png',
                show = False,
                legend = False
            )
            plt.figure()
            _ = EGD_model(adata_query_eval).reg_var_plot(
                adata_query_eval,
                axis_keys = {'x': 'pred', 'y': stim_key},
                gene_list = diff_genes[: 10],
                top_100_genes = diff_genes,
                labels = {'x': 'Prediction', 'y': 'Ground truth'},
                path_to_save = f'./Figures/{data_file}_reg_var_{query_key}_{mode}.png',
                show = False,
                legend = False
            )
            adata_query_eval = ad.concat([adata_query_ctrl, adata_query_eval])
            sc.pl.violin(adata_query_eval, keys = diff_genes[0], groupby = cond_key, show = False)
            plt.savefig(f'./Figures/{data_file}_diff_genes[0]_violin_{query_key}_{mode}.png', dpi = 200, bbox_inches = 'tight')
            adata_query_eval.write_h5ad(f'./Result_anndata/{experiment_name}/{experiment_name}_CellOT_{data_file}_{query_key}{file_type}')
if __name__ == '__main__':
    predict_perturbation()