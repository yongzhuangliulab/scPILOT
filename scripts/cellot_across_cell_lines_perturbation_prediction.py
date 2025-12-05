import scanpy as sc
import anndata as ad
import torch
from scipy import sparse
from cellot.models import load_autoencoder_model, load_cellot_model
from cellot.utils import load_config
def predict_perturbation(
    experiment_name = 'across_cell_lines',
    data_file = 'IFNGR2',
    file_type = '.h5ad',
    cond_key = 'target_gene',
    ctrl_key = 'non-targeting',
    stim_key = 'IFNGR2',
    cell_label_key = 'cell_type',
):
    adata = sc.read_h5ad(f"../Data/{experiment_name}/{data_file}{file_type}")
    adata = adata[(adata.obs[cond_key] == ctrl_key) | (adata.obs[cond_key] == stim_key)].copy()
    query_keys = sorted(adata.obs[cell_label_key].unique().tolist())
    for i, query_key in enumerate(query_keys):
        print(f'======Predicting {i + 1}: {query_key}======')
        path_ae = f'../cellot_results/{experiment_name}/holdout-{stim_key}_{query_key}/mode-ood/model-scgen'
        ae_model, _ = load_autoencoder_model(
            load_config(f'{path_ae}/config.yaml'),
            restore = f'{path_ae}/cache/model.pt',
            input_dim = adata.n_vars
        )
        adata_query_ctrl = adata[(adata.obs[cell_label_key] == query_key) & (adata.obs[cond_key] == ctrl_key)]
        adata_query_stim = adata[(adata.obs[cell_label_key] == query_key) & (adata.obs[cond_key] == stim_key)]
        inputs = torch.Tensor(
            adata_query_ctrl.X if not sparse.issparse(adata_query_ctrl.X) else adata_query_ctrl.X.todense()
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        inputs = inputs.to(device)
        latent_query_ctrl = ad.AnnData(
            ae_model.eval().encode(inputs).detach().cpu().numpy(),
            obs = adata_query_ctrl.obs.copy(),
            obsm = adata_query_ctrl.obsm.copy()
        )
        path_cellot = f'../cellot_results/{experiment_name}/holdout-{query_key}/mode-ood/model-cellot'
        cellot_model, _ = load_cellot_model(
            load_config(f'{path_cellot}/config.yaml'),
            restore = f'{path_cellot}/cache/model.pt',
            input_dim = latent_query_ctrl.n_vars
        )
        inputs = torch.Tensor(
            latent_query_ctrl.X if not sparse.issparse(latent_query_ctrl.X) else latent_query_ctrl.X.todense()
        )
        inputs = inputs.to(device)
        inputs.requires_grad_(True)
        latent_query_pred = ad.AnnData(
            cellot_model[1].eval().transport(inputs).detach().cpu().numpy(),
            obs = adata_query_ctrl.obs.copy(),
            obsm = adata_query_ctrl.obsm.copy()
        )
        inputs = torch.Tensor(
            latent_query_pred.X if not sparse.issparse(latent_query_pred.X) else latent_query_pred.X.todense()
        )
        inputs = inputs.to(device)
        adata_query_pred = ad.AnnData(
            ae_model.eval().decode(inputs).detach().cpu().numpy(),
            obs = adata_query_ctrl.obs.copy(),
            obsm = adata_query_ctrl.obsm.copy(),
        )
        adata_query_pred.var_names = adata_query_ctrl.var_names
        adata_query_pred.obs[cond_key] = 'pred'
        adata_query_eval = ad.concat([adata_query_ctrl, adata_query_stim, adata_query_pred])
        adata_query_eval.write_h5ad(f'../Result_anndata/{experiment_name}/{experiment_name}_CellOT_{data_file}_{query_key}{file_type}')
if __name__ == '__main__':
    predict_perturbation()
    print('Done')