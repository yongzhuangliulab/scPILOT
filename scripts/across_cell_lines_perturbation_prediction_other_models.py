import argparse
import scanpy as sc
import anndata as ad
import numpy as np
from scipy import stats
from scgen import SCGEN
from biolord import Biolord
parser = argparse.ArgumentParser(description = 'across_cell_lines_perturbation_prediction_other_models')
parser.add_argument(
    '--model_name',
    type = str,
    default = 'biolord',
    help = 'biolord, identity, scGen',
)
parser.add_argument(
    '--query_key',
    type = str,
    default = 'A549',
    help = 'A549, BXPC3, HAP1, HT29, K562, MCF7',
)
args = parser.parse_args()
def predict_perturbation(
    experiment_name = 'across_cell_lines',
    model_name = 'biolord',
    data_file = 'IFNGR2',
    file_type = '.h5ad',
    cond_key = 'target_gene',
    ctrl_key = 'non-targeting',
    stim_key = 'IFNGR2',
    cell_label_key = 'cell_type',
    query_key = 'A549',
):
    adata = sc.read_h5ad(f'../Data/{experiment_name}/{data_file}{file_type}')
    adata = adata[(adata.obs[cond_key] == ctrl_key) | (adata.obs[cond_key] == stim_key)].copy()
    print(f'======Predicting {query_key}======')
    train = adata[~((adata.obs[cell_label_key] == query_key) &
                        (adata.obs[cond_key] == stim_key))].copy()
    print('train:')
    print(train)
    if model_name == 'biolord':
        adata.obs['split'] = ''
        adata.obs['_indices'] = np.arange(adata.n_obs)
        train_mask = ~((adata.obs[cell_label_key] == query_key) & (adata.obs[cond_key] == stim_key))
        adata.obs.loc[train_mask, 'split'] = 'train'
        adata.obs.loc[~train_mask, 'split'] = 'test'
        Biolord.setup_anndata(
            adata,
            ordered_attributes_keys = None,
            categorical_attributes_keys = [cond_key, cell_label_key],
        )
        model = Biolord(adata, n_latent = 256, split_key = 'split', train_split = 'train')
        model.train(max_epochs = 300, batch_size = 64, early_stopping = True, early_stopping_patience = 20, enable_checkpointing = False, enable_progress_bar = True)
        idx_source = np.where((adata.obs[cell_label_key] == query_key) & (adata.obs[cond_key] == ctrl_key))[0]
        adata_source = adata[idx_source].copy()
        adata_query_pred = model.compute_prediction_adata(adata, adata_source, target_attributes = [cond_key])
        adata_query_pred = adata_query_pred[~adata_query_pred.obs.index.str.endswith(f'_{ctrl_key}')].copy()
        adata_query_pred.obs.index = adata_query_pred.obs.index.str.removesuffix(f'_{stim_key}')
        adata_query_pred.obs[cond_key] = 'pred'
        adata_query_ctrl = adata[((adata.obs[cell_label_key] == query_key) & (adata.obs[cond_key] == ctrl_key))].copy()
        adata_query_stim = adata[((adata.obs[cell_label_key] == query_key) & (adata.obs[cond_key] == stim_key))].copy()
        adata_query_eval = ad.concat([adata_query_ctrl, adata_query_stim, adata_query_pred])
        adata_query_eval.write_h5ad(f'../Result_anndata/{experiment_name}/{experiment_name}_{model_name}_{data_file}_{query_key}{file_type}')
        print(f'{model_name}_{query_key}:\n{adata_query_eval.obs}')
        x = np.average(
            adata_query_pred.X.toarray()
            if hasattr(adata_query_pred.X, 'toarray')
            else adata_query_pred.X,
            axis = 0,
        )
        y = np.average(
            adata_query_stim.X.toarray()
            if hasattr(adata_query_stim.X, 'toarray')
            else adata_query_stim.X,
            axis = 0,
        )
        _, _, r_value, _, _ = stats.linregress(x, y)
        print(f'{model_name}_{query_key}_r2mean = {r_value ** 2}')
    elif model_name == 'identity':
        adata_query_ctrl = adata[((adata.obs[cell_label_key] == query_key) & (adata.obs[cond_key] == ctrl_key))].copy()
        adata_query_stim = adata[((adata.obs[cell_label_key] == query_key) & (adata.obs[cond_key] == stim_key))].copy()
        adata_query_pred = adata_query_ctrl.copy()
        adata_query_pred.obs[cond_key] = 'pred'
        adata_query_eval = ad.concat([adata_query_ctrl, adata_query_stim, adata_query_pred])
        adata_query_eval.write_h5ad(f'../Result_anndata/{experiment_name}/{experiment_name}_{model_name}_{data_file}_{query_key}{file_type}')
        print(f'{model_name}_{query_key}:\n{adata_query_eval.obs}')
        x = np.average(
            adata_query_pred.X.toarray()
            if hasattr(adata_query_pred.X, 'toarray')
            else adata_query_pred.X,
            axis = 0,
        )
        y = np.average(
            adata_query_stim.X.toarray()
            if hasattr(adata_query_stim.X, 'toarray')
            else adata_query_stim.X,
            axis = 0,
        )
        _, _, r_value, _, _ = stats.linregress(x, y)
        print(f'{model_name}_{query_key}_r2mean = {r_value ** 2}')
    elif model_name == 'scGen':
        SCGEN.setup_anndata(train, batch_key = cond_key, labels_key = cell_label_key)
        model = SCGEN(train)
        model.train(
            max_epochs = 100,
            batch_size = 32,
            early_stopping = True,
            early_stopping_patience = 25,
            enable_progress_bar = True,
        )
        adata_query_pred, _ = model.predict(
            ctrl_key = ctrl_key,
            stim_key = stim_key,
            celltype_to_predict = query_key,
        )
        adata_query_pred.obs[cond_key] = 'pred'
        adata_query_ctrl = adata[((adata.obs[cell_label_key] == query_key) & (adata.obs[cond_key] == ctrl_key))].copy()
        adata_query_stim = adata[((adata.obs[cell_label_key] == query_key) & (adata.obs[cond_key] == stim_key))].copy()
        adata_query_eval = ad.concat([adata_query_ctrl, adata_query_stim, adata_query_pred])
        adata_query_eval.write_h5ad(f'../Result_anndata/{experiment_name}/{experiment_name}_{model_name}_{data_file}_{query_key}{file_type}')
        print(f'{model_name}_{query_key}:\n{adata_query_eval.obs}')
        x = np.average(
            adata_query_pred.X.toarray()
            if hasattr(adata_query_pred.X, 'toarray')
            else adata_query_pred.X,
            axis = 0,
        )
        y = np.average(
            adata_query_stim.X.toarray()
            if hasattr(adata_query_stim.X, 'toarray')
            else adata_query_stim.X,
            axis = 0,
        )
        _, _, r_value, _, _ = stats.linregress(x, y)
        print(f'{model_name}_{query_key}_r2mean = {r_value ** 2}')
    else:
        pass
if __name__ == '__main__':
    model_name = args.model_name
    query_key = args.query_key
    predict_perturbation(model_name = model_name, query_key = query_key)
    print('Done')