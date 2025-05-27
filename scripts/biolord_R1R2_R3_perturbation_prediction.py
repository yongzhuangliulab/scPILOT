import numpy as np
from biolord import Biolord
import scanpy as sc
import anndata as ad
def predict_perturbation(
    experiment_name = "R1R2",
    model_name = 'biolord',
    data_file='pbmc',
    file_type='.h5ad',
    cond_key='condition',
    ctrl_key='control',
    stim_key='stimulated',
    cell_type_key='cell_type',
):
    adata = sc.read_h5ad(f"../Data/{experiment_name}/{data_file}{file_type}")
    adata = adata[(adata.obs[cond_key] == ctrl_key) | (adata.obs[cond_key] == stim_key)].copy()
    cell_types = adata.obs[cell_type_key].unique().tolist()
    for i in range(len(cell_types)):
        print(f'======Predicting {i + 1}: {cell_types[i]}======')
        print('adata:')
        print(adata)
        train = adata[~((adata.obs[cell_type_key] == cell_types[i]) & (adata.obs[cond_key] == stim_key))].copy()
        print('train:')
        print(train)
        adata.obs['split'] = ''
        adata.obs['_indices'] = np.arange(adata.n_obs)
        train_mask = ~((adata.obs[cell_type_key] == cell_types[i]) & (adata.obs[cond_key] == stim_key))
        adata.obs.loc[train_mask, 'split'] = 'train'
        adata.obs.loc[~train_mask, 'split'] = 'test'
        Biolord.setup_anndata(
            adata,
            ordered_attributes_keys=None,
            categorical_attributes_keys=[cond_key, cell_type_key],
        )
        model = Biolord(adata, n_latent=256, split_key="split",train_split='train')
        model.train(max_epochs=300, batch_size=64,early_stopping=True,early_stopping_patience=20,enable_checkpointing=False)
        model.save(f'../model_trained/{experiment_name}/Biolord_model_trained_on_{data_file}_{cell_types[i]}.model', overwrite=True, save_anndata=True)
        #model = Biolord.load(f'../model_trained/{experiment_name}/Biolord_model_trained_on_{data_file}_{cell_types[i]}.model')
        ctrl_adata = adata[((adata.obs[cell_type_key] == cell_types[i]) & (adata.obs[cond_key] == ctrl_key))]
        stim_adata = adata[((adata.obs[cell_type_key] == cell_types[i]) & (adata.obs[cond_key] == stim_key))]
        idx_source = np.where((adata.obs[cell_type_key] == cell_types[i]) & (adata.obs[cond_key] == ctrl_key))[0]
        adata_source = adata[idx_source].copy()
        pred_biolord = model.compute_prediction_adata(adata, adata_source, target_attributes=[cond_key])
        pred_biolord.obs[cond_key] = 'pred'
        eval_adata_biolord = ad.concat([ctrl_adata,stim_adata, pred_biolord])
        eval_adata_biolord.write_h5ad(f"../Result_anndata/{experiment_name}/{experiment_name}_{model_name}_{data_file}_{cell_types[i]}.h5ad")
if __name__ == '__main__':
    predict_perturbation(
        experiment_name = "R1R2",
        data_file='pbmc',
        file_type='.h5ad',
        cond_key='condition',
        ctrl_key='control',
        stim_key='stimulated',
        cell_type_key='cell_type',
    )
    predict_perturbation(
        experiment_name = "R3",
        data_file='pbmc_patients',
        file_type='.h5ad',
        cond_key='condition',
        ctrl_key='ctrl',
        stim_key='stim',
        cell_type_key='sample_id',
    )