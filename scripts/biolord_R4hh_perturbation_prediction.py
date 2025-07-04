import scanpy as sc
import anndata as ad
import seaborn as sns
from biolord import Biolord
from sklearn.model_selection import train_test_split
def predict_perturbation(
    experiment_name = 'R4hh',
    model_name = 'biolord',
    data_file = 'sciplex3_partial',
    file_type = '.h5ad',
    cond_key = 'drug',
    ctrl_key = 'control',
    drug_no = -1,
    drug_name = 'stim',
    cell_label_key = 'cell_type',
    random_state = 0,
):
    sns.set_theme(font = 'Times New Roman', font_scale = 2)
    adata = sc.read_h5ad(f"../Data/{experiment_name}/{data_file}{file_type}")
    adata = adata[(adata.obs[cond_key] == ctrl_key) | (adata.obs[cond_key] == drug_name)].copy()
    print('adata:')
    print(adata)
    query_keys = sorted(adata.obs[cell_label_key].unique().tolist())
    for query_no, query_key in enumerate(query_keys):
        print(f'======Predicting {drug_no + 1}: {drug_name} on {query_no + 1}: {query_key}======')
        adata.obs[cell_label_key] = adata.obs[cell_label_key].cat.add_categories(f'{query_key}_rsv')
        query_key_ctrl_obs_names = adata.obs_names[(adata.obs[cell_label_key] == query_key) & (adata.obs[cond_key] == ctrl_key)]
        ctrl_reserve_obs_names, ctrl_holdout_obs_names = train_test_split(query_key_ctrl_obs_names, random_state = random_state, test_size = 0.5)
        adata.obs[cell_label_key].loc[ctrl_reserve_obs_names] = f'{query_key}_rsv'
        query_key_stim_obs_names = adata.obs_names[(adata.obs[cell_label_key] == query_key) & (adata.obs[cond_key] == drug_name)]
        stim_reserve_obs_names, stim_holdout_obs_names = train_test_split(query_key_stim_obs_names, random_state = random_state, test_size = 0.5)
        adata.obs[cell_label_key].loc[stim_reserve_obs_names] = f'{query_key}_rsv'
        train_mask = ~((adata.obs[cell_label_key] == query_key) & (adata.obs[cond_key] == drug_name))
        adata.obs.loc[train_mask, 'split'] = 'train'
        adata.obs.loc[~train_mask, 'split'] = 'test'
        Biolord.setup_anndata(
            adata,
            ordered_attributes_keys=None,
            categorical_attributes_keys=[cond_key, cell_label_key],
        )
        model = Biolord(adata, n_latent=256, split_key="split",train_split='train')
        model.train(max_epochs=300, batch_size=64,early_stopping=True,early_stopping_patience=20,enable_checkpointing=False)
        model.save(f'../model_trained/{experiment_name}/Biolord_model_trained_on_{data_file}_{query_key}_{drug_name}.model', overwrite=True, save_anndata=True)
        adata_source = adata[adata.obs_names.isin(ctrl_holdout_obs_names)].copy()
        adata_query_pred = model.compute_prediction_adata(adata, adata_source, target_attributes=[cond_key])
        adata_query_pred.obs[cond_key] = 'pred'
        adata_query_ctrl = adata[adata.obs_names.isin(ctrl_holdout_obs_names)].copy()
        adata_query_stim = adata[adata.obs_names.isin(stim_holdout_obs_names)].copy()
        adata_query = ad.concat([adata_query_ctrl, adata_query_stim, adata_query_pred])
        adata_query.write_h5ad(f'../Result_anndata/{experiment_name}/{experiment_name}_{model_name}_{data_file}_{query_key}_{drug_name}{file_type}')
if __name__ == '__main__':
    drug_names = [
        'abexinostat',
        'azacitidine',
        'baricitinib',
        'belinostat',
        'dacinostat',
        'decitabine',
        'entinostat',
        'givinostat',
        'iniparib',
        'mocetinostat',
        'ofloxacin',
        'pracinostat',
        'roxadustat',
        'ruxolitinib',
        'selisistat',
        'tacedinaline',
        'tazemetostat',
        'trametinib',
        'tucidinostat',
    ]
    for drug_no, drug_name in enumerate(drug_names):
        predict_perturbation(drug_no = drug_no, drug_name = drug_name)