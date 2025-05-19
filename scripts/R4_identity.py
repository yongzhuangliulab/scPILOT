import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Literal
from scgen import SCGEN
from sklearn.model_selection import train_test_split
from scpilot.egd_model import EGD_model
def predict_perturbation(
    experiment_name = 'R4',
    model_name = 'identity',
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
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.pl.umap(adata, color = [cond_key], wspace = 0.4, frameon = False, show = False)
    plt.savefig(f'../Figures/{experiment_name}/temp0.png', dpi = 200, bbox_inches = 'tight')
    plt.close()
    sc.pl.umap(adata, color = [cell_label_key], wspace = 0.4, frameon = False, show = False)
    plt.savefig(f'../Figures/{experiment_name}/temp1.png', dpi = 200, bbox_inches = 'tight')
    plt.close()
    plt.figure()
    temp_fig = [0, 1]
    for temp_fig_no in range(2):
        plt.subplot2grid((2, 1), (temp_fig[temp_fig_no], 0), colspan = 1, rowspan = 1)
        plt.axis('off')
        plt.imshow(plt.imread(f'../Figures/{experiment_name}/temp{temp_fig[temp_fig_no]}.png'))
    plt.savefig(f'../Figures/{experiment_name}/{data_file}_UMAP.png', dpi = 200, bbox_inches = 'tight')
    plt.close()
    query_keys = sorted(adata.obs[cell_label_key].unique().tolist())
    for query_no, query_key in enumerate(query_keys):
        print(f'======Predicting {drug_no + 1}: {drug_name} on {query_no + 1}: {query_key}======')
        query_key_ctrl_obs_names = adata.obs_names[(adata.obs[cell_label_key] == query_key) & (adata.obs[cond_key] == ctrl_key)]
        ctrl_reserve_obs_names, ctrl_holdout_obs_names = train_test_split(query_key_ctrl_obs_names, random_state = random_state, test_size = 0.5)
        query_key_stim_obs_names = adata.obs_names[(adata.obs[cell_label_key] == query_key) & (adata.obs[cond_key] == drug_name)]
        stim_reserve_obs_names, stim_holdout_obs_names = train_test_split(query_key_stim_obs_names, random_state = random_state, test_size = 0.5)
        adata_query_ctrl = adata[adata.obs_names.isin(ctrl_holdout_obs_names)].copy()
        adata_query_stim = adata[adata.obs_names.isin(stim_holdout_obs_names)].copy()
        adata_query = ad.concat([adata_query_ctrl, adata_query_stim])
        sc.tl.rank_genes_groups(adata_query, groupby = cond_key, method = 'wilcoxon', n_genes = 100)
        diff_genes = adata_query.uns['rank_genes_groups']['names'][drug_name]
        adata_query_pred = adata_query_ctrl.copy()
        adata_query_pred.obs[cond_key] = 'pred'
        adata_query_eval = ad.concat([adata_query_stim, adata_query_pred])
        sc.pp.pca(adata_query_eval)
        sc.pp.neighbors(adata_query_eval)
        sc.tl.umap(adata_query_eval)
        sc.pl.umap(adata_query_eval, color = cond_key, frameon = False, show = False)
        plt.savefig(f'../Figures/{experiment_name}/{model_name}_{data_file}_prediction_{query_key}_{drug_name}.png', dpi = 200, bbox_inches = 'tight')
        plt.close()
        plt.figure()
        EGD_model.setup_anndata(adata_query_eval, batch_key = cond_key, labels_key = cell_label_key)
        r2mean_all, r2mean_top100 = EGD_model(adata_query_eval).reg_mean_plot(
            adata_query_eval,
            axis_keys = {"x": "pred", "y": drug_name},
            gene_list = diff_genes[:10],
            top_100_genes = diff_genes,
            labels = {"x": "Prediction","y": "Ground truth"},
            path_to_save = f"../Figures/{experiment_name}/{model_name}_{data_file}_reg_mean_{query_key}_{drug_name}.png",
            show = False,
            legend = False,
        )
        plt.figure()
        r2var_all, r2var_top100 = EGD_model(adata_query_eval).reg_var_plot(
            adata_query_eval,
            axis_keys = {"x": "pred", "y": drug_name},
            gene_list = diff_genes[:10],
            top_100_genes = diff_genes,
            labels = {"x": "Prediction","y": "Ground truth"},
            path_to_save = f"../Figures/{experiment_name}/{model_name}_{data_file}_reg_var_{query_key}_{drug_name}.png",
            show = False,
            legend = False,
        )
        print(f'{model_name}:')
        print(f'{query_key}_{drug_name}: r2mean_all = {r2mean_all}, r2mean_top100 = {r2mean_top100}')
        print(f'{query_key}_{drug_name}: r2var_all = {r2var_all}, r2var_top100 = {r2var_top100}')
        adata_query_eval = ad.concat([adata_query_ctrl, adata_query_eval])
        sc.pl.violin(adata_query_eval, keys = diff_genes[0], groupby = cond_key, rotation = 90, show = False)
        plt.savefig(f'../Figures/{experiment_name}/{model_name}_{data_file}_diff_genes[0]_violin_{query_key}_{drug_name}.png', dpi = 200, bbox_inches = 'tight')
        plt.close()
        adata_query_eval.write_h5ad(f'../Result_anndata/{experiment_name}/{experiment_name}_{model_name}_{data_file}_{query_key}_{drug_name}{file_type}')
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