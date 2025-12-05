import scanpy as sc
import anndata as ad
import seaborn as sns
from matplotlib import pyplot as plt
def perturbation_prediction_dotplot(
    experiment_name = 'across_cell_types',
    model_name = 'scPILOT',
    data_file = 'pbmc',
    file_type = '.h5ad',
    cond_key = 'condition',
    ctrl_key = 'control',
    stim_key = 'stimulated',
    cell_label_key = 'cell_type',
):
    sns.set_theme(style = 'white', font = 'Arial', font_scale = 2)
    adata = sc.read_h5ad(f'../Data/{experiment_name}/{data_file}{file_type}')
    adata = adata[(adata.obs[cond_key] == ctrl_key) | (adata.obs[cond_key] == stim_key)].copy()
    print('adata:')
    print(adata)
    query_keys = sorted(adata.obs[cell_label_key].unique().tolist())
    adata_query_eval_list = []
    for query_no, query_key in enumerate(query_keys):
        adata_query_eval = ad.read_h5ad(f'../Result_anndata/{experiment_name}/{experiment_name}_{model_name}_{data_file}_{query_key}{file_type}')
        adata_query_eval_list.append(adata_query_eval)
    adata_eval = ad.concat(adata_query_eval_list)
    gene_list = ['CD3D', 'CCL5', 'GNLY', 'CD79A', 'FCGR3A', 'S100A9', 'HLA-DQA1', 'ISG15', 'IFI6', 'IFIT1', 'CXCL10', 'CXCL11', 'APOBEC3A', 'DEFB1', 'CCL8', 'TARBP1']
    adata_eval.obs['condition'] = adata_eval.obs['condition'].replace('stimulated', 'stim').replace('control', 'ctrl')
    adata_eval.obs['cell_type_condition'] = (
        adata_eval.obs['cell_type'].astype(str) + '_' + adata_eval.obs['condition'].astype(str)
    ).astype('category')
    categories_order = [
        'B_ctrl', 'B_stim', 'B_pred',
        'CD4T_ctrl', 'CD4T_stim', 'CD4T_pred',
        'CD8T_ctrl', 'CD8T_stim', 'CD8T_pred',
        'CD14+Mono_ctrl', 'CD14+Mono_stim', 'CD14+Mono_pred',
        'Dendritic_ctrl', 'Dendritic_stim', 'Dendritic_pred',
        'FCGR3A+Mono_ctrl', 'FCGR3A+Mono_stim', 'FCGR3A+Mono_pred',
        'NK_ctrl', 'NK_stim', 'NK_pred',
    ]
    dp = sc.pl.dotplot(
        adata_eval,
        var_names = gene_list,
        groupby = 'cell_type_condition',
        categories_order = categories_order,
        show = False,
        return_fig = True,
    )
    dp.style(cmap = 'Purples', dot_edge_color = None, dot_edge_lw = 0)
    dp_ax_dict = dp.get_axes()
    print(dp_ax_dict)
    mainplot_ax = dp_ax_dict['mainplot_ax']
    mainplot_ax.tick_params(labelsize = 14)
    size_legend_ax = dp_ax_dict['size_legend_ax']
    size_legend_ax.set_title('Fraction of cells\nin group (%)', fontsize = 14)
    size_legend_ax.tick_params(axis = 'x', labelsize = 12)
    color_legend_ax = dp_ax_dict['color_legend_ax']
    color_legend_ax.set_title('Mean expression\nin group', fontsize = 14)
    color_legend_ax.tick_params(axis = 'x', labelsize = 12)
    plt.savefig(f'../Figures/{experiment_name}/{model_name}_dotplot.jpg', dpi = 300, bbox_inches = 'tight')
    plt.close()
if __name__ == '__main__':
    perturbation_prediction_dotplot()
    print('Done')