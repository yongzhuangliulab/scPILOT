import scanpy as sc
import anndata as ad
import seaborn as sns
def predict_perturbation_dotpolt(
    data_file='pbmc',
    file_type='.h5ad',
    cond_key='condition',
    ctrl_key='control',
    stim_key='stimulated',
    cell_type_key='cell_type',
):
    sns.set_theme(font='Times New Roman', font_scale=2)
    adata = sc.read_h5ad(f"../Data/R1R2/{data_file}{file_type}")
    adata = adata[(adata.obs[cond_key] == ctrl_key) | (adata.obs[cond_key] == stim_key)].copy()
    cell_types = adata.obs[cell_type_key].unique().tolist()
    pred_dict = {}
    for i in range(len(cell_types)):
        a = sc.read_h5ad(f'../Result_anndata/R1R2/R1R2_scPILOT_pbmc_{cell_types[i]}.h5ad')
        pred = a[a.obs["condition"] == "pred"].copy()
        pred.obs[cond_key] = 'pred'
        pred_dict[cell_types[i]] = pred
    adata_eval =  ad.concat([adata] + list(pred_dict.values()))
    gene_list = ["CD3D", "CCL5", "GNLY", "CD79A", "FCGR3A", "S100A9", "HLA-DQA1",
                 "ISG15", "IFI6", "IFIT1", "CXCL10", "CXCL11", "APOBEC3A", "DEFB1",
                 "CCL8", "TARBP1"]
    adata_eval.obs["condition"].replace("stimulated", "stim", inplace=True)
    adata_eval.obs["condition"].replace("control", "ctrl", inplace=True)
    adata_eval.obs['cell_type'] = adata_eval.obs['cell_type'].astype(str)
    adata_eval.obs['condition'] = adata_eval.obs['condition'].astype(str)
    adata_eval.obs['new_condition'] = adata_eval.obs['cell_type'].str.cat(adata_eval.obs['condition'], sep='_')
    adata_eval.obs['condition'] = adata_eval.obs['new_condition']
    sc.set_figure_params(fontsize=14)
    categories_order = [
        'B_ctrl', 'B_stim', 'B_pred',
        'CD4T_ctrl', 'CD4T_stim', 'CD4T_pred',
        'CD8T_ctrl', 'CD8T_stim', 'CD8T_pred',
        'CD14+Mono_ctrl', 'CD14+Mono_stim', 'CD14+Mono_pred',
        'Dendritic_ctrl', 'Dendritic_stim', 'Dendritic_pred',
        'FCGR3A+Mono_ctrl', 'FCGR3A+Mono_stim', 'FCGR3A+Mono_pred',
        'NK_ctrl', 'NK_stim', 'NK_pred'
    ]
    sc.pl.dotplot(
        adata_eval,
        var_names=gene_list,
        groupby="condition",
        categories_order=categories_order,
        save="../Figures/R1R2/fig2i_dotplot.png",
        show=True
    )
if __name__ == '__main__':
    predict_perturbation_dotpolt()