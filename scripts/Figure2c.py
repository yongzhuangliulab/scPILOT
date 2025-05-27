import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
def predict_perturbation(
    data_file = 'pbmc',
    file_type = '.h5ad',
    cond_key = 'condition',
    ctrl_key = 'control',
    stim_key = 'stimulated',
    cell_type_key = 'cell_type',
):
    sns.set_theme(font='Times New Roman', font_scale=2)
    adata = sc.read_h5ad(f"../Data/R1R2/{data_file}{file_type}")
    adata = adata[(adata.obs[cond_key] == ctrl_key) | (adata.obs[cond_key] == stim_key)].copy()
    cell_types = sorted(adata.obs[cell_type_key].unique().tolist())
    r2_means = []
    r2_stds = []
    for i in range(len(cell_types)):
        print(f'======Calculating r2 {i + 1}: {cell_types[i]}======')
        print('adata:')
        print(adata)
        cell_type_i = adata[adata.obs[cell_type_key] == cell_types[i]]
        sc.tl.rank_genes_groups(cell_type_i, groupby=cond_key, method="wilcoxon", n_genes=100)
        diff_genes = cell_type_i.uns["rank_genes_groups"]["names"][stim_key]
        print(diff_genes)
        with_ot = ['', '_with_ot']
        for ot_flag in range(2):
            if ot_flag == 0:
                eval_adata = sc.read_h5ad(f'../Result_anndata/R1R2/R1R2_VAEGAN_pbmc_{cell_types[i]}.h5ad')
            elif ot_flag == 1:
                eval_adata = sc.read_h5ad(f'../Result_anndata/R1R2/R1R2_scPILOT_pbmc_{cell_types[i]}.h5ad')
            adata_i_pred = eval_adata[eval_adata.obs[cond_key] == 'pred']
            adata_i_stim = eval_adata[eval_adata.obs[cond_key] == stim_key]
            for gene_set in ['all', 'top100']:
                r2_values = np.zeros((1, 100))
                for j in range(100):
                    adata_i_pred_idx = np.random.choice(range(adata_i_pred.shape[0]), int(0.8 * adata_i_pred.shape[0]))
                    adata_i_stim_idx = np.random.choice(range(adata_i_stim.shape[0]), int(0.8 * adata_i_stim.shape[0]))
                    if gene_set == 'all':
                        x = np.average(adata_i_pred.X.toarray()[adata_i_pred_idx], axis = 0)
                        y = np.average(adata_i_stim.X.toarray()[adata_i_stim_idx], axis = 0)
                    else:
                        x = np.average(adata_i_pred[: , diff_genes.tolist()].X.toarray()[adata_i_pred_idx], axis = 0)
                        y = np.average(adata_i_stim[: , diff_genes.tolist()].X.toarray()[adata_i_stim_idx], axis = 0)
                    _, _, r_value, _, _ = stats.linregress(x, y)
                    r2_values[0, j] = r_value ** 2
                r2_means.append(r2_values.mean())
                r2_stds.append(r2_values.std())
                print(f'{cell_types[i]}: r2_{gene_set}_mean{with_ot[ot_flag]} = {r2_values.mean()}, r2_{gene_set}_std{with_ot[ot_flag]} = {r2_values.std()}')
    with_ots = [False, False, True, True] * 7
    gene_sets = ['all genes', 'top 100 DEGs'] * 2 * 7
    cell_types_fordf = []
    for ct in cell_types:
        cell_types_fordf.extend([ct] * 2 * 2)
    df = pd.DataFrame({'r2 means': r2_means, 'r2 stds': r2_stds, 'with ot?': with_ots, 'gene set': gene_sets, 'cell type': cell_types_fordf})
    df = df.sort_values(by=['with ot?', 'cell type','gene set'], ascending=[True, False,True])
    df['gene set'] = df['gene set'].replace({'all genes': 'All genes', 'top 100 DEGs': 'Top 100 DEGs'})
    def autolabel(rects, ax, values):
        for rect, value in zip(rects, values):
            width = rect.get_width()
            ax.text(
                rect.get_x() + width + 0.05,
                rect.get_y() + rect.get_height() / 2,
                '%.2f' % value,
                ha='left',
                va='center',
                fontsize=25,
            )
    def grouped_barplot(df, cat, subcat, val, err, filename, put_label=False, legend=False, height=0.40):
        sns.set_theme(font='Times New Roman', font_scale=2)
        plt.close('all')
        matplotlib.rc('ytick', labelsize=25)
        matplotlib.rc('xtick', labelsize=30)
        u = df[cat].unique()
        y_pos = np.arange(len(u))
        suby = df[subcat].unique()
        suby = ['Top 100 DEGs', 'All genes']
        plt.figure(figsize=(12, 12))
        colors = sns.color_palette("husl", len(suby))
        gene_set_colors = {
            'All genes': colors[0],
            'Top 100 DEGs': colors[1]
        }
        color_map = [gene_set_colors.get(gr, colors.pop(0)) for gr in suby]
        for i, gr in enumerate(suby):
            dfg = df[df[subcat] == gr]
            b = plt.barh(
                y_pos + i * height,
                dfg[val].values,
                height=height,
                capsize=10,
                alpha=0.95,
                label=f'{gr}',
                xerr=dfg[err].values,
                color=color_map[i]
            )
            a = np.random.normal(dfg[val].values, dfg[err].values, (10, len(u)))
            plt.plot(a.T, y_pos + i * height, '.', color='black', alpha=0.5)
            if put_label:
                autolabel(b, plt.gca(), dfg[val].values)
        plt.xlabel(r'$\mathrm{R^2_{mean}}$', fontsize=25)
        plt.yticks(y_pos + height * (len(suby) - 1) / 4 + height / 4, u, rotation=0)
        if legend:
            plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', borderaxespad=0., ncol=len(suby))
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()
    grouped_barplot(
        df[df['with ot?'] == True],
        'cell type',
        'gene set',
        'r2 means',
        'r2 stds',
        legend= False,
        filename='../Figures/R1R2/fig2c_barplot.png',
        put_label=True,
    )
if __name__ == '__main__':
    predict_perturbation()