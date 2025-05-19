import scanpy as sc
import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from math import sqrt
from sklearn.metrics.pairwise import rbf_kernel
def mmd_distance(x, y, gamma):
    xx = rbf_kernel(x, x, gamma)
    xy = rbf_kernel(x, y, gamma)
    yy = rbf_kernel(y, y, gamma)
    return xx.mean() + yy.mean() - 2 * xy.mean()
def compute_mmd_loss(lhs, rhs, gammas):
    return np.mean([mmd_distance(lhs, rhs, g) for g in gammas])
def plot_result(
    experiment_name = 'R1R2',
    model_names = ['scPILOT', 'scGen', 'CellOT', 'biolord', 'identity', 'VAEGAN'],
    data_file = 'pbmc',
    file_type = '.h5ad',
    cond_key = 'condition',
    ctrl_key = 'control',
    stim_key = 'stimulated',
    cell_label_key = 'cell_type',
):
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2., 1.01 * height,
                     '%.2f' % float(height),
                     ha = 'center', va = 'bottom', fontsize = 18)
    def grouped_barplot(df, cat, subcat, val, err, filename, put_label = False, legend = False, offset = 0.375, width = 5, ylabel = r'$\mathrm{R^2}$'):
        plt.close('all')
        matplotlib.rc('ytick', labelsize = 25)
        matplotlib.rc('xtick', labelsize = 30)
        u = sorted(df[cat].unique().tolist())
        x_pos = np.arange(0, width * len(u), width)
        subx = sorted(df[subcat].unique().tolist())
        plt.figure(figsize = (12, 10))
        for i, gr in enumerate(subx):
            dfg = df[df[subcat] == gr]
            b = plt.bar(x_pos + i / 1.25, dfg[val].values, capsize = 10, alpha = 0.95, label = f'{gr}', yerr = dfg[err].values)
            a = np.random.normal(dfg[val].values, dfg[err].values, (10, len(u)))
            plt.plot(x_pos + i / 1.25, a.T, '.', color = 'black', alpha = 0.5)
            if put_label:
                autolabel(b)
        plt.ylabel(ylabel, fontsize = 25)
        plt.xticks(x_pos + offset, u, rotation = 90)
        if legend:
            plt.legend(bbox_to_anchor = (1.05, 0.5), loc = 'center left', borderaxespad = 0, prop = {'size': 18})
        plt.tight_layout()
        plt.savefig(filename, dpi = 300)
        plt.show()
    sns.set_theme(font = 'Times New Roman', font_scale = 2)
    adata = ad.read_h5ad(f'../Data/{experiment_name}/{data_file}{file_type}')
    adata = adata[(adata.obs[cond_key] == ctrl_key) | (adata.obs[cond_key] == stim_key)].copy()
    print('adata:')
    print(adata)
    query_keys = sorted(adata.obs[cell_label_key].unique().tolist())
    r2mean_dfs = []
    l2mean_dfs = []
    r2var_dfs = []
    l2var_dfs = []
    gammas = np.logspace(1, -3, num = 50)
    mmd_dfs = []
    for query_no, query_key in enumerate(query_keys):
        query_adatas = []
        for model_no, model_name in enumerate(sorted(model_names)):
            adata_query_eval = ad.read_h5ad(f'../Result_anndata/{experiment_name}/{experiment_name}_{model_name}_{data_file}_{query_key}{file_type}')
            adata_query_ctrl = adata_query_eval[adata_query_eval.obs[cond_key] == ctrl_key].copy()
            adata_query_stim = adata_query_eval[adata_query_eval.obs[cond_key] == stim_key].copy()
            adata_query_pred = adata_query_eval[adata_query_eval.obs[cond_key] == 'pred'].copy()
            adata_query_stpr = ad.concat([adata_query_stim.copy(), adata_query_pred.copy()])
            adata_query_stpr.obs = adata_query_stpr.obs.replace(stim_key, stim_key.capitalize()).replace('pred', 'Pred')
            adata_query_stpr.obs.rename(columns = {cond_key: cond_key.capitalize()}, inplace = True)
            sc.pp.pca(adata_query_stpr)
            sc.pp.neighbors(adata_query_stpr)
            sc.tl.umap(adata_query_stpr)
            if model_no < len(model_names) - 1:
                sc.pl.umap(adata_query_stpr, color = cond_key.capitalize(), legend_loc = None, title = '', frameon = False, show = False)
            else:
                sc.pl.umap(adata_query_stpr, color = cond_key.capitalize(), title = '', frameon = False, show = False)
            plt.savefig(f'../Figures/{experiment_name}/{model_name}_{data_file}_{query_key}_prediction.png', dpi = 200, bbox_inches = 'tight')
            plt.close()
            if model_no == 0:
                query_adatas.extend([adata_query_ctrl.copy(), adata_query_stim.copy()])
            if model_name != 'identity':
                adata_query_pred_copy = adata_query_pred.copy()
                adata_query_pred_copy.obs[cond_key] = model_name
                query_adatas.append(adata_query_pred_copy)
        grid = (1, len(model_names))
        unit_len = 6
        loc = []
        for j in range(len(model_names)):
            loc.append((0, j))
        colspan = 1
        rowspan = 1
        plt.figure(figsize=(len(model_names) * unit_len, 1 * unit_len))
        model_names_in_order = sorted(
            [
                model_name.capitalize()
                if model_name in ['biolord', 'identity']
                else model_name 
                for model_name in model_names
            ]
        )
        for model_no, model_name in enumerate(model_names_in_order):
            plt.subplot2grid(grid, loc[model_no], colspan=colspan, rowspan=rowspan)
            plt.axis('off')
            plt.imshow(plt.imread(f'../Figures/{experiment_name}/{model_name.lower() if model_name in [
                'Biolord', 'Identity'
            ] else model_name}_{data_file}_{query_key}_prediction.png'))
            plt.title(model_name)
        plt.savefig(f'../Figures/{experiment_name}/All_models_{data_file}_{query_key}_prediction.png', dpi = 400, bbox_inches = 'tight')
        plt.close()
        adata_query_models = ad.concat(query_adatas)
        adata_query_models.obs = adata_query_models.obs.replace(
            ctrl_key, ctrl_key.capitalize()
        ).replace(stim_key, stim_key.capitalize()).replace('biolord', 'Biolord')
        groupby_order = [ctrl_key.capitalize(), stim_key.capitalize(), 'Biolord', 'CellOT', 'VAEGAN', 'scGen', 'scPILOT']
        adata_query = adata[adata.obs[cell_label_key] == query_key].copy()
        sc.tl.rank_genes_groups(adata_query, groupby = cond_key, method = 'wilcoxon', n_genes = 100)
        important_genes = adata_query.uns['rank_genes_groups']['names'][stim_key]
        for gene_no, gene_name in enumerate(important_genes):
            sc.pl.violin(
                adata_query_models,
                keys = gene_name,
                groupby = cond_key,
                rotation = 90,
                show = False,
                order = groupby_order,
                xlabel = cond_key.capitalize(),
            )
            plt.savefig(f'../Figures/{experiment_name}/All_models_on_{data_file}_{query_key}_{gene_name}_violin.png', dpi = 200, bbox_inches = 'tight')
            plt.close()
        grid = (1, 3)
        unit_len = 6
        loc = [(0, 0), (0, 1), (0, 2)]
        colspan = 1
        rowspan = 1
        plt.figure(figsize=(3 * unit_len, 1 * unit_len))
        for gene_no, gene_name in enumerate(important_genes[: 3]):
            plt.subplot2grid(grid, loc[gene_no], colspan = colspan, rowspan = rowspan)
            plt.axis('off')
            plt.imshow(plt.imread(f'../Figures/{experiment_name}/All_models_on_{data_file}_{query_key}_{gene_name}_violin.png'))
        plt.savefig(f'../Figures/{experiment_name}/All_models_on_{data_file}_{query_key}_important_genes_violin.png', dpi = 400, bbox_inches = 'tight')
        plt.close()
    for model_no, model_name in enumerate(sorted(model_names)):
        r2mean_means = []
        r2mean_stds = []
        l2mean_means = []
        l2mean_stds = []
        r2var_means = []
        r2var_stds = []
        l2var_means = []
        l2var_stds = []
        mmd_means = []
        mmd_stds = []
        for query_no, query_key in enumerate(query_keys):
            print(f'======Plotting {model_no + 1}: {model_name}\'s result on {query_no + 1}: {query_key}======')
            adata_query_eval = ad.read_h5ad(f'../Result_anndata/{experiment_name}/{experiment_name}_{model_name}_{data_file}_{query_key}{file_type}')
            adata_query_pred = adata_query_eval[adata_query_eval.obs[cond_key] == 'pred'].copy()
            adata_query_ctrl = adata_query_eval[adata_query_eval.obs[cond_key] == ctrl_key].copy()
            adata_query_stim = adata_query_eval[adata_query_eval.obs[cond_key] == stim_key].copy()
            adata_query = ad.concat([adata_query_ctrl, adata_query_stim])
            sc.tl.rank_genes_groups(adata_query, groupby = cond_key, method = 'wilcoxon', n_genes = 100)
            diff_genes = adata_query.uns['rank_genes_groups']['names'][stim_key]
            print(diff_genes)
            for gene_set in ['All genes', 'Top 100 DEGs']:
                r2mean_values = np.zeros((1, 100))
                l2mean_values = np.zeros((1, 100))
                r2var_values = np.zeros((1, 100))
                l2var_values = np.zeros((1, 100))
                for j in range(100):
                    adata_query_pred_idx = np.random.choice(range(adata_query_pred.shape[0]), int(0.8 * adata_query_pred.shape[0]))
                    adata_query_stim_idx = np.random.choice(range(adata_query_stim.shape[0]), int(0.8 * adata_query_stim.shape[0]))
                    if gene_set == 'All genes':
                        x = np.average(adata_query_pred.X.toarray()[adata_query_pred_idx], axis = 0)
                        y = np.average(adata_query_stim.X.toarray()[adata_query_stim_idx], axis = 0)
                    else:
                        x = np.average(adata_query_pred[: , diff_genes.tolist()].X.toarray()[adata_query_pred_idx], axis = 0)
                        y = np.average(adata_query_stim[: , diff_genes.tolist()].X.toarray()[adata_query_stim_idx], axis = 0)
                    _, _, r_value, _, _ = stats.linregress(x, y)
                    r2mean_values[0, j] = r_value ** 2
                    l2mean_values[0, j] = sqrt(((x - y) ** 2).sum())
                    if gene_set == 'All genes':
                        x = np.var(adata_query_pred.X.toarray()[adata_query_pred_idx], axis = 0)
                        y = np.var(adata_query_stim.X.toarray()[adata_query_stim_idx], axis = 0)
                    else:
                        x = np.var(adata_query_pred[: , diff_genes.tolist()].X.toarray()[adata_query_pred_idx], axis = 0)
                        y = np.var(adata_query_stim[: , diff_genes.tolist()].X.toarray()[adata_query_stim_idx], axis = 0)
                    _, _, r_value, _, _ = stats.linregress(x, y)
                    r2var_values[0, j] = r_value ** 2
                    l2var_values[0, j] = sqrt(((x - y) ** 2).sum())
                r2mean_means.append(r2mean_values.mean())
                r2mean_stds.append(r2mean_values.std())
                l2mean_means.append(l2mean_values.mean())
                l2mean_stds.append(l2mean_values.std())
                r2var_means.append(r2var_values.mean())
                r2var_stds.append(r2var_values.std())
                l2var_means.append(l2var_values.mean())
                l2var_stds.append(l2var_values.std())
                print(f'{model_name} on {query_key}: R2mean_{gene_set}_mean = {r2mean_values.mean()}, R2mean_{gene_set}_std = {r2mean_values.std()}')
                print(f'{model_name} on {query_key}: L2mean_{gene_set}_mean = {l2mean_values.mean()}, L2mean_{gene_set}_std = {l2mean_values.std()}')
                print(f'{model_name} on {query_key}: R2var_{gene_set}_mean = {r2var_values.mean()}, R2var_{gene_set}_std = {r2var_values.std()}')
                print(f'{model_name} on {query_key}: L2var_{gene_set}_mean = {l2var_values.mean()}, L2var_{gene_set}_std = {l2var_values.std()}')
            mmd_values = np.zeros((1, 10))
            for j in range(10):
                adata_query_pred_idx = np.random.choice(range(adata_query_pred.shape[0]), int(0.8 * adata_query_pred.shape[0]))
                adata_query_stim_idx = np.random.choice(range(adata_query_stim.shape[0]), int(0.8 * adata_query_stim.shape[0]))
                x = adata_query_pred[: , diff_genes.tolist()[: 50]].X.toarray()[adata_query_pred_idx]
                y = adata_query_stim[: , diff_genes.tolist()[: 50]].X.toarray()[adata_query_stim_idx]
                mmd_values[0, j] = compute_mmd_loss(x, y, gammas = gammas)
            mmd_means.append(mmd_values.mean())
            mmd_stds.append(mmd_values.std())
            print(f'{model_name} on {query_key}: MMD_mean = {mmd_values.mean()}, MMD_std = {mmd_values.std()}')
        gene_sets = ['All genes', 'Top 100 DEGs'] * len(query_keys)
        query_keys_fordf = []
        for query_key in query_keys:
            query_keys_fordf.extend([query_key] * 2)
        r2mean_df = pd.DataFrame({
            'R2mean means': r2mean_means,
            'R2mean stds': r2mean_stds,
            'Gene set': gene_sets,
            'Cell label': query_keys_fordf,
            'Model': [model_name] * 2 * len(query_keys),
        })
        r2mean_df = r2mean_df.sort_values(by = 'Cell label')
        r2mean_dfs.append(r2mean_df[r2mean_df['Gene set'] == 'All genes'])
        print(f'{model_name} R2mean_df:')
        print(r2mean_df)
        l2mean_df = pd.DataFrame({
            'L2mean means': l2mean_means,
            'L2mean stds': l2mean_stds,
            'Gene set': gene_sets,
            'Cell label': query_keys_fordf,
            'Model': [model_name] * 2 * len(query_keys),
        })
        l2mean_df = l2mean_df.sort_values(by = 'Cell label')
        l2mean_dfs.append(l2mean_df[l2mean_df['Gene set'] == 'All genes'])
        print(f'{model_name} L2mean_df:')
        print(l2mean_df)
        r2var_df = pd.DataFrame({
            'R2var means': r2var_means,
            'R2var stds': r2var_stds,
            'Gene set': gene_sets,
            'Cell label': query_keys_fordf,
            'Model': [model_name] * 2 * len(query_keys),
        })
        r2var_df = r2var_df.sort_values(by = 'Cell label')
        r2var_dfs.append(r2var_df[r2var_df['Gene set'] == 'All genes'])
        print(f'{model_name} R2var_df:')
        print(r2var_df)
        l2var_df = pd.DataFrame({
            'L2var means': l2var_means,
            'L2var stds': l2var_stds,
            'Gene set': gene_sets,
            'Cell label': query_keys_fordf,
            'Model': [model_name] * 2 * len(query_keys),
        })
        l2var_df = l2var_df.sort_values(by = 'Cell label')
        l2var_dfs.append(l2var_df[l2var_df['Gene set'] == 'All genes'])
        print(f'{model_name} L2var_df:')
        print(l2var_df)
        mmd_df = pd.DataFrame({
            'MMD means': mmd_means,
            'MMD stds': mmd_stds,
            'Cell label': query_keys,
            'Model': [model_name] * len(query_keys),
        })
        mmd_df = mmd_df.sort_values(by = 'Cell label')
        mmd_dfs.append(mmd_df)
        print(f'{model_name} MMD_df:')
        print(mmd_df)
        grouped_barplot(
            r2mean_df,
            'Cell label',
            'Gene set',
            'R2mean means',
            'R2mean stds',
            legend = True,
            filename = f'../Figures/{experiment_name}/{model_name}_on_{data_file}_barplot_R2mean.png',
            width = 2,
            ylabel = r'$\mathrm{R^2_{\mathrm{\mathsf{mean}}}}$',
        )
        grouped_barplot(
            l2mean_df,
            'Cell label',
            'Gene set',
            'L2mean means',
            'L2mean stds',
            legend = True,
            filename = f'../Figures/{experiment_name}/{model_name}_on_{data_file}_barplot_L2mean.png',
            width = 2,
            ylabel = r'$\mathrm{L^2_{\mathrm{\mathsf{mean}}}}$',
        )
        grouped_barplot(
            r2var_df,
            'Cell label',
            'Gene set',
            'R2var means',
            'R2var stds',
            legend = True,
            filename = f'../Figures/{experiment_name}/{model_name}_on_{data_file}_barplot_R2var.png',
            width = 2,
            ylabel = r'$\mathrm{R^2_{\mathrm{\mathsf{var}}}}$'
        )
        grouped_barplot(
            l2var_df,
            'Cell label',
            'Gene set',
            'L2var means',
            'L2var stds',
            legend = True,
            filename = f'../Figures/{experiment_name}/{model_name}_on_{data_file}_barplot_L2var.png',
            width = 2,
            ylabel = r'$\mathrm{L^2_{\mathrm{\mathsf{var}}}}$'
        )
        grouped_barplot(
            mmd_df,
            'Cell label',
            'Model',
            'MMD means',
            'MMD stds',
            legend = False,
            filename = f'../Figures/{experiment_name}/{model_name}_on_{data_file}_barplot_MMD.png',
            offset = 0,
            width = 1,
            ylabel = r'$\mathrm{MMD}$'
        )
    r2mean_final_df = pd.concat(r2mean_dfs, ignore_index = True)
    l2mean_final_df = pd.concat(l2mean_dfs, ignore_index = True)
    r2var_final_df = pd.concat(r2var_dfs, ignore_index = True)
    l2var_final_df = pd.concat(l2var_dfs, ignore_index = True)
    mmd_final_df = pd.concat(mmd_dfs, ignore_index = True)
    r2mean_final_df = r2mean_final_df.sort_values(by = ['Cell label', 'Model'])
    print(r2mean_final_df)
    l2mean_final_df = l2mean_final_df.sort_values(by = ['Cell label', 'Model'])
    print(l2mean_final_df)
    r2var_final_df = r2var_final_df.sort_values(by = ['Cell label', 'Model'])
    print(r2var_final_df)
    l2var_final_df = l2var_final_df.sort_values(by = ['Cell label', 'Model'])
    print(l2var_final_df)
    mmd_final_df = mmd_final_df.sort_values(by = ['Cell label', 'Model'])
    print(mmd_final_df)
    grouped_barplot(
        r2mean_final_df.replace(
            'biolord',
            'Biolord',
        ).replace(
            'identity',
            'Identity',
        ),
        'Cell label',
        'Model',
        'R2mean means',
        'R2mean stds',
        put_label = False,
        legend = True,
        filename = f'../Figures/{experiment_name}/All_models_on_{data_file}_barplot_R2mean.png',
        offset = 1.875,
        width = 6,
        ylabel = r'$\mathrm{R^2_{\mathrm{\mathsf{mean}}}}$'
    )
    grouped_barplot(
        l2mean_final_df.replace(
            'biolord',
            'Biolord',
        ).replace(
            'identity',
            'Identity',
        ),
        'Cell label',
        'Model',
        'L2mean means',
        'L2mean stds',
        put_label = False,
        legend = True,
        filename = f'../Figures/{experiment_name}/All_models_on_{data_file}_barplot_L2mean.png',
        offset = 1.875,
        width = 6,
        ylabel = r'$\mathrm{L^2_{\mathrm{\mathsf{mean}}}}$'
    )
    grouped_barplot(
        r2var_final_df.replace(
            'biolord',
            'Biolord',
        ).replace(
            'identity',
            'Identity',
        ),
        'Cell label',
        'Model',
        'R2var means',
        'R2var stds',
        put_label = False,
        legend = True,
        filename = f'../Figures/{experiment_name}/All_models_on_{data_file}_barplot_R2var.png',
        offset = 1.875,
        width = 6,
        ylabel = r'$\mathrm{R^2_{\mathrm{\mathsf{var}}}}$'
    )
    grouped_barplot(
        l2var_final_df.replace(
            'biolord',
            'Biolord',
        ).replace(
            'identity',
            'Identity',
        ),
        'Cell label',
        'Model',
        'L2var means',
        'L2var stds',
        put_label = False,
        legend = True,
        filename = f'../Figures/{experiment_name}/All_models_on_{data_file}_barplot_L2var.png',
        offset = 1.875,
        width = 6,
        ylabel = r'$\mathrm{L^2_{\mathrm{\mathsf{var}}}}$'
    )
    grouped_barplot(
        mmd_final_df.replace(
            'biolord',
            'Biolord',
        ).replace(
            'identity',
            'Identity',
        ),
        'Cell label',
        'Model',
        'MMD means',
        'MMD stds',
        put_label = False,
        legend = True,
        filename = f'../Figures/{experiment_name}/All_models_on_{data_file}_barplot_MMD.png',
        offset = 1.875,
        width = 6,
        ylabel = r'$\mathrm{MMD}$'
    )
if __name__ == '__main__':
    plot_result()