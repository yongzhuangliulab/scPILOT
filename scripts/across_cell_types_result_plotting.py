import time
import math
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import pairwise
from matplotlib import pyplot as plt
def mmd_distance(x, y, gamma):
    xx = pairwise.rbf_kernel(x, x, gamma)
    xy = pairwise.rbf_kernel(x, y, gamma)
    yy = pairwise.rbf_kernel(y, y, gamma)
    return xx.mean() + yy.mean() - 2 * xy.mean()
def compute_mmd_loss(lhs, rhs, gammas):
    return np.mean([mmd_distance(lhs, rhs, g) for g in gammas])
def plot_result(
    experiment_name = 'across_cell_types',
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
                     ha = 'center', va = 'bottom', fontsize = 25, rotation = 90)
    def grouped_barplot(df, cat, subcat, mean, std, val, filename, put_label = True, legend = False, offset = 0.375, width = 5, ylabel = r'$\mathrm{R^2}$'):
        plt.close('all')
        u = sorted(df[cat].unique().tolist())
        x_pos = np.arange(0, width * len(u), width)
        subx = sorted(df[subcat].unique().tolist())
        plt.figure(figsize = (6 * width, 10))
        for i, gr in enumerate(subx):
            dfg = df[df[subcat] == gr]
            b = plt.bar(
                x_pos + i / 1.25,
                dfg[mean].values,
                capsize = 10,
                alpha = 0.95,
                label = f'{gr}',
                yerr = np.vstack([
                    dfg[mean].values - np.array([np.percentile(values, 2.5) for values in dfg[val].values]),
                    np.array([np.percentile(values, 97.5) for values in dfg[val].values]) - dfg[mean].values,
                ]),
            )
            a = np.array([values.ravel() for values in dfg[val].values])
            x_base = x_pos + i / 1.25
            jitter = np.random.normal(loc = 0.0, scale = 0.01, size = a.shape)
            x_jittered = x_base.reshape(-1, 1) + jitter
            plt.plot(x_jittered, a, '.', color = 'black', alpha = 0.5)
            if put_label:
                autolabel(b)
        for j, l in enumerate(u):
            dfl = df[df[cat] == l]
            x_posl = x_pos[j]
            x_right = x_posl + (len(subx) - 1) / 1.25
            values_right = dfl[dfl[subcat] == subx[-1]][val].values[0].ravel()
            y_interval = df[mean].values.max() / 8
            y_base = df[mean].values.max() + y_interval * (len(subx) - 1)
            for i, gr in enumerate(subx[: -1]):
                x_left = x_posl + i / 1.25
                y_lgr = y_base - y_interval * i
                plt.hlines(y_lgr, x_left, x_right)
                values_left = dfl[dfl[subcat] == gr][val].values[0].ravel()
                wsrt_less_res = stats.wilcoxon(values_left, values_right, alternative = 'less')
                wsrt_greater_res = stats.wilcoxon(values_left, values_right, alternative = 'greater')
                plt.text(
                    (x_left + x_right) / 2,
                    y_lgr + y_interval / 5,
                    f'p={wsrt_less_res.pvalue:.3g}, {wsrt_greater_res.pvalue:.3g}',
                    ha = 'center',
                    va = 'bottom',
                    fontsize = 25,
                )
        plt.ylabel(ylabel, fontsize = 25)
        plt.yticks(fontsize = 25)
        plt.xticks(x_pos + offset, u, fontsize = 30)
        if legend:
            plt.legend(bbox_to_anchor = (1.01, 0.5), loc = 'center left', borderaxespad = 0, fontsize = 30)
        sns.despine(ax = plt.gca(), top = True, right = True)
        plt.tight_layout()
        plt.savefig(filename, dpi = 300)
        plt.show()
    sns.set_theme(style = 'white', font = 'Arial', font_scale = 2)
    adata = ad.read_h5ad(f'../Data/{experiment_name}/{data_file}{file_type}')
    adata = adata[(adata.obs[cond_key] == ctrl_key) | (adata.obs[cond_key] == stim_key)].copy()
    print('adata:')
    print(adata)
    query_keys = sorted(adata.obs[cell_label_key].unique().tolist())
    for query_no, query_key in enumerate(query_keys):
        query_adatas = []
        start_time = time.time()
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
            sc.pl.umap(
                adata_query_stpr,
                color = cond_key.capitalize(),
                palette = {
                    stim_key.capitalize(): sns.color_palette()[0],
                    'Pred': sns.color_palette()[1],
                },
                legend_loc = None,
                title = '',
                frameon = False,
                show = False,
            )
            plt.savefig(f'../Figures/{experiment_name}/{model_name}_{data_file}_{query_key}_prediction.jpg', dpi = 300, bbox_inches = 'tight')
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
            ] else model_name}_{data_file}_{query_key}_prediction.jpg'))
            plt.title(model_name)
        plt.savefig(f'../Figures/{experiment_name}/All_models_{data_file}_{query_key}_prediction.jpg', dpi = 300, bbox_inches = 'tight')
        plt.close()
        end_time = time.time()
        print(f'======Prediction plotting for {query_no + 1}: {query_key} costs {end_time - start_time:.3f} secs.')
        start_time = time.time()
        adata_query_models = ad.concat(query_adatas)
        adata_query_models.obs = adata_query_models.obs.replace(
            ctrl_key, ctrl_key.capitalize()
        ).replace(stim_key, stim_key.capitalize()).replace('biolord', 'Biolord')
        groupby_order = [ctrl_key.capitalize(), stim_key.capitalize(), 'Biolord', 'CellOT', 'VAEGAN', 'scGen', 'scPILOT']
        adata_query = adata[adata.obs[cell_label_key] == query_key].copy()
        sc.tl.rank_genes_groups(adata_query, groupby = cond_key, method = 'wilcoxon', n_genes = 100)
        important_genes = adata_query.uns['rank_genes_groups']['names'][stim_key][: 10]
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
            sns.despine(ax = plt.gca(), top = True, right = True)
            plt.savefig(f'../Figures/{experiment_name}/All_models_on_{data_file}_{query_key}_{gene_name}_violin.jpg', dpi = 300, bbox_inches = 'tight')
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
            plt.imshow(plt.imread(f'../Figures/{experiment_name}/All_models_on_{data_file}_{query_key}_{gene_name}_violin.jpg'))
        plt.savefig(f'../Figures/{experiment_name}/All_models_on_{data_file}_{query_key}_important_genes_violin.jpg', dpi = 300, bbox_inches = 'tight')
        plt.close()
        sv = sc.pl.stacked_violin(
            adata_query_models,
            var_names = important_genes,
            groupby = cond_key,
            categories_order = groupby_order,
            swap_axes = True,
            dendrogram = False,
            show = False,
            return_fig = True,
        )
        sv_ax_dict = sv.get_axes()
        print(sv_ax_dict)
        mainplot_ax = sv_ax_dict['mainplot_ax']
        mainplot_ax.tick_params(labelsize = 13)
        color_legend_ax = sv_ax_dict['color_legend_ax']
        color_legend_ax.set_title('Median expression\nin group', fontsize = 13)
        color_legend_ax.tick_params(axis = 'x', labelsize = 12)
        plt.tight_layout()
        plt.savefig(f'../Figures/{experiment_name}/All_models_on_{data_file}_{query_key}_stacked_violin.jpg', dpi = 300, bbox_inches = 'tight')
        plt.close()
        end_time = time.time()
        print(f'======Violin plotting for {query_no + 1}: {query_key} costs {end_time - start_time:.3f} secs.')
    r2mean_dfs = []
    l2mean_dfs = []
    r2var_dfs = []
    l2var_dfs = []
    gammas = np.logspace(1, -3, num = 50)
    mmd_dfs = []
    for model_no, model_name in enumerate(sorted(model_names)):
        r2mean_means = []
        r2mean_stds = []
        r2mean_values_list = []
        l2mean_means = []
        l2mean_stds = []
        l2mean_values_list = []
        r2var_means = []
        r2var_stds = []
        r2var_values_list = []
        l2var_means = []
        l2var_stds = []
        l2var_values_list = []
        mmd_means = []
        mmd_stds = []
        mmd_values_list = []
        start_time = time.time()
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
                r2mean_values = np.zeros((1, 200))
                l2mean_values = np.zeros((1, 200))
                r2var_values = np.zeros((1, 200))
                l2var_values = np.zeros((1, 200))
                # 80% random sampling 200 times
                for j in range(200):
                    adata_query_pred_idx = np.random.choice(range(adata_query_pred.shape[0]), int(0.8 * adata_query_pred.shape[0]))
                    adata_query_stim_idx = np.random.choice(range(adata_query_stim.shape[0]), int(0.8 * adata_query_stim.shape[0]))
                    if gene_set == 'All genes':
                        x = np.average(
                            adata_query_pred[adata_query_pred_idx].X.toarray()
                            if hasattr(adata_query_pred[adata_query_pred_idx].X, 'toarray')
                            else adata_query_pred[adata_query_pred_idx].X,
                            axis = 0,
                        )
                        y = np.average(
                            adata_query_stim[adata_query_stim_idx].X.toarray()
                            if hasattr(adata_query_stim[adata_query_stim_idx].X, 'toarray')
                            else adata_query_stim[adata_query_stim_idx].X,
                            axis = 0,
                        )
                    else:
                        x = np.average(
                            adata_query_pred[adata_query_pred_idx, diff_genes.tolist()].X.toarray()
                            if hasattr(adata_query_pred[adata_query_pred_idx, diff_genes.tolist()].X, 'toarray')
                            else adata_query_pred[adata_query_pred_idx, diff_genes.tolist()].X,
                            axis = 0,
                        )
                        y = np.average(
                            adata_query_stim[adata_query_stim_idx, diff_genes.tolist()].X.toarray()
                            if hasattr(adata_query_stim[adata_query_stim_idx, diff_genes.tolist()].X, 'toarray')
                            else adata_query_stim[adata_query_stim_idx, diff_genes.tolist()].X,
                            axis = 0,
                        )
                    _, _, r_value, _, _ = stats.linregress(x, y)
                    r2mean_values[0, j] = r_value ** 2
                    l2mean_values[0, j] = math.sqrt(((x - y) ** 2).sum())
                    if gene_set == 'All genes':
                        x = np.var(
                            adata_query_pred[adata_query_pred_idx].X.toarray()
                            if hasattr(adata_query_pred[adata_query_pred_idx].X, 'toarray')
                            else adata_query_pred[adata_query_pred_idx].X,
                            axis = 0,
                        )
                        y = np.var(
                            adata_query_stim[adata_query_stim_idx].X.toarray()
                            if hasattr(adata_query_stim[adata_query_stim_idx].X, 'toarray')
                            else adata_query_stim[adata_query_stim_idx].X,
                            axis = 0,
                        )
                    else:
                        x = np.var(
                            adata_query_pred[adata_query_pred_idx, diff_genes.tolist()].X.toarray()
                            if hasattr(adata_query_pred[adata_query_pred_idx, diff_genes.tolist()].X, 'toarray')
                            else adata_query_pred[adata_query_pred_idx, diff_genes.tolist()].X,
                            axis = 0,
                        )
                        y = np.var(
                            adata_query_stim[adata_query_stim_idx, diff_genes.tolist()].X.toarray()
                            if hasattr(adata_query_stim[adata_query_stim_idx, diff_genes.tolist()].X, 'toarray')
                            else adata_query_stim[adata_query_stim_idx, diff_genes.tolist()].X,
                            axis = 0,
                        )
                    _, _, r_value, _, _ = stats.linregress(x, y)
                    r2var_values[0, j] = r_value ** 2
                    l2var_values[0, j] = math.sqrt(((x - y) ** 2).sum())
                r2mean_means.append(r2mean_values.mean())
                r2mean_stds.append(r2mean_values.std())
                r2mean_values_list.append(r2mean_values)
                l2mean_means.append(l2mean_values.mean())
                l2mean_stds.append(l2mean_values.std())
                l2mean_values_list.append(l2mean_values)
                r2var_means.append(r2var_values.mean())
                r2var_stds.append(r2var_values.std())
                r2var_values_list.append(r2var_values)
                l2var_means.append(l2var_values.mean())
                l2var_stds.append(l2var_values.std())
                l2var_values_list.append(l2var_values)
                print(f'{model_name} on {query_key}: R2mean_{gene_set}_mean = {r2mean_values.mean()}, R2mean_{gene_set}_std = {r2mean_values.std()}')
                print(f'{model_name} on {query_key}: L2mean_{gene_set}_mean = {l2mean_values.mean()}, L2mean_{gene_set}_std = {l2mean_values.std()}')
                print(f'{model_name} on {query_key}: R2var_{gene_set}_mean = {r2var_values.mean()}, R2var_{gene_set}_std = {r2var_values.std()}')
                print(f'{model_name} on {query_key}: L2var_{gene_set}_mean = {l2var_values.mean()}, L2var_{gene_set}_std = {l2var_values.std()}')
            mmd_values = np.zeros((1, 200))
            for j in range(200):
                adata_query_pred_idx = np.random.choice(range(adata_query_pred.shape[0]), int(0.8 * adata_query_pred.shape[0]))
                adata_query_stim_idx = np.random.choice(range(adata_query_stim.shape[0]), int(0.8 * adata_query_stim.shape[0]))
                x = adata_query_pred[adata_query_pred_idx, diff_genes.tolist()[: 50]].X.toarray() if hasattr(
                    adata_query_pred[adata_query_pred_idx, diff_genes.tolist()[: 50]].X, 'toarray'
                ) else adata_query_pred[adata_query_pred_idx, diff_genes.tolist()[: 50]].X
                y = adata_query_stim[adata_query_stim_idx, diff_genes.tolist()[: 50]].X.toarray() if hasattr(
                    adata_query_stim[adata_query_stim_idx, diff_genes.tolist()[: 50]].X, 'toarray'
                ) else adata_query_stim[adata_query_stim_idx, diff_genes.tolist()[: 50]].X
                mmd_values[0, j] = compute_mmd_loss(x, y, gammas = gammas)
            mmd_means.append(mmd_values.mean())
            mmd_stds.append(mmd_values.std())
            mmd_values_list.append(mmd_values)
            print(f'{model_name} on {query_key}: MMD_mean = {mmd_values.mean()}, MMD_std = {mmd_values.std()}')
        end_time = time.time()
        print(f'======Metrics computing for {model_no + 1}: {model_name} costs {end_time - start_time:.3f} secs.')
        gene_sets = ['All genes', 'Top 100 DEGs'] * len(query_keys)
        query_keys_fordf = []
        for query_key in query_keys:
            query_keys_fordf.extend([query_key] * 2)
        r2mean_df = pd.DataFrame({
            'R2mean means': r2mean_means,
            'R2mean stds': r2mean_stds,
            'R2mean values ndarrays': r2mean_values_list,
            'Gene set': gene_sets,
            'Cell label': query_keys_fordf,
            'Model': [model_name] * 2 * len(query_keys),
        })
        r2mean_df = r2mean_df.sort_values(by = 'Cell label')
        r2mean_dfs.append(r2mean_df)
        print(f'{model_name} R2mean_df:')
        print(r2mean_df)
        l2mean_df = pd.DataFrame({
            'L2mean means': l2mean_means,
            'L2mean stds': l2mean_stds,
            'L2mean values ndarrays': l2mean_values_list,
            'Gene set': gene_sets,
            'Cell label': query_keys_fordf,
            'Model': [model_name] * 2 * len(query_keys),
        })
        l2mean_df = l2mean_df.sort_values(by = 'Cell label')
        l2mean_dfs.append(l2mean_df)
        print(f'{model_name} L2mean_df:')
        print(l2mean_df)
        r2var_df = pd.DataFrame({
            'R2var means': r2var_means,
            'R2var stds': r2var_stds,
            'R2var values ndarrays': r2var_values_list,
            'Gene set': gene_sets,
            'Cell label': query_keys_fordf,
            'Model': [model_name] * 2 * len(query_keys),
        })
        r2var_df = r2var_df.sort_values(by = 'Cell label')
        r2var_dfs.append(r2var_df)
        print(f'{model_name} R2var_df:')
        print(r2var_df)
        l2var_df = pd.DataFrame({
            'L2var means': l2var_means,
            'L2var stds': l2var_stds,
            'L2var values ndarrays': l2var_values_list,
            'Gene set': gene_sets,
            'Cell label': query_keys_fordf,
            'Model': [model_name] * 2 * len(query_keys),
        })
        l2var_df = l2var_df.sort_values(by = 'Cell label')
        l2var_dfs.append(l2var_df)
        print(f'{model_name} L2var_df:')
        print(l2var_df)
        mmd_df = pd.DataFrame({
            'MMD means': mmd_means,
            'MMD stds': mmd_stds,
            'MMD values ndarrays': mmd_values_list,
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
            'R2mean values ndarrays',
            legend = True,
            filename = f'../Figures/{experiment_name}/{model_name}_on_{data_file}_barplot_R2mean.jpg',
            width = 2,
            ylabel = r'$\mathrm{R^2_{\mathrm{\mathsf{mean}}}}$',
        )
        grouped_barplot(
            l2mean_df,
            'Cell label',
            'Gene set',
            'L2mean means',
            'L2mean stds',
            'L2mean values ndarrays',
            legend = True,
            filename = f'../Figures/{experiment_name}/{model_name}_on_{data_file}_barplot_L2mean.jpg',
            width = 2,
            ylabel = r'$\mathrm{L^2_{\mathrm{\mathsf{mean}}}}$',
        )
        grouped_barplot(
            r2var_df,
            'Cell label',
            'Gene set',
            'R2var means',
            'R2var stds',
            'R2var values ndarrays',
            legend = True,
            filename = f'../Figures/{experiment_name}/{model_name}_on_{data_file}_barplot_R2var.jpg',
            width = 2,
            ylabel = r'$\mathrm{R^2_{\mathrm{\mathsf{var}}}}$'
        )
        grouped_barplot(
            l2var_df,
            'Cell label',
            'Gene set',
            'L2var means',
            'L2var stds',
            'L2var values ndarrays',
            legend = True,
            filename = f'../Figures/{experiment_name}/{model_name}_on_{data_file}_barplot_L2var.jpg',
            width = 2,
            ylabel = r'$\mathrm{L^2_{\mathrm{\mathsf{var}}}}$'
        )
        grouped_barplot(
            mmd_df,
            'Cell label',
            'Model',
            'MMD means',
            'MMD stds',
            'MMD values ndarrays',
            legend = False,
            filename = f'../Figures/{experiment_name}/{model_name}_on_{data_file}_barplot_MMD.jpg',
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
    r2mean_final_df.to_pickle(f'../DataFrames/{experiment_name}/r2mean_final.pkl')
    l2mean_final_df = l2mean_final_df.sort_values(by = ['Cell label', 'Model'])
    l2mean_final_df.to_pickle(f'../DataFrames/{experiment_name}/l2mean_final.pkl')
    r2var_final_df = r2var_final_df.sort_values(by = ['Cell label', 'Model'])
    r2var_final_df.to_pickle(f'../DataFrames/{experiment_name}/r2var_final.pkl')
    l2var_final_df = l2var_final_df.sort_values(by = ['Cell label', 'Model'])
    l2var_final_df.to_pickle(f'../DataFrames/{experiment_name}/l2var_final.pkl')
    mmd_final_df = mmd_final_df.sort_values(by = ['Cell label', 'Model'])
    mmd_final_df.to_pickle(f'../DataFrames/{experiment_name}/mmd_final.pkl')
    print(r2mean_final_df)
    print(l2mean_final_df)
    print(r2var_final_df)
    print(l2var_final_df)
    print(mmd_final_df)
    r2mean_final_df['Model'] = r2mean_final_df['Model'].replace({'biolord': 'Biolord', 'identity': 'Identity'})
    grouped_barplot(
        r2mean_final_df[r2mean_final_df['Gene set'] == 'All genes'],
        'Cell label',
        'Model',
        'R2mean means',
        'R2mean stds',
        'R2mean values ndarrays',
        legend = True,
        filename = f'../Figures/{experiment_name}/All_models_on_{data_file}_barplot_R2mean.jpg',
        offset = 1.875,
        width = 6,
        ylabel = r'$\mathrm{R^2_{\mathrm{\mathsf{mean}}}}$'
    )
    grouped_barplot(
        r2mean_final_df[r2mean_final_df['Gene set'] == 'Top 100 DEGs'],
        'Cell label',
        'Model',
        'R2mean means',
        'R2mean stds',
        'R2mean values ndarrays',
        legend = True,
        filename = f'../Figures/{experiment_name}/All_models_on_{data_file}_barplot_R2mean_T100.jpg',
        offset = 1.875,
        width = 6,
        ylabel = r'$\mathrm{R^2_{\mathrm{\mathsf{mean, T100}}}}$'
    )
    l2mean_final_df['Model'] = l2mean_final_df['Model'].replace({'biolord': 'Biolord', 'identity': 'Identity'})
    grouped_barplot(
        l2mean_final_df[l2mean_final_df['Gene set'] == 'All genes'],
        'Cell label',
        'Model',
        'L2mean means',
        'L2mean stds',
        'L2mean values ndarrays',
        legend = True,
        filename = f'../Figures/{experiment_name}/All_models_on_{data_file}_barplot_L2mean.jpg',
        offset = 1.875,
        width = 6,
        ylabel = r'$\mathrm{L^2_{\mathrm{\mathsf{mean}}}}$'
    )
    grouped_barplot(
        l2mean_final_df[l2mean_final_df['Gene set'] == 'Top 100 DEGs'],
        'Cell label',
        'Model',
        'L2mean means',
        'L2mean stds',
        'L2mean values ndarrays',
        legend = True,
        filename = f'../Figures/{experiment_name}/All_models_on_{data_file}_barplot_L2mean_T100.jpg',
        offset = 1.875,
        width = 6,
        ylabel = r'$\mathrm{L^2_{\mathrm{\mathsf{mean, T100}}}}$'
    )
    r2var_final_df['Model'] = r2var_final_df['Model'].replace({'biolord': 'Biolord', 'identity': 'Identity'})
    grouped_barplot(
        r2var_final_df[r2var_final_df['Gene set'] == 'All genes'],
        'Cell label',
        'Model',
        'R2var means',
        'R2var stds',
        'R2var values ndarrays',
        legend = True,
        filename = f'../Figures/{experiment_name}/All_models_on_{data_file}_barplot_R2var.jpg',
        offset = 1.875,
        width = 6,
        ylabel = r'$\mathrm{R^2_{\mathrm{\mathsf{var}}}}$'
    )
    grouped_barplot(
        r2var_final_df[r2var_final_df['Gene set'] == 'Top 100 DEGs'],
        'Cell label',
        'Model',
        'R2var means',
        'R2var stds',
        'R2var values ndarrays',
        legend = True,
        filename = f'../Figures/{experiment_name}/All_models_on_{data_file}_barplot_R2var_T100.jpg',
        offset = 1.875,
        width = 6,
        ylabel = r'$\mathrm{R^2_{\mathrm{\mathsf{var, T100}}}}$'
    )
    l2var_final_df['Model'] = l2var_final_df['Model'].replace({'biolord': 'Biolord', 'identity': 'Identity'})
    grouped_barplot(
        l2var_final_df[l2var_final_df['Gene set'] == 'All genes'],
        'Cell label',
        'Model',
        'L2var means',
        'L2var stds',
        'L2var values ndarrays',
        legend = True,
        filename = f'../Figures/{experiment_name}/All_models_on_{data_file}_barplot_L2var.jpg',
        offset = 1.875,
        width = 6,
        ylabel = r'$\mathrm{L^2_{\mathrm{\mathsf{var}}}}$'
    )
    grouped_barplot(
        l2var_final_df[l2var_final_df['Gene set'] == 'Top 100 DEGs'],
        'Cell label',
        'Model',
        'L2var means',
        'L2var stds',
        'L2var values ndarrays',
        legend = True,
        filename = f'../Figures/{experiment_name}/All_models_on_{data_file}_barplot_L2var_T100.jpg',
        offset = 1.875,
        width = 6,
        ylabel = r'$\mathrm{L^2_{\mathrm{\mathsf{var, T100}}}}$'
    )
    mmd_final_df['Model'] = mmd_final_df['Model'].replace({'biolord': 'Biolord', 'identity': 'Identity'})
    grouped_barplot(
        mmd_final_df,
        'Cell label',
        'Model',
        'MMD means',
        'MMD stds',
        'MMD values ndarrays',
        legend = True,
        filename = f'../Figures/{experiment_name}/All_models_on_{data_file}_barplot_MMD.jpg',
        offset = 1.875,
        width = 6,
        ylabel = r'$\mathrm{MMD}$'
    )
if __name__ == '__main__':
    plot_result()
    print('Done')