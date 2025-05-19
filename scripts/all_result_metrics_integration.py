import scanpy as sc
import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from math import sqrt
def metric41(
    experiment_name,
    model_names,
    data_file,
    file_type,
    cond_key,
    ctrl_key,
    stim_key,
    cell_label_key,
    drug_name,
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
    query_keys = sorted(adata.obs[cell_label_key].unique().tolist())
    r2mean_dfs = []
    l2mean_dfs = []
    r2var_dfs = []
    l2var_dfs = []
    for model_no, model_name in enumerate(sorted(model_names)):
        r2mean_means = []
        r2mean_stds = []
        l2mean_means = []
        l2mean_stds = []
        r2var_means = []
        r2var_stds = []
        l2var_means = []
        l2var_stds = []
        for query_no, query_key in enumerate(query_keys):
            adata_query_eval = ad.read_h5ad(f'../Result_anndata/{experiment_name}/{experiment_name}_{model_name}_{data_file}_{query_key}{drug_name}{file_type}')
            adata_query_pred = adata_query_eval[adata_query_eval.obs[cond_key] == 'pred']
            adata_query_ctrl = adata_query_eval[adata_query_eval.obs[cond_key] == ctrl_key]
            adata_query_stim = adata_query_eval[adata_query_eval.obs[cond_key] == stim_key]
            adata_query = ad.concat([adata_query_ctrl, adata_query_stim])
            sc.tl.rank_genes_groups(adata_query, groupby = cond_key, method = 'wilcoxon', n_genes = 100)
            diff_genes = adata_query.uns['rank_genes_groups']['names'][stim_key]
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
            for j in range(10):
                adata_query_pred_idx = np.random.choice(range(adata_query_pred.shape[0]), int(0.8 * adata_query_pred.shape[0]))
                adata_query_stim_idx = np.random.choice(range(adata_query_stim.shape[0]), int(0.8 * adata_query_stim.shape[0]))
                x = adata_query_pred[: , diff_genes.tolist()[: 50]].X.toarray()[adata_query_pred_idx]
                y = adata_query_stim[: , diff_genes.tolist()[: 50]].X.toarray()[adata_query_stim_idx]
        gene_sets = ['All genes', 'Top 100 DEGs'] * len(query_keys)
        cell_labels_fordf = []
        for cl in query_keys:
            cell_labels_fordf.extend([cl] * 2)
        r2mean_df = pd.DataFrame({
            'R2mean means': r2mean_means,
            'R2mean stds': r2mean_stds,
            'Gene set': gene_sets,
            'Cell label': cell_labels_fordf,
            'Model': [model_name] * 2 * len(query_keys),
        })
        r2mean_df = r2mean_df.sort_values(by = 'Cell label')
        r2mean_dfs.append(r2mean_df)
        l2mean_df = pd.DataFrame({
            'L2mean means': l2mean_means,
            'L2mean stds': l2mean_stds,
            'Gene set': gene_sets,
            'Cell label': cell_labels_fordf,
            'Model': [model_name] * 2 * len(query_keys),
        })
        l2mean_df = l2mean_df.sort_values(by = 'Cell label')
        l2mean_dfs.append(l2mean_df)
        r2var_df = pd.DataFrame({
            'R2var means': r2var_means,
            'R2var stds': r2var_stds,
            'Gene set': gene_sets,
            'Cell label': cell_labels_fordf,
            'Model': [model_name] * 2 * len(query_keys),
        })
        r2var_df = r2var_df.sort_values(by = 'Cell label')
        r2var_dfs.append(r2var_df)
        l2var_df = pd.DataFrame({
            'L2var means': l2var_means,
            'L2var stds': l2var_stds,
            'Gene set': gene_sets,
            'Cell label': cell_labels_fordf,
            'Model': [model_name] * 2 * len(query_keys),
        })
        l2var_df = l2var_df.sort_values(by = 'Cell label')
        l2var_dfs.append(l2var_df)
        grouped_barplot(
            r2mean_df,
            'Cell label',
            'Gene set',
            'R2mean means',
            'R2mean stds',
            legend = True,
            filename = f'../Figures/{experiment_name}/{model_name}_on_{data_file}{drug_name}_barplot_R2mean.png',
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
            filename = f'../Figures/{experiment_name}/{model_name}_on_{data_file}{drug_name}_barplot_L2mean.png',
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
            filename = f'../Figures/{experiment_name}/{model_name}_on_{data_file}{drug_name}_barplot_R2var.png',
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
            filename = f'../Figures/{experiment_name}/{model_name}_on_{data_file}{drug_name}_barplot_L2var.png',
            width = 2,
            ylabel = r'$\mathrm{L^2_{\mathrm{\mathsf{var}}}}$'
        )
    r2mean_final_df = pd.concat(r2mean_dfs, ignore_index = True)
    l2mean_final_df = pd.concat(l2mean_dfs, ignore_index = True)
    r2var_final_df = pd.concat(r2var_dfs, ignore_index = True)
    l2var_final_df = pd.concat(l2var_dfs, ignore_index = True)
    r2mean_final_df = r2mean_final_df.sort_values(by = ['Cell label', 'Model'])
    l2mean_final_df = l2mean_final_df.sort_values(by = ['Cell label', 'Model'])
    r2var_final_df = r2var_final_df.sort_values(by = ['Cell label', 'Model'])
    l2var_final_df = l2var_final_df.sort_values(by = ['Cell label', 'Model'])
    for gene_set in ['All genes', 'Top 100 DEGs']:
        grouped_barplot(
            r2mean_final_df[r2mean_final_df['Gene set'] == gene_set].replace(
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
            filename = f'../Figures/{experiment_name}/All_models_on_{data_file}{drug_name}_barplot_R2mean_{gene_set}.png',
            offset = 1.875,
            width = 6,
            ylabel = r'$\mathrm{R^2_{\mathrm{\mathsf{mean}}}}$'
        )
        grouped_barplot(
            l2mean_final_df[l2mean_final_df['Gene set'] == gene_set].replace(
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
            filename = f'../Figures/{experiment_name}/All_models_on_{data_file}{drug_name}_barplot_L2mean_{gene_set}.png',
            offset = 1.875,
            width = 6,
            ylabel = r'$\mathrm{L^2_{\mathrm{\mathsf{mean}}}}$'
        )
        grouped_barplot(
            r2var_final_df[r2var_final_df['Gene set'] == gene_set].replace(
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
            filename = f'../Figures/{experiment_name}/All_models_on_{data_file}{drug_name}_barplot_R2var_{gene_set}.png',
            offset = 1.875,
            width = 6,
            ylabel = r'$\mathrm{R^2_{\mathrm{\mathsf{var}}}}$'
        )
        grouped_barplot(
            l2var_final_df[l2var_final_df['Gene set'] == gene_set].replace(
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
            filename = f'../Figures/{experiment_name}/All_models_on_{data_file}{drug_name}_barplot_L2var_{gene_set}.png',
            offset = 1.875,
            width = 6,
            ylabel = r'$\mathrm{L^2_{\mathrm{\mathsf{var}}}}$'
        )
def metrics_integration(argument_dict):
    experiment_name = argument_dict['experiment_name']
    model_names = argument_dict['model_names']
    data_file = argument_dict['data_file']
    file_type = argument_dict['file_type']
    cond_key = argument_dict['cond_key']
    ctrl_key = argument_dict['ctrl_key']
    stim_key = argument_dict['stim_key']
    cell_label_key = argument_dict['cell_label_key']
    if isinstance(stim_key, str):
        metric41(
            experiment_name,
            model_names,
            data_file,
            file_type,
            cond_key,
            ctrl_key,
            stim_key,
            cell_label_key,
            '',
        )
    else:
        for drug_name in stim_key:
            metric41(
                experiment_name,
                model_names,
                data_file,
                file_type,
                cond_key,
                ctrl_key,
                drug_name,
                cell_label_key,
                f'_{drug_name}',
            )
if __name__ == '__main__':
    experiment_names = ['R1R2', 'R3', 'R4', 'R5', 'R6', 'R4hh', 'R5hh', 'R6hh']
    model_names = ['scPILOT', 'scGen', 'CellOT', 'biolord', 'identity', 'VAEGAN']
    data_files = ['pbmc', 'pbmc_patients', 'sciplex3_partial', 'species', 'differentiation', 'sciplex3_partial', 'species', 'differentiation']
    file_type = '.h5ad'
    cond_keys = ['condition', 'condition', 'drug', 'condition', 'condition', 'drug', 'condition', 'condition']
    ctrl_keys = ['control', 'ctrl', 'control', 'unst', 'control', 'control', 'unst', 'control']
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
    stim_keys = ['stimulated', 'stim', drug_names, 'LPS6', 'developed', drug_names, 'LPS6', 'developed']
    cell_label_keys = ['cell_type', 'sample_id', 'cell_type', 'species', 'population', 'cell_type', 'species', 'population']
    argument_dicts = []
    for i in range(len(experiment_names)):
        argument_dicts.append(
            {
                'experiment_name': experiment_names[i],
                'model_names': model_names,
                'data_file': data_files[i],
                'file_type': file_type,
                'cond_key': cond_keys[i],
                'ctrl_key': ctrl_keys[i],
                'stim_key': stim_keys[i],
                'cell_label_key': cell_label_keys[i],
            }
        )
    for argument_dict in argument_dicts:
        metrics_integration(argument_dict)