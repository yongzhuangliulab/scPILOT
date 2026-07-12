import os
import time
import argparse

import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise
from matplotlib import pyplot as plt


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------


def ensure_dirs(*paths: str):
    for path in paths:
        os.makedirs(path, exist_ok=True)


def save_figure_jpg_pdf(path_to_save: str, dpi: int = 300, bbox_inches: str = 'tight'):
    """Save current matplotlib figure as both JPG and PDF."""
    root, _ = os.path.splitext(path_to_save)
    plt.savefig(root + '.jpg', dpi=dpi, bbox_inches=bbox_inches)
    plt.savefig(root + '.pdf', bbox_inches=bbox_inches)


def to_dense_array(x):
    return x.toarray() if hasattr(x, 'toarray') else np.asarray(x)


def get_X(adata_obj, genes=None):
    x = adata_obj[:, genes].X if genes is not None else adata_obj.X
    return to_dense_array(x)


def model_display_name(model_name: str) -> str:
    name_map = {
        'identity': 'Identity',
        'biolord': 'Biolord',
        'CellOT': 'CellOT',
        'VAEGAN': 'VAEGAN',
        'scGen': 'scGen',
        'scPILOT': 'scPILOT',
    }
    return name_map.get(model_name, model_name)


def model_marker(model_name: str) -> str:
    marker_map = {
        'identity': 'o',   # circle
        'scGen': 's',      # square
        'biolord': '^',    # triangle up
        'CellOT': 'D',     # diamond
        'VAEGAN': 'P',     # filled plus
        'scPILOT': '*',    # star
    }
    return marker_map.get(model_name, 'o')


def model_marker_size(model_name: str, base_size: float = 4.0) -> float:
    size_map = {
        'scPILOT': base_size + 1.2,
        'VAEGAN': base_size + 0.4,
    }
    return size_map.get(model_name, base_size)


def default_model_order(model_names):
    ordered = ['identity', 'scGen', 'biolord', 'CellOT', 'VAEGAN', 'scPILOT']
    return [m for m in ordered if m in model_names]


def mmd_distance(x, y, gamma):
    xx = pairwise.rbf_kernel(x, x, gamma)
    xy = pairwise.rbf_kernel(x, y, gamma)
    yy = pairwise.rbf_kernel(y, y, gamma)
    return xx.mean() + yy.mean() - 2 * xy.mean()


def compute_mmd_loss(lhs, rhs, gammas):
    return float(np.mean([mmd_distance(lhs, rhs, g) for g in gammas]))


def safe_r2(x, y):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    _, _, r_value, _, _ = stats.linregress(x, y)
    return float(r_value ** 2)


def l2_distance(x, y):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    return float(np.sqrt(((x - y) ** 2).sum()))


def prediction_h5ad_path(experiment_name, model_name, data_file, query_key, seed, file_type):
    return (
        f'../Result_anndata/{experiment_name}/'
        f'{experiment_name}_{model_name}_{data_file}_{query_key}_seed{seed}{file_type}'
    )


def load_prediction_h5ad(
    experiment_name,
    model_name,
    data_file,
    query_key,
    seed,
    file_type,
    skip_missing=False,
):
    file_path = prediction_h5ad_path(
        experiment_name=experiment_name,
        model_name=model_name,
        data_file=data_file,
        query_key=query_key,
        seed=seed,
        file_type=file_type,
    )
    if not os.path.exists(file_path):
        msg = f'Missing prediction h5ad: {file_path}'
        if skip_missing:
            print(f'[SKIP] {msg}', flush=True)
            return None
        raise FileNotFoundError(msg)
    return ad.read_h5ad(file_path)


def get_top50_genes(adata, query_key, cond_key, stim_key, cell_label_key):
    """Define top50 DEGs once from real control vs stimulated cells."""
    adata_query = adata[adata.obs[cell_label_key] == query_key].copy()
    sc.tl.rank_genes_groups(
        adata_query,
        groupby=cond_key,
        method='wilcoxon',
        n_genes=50,
    )
    return adata_query.uns['rank_genes_groups']['names'][stim_key].tolist()


# -----------------------------------------------------------------------------
# Shared-coordinate UMAPs
# -----------------------------------------------------------------------------


def plot_shared_umaps_from_h5ad(
    adata,
    experiment_name='across_cell_types',
    model_names=('scPILOT', 'scGen', 'CellOT', 'biolord', 'identity', 'VAEGAN'),
    seeds=(1327, 1337, 1347),
    data_file='pbmc',
    file_type='.h5ad',
    cond_key='condition',
    ctrl_key='control',
    stim_key='stimulated',
    cell_label_key='cell_type',
    query_keys=None,
    skip_missing=False,
):
    ensure_dirs(f'../Figures/{experiment_name}')
    model_order = default_model_order(model_names)
    query_keys = query_keys or sorted(adata.obs[cell_label_key].unique().tolist())

    for seed in seeds:
        for query_no, query_key in enumerate(query_keys):
            start_time = time.time()
            print(
                f'======Shared UMAP plotting | seed={seed} | '
                f'{query_no + 1}: {query_key}======',
                flush=True,
            )

            query_umap_adatas = []
            stim_added = False

            for model_name in model_order:
                adata_eval = load_prediction_h5ad(
                    experiment_name=experiment_name,
                    model_name=model_name,
                    data_file=data_file,
                    query_key=query_key,
                    seed=seed,
                    file_type=file_type,
                    skip_missing=skip_missing,
                )
                if adata_eval is None:
                    continue

                adata_query_stim = adata_eval[adata_eval.obs[cond_key] == stim_key].copy()
                adata_query_pred = adata_eval[adata_eval.obs[cond_key] == 'pred'].copy()

                if not stim_added:
                    adata_stim_plot = adata_query_stim.copy()
                    adata_stim_plot.obs['UMAP group'] = 'Stimulated'
                    adata_stim_plot.obs['Model'] = 'Stimulated'
                    adata_stim_plot.obs_names = [
                        f'{idx}_stim_seed{seed}'
                        for idx in adata_stim_plot.obs_names.astype(str)
                    ]
                    query_umap_adatas.append(adata_stim_plot)
                    stim_added = True

                adata_pred_plot = adata_query_pred.copy()
                display_name = model_display_name(model_name)
                adata_pred_plot.obs['UMAP group'] = display_name
                adata_pred_plot.obs['Model'] = display_name
                adata_pred_plot.obs_names = [
                    f'{idx}_{model_name}_pred_seed{seed}'
                    for idx in adata_pred_plot.obs_names.astype(str)
                ]
                query_umap_adatas.append(adata_pred_plot)

            if len(query_umap_adatas) == 0:
                print(f'[SKIP] No UMAP data for {query_key}, seed={seed}', flush=True)
                continue

            adata_query_umap = ad.concat(query_umap_adatas)
            adata_query_umap.obs_names_make_unique()

            umap_group_order = [
                'Stimulated',
                *[model_display_name(m) for m in model_order],
            ]

            adata_query_umap.obs['UMAP group'] = pd.Categorical(
                adata_query_umap.obs['UMAP group'],
                categories=umap_group_order,
                ordered=True,
            )

            sc.pp.pca(adata_query_umap)
            sc.pp.neighbors(adata_query_umap)
            sc.tl.umap(adata_query_umap)

            # 1) All-in-one UMAP.
            sc.pl.umap(
                adata_query_umap,
                color='UMAP group',
                legend_loc='right margin',
                title=f'{query_key}, seed {seed}',
                frameon=False,
                show=False,
            )
            save_figure_jpg_pdf(
                f'../Figures/{experiment_name}/'
                f'All_models_{data_file}_{query_key}_seed{seed}_shared_umap_all_in_one.jpg'
            )
            plt.close('all')

            # 2) One panel per model, all sharing the same UMAP coordinates.
            xy = adata_query_umap.obsm['X_umap']
            xlim = (xy[:, 0].min(), xy[:, 0].max())
            ylim = (xy[:, 1].min(), xy[:, 1].max())

            n_models = len(model_order)
            fig, axes = plt.subplots(
                1,
                n_models,
                figsize=(5 * n_models, 5),
                squeeze=False,
            )
            axes = axes.ravel()
            mask_stim = adata_query_umap.obs['UMAP group'].values == 'Stimulated'

            for ax, model_name in zip(axes, model_order):
                display_name = model_display_name(model_name)
                mask_pred = adata_query_umap.obs['UMAP group'].values == display_name

                ax.scatter(
                    xy[mask_stim, 0],
                    xy[mask_stim, 1],
                    s=5,
                    alpha=0.55,
                    label='Stimulated',
                )
                ax.scatter(
                    xy[mask_pred, 0],
                    xy[mask_pred, 1],
                    s=5,
                    alpha=0.55,
                    label='Prediction',
                )

                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_title(display_name, fontsize=18)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel('')
                ax.set_ylabel('')
                for spine in ax.spines.values():
                    spine.set_visible(False)

            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc='center left',
                bbox_to_anchor=(1.01, 0.5),
                frameon=False,
            )
            plt.suptitle(f'{query_key}, seed {seed}', fontsize=20)
            plt.tight_layout()
            save_figure_jpg_pdf(
                f'../Figures/{experiment_name}/'
                f'All_models_{data_file}_{query_key}_seed{seed}_shared_umap_panels.jpg'
            )
            plt.close('all')

            print(
                f'======Shared UMAP plotting | seed={seed} | '
                f'{query_key} costs {time.time() - start_time:.3f} secs.======',
                flush=True,
            )


# -----------------------------------------------------------------------------
# Stacked violin and top-DEG recovery scores
# -----------------------------------------------------------------------------


def compute_gene_recovery_records(
    adata_ctrl,
    adata_stim,
    adata_pred,
    top10_genes,
    query_key,
    seed,
    model_name,
    eps=1e-8,
):
    records = []
    for rank, gene in enumerate(top10_genes, start=1):
        ctrl_values = get_X(adata_ctrl, [gene]).ravel()
        stim_values = get_X(adata_stim, [gene]).ravel()
        pred_values = get_X(adata_pred, [gene]).ravel()

        d_ctrl = wasserstein_distance(ctrl_values, stim_values)
        d_pred = wasserstein_distance(pred_values, stim_values)
        recovery_score = 1.0 - d_pred / (d_ctrl + eps)

        mean_ctrl = float(np.mean(ctrl_values))
        mean_stim = float(np.mean(stim_values))
        mean_pred = float(np.mean(pred_values))
        true_delta = mean_stim - mean_ctrl
        pred_delta = mean_pred - mean_ctrl
        direction_match = bool(np.sign(true_delta) == np.sign(pred_delta))

        records.append({
            'query_key': query_key,
            'seed': seed,
            'model': model_name,
            'model_display': model_display_name(model_name),
            'gene': gene,
            'rank': rank,
            'wasserstein_ctrl_to_stim': float(d_ctrl),
            'wasserstein_pred_to_stim': float(d_pred),
            'recovery_score': float(recovery_score),
            'mean_ctrl': mean_ctrl,
            'mean_stim': mean_stim,
            'mean_pred': mean_pred,
            'true_delta': float(true_delta),
            'pred_delta': float(pred_delta),
            'direction_match': direction_match,
        })
    return records


def summarize_recovery_gene_level(gene_df, experiment_name, model_names):
    if gene_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def delta_corr(group):
        x = group['true_delta'].values
        y = group['pred_delta'].values
        if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
            return np.nan
        return float(np.corrcoef(x, y)[0, 1])

    summary_records = []
    for (query_key, seed, model), group in gene_df.groupby(['query_key', 'seed', 'model']):
        summary_records.append({
            'query_key': query_key,
            'seed': seed,
            'model': model,
            'model_display': model_display_name(model),
            'top10_recovery_mean': float(group['recovery_score'].mean()),
            'top10_recovery_median': float(group['recovery_score'].median()),
            'top10_direction_accuracy': float(group['direction_match'].mean()),
            'top10_delta_corr': delta_corr(group),
        })

    seed_summary = pd.DataFrame(summary_records)
    seed_summary.to_csv(
        f'../DataFrames/{experiment_name}/topdeg_summary_recovery_seed_level.csv',
        index=False,
    )

    metric_cols = [
        'top10_recovery_mean',
        'top10_recovery_median',
        'top10_direction_accuracy',
        'top10_delta_corr',
    ]

    query_mean = (
        seed_summary
        .groupby(['query_key', 'model', 'model_display'])[metric_cols]
        .mean()
        .reset_index()
    )
    query_sd = (
        seed_summary
        .groupby(['query_key', 'model', 'model_display'])[metric_cols]
        .std()
        .reset_index()
    )
    query_summary = query_mean.merge(
        query_sd,
        on=['query_key', 'model', 'model_display'],
        suffixes=('_mean', '_sd'),
    )
    query_summary.to_csv(
        f'../DataFrames/{experiment_name}/topdeg_summary_recovery_query_level.csv',
        index=False,
    )

    # Paired two-sided Wilcoxon on seven query-level seed averages.
    wilcoxon_records = []
    baselines = [m for m in model_names if m != 'scPILOT']
    for metric in metric_cols:
        metric_query_col = f'{metric}_mean'
        pivot = query_summary.pivot(index='query_key', columns='model', values=metric_query_col)
        if 'scPILOT' not in pivot.columns:
            continue
        for baseline in baselines:
            if baseline not in pivot.columns:
                continue
            paired = pivot[['scPILOT', baseline]].dropna()
            if paired.shape[0] == 0:
                continue
            try:
                res = stats.wilcoxon(
                    paired['scPILOT'],
                    paired[baseline],
                    alternative='two-sided',
                )
                statistic = res.statistic
                pvalue = res.pvalue
            except ValueError:
                statistic = np.nan
                pvalue = np.nan

            scpilot_mean = paired['scPILOT'].mean()
            baseline_mean = paired[baseline].mean()
            raw_difference = scpilot_mean - baseline_mean
            wilcoxon_records.append({
                'metric': metric,
                'baseline': baseline,
                'n_query_keys': paired.shape[0],
                'scPILOT_mean': scpilot_mean,
                'baseline_mean': baseline_mean,
                'raw_difference_scPILOT_minus_baseline': raw_difference,
                'advantage_for_scPILOT': raw_difference,
                'better_direction': 'higher',
                'wilcoxon_statistic': statistic,
                'pvalue_two_sided': pvalue,
            })

    recovery_wilcoxon = pd.DataFrame(wilcoxon_records)
    recovery_wilcoxon.to_csv(
        f'../DataFrames/{experiment_name}/topdeg_recovery_wilcoxon_scPILOT_vs_baselines.csv',
        index=False,
    )

    return seed_summary, query_summary, recovery_wilcoxon


def plot_stacked_violins_and_recovery_from_h5ad(
    adata,
    experiment_name='across_cell_types',
    model_names=('scPILOT', 'scGen', 'CellOT', 'biolord', 'identity', 'VAEGAN'),
    seeds=(1327, 1337, 1347),
    data_file='pbmc',
    file_type='.h5ad',
    cond_key='condition',
    ctrl_key='control',
    stim_key='stimulated',
    cell_label_key='cell_type',
    query_keys=None,
    skip_missing=False,
):
    ensure_dirs(f'../Figures/{experiment_name}', f'../DataFrames/{experiment_name}')
    model_order = default_model_order(model_names)
    query_keys = query_keys or sorted(adata.obs[cell_label_key].unique().tolist())

    groupby_order = [
        ctrl_key.capitalize(),
        stim_key.capitalize(),
        *[model_display_name(m) for m in model_order],
    ]

    plot_group_key = 'Plot group'

    gene_level_records = []

    for query_no, query_key in enumerate(query_keys):
        start_time = time.time()
        top50_genes = get_top50_genes(
            adata=adata,
            query_key=query_key,
            cond_key=cond_key,
            stim_key=stim_key,
            cell_label_key=cell_label_key,
        )
        top10_genes = top50_genes[:10]

        for seed in seeds:
            print(
                f'======Stacked violin and recovery | seed={seed} | '
                f'{query_no + 1}: {query_key}======',
                flush=True,
            )
            query_violin_adatas = []
            truth_added = False
            adata_ctrl_ref = None
            adata_stim_ref = None
            heatmap_records_for_plot = []

            for model_name in model_order:
                adata_eval = load_prediction_h5ad(
                    experiment_name=experiment_name,
                    model_name=model_name,
                    data_file=data_file,
                    query_key=query_key,
                    seed=seed,
                    file_type=file_type,
                    skip_missing=skip_missing,
                )
                if adata_eval is None:
                    continue

                if not truth_added:
                    adata_ctrl_ref = adata_eval[adata_eval.obs[cond_key] == ctrl_key].copy()
                    adata_stim_ref = adata_eval[adata_eval.obs[cond_key] == stim_key].copy()

                    adata_ctrl_plot = adata_ctrl_ref.copy()
                    adata_stim_plot = adata_stim_ref.copy()
                    adata_ctrl_plot.obs[plot_group_key] = ctrl_key.capitalize()
                    adata_stim_plot.obs[plot_group_key] = stim_key.capitalize()
                    adata_ctrl_plot.obs_names = [
                        f'{idx}_ctrl_seed{seed}' for idx in adata_ctrl_plot.obs_names.astype(str)
                    ]
                    adata_stim_plot.obs_names = [
                        f'{idx}_stim_seed{seed}' for idx in adata_stim_plot.obs_names.astype(str)
                    ]
                    query_violin_adatas.extend([adata_ctrl_plot, adata_stim_plot])
                    truth_added = True

                adata_pred = adata_eval[adata_eval.obs[cond_key] == 'pred'].copy()
                display_name = model_display_name(model_name)

                gene_records = compute_gene_recovery_records(
                    adata_ctrl=adata_ctrl_ref,
                    adata_stim=adata_stim_ref,
                    adata_pred=adata_pred,
                    top10_genes=top10_genes,
                    query_key=query_key,
                    seed=seed,
                    model_name=model_name,
                )
                gene_level_records.extend(gene_records)
                heatmap_records_for_plot.extend(gene_records)

                adata_pred_plot = adata_pred.copy()
                adata_pred_plot.obs[plot_group_key] = display_name
                adata_pred_plot.obs_names = [
                    f'{idx}_{model_name}_pred_seed{seed}'
                    for idx in adata_pred_plot.obs_names.astype(str)
                ]
                query_violin_adatas.append(adata_pred_plot)

            if len(query_violin_adatas) == 0:
                print(f'[SKIP] No violin data for {query_key}, seed={seed}', flush=True)
                continue

            adata_query_models = ad.concat(query_violin_adatas)
            adata_query_models.obs_names_make_unique()

            # Stacked violin for top 10 DEGs.
            sv = sc.pl.stacked_violin(
                adata_query_models,
                var_names=top10_genes,
                groupby=plot_group_key,
                categories_order=groupby_order,
                swap_axes=True,
                dendrogram=False,
                show=False,
                return_fig=True,
            )
            try:
                sv_ax_dict = sv.get_axes()
                sv_ax_dict['mainplot_ax'].tick_params(labelsize=13)
                sv_ax_dict['color_legend_ax'].set_title('Median expression\nin group', fontsize=13)
                sv_ax_dict['color_legend_ax'].tick_params(axis='x', labelsize=12)
            except Exception:
                pass
            plt.suptitle(f'{query_key}, seed {seed}', fontsize=16, x=0.45, y=0.8)
            plt.tight_layout()
            save_figure_jpg_pdf(
                f'../Figures/{experiment_name}/'
                f'All_models_on_{data_file}_{query_key}_seed{seed}_stacked_violin.jpg'
            )
            plt.close('all')

            # Matched quantitative heatmap: recovery score, models x top10 genes.
            heatmap_df = pd.DataFrame(heatmap_records_for_plot)
            if not heatmap_df.empty:
                heatmap_matrix = heatmap_df.pivot(
                    index='gene',
                    columns='model_display',
                    values='recovery_score',
                )

                heatmap_matrix = heatmap_matrix.reindex(
                    [g for g in top10_genes if g in heatmap_matrix.index]
                )

                heatmap_matrix = heatmap_matrix[
                    [model_display_name(m) for m in model_order if model_display_name(m) in heatmap_matrix.columns]
                ]

                plt.figure(figsize=(0.85 * len(model_order) + 3, max(6, 0.45 * len(top10_genes) + 2)))
                sns.heatmap(
                    heatmap_matrix,
                    annot=True,
                    fmt='.2f',
                    cmap='vlag',
                    center=0,
                    linewidths=0.5,
                    linecolor='white',
                    cbar_kws={'label': 'Recovery score'},
                )
                plt.xlabel('Model')
                plt.ylabel('Top 10 DEGs')
                plt.title(f'{query_key}, seed {seed}')
                plt.tight_layout()
                save_figure_jpg_pdf(
                    f'../Figures/{experiment_name}/'
                    f'All_models_on_{data_file}_{query_key}_seed{seed}_top10_deg_recovery_heatmap.jpg'
                )
                plt.close('all')

        print(
            f'======Stacked violin and recovery for {query_no + 1}: {query_key} '
            f'costs {time.time() - start_time:.3f} secs.======',
            flush=True,
        )

    gene_df = pd.DataFrame(gene_level_records)
    gene_df.to_csv(
        f'../DataFrames/{experiment_name}/topdeg_gene_level_recovery.csv',
        index=False,
    )
    summarize_recovery_gene_level(gene_df, experiment_name, model_names)
    return gene_df


# -----------------------------------------------------------------------------
# Full-cell metrics from h5ad and CSV summaries
# -----------------------------------------------------------------------------


def compute_full_metrics_for_one_result(adata_query_pred, adata_query_stim, top50_genes, gammas):
    records = {}
    for suffix, genes in [('all', None), ('top50', top50_genes)]:
        x_pred = get_X(adata_query_pred, genes)
        x_stim = get_X(adata_query_stim, genes)

        pred_mean = np.mean(x_pred, axis=0)
        stim_mean = np.mean(x_stim, axis=0)
        pred_var = np.var(x_pred, axis=0)
        stim_var = np.var(x_stim, axis=0)

        records[f'r2mean_{suffix}'] = safe_r2(pred_mean, stim_mean)
        records[f'l2mean_{suffix}'] = l2_distance(pred_mean, stim_mean)
        records[f'r2var_{suffix}'] = safe_r2(pred_var, stim_var)
        records[f'l2var_{suffix}'] = l2_distance(pred_var, stim_var)

    x_mmd = get_X(adata_query_pred, top50_genes)
    y_mmd = get_X(adata_query_stim, top50_genes)
    records['mmd_top50'] = compute_mmd_loss(x_mmd, y_mmd, gammas=gammas)
    return records


def compute_metrics_from_h5ad(
    adata,
    experiment_name='across_cell_types',
    model_names=('scPILOT', 'scGen', 'CellOT', 'biolord', 'identity', 'VAEGAN'),
    seeds=(1327, 1337, 1347),
    data_file='pbmc',
    file_type='.h5ad',
    cond_key='condition',
    stim_key='stimulated',
    cell_label_key='cell_type',
    query_keys=None,
    skip_missing=False,
):
    ensure_dirs(f'../DataFrames/{experiment_name}')
    model_order = default_model_order(model_names)
    query_keys = query_keys or sorted(adata.obs[cell_label_key].unique().tolist())
    gammas = np.logspace(1, -3, num=50)
    metric_records = []

    for query_key in query_keys:
        top50_genes = get_top50_genes(
            adata=adata,
            query_key=query_key,
            cond_key=cond_key,
            stim_key=stim_key,
            cell_label_key=cell_label_key,
        )

        for seed in seeds:
            for model_name in model_order:
                print(
                    f'======Computing metrics | model={model_name} | '
                    f'query={query_key} | seed={seed}======',
                    flush=True,
                )
                adata_eval = load_prediction_h5ad(
                    experiment_name=experiment_name,
                    model_name=model_name,
                    data_file=data_file,
                    query_key=query_key,
                    seed=seed,
                    file_type=file_type,
                    skip_missing=skip_missing,
                )
                if adata_eval is None:
                    continue
                adata_query_pred = adata_eval[adata_eval.obs[cond_key] == 'pred'].copy()
                adata_query_stim = adata_eval[adata_eval.obs[cond_key] == stim_key].copy()

                metrics = compute_full_metrics_for_one_result(
                    adata_query_pred=adata_query_pred,
                    adata_query_stim=adata_query_stim,
                    top50_genes=top50_genes,
                    gammas=gammas,
                )

                metric_records.append({
                    'experiment': experiment_name,
                    'data_file': data_file,
                    'query_key': query_key,
                    'seed': seed,
                    'model': model_name,
                    'model_display': model_display_name(model_name),
                    **metrics,
                })

    metrics_seed_df = pd.DataFrame(metric_records)
    metrics_seed_df.to_csv(
        f'../DataFrames/{experiment_name}/metrics_seed_level_full.csv',
        index=False,
    )
    print(metrics_seed_df.groupby('model').size(), flush=True)
    return metrics_seed_df


def summarize_metrics_from_csv(
    experiment_name='across_cell_types',
    model_names=('scPILOT', 'scGen', 'CellOT', 'biolord', 'identity', 'VAEGAN'),
):
    metrics_path = f'../DataFrames/{experiment_name}/metrics_seed_level_full.csv'
    metrics_seed_df = pd.read_csv(metrics_path)

    metric_cols = [
        'r2mean_all',
        'l2mean_all',
        'r2var_all',
        'l2var_all',
        'r2mean_top50',
        'l2mean_top50',
        'r2var_top50',
        'l2var_top50',
        'mmd_top50',
    ]

    query_mean = (
        metrics_seed_df
        .groupby(['experiment', 'data_file', 'query_key', 'model', 'model_display'])[metric_cols]
        .mean()
        .reset_index()
    )
    query_sd = (
        metrics_seed_df
        .groupby(['experiment', 'data_file', 'query_key', 'model', 'model_display'])[metric_cols]
        .std()
        .reset_index()
    )
    query_summary = query_mean.merge(
        query_sd,
        on=['experiment', 'data_file', 'query_key', 'model', 'model_display'],
        suffixes=('_mean', '_sd'),
    )
    query_summary.to_csv(
        f'../DataFrames/{experiment_name}/metrics_query_level_summary.csv',
        index=False,
    )

    return metrics_seed_df, query_summary


def format_radar_tick(x):
    """Format radial tick labels."""
    if abs(x) >= 10:
        return f'{x:.1f}'
    if abs(x) >= 1:
        return f'{x:.2f}'
    return f'{x:.3f}'.rstrip('0').rstrip('.')


def metric_higher_is_better(metric):
    return metric.startswith('r2')


def metric_direction_symbol(metric):
    return '↑' if metric_higher_is_better(metric) else '↓'


def format_test_annotation(pvalue):
    """
    Format p-value annotations.

    Significant results are shown as stars.
    Non-significant results are shown as numeric p-values.
    """
    if pvalue is None or not np.isfinite(pvalue):
        return 'p=NA'

    if pvalue < 1e-4:
        return '****'

    return f'p={pvalue:.4f}'


def paired_test_scpilot_vs_baseline(query_summary, metric, baseline, test_name):
    """
    Compare scPILOT against one baseline using paired held-out split means.

    Each paired sample is one held-out cell type.
    The value for each held-out cell type is the average across three seeds.
    """
    mean_col = f'{metric}_mean'

    pivot = query_summary.pivot(
        index='query_key',
        columns='model',
        values=mean_col,
    )

    if 'scPILOT' not in pivot.columns or baseline not in pivot.columns:
        return np.nan, np.nan, 0

    paired = pivot[['scPILOT', baseline]].dropna()
    n_pairs = paired.shape[0]

    if n_pairs < 2:
        return np.nan, np.nan, n_pairs

    x = paired['scPILOT'].values.astype(float)
    y = paired[baseline].values.astype(float)

    try:
        if test_name == 'wilcoxon':
            res = stats.wilcoxon(
                x,
                y,
                alternative='two-sided',
            )
            statistic = float(res.statistic)
            pvalue = float(res.pvalue)

        elif test_name == 'paired_ttest':
            res = stats.ttest_rel(
                x,
                y,
                nan_policy='omit',
            )
            statistic = float(res.statistic)
            pvalue = float(res.pvalue)

        else:
            raise ValueError(f'Unknown test_name: {test_name}')

    except ValueError:
        statistic = np.nan
        pvalue = np.nan

    return statistic, pvalue, n_pairs


def summarize_worst_case_performance(
    query_summary,
    metrics,
    experiment_name,
    data_file,
    model_order,
):
    """
    Summarize worst-case performance across held-out cell types.

    Each split-level value is the mean across three seeds.
    For higher-is-better metrics, the worst case is the minimum value.
    For lower-is-better metrics, the worst case is the maximum value.
    """
    query_keys = sorted(query_summary['query_key'].unique().tolist())
    records = []

    for metric in metrics:
        mean_col = f'{metric}_mean'
        higher_is_better = metric_higher_is_better(metric)

        for model_name in model_order:
            dfg = (
                query_summary[query_summary['model'] == model_name]
                .set_index('query_key')
                .reindex(query_keys)
            )

            if mean_col not in dfg.columns:
                continue

            values = dfg[mean_col].astype(float).dropna()

            if values.empty:
                continue

            if higher_is_better:
                worst_query_key = values.idxmin()
                worst_case_value = float(values.min())
                best_query_key = values.idxmax()
                best_case_value = float(values.max())
            else:
                worst_query_key = values.idxmax()
                worst_case_value = float(values.max())
                best_query_key = values.idxmin()
                best_case_value = float(values.min())

            value_array = values.values.astype(float)

            records.append({
                'experiment': experiment_name,
                'data_file': data_file,
                'metric': metric,
                'model': model_name,
                'model_display': model_display_name(model_name),
                'higher_is_better': higher_is_better,
                'worst_case_value': worst_case_value,
                'worst_case_query_key': worst_query_key,
                'best_case_value': best_case_value,
                'best_case_query_key': best_query_key,
                'mean_across_splits': float(np.mean(value_array)),
                'median_across_splits': float(np.median(value_array)),
                'std_across_splits': float(np.std(value_array, ddof=1)) if len(value_array) > 1 else np.nan,
                'iqr_across_splits': float(np.percentile(value_array, 75) - np.percentile(value_array, 25)),
                'range_across_splits': float(np.max(value_array) - np.min(value_array)),
                'n_splits': int(len(value_array)),
            })

    worst_df = pd.DataFrame(records)
    worst_df.to_csv(
        f'../DataFrames/{experiment_name}/metrics_worst_case_summary.csv',
        index=False,
    )

    return worst_df


def plot_worst_case_performance_combined(
    worst_df,
    metrics,
    metric_labels,
    experiment_name,
    data_file,
    model_order,
    annotate_worst_split=True,
    annotate_scpilot_advantage=True,
):
    """
    Plot mean and worst-case performance for selected metrics.

    For each model:
    - Open marker: mean performance across held-out splits.
    - Filled marker: worst-case split-level performance.
    - Gray segment: gap between mean and worst-case performance.

    scPILOT is used as the visual reference:
    - Solid vertical line: scPILOT mean.
    - Dashed vertical line: scPILOT worst case.
    - Right-side labels indicate whether scPILOT has better mean and/or worst-case
      performance than each baseline.

    For R2 metrics, larger values are better.
    For MMD and L2 metrics, smaller values are better.
    """
    plot_order = model_order
    y_positions = np.arange(len(plot_order))

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    model_colors = {
        model_name: color_cycle[i % len(color_cycle)]
        for i, model_name in enumerate(plot_order)
    }

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(5.8 * len(metrics), 6.2),
        sharey=True,
    )

    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        higher_is_better = metric_higher_is_better(metric)

        dfg_metric = (
            worst_df[worst_df['metric'] == metric]
            .set_index('model')
            .reindex(plot_order)
        )

        # Use both mean and worst-case values to determine the axis range.
        axis_values = []
        for col in ['worst_case_value', 'mean_across_splits']:
            if col in dfg_metric.columns:
                vals = dfg_metric[col].astype(float).values
                axis_values.extend(vals[np.isfinite(vals)])

        if len(axis_values) == 0:
            continue

        xmin, xmax, xticks = get_radar_axis_limits(axis_values, metric)
        span = xmax - xmin if xmax > xmin else 1.0

        # scPILOT reference values.
        scpilot_mean = np.nan
        scpilot_worst = np.nan

        if 'scPILOT' in dfg_metric.index:
            scpilot_mean = dfg_metric.loc['scPILOT', 'mean_across_splits']
            scpilot_worst = dfg_metric.loc['scPILOT', 'worst_case_value']

        # Reference lines for scPILOT.
        if np.isfinite(scpilot_mean):
            ax.axvline(
                scpilot_mean,
                color='black',
                linestyle='-',
                linewidth=1.2,
                alpha=0.70,
                zorder=1,
            )

        if np.isfinite(scpilot_worst):
            ax.axvline(
                scpilot_worst,
                color='black',
                linestyle='--',
                linewidth=1.2,
                alpha=0.70,
                zorder=1,
            )

        for y_pos, model_name in zip(y_positions, plot_order):
            if model_name not in dfg_metric.index:
                continue

            row = dfg_metric.loc[model_name]
            worst_value = row['worst_case_value']
            mean_value = row['mean_across_splits']

            if not np.isfinite(worst_value) or not np.isfinite(mean_value):
                continue

            color = model_colors[model_name]
            marker = model_marker(model_name)

            # Make scPILOT more visually prominent.
            is_scpilot = model_name == 'scPILOT'
            point_size = 92 if is_scpilot else 58
            line_width = 2.0 if is_scpilot else 1.2
            alpha = 1.0 if is_scpilot else 0.85

            # Connect mean and worst-case values to show robustness gap.
            ax.plot(
                [mean_value, worst_value],
                [y_pos, y_pos],
                color='black' if is_scpilot else 'gray',
                linewidth=line_width,
                alpha=0.65 if is_scpilot else 0.45,
                zorder=2,
            )

            # Mean performance across held-out splits: open marker.
            ax.scatter(
                mean_value,
                y_pos,
                s=point_size,
                marker=marker,
                facecolors='white',
                edgecolors='black' if is_scpilot else color,
                linewidths=1.6 if is_scpilot else 1.2,
                alpha=alpha,
                zorder=5 if is_scpilot else 4,
            )

            # Worst-case performance: filled marker.
            ax.scatter(
                worst_value,
                y_pos,
                s=point_size,
                marker=marker,
                color=color,
                edgecolors='black',
                linewidths=1.0 if is_scpilot else 0.7,
                alpha=alpha,
                zorder=6 if is_scpilot else 5,
            )

            if annotate_worst_split:
                worst_query_key = str(row['worst_case_query_key'])

                dx = 0.05 * span
                dy = 0.18

                # Default rule from the previous version:
                # put the query key centered below the worst-case point;
                # identity is placed above because it is at the bottom.
                text_x = worst_value
                text_y = y_pos - dy
                ha = 'center'
                va = 'top'

                if model_name == 'identity':
                    text_y = y_pos + dy
                    va = 'bottom'

                # For R2mean panels:
                # CellOT and identity keep the default placement;
                # all other models are labeled on the left side of the point.
                if metric in ['r2mean_all', 'r2mean_top50']:
                    if model_name not in ['CellOT', 'identity']:
                        text_x = worst_value - dx
                        text_y = y_pos
                        ha = 'right'
                        va = 'center'

                # For MMD panel:
                # scPILOT is labeled on the right side of the point;
                # all other models keep the default placement.
                elif metric == 'mmd_top50':
                    if model_name == 'scPILOT':
                        text_x = worst_value + dx
                        text_y = y_pos
                        ha = 'left'
                        va = 'center'

                ax.text(
                    text_x,
                    text_y,
                    worst_query_key,
                    ha=ha,
                    va=va,
                    fontsize=8.0,
                    alpha=0.90 if is_scpilot else 0.70,
                    fontweight='bold' if is_scpilot else 'normal',
                    zorder=7,
                    clip_on=False,
                )

            # Right-side label: whether scPILOT is better in mean and worst case.
            if annotate_scpilot_advantage and model_name != 'scPILOT':
                if np.isfinite(scpilot_mean) and np.isfinite(scpilot_worst):
                    if higher_is_better:
                        mean_advantage = scpilot_mean - mean_value
                        worst_advantage = scpilot_worst - worst_value
                    else:
                        mean_advantage = mean_value - scpilot_mean
                        worst_advantage = worst_value - scpilot_worst

                    mean_better = mean_advantage > 0
                    worst_better = worst_advantage > 0

                    if mean_better and worst_better:
                        label = 'Mean√ Worst√'
                        label_weight = 'bold'
                        label_alpha = 1.0
                    elif mean_better:
                        label = 'Mean√'
                        label_weight = 'normal'
                        label_alpha = 0.80
                    elif worst_better:
                        label = 'Worst√'
                        label_weight = 'normal'
                        label_alpha = 0.80
                    else:
                        label = ''
                        label_weight = 'normal'
                        label_alpha = 0.50

                    if label:
                        ax.text(
                            1.02,
                            y_pos,
                            label,
                            transform=ax.get_yaxis_transform(),
                            ha='left',
                            va='center',
                            fontsize=9.5,
                            fontweight=label_weight,
                            alpha=label_alpha,
                        )

        ax.set_xlim(xmin, xmax)
        ax.set_xticks(xticks)
        ax.set_xticklabels([format_radar_tick(t) for t in xticks], fontsize=10)

        ax.grid(True, axis='x', linewidth=0.8, alpha=0.35)
        sns.despine(ax=ax, top=True, right=True)

        ax.set_title(
            metric_labels[metric],
            fontsize=13,
            pad=10,
        )

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels([model_display_name(m) for m in plot_order], fontsize=12)
    axes[0].set_ylabel('Model', fontsize=13)

    # Dummy handles for legend.
    mean_handle = axes[-1].scatter(
        [],
        [],
        s=58,
        marker='o',
        facecolors='white',
        edgecolors='black',
        linewidths=1.3,
        label='Mean across splits',
    )
    worst_handle = axes[-1].scatter(
        [],
        [],
        s=58,
        marker='o',
        color='black',
        edgecolors='black',
        linewidths=0.7,
        label='Worst-case split',
    )
    gap_handle, = axes[-1].plot(
        [],
        [],
        color='gray',
        linewidth=1.2,
        alpha=0.55,
        label='Mean–worst gap',
    )
    mean_ref_handle, = axes[-1].plot(
        [],
        [],
        color='black',
        linestyle='-',
        linewidth=1.2,
        alpha=0.70,
        label='scPILOT mean',
    )
    worst_ref_handle, = axes[-1].plot(
        [],
        [],
        color='black',
        linestyle='--',
        linewidth=1.2,
        alpha=0.70,
        label='scPILOT worst case',
    )

    axes[-1].legend(
        handles=[
            mean_handle,
            worst_handle,
            gap_handle,
            mean_ref_handle,
            worst_ref_handle,
        ],
        bbox_to_anchor=(1.5, 1.0),
        loc='upper left',
        frameon=False,
        fontsize=9,
    )

    fig.suptitle(
        'Mean and worst-case performance across held-out cell types',
        fontsize=16,
        x=0.4,
        y=1.03,
    )

    plt.tight_layout(rect=[0, 0, 0.86, 0.95])
    save_figure_jpg_pdf(
        f'../Figures/{experiment_name}/'
        f'All_models_on_{data_file}_worst_case_performance_combined.jpg'
    )
    plt.close('all')


def get_radar_axis_limits(values, metric, n_ticks=5):
    """
    Choose radial axis limits according to the displayed data range.

    The radial axis is intentionally not forced to start from 0.
    This helps distinguish close values such as 0.95 and 0.92.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return 0.0, 1.0, np.linspace(0.0, 1.0, n_ticks)

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    span = vmax - vmin

    if span == 0:
        span = max(abs(vmax) * 0.05, 0.01)

    if metric.startswith('r2'):
        # R2 metrics are bounded by [0, 1].
        pad = max(span * 0.20, 0.01)

        rmin = max(0.0, vmin - pad)
        rmax = min(1.0, vmax + pad)

        # Use clean two-decimal limits for R2.
        rmin = np.floor(rmin * 100) / 100
        rmax = np.ceil(rmax * 100) / 100

        # Avoid an overly narrow radial range.
        if rmax - rmin < 0.03:
            mid = (rmin + rmax) / 2
            rmin = max(0.0, mid - 0.02)
            rmax = min(1.0, mid + 0.02)

    else:
        # MMD and L2 metrics are non-negative, but we still avoid forcing 0
        # unless the padded lower bound goes below 0.
        pad = max(span * 0.20, abs(vmax) * 0.03, 1e-4)

        rmin = vmin - pad
        rmax = vmax + pad

        if rmin < 0:
            rmin = 0.0

        if rmax - rmin <= 0:
            rmax = rmin + max(abs(rmin) * 0.05, 1e-3)

    rticks = np.linspace(rmin, rmax, n_ticks)
    return rmin, rmax, rticks


def plot_splitwise_radar_chart(
    seed_df,
    query_summary,
    metric,
    experiment_name,
    data_file,
    model_order,
    ylabel,
    show_seed_points=False,
    show_error_bars=False,
    show_sd_band=False,
):
    mean_col = f'{metric}_mean'
    sd_col = f'{metric}_sd'

    query_keys = sorted(query_summary['query_key'].unique().tolist())
    n_axes = len(query_keys)

    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False)
    angles_closed = np.concatenate([angles, [angles[0]]])

    # Collect all displayed mean values to determine the radial axis range.
    all_values = []
    for model_name in model_order:
        dfg = (
            query_summary[query_summary['model'] == model_name]
            .set_index('query_key')
            .reindex(query_keys)
        )
        all_values.extend(dfg[mean_col].values.astype(float))

    rmin, rmax, rticks = get_radar_axis_limits(all_values, metric)

    fig, ax = plt.subplots(
        figsize=(8.8, 8.8),
        subplot_kw={'polar': True},
    )

    # Place the first axis at the top and draw axes clockwise.
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles)
    ax.set_xticklabels(query_keys, fontsize=12)

    ax.set_ylim(rmin, rmax)
    ax.set_yticks(rticks)
    ax.set_yticklabels([format_radar_tick(t) for t in rticks], fontsize=10)
    ax.set_rlabel_position(90)

    ax.grid(True, linewidth=0.8, alpha=0.45)
    ax.spines['polar'].set_alpha(0.45)

    # Small angular offsets help avoid overplotting among models.
    model_offsets = np.linspace(-0.045, 0.045, len(model_order))

    for i, model_name in enumerate(model_order):
        display_name = model_display_name(model_name)
        marker = model_marker(model_name)

        # Query-level mean and SD for the current model.
        dfg_mean = (
            query_summary[query_summary['model'] == model_name]
            .set_index('query_key')
            .reindex(query_keys)
        )

        mean_values = dfg_mean[mean_col].values.astype(float)
        mean_values_closed = np.concatenate([mean_values, [mean_values[0]]])

        # Draw the mean radar line.
        line, = ax.plot(
            angles_closed,
            mean_values_closed,
            linewidth=2.0,
            marker=marker,
            markersize=model_marker_size(model_name, base_size=4.2),
            label=display_name,
        )
        line_color = line.get_color()

        ax.fill(
            angles_closed,
            mean_values_closed,
            color=line_color,
            alpha=0.04,
        )

        if sd_col in dfg_mean.columns:
            sd_values = dfg_mean[sd_col].values.astype(float)

            # Clip mean ± SD to the displayed radial axis range.
            lower = np.clip(mean_values - sd_values, rmin, rmax)
            upper = np.clip(mean_values + sd_values, rmin, rmax)

            lower_closed = np.concatenate([lower, [lower[0]]])
            upper_closed = np.concatenate([upper, [upper[0]]])

            # Optional SD band around the mean line.
            if show_sd_band:
                ax.fill_between(
                    angles_closed,
                    lower_closed,
                    upper_closed,
                    color=line_color,
                    alpha=0.06,
                    linewidth=0,
                )

            # Optional error bars showing mean ± SD at each held-out cell type.
            if show_error_bars:
                theta_error = angles + model_offsets[i]

                yerr_lower = mean_values - lower
                yerr_upper = upper - mean_values
                yerr = np.vstack([yerr_lower, yerr_upper])

                ax.errorbar(
                    theta_error,
                    mean_values,
                    yerr=yerr,
                    fmt='none',
                    color=line_color,
                    capsize=3,
                    elinewidth=1.0,
                    capthick=1.0,
                    alpha=0.85,
                    zorder=3,
                )

        # Optional seed-level points.
        if show_seed_points:
            dfg_seed = seed_df[seed_df['model'] == model_name].copy()

            for j, query_key in enumerate(query_keys):
                sub = (
                    dfg_seed[dfg_seed['query_key'] == query_key]
                    .sort_values('seed')
                )

                seed_values = sub[metric].values.astype(float)
                n_seed = len(seed_values)

                if n_seed == 0:
                    continue

                # Small local offsets separate the seed points within each model.
                if n_seed == 1:
                    local_offsets = np.array([0.0])
                else:
                    local_offsets = np.linspace(-0.018, 0.018, n_seed)

                theta_seed = (
                    np.full(n_seed, angles[j] + model_offsets[i])
                    + local_offsets
                )

                ax.scatter(
                    theta_seed,
                    seed_values,
                    s=18,
                    color=line_color,
                    alpha=0.85,
                    zorder=4,
                )

    ax.set_title(
        f'{ylabel}\nHeld-out cell type',
        fontsize=15,
        pad=24,
    )

    ax.legend(
        bbox_to_anchor=(1.22, 0.5),
        loc='center left',
        frameon=False,
        fontsize=11,
    )

    direction = metric_direction_symbol(metric)

    plt.tight_layout()
    save_figure_jpg_pdf(
        f'../Figures/{experiment_name}/'
        f'All_models_on_{data_file}_splitwise_radar_{metric}.jpg'
    )
    plt.close('all')


def plot_splitwise_line_chart(
    seed_df,
    query_summary,
    metric,
    experiment_name,
    data_file,
    model_order,
    ylabel,
    show_seed_points=True,
    show_error_bars=True,
    show_sd_band=True,
):
    mean_col = f'{metric}_mean'
    sd_col = f'{metric}_sd'

    query_keys = sorted(query_summary['query_key'].unique().tolist())
    x = np.arange(len(query_keys))

    # Use the displayed mean values to determine the y-axis range.
    all_values = []
    for model_name in model_order:
        dfg = (
            query_summary[query_summary['model'] == model_name]
            .set_index('query_key')
            .reindex(query_keys)
        )
        all_values.extend(dfg[mean_col].values.astype(float))

    ymin, ymax, yticks = get_radar_axis_limits(all_values, metric)

    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    # Small horizontal offsets help avoid overplotting among models.
    model_offsets = np.linspace(-0.20, 0.20, len(model_order))

    for i, model_name in enumerate(model_order):
        display_name = model_display_name(model_name)
        marker = model_marker(model_name)

        # Query-level mean and SD for the current model.
        dfg_mean = (
            query_summary[query_summary['model'] == model_name]
            .set_index('query_key')
            .reindex(query_keys)
        )

        mean_values = dfg_mean[mean_col].values.astype(float)

        # Draw the mean line.
        x_model = x

        line, = ax.plot(
            x_model,
            mean_values,
            linewidth=1.2,
            marker=marker,
            markersize=model_marker_size(model_name, base_size=3.8),
            markeredgewidth=0.0,
            label=display_name,
            zorder=5,
        )

        line_color = line.get_color()
        line.set_markerfacecolor(line_color)
        line.set_markeredgecolor(line_color)

        if sd_col in dfg_mean.columns:
            sd_values = dfg_mean[sd_col].values.astype(float)

            lower = np.clip(mean_values - sd_values, ymin, ymax)
            upper = np.clip(mean_values + sd_values, ymin, ymax)

            if show_sd_band:
                ax.fill_between(
                    x_model,
                    lower,
                    upper,
                    color=line_color,
                    alpha=0.08,
                    linewidth=0,
                    zorder=1,
                )

            if show_error_bars:
                yerr_lower = mean_values - lower
                yerr_upper = upper - mean_values
                yerr = np.vstack([yerr_lower, yerr_upper])

                ax.errorbar(
                    x_model,
                    mean_values,
                    yerr=yerr,
                    fmt='none',
                    color=line_color,
                    capsize=3,
                    elinewidth=1.0,
                    capthick=1.0,
                    alpha=0.90,
                    zorder=4,
                )

        if show_seed_points:
            dfg_seed = seed_df[seed_df['model'] == model_name].copy()

            for j, query_key in enumerate(query_keys):
                sub = (
                    dfg_seed[dfg_seed['query_key'] == query_key]
                    .sort_values('seed')
                )

                seed_values = sub[metric].values.astype(float)
                n_seed = len(seed_values)

                if n_seed == 0:
                    continue

                if n_seed == 1:
                    local_offsets = np.array([0.0])
                else:
                    local_offsets = np.linspace(-0.025, 0.025, n_seed)

                x_seed = (
                    np.full(n_seed, x_model[j])
                    + local_offsets
                )

                ax.scatter(
                    x_seed,
                    seed_values,
                    s=10,
                    marker=marker,
                    facecolors='none',
                    edgecolors=line_color,
                    linewidths=0.8,
                    alpha=0.85,
                    zorder=6,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(query_keys, rotation=25, ha='right', fontsize=11)

    ax.set_ylim(ymin, ymax)
    ax.set_yticks(yticks)
    ax.set_yticklabels([format_radar_tick(t) for t in yticks], fontsize=10)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel('Held-out cell type', fontsize=12)

    ax.grid(True, axis='y', linewidth=0.8, alpha=0.45)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    direction = metric_direction_symbol(metric)

    ax.legend(
        bbox_to_anchor=(1.02, 1.0),
        loc='upper left',
        frameon=False,
        fontsize=11,
    )

    plt.tight_layout()
    save_figure_jpg_pdf(
        f'../Figures/{experiment_name}/'
        f'All_models_on_{data_file}_splitwise_line_{metric}.jpg'
    )
    plt.close('all')


def plot_aggregate_boxplot_combined(
    query_summary,
    metrics,
    metric_labels,
    experiment_name,
    data_file,
    model_order,
    test_name='wilcoxon',
):
    """
    CRISP-style aggregate boxplot with three metrics in one row.

    Each box summarizes split-level mean values across held-out cell types.
    Each point is one held-out split averaged across three seeds.
    scPILOT is compared with each baseline using paired split-level tests.
    """
    query_keys = sorted(query_summary['query_key'].unique().tolist())

    # Keep the same bottom-to-top model order as the splitwise line chart.
    # Because scPILOT is the last model in model_order, it appears at the top.
    plot_order = model_order
    y_positions = np.arange(len(plot_order))

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    model_colors = {
        model_name: color_cycle[i % len(color_cycle)]
        for i, model_name in enumerate(plot_order)
    }

    test_title_map = {
        'wilcoxon': 'Wilcoxon signed-rank test',
        'paired_ttest': 'Paired t-test',
    }
    test_title = test_title_map.get(test_name, test_name)

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(5.2 * len(metrics), 5.8),
        sharey=True,
    )

    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        mean_col = f'{metric}_mean'

        # Determine x-axis range from split-level mean values.
        all_values = []
        for model_name in plot_order:
            dfg = (
                query_summary[query_summary['model'] == model_name]
                .set_index('query_key')
                .reindex(query_keys)
            )
            values = dfg[mean_col].values.astype(float)
            values = values[np.isfinite(values)]
            all_values.extend(values)

        xmin, xmax, xticks = get_radar_axis_limits(all_values, metric)

        for y_pos, model_name in zip(y_positions, plot_order):
            color = model_colors[model_name]

            dfg = (
                query_summary[query_summary['model'] == model_name]
                .set_index('query_key')
                .reindex(query_keys)
            )

            values = dfg[mean_col].values.astype(float)
            values_valid = values[np.isfinite(values)]

            if values_valid.size == 0:
                continue

            # Horizontal CRISP-style boxplot.
            ax.boxplot(
                values_valid,
                positions=[y_pos],
                vert=False,
                widths=0.55,
                patch_artist=True,
                showfliers=False,
                whis=(0, 100),
                boxprops=dict(
                    facecolor=color,
                    edgecolor='black',
                    linewidth=0.9,
                    alpha=0.75,
                ),
                medianprops=dict(
                    color='black',
                    linewidth=1.2,
                ),
                whiskerprops=dict(
                    color='black',
                    linewidth=0.8,
                ),
                capprops=dict(
                    color='black',
                    linewidth=0.8,
                ),
            )

            # Plot one point per held-out split.
            # Each point is the average across three seeds.
            point_offsets = np.linspace(-0.16, 0.16, len(query_keys))

            for q_idx, query_key in enumerate(query_keys):
                if query_key not in dfg.index:
                    continue

                value = dfg.loc[query_key, mean_col]
                if not np.isfinite(value):
                    continue

                ax.scatter(
                    value,
                    y_pos + point_offsets[q_idx],
                    s=18,
                    marker='o',
                    facecolors='white',
                    edgecolors='black',
                    linewidths=0.7,
                    alpha=0.95,
                    zorder=4,
                )

        ax.set_xlim(xmin, xmax)
        ax.set_xticks(xticks)
        ax.set_xticklabels([format_radar_tick(t) for t in xticks], fontsize=9)

        ax.grid(True, axis='x', linewidth=0.8, alpha=0.35)
        sns.despine(ax=ax, top=True, right=True)

        # Annotate scPILOT-vs-baseline p-values.
        if 'scPILOT' in plot_order:
            for y_pos, baseline in zip(y_positions, plot_order):
                if baseline == 'scPILOT':
                    continue

                statistic, pvalue, n_pairs = paired_test_scpilot_vs_baseline(
                    query_summary=query_summary,
                    metric=metric,
                    baseline=baseline,
                    test_name=test_name,
                )

                ax.text(
                    1.02,
                    y_pos,
                    format_test_annotation(pvalue),
                    transform=ax.get_yaxis_transform(),
                    ha='left',
                    va='center',
                    fontsize=8.5,
                )

        direction = metric_direction_symbol(metric)
        ax.set_title(
            metric_labels[metric],
            fontsize=12,
            pad=10,
        )

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels([model_display_name(m) for m in plot_order], fontsize=11)
    axes[0].set_ylabel('Model', fontsize=12)

    fig.suptitle(
        f'Aggregate performance across held-out cell types ({test_title})',
        fontsize=15,
        y=1.03,
    )

    plt.tight_layout()
    save_figure_jpg_pdf(
        f'../Figures/{experiment_name}/'
        f'All_models_on_{data_file}_aggregate_boxplot_combined_{test_name}.jpg'
    )
    plt.close('all')


def plot_metrics_from_csv(
    experiment_name='across_cell_types',
    model_names=('scPILOT', 'scGen', 'CellOT', 'biolord', 'identity', 'VAEGAN'),
    data_file='pbmc',
):
    ensure_dirs(f'../Figures/{experiment_name}')
    seed_df = pd.read_csv(
        f'../DataFrames/{experiment_name}/metrics_seed_level_full.csv'
    )
    query_summary = pd.read_csv(
        f'../DataFrames/{experiment_name}/metrics_query_level_summary.csv'
    )

    model_order = default_model_order(model_names)
    metric_labels = {
        'r2mean_all': r'$R^2_{\mathrm{mean}}$ (↑; all genes)',
        'r2mean_top50': r'$R^2_{\mathrm{mean}}$ (↑; top 50 DEGs)',
        'mmd_top50': r'$\mathrm{MMD}$ (↓; top 50 DEGs)',

        'l2mean_all': r'$L^2_{\mathrm{mean}}$ (↓; all genes)',
        'l2mean_top50': r'$L^2_{\mathrm{mean}}$ (↓; top 50 DEGs)',

        'r2var_all': r'$R^2_{\mathrm{var}}$ (↑; all genes)',
        'r2var_top50': r'$R^2_{\mathrm{var}}$ (↑; top 50 DEGs)',

        'l2var_all': r'$L^2_{\mathrm{var}}$ (↓; all genes)',
        'l2var_top50': r'$L^2_{\mathrm{var}}$ (↓; top 50 DEGs)',
    }

    main_metrics = ['r2mean_all', 'r2mean_top50', 'mmd_top50']
    supplementary_metrics = [
        'l2mean_all',
        'l2mean_top50',
        'r2var_all',
        'r2var_top50',
        'l2var_all',
        'l2var_top50',
    ]

    for metric in main_metrics + supplementary_metrics:
        # Radar chart: mean only, cleaner visual summary.
        plot_splitwise_radar_chart(
            seed_df=seed_df,
            query_summary=query_summary,
            metric=metric,
            experiment_name=experiment_name,
            data_file=data_file,
            model_order=model_order,
            ylabel=metric_labels[metric],
            show_seed_points=False,
            show_error_bars=False,
            show_sd_band=False,
        )

        # Line chart: full variability display with seed points, error bars, and SD band.
        plot_splitwise_line_chart(
            seed_df=seed_df,
            query_summary=query_summary,
            metric=metric,
            experiment_name=experiment_name,
            data_file=data_file,
            model_order=model_order,
            ylabel=metric_labels[metric],
            show_seed_points=True,
            show_error_bars=True,
            show_sd_band=True,
        )
    aggregate_metrics = ['r2mean_all', 'r2mean_top50', 'mmd_top50']

    plot_aggregate_boxplot_combined(
        query_summary=query_summary,
        metrics=aggregate_metrics,
        metric_labels=metric_labels,
        experiment_name=experiment_name,
        data_file=data_file,
        model_order=model_order,
        test_name='wilcoxon',
    )

    plot_aggregate_boxplot_combined(
        query_summary=query_summary,
        metrics=aggregate_metrics,
        metric_labels=metric_labels,
        experiment_name=experiment_name,
        data_file=data_file,
        model_order=model_order,
        test_name='paired_ttest',
    )

    worst_df = summarize_worst_case_performance(
        query_summary=query_summary,
        metrics=aggregate_metrics,
        experiment_name=experiment_name,
        data_file=data_file,
        model_order=model_order,
    )

    plot_worst_case_performance_combined(
        worst_df=worst_df,
        metrics=aggregate_metrics,
        metric_labels=metric_labels,
        experiment_name=experiment_name,
        data_file=data_file,
        model_order=model_order,
        annotate_worst_split=True,
    )


# -----------------------------------------------------------------------------
# Top-level entry point
# -----------------------------------------------------------------------------


def plot_result(
    experiment_name='across_cell_types',
    model_names=('scPILOT', 'scGen', 'CellOT', 'biolord', 'identity', 'VAEGAN'),
    seeds=(1327, 1337, 1347),
    data_file='pbmc',
    file_type='.h5ad',
    cond_key='condition',
    ctrl_key='control',
    stim_key='stimulated',
    cell_label_key='cell_type',
    query_key='all',
    steps='all',
    skip_missing=False,
):
    sns.set_theme(style='white', font='Arial', font_scale=1.4)
    ensure_dirs(
        f'../Figures/{experiment_name}',
        f'../DataFrames/{experiment_name}',
    )

    adata = sc.read_h5ad(f'../Data/{experiment_name}/{data_file}{file_type}')
    adata = adata[(adata.obs[cond_key] == ctrl_key) | (adata.obs[cond_key] == stim_key)].copy()
    print('adata:')
    print(adata)

    if query_key == 'all':
        query_keys = sorted(adata.obs[cell_label_key].unique().tolist())
    else:
        query_keys = [query_key]

    if steps == 'all':
        step_set = {'umap', 'violin', 'metrics', 'summary', 'barplots'}
    else:
        step_set = {s.strip() for s in steps.split(',') if s.strip()}

    if 'umap' in step_set:
        plot_shared_umaps_from_h5ad(
            adata=adata,
            experiment_name=experiment_name,
            model_names=model_names,
            seeds=seeds,
            data_file=data_file,
            file_type=file_type,
            cond_key=cond_key,
            ctrl_key=ctrl_key,
            stim_key=stim_key,
            cell_label_key=cell_label_key,
            query_keys=query_keys,
            skip_missing=skip_missing,
        )

    if 'violin' in step_set:
        plot_stacked_violins_and_recovery_from_h5ad(
            adata=adata,
            experiment_name=experiment_name,
            model_names=model_names,
            seeds=seeds,
            data_file=data_file,
            file_type=file_type,
            cond_key=cond_key,
            ctrl_key=ctrl_key,
            stim_key=stim_key,
            cell_label_key=cell_label_key,
            query_keys=query_keys,
            skip_missing=skip_missing,
        )

    if 'metrics' in step_set:
        compute_metrics_from_h5ad(
            adata=adata,
            experiment_name=experiment_name,
            model_names=model_names,
            seeds=seeds,
            data_file=data_file,
            file_type=file_type,
            cond_key=cond_key,
            stim_key=stim_key,
            cell_label_key=cell_label_key,
            query_keys=query_keys,
            skip_missing=skip_missing,
        )

    if 'summary' in step_set:
        summarize_metrics_from_csv(
            experiment_name=experiment_name,
            model_names=model_names,
        )

    if 'barplots' in step_set:
        plot_metrics_from_csv(
            experiment_name=experiment_name,
            model_names=model_names,
            data_file=data_file,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Across-cell-type result plotting and summary.')
    parser.add_argument('--query_key', type=str, default='all')
    parser.add_argument('--seed', type=str, default='all')
    parser.add_argument(
        '--steps',
        type=str,
        default='all',
        help='Comma-separated steps: umap,violin,metrics,summary,barplots. Use all to run all steps.',
    )
    parser.add_argument('--skip_missing', action='store_true')
    args = parser.parse_args()

    if args.seed == 'all':
        selected_seeds = (1327, 1337, 1347)
    else:
        selected_seeds = tuple(int(s.strip()) for s in args.seed.split(',') if s.strip())

    plot_result(
        query_key=args.query_key,
        seeds=selected_seeds,
        steps=args.steps,
        skip_missing=args.skip_missing,
    )
    print('Done', flush=True)
