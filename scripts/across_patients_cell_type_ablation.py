import os
import time
import argparse

import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import pairwise
from matplotlib import pyplot as plt


# -----------------------------------------------------------------------------
# Ablation configuration
# -----------------------------------------------------------------------------

DEFAULT_ABLATION_MODELS = (
    'scPILOT_w_o_discriminator',
    'scPILOT_w_o_latent_OT',
    'scPILOT_w_o_Leiden',
    'scPILOT_w_o_adaptive_weighting',
    'scPILOT',
)

ABLATION_MODEL_ORDER = (
    'scPILOT_w_o_discriminator',
    'scPILOT_w_o_latent_OT',
    'scPILOT_w_o_Leiden',
    'scPILOT_w_o_adaptive_weighting',
    'scPILOT',
)

MODEL_CONFIG = {
    'scPILOT_w_o_discriminator': {
        'display_name': 'w/o discriminator',
    },
    'scPILOT_w_o_latent_OT': {
        'display_name': 'w/o latent OT',
    },
    'scPILOT_w_o_Leiden': {
        'display_name': 'w/o Leiden',
    },
    'scPILOT_w_o_adaptive_weighting': {
        'display_name': 'w/o adaptive weighting',
    },
    'scPILOT': {
        'display_name': 'scPILOT',
    },
}


parser = argparse.ArgumentParser(description='across_patients_cell_type_ablation')
parser.add_argument(
    '--query_key',
    type=str,
    default='all',
    help='Patient/sample ID to analyze, e.g. 101, 107, 1015, or all.',
)
parser.add_argument(
    '--seed',
    type=str,
    default='all',
    help='Seed to analyze, e.g. 1327, 1337, 1347, or all.',
)
parser.add_argument(
    '--skip_missing',
    action='store_true',
    help='Skip missing prediction h5ad files.',
)
parser.add_argument(
    '--models',
    type=str,
    default=','.join(DEFAULT_ABLATION_MODELS),
    help='Comma-separated ablation model identifiers.',
)
parser.add_argument(
    '--output_name',
    type=str,
    default='across_patients_ablation',
    help='Shared output subdirectory under ../Figures and ../DataFrames.',
)
args = parser.parse_args()


def ensure_dirs(*paths: str):
    for path in paths:
        os.makedirs(path, exist_ok=True)


def save_figure_jpg_pdf(path_to_save: str, dpi: int = 300, bbox_inches: str = 'tight'):
    root, _ = os.path.splitext(path_to_save)
    plt.savefig(root + '.jpg', dpi=dpi, bbox_inches=bbox_inches)
    plt.savefig(root + '.pdf', bbox_inches=bbox_inches)


def to_dense_array(x):
    return x.toarray() if hasattr(x, 'toarray') else np.asarray(x)


def get_X(adata_obj, genes=None):
    x = adata_obj[:, genes].X if genes is not None else adata_obj.X
    return to_dense_array(x)


def obs_value_mask(adata_obj, obs_key, value):
    return adata_obj.obs[obs_key].astype(str) == str(value)


def model_display_name(model_name: str) -> str:
    return MODEL_CONFIG.get(
        model_name,
        {},
    ).get(
        'display_name',
        model_name,
    )


def default_model_order(model_names):
    """Return ablation models in a fixed, publication-ready order."""
    model_names = tuple(model_names)

    configured = [
        model_name
        for model_name in ABLATION_MODEL_ORDER
        if model_name in model_names
    ]
    additional = [
        model_name
        for model_name in model_names
        if model_name not in ABLATION_MODEL_ORDER
    ]
    return configured + additional


def get_figure_dir(output_name: str) -> str:
    return f'../Figures/{output_name}'


def get_dataframe_dir(output_name: str) -> str:
    return f'../DataFrames/{output_name}'


def sanitize_result_tag(value: str) -> str:
    """Create a filename-safe tag for one held-out patient or an all-patient run."""
    value = str(value).strip()
    safe = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in value)
    return safe or 'all'


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
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
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


def get_top50_genes_for_cell_type(
    adata_eval_sub,
    cond_key,
    ctrl_key,
    stim_key,
):
    """
    Define top50 DEGs using true ctrl and true stim cells within one held-out patient
    and one cell type.
    """
    adata_truth = adata_eval_sub[
        adata_eval_sub.obs[cond_key].isin([ctrl_key, stim_key])
    ].copy()

    if adata_truth.n_obs == 0:
        return None

    n_ctrl = int(np.sum(adata_truth.obs[cond_key] == ctrl_key))
    n_stim = int(np.sum(adata_truth.obs[cond_key] == stim_key))

    if n_ctrl < 2 or n_stim < 2:
        print(
            f'[SKIP DEG] Too few cells: ctrl={n_ctrl}, stim={n_stim}',
            flush=True,
        )
        return None

    sc.tl.rank_genes_groups(
        adata_truth,
        groupby=cond_key,
        method='wilcoxon',
        n_genes=50,
    )

    return adata_truth.uns['rank_genes_groups']['names'][stim_key].tolist()


def compute_metrics_for_one_cell_type(
    adata_query_pred,
    adata_query_stim,
    top50_genes,
    gammas,
):
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


def compute_cell_type_metrics_from_h5ad(
    experiment_name='across_patients',
    output_name='across_patients_ablation',
    model_names=DEFAULT_ABLATION_MODELS,
    seeds=(1327, 1337, 1347),
    data_file='pbmc_patients',
    file_type='.h5ad',
    cond_key='condition',
    ctrl_key='ctrl',
    stim_key='stim',
    cell_label_key='sample_id',
    query_keys=None,
    sub_key='cell_type',
    skip_missing=False,
    result_tag='all',
):
    ensure_dirs(get_dataframe_dir(output_name), get_figure_dir(output_name))

    model_order = default_model_order(model_names)
    gammas = np.logspace(1, -3, num=50)
    metric_records = []

    # Use the original data only to infer query_keys if needed.
    adata = sc.read_h5ad(f'../Data/{experiment_name}/{data_file}{file_type}')
    adata = adata[
        (adata.obs[cond_key] == ctrl_key)
        | (adata.obs[cond_key] == stim_key)
    ].copy()

    if query_keys is None:
        query_keys = sorted(adata.obs[cell_label_key].astype(str).unique().tolist())

    for query_key in query_keys:
        for seed in seeds:
            for model_name in model_order:
                print(
                    f'======Cell-type metrics | patient={query_key} | '
                    f'seed={seed} | model={model_name}======',
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

                sub_classes = sorted(adata_eval.obs[sub_key].astype(str).unique().tolist())

                for sub_class in sub_classes:
                    adata_eval_sub = adata_eval[
                        adata_eval.obs[sub_key].astype(str) == str(sub_class)
                    ].copy()

                    adata_query_ctrl = adata_eval_sub[
                        adata_eval_sub.obs[cond_key] == ctrl_key
                    ].copy()
                    adata_query_stim = adata_eval_sub[
                        adata_eval_sub.obs[cond_key] == stim_key
                    ].copy()
                    adata_query_pred = adata_eval_sub[
                        adata_eval_sub.obs[cond_key] == 'pred'
                    ].copy()

                    if (
                        adata_query_ctrl.n_obs < 2
                        or adata_query_stim.n_obs < 2
                        or adata_query_pred.n_obs < 2
                    ):
                        print(
                            f'[SKIP] patient={query_key}, seed={seed}, '
                            f'model={model_name}, cell_type={sub_class}: '
                            f'n_ctrl={adata_query_ctrl.n_obs}, '
                            f'n_stim={adata_query_stim.n_obs}, '
                            f'n_pred={adata_query_pred.n_obs}',
                            flush=True,
                        )
                        continue

                    top50_genes = get_top50_genes_for_cell_type(
                        adata_eval_sub=adata_eval_sub,
                        cond_key=cond_key,
                        ctrl_key=ctrl_key,
                        stim_key=stim_key,
                    )

                    if top50_genes is None or len(top50_genes) == 0:
                        continue

                    metrics = compute_metrics_for_one_cell_type(
                        adata_query_pred=adata_query_pred,
                        adata_query_stim=adata_query_stim,
                        top50_genes=top50_genes,
                        gammas=gammas,
                    )

                    metric_records.append({
                        'experiment': experiment_name,
                        'data_file': data_file,
                        'query_key': str(query_key),
                        'cell_type': str(sub_class),
                        'seed': seed,
                        'model': model_name,
                        'model_display': model_display_name(model_name),
                        'n_ctrl': adata_query_ctrl.n_obs,
                        'n_stim': adata_query_stim.n_obs,
                        'n_pred': adata_query_pred.n_obs,
                        **metrics,
                    })

    metrics_seed_df = pd.DataFrame(metric_records)
    metrics_seed_df.to_csv(
        f'{get_dataframe_dir(output_name)}/'
        f'cell_type_metrics_seed_level_ablation_{result_tag}.csv',
        index=False,
    )

    print(metrics_seed_df.groupby(['model']).size(), flush=True)

    return metrics_seed_df


def summarize_cell_type_metrics(
    output_name='across_patients_ablation',
    result_tag='all',
):
    metrics_path = (
        f'{get_dataframe_dir(output_name)}/'
        f'cell_type_metrics_seed_level_ablation_{result_tag}.csv'
    )
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

    group_cols = [
        'experiment',
        'data_file',
        'query_key',
        'cell_type',
        'model',
        'model_display',
    ]

    query_mean = (
        metrics_seed_df
        .groupby(group_cols)[metric_cols]
        .mean()
        .reset_index()
    )

    query_sd = (
        metrics_seed_df
        .groupby(group_cols)[metric_cols]
        .std()
        .reset_index()
    )

    summary = query_mean.merge(
        query_sd,
        on=group_cols,
        suffixes=('_mean', '_sd'),
    )

    summary.to_csv(
        f'{get_dataframe_dir(output_name)}/'
        f'cell_type_metrics_summary_ablation_{result_tag}.csv',
        index=False,
    )

    return metrics_seed_df, summary


def plot_cell_type_metric_heatmaps(
    output_name='across_patients_ablation',
    result_tag='all',
    data_file='pbmc_patients',
    model_names=DEFAULT_ABLATION_MODELS,
    metrics=('r2mean_all', 'r2mean_top50', 'mmd_top50'),
):
    ensure_dirs(get_figure_dir(output_name))

    summary = pd.read_csv(
        f'{get_dataframe_dir(output_name)}/'
        f'cell_type_metrics_summary_ablation_{result_tag}.csv'
    )

    model_order = default_model_order(model_names)

    metric_labels = {
        'r2mean_all': r'$R^2_{\mathrm{mean}}$ (↑; all genes)',
        'r2mean_top50': r'$R^2_{\mathrm{mean}}$ (↑; top 50 DEGs)',
        'mmd_top50': r'$\mathrm{MMD}^2$ (↓)',
    }

    for metric in metrics:
        mean_col = f'{metric}_mean'

        for query_key in sorted(summary['query_key'].astype(str).unique().tolist()):
            dfq = summary[summary['query_key'].astype(str) == str(query_key)].copy()

            heatmap_matrix = dfq.pivot(
                index='model',
                columns='cell_type',
                values=mean_col,
            )

            heatmap_matrix = heatmap_matrix.reindex(model_order)
            heatmap_matrix.index = [
                model_display_name(model_name)
                for model_name in heatmap_matrix.index
            ]

            plt.figure(
                figsize=(
                    max(7, 0.75 * heatmap_matrix.shape[1] + 3),
                    max(4, 0.55 * heatmap_matrix.shape[0] + 2),
                )
            )

            if metric.startswith('r2'):
                cmap = 'viridis'
                center = None
            else:
                cmap = 'mako_r'
                center = None

            sns.heatmap(
                heatmap_matrix,
                annot=True,
                fmt='.3f',
                cmap=cmap,
                center=center,
                linewidths=0.5,
                linecolor='white',
                cbar_kws={'label': metric_labels.get(metric, metric)},
            )

            plt.xlabel('Cell type')
            plt.ylabel('Model')
            plt.title(f'Held-out patient {query_key}: {metric_labels.get(metric, metric)}')
            plt.tight_layout()

            save_figure_jpg_pdf(
                f'{get_figure_dir(output_name)}/'
                f'Ablation_models_on_{data_file}_{query_key}_cell_type_heatmap_{metric}.jpg'
            )

            plt.close('all')


def plot_result(
    experiment_name='across_patients',
    output_name='across_patients_ablation',
    model_names=DEFAULT_ABLATION_MODELS,
    seeds=(1327, 1337, 1347),
    data_file='pbmc_patients',
    file_type='.h5ad',
    cond_key='condition',
    ctrl_key='ctrl',
    stim_key='stim',
    cell_label_key='sample_id',
    query_key='all',
    sub_key='cell_type',
    skip_missing=False,
):
    sns.set_theme(style='white', font='Arial', font_scale=1.2)

    ensure_dirs(
        get_figure_dir(output_name),
        get_dataframe_dir(output_name),
    )

    adata = sc.read_h5ad(f'../Data/{experiment_name}/{data_file}{file_type}')
    adata = adata[
        (adata.obs[cond_key] == ctrl_key)
        | (adata.obs[cond_key] == stim_key)
    ].copy()

    if query_key == 'all':
        query_keys = sorted(adata.obs[cell_label_key].astype(str).unique().tolist())
        result_tag = 'all'
    else:
        query_keys = [str(query_key)]
        result_tag = sanitize_result_tag(query_key)

    compute_cell_type_metrics_from_h5ad(
        experiment_name=experiment_name,
        output_name=output_name,
        model_names=model_names,
        seeds=seeds,
        data_file=data_file,
        file_type=file_type,
        cond_key=cond_key,
        ctrl_key=ctrl_key,
        stim_key=stim_key,
        cell_label_key=cell_label_key,
        query_keys=query_keys,
        sub_key=sub_key,
        skip_missing=skip_missing,
        result_tag=result_tag,
    )

    summarize_cell_type_metrics(
        output_name=output_name,
        result_tag=result_tag,
    )

    plot_cell_type_metric_heatmaps(
        output_name=output_name,
        result_tag=result_tag,
        data_file=data_file,
        model_names=model_names,
        metrics=('r2mean_all', 'r2mean_top50', 'mmd_top50'),
    )


if __name__ == '__main__':
    if args.seed == 'all':
        selected_seeds = (1327, 1337, 1347)
    else:
        selected_seeds = tuple(int(s.strip()) for s in args.seed.split(',') if s.strip())

    selected_models = tuple(
        model_name.strip()
        for model_name in args.models.split(',')
        if model_name.strip()
    )

    plot_result(
        experiment_name='across_patients',
        output_name=args.output_name,
        model_names=selected_models,
        query_key=args.query_key,
        seeds=selected_seeds,
        skip_missing=args.skip_missing,
    )

    print('Done', flush=True)