import numpy as np
import pandas as pd
import anndata as ad
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt

def load_three_seed_metrics(
    experiment_name: str,
    data_file: str,
    query_keys,
    model_names,
    seeds,
):
    metric_dir = Path(f'../DataFrames/{experiment_name}')
    metric_paths = sorted(metric_dir.glob('*_metrics.csv'))

    if not metric_paths:
        raise FileNotFoundError(
            f'No metric CSV files found in {metric_dir}'
        )

    required_columns = {
        'experiment',
        'data_file',
        'query_key',
        'seed',
        'model',
        'r2mean_all',
        'mmd_top50',
    }

    frames = []

    for path in metric_paths:
        df = pd.read_csv(path)

        if not required_columns.issubset(df.columns):
            continue

        df['source_file'] = path.name
        frames.append(df)

    if not frames:
        raise ValueError(
            f'No valid metric files found for {experiment_name}'
        )

    metrics_df = pd.concat(
        frames,
        ignore_index=True,
    )

    metrics_df['query_key'] = (
        metrics_df['query_key']
        .astype(str)
    )

    metrics_df['seed'] = pd.to_numeric(
        metrics_df['seed'],
        errors='raise',
    ).astype(int)

    expected_query_keys = [
        str(query_key)
        for query_key in query_keys
    ]

    metrics_df = metrics_df[
        (metrics_df['experiment'] == experiment_name)
        & (metrics_df['data_file'] == data_file)
        & (metrics_df['query_key'].isin(expected_query_keys))
        & (metrics_df['model'].isin(model_names))
        & (metrics_df['seed'].isin(seeds))
    ].copy()

    # Remove exact duplicate rows, but do not silently accept
    # conflicting results for the same run.
    metrics_df = metrics_df.drop_duplicates()

    run_keys = [
        'query_key',
        'model',
        'seed',
    ]

    duplicated = metrics_df.duplicated(
        run_keys,
        keep=False,
    )

    if duplicated.any():
        duplicated_rows = metrics_df.loc[
            duplicated,
            run_keys + [
                'r2mean_all',
                'mmd_top50',
                'source_file',
            ],
        ].sort_values(run_keys)

        raise ValueError(
            'Multiple metric records were found for the same '
            f'query/model/seed combination:\n{duplicated_rows}'
        )

    # Check that every held-out context and model has all three seeds.
    expected_runs = pd.MultiIndex.from_product(
        [
            expected_query_keys,
            list(model_names),
            list(seeds),
        ],
        names=run_keys,
    )

    observed_runs = pd.MultiIndex.from_frame(
        metrics_df[run_keys]
    )

    missing_runs = expected_runs.difference(
        observed_runs
    )

    if len(missing_runs) > 0:
        raise ValueError(
            'Missing three-seed metric records:\n'
            f'{missing_runs.tolist()}'
        )

    # First level: average the three seeds within each held-out context.
    context_mean_df = (
        metrics_df
        .groupby(
            ['query_key', 'model'],
            as_index=False,
        )
        .agg(
            r2mean_all=('r2mean_all', 'mean'),
            mmd2_top50=('mmd_top50', 'mean'),
            seed_count=('seed', 'nunique'),
        )
    )

    # Second level: average equally across held-out contexts.
    dataset_mean_df = (
        context_mean_df
        .groupby(
            'model',
            as_index=False,
        )
        .agg(
            r2mean_all=('r2mean_all', 'mean'),
            mmd2_top50=('mmd2_top50', 'mean'),
            context_count=('query_key', 'nunique'),
        )
    )

    context_mean_df.to_csv(
        metric_dir / 'three_seed_context_summary.csv',
        index=False,
    )

    dataset_mean_df.to_csv(
        metric_dir / 'three_seed_dataset_summary.csv',
        index=False,
    )

    return context_mean_df, dataset_mean_df

def datasets_assess(
    datasets = [
        dict(
            author = 'Kang et al.\n(7 cell types)',
            experiment_name = 'across_cell_types',
            data_file = 'pbmc',
            file_type = '.h5ad',
            cond_key = 'condition',
            ctrl_key = 'control',
            stim_key = 'stimulated',
            cell_label_key = 'cell_type',
        ),
        dict(
            author = 'Kang et al.\n(8 patients)',
            experiment_name = 'across_patients',
            data_file = 'pbmc_patients',
            file_type = '.h5ad',
            cond_key = 'condition',
            ctrl_key = 'ctrl',
            stim_key = 'stim',
            cell_label_key = 'sample_id',
        ),
        dict(
            author = 'Hagai et al.',
            experiment_name = 'across_species',
            data_file = 'species',
            file_type = '.h5ad',
            cond_key = 'condition',
            ctrl_key = 'unst',
            stim_key = 'LPS6',
            cell_label_key = 'species',
        ),
        dict(
            author = 'Jiang et al.',
            experiment_name = 'across_cell_lines',
            data_file = 'IFNGR2',
            file_type = '.h5ad',
            cond_key = 'target_gene',
            ctrl_key = 'non-targeting',
            stim_key = 'IFNGR2',
            cell_label_key = 'cell_type',
        ),
    ],
    model_names: list[str] = ['scPILOT', 'scGen', 'CellOT', 'biolord', 'identity', 'VAEGAN'],
    seeds: tuple[int, ...] = (1327, 1337, 1347),
):
    def distance(x: np.ndarray, y: np.ndarray):
        return ((x - y) ** 2).sum()
    sns.set_theme(style = 'white', font = 'Arial', font_scale = 1.5)
    authors = []
    affiliations = []
    values = []
    values_r2 = []
    values_mmd = []
    for dataset in datasets:
        author = dataset['author']
        experiment_name = dataset['experiment_name']
        data_file = dataset['data_file']
        file_type = dataset['file_type']
        cond_key = dataset['cond_key']
        ctrl_key = dataset['ctrl_key']
        stim_key = dataset['stim_key']
        cell_label_key = dataset['cell_label_key']
        adata = ad.read_h5ad(f'../Data/{experiment_name}/{data_file}{file_type}')
        query_keys = sorted(adata.obs[cell_label_key].unique().tolist())
        learnability_dict = {}
        for query_key in query_keys:
            adata_query_ctrl = adata[(adata.obs[cell_label_key] == query_key) & (adata.obs[cond_key] == ctrl_key)].copy()
            adata_query_stim = adata[(adata.obs[cell_label_key] == query_key) & (adata.obs[cond_key] == stim_key)].copy()
            adata_other_ctrl = adata[(adata.obs[cell_label_key] != query_key) & (adata.obs[cond_key] == ctrl_key)].copy()
            mean_query_ctrl = np.mean(
                adata_query_ctrl.X.toarray()
                if hasattr(adata_query_ctrl.X, 'toarray')
                else adata_query_ctrl.X,
                axis = 0,
            )
            mean_query_stim = np.mean(
                adata_query_stim.X.toarray()
                if hasattr(adata_query_stim.X, 'toarray')
                else adata_query_stim.X,
                axis = 0,
            )
            mean_other_ctrl = np.mean(
                adata_other_ctrl.X.toarray()
                if hasattr(adata_other_ctrl.X, 'toarray')
                else adata_other_ctrl.X,
                axis = 0,
            )
            learnability = distance(
                mean_query_ctrl, mean_query_stim
            ) / (distance(
                mean_query_ctrl, mean_query_stim
            ) + distance(
                mean_query_ctrl, mean_other_ctrl
            ))
            learnability_dict.update(
                {query_key: learnability}
            )
        learnability_array = np.fromiter(learnability_dict.values(), dtype = np.float32)
        authors.append(author)
        affiliations.append('Learnability')
        values.append(learnability_array.mean())
        values_r2.append(learnability_array.mean())
        values_mmd.append(learnability_array.mean())
        context_metric_df, dataset_metric_df = (
            load_three_seed_metrics(
                experiment_name=experiment_name,
                data_file=data_file,
                query_keys=query_keys,
                model_names=model_names,
                seeds=seeds,
            )
        )

        identity_row = dataset_metric_df[
            dataset_metric_df['model'] == 'identity'
        ]

        if len(identity_row) != 1:
            raise ValueError(
                f'Expected one identity summary for {experiment_name}, '
                f'but found {len(identity_row)}.'
            )

        baseline_r2 = float(
            identity_row['r2mean_all'].iloc[0]
        )

        baseline_mmd2 = float(
            identity_row['mmd2_top50'].iloc[0]
        )

        for model_name in sorted(model_names):
            model_row = dataset_metric_df[
                dataset_metric_df['model'] == model_name
            ]

            if len(model_row) != 1:
                raise ValueError(
                    f'Expected one summary for {model_name} in '
                    f'{experiment_name}, but found {len(model_row)}.'
                )

            model_r2 = float(
                model_row['r2mean_all'].iloc[0]
            )

            model_mmd2 = float(
                model_row['mmd2_top50'].iloc[0]
            )

            score = (
                model_r2
                / (model_r2 + baseline_r2)
                + baseline_mmd2
                / (baseline_mmd2 + model_mmd2)
            ) / 2

            authors.append(author)
            affiliations.append(model_name)
            values.append(score)
            values_r2.append(model_r2)
            values_mmd.append(model_mmd2)
    learnabilityNscore_df = pd.DataFrame({
        'Dataset': authors,
        'Affiliations': affiliations,
        'Score': values,
    })
    print(f'learnabilityNscore_df:\n{learnabilityNscore_df}')
    learnabilityNscore_df = learnabilityNscore_df.replace('biolord', 'Biolord').replace('identity', 'Identity')
    ax = sns.lineplot(data = learnabilityNscore_df, x = 'Dataset', y = 'Score', hue = 'Affiliations', hue_order = [
        'Learnability', 'Biolord', 'CellOT', 'Identity', 'VAEGAN', 'scGen', 'scPILOT'
    ], style = 'Affiliations', markers = True)
    plt.legend(bbox_to_anchor = (1.01, 0.5), loc = 'center left', borderaxespad = 0)
    plt.savefig(f'../Figures/datasets_assess/learnabilityNscore_lineplot.jpg', dpi = 300, bbox_inches = 'tight')
    plt.savefig(f'../Figures/datasets_assess/learnabilityNscore_lineplot.pdf', bbox_inches = 'tight')
    plt.close()
    learnabilityNr2_df = pd.DataFrame({
        'Dataset': authors,
        'Affiliations': affiliations,
        'Score_r2': values_r2,
    })
    print(f'learnabilityNr2_df:\n{learnabilityNr2_df}')
    learnabilityNmmd_df = pd.DataFrame({
        'Dataset': authors,
        'Affiliations': affiliations,
        'Score_mmd': values_mmd,
    })
    print(f'learnabilityNmmd_df:\n{learnabilityNmmd_df}')
if __name__ == '__main__':
    datasets_assess()
    print('Done')