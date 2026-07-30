import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_DIR = '../DataFrames/discriminator_training_curves'
OUTPUT_DIR = '../Figures/discriminator_training_curves'

EXPECTED_SEEDS = 3

TRAIN_METRIC = 'gene_reconstruction_mmd_train'
VALIDATION_METRIC = 'gene_reconstruction_mmd_validation'
VALUE_COLUMN = 'gene_reconstruction_mmd'


# Slightly enlarge all text in the figures.
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 13,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 13,
})


def save_figure(path_without_suffix):
    plt.savefig(
        path_without_suffix + '.jpg',
        dpi=300,
        bbox_inches='tight',
    )
    plt.savefig(
        path_without_suffix + '.pdf',
        bbox_inches='tight',
    )


def load_curve_files():
    paths = sorted(
        glob(f'{INPUT_DIR}/across_*_seed*.csv')
    )

    if not paths:
        raise FileNotFoundError(
            f'No curve CSV files found in {INPUT_DIR}'
        )

    frames = []

    for path in paths:
        df = pd.read_csv(path)
        df['source_file'] = os.path.basename(path)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def make_long_dataframe(curve_df):
    id_columns = [
        'benchmark',
        'query_key',
        'seed',
        'split_seed',
        'model',
        'use_discriminator',
        'epoch',
        'global_step',
    ]

    long_df = curve_df.melt(
        id_vars=id_columns,
        value_vars=[
            TRAIN_METRIC,
            VALIDATION_METRIC,
        ],
        var_name='split',
        value_name=VALUE_COLUMN,
    )

    long_df['split'] = long_df['split'].replace({
        TRAIN_METRIC: 'Training',
        VALIDATION_METRIC: 'Validation',
    })

    return long_df


def summarize_curves(long_df):
    summary = (
        long_df
        .groupby(
            [
                'benchmark',
                'query_key',
                'model',
                'split',
                'epoch',
            ]
        )[VALUE_COLUMN]
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )

    # Only retain epochs for which all three seeds are available.
    summary = summary[
        summary['count'] == EXPECTED_SEEDS
    ].copy()

    summary.to_csv(
        f'{INPUT_DIR}/'
        f'discriminator_training_curve_summary.csv',
        index=False,
    )

    return summary


def plot_one_benchmark(summary, benchmark, query_key):
    subset = summary[
        (summary['benchmark'] == benchmark)
        & (summary['query_key'].astype(str) == str(query_key))
    ].copy()

    if subset.empty:
        raise ValueError(
            f'No summarized data for {benchmark}, {query_key}'
        )

    plt.figure(figsize=(8.5, 6.0))
    ax = plt.gca()

    model_order = [
        'scPILOT',
        'scPILOT_w_o_discriminator',
    ]

    for model_name in model_order:
        model_df = subset[
            subset['model'] == model_name
        ]

        train_df = (
            model_df[model_df['split'] == 'Training']
            .sort_values('epoch')
        )

        validation_df = (
            model_df[model_df['split'] == 'Validation']
            .sort_values('epoch')
        )

        if train_df.empty or validation_df.empty:
            continue

        train_line, = ax.plot(
            train_df['epoch'],
            train_df['mean'],
            linestyle='-',
            linewidth=1.8,
            label=f'{model_name}: training',
        )

        model_color = train_line.get_color()

        ax.fill_between(
            train_df['epoch'],
            train_df['mean'] - train_df['std'].fillna(0),
            train_df['mean'] + train_df['std'].fillna(0),
            color=model_color,
            alpha=0.12,
            linewidth=0,
        )

        ax.plot(
            validation_df['epoch'],
            validation_df['mean'],
            linestyle='--',
            linewidth=1.8,
            color=model_color,
            label=f'{model_name}: validation',
        )

        ax.fill_between(
            validation_df['epoch'],
            validation_df['mean']
            - validation_df['std'].fillna(0),
            validation_df['mean']
            + validation_df['std'].fillna(0),
            color=model_color,
            alpha=0.12,
            linewidth=0,
        )

    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'Minibatch_averaged reconstruction $\mathrm{MMD}^2$')
    ax.set_title(
        f'{benchmark}: held-out {query_key}\n'
        f'Mean ± SD across three seeds'
    )

    ax.grid(
        True,
        axis='y',
        linewidth=0.8,
        alpha=0.35,
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(
        frameon=False,
    )

    plt.tight_layout()

    save_figure(
        f'{OUTPUT_DIR}/'
        f'{benchmark}_{query_key}_'
        f'discriminator_training_validation_curve'
    )

    plt.close('all')


def plot_seed_panels(curve_df, benchmark, query_key):
    subset = curve_df[
        (curve_df['benchmark'] == benchmark)
        & (
            curve_df['query_key'].astype(str)
            == str(query_key)
        )
    ].copy()

    if subset.empty:
        raise ValueError(
            f'No data for {benchmark}, query={query_key}'
        )

    seeds = sorted(subset['seed'].unique().tolist())

    fig, axes = plt.subplots(
        1,
        len(seeds),
        figsize=(5.4 * len(seeds), 4.8),
        sharey=True,
    )

    if len(seeds) == 1:
        axes = [axes]

    model_order = [
        'scPILOT',
        'scPILOT_w_o_discriminator',
    ]

    for ax, seed in zip(axes, seeds):
        seed_df = subset[
            subset['seed'] == seed
        ].copy()

        for model_name in model_order:
            model_df = (
                seed_df[
                    seed_df['model'] == model_name
                ]
                .sort_values('epoch')
            )

            if model_df.empty:
                continue

            train_line, = ax.plot(
                model_df['epoch'],
                model_df[
                    TRAIN_METRIC
                ],
                linestyle='-',
                linewidth=1.7,
                label=f'{model_name}: training',
            )

            ax.plot(
                model_df['epoch'],
                model_df[
                    VALIDATION_METRIC
                ],
                linestyle='--',
                linewidth=1.7,
                color=train_line.get_color(),
                label=f'{model_name}: validation',
            )

            best_index = model_df[
                VALIDATION_METRIC
            ].idxmin()

            best_epoch = model_df.loc[
                best_index,
                'epoch',
            ]

            best_value = model_df.loc[
                best_index,
                VALIDATION_METRIC,
            ]

            ax.scatter(
                [best_epoch],
                [best_value],
                s=35,
                color=train_line.get_color(),
                zorder=3,
            )

        ax.set_title(f'Seed {seed}')
        ax.set_xlabel('Epoch')
        ax.grid(
            True,
            axis='y',
            linewidth=0.8,
            alpha=0.35,
        )
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylabel(
        r'Gene-space reconstruction $\mathrm{MMD}^2$'
    )

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        frameon=False,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.04),
        ncol=2,
        fontsize=10,
    )

    fig.suptitle(
        f'{benchmark}: held-out {query_key}',
        y=1.10,
        fontsize=13,
    )

    fig.tight_layout()

    save_figure(
        f'{OUTPUT_DIR}/'
        f'{benchmark}_{query_key}_'
        f'discriminator_curves_by_seed'
    )

    plt.close(fig)


def summarize_runs(curve_df):
    rows = []

    group_columns = [
        'benchmark',
        'query_key',
        'model',
        'seed',
    ]

    for group_values, group_df in curve_df.groupby(
        group_columns
    ):
        group_df = group_df.sort_values('epoch')

        best_index = group_df[
            VALIDATION_METRIC
        ].idxmin()

        best_row = group_df.loc[best_index]

        rows.append({
            'benchmark': group_values[0],
            'query_key': group_values[1],
            'model': group_values[2],
            'seed': group_values[3],
            'last_epoch': int(
                group_df['epoch'].max()
            ),
            'best_validation_epoch': int(
                best_row['epoch']
            ),
            'best_validation_mmd': float(
                best_row[
                    VALIDATION_METRIC
                ]
            ),
            'training_mmd_at_best_epoch': float(
                best_row[
                    TRAIN_METRIC
                ]
            ),
            'gap_at_best_epoch': float(
                best_row[
                    VALIDATION_METRIC
                ]
                - best_row[
                    TRAIN_METRIC
                ]
            ),
        })

    run_summary = pd.DataFrame(rows)

    run_summary.to_csv(
        f'{INPUT_DIR}/'
        f'discriminator_training_run_summary.csv',
        index=False,
    )

    return run_summary


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    curve_df = load_curve_files()
    long_df = make_long_dataframe(curve_df)
    summary = summarize_curves(long_df)

    plot_one_benchmark(
        summary=summary,
        benchmark='across_patients',
        query_key='101',
    )

    plot_one_benchmark(
        summary=summary,
        benchmark='across_species',
        query_key='mouse',
    )

    plot_seed_panels(
        curve_df=curve_df,
        benchmark='across_patients',
        query_key='101',
    )

    plot_seed_panels(
        curve_df=curve_df,
        benchmark='across_species',
        query_key='mouse',
    )

    run_summary = summarize_runs(curve_df)

    print(run_summary.to_string(index=False))

    print('Done', flush=True)


if __name__ == '__main__':
    main()