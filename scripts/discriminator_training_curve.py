import os
import random
import argparse

import numpy as np
import scanpy as sc
import torch
import wandb

from lightning import pytorch as pl
from scpilot.egd_model import EGD_model

from training_curve_callbacks import (
    EpochCurveCSVLogger,
    EpochProgressPrinter,
)


BENCHMARK_CONFIGS = {
    'across_patients': {
        'data_path': '../Data/across_patients/pbmc_patients.h5ad',
        'cond_key': 'condition',
        'ctrl_key': 'ctrl',
        'stim_key': 'stim',
        'context_key': 'sample_id',
        'default_query_key': '101',
    },
    'across_species': {
        'data_path': '../Data/across_species/species.h5ad',
        'cond_key': 'condition',
        'ctrl_key': 'unst',
        'stim_key': 'LPS6',
        'context_key': 'species',
        'default_query_key': 'mouse',
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Train full scPILOT or the no-discriminator variant and save '
            'comparable train/validation reconstruction curves.'
        )
    )

    parser.add_argument(
        '--benchmark',
        type=str,
        required=True,
        choices=tuple(BENCHMARK_CONFIGS.keys()),
    )

    parser.add_argument(
        '--query_key',
        type=str,
        required=True,
    )

    parser.add_argument(
        '--seed',
        type=int,
        required=True,
        choices=(1327, 1337, 1347),
    )

    parser.add_argument(
        '--split_seed',
        type=int,
        default=0,
    )

    parser.add_argument(
        '--use_discriminator',
        type=int,
        required=True,
        choices=(0, 1),
        help='1: complete scPILOT; 0: no-discriminator variant.',
    )

    return parser.parse_args()


def set_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    pl.seed_everything(seed, workers=True)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    args = parse_args()

    config = BENCHMARK_CONFIGS[args.benchmark]
    use_discriminator = bool(args.use_discriminator)

    model_name = (
        'scPILOT'
        if use_discriminator
        else 'scPILOT_w_o_discriminator'
    )

    output_dir = '../DataFrames/discriminator_training_curves'
    os.makedirs(output_dir, exist_ok=True)

    curve_csv_path = (
        f'{output_dir}/'
        f'{args.benchmark}_{args.query_key}_{model_name}_'
        f'seed{args.seed}.csv'
    )

    adata = sc.read_h5ad(config['data_path'])
    adata.obs_names_make_unique()

    cond_key = config['cond_key']
    ctrl_key = config['ctrl_key']
    stim_key = config['stim_key']
    context_key = config['context_key']

    adata = adata[
        adata.obs[cond_key].astype(str).isin(
            [str(ctrl_key), str(stim_key)]
        )
    ].copy()

    query_mask = (
        adata.obs[context_key].astype(str)
        == str(args.query_key)
    )

    stimulated_mask = (
        adata.obs[cond_key].astype(str)
        == str(stim_key)
    )

    # Same outer holdout protocol as the original experiments:
    # only stimulated cells from the query context are excluded.
    train = adata[
        ~(query_mask & stimulated_mask)
    ].copy()

    print(
        f'benchmark={args.benchmark} | '
        f'query_key={args.query_key} | '
        f'model={model_name} | '
        f'seed={args.seed} | '
        f'split_seed={args.split_seed}',
        flush=True,
    )

    print('Training AnnData:')
    print(train)

    set_seed(args.seed)

    model = EGD_model(
        train,
        use_discriminator=use_discriminator,
    )

    assert (
        model.module.use_discriminator
        is use_discriminator
    )

    if use_discriminator:
        assert model.module.discriminator is not None
    else:
        assert model.module.discriminator is None

    curve_callback = EpochCurveCSVLogger(
        csv_path=curve_csv_path,
        benchmark=args.benchmark,
        query_key=args.query_key,
        seed=args.seed,
        split_seed=args.split_seed,
        model_name=model_name,
        use_discriminator=use_discriminator,
    )

    progress_callback = EpochProgressPrinter()

    try:
        model.train(
            max_epochs=400,
            batch_size=32,
            early_stopping=True,
            early_stopping_patience=25,
            enable_progress_bar=False,
            callbacks=[
                curve_callback,
                progress_callback,
            ],
            datasplitter_kwargs={
                'random_state': args.split_seed,
            },
            wandb_project=(
                f'discriminator_curve_{args.benchmark}'
            ),
            experiment_name=(
                f'discriminator_curve_'
                f'{args.benchmark}_'
                f'{args.query_key}_'
                f'{model_name}_'
                f'seed{args.seed}'
            ),
        )
    finally:
        wandb.finish()

    print(f'Curve saved to: {curve_csv_path}', flush=True)
    print('Done', flush=True)


if __name__ == '__main__':
    main()