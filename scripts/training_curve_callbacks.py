from pathlib import Path

import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback


def metric_to_float(value):
    if value is None:
        return None

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu().item())

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class EpochCurveCSVLogger(Callback):
    """
    Save one row per completed validation epoch.

    The common gene-space reconstruction MSE is directly comparable between
    the full scPILOT model and the no-discriminator variant.
    """

    METRIC_KEYS = (
        'gene_reconstruction_mmd_train',
        'gene_reconstruction_mmd_validation',
        'VAE_loss_train',
        'VAE_loss_val',
        'elbo_train',
        'elbo_validation',
        'rl_train',
        'rl_validation',
        'kld_train',
        'kld_validation',
        'dl_train',
        'dl_validation',
    )

    def __init__(
        self,
        csv_path: str,
        benchmark: str,
        query_key: str,
        seed: int,
        split_seed: int,
        model_name: str,
        use_discriminator: bool,
    ):
        super().__init__()

        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        self.metadata = {
            'benchmark': benchmark,
            'query_key': str(query_key),
            'seed': int(seed),
            'split_seed': int(split_seed),
            'model': model_name,
            'use_discriminator': bool(use_discriminator),
        }

        self.rows = []
        self.last_written_epoch = None

    def on_validation_epoch_end(self, trainer, pl_module):
        # Avoid writing rows during Lightning's optional sanity check.
        if trainer.sanity_checking:
            return

        epoch = int(trainer.current_epoch + 1)

        # Prevent accidental duplicate rows.
        if self.last_written_epoch == epoch:
            return

        metrics = trainer.callback_metrics

        required_keys = (
            'gene_reconstruction_mmd_train',
            'gene_reconstruction_mmd_validation',
        )

        # Do not write an incomplete row.
        if not all(key in metrics for key in required_keys):
            return

        row = {
            **self.metadata,
            'epoch': epoch,
            'global_step': int(trainer.global_step),
        }

        for key in self.METRIC_KEYS:
            row[key] = metric_to_float(metrics.get(key))

        self.rows.append(row)
        self.last_written_epoch = epoch

        pd.DataFrame(self.rows).to_csv(
            self.csv_path,
            index=False,
        )

        print(
            '[Curve CSV] '
            f'epoch={epoch} | '
            f'train_mmd={row["gene_reconstruction_mmd_train"]:.6f} | '
            f'validation_mmd={row["gene_reconstruction_mmd_validation"]:.6f} | '
            f'path={self.csv_path}',
            flush=True,
        )


class EpochProgressPrinter(Callback):
    """Print the common reconstruction metrics after each validation epoch."""

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch + 1

        keys = (
            'gene_reconstruction_mmd_train',
            'gene_reconstruction_mmd_validation',
            'VAE_loss_train',
            'VAE_loss_val',
            'dl_train',
            'dl_validation',
        )

        message = f'[Epoch {epoch}/{trainer.max_epochs}]'

        for key in keys:
            if key not in metrics:
                continue

            value = metric_to_float(metrics[key])
            if value is not None:
                message += f' {key}={value:.6f}'

        print(message, flush=True)