from scvi.train import TrainingPlan
from .egd_network import EGD_network
from scvi._types import Tunable
from typing import Literal, Optional, Callable
from collections.abc import Iterable
from scvi.module.base import BaseModuleClass
import torch
import itertools
from torch.optim.lr_scheduler import ReduceLROnPlateau
TorchOptimizerCreator = Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer]
class EGDTrainingPlan(TrainingPlan):
    def __init__(
        self,
        module: BaseModuleClass,
        *,
        optimizer: Tunable[Literal['Adam', 'AdamW', 'Custom']] = 'Adam',
        optimizer_creator: Optional[TorchOptimizerCreator] = None,
        lr: Tunable[float] = 1e-3,
        weight_decay: Tunable[float] = 1e-6,
        n_steps_kl_warmup: Tunable[int] = None,
        n_epochs_kl_warmup: Tunable[int] = 400,
        reduce_lr_on_plateau: Tunable[bool] = False,
        lr_factor: Tunable[float] = 0.6,
        lr_patience: Tunable[int] = 30,
        lr_threshold: Tunable[float] = 0.0,
        lr_scheduler_metric: Literal[
            'elbo_validation', 'reconstruction_loss_validation', 'kl_local_validation'
        ] = 'elbo_validation',
        lr_min: float = 0,
        **loss_kwargs,
    ):
        super().__init__(
            module = module,
            optimizer = optimizer,
            optimizer_creator = optimizer_creator,
            lr = lr,
            weight_decay = weight_decay,
            n_steps_kl_warmup = n_steps_kl_warmup,
            n_epochs_kl_warmup = n_epochs_kl_warmup,
            reduce_lr_on_plateau = reduce_lr_on_plateau,
            lr_factor = lr_factor,
            lr_patience = lr_patience,
            lr_threshold = lr_threshold,
            lr_scheduler_metric = lr_scheduler_metric,
            lr_min = lr_min,
            **loss_kwargs,
        )
        self.automatic_optimization = False
    def training_step(self, batch, batch_idx):
        opts = self.optimizers()
        opt1, opt2 = opts
        _, _, egd_loss = self.forward(batch, loss_kwargs = self.loss_kwargs)
        loss = egd_loss.loss + egd_loss.extra_metrics['dl']
        self.log('train_loss', loss, on_step = False, on_epoch = True, prog_bar = True)
        self.log('vae_loss', egd_loss.loss, on_step = False, on_epoch = True, prog_bar = True)
        self.log('dl', egd_loss.extra_metrics['dl'], on_step = False, on_epoch = True, prog_bar = True)
        self.compute_and_log_metrics(egd_loss, self.train_metrics, 'train')
        opt1.zero_grad()
        self.manual_backward(egd_loss.loss)
        opt1.step()
        opt2.zero_grad()
        self.manual_backward(egd_loss.extra_metrics['dl'])
        opt2.step()
    def configure_optimizers(self):
        assert isinstance(self.module, EGD_network)
        params1 = itertools.chain(
            self.module.z_encoder.parameters(),
            self.module.generator.parameters()
        )
        optimizer1 = self.get_optimizer_creator()(params1)
        config1 = {'optimizer': optimizer1}
        if self.reduce_lr_on_plateau:
            scheduler1 = ReduceLROnPlateau(
                optimizer1,
                patience = self.lr_patience,
                factor = self.lr_factor,
                threshold = self.lr_threshold,
                min_lr = self.lr_min,
                threshold_mode = 'abs',
                verbose = True,
            )
            config1.update(
                {
                    'lr_scheduler':{
                        'scheduler': scheduler1,
                        'monitor': self.lr_scheduler_metric,
                    },
                },
            )
        params2 = self.module.discriminator.parameters()
        optimizer2 = torch.optim.Adam(
            params2, lr = 1e-3, eps = 0.01, weight_decay = self.weight_decay
        )
        config2 = {'optimizer': optimizer2}
        opts = [config1.pop('optimizer'), config2['optimizer']]
        if 'lr_scheduler' in config1:
            scheds = [config1['lr_scheduler']]
            return opts, scheds
        else:
            return opts