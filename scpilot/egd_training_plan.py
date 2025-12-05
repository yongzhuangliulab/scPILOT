import torch
import itertools
from inspect import signature
from torchmetrics import Metric
from collections import OrderedDict
from typing import Literal, Optional, Callable, Iterable, Union
from lightning.pytorch.strategies.ddp import DDPStrategy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning import pytorch as pl
from .egd_network import EGD_network
TorchOptimizerCreator = Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer]
class ElboMetric(Metric):
    full_state_update = False
    _N_OBS_MINIBATCH_KEY = 'n_obs_minibatch'
    def __init__(
        self,
        name: str,
        mode: Literal['train', 'validation'],
        interval: Literal['obs', 'batch'],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._name = name
        self._mode = mode
        self._interval = interval
        self.add_state('elbo_component', default = torch.tensor(0.0), dist_reduce_fx = 'sum')
        self.add_state('n_obs', default = torch.tensor(0.0), dist_reduce_fx = 'sum')
        self.add_state('n_batches', default = torch.tensor(0.0), dist_reduce_fx = 'sum')
    @property
    def mode(self):
        return self._mode
    @property
    def name(self):
        return f'{self._name}_{self.mode}'
    @name.setter
    def name(self, new_name):
        self._name = new_name
    @property
    def interval(self):
        return self._interval
    def get_intervals_recorded(self):
        if self.interval == 'obs':
            return self.n_obs
        elif self.interval == 'batch':
            return self.n_batches
        raise ValueError(f'Unrecognized interval: {self.interval}.')
    def update(
        self,
        **kwargs,
    ):
        if self._N_OBS_MINIBATCH_KEY not in kwargs:
            raise ValueError(f'Missing {self._N_OBS_MINIBATCH_KEY} value in metrics update.')
        if self._name not in kwargs:
            raise ValueError(f'Missing {self._name} value in metrics update.')
        elbo_component = kwargs[self._name]
        self.elbo_component += elbo_component
        n_obs_minibatch = kwargs[self._N_OBS_MINIBATCH_KEY]
        self.n_obs += n_obs_minibatch
        self.n_batches += 1
    def compute(self):
        return self.elbo_component / self.get_intervals_recorded()
class EGDTrainingPlan(pl.LightningModule):
    @staticmethod
    def _create_elbo_metric_components(mode: str, n_total: Optional[int] = None):
        rl = ElboMetric('rl', mode, 'obs')
        kld = ElboMetric('kld', mode, 'obs')
        elbo = rl + kld
        elbo.name = f'elbo_{mode}'
        collection = OrderedDict(
            [(metric.name, metric) for metric in [elbo, rl, kld]]
        )
        return elbo, rl, kld, collection
    def initialize_train_metrics(self):
        (
            self.elbo_train,
            self.rl_train,
            self.kld_train,
            self.train_metrics,
        ) = self._create_elbo_metric_components(mode = 'train', n_total = self.n_obs_training)
        self.elbo_train.reset()
    def initialize_val_metrics(self):
        (
            self.elbo_val,
            self.rl_val,
            self.kld_val,
            self.val_metrics,
        ) = self._create_elbo_metric_components(mode = 'validation', n_total = self.n_obs_validation)
        self.elbo_val.reset()
    def __init__(
        self,
        module: EGD_network,
        *,
        optimizer: Literal['Adam', 'AdamW', 'Custom'] = 'Adam',
        optimizer_creator: Optional[TorchOptimizerCreator] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        eps: float = 0.01,
        reduce_lr_on_plateau: bool = False,
        lr_factor: float = 0.6,
        lr_patience: int = 30,
        lr_threshold: float = 0.0,
        lr_scheduler_metric: Literal[
            'elbo_validation',
            'rl_validation',
            'kld_validation',
        ] = 'elbo_validation',
        lr_min: float = 0,
    ):
        super().__init__()
        self.module = module
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self.optimizer_name = optimizer
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.lr_scheduler_metric = lr_scheduler_metric
        self.lr_threshold = lr_threshold
        self.lr_min = lr_min
        self.optimizer_creator = optimizer_creator
        if self.optimizer_name == 'Custom' and self.optimizer_creator is None:
            raise ValueError('If optimizer is \'Custom\', `optimizer_creator` must be provided.')
        self._n_obs_training = None
        self._n_obs_validation = None
        # get argument information of self.module.loss()
        self._loss_args = set(signature(self.module.loss).parameters.keys())
        self.initialize_train_metrics()
        self.initialize_val_metrics()
        self.automatic_optimization = False
    @property
    def use_sync_dist(self):
        return isinstance(self.trainer.strategy, DDPStrategy)
    @property
    def n_obs_training(self):
        return self._n_obs_training
    @n_obs_training.setter
    def n_obs_training(self, n_obs: int):
        self._n_obs_training = n_obs
        self.initialize_train_metrics()
    @property
    def n_obs_validation(self):
        return self._n_obs_validation
    @n_obs_validation.setter
    def n_obs_validation(self, n_obs: int):
        self._n_obs_validation = n_obs
        self.initialize_val_metrics()
    def _optimizer_creator_fn(self, optimizer_cls: Union[torch.optim.Adam, torch.optim.AdamW]):
        return lambda params: optimizer_cls(params, lr = self.lr, eps = self.eps, weight_decay = self.weight_decay)
    def get_optimizer_creator(self):
        if self.optimizer_name == 'Adam':
            optim_creator = self._optimizer_creator_fn(torch.optim.Adam)
        elif self.optimizer_name == 'AdamW':
            optim_creator = self._optimizer_creator_fn(torch.optim.AdamW)
        elif self.optimizer_name == 'Custom':
            optim_creator = self.optimizer_creator
        else:
            raise ValueError('Optimizer not understood.')
        return optim_creator
    def configure_optimizers(self):
        assert isinstance(self.module, EGD_network)
        params1 = itertools.chain(
            self.module.z_encoder.parameters(),
            self.module.generator.parameters(),
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
                    'lr_scheduler': {
                        'scheduler': scheduler1,
                        'monitor': self.lr_scheduler_metric,
                    }
                }
            )
        params2 = self.module.discriminator.parameters()
        optimizer2 = self.get_optimizer_creator()(params2)
        config2 = {'optimizer': optimizer2}
        opts = [config1.pop('optimizer'), config2['optimizer']]
        if 'lr_scheduler' in config1:
            scheds = [config1['lr_scheduler']]
            return opts, scheds
        else:
            return opts
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    @torch.inference_mode()
    def compute_and_log_metrics(
        self,
        rl: torch.Tensor,
        kld: torch.Tensor,
        dl: torch.Tensor,
        metrics: dict[str, ElboMetric],
        mode: str,
    ):
        n_obs_minibatch = rl.shape[0]
        rl = rl.sum()
        kld = kld.sum()
        metrics[f'elbo_{mode}'].update(
            rl = rl,
            kld = kld,
            n_obs_minibatch = n_obs_minibatch,
        )
        self.log_dict(
            metrics,
            on_step = False,
            on_epoch = True,
            batch_size = n_obs_minibatch,
            sync_dist = self.use_sync_dist,
        )
        self.log(
            f'dl_{mode}',
            dl,
            on_step = False,
            on_epoch = True,
            batch_size = n_obs_minibatch,
            sync_dist = self.use_sync_dist,
        )
    def training_step(self, batch, batch_idx):
        opts = self.optimizers()
        opt1, opt2 = opts
        _, _, _, loss = self.forward(batch)
        self.log('VAE_loss_train', loss['VAE_loss'], prog_bar = True, logger = True, on_step = False, on_epoch = True)
        self.compute_and_log_metrics(loss['rl'], loss['kld'], loss['dl'], self.train_metrics, 'train')
        opt1.zero_grad()
        self.manual_backward(loss['VAE_loss'])
        opt1.step()
        opt2.zero_grad()
        self.manual_backward(loss['dl'])
        opt2.step()
    def validation_step(self, batch, batch_idx):
        _, _, _, loss = self.forward(batch)
        self.log('VAE_loss_val', loss['VAE_loss'], prog_bar = True, logger = True, on_step = False, on_epoch = True)
        self.compute_and_log_metrics(loss['rl'], loss['kld'], loss['dl'], self.val_metrics, 'validation')