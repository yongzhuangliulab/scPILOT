import sys
import warnings
import torch
import jax
import numpy as np
import pandas as pd
from typing import Optional, Union, Literal, Any
from lightning.pytorch.accelerators.accelerator import Accelerator
from lightning.pytorch.loggers import Logger, WandbLogger
from lightning.pytorch.trainer.connectors.accelerator_connector import _AcceleratorConnector
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning import pytorch as pl
from .egd_training_plan import EGDTrainingPlan
from .ann_data_splitter import AnnDataSplitter
class LoudEarlyStopping(EarlyStopping):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.early_stopping_reason = None
    def _evaluate_stopping_criteria(self, current: torch.Tensor) -> tuple[bool, str]:
        should_stop, reason = super()._evaluate_stopping_criteria(current)
        if should_stop:
            self.early_stopping_reason = reason
        return should_stop, reason
    def teardown(
        self,
        _trainer: pl.Trainer,
        _pl_module: pl.LightningModule,
        stage: Optional[str] = None,
    ) -> None:
        if self.early_stopping_reason is not None:
            print(self.early_stopping_reason)
class EGDTrainer(pl.Trainer):
    def __init__(
        self,
        accelerator: Optional[Union[str, Accelerator]] = None,
        devices: Optional[Union[list[int], str, int]] = None,
        benchmark: bool = True,
        check_val_every_n_epoch: Optional[int] = None,
        max_epochs: int = 400,
        default_root_dir: str = '../logging_dir/',
        enable_checkpointing: bool = False,
        num_sanity_val_steps: int = 0,
        enable_model_summary: bool = False,
        early_stopping: bool = True,
        early_stopping_monitor: Literal[
            'elbo_validation',
            'rl_validation',
            'kld_validation',
        ] = 'elbo_validation',
        early_stopping_min_delta: float = 0.00,
        early_stopping_patience: int = 25,
        early_stopping_mode: Literal['min', 'max'] = 'min',
        enable_progress_bar: bool = True,
        logger: Union[Optional[Logger], bool] = None,
        experiment_name: Optional[str] = None,
        log_every_n_steps: int = 10,
        learning_rate_monitor: bool = False,
        **kwargs,
    ):
        check_val_every_n_epoch = check_val_every_n_epoch if check_val_every_n_epoch is not None else sys.maxsize
        callbacks = kwargs.pop('callbacks', [])
        if early_stopping:
            early_stopping_callback = LoudEarlyStopping(
                monitor = early_stopping_monitor,
                min_delta = early_stopping_min_delta,
                patience = early_stopping_patience,
                mode = early_stopping_mode,
            )
            callbacks.append(early_stopping_callback)
            check_val_every_n_epoch = 1
        if learning_rate_monitor and not any(
            isinstance(c, LearningRateMonitor) for c in callbacks
        ):
            callbacks.append(LearningRateMonitor())
            check_val_every_n_epoch = 1
        if logger is None:
            logger = WandbLogger(
                name = experiment_name,
                save_dir = '../Wandb_logging_dir/',
                offline = True,
                project = kwargs.pop('wandb_project'),
            )
        super().__init__(
            accelerator = accelerator,
            devices = devices,
            benchmark = benchmark,
            check_val_every_n_epoch = check_val_every_n_epoch,
            max_epochs = max_epochs,
            default_root_dir = default_root_dir,
            enable_checkpointing = enable_checkpointing,
            num_sanity_val_steps = num_sanity_val_steps,
            enable_model_summary = enable_model_summary,
            logger = logger,
            log_every_n_steps = log_every_n_steps,
            enable_progress_bar = enable_progress_bar,
            callbacks = callbacks,
            **kwargs,
        )
class InvalidParameterError(Exception):
    def __init__(
        self,
        param: str,
        value: Any,
        valid: Optional[list[Any]] = None,
        additional_message: Optional[str] = None,
    ):
        self.message = f'Invalid value for `{param}`: {value}.'
        if valid is not None:
            self.message += f' Must be one of {valid}.'
        if additional_message is not None:
            self.message += f' {additional_message}'
        super().__init__(self.message)
    def __str__(self):
        return self.message
def parse_device_args(
    accelerator: str = 'auto',
    devices: Union[int, list[int], str] = 'auto',
    return_device: Optional[Literal['torch', 'jax']] = None,
    validate_single_device: bool = False,
):
    valid = [None, 'torch', 'jax']
    if return_device not in valid:
        raise InvalidParameterError(param = 'return_device', value = return_device, valid = valid)
    _validate_single_device = validate_single_device and devices != 'auto'
    cond1 = isinstance(devices, list) and len(devices) > 1
    cond2 = isinstance(devices, str) and ',' in devices
    cond3 = devices == -1
    if _validate_single_device and (cond1 or cond2 or cond3):
        raise ValueError('Only a single device can be specified for `device`.')
    connector = _AcceleratorConnector(accelerator = accelerator, devices = devices)
    _accelerator = connector._accelerator_flag
    _devices = connector._devices_flag
    if isinstance(_devices, list):
        device_idx = _devices[0]
    elif isinstance(_devices, str) and ',' in _devices:
        device_idx = _devices.split(',')[0]
    else:
        device_idx = _devices
    if devices == 'auto' and _accelerator != 'cpu':
        _devices = [device_idx]
    if return_device == 'torch':
        device = torch.device('cpu')
        if _accelerator != 'cpu':
            device = torch.device(f'{_accelerator}:{device_idx}')
        return _accelerator, _devices, device
    elif return_device == 'jax':
        device = jax.devices('cpu')[0]
        if _accelerator != 'cpu':
            device = jax.devices(_accelerator)[device_idx]
        return _accelerator, _devices, device
    return _accelerator, _devices
class EGDTrainRunner:
    def __init__(
        self,
        model,
        training_plan: EGDTrainingPlan,
        data_splitter: AnnDataSplitter,
        max_epochs: int,
        accelerator: str = 'auto',
        devices: Union[int, list[int], str] = 'auto',
        **trainer_kwargs,
    ):
        self.model = model
        self.training_plan = training_plan
        self.data_splitter = data_splitter
        accelerator, lightning_devices, device = parse_device_args(
            accelerator = accelerator,
            devices = devices,
            return_device = 'torch',
        )
        self.accelerator = accelerator
        self.lightning_devices = lightning_devices
        self.device = device
        if getattr(self.training_plan, 'reduce_lr_on_plateau', False):
            trainer_kwargs['learning_rate_monitor'] = True
        self.trainer = EGDTrainer(
            max_epochs = max_epochs,
            accelerator = accelerator,
            devices = lightning_devices,
            **trainer_kwargs,
        )
    def _update_history(self):
        if self.model.is_trained_ is True:
            if not isinstance(self.model.history_, dict):
                warnings.warn(
                    'Training history cannot be updated. Logger can be accessed from '
                    '`model.trainer.logger`',
                    UserWarning,
                )
                return
            else:
                new_history = self.trainer.logger.history
                for key, val in self.model.history_.items():
                    if key not in new_history:
                        continue
                    prev_len = len(val)
                    new_len = len(new_history[key])
                    index = np.arange(prev_len, prev_len + new_len)
                    new_history[key].index = index
                    self.model.history_[key] = pd.concat(
                        [
                            val,
                            new_history[key],
                        ]
                    )
                    self.model.history_[key].index.name = val.index.name
        else:
            try:
                self.model.history_ = self.trainer.logger.history
            except AttributeError:
                self.model.history_ = None
    def __call__(self):
        if hasattr(self.data_splitter, 'n_train'):
            self.training_plan.n_obs_training = self.data_splitter.n_train
        if hasattr(self.data_splitter, 'n_val'):
            self.training_plan.n_obs_validation = self.data_splitter.n_val
        self.trainer.fit(self.training_plan, self.data_splitter)
        self._update_history()
        self.model.train_indices_ = self.data_splitter.train_idx
        self.model.validation_indices_ = self.data_splitter.val_idx
        self.model.test_indices_ = self.data_splitter.test_idx
        self.model.module.eval()
        self.model.is_trained_ = True
        self.model.to_device(self.device)
        self.model.trainer = self.trainer