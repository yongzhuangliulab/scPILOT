import numpy as np
import torch
from scvi import REGISTRY_KEYS
from typing import Literal, Optional
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from .encoder import Encoder
from .generator import Generator
from .discriminator import Discriminator
def _get_dict_if_none(param):
    param = {} if not isinstance(param, dict) else param

    return param
class EGD_network(BaseModuleClass):
    def __init__(
        self,
        n_input: int,
        n_hidden: int = 800,
        n_latent: int = 10,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        log_variational: bool = False,
        latent_distribution: str = "normal",
        use_batch_norm: Literal["none", "encoder", "generator", "discriminator", "E&G", "E&D", "G&D", "all"] = "all",
        use_layer_norm: Literal["none", "encoder", "generator", "discriminator", "E&G", "E&D", "G&D", "all"] = "none",
        # kl_weight: float = 0.00005,
        lambd1: float = 0.2,
        lambd2: float = 1.0,
        lambd3: float = 1e-3,
        lambd4: float = 0.05,
        eps = 1e-6,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.latent_distribution = "normal"
        # self.kl_weight = kl_weight
        self.lambd1 = lambd1
        self.lambd2 = lambd2
        self.lambd3 = lambd3
        self.lambd4 = lambd4
        self.eps = eps
        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "E&G" or use_batch_norm == "E&D" or use_batch_norm == "all"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "E&G" or use_layer_norm == "E&D" or use_layer_norm == "all"
        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            activation_fn=torch.nn.LeakyReLU,
        )
        n_input_generator = n_latent
        self.generator = Generator(
            n_input_generator,
            n_input,
            n_layers=n_layers,
            n_hidden=n_hidden,
            activation_fn=torch.nn.LeakyReLU,
            dropout_rate=dropout_rate,
        )
        self.discriminator = Discriminator(
            n_input,
            1,
            n_layers = n_layers,
            n_hidden = n_hidden,
            activation_fn = torch.nn.LeakyReLU,
            dropout_rate = dropout_rate,
        )
    @auto_move_data
    def forward(
        self,
        tensors,
        get_inference_input_kwargs: Optional[dict] = None,
        get_generative_input_kwargs: Optional[dict] = None,
        inference_kwargs: Optional[dict] = None,
        generative_kwargs: Optional[dict] = None,
        loss_kwargs: Optional[dict] = None,
        compute_loss=True,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, LossOutput]:
        inference_kwargs = _get_dict_if_none(inference_kwargs)
        generative_kwargs = _get_dict_if_none(generative_kwargs)
        loss_kwargs = _get_dict_if_none(loss_kwargs)
        get_inference_input_kwargs = _get_dict_if_none(get_inference_input_kwargs)
        get_generative_input_kwargs = _get_dict_if_none(get_generative_input_kwargs)
        inference_inputs = self._get_inference_input(tensors, **get_inference_input_kwargs)
        inference_outputs = self.inference(**inference_inputs, **inference_kwargs)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        get_generative_input_kwargs.update(
            {'z_p': Normal(0, 1).sample(inference_outputs['z'].size()).to(device)}
        )
        generative_inputs = self._get_generative_input(
            tensors, inference_outputs, **get_generative_input_kwargs
        )
        generative_outputs = self.generative(**generative_inputs, **generative_kwargs)
        if compute_loss:
            losses = self.loss(tensors, inference_outputs, generative_outputs, **loss_kwargs)
            return inference_outputs, generative_outputs, losses
        else:
            return inference_outputs, generative_outputs
    def _get_inference_input(self, tensors: dict[str, torch.Tensor]):
        x = tensors[REGISTRY_KEYS.X_KEY] # x = tensors['X']
        input_dict = dict(x = x)
        # input_dict = {"x": x}
        return input_dict
    def _get_generative_input(
        self,
        tensors,
        inference_outputs: dict[str, torch.Tensor],
        z_p: torch.Tensor
    ):
        input_dict = {
            'z': inference_outputs['z'],
            'z_p': z_p,
            'x': tensors[REGISTRY_KEYS.X_KEY],
        }
        return input_dict
    @auto_move_data
    def inference(self, x):
        qz_m, qz_v, z = self.z_encoder(x)
        outputs: dict[str, torch.Tensor] = dict(z = z, qz_m = qz_m, qz_v = qz_v)
        return outputs
    @auto_move_data
    def generative(self, z, z_p = None, x = None):
        px = self.generator(z)
        x_p = self.generator(z_p) if z_p is not None else None
        f_D_x, x_r = self.discriminator(x) if x is not None else (None, None)
        f_D_px, px_r = self.discriminator(px)
        f_D_x_p, x_p_r = self.discriminator(x_p) if x_p is not None else (None, None)
        return {
            'px': px,
            'x_p': x_p,
            'f_D_x': f_D_x,
            'x_r': x_r,
            'f_D_px': f_D_px,
            'px_r': px_r,
            'f_D_x_p': f_D_x_p,
            'x_p_r': x_p_r
        }
    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor],
        generative_outputs: dict[str, torch.Tensor],
    ):
        x = tensors[REGISTRY_KEYS.X_KEY]
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        px = generative_outputs["px"]
        f_D_x = generative_outputs['f_D_x']
        x_r = generative_outputs['x_r']
        f_D_px = generative_outputs['f_D_px']
        px_r = generative_outputs['px_r']
        f_D_x_p = generative_outputs['f_D_x_p']
        x_p_r = generative_outputs['x_p_r']
        kld = kl(
            Normal(qz_m, torch.sqrt(qz_v)),
            Normal(0, 1),
        ).sum(dim = 1)
        rl = self.lambd2 * (
                self.get_reconstruction_loss(px, x) + 
                self.get_reconstruction_loss(f_D_px, f_D_x)
            ) + \
            self.lambd3 * self.get_reconstruction_loss(f_D_x_p.mean(dim = 0, keepdim = True), f_D_x.mean(dim = 0, keepdim = True))
        dl = -(torch.log10(x_r + self.eps) + torch.log10(1 - px_r + self.eps) + torch.log10(1 - x_p_r + self.eps))
        loss = (0.5 * self.lambd1 * kld + 0.5 * rl).mean()
        return LossOutput(loss, rl, kld, extra_metrics = {'dl': (self.lambd4 * dl).mean()})
    @torch.no_grad()
    def sample(
        self,
        tensors,
        n_samples=1,
    ) -> np.ndarray:
        inference_kwargs = dict(n_samples=n_samples)
        _, generative_outputs, = self.forward(
            tensors,
            inference_kwargs=inference_kwargs,
            compute_loss=False,
        )
        px = Normal(generative_outputs["px"], 1).sample()
        return px.cpu().numpy()
    def get_reconstruction_loss(self, x: torch.Tensor, px: torch.Tensor) -> torch.Tensor:
        loss = ((x - px) ** 2).sum(dim = 1)
        return loss