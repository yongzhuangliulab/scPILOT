import torch
from typing import Iterable, Optional, Callable
from .mlp import MLP
def _identity(x):
    return x
class Encoder(torch.nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        distribution: str = "normal",
        var_eps: float = 1e-4,
        var_activation: Optional[Callable] = None,
        return_latent_distribution: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.distribution = distribution
        self.var_eps = var_eps
        self.encoder = MLP(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        self.mean_encoder = torch.nn.Linear(n_hidden, n_output)
        self.var_encoder = torch.nn.Linear(n_hidden, n_output)
        self.return_latent_distribution = return_latent_distribution
        if distribution == 'ln':
            self.z_transformation = torch.nn.Softmax(dim=-1)
        else:
            self.z_transformation = _identity
        self.var_activation = torch.exp if var_activation is None else var_activation
    def forward(self, x: torch.Tensor, *cat_list: torch.Tensor):
        q: torch.Tensor = self.encoder(x, *cat_list)
        q_m: torch.Tensor = self.mean_encoder(q)
        q_v: torch.Tensor = self.var_activation(self.var_encoder(q)) + self.var_eps
        distribution = torch.distributions.Normal(q_m, q_v.sqrt())
        latent: torch.Tensor = self.z_transformation(distribution.rsample())
        if self.return_latent_distribution:
            return distribution, latent
        return q_m, q_v, latent