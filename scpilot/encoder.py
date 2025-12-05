import torch
from typing import Optional, Callable
from .mlp import MLP
class Encoder(torch.nn.Module):
    def __init__(
        self,
        n_input: int = 9999,
        n_output: int = 100,
        n_layers: int = 2,
        n_hidden: int = 800,
        dropout_rate: float = 0.2,
        distribution: str = 'normal',
        var_eps: float = 1e-4,
        var_activation: Optional[Callable] = None,
        return_latent_distribution: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.distribution = distribution
        self.var_eps = var_eps
        self.encoder = MLP(
            n_input = n_input,
            n_output = n_hidden,
            n_layers = n_layers,
            n_hidden = n_hidden,
            dropout_rate = dropout_rate,
            **kwargs,
        )
        self.mean_encoder = torch.nn.Linear(n_hidden, n_output)
        self.var_encoder = torch.nn.Linear(n_hidden, n_output)
        self.return_latent_distribution = return_latent_distribution
        self.var_activation = torch.exp if var_activation is None else var_activation
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q: torch.Tensor = self.encoder(x)
        q_m: torch.Tensor = self.mean_encoder(q)
        q_v: torch.Tensor = self.var_activation(self.var_encoder(q)) + self.var_eps
        distribution = torch.distributions.Normal(q_m, q_v.sqrt())
        z: torch.Tensor = distribution.rsample().to(self.device)
        if self.return_latent_distribution:
            return distribution, z
        return q_m, q_v, z