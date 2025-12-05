import torch
from .mlp import MLP
class Discriminator(torch.nn.Module):
    def __init__(
        self,
        n_input: int = 9999,
        n_output: int = 1,
        n_layers: int = 2,
        n_hidden: int = 800,
        dropout_rate: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        self.discriminator = MLP(
            n_input = n_input,
            n_output = n_hidden,
            n_layers = n_layers,
            n_hidden = n_hidden,
            dropout_rate = dropout_rate,
            **kwargs,
        )
        self.linear_out = torch.nn.Linear(n_hidden, n_output)
        self.regularize = torch.nn.Sigmoid()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f_D = self.discriminator(x)
        proba = self.regularize(self.linear_out(self.discriminator(x.detach())))
        return f_D, proba