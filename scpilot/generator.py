import torch
from .mlp import MLP
class Generator(torch.nn.Module):
    def __init__(
        self,
        n_input: int = 100,
        n_output: int = 9999,
        n_layers: int = 2,
        n_hidden: int = 800,
        dropout_rate: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        self.generator = MLP(
            n_input = n_input,
            n_output = n_hidden,
            n_layers = n_layers,
            n_hidden = n_hidden,
            dropout_rate = dropout_rate,
            **kwargs,
        )
        self.linear_out = torch.nn.Linear(n_hidden, n_output)
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        xHat = self.linear_out(self.generator(z))
        return xHat