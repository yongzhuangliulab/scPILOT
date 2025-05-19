import torch
from typing import Iterable
from .mlp import MLP
class Generator(torch.nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        self.generator = MLP(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        self.linear_out = torch.nn.Linear(n_hidden, n_output)
    def forward(self, x: torch.Tensor, *cat_list: torch.Tensor):
        p = self.linear_out(self.generator(x, *cat_list))
        return p