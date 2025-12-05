import torch
import collections
class MLP(torch.nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int = 2,
        n_hidden: int = 800,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_activation: bool = True,
        bias: bool = True,
        activation_fn: torch.nn.Module = torch.nn.LeakyReLU,
    ):
        super().__init__()
        layers_dim = [n_input] + (n_layers - 1) * [n_hidden] + [n_output]
        self.mlp = torch.nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        f"Layer {i}",
                        torch.nn.Sequential(
                            torch.nn.Linear(
                                n_in,
                                n_out,
                                bias=bias,
                            ),
                            torch.nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
                            if use_batch_norm
                            else None,
                            torch.nn.LayerNorm(n_out, elementwise_affine=False)
                            if use_layer_norm
                            else None,
                            activation_fn() if use_activation else None,
                            torch.nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:]))
                ]
            )
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layers in enumerate(self.mlp):
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, torch.nn.BatchNorm1d):
                        if x.shape[0] == 1:
                            x = torch.zeros_like(x, device = self.device)
                        else:
                            x = layer(x)
                    else:
                        x = layer(x)
        return x