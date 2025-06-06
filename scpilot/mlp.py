import torch
from typing import Iterable
import collections
def one_hot(index: torch.Tensor, n_cat: int) -> torch.Tensor:
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(torch.float32)
class MLP(torch.nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_activation: bool = True,
        bias: bool = True,
        inject_covariates: bool = True,
        activation_fn: torch.nn.Module = torch.nn.ReLU,
    ):
        super().__init__()
        self.inject_covariates = inject_covariates
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]
        if n_cat_list is not None:
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []
        cat_dim = sum(self.n_cat_list)
        self.mlp = torch.nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        f"Layer {i}",
                        torch.nn.Sequential(
                            torch.nn.Linear(
                                n_in + cat_dim * self.inject_into_layer(i),
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
    def inject_into_layer(self, layer_num) -> bool:
        user_cond = layer_num == 0 or (layer_num > 0 and self.inject_covariates)
        return user_cond
    def set_online_update_hooks(self, hook_first_layer=True):
        self.hooks = []
        def _hook_fn_weight(grad):
            categorical_dims = sum(self.n_cat_list)
            new_grad = torch.zeros_like(grad)
            if categorical_dims > 0:
                new_grad[:, -categorical_dims:] = grad[:, -categorical_dims:]
            return new_grad
        def _hook_fn_zero_out(grad):
            return grad * 0
        for i, layers in enumerate(self.mlp):
            for layer in layers:
                if i == 0 and not hook_first_layer:
                    continue
                if isinstance(layer, torch.nn.Linear):
                    if self.inject_into_layer(i):
                        w = layer.weight.register_hook(_hook_fn_weight)
                    else:
                        w = layer.weight.register_hook(_hook_fn_zero_out)
                    self.hooks.append(w)
                    b = layer.bias.register_hook(_hook_fn_zero_out)
                    self.hooks.append(b)
    def forward(self, x: torch.Tensor, *cat_list: torch.Tensor):
        one_hot_cat_list: list[torch.Tensor] = []
        if len(self.n_cat_list) > len(cat_list):
            raise ValueError("nb. categorical args provided doesn't match init. params.")
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0 in init. params.")
            if n_cat > 1:
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat
                one_hot_cat_list += [one_hot_cat]
        for i, layers in enumerate(self.mlp):
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, torch.nn.BatchNorm1d):
                        if x.dim() == 3: # with copy of this batch
                            layer_slice_x_list = []
                            for slice_x in x:
                                if slice_x.shape[0] == 1:
                                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                                    layer_slice_x = torch.zeros(slice_x.shape).to(device)
                                else:
                                    layer_slice_x = layer(slice_x)
                                layer_slice_x_list.append(layer_slice_x.unsqueeze(0))
                            x = torch.cat(layer_slice_x_list, dim = 0)
                            # x = torch.cat([(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0)
                        else:
                            if x.shape[0] == 1:
                                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                                x = torch.zeros(x.shape).to(device)
                            else:
                                x = layer(x)
                    else:
                        if isinstance(layer, torch.nn.Linear) and self.inject_into_layer(i):
                            if x.dim() == 3: # with copy of this batch
                                one_hot_cat_list_layer = [
                                    o.unsqueeze(0).expand((x.size(0), o.size(0), o.size(1)))
                                    for o in one_hot_cat_list
                                ]
                            else:
                                one_hot_cat_list_layer = one_hot_cat_list
                            x = torch.cat((x, *one_hot_cat_list_layer), dim=-1)
                        x = layer(x)
        return x