import os
import random
import argparse

import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.metrics import pairwise
from matplotlib import pyplot as plt

from cellot.models import load_autoencoder_model, load_cellot_model
from cellot.utils import load_config
from scpilot.egd_model import EGD_model


parser = argparse.ArgumentParser(description="CellOT across-species perturbation prediction")
parser.add_argument(
    "--query_key",
    type=str,
    default=None,
    help="Species to predict. Use all or omit to run all species.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=1327,
    help="Training/inference seed. Recommended: 1327, 1337, 1347.",
)
parser.add_argument(
    "--split_seed",
    type=int,
    default=0,
    help="Recorded split seed for consistency with other methods.",
)
args = parser.parse_args()


def set_seed(seed: int, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print(f"Global seed set to {seed}", flush=True)


def ensure_dirs(*paths: str):
    for path in paths:
        os.makedirs(path, exist_ok=True)


def to_dense_array(x):
    if sparse.issparse(x) or hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x)


def mmd_distance(x, y, gamma):
    xx = pairwise.rbf_kernel(x, x, gamma)
    xy = pairwise.rbf_kernel(x, y, gamma)
    yy = pairwise.rbf_kernel(y, y, gamma)
    return xx.mean() + yy.mean() - 2 * xy.mean()


def compute_mmd_loss(lhs, rhs, gammas):
    return float(np.mean([mmd_distance(lhs, rhs, g) for g in gammas]))


def move_model_to_device(model, device):
    if isinstance(model, (tuple, list)):
        for module in model:
            if hasattr(module, "to"):
                module.to(device)
    elif hasattr(model, "to"):
        model.to(device)
    return model


def predict_one_query(
    adata,
    experiment_name,
    data_file,
    file_type,
    cond_key,
    ctrl_key,
    stim_key,
    cell_label_key,
    query_key,
    seed,
    split_seed,
):
    set_seed(seed)
    print(
        f"======Predicting {query_key} | seed={seed} | split_seed={split_seed}======",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # condition_species.txt is expected to contain entries such as LPS6_mouse.
    condition_species = f"{stim_key}_{query_key}"
    path_ae = (
        f"../cellot_results/{experiment_name}/"
        f"seed-{seed}/holdout-{condition_species}/mode-ood/model-scgen"
    )
    path_cellot = (
        f"../cellot_results/{experiment_name}/"
        f"seed-{seed}/holdout-{query_key}/mode-ood/model-cellot"
    )

    print(f"AE path: {path_ae}", flush=True)
    print(f"CellOT path: {path_cellot}", flush=True)

    if not os.path.exists(f"{path_ae}/cache/model.pt"):
        raise FileNotFoundError(f"Missing AE checkpoint: {path_ae}/cache/model.pt")
    if not os.path.exists(f"{path_cellot}/cache/model.pt"):
        raise FileNotFoundError(f"Missing CellOT checkpoint: {path_cellot}/cache/model.pt")

    ae_model, _ = load_autoencoder_model(
        load_config(f"{path_ae}/config.yaml"),
        restore=f"{path_ae}/cache/model.pt",
        input_dim=adata.n_vars,
    )
    ae_model = ae_model.to(device).eval()

    adata_query_ctrl = adata[
        (adata.obs[cell_label_key].astype(str) == str(query_key))
        & (adata.obs[cond_key].astype(str) == str(ctrl_key))
    ].copy()
    adata_query_stim = adata[
        (adata.obs[cell_label_key].astype(str) == str(query_key))
        & (adata.obs[cond_key].astype(str) == str(stim_key))
    ].copy()
    adata_query = adata[adata.obs[cell_label_key].astype(str) == str(query_key)].copy()

    n_top_degs = 50
    sc.tl.rank_genes_groups(
        adata_query,
        groupby=cond_key,
        method="wilcoxon",
        n_genes=n_top_degs,
    )
    top50_genes = adata_query.uns["rank_genes_groups"]["names"][stim_key].tolist()

    inputs = torch.as_tensor(
        to_dense_array(adata_query_ctrl.X),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        latent_query_ctrl_x = ae_model.encode(inputs).detach().cpu().numpy()

    latent_query_ctrl = ad.AnnData(
        latent_query_ctrl_x,
        obs=adata_query_ctrl.obs.copy(),
        obsm=adata_query_ctrl.obsm.copy(),
    )

    cellot_model, _ = load_cellot_model(
        load_config(f"{path_cellot}/config.yaml"),
        restore=f"{path_cellot}/cache/model.pt",
        input_dim=latent_query_ctrl.n_vars,
    )
    cellot_model = move_model_to_device(cellot_model, device)

    # CellOT transport is implemented as a gradient of the learned potential,
    # so gradients must be enabled for the input tensor even during inference.
    inputs = torch.as_tensor(
        to_dense_array(latent_query_ctrl.X),
        dtype=torch.float32,
        device=device,
    )
    inputs = inputs.detach().clone().requires_grad_(True)

    transport_net = cellot_model[1].eval()
    with torch.enable_grad():
        transported = transport_net.transport(inputs)

    transported = transported.detach().cpu().numpy()

    latent_inputs = torch.as_tensor(
        transported,
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        adata_query_pred_x = ae_model.decode(latent_inputs).detach().cpu().numpy()

    adata_query_pred = ad.AnnData(
        adata_query_pred_x,
        obs=adata_query_ctrl.obs.copy(),
        obsm=adata_query_ctrl.obsm.copy(),
    )
    adata_query_pred.var_names = adata_query_ctrl.var_names
    adata_query_pred.obs[cond_key] = "pred"
    adata_query_pred.obs_names = [
        f"{idx}_pred" for idx in adata_query_ctrl.obs_names.astype(str)
    ]

    adata_query_eval = ad.concat([adata_query_stim, adata_query_pred])
    adata_query_eval.obs_names_make_unique()

    plt.figure()
    r2mean_all, r2mean_top50 = EGD_model.reg_mean_plot(
        adata_query_eval,
        cond_key=cond_key,
        axis_keys={"x": "pred", "y": stim_key},
        labels={"x": "Prediction", "y": "Ground truth"},
        path_to_save=(
            f"../Figures/{experiment_name}/"
            f"CellOT_{data_file}_reg_mean_{query_key}_seed{seed}.jpg"
        ),
        gene_list=top50_genes[:10],
        show=False,
        top_genes=top50_genes,
        top_gene_label="T50",
        legend=False,
    )
    plt.close("all")

    gammas = np.logspace(1, -3, num=50)
    x = to_dense_array(adata_query_pred[:, top50_genes].X)
    y = to_dense_array(adata_query_stim[:, top50_genes].X)
    mmd = compute_mmd_loss(x, y, gammas=gammas)

    print("CellOT:", flush=True)
    print(f"{query_key}: r2mean_all = {r2mean_all}", flush=True)
    print(f"{query_key}: r2mean_top50 = {r2mean_top50}", flush=True)
    print(f"{query_key}: mmd = {mmd}", flush=True)

    adata_query_eval_to_save = ad.concat([
        adata_query_ctrl,
        adata_query_stim,
        adata_query_pred,
    ])
    adata_query_eval_to_save.obs_names_make_unique()
    adata_query_eval_to_save.write_h5ad(
        f"../Result_anndata/{experiment_name}/"
        f"{experiment_name}_CellOT_{data_file}_{query_key}_seed{seed}{file_type}"
    )

    metrics_df = pd.DataFrame([
        {
            "experiment": experiment_name,
            "data_file": data_file,
            "query_key": query_key,
            "seed": seed,
            "split_seed": split_seed,
            "model": "CellOT",
            "r2mean_all": r2mean_all,
            "r2mean_top50": r2mean_top50,
            "mmd_top50": mmd,
        }
    ])
    metrics_df.to_csv(
        f"../DataFrames/{experiment_name}/"
        f"{experiment_name}_{data_file}_{query_key}_seed{seed}_CellOT_metrics.csv",
        index=False,
    )


def predict_perturbation(
    experiment_name="across_species",
    data_file="species",
    file_type=".h5ad",
    cond_key="condition",
    ctrl_key="unst",
    stim_key="LPS6",
    cell_label_key="species",
    query_key=None,
    seed=1327,
    split_seed=0,
):
    set_seed(seed)
    ensure_dirs(
        f"../Result_anndata/{experiment_name}",
        f"../Figures/{experiment_name}",
        f"../DataFrames/{experiment_name}",
    )

    adata = sc.read_h5ad(f"../Data/{experiment_name}/{data_file}{file_type}")
    adata = adata[
        (adata.obs[cond_key].astype(str) == str(ctrl_key))
        | (adata.obs[cond_key].astype(str) == str(stim_key))
    ].copy()

    if query_key is None or str(query_key).lower() == "all":
        query_keys = sorted(adata.obs[cell_label_key].astype(str).unique().tolist())
    else:
        query_keys = [str(query_key)]

    for current_query_key in query_keys:
        predict_one_query(
            adata=adata,
            experiment_name=experiment_name,
            data_file=data_file,
            file_type=file_type,
            cond_key=cond_key,
            ctrl_key=ctrl_key,
            stim_key=stim_key,
            cell_label_key=cell_label_key,
            query_key=current_query_key,
            seed=seed,
            split_seed=split_seed,
        )


if __name__ == "__main__":
    predict_perturbation(
        query_key=args.query_key,
        seed=args.seed,
        split_seed=args.split_seed,
    )
    print("Done", flush=True)
