from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from typing import Iterable, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise

from scpilot.egd_model import EGD_model, ot_naive


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set Python, NumPy, PyTorch, and Lightning random seeds."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def ensure_dirs(*paths: str | Path) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def obs_value_mask(adata: ad.AnnData, key: str, value: object) -> np.ndarray:
    """Robust equality mask for integer/string observation labels."""
    return adata.obs[key].astype(str).to_numpy() == str(value)


def obs_values_mask(
    adata: ad.AnnData,
    key: str,
    values: Iterable[object],
) -> np.ndarray:
    value_set = {str(value) for value in values}
    return adata.obs[key].astype(str).isin(value_set).to_numpy()


def to_dense_array(x) -> np.ndarray:
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def mmd_distance(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    xx = pairwise.rbf_kernel(x, x, gamma)
    xy = pairwise.rbf_kernel(x, y, gamma)
    yy = pairwise.rbf_kernel(y, y, gamma)
    return float(xx.mean() + yy.mean() - 2.0 * xy.mean())


def compute_mmd2(
    lhs: np.ndarray,
    rhs: np.ndarray,
    gammas: Sequence[float],
) -> float:
    return float(np.mean([mmd_distance(lhs, rhs, gamma) for gamma in gammas]))


def _stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def deterministic_subsample(
    adata: ad.AnnData,
    max_cells: int,
    *seed_parts: object,
) -> ad.AnnData:
    """Subsample cells without replacement using a stable group-specific seed."""
    if adata.n_obs == 0:
        raise ValueError("Cannot subsample an empty AnnData object.")

    if max_cells <= 0 or adata.n_obs <= max_cells:
        return adata.copy()

    rng = np.random.default_rng(_stable_seed(*seed_parts))
    indices = np.sort(
        rng.choice(adata.n_obs, size=max_cells, replace=False)
    )
    return adata[indices].copy()


def posterior_mean_embedding(
    model: EGD_model,
    adata: ad.AnnData,
    max_cells: int,
    *seed_parts: object,
) -> np.ndarray:
    """Return deterministic posterior-mean embeddings for a bounded cell sample."""
    sampled = deterministic_subsample(
        adata,
        max_cells,
        *seed_parts,
    )
    q_mean, _, _ = model.get_latent_representation(sampled)
    return np.asarray(q_mean, dtype=np.float64)


def latent_ot_cost(
    model: EGD_model,
    query_adata: ad.AnnData,
    reference_adata: ad.AnnData,
    max_cells: int,
    *seed_parts: object,
) -> float:
    """Compute balanced OT cost between posterior-mean latent embeddings."""
    query_mu = posterior_mean_embedding(
        model,
        query_adata,
        max_cells,
        *seed_parts,
        "query",
    )
    reference_mu = posterior_mean_embedding(
        model,
        reference_adata,
        max_cells,
        *seed_parts,
        "reference",
    )
    _, cost = ot_naive(query_mu, reference_mu)
    return float(cost)


def rank_cell_type_references(
    model: EGD_model,
    model_adata: ad.AnnData,
    query_key: object,
    cell_label_key: str,
    cond_key: str,
    ctrl_key: object,
    max_cells_per_group: int,
    benchmark: str,
    seed: int,
) -> pd.DataFrame:
    """Rank reference cell types by control-state latent OT cost."""
    query_ctrl = model_adata[
        obs_value_mask(model_adata, cell_label_key, query_key)
        & obs_value_mask(model_adata, cond_key, ctrl_key)
    ].copy()

    if query_ctrl.n_obs == 0:
        raise ValueError(
            f"No query control cells found for {cell_label_key}={query_key!r}."
        )

    # Use contexts with both control and non-control cells in model.adata.
    reference_keys = []
    for value in sorted(model_adata.obs[cell_label_key].astype(str).unique()):
        if value == str(query_key):
            continue
        context = model_adata[obs_value_mask(model_adata, cell_label_key, value)]
        has_ctrl = np.any(obs_value_mask(context, cond_key, ctrl_key))
        has_non_ctrl = np.any(~obs_value_mask(context, cond_key, ctrl_key))
        if has_ctrl and has_non_ctrl:
            reference_keys.append(value)

    rows = []
    for reference_key in reference_keys:
        reference_ctrl = model_adata[
            obs_value_mask(model_adata, cell_label_key, reference_key)
            & obs_value_mask(model_adata, cond_key, ctrl_key)
        ].copy()

        score = latent_ot_cost(
            model,
            query_ctrl,
            reference_ctrl,
            max_cells_per_group,
            benchmark,
            query_key,
            reference_key,
            seed,
        )

        rows.append(
            {
                "benchmark": benchmark,
                "query_key": str(query_key),
                "seed": int(seed),
                "reference_key": str(reference_key),
                "mismatch_score": score,
                "n_query_ctrl": int(query_ctrl.n_obs),
                "n_reference_ctrl": int(reference_ctrl.n_obs),
                "n_covariates": 1,
                "max_cells_per_group": int(max_cells_per_group),
            }
        )

    ranking = pd.DataFrame(rows).sort_values(
        ["mismatch_score", "reference_key"],
        ascending=[True, True],
    ).reset_index(drop=True)
    ranking["rank_nearest"] = np.arange(1, len(ranking) + 1)
    ranking["rank_farthest"] = np.arange(len(ranking), 0, -1)
    return ranking


def rank_patient_references(
    model: EGD_model,
    model_adata: ad.AnnData,
    query_key: object,
    patient_key: str,
    cov_key: str,
    cond_key: str,
    ctrl_key: object,
    stim_key: object,
    max_cells_per_group: int,
    benchmark: str,
    seed: int,
) -> pd.DataFrame:
    """
    Rank reference patients using a query-cell-count-weighted mean of
    within-cell-type control-state latent OT costs.

    Every candidate reference patient must contain control cells for every
    query cell type. This is also required by the cov_weight=1.0 prediction
    protocol when a restricted reference subset is used.
    """
    query_ctrl = model_adata[
        obs_value_mask(model_adata, patient_key, query_key)
        & obs_value_mask(model_adata, cond_key, ctrl_key)
    ].copy()

    if query_ctrl.n_obs == 0:
        raise ValueError(
            f"No query control cells found for {patient_key}={query_key!r}."
        )

    query_covariates = sorted(query_ctrl.obs[cov_key].astype(str).unique())
    query_counts = {
        covariate: int(
            np.sum(query_ctrl.obs[cov_key].astype(str).to_numpy() == covariate)
        )
        for covariate in query_covariates
    }
    total_query_cells = float(sum(query_counts.values()))

    reference_keys = []
    for value in sorted(model_adata.obs[patient_key].astype(str).unique()):
        if value == str(query_key):
            continue
        context = model_adata[obs_value_mask(model_adata, patient_key, value)]
        has_ctrl = np.any(obs_value_mask(context, cond_key, ctrl_key))
        has_stim = np.any(obs_value_mask(context, cond_key, stim_key))
        if has_ctrl and has_stim:
            reference_keys.append(value)

    rows = []
    invalid_references = []

    for reference_key in reference_keys:
        weighted_cost = 0.0
        per_covariate_costs = []
        missing_covariates = []
        n_reference_ctrl = 0

        for covariate in query_covariates:
            query_group = query_ctrl[
                query_ctrl.obs[cov_key].astype(str).to_numpy() == covariate
            ].copy()
            reference_group = model_adata[
                obs_value_mask(model_adata, patient_key, reference_key)
                & obs_value_mask(model_adata, cond_key, ctrl_key)
                & (
                    model_adata.obs[cov_key].astype(str).to_numpy()
                    == covariate
                )
            ].copy()

            if reference_group.n_obs == 0:
                missing_covariates.append(covariate)
                continue

            cost = latent_ot_cost(
                model,
                query_group,
                reference_group,
                max_cells_per_group,
                benchmark,
                query_key,
                reference_key,
                covariate,
                seed,
            )
            weight = query_counts[covariate] / total_query_cells
            weighted_cost += weight * cost
            per_covariate_costs.append(cost)
            n_reference_ctrl += int(reference_group.n_obs)

        if missing_covariates:
            invalid_references.append(
                (reference_key, missing_covariates)
            )
            continue

        rows.append(
            {
                "benchmark": benchmark,
                "query_key": str(query_key),
                "seed": int(seed),
                "reference_key": str(reference_key),
                "mismatch_score": float(weighted_cost),
                "mean_covariate_ot_cost": float(np.mean(per_covariate_costs)),
                "n_query_ctrl": int(query_ctrl.n_obs),
                "n_reference_ctrl": int(n_reference_ctrl),
                "n_covariates": int(len(query_covariates)),
                "max_cells_per_group": int(max_cells_per_group),
            }
        )

    if invalid_references:
        details = "; ".join(
            f"{reference}: missing {missing}"
            for reference, missing in invalid_references
        )
        raise ValueError(
            "The patient mismatch stress test requires every candidate "
            "reference patient to contain control cells for every query cell "
            f"type. Invalid references: {details}"
        )

    ranking = pd.DataFrame(rows).sort_values(
        ["mismatch_score", "reference_key"],
        ascending=[True, True],
    ).reset_index(drop=True)
    ranking["rank_nearest"] = np.arange(1, len(ranking) + 1)
    ranking["rank_farthest"] = np.arange(len(ranking), 0, -1)
    return ranking


def choose_reference_sets(
    ranking: pd.DataFrame,
    k: int,
) -> dict[str, list[str]]:
    if len(ranking) < 2 * k:
        raise ValueError(
            f"Need at least {2 * k} references for disjoint nearest-{k} and "
            f"farthest-{k} sets, but found {len(ranking)}."
        )

    nearest = ranking.head(k)["reference_key"].astype(str).tolist()
    farthest = ranking.tail(k)["reference_key"].astype(str).tolist()
    all_references = ranking["reference_key"].astype(str).tolist()

    overlap = set(nearest) & set(farthest)
    if overlap:
        raise RuntimeError(
            f"Nearest and farthest reference sets overlap: {sorted(overlap)}"
        )

    return {
        "all": all_references,
        f"nearest_{k}": nearest,
        f"farthest_{k}": farthest,
    }


def make_restricted_model_adata(
    original_model_adata: ad.AnnData,
    context_key: str,
    query_key: object,
    selected_reference_keys: Sequence[object],
    cond_key: str,
    ctrl_key: object,
    stim_key: object,
) -> ad.AnnData:
    """Keep query controls plus both conditions for selected references."""
    query_ctrl_mask = (
        obs_value_mask(original_model_adata, context_key, query_key)
        & obs_value_mask(original_model_adata, cond_key, ctrl_key)
    )
    reference_mask = obs_values_mask(
        original_model_adata,
        context_key,
        selected_reference_keys,
    )

    restricted = original_model_adata[query_ctrl_mask | reference_mask].copy()

    if np.any(
        obs_value_mask(restricted, context_key, query_key)
        & obs_value_mask(restricted, cond_key, stim_key)
    ):
        raise RuntimeError("Held-out stimulated query cells leaked into model.adata.")

    for reference_key in selected_reference_keys:
        context = restricted[
            obs_value_mask(restricted, context_key, reference_key)
        ]
        has_ctrl = np.any(obs_value_mask(context, cond_key, ctrl_key))
        has_stim = np.any(obs_value_mask(context, cond_key, stim_key))
        if not (has_ctrl and has_stim):
            raise ValueError(
                f"Selected reference {reference_key!r} does not contain both "
                f"{ctrl_key!r} and {stim_key!r} cells."
            )

    return restricted


def evaluate_prediction(
    adata_query_stim: ad.AnnData,
    adata_query_pred: ad.AnnData,
    cond_key: str,
    stim_key: object,
    top50_genes: Sequence[str],
    gammas: Sequence[float],
) -> tuple[float, float, float]:
    """Evaluate R2mean_all, R2mean_top50, and biased MMD2 on top-50 genes."""
    adata_query_pred = adata_query_pred.copy()
    adata_query_pred.obs[cond_key] = "pred"

    adata_query_eval = ad.concat(
        [adata_query_stim, adata_query_pred],
        join="inner",
    )

    plt.figure()
    r2mean_all, r2mean_top50 = EGD_model.reg_mean_plot(
        adata_query_eval,
        cond_key=cond_key,
        axis_keys={"x": "pred", "y": stim_key},
        labels={"x": "Prediction", "y": "Ground truth"},
        save=False,
        gene_list=list(top50_genes[:10]),
        show=False,
        top_genes=list(top50_genes),
        top_gene_label="T50",
        legend=False,
    )
    plt.close("all")

    predicted = to_dense_array(adata_query_pred[:, list(top50_genes)].X)
    observed = to_dense_array(adata_query_stim[:, list(top50_genes)].X)
    mmd2_top50 = compute_mmd2(predicted, observed, gammas)

    return float(r2mean_all), float(r2mean_top50), float(mmd2_top50)


def run_restricted_prediction(
    model: EGD_model,
    original_model_adata: ad.AnnData,
    selected_reference_keys: Sequence[object],
    context_key: str,
    query_key: object,
    cond_key: str,
    ctrl_key: object,
    stim_key: object,
    seed: int,
    predict_kwargs: dict,
) -> ad.AnnData:
    """Temporarily restrict model.adata, run predict_new, then restore it."""
    restricted_adata = make_restricted_model_adata(
        original_model_adata=original_model_adata,
        context_key=context_key,
        query_key=query_key,
        selected_reference_keys=selected_reference_keys,
        cond_key=cond_key,
        ctrl_key=ctrl_key,
        stim_key=stim_key,
    )

    model.adata = restricted_adata
    model.module.eval()
    set_seed(seed)

    try:
        adata_query_pred, _ = model.predict_new(**predict_kwargs)
    finally:
        model.adata = original_model_adata

    return adata_query_pred
