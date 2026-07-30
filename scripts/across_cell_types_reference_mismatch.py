import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

from scpilot.egd_model import EGD_model
from reference_mismatch_utils import (
    choose_reference_sets,
    ensure_dirs,
    evaluate_prediction,
    obs_value_mask,
    rank_cell_type_references,
    run_restricted_prediction,
    set_seed,
)


parser = argparse.ArgumentParser(
    description="Across-cell-type query-reference mismatch stress test."
)
parser.add_argument("--query_key", type=str, required=True)
parser.add_argument(
    "--seed",
    type=int,
    required=True,
    choices=(1327, 1337, 1347),
)
parser.add_argument("--split_seed", type=int, default=0)
parser.add_argument("--k", type=int, default=3)
parser.add_argument("--max_cells_per_group", type=int, default=300)
args = parser.parse_args()


def main() -> None:
    experiment_name = "across_cell_types"
    stress_name = "across_cell_types_reference_mismatch"
    data_file = "pbmc"
    file_type = ".h5ad"
    cond_key = "condition"
    ctrl_key = "control"
    stim_key = "stimulated"
    cell_label_key = "cell_type"

    output_dir = Path(f"../DataFrames/{stress_name}")
    ensure_dirs(output_dir)

    adata = sc.read_h5ad(
        f"../Data/{experiment_name}/{data_file}{file_type}"
    )
    adata.obs_names_make_unique()
    adata = adata[
        obs_value_mask(adata, cond_key, ctrl_key)
        | obs_value_mask(adata, cond_key, stim_key)
    ].copy()

    model_path = (
        f"../model_trained/{experiment_name}/"
        f"EGD_model_trained_on_{data_file}_{args.query_key}_"
        f"seed{args.seed}.model"
    )
    model = EGD_model.load(model_path)
    model.module.eval()
    original_model_adata = model.adata.copy()
    original_model_adata.obs_names_make_unique()

    adata_query_stim = adata[
        obs_value_mask(adata, cell_label_key, args.query_key)
        & obs_value_mask(adata, cond_key, stim_key)
    ].copy()
    adata_query = adata[
        obs_value_mask(adata, cell_label_key, args.query_key)
    ].copy()

    if adata_query_stim.n_obs == 0:
        raise ValueError(f"No stimulated truth found for {args.query_key!r}.")

    sc.tl.rank_genes_groups(
        adata_query,
        groupby=cond_key,
        method="wilcoxon",
        n_genes=50,
    )
    top50_genes = (
        adata_query.uns["rank_genes_groups"]["names"][stim_key]
        .tolist()
    )
    gammas = np.logspace(1, -3, num=50)

    set_seed(args.seed)
    ranking = rank_cell_type_references(
        model=model,
        model_adata=original_model_adata,
        query_key=args.query_key,
        cell_label_key=cell_label_key,
        cond_key=cond_key,
        ctrl_key=ctrl_key,
        max_cells_per_group=args.max_cells_per_group,
        benchmark=experiment_name,
        seed=args.seed,
    )
    reference_sets = choose_reference_sets(ranking, k=args.k)

    nearest_name = f"nearest_{args.k}"
    farthest_name = f"farthest_{args.k}"
    ranking["selected_nearest"] = ranking["reference_key"].isin(
        reference_sets[nearest_name]
    )
    ranking["selected_farthest"] = ranking["reference_key"].isin(
        reference_sets[farthest_name]
    )

    ranking_path = output_dir / (
        f"{stress_name}_{args.query_key}_seed{args.seed}_"
        "reference_scores.csv"
    )
    ranking.to_csv(ranking_path, index=False)

    metric_rows = []
    score_map = dict(
        zip(ranking["reference_key"], ranking["mismatch_score"])
    )

    for condition_name, selected_references in reference_sets.items():
        print(
            f"[{experiment_name}] query={args.query_key} seed={args.seed} "
            f"condition={condition_name} references={selected_references}",
            flush=True,
        )

        adata_query_pred = run_restricted_prediction(
            model=model,
            original_model_adata=original_model_adata,
            selected_reference_keys=selected_references,
            context_key=cell_label_key,
            query_key=args.query_key,
            cond_key=cond_key,
            ctrl_key=ctrl_key,
            stim_key=stim_key,
            seed=args.seed,
            predict_kwargs={
                "cell_label_key": cell_label_key,
                "cond_key": cond_key,
                "ctrl_key": ctrl_key,
                "stim_key": stim_key,
                "query_key": args.query_key,
            },
        )

        r2mean_all, r2mean_top50, mmd2_top50 = evaluate_prediction(
            adata_query_stim=adata_query_stim,
            adata_query_pred=adata_query_pred,
            cond_key=cond_key,
            stim_key=stim_key,
            top50_genes=top50_genes,
            gammas=gammas,
        )

        metric_rows.append(
            {
                "benchmark": experiment_name,
                "query_key": str(args.query_key),
                "seed": int(args.seed),
                "split_seed": int(args.split_seed),
                "condition": condition_name,
                "k": int(args.k),
                "n_references": int(len(selected_references)),
                "selected_references": "|".join(selected_references),
                "mean_reference_mismatch": float(
                    np.mean([score_map[key] for key in selected_references])
                ),
                "min_reference_mismatch": float(
                    np.min([score_map[key] for key in selected_references])
                ),
                "max_reference_mismatch": float(
                    np.max([score_map[key] for key in selected_references])
                ),
                "r2mean_all": r2mean_all,
                "r2mean_top50": r2mean_top50,
                "mmd2_top50": mmd2_top50,
                "max_cells_per_group": int(args.max_cells_per_group),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    metrics_path = output_dir / (
        f"{stress_name}_{args.query_key}_seed{args.seed}_metrics.csv"
    )
    metrics.to_csv(metrics_path, index=False)

    print(f"Saved: {ranking_path}", flush=True)
    print(f"Saved: {metrics_path}", flush=True)
    print("Done", flush=True)


if __name__ == "__main__":
    main()
