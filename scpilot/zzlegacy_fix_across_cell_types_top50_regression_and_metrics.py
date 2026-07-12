# import os
# import argparse

# import scanpy as sc
# import anndata as ad
# import numpy as np
# import pandas as pd
# from sklearn.metrics import pairwise
# from matplotlib import pyplot as plt

# from scpilot.egd_model import EGD_model


# parser = argparse.ArgumentParser(
#     description="Regenerate top50 regression figures and per-run metrics CSVs from saved h5ad files."
# )

# parser.add_argument(
#     "--query_key",
#     type=str,
#     default="all",
#     help="Cell type to fix. Use all to process all cell types.",
# )

# parser.add_argument(
#     "--seed",
#     type=str,
#     default="all",
#     help="Seed to fix. Use all to process 1327, 1337, 1347.",
# )

# parser.add_argument(
#     "--model_name",
#     type=str,
#     default="all",
#     help="Model to fix. Use all, or one of VAEGAN, scPILOT, biolord, identity, scGen, CellOT.",
# )

# parser.add_argument(
#     "--split_seed",
#     type=int,
#     default=0,
#     help="Recorded split seed for metrics CSV.",
# )

# parser.add_argument(
#     "--skip_missing",
#     action="store_true",
#     help="Skip missing h5ad files instead of raising an error.",
# )

# args = parser.parse_args()


# def ensure_dirs(*paths):
#     for path in paths:
#         os.makedirs(path, exist_ok=True)


# def to_dense_array(x):
#     return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


# def mmd_distance(x, y, gamma):
#     xx = pairwise.rbf_kernel(x, x, gamma)
#     xy = pairwise.rbf_kernel(x, y, gamma)
#     yy = pairwise.rbf_kernel(y, y, gamma)
#     return xx.mean() + yy.mean() - 2 * xy.mean()


# def compute_mmd_loss(lhs, rhs, gammas):
#     return np.mean([mmd_distance(lhs, rhs, g) for g in gammas])


# def get_top50_genes(
#     adata,
#     query_key,
#     cond_key="condition",
#     stim_key="stimulated",
#     cell_label_key="cell_type",
# ):
#     adata_query = adata[adata.obs[cell_label_key] == query_key].copy()

#     sc.tl.rank_genes_groups(
#         adata_query,
#         groupby=cond_key,
#         method="wilcoxon",
#         n_genes=50,
#     )

#     top50_genes = adata_query.uns["rank_genes_groups"]["names"][stim_key].tolist()
#     return top50_genes


# def prediction_h5ad_path(
#     experiment_name,
#     model_name,
#     data_file,
#     query_key,
#     seed,
#     file_type,
# ):
#     return (
#         f"../Result_anndata/{experiment_name}/"
#         f"{experiment_name}_{model_name}_{data_file}_{query_key}_seed{seed}{file_type}"
#     )


# def regenerate_regression_and_metrics_for_model(
#     experiment_name,
#     data_file,
#     file_type,
#     cond_key,
#     stim_key,
#     query_key,
#     seed,
#     split_seed,
#     model_name,
#     top50_genes,
#     gammas,
#     skip_missing=False,
# ):
#     h5ad_path = prediction_h5ad_path(
#         experiment_name=experiment_name,
#         model_name=model_name,
#         data_file=data_file,
#         query_key=query_key,
#         seed=seed,
#         file_type=file_type,
#     )

#     if not os.path.exists(h5ad_path):
#         msg = f"Missing prediction h5ad: {h5ad_path}"
#         if skip_missing:
#             print(f"[SKIP] {msg}", flush=True)
#             return None
#         raise FileNotFoundError(msg)

#     adata_eval = ad.read_h5ad(h5ad_path)

#     adata_query_stim = adata_eval[adata_eval.obs[cond_key] == stim_key].copy()
#     adata_query_pred = adata_eval[adata_eval.obs[cond_key] == "pred"].copy()

#     if adata_query_stim.n_obs == 0:
#         raise ValueError(f"No stimulated cells found in {h5ad_path}")
#     if adata_query_pred.n_obs == 0:
#         raise ValueError(f"No predicted cells found in {h5ad_path}")

#     adata_query_eval = ad.concat([adata_query_stim, adata_query_pred])
#     adata_query_eval.obs_names_make_unique()

#     fig_path = (
#         f"../Figures/{experiment_name}/"
#         f"{model_name}_{data_file}_reg_mean_{query_key}_seed{seed}.jpg"
#     )

#     plt.figure()
#     r2mean_all, r2mean_top50 = EGD_model.reg_mean_plot(
#         adata_query_eval,
#         cond_key=cond_key,
#         axis_keys={"x": "pred", "y": stim_key},
#         labels={"x": "Prediction", "y": "Ground truth"},
#         path_to_save=fig_path,
#         gene_list=top50_genes[:10],
#         show=False,
#         top_genes=top50_genes,
#         top_gene_label="T50",
#         legend=False,
#     )
#     plt.close("all")

#     x = to_dense_array(adata_query_pred[:, top50_genes].X)
#     y = to_dense_array(adata_query_stim[:, top50_genes].X)
#     mmd_top50 = compute_mmd_loss(x, y, gammas=gammas)

#     print(f"{model_name}:")
#     print(f"{query_key}, seed={seed}: r2mean_all = {r2mean_all}")
#     print(f"{query_key}, seed={seed}: r2mean_top50 = {r2mean_top50}")
#     print(f"{query_key}, seed={seed}: mmd_top50 = {mmd_top50}")

#     return {
#         "experiment": experiment_name,
#         "data_file": data_file,
#         "query_key": query_key,
#         "seed": seed,
#         "split_seed": split_seed,
#         "model": model_name,
#         "r2mean_all": r2mean_all,
#         "r2mean_top50": r2mean_top50,
#         "mmd_top50": mmd_top50,
#     }


# def write_metrics_csvs(
#     experiment_name,
#     data_file,
#     query_key,
#     seed,
#     records,
# ):
#     records = [r for r in records if r is not None]
#     if len(records) == 0:
#         print(
#             f"[SKIP] No metrics records to write for query_key={query_key}, seed={seed}",
#             flush=True,
#         )
#         return

#     records_by_model = {record["model"]: record for record in records}

#     # 1. Keep the original scPILOT/VAEGAN CSV format:
#     #    one CSV containing two rows: VAEGAN and scPILOT.
#     main_models = ["VAEGAN", "scPILOT"]
#     main_records = [
#         records_by_model[m]
#         for m in main_models
#         if m in records_by_model
#     ]

#     if len(main_records) > 0:
#         main_metrics_path = (
#             f"../DataFrames/{experiment_name}/"
#             f"{experiment_name}_{data_file}_{query_key}_seed{seed}_metrics.csv"
#         )
#         pd.DataFrame(main_records).to_csv(main_metrics_path, index=False)
#         print(f"Saved corrected metrics: {main_metrics_path}", flush=True)

#     # 2. Keep the original one-model-one-CSV format for other baselines.
#     for model_name in ["biolord", "identity", "scGen", "CellOT"]:
#         if model_name not in records_by_model:
#             continue

#         model_metrics_path = (
#             f"../DataFrames/{experiment_name}/"
#             f"{experiment_name}_{data_file}_{query_key}_seed{seed}_{model_name}_metrics.csv"
#         )

#         pd.DataFrame([records_by_model[model_name]]).to_csv(
#             model_metrics_path,
#             index=False,
#         )
#         print(f"Saved corrected metrics: {model_metrics_path}", flush=True)


# def main():
#     experiment_name = "across_cell_types"
#     data_file = "pbmc"
#     file_type = ".h5ad"

#     cond_key = "condition"
#     ctrl_key = "control"
#     stim_key = "stimulated"
#     cell_label_key = "cell_type"

#     all_models = ["VAEGAN", "scPILOT", "biolord", "identity", "scGen", "CellOT"]
#     all_seeds = [1327, 1337, 1347]

#     ensure_dirs(
#         f"../Figures/{experiment_name}",
#         f"../DataFrames/{experiment_name}",
#     )

#     adata = sc.read_h5ad(f"../Data/{experiment_name}/{data_file}{file_type}")
#     adata = adata[
#         (adata.obs[cond_key] == ctrl_key)
#         | (adata.obs[cond_key] == stim_key)
#     ].copy()

#     if args.query_key == "all":
#         query_keys = sorted(adata.obs[cell_label_key].unique().tolist())
#     else:
#         query_keys = [args.query_key]

#     if args.seed == "all":
#         seeds = all_seeds
#     else:
#         seeds = [int(args.seed)]

#     if args.model_name == "all":
#         model_names = all_models
#     else:
#         if args.model_name not in all_models:
#             raise ValueError(
#                 f"Unknown model_name: {args.model_name}. "
#                 f"Expected one of: {all_models}"
#             )
#         model_names = [args.model_name]

#     gammas = np.logspace(1, -3, num=50)

#     for query_key in query_keys:
#         top50_genes = get_top50_genes(
#             adata=adata,
#             query_key=query_key,
#             cond_key=cond_key,
#             stim_key=stim_key,
#             cell_label_key=cell_label_key,
#         )

#         print(
#             f"Top50 genes for {query_key}: {top50_genes[:10]} ...",
#             flush=True,
#         )

#         for seed in seeds:
#             print(
#                 f"======Regenerating outputs | query_key={query_key} | seed={seed}======",
#                 flush=True,
#             )

#             records = []

#             for model_name in model_names:
#                 record = regenerate_regression_and_metrics_for_model(
#                     experiment_name=experiment_name,
#                     data_file=data_file,
#                     file_type=file_type,
#                     cond_key=cond_key,
#                     stim_key=stim_key,
#                     query_key=query_key,
#                     seed=seed,
#                     split_seed=args.split_seed,
#                     model_name=model_name,
#                     top50_genes=top50_genes,
#                     gammas=gammas,
#                     skip_missing=args.skip_missing,
#                 )
#                 records.append(record)

#             write_metrics_csvs(
#                 experiment_name=experiment_name,
#                 data_file=data_file,
#                 query_key=query_key,
#                 seed=seed,
#                 records=records,
#             )

#     print("Done", flush=True)


# if __name__ == "__main__":
#     main()