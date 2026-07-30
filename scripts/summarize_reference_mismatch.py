from pathlib import Path

import numpy as np
import pandas as pd


BENCHMARK_DIRS = {
    "across_cell_types": Path(
        "../DataFrames/across_cell_types_reference_mismatch"
    ),
    "across_patients": Path(
        "../DataFrames/across_patients_reference_mismatch"
    ),
}
OUTPUT_DIR = Path("../DataFrames/reference_mismatch_summary")
EXPECTED_SEEDS = {1327, 1337, 1347}
METRICS = ("r2mean_all", "r2mean_top50", "mmd2_top50")


def load_seed_level_metrics() -> pd.DataFrame:
    frames = []
    for benchmark, directory in BENCHMARK_DIRS.items():
        paths = sorted(directory.glob("*_metrics.csv"))
        if not paths:
            raise FileNotFoundError(
                f"No metric CSV files found for {benchmark} in {directory}."
            )
        for path in paths:
            frame = pd.read_csv(path)
            frame["source_file"] = path.name
            frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    duplicate_mask = data.duplicated(
        ["benchmark", "query_key", "seed", "condition"],
        keep=False,
    )
    if duplicate_mask.any():
        raise ValueError(
            "Duplicate benchmark/query/seed/condition rows found:\n"
            + data.loc[duplicate_mask].to_string(index=False)
        )
    return data


def validate_completeness(seed_level: pd.DataFrame) -> None:
    for (benchmark, query_key, condition), group in seed_level.groupby(
        ["benchmark", "query_key", "condition"]
    ):
        seeds = set(group["seed"].astype(int))
        if seeds != EXPECTED_SEEDS:
            raise ValueError(
                f"Incomplete seeds for {benchmark}, query={query_key}, "
                f"condition={condition}: found {sorted(seeds)}"
            )


def make_query_level(seed_level: pd.DataFrame) -> pd.DataFrame:
    aggregation = {
        metric: "mean" for metric in METRICS
    }
    aggregation.update(
        {
            "mean_reference_mismatch": "mean",
            "min_reference_mismatch": "mean",
            "max_reference_mismatch": "mean",
            "n_references": "first",
            "selected_references": lambda values: " || ".join(
                sorted(set(values.astype(str)))
            ),
        }
    )

    query_level = (
        seed_level.groupby(
            ["benchmark", "query_key", "condition"],
            as_index=False,
        )
        .agg(aggregation)
    )
    return query_level


def make_paired_effects(query_level: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (benchmark, query_key), group in query_level.groupby(
        ["benchmark", "query_key"]
    ):
        indexed = group.set_index("condition")
        nearest_names = [name for name in indexed.index if name.startswith("nearest_")]
        farthest_names = [name for name in indexed.index if name.startswith("farthest_")]
        if len(nearest_names) != 1 or len(farthest_names) != 1:
            raise ValueError(
                f"Expected one nearest and one farthest condition for "
                f"{benchmark}, query={query_key}."
            )
        nearest = indexed.loc[nearest_names[0]]
        farthest = indexed.loc[farthest_names[0]]

        row = {
            "benchmark": benchmark,
            "query_key": query_key,
            "nearest_condition": nearest_names[0],
            "farthest_condition": farthest_names[0],
            "nearest_mismatch": nearest["mean_reference_mismatch"],
            "farthest_mismatch": farthest["mean_reference_mismatch"],
            "mismatch_increase": (
                farthest["mean_reference_mismatch"]
                - nearest["mean_reference_mismatch"]
            ),
        }
        for metric in METRICS:
            row[f"nearest_{metric}"] = nearest[metric]
            row[f"farthest_{metric}"] = farthest[metric]
            if metric == "mmd2_top50":
                row[f"deterioration_{metric}"] = (
                    farthest[metric] - nearest[metric]
                )
            else:
                row[f"deterioration_{metric}"] = (
                    nearest[metric] - farthest[metric]
                )
        rows.append(row)

    return pd.DataFrame(rows)


def make_aggregate_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for benchmark, group in paired.groupby("benchmark"):
        row = {
            "benchmark": benchmark,
            "n_queries": int(len(group)),
            "mean_nearest_mismatch": float(group["nearest_mismatch"].mean()),
            "mean_farthest_mismatch": float(group["farthest_mismatch"].mean()),
            "mean_mismatch_increase": float(group["mismatch_increase"].mean()),
        }
        for metric in METRICS:
            deterioration = group[f"deterioration_{metric}"]
            row[f"mean_nearest_{metric}"] = float(
                group[f"nearest_{metric}"].mean()
            )
            row[f"mean_farthest_{metric}"] = float(
                group[f"farthest_{metric}"].mean()
            )
            row[f"mean_deterioration_{metric}"] = float(
                deterioration.mean()
            )
            row[f"median_deterioration_{metric}"] = float(
                deterioration.median()
            )
            row[f"n_queries_deteriorated_{metric}"] = int(
                np.sum(deterioration > 0)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    seed_level = load_seed_level_metrics()
    validate_completeness(seed_level)
    query_level = make_query_level(seed_level)
    paired = make_paired_effects(query_level)
    aggregate = make_aggregate_summary(paired)

    seed_level.to_csv(
        OUTPUT_DIR / "reference_mismatch_seed_level.csv",
        index=False,
    )
    query_level.to_csv(
        OUTPUT_DIR / "reference_mismatch_query_level.csv",
        index=False,
    )
    paired.to_csv(
        OUTPUT_DIR / "reference_mismatch_nearest_vs_farthest.csv",
        index=False,
    )
    aggregate.to_csv(
        OUTPUT_DIR / "reference_mismatch_aggregate_summary.csv",
        index=False,
    )

    print(aggregate.to_string(index=False), flush=True)
    print("Done", flush=True)


if __name__ == "__main__":
    main()
