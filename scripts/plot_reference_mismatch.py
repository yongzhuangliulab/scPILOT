from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_DIR = Path("../DataFrames/reference_mismatch_summary")
OUTPUT_DIR = Path("../Figures/reference_mismatch_summary")

QUERY_LEVEL_PATH = INPUT_DIR / "reference_mismatch_query_level.csv"
PAIRED_PATH = INPUT_DIR / "reference_mismatch_nearest_vs_farthest.csv"

METRICS = [
    ("r2mean_all", r"$R^2_{\mathrm{mean,all}}$", "↑"),
    ("r2mean_top50", r"$R^2_{\mathrm{mean,top50}}$", "↑"),
    ("mmd2_top50", r"$\mathrm{MMD}^2$", "↓"),
]

BENCHMARK_TITLES = {
    "across_cell_types": "Across cell types",
    "across_patients": "Across patients",
}


def save_figure(fig: plt.Figure, path_without_suffix: Path) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path_without_suffix.with_suffix(".jpg"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        path_without_suffix.with_suffix(".pdf"),
        bbox_inches="tight",
    )


def validate_inputs(query_level: pd.DataFrame, paired: pd.DataFrame) -> None:
    required_query_columns = {
        "benchmark",
        "query_key",
        "condition",
        "mean_reference_mismatch",
        *[metric for metric, _, _ in METRICS],
    }
    missing_query = required_query_columns.difference(query_level.columns)
    if missing_query:
        raise ValueError(
            "reference_mismatch_query_level.csv is missing columns: "
            + ", ".join(sorted(missing_query))
        )

    required_paired_columns = {
        "benchmark",
        "query_key",
        "nearest_mismatch",
        "farthest_mismatch",
    }
    for metric, _, _ in METRICS:
        required_paired_columns.update(
            {
                f"nearest_{metric}",
                f"farthest_{metric}",
            }
        )

    missing_paired = required_paired_columns.difference(paired.columns)
    if missing_paired:
        raise ValueError(
            "reference_mismatch_nearest_vs_farthest.csv is missing columns: "
            + ", ".join(sorted(missing_paired))
        )


def get_condition_names(benchmark_df: pd.DataFrame) -> tuple[str, str, str]:
    conditions = benchmark_df["condition"].astype(str).unique().tolist()

    all_names = [name for name in conditions if name == "all"]
    nearest_names = [name for name in conditions if name.startswith("nearest_")]
    farthest_names = [name for name in conditions if name.startswith("farthest_")]

    if len(all_names) != 1 or len(nearest_names) != 1 or len(farthest_names) != 1:
        raise ValueError(
            "Expected exactly one all, nearest_k, and farthest_k condition; "
            f"found {conditions}."
        )

    return all_names[0], nearest_names[0], farthest_names[0]


def pretty_condition_name(condition: str) -> str:
    if condition == "all":
        return "All "r"$ρ_{\mathrm{U}}$"
    if condition.startswith("nearest_"):
        return "Nearest-" + condition.split("_", maxsplit=1)[1]
    if condition.startswith("farthest_"):
        return "Farthest-" + condition.split("_", maxsplit=1)[1]
    return condition


def plot_condition_comparison(
    query_level: pd.DataFrame,
    benchmark: str,
) -> None:
    subset = query_level[query_level["benchmark"] == benchmark].copy()
    if subset.empty:
        raise ValueError(f"No query-level rows found for {benchmark}.")

    all_name, nearest_name, farthest_name = get_condition_names(subset)
    condition_order = [all_name, nearest_name, farthest_name]
    x_values = np.arange(len(condition_order))

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))

    for ax, (metric, metric_label, direction) in zip(axes, METRICS):
        pivot = subset.pivot(
            index="query_key",
            columns="condition",
            values=metric,
        )
        pivot = pivot[condition_order].sort_index()

        for row_number, (_, row) in enumerate(pivot.iterrows()):
            ax.plot(
                x_values,
                row.to_numpy(dtype=float),
                marker="o",
                linewidth=1.0,
                alpha=0.55,
                label="Held-out query" if row_number == 0 else None,
            )

        means = pivot.mean(axis=0).to_numpy(dtype=float)
        standard_deviations = pivot.std(axis=0, ddof=1).to_numpy(dtype=float)

        ax.errorbar(
            x_values,
            means,
            yerr=standard_deviations,
            marker="D",
            linewidth=2.5,
            capsize=4,
            label="Mean ± SD",
        )

        ax.set_xticks(x_values)
        ax.set_xticklabels(
            [pretty_condition_name(name) for name in condition_order],
            rotation=15,
            ha="right",
        )
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} ({direction})")
        ax.grid(True, axis="y", linewidth=0.8, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.suptitle(
        f"{BENCHMARK_TITLES.get(benchmark, benchmark)}: "r"$ρ_{\mathrm{U}}$""–mismatch stress test",
        y=1.10,
    )
    fig.tight_layout()

    save_figure(
        fig,
        OUTPUT_DIR / f"{benchmark}_reference_condition_comparison",
    )
    plt.close(fig)


def plot_nearest_vs_farthest(
    paired: pd.DataFrame,
    benchmark: str,
) -> None:
    subset = paired[paired["benchmark"] == benchmark].copy()
    if subset.empty:
        raise ValueError(f"No paired rows found for {benchmark}.")

    subset = subset.sort_values("query_key")
    x_values = np.array([0, 1], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8))

    for ax, (metric, metric_label, direction) in zip(axes, METRICS):
        nearest_column = f"nearest_{metric}"
        farthest_column = f"farthest_{metric}"

        for row_number, (_, row) in enumerate(subset.iterrows()):
            values = np.array(
                [row[nearest_column], row[farthest_column]],
                dtype=float,
            )
            ax.plot(
                x_values,
                values,
                marker="o",
                linewidth=1.2,
                alpha=0.65,
                label="Held-out query" if row_number == 0 else None,
            )

        nearest_values = subset[nearest_column].to_numpy(dtype=float)
        farthest_values = subset[farthest_column].to_numpy(dtype=float)
        means = np.array([nearest_values.mean(), farthest_values.mean()])
        standard_deviations = np.array(
            [
                nearest_values.std(ddof=1),
                farthest_values.std(ddof=1),
            ]
        )

        ax.errorbar(
            x_values,
            means,
            yerr=standard_deviations,
            marker="D",
            linewidth=2.5,
            capsize=4,
            label="Mean ± SD",
        )

        ax.set_xticks(x_values)
        ax.set_xticklabels(["Nearest "r"$ρ_{\mathrm{U}}$", "Farthest "r"$ρ_{\mathrm{U}}$"])
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} ({direction})")
        ax.grid(True, axis="y", linewidth=0.8, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.suptitle(
        f"{BENCHMARK_TITLES.get(benchmark, benchmark)}: paired mismatch comparison",
        y=1.10,
    )
    fig.tight_layout()

    save_figure(
        fig,
        OUTPUT_DIR / f"{benchmark}_nearest_vs_farthest_paired",
    )
    plt.close(fig)


def plot_mismatch_performance_relationship(
    paired: pd.DataFrame,
    benchmark: str,
) -> None:
    subset = paired[paired["benchmark"] == benchmark].copy()
    if subset.empty:
        raise ValueError(f"No paired rows found for {benchmark}.")

    subset = subset.sort_values("query_key")

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))

    for ax, (metric, metric_label, direction) in zip(axes, METRICS):
        nearest_column = f"nearest_{metric}"
        farthest_column = f"farthest_{metric}"

        for _, row in subset.iterrows():
            mismatch_values = np.array(
                [row["nearest_mismatch"], row["farthest_mismatch"]],
                dtype=float,
            )
            performance_values = np.array(
                [row[nearest_column], row[farthest_column]],
                dtype=float,
            )

            ax.plot(
                mismatch_values,
                performance_values,
                marker="o",
                linewidth=1.1,
                alpha=0.65,
            )

            ax.annotate(
                str(row["query_key"]),
                (mismatch_values[1], performance_values[1]),
                xytext=(4, 2),
                textcoords="offset points",
                fontsize=8,
            )

        ax.set_xlabel("Mean "r"$Q$""–"r"$ρ_{\mathrm{U}}$"" mismatch score")
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} ({direction})")
        ax.grid(True, linewidth=0.8, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"{BENCHMARK_TITLES.get(benchmark, benchmark)}: mismatch–performance relationship",
        y=1.04,
    )
    fig.tight_layout()

    save_figure(
        fig,
        OUTPUT_DIR / f"{benchmark}_mismatch_performance_relationship",
    )
    plt.close(fig)


def main() -> None:
    if not QUERY_LEVEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {QUERY_LEVEL_PATH}. Run summarize_reference_mismatch.py first."
        )
    if not PAIRED_PATH.exists():
        raise FileNotFoundError(
            f"Missing {PAIRED_PATH}. Run summarize_reference_mismatch.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    query_level = pd.read_csv(QUERY_LEVEL_PATH)
    paired = pd.read_csv(PAIRED_PATH)
    validate_inputs(query_level, paired)

    benchmarks = sorted(query_level["benchmark"].astype(str).unique())
    for benchmark in benchmarks:
        plot_condition_comparison(query_level, benchmark)
        plot_nearest_vs_farthest(paired, benchmark)
        plot_mismatch_performance_relationship(paired, benchmark)

    print(f"Figures saved to: {OUTPUT_DIR}", flush=True)
    print("Done", flush=True)


if __name__ == "__main__":
    main()
