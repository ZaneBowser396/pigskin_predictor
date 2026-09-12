"""Estimate whether ablation-test improvements are likely to be real.

The script resamples complete NFL weeks rather than individual games. This
preserves the confidence-pool structure because all picks and weights within a
week stay together.

Run model 09 first, then run from the repository root:
    python models/10_weekly_bootstrap.py

Output:
    outputs/bootstrap_comparison.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "ablation_by_week.csv"
OUTPUT = ROOT / "outputs" / "bootstrap_comparison.csv"

BASELINE = "Market baseline"
BOOTSTRAP_SAMPLES = 10_000
RANDOM_SEED = 42


def bootstrap_difference(
    model_values: np.ndarray,
    baseline_values: np.ndarray,
    denominators: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    """Return observed difference, 95% CI and chance model is better."""
    observed = 100 * (
        model_values.sum() / denominators.sum()
        - baseline_values.sum() / denominators.sum()
    )

    sample_count = len(model_values)
    differences = np.empty(BOOTSTRAP_SAMPLES)

    for sample in range(BOOTSTRAP_SAMPLES):
        indices = rng.integers(0, sample_count, sample_count)
        sampled_denominators = denominators[indices].sum()
        differences[sample] = 100 * (
            model_values[indices].sum() / sampled_denominators
            - baseline_values[indices].sum() / sampled_denominators
        )

    low, high = np.percentile(differences, [2.5, 97.5])
    probability_better = np.mean(differences > 0) * 100

    return (
        round(float(observed), 3),
        round(float(low), 3),
        round(float(high), 3),
        round(float(probability_better), 1),
    )


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(
            "outputs/ablation_by_week.csv was not found. "
            "Run models/09_ablation_test.py first."
        )

    weekly = pl.read_csv(INPUT)
    baseline = (
        weekly
        .filter(pl.col("model") == BASELINE)
        .select(
            "season",
            "week",
            pl.col("points").alias("baseline_points"),
            pl.col("correct_picks").alias("baseline_correct"),
        )
    )

    model_names = (
        weekly
        .filter(pl.col("model") != BASELINE)
        .get_column("model")
        .unique()
        .sort()
        .to_list()
    )

    rows: list[dict] = []

    for position, model_name in enumerate(model_names):
        paired = (
            weekly
            .filter(pl.col("model") == model_name)
            .join(baseline, on=["season", "week"], how="inner")
            .sort(["season", "week"])
        )

        # A separate deterministic stream for each model prevents ordering
        # changes from altering another model's bootstrap result.
        rng = np.random.default_rng(RANDOM_SEED + position)

        points = bootstrap_difference(
            paired["points"].to_numpy(),
            paired["baseline_points"].to_numpy(),
            paired["max_points"].to_numpy(),
            rng,
        )
        accuracy = bootstrap_difference(
            paired["correct_picks"].to_numpy(),
            paired["baseline_correct"].to_numpy(),
            paired["games"].to_numpy(),
            rng,
        )

        rows.append({
            "model": model_name,
            "weeks": paired.height,
            "points_difference": points[0],
            "points_ci_low": points[1],
            "points_ci_high": points[2],
            "probability_points_better": points[3],
            "accuracy_difference": accuracy[0],
            "accuracy_ci_low": accuracy[1],
            "accuracy_ci_high": accuracy[2],
            "probability_accuracy_better": accuracy[3],
        })

    comparison = (
        pl.DataFrame(rows)
        .sort("points_difference", descending=True)
        .with_row_index("points_rank", offset=1)
    )

    comparison.write_csv(OUTPUT)

    with pl.Config(tbl_rows=20, tbl_cols=12, tbl_width_chars=180):
        print("\nPAIRED WEEKLY BOOTSTRAP — VERSUS RAW MARKET BASELINE")
        print(comparison)

    print("\nHow to read this:")
    print("- A 95% interval entirely above zero suggests a real improvement.")
    print("- An interval crossing zero means the apparent edge may be noise.")
    print("- Probability better is the share of bootstrap samples above zero.")
    print("\nSaved outputs/bootstrap_comparison.csv")


if __name__ == "__main__":
    main()
