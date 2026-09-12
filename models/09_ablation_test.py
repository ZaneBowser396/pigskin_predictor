"""Walk-forward ablation test for the Pigskin Predictor.

This script tests every non-empty combination of the four features used by
08_walk_forward_model.py. Each test season is predicted using only games from
earlier seasons, so the comparison remains genuinely out of sample.

Run from the repository root:
    python models/09_ablation_test.py

Outputs:
    outputs/ablation_overall.csv
    outputs/ablation_by_season.csv
    outputs/ablation_by_week.csv
"""

from __future__ import annotations

from collections import defaultdict, deque
from itertools import combinations
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ==================================================
# SETTINGS
# ==================================================

STARTING_ELO = 1500
K_FACTOR = 20
HOME_FIELD_ADVANTAGE = 55

START_SEASON = 2015
TRAIN_START = 2018
TEST_START = 2021
TEST_END = 2025
RECENT_GAMES = 5

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"


# Keep this order so the result names are easy to compare.
FEATURES = {
    "Market": "market_home_probability",
    "ELO": "elo_home_probability",
    "Recent form": "recent_margin_difference",
    "Rest": "rest_difference",
}


# ==================================================
# HELPERS
# ==================================================

def elo_probability(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def moneyline_probability(moneyline: float) -> float:
    if moneyline < 0:
        return -moneyline / (-moneyline + 100)
    return 100 / (moneyline + 100)


def average(values: deque[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def all_feature_sets() -> dict[str, list[str]]:
    """Return all 15 non-empty combinations of the four model features."""
    feature_items = list(FEATURES.items())
    tests: dict[str, list[str]] = {}

    for size in range(1, len(feature_items) + 1):
        for group in combinations(feature_items, size):
            label = " + ".join(item[0] for item in group)
            tests[label] = [item[1] for item in group]

    return tests


def add_pool_scoring(
    frame: pl.DataFrame,
    probability_column: str,
) -> pl.DataFrame:
    """Turn home-win probabilities into picks and confidence-pool points."""
    return (
        frame
        .with_columns(
            pl.when(pl.col(probability_column) >= 0.5)
            .then(pl.col("home_team"))
            .otherwise(pl.col("away_team"))
            .alias("pick"),

            pl.max_horizontal(
                pl.col(probability_column),
                1 - pl.col(probability_column),
            ).alias("confidence"),
        )
        .with_columns(
            (pl.col("pick") == pl.col("winner")).alias("correct"),

            pl.col("confidence")
            .rank(method="ordinal")
            .over(["season", "week"])
            .cast(pl.Int32)
            .alias("weight"),
        )
        .with_columns(
            pl.when(pl.col("correct"))
            .then(pl.col("weight"))
            .otherwise(0)
            .alias("points")
        )
    )


def metrics(
    frame: pl.DataFrame,
    model_name: str,
    probability_column: str,
    season: int | str,
) -> dict[str, int | float | str]:
    """Calculate the metrics needed to compare picks and probabilities."""
    probabilities = np.clip(
        frame[probability_column].to_numpy(),
        1e-15,
        1 - 1e-15,
    )
    outcomes = frame["home_win"].to_numpy()

    brier_score = np.mean((probabilities - outcomes) ** 2)
    log_loss = -np.mean(
        outcomes * np.log(probabilities)
        + (1 - outcomes) * np.log(1 - probabilities)
    )

    return {
        "model": model_name,
        "season": season,
        "games": frame.height,
        "accuracy": round(float(frame["correct"].mean() * 100), 2),
        "confidence_points": round(
            float(frame["points"].sum() / frame["weight"].sum() * 100),
            2,
        ),
        "brier_score": round(float(brier_score), 5),
        "log_loss": round(float(log_loss), 5),
    }


def weekly_metrics(
    frame: pl.DataFrame,
    model_name: str,
    probability_column: str,
) -> pl.DataFrame:
    """Keep paired weekly results for the uncertainty test in model 10."""
    probability = pl.col(probability_column).clip(1e-15, 1 - 1e-15)

    return (
        frame
        .with_columns(
            (
                (pl.col(probability_column) - pl.col("home_win")) ** 2
            ).alias("brier_value"),
            (
                -(
                    pl.col("home_win") * probability.log()
                    + (1 - pl.col("home_win"))
                    * (1 - probability).log()
                )
            ).alias("log_loss_value"),
        )
        .group_by(["season", "week"])
        .agg(
            pl.len().alias("games"),
            pl.col("correct").sum().alias("correct_picks"),
            pl.col("points").sum().alias("points"),
            pl.col("weight").sum().alias("max_points"),
            pl.col("brier_value").mean().alias("brier_score"),
            pl.col("log_loss_value").mean().alias("log_loss"),
        )
        .with_columns(
            pl.lit(model_name).alias("model"),
            (pl.col("correct_picks") / pl.col("games") * 100)
            .round(2)
            .alias("accuracy"),
            (pl.col("points") / pl.col("max_points") * 100)
            .round(2)
            .alias("confidence_points"),
            pl.col("brier_score").round(5),
            pl.col("log_loss").round(5),
        )
        .select(
            "model",
            "season",
            "week",
            "games",
            "correct_picks",
            "points",
            "max_points",
            "accuracy",
            "confidence_points",
            "brier_score",
            "log_loss",
        )
        .sort(["season", "week"])
    )


# ==================================================
# BUILD PRE-GAME FEATURES
# ==================================================

def build_feature_data() -> pl.DataFrame:
    games = nfl.load_schedules(
        list(range(START_SEASON, TEST_END + 1))
    )

    games = (
        games
        .filter(
            (pl.col("game_type") == "REG")
            & pl.col("home_score").is_not_null()
            & pl.col("away_score").is_not_null()
            & pl.col("home_moneyline").is_not_null()
            & pl.col("away_moneyline").is_not_null()
        )
        .sort(["season", "week", "gameday", "gametime"])
    )

    ratings: dict[str, float] = {}
    recent_margins: defaultdict[str, deque[float]] = defaultdict(
        lambda: deque(maxlen=RECENT_GAMES)
    )
    rows: list[dict] = []
    previous_season: int | None = None

    for game in games.iter_rows(named=True):
        season = game["season"]
        home_team = game["home_team"]
        away_team = game["away_team"]
        home_score = game["home_score"]
        away_score = game["away_score"]

        if previous_season is not None and season != previous_season:
            for team in ratings:
                ratings[team] = (
                    STARTING_ELO
                    + 0.75 * (ratings[team] - STARTING_ELO)
                )
            recent_margins.clear()

        previous_season = season
        ratings.setdefault(home_team, STARTING_ELO)
        ratings.setdefault(away_team, STARTING_ELO)

        home_raw = moneyline_probability(game["home_moneyline"])
        away_raw = moneyline_probability(game["away_moneyline"])
        market_home_probability = home_raw / (home_raw + away_raw)

        elo_home_probability = elo_probability(
            ratings[home_team] + HOME_FIELD_ADVANTAGE,
            ratings[away_team],
        )

        recent_margin_difference = (
            average(recent_margins[home_team])
            - average(recent_margins[away_team])
        )

        home_rest = game["home_rest"] if game["home_rest"] is not None else 7
        away_rest = game["away_rest"] if game["away_rest"] is not None else 7
        rest_difference = home_rest - away_rest

        if home_score > away_score:
            home_win = 1
            actual_home_result = 1.0
            winner = home_team
        elif away_score > home_score:
            home_win = 0
            actual_home_result = 0.0
            winner = away_team
        else:
            home_win = None
            actual_home_result = 0.5
            winner = None

        # Store features before updating either team's ratings or recent form.
        if season >= TRAIN_START:
            rows.append({
                "season": season,
                "week": game["week"],
                "home_team": home_team,
                "away_team": away_team,
                "winner": winner,
                "home_win": home_win,
                "market_home_probability": market_home_probability,
                "elo_home_probability": elo_home_probability,
                "recent_margin_difference": recent_margin_difference,
                "rest_difference": rest_difference,
            })

        elo_change = K_FACTOR * (
            actual_home_result - elo_home_probability
        )
        ratings[home_team] += elo_change
        ratings[away_team] -= elo_change

        home_margin = home_score - away_score
        recent_margins[home_team].append(home_margin)
        recent_margins[away_team].append(-home_margin)

    return (
        pl.DataFrame(rows)
        .filter(pl.col("home_win").is_not_null())
    )


# ==================================================
# WALK-FORWARD ABLATION
# ==================================================

def run_ablation(
    data: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    overall_rows: list[dict] = []
    season_rows: list[dict] = []
    weekly_frames: list[pl.DataFrame] = []

    test_data = data.filter(
        pl.col("season").is_between(TEST_START, TEST_END)
    )

    # Unchanged market probabilities are the benchmark every model must beat.
    market_scored = add_pool_scoring(
        test_data,
        "market_home_probability",
    )
    overall_rows.append(
        metrics(
            market_scored,
            "Market baseline",
            "market_home_probability",
            "Overall",
        )
    )
    weekly_frames.append(
        weekly_metrics(
            market_scored,
            "Market baseline",
            "market_home_probability",
        )
    )

    for season in range(TEST_START, TEST_END + 1):
        season_market = market_scored.filter(pl.col("season") == season)
        season_rows.append(
            metrics(
                season_market,
                "Market baseline",
                "market_home_probability",
                season,
            )
        )

    for model_name, feature_columns in all_feature_sets().items():
        predictions: list[pl.DataFrame] = []

        for test_season in range(TEST_START, TEST_END + 1):
            train = data.filter(
                (pl.col("season") >= TRAIN_START)
                & (pl.col("season") < test_season)
            )
            test = data.filter(pl.col("season") == test_season)

            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, random_state=42),
            )
            model.fit(
                train.select(feature_columns).to_numpy(),
                train["home_win"].to_numpy(),
            )

            probability_column = "model_home_probability"
            probabilities = model.predict_proba(
                test.select(feature_columns).to_numpy()
            )[:, 1]

            scored = add_pool_scoring(
                test.with_columns(
                    pl.Series(probability_column, probabilities)
                ),
                probability_column,
            )
            predictions.append(scored)
            weekly_frames.append(
                weekly_metrics(
                    scored,
                    model_name,
                    probability_column,
                )
            )
            season_rows.append(
                metrics(
                    scored,
                    model_name,
                    probability_column,
                    test_season,
                )
            )

        combined = pl.concat(predictions)
        overall_rows.append(
            metrics(
                combined,
                model_name,
                "model_home_probability",
                "Overall",
            )
        )

    overall = (
        pl.DataFrame(overall_rows)
        .sort(
            ["confidence_points", "accuracy"],
            descending=[True, True],
        )
        .with_row_index("points_rank", offset=1)
    )
    by_season = pl.DataFrame(season_rows).sort(["season", "model"])
    by_week = pl.concat(weekly_frames).sort(["season", "week", "model"])
    return overall, by_season, by_week


def main() -> None:
    print("Building pre-game features...")
    data = build_feature_data()

    print("Running 15 feature combinations with walk-forward validation...")
    overall, by_season, by_week = run_ablation(data)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    overall.write_csv(OUTPUT_DIR / "ablation_overall.csv")
    by_season.write_csv(OUTPUT_DIR / "ablation_by_season.csv")
    by_week.write_csv(OUTPUT_DIR / "ablation_by_week.csv")

    print("\nABLATION RESULTS — RANKED BY CONFIDENCE POINTS")
    print(overall)
    print("\nSaved outputs/ablation_overall.csv")
    print("Saved outputs/ablation_by_season.csv")
    print("Saved outputs/ablation_by_week.csv")


if __name__ == "__main__":
    main()
