"""Generate a confidence-pool card for one upcoming NFL week.

The raw, no-vig betting market is the official baseline because no challenger
in the walk-forward bootstrap test showed a statistically reliable improvement.

Example:
    python models/11_generate_weekly_picks.py --season 2026 --week 1 --first-game TB

Output:
    outputs/weekly_picks.csv
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import nflreadpy as nfl
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "weekly_picks.csv"


def default_season() -> int:
    """Use the previous calendar year during January and February."""
    today = date.today()
    return today.year if today.month >= 3 else today.year - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create market-based NFL confidence-pool picks."
    )
    parser.add_argument(
        "--season",
        type=int,
        default=default_season(),
        help="NFL season year (default: current NFL season).",
    )
    parser.add_argument(
        "--week",
        type=int,
        required=True,
        help="Regular-season week to generate.",
    )
    parser.add_argument(
        "--first-game",
        type=str.upper,
        default=None,
        metavar="TEAM",
        help=(
            "Only include this team's matchup and games kicking off later "
            "(for example: --first-game TB)."
        ),
    )
    return parser.parse_args()


def moneyline_probability(column: str) -> pl.Expr:
    moneyline = pl.col(column)
    return (
        pl.when(moneyline < 0)
        .then(-moneyline / (-moneyline + 100))
        .otherwise(100 / (moneyline + 100))
    )


def generate_picks(
    season: int,
    week: int,
    first_game_team: str | None = None,
) -> pl.DataFrame:
    schedule = nfl.load_schedules([season])
    games = (
        schedule
        .filter(
            (pl.col("game_type") == "REG")
            & (pl.col("week") == week)
        )
        .sort(["gameday", "gametime", "game_id"])
    )

    if games.is_empty():
        raise ValueError(
            f"No regular-season games were found for {season} Week {week}."
        )

    if first_game_team is not None:
        first_game = games.filter(
            (pl.col("home_team") == first_game_team)
            | (pl.col("away_team") == first_game_team)
        )

        if first_game.is_empty():
            raise ValueError(
                f"{first_game_team} was not found in {season} Week {week}. "
                "Use the nflverse team abbreviation, such as TB."
            )

        first_gameday = first_game["gameday"][0]
        first_gametime = first_game["gametime"][0]
        games = games.filter(
            (pl.col("gameday") > first_gameday)
            | (
                (pl.col("gameday") == first_gameday)
                & (pl.col("gametime") >= first_gametime)
            )
        )

    missing_odds = games.filter(
        pl.col("home_moneyline").is_null()
        | pl.col("away_moneyline").is_null()
    )

    if not missing_odds.is_empty():
        matchups = ", ".join(
            f"{row['away_team']} at {row['home_team']}"
            for row in missing_odds.iter_rows(named=True)
        )
        raise ValueError(
            "Moneylines are not available for every game yet. "
            f"Missing: {matchups}. Run the script again when odds are posted."
        )

    games = games.with_columns(
        moneyline_probability("home_moneyline").alias("home_raw_probability"),
        moneyline_probability("away_moneyline").alias("away_raw_probability"),
    )

    games = games.with_columns(
        (
            pl.col("home_raw_probability")
            / (
                pl.col("home_raw_probability")
                + pl.col("away_raw_probability")
            )
        ).alias("home_probability"),
        (
            pl.col("away_raw_probability")
            / (
                pl.col("home_raw_probability")
                + pl.col("away_raw_probability")
            )
        ).alias("away_probability"),
    )

    games = games.with_columns(
        pl.when(pl.col("home_probability") >= pl.col("away_probability"))
        .then(pl.col("home_team"))
        .otherwise(pl.col("away_team"))
        .alias("pick"),
        pl.when(pl.col("home_probability") >= pl.col("away_probability"))
        .then(pl.col("away_team"))
        .otherwise(pl.col("home_team"))
        .alias("opponent"),
        pl.max_horizontal("home_probability", "away_probability")
        .alias("pick_probability"),
        pl.when(pl.col("home_probability") >= pl.col("away_probability"))
        .then(pl.col("home_moneyline"))
        .otherwise(pl.col("away_moneyline"))
        .alias("pick_odds"),
    )

    games = games.with_columns(
        pl.when(pl.col("pick_odds") < 0)
        .then(1 + 100 / -pl.col("pick_odds"))
        .otherwise(1 + pl.col("pick_odds") / 100)
        .round(3)
        .alias("decimal_price")
    )

    # Lowest win probability receives weight 1; highest receives weight N.
    return (
        games
        .with_columns(
            pl.col("pick_probability")
            .rank(method="ordinal")
            .cast(pl.Int32)
            .alias("rank")
        )
        .select(
            pl.lit(season).alias("season"),
            pl.lit(week).alias("week_number"),
            pl.lit(f"{season} Week {week}").alias("week"),
            "rank",
            "pick",
            "opponent",
            pl.col("pick_odds").alias("odds"),
            "decimal_price",
            (pl.col("pick_probability") * 100)
            .round(2)
            .alias("probability"),
            (pl.col("pick_probability") * 100)
            .round(2)
            .alias("confidence"),
        )
        .sort("rank", descending=True)
    )


def main() -> None:
    args = parse_args()
    picks = generate_picks(
        args.season,
        args.week,
        args.first_game,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    picks.write_csv(OUTPUT)

    with pl.Config(tbl_rows=25, tbl_width_chars=120):
        print(f"\nPIGSKIN PREDICTOR — {args.season} WEEK {args.week}")
        print(picks)

    print("\nSaved outputs/weekly_picks.csv")
    print("Next run: python scripts/update_dashboard.py")


if __name__ == "__main__":
    main()
