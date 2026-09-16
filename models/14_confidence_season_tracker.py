"""Track Pigskin Predictor confidence-pool picks across the full NFL season.

Workflow each week:

1. Generate weekly picks:
   python models/11_generate_weekly_picks.py --season 2026 --week 1

2. Record the card before generating another week:
   python models/14_confidence_season_tracker.py record

3. After the games are finished:
   python models/14_confidence_season_tracker.py settle

Outputs:
    outputs/confidence_pool_picks.csv
    outputs/confidence_pool_by_week.csv
    outputs/confidence_pool_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nflreadpy as nfl
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
PICKS_FILE = ROOT / "outputs" / "weekly_picks.csv"
LEDGER_FILE = ROOT / "outputs" / "confidence_pool_picks.csv"
WEEKLY_FILE = ROOT / "outputs" / "confidence_pool_by_week.csv"
SUMMARY_FILE = ROOT / "outputs" / "confidence_pool_summary.csv"


LEDGER_COLUMNS = [
    "pick_id",
    "season",
    "week_number",
    "week",
    "weight",
    "pick",
    "opponent",
    "probability",
    "status",
    "winner",
    "result",
    "points",
]


def empty_ledger() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "pick_id": pl.String,
            "season": pl.Int64,
            "week_number": pl.Int64,
            "week": pl.String,
            "weight": pl.Int64,
            "pick": pl.String,
            "opponent": pl.String,
            "probability": pl.Float64,
            "status": pl.String,
            "winner": pl.String,
            "result": pl.String,
            "points": pl.Int64,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track confidence-pool picks throughout the NFL season."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "record",
        help="Save the current weekly_picks.csv into the permanent season ledger.",
    )
    subparsers.add_parser(
        "settle",
        help="Score any pending picks whose NFL games have finished.",
    )
    return parser.parse_args()


def matchup_id(season: int, week: int, team_a: str, team_b: str) -> str:
    teams = sorted([team_a, team_b])
    return f"{season}-{week:02d}-{teams[0]}-{teams[1]}"


def load_ledger() -> pl.DataFrame:
    if not LEDGER_FILE.exists():
        return empty_ledger()

    ledger = pl.read_csv(LEDGER_FILE)
    target_schema = empty_ledger().schema

    ledger = ledger.with_columns(
        [
            pl.col(column).cast(dtype, strict=False)
            for column, dtype in target_schema.items()
            if column in ledger.columns
        ]
    )

    missing_columns = [
        pl.lit(None, dtype=dtype).alias(column)
        for column, dtype in target_schema.items()
        if column not in ledger.columns
    ]
    if missing_columns:
        ledger = ledger.with_columns(missing_columns)

    return ledger.select(LEDGER_COLUMNS)


def record_current_week() -> pl.DataFrame:
    if not PICKS_FILE.exists():
        raise FileNotFoundError(
            "outputs/weekly_picks.csv does not exist. Run model 11 first."
        )

    picks = pl.read_csv(PICKS_FILE)
    required = {
        "season",
        "week_number",
        "week",
        "rank",
        "pick",
        "opponent",
        "probability",
    }
    missing = required - set(picks.columns)
    if missing:
        raise ValueError(
            "weekly_picks.csv is missing: " + ", ".join(sorted(missing))
        )

    rows = []
    for pick in picks.iter_rows(named=True):
        season = int(pick["season"])
        week_number = int(pick["week_number"])
        team = pick["pick"]
        opponent = pick["opponent"]
        rows.append(
            {
                "pick_id": matchup_id(season, week_number, team, opponent),
                "season": season,
                "week_number": week_number,
                "week": pick["week"],
                "weight": int(pick["rank"]),
                "pick": team,
                "opponent": opponent,
                "probability": float(pick["probability"]),
                "status": "Pending",
                "winner": None,
                "result": None,
                "points": None,
            }
        )

    new_picks = pl.DataFrame(rows, schema=empty_ledger().schema)
    ledger = load_ledger()
    new_ids = new_picks["pick_id"].to_list()

    # A regenerated card replaces matching pending picks, but settled history
    # is immutable once the games have been scored.
    ledger = ledger.filter(
        ~(
            pl.col("pick_id").is_in(new_ids)
            & (pl.col("status") == "Pending")
        )
    )

    settled_duplicates = ledger.filter(
        pl.col("pick_id").is_in(new_ids)
        & (pl.col("status") == "Settled")
    )
    if not settled_duplicates.is_empty():
        settled_ids = set(settled_duplicates["pick_id"].to_list())
        new_picks = new_picks.filter(~pl.col("pick_id").is_in(settled_ids))

    return (
        pl.concat([ledger, new_picks], how="vertical")
        .sort(
            ["season", "week_number", "weight"],
            descending=[False, False, True],
        )
    )


def settle_picks(ledger: pl.DataFrame) -> pl.DataFrame:
    pending = ledger.filter(pl.col("status") == "Pending")
    if pending.is_empty():
        print("There are no pending confidence picks to settle.")
        return ledger

    seasons = sorted(set(pending["season"].to_list()))
    schedules = nfl.load_schedules(seasons)
    rows = []

    for pick in ledger.iter_rows(named=True):
        if pick["status"] == "Settled":
            rows.append(pick)
            continue

        game = schedules.filter(
            (pl.col("season") == pick["season"])
            & (pl.col("week") == pick["week_number"])
            & (
                (
                    (pl.col("home_team") == pick["pick"])
                    & (pl.col("away_team") == pick["opponent"])
                )
                | (
                    (pl.col("away_team") == pick["pick"])
                    & (pl.col("home_team") == pick["opponent"])
                )
            )
        )

        if game.is_empty():
            rows.append(pick)
            continue

        game_row = game.row(0, named=True)
        home_score = game_row["home_score"]
        away_score = game_row["away_score"]
        if home_score is None or away_score is None:
            rows.append(pick)
            continue

        if home_score > away_score:
            winner = game_row["home_team"]
        elif away_score > home_score:
            winner = game_row["away_team"]
        else:
            winner = None

        if winner is None:
            result = "Tie"
            points = 0
        elif winner == pick["pick"]:
            result = "Win"
            points = int(pick["weight"])
        else:
            result = "Loss"
            points = 0

        pick.update(
            {
                "status": "Settled",
                "winner": winner,
                "result": result,
                "points": points,
            }
        )
        rows.append(pick)

    return pl.DataFrame(rows, schema=empty_ledger().schema)


def write_metrics(ledger: pl.DataFrame) -> None:
    settled = ledger.filter(pl.col("status") == "Settled")

    if settled.is_empty():
        pl.DataFrame(
            schema={
                "season": pl.Int64,
                "week_number": pl.Int64,
                "week": pl.String,
                "games": pl.Int64,
                "correct_picks": pl.Int64,
                "incorrect_picks": pl.Int64,
                "ties": pl.Int64,
                "confidence_points": pl.Int64,
                "max_points": pl.Int64,
                "accuracy_percent": pl.Float64,
                "points_percent": pl.Float64,
            }
        ).write_csv(WEEKLY_FILE)
        pl.DataFrame(
            schema={
                "season": pl.Int64,
                "weeks": pl.Int64,
                "games": pl.Int64,
                "correct_picks": pl.Int64,
                "incorrect_picks": pl.Int64,
                "ties": pl.Int64,
                "confidence_points": pl.Int64,
                "max_points": pl.Int64,
                "accuracy_percent": pl.Float64,
                "points_percent": pl.Float64,
            }
        ).write_csv(SUMMARY_FILE)
        return

    weekly = (
        settled
        .group_by(["season", "week_number", "week"])
        .agg(
            pl.len().alias("games"),
            (pl.col("result") == "Win").sum().alias("correct_picks"),
            (pl.col("result") == "Loss").sum().alias("incorrect_picks"),
            (pl.col("result") == "Tie").sum().alias("ties"),
            pl.col("points").sum().alias("confidence_points"),
            pl.col("weight").sum().alias("max_points"),
        )
        .with_columns(
            (
                pl.col("correct_picks")
                / (pl.col("correct_picks") + pl.col("incorrect_picks"))
                * 100
            )
            .round(2)
            .alias("accuracy_percent"),
            (
                pl.col("confidence_points") / pl.col("max_points") * 100
            )
            .round(2)
            .alias("points_percent"),
        )
        .sort(["season", "week_number"])
    )

    season_summary = (
        weekly
        .group_by("season")
        .agg(
            pl.len().alias("weeks"),
            pl.col("games").sum().alias("games"),
            pl.col("correct_picks").sum().alias("correct_picks"),
            pl.col("incorrect_picks").sum().alias("incorrect_picks"),
            pl.col("ties").sum().alias("ties"),
            pl.col("confidence_points").sum().alias("confidence_points"),
            pl.col("max_points").sum().alias("max_points"),
        )
        .with_columns(
            (
                pl.col("correct_picks")
                / (pl.col("correct_picks") + pl.col("incorrect_picks"))
                * 100
            )
            .round(2)
            .alias("accuracy_percent"),
            (
                pl.col("confidence_points") / pl.col("max_points") * 100
            )
            .round(2)
            .alias("points_percent"),
        )
        .sort("season")
    )

    weekly.write_csv(WEEKLY_FILE)
    season_summary.write_csv(SUMMARY_FILE)


def print_results(ledger: pl.DataFrame) -> None:
    with pl.Config(tbl_rows=40, tbl_width_chars=150):
        print("\nCONFIDENCE POOL LEDGER")
        print(
            ledger.select(
                "week",
                "weight",
                "pick",
                "opponent",
                "probability",
                "status",
                "winner",
                "result",
                "points",
            )
        )

    if WEEKLY_FILE.exists():
        weekly = pl.read_csv(WEEKLY_FILE)
        if not weekly.is_empty():
            print("\nWEEKLY PERFORMANCE")
            print(weekly)

    if SUMMARY_FILE.exists():
        summary = pl.read_csv(SUMMARY_FILE)
        if not summary.is_empty():
            print("\nSEASON PERFORMANCE")
            print(summary)


def main() -> None:
    args = parse_args()
    ledger = load_ledger()

    if args.command == "record":
        ledger = record_current_week()
        pending = ledger.filter(pl.col("status") == "Pending").height
        print(f"Recorded current confidence card. {pending} pending picks.")
    else:
        before = ledger.filter(pl.col("status") == "Settled").height
        ledger = settle_picks(ledger)
        after = ledger.filter(pl.col("status") == "Settled").height
        print(f"Settled {after - before} new confidence picks.")

    LEDGER_FILE.parent.mkdir(parents=True, exist_ok=True)
    ledger.select(LEDGER_COLUMNS).write_csv(LEDGER_FILE)
    write_metrics(ledger)
    print_results(ledger)

    print("\nSaved:")
    print("outputs/confidence_pool_picks.csv")
    print("outputs/confidence_pool_by_week.csv")
    print("outputs/confidence_pool_summary.csv")


if __name__ == "__main__":
    main()
