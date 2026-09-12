"""Record and settle fixed-stake paper bets on Pigskin Predictor picks.

This tracker never chooses a team. It reads the exact selections already saved
in outputs/weekly_picks.csv by model 11 and records the odds captured there.

Record this week's picks:
    python models/13_paper_bet_tracker.py record --stake 10

Settle completed picks later:
    python models/13_paper_bet_tracker.py settle

Outputs:
    outputs/paper_bets.csv
    outputs/paper_betting_summary.csv
    outputs/paper_betting_by_week.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy as nfl
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
PICKS_FILE = ROOT / "outputs" / "weekly_picks.csv"
LEDGER_FILE = ROOT / "outputs" / "paper_bets.csv"
SUMMARY_FILE = ROOT / "outputs" / "paper_betting_summary.csv"
WEEKLY_FILE = ROOT / "outputs" / "paper_betting_by_week.csv"


LEDGER_COLUMNS = [
    "bet_id",
    "season",
    "week_number",
    "week",
    "rank",
    "pick",
    "opponent",
    "probability",
    "odds",
    "stake",
    "recorded_at_utc",
    "status",
    "winner",
    "result",
    "gross_return",
    "net_profit",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track paper bets on the generated weekly picks."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser(
        "record",
        help="Record outputs/weekly_picks.csv as pending paper bets.",
    )
    record.add_argument(
        "--stake",
        type=float,
        default=10.0,
        help="Paper stake on every selected team (default: $10).",
    )

    subparsers.add_parser(
        "settle",
        help="Settle pending bets whose games have finished.",
    )

    args = parser.parse_args()
    if args.command == "record" and args.stake <= 0:
        parser.error("--stake must be greater than zero.")
    return args


def empty_ledger() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "bet_id": pl.String,
            "season": pl.Int64,
            "week_number": pl.Int64,
            "week": pl.String,
            "rank": pl.Int64,
            "pick": pl.String,
            "opponent": pl.String,
            "probability": pl.Float64,
            "odds": pl.Float64,
            "stake": pl.Float64,
            "recorded_at_utc": pl.String,
            "status": pl.String,
            "winner": pl.String,
            "result": pl.String,
            "gross_return": pl.Float64,
            "net_profit": pl.Float64,
        }
    )


def load_ledger() -> pl.DataFrame:
    if not LEDGER_FILE.exists():
        return empty_ledger()
    return pl.read_csv(LEDGER_FILE)


def matchup_id(
    season: int,
    week: int,
    team_a: str,
    team_b: str,
) -> str:
    teams = sorted([team_a, team_b])
    return f"{season}-{week:02d}-{teams[0]}-{teams[1]}"


def record_picks(stake: float) -> pl.DataFrame:
    if not PICKS_FILE.exists():
        raise FileNotFoundError(
            "outputs/weekly_picks.csv was not found. Run model 11 first."
        )

    picks = pl.read_csv(PICKS_FILE)
    required = {
        "season", "week_number", "week", "rank", "pick",
        "opponent", "probability", "odds",
    }
    missing = required - set(picks.columns)
    if missing:
        raise ValueError(
            "weekly_picks.csv is missing: "
            + ", ".join(sorted(missing))
            + ". Pull the latest model 11 and regenerate the picks."
        )

    recorded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new_rows = []

    for pick in picks.iter_rows(named=True):
        new_rows.append({
            "bet_id": matchup_id(
                int(pick["season"]),
                int(pick["week_number"]),
                pick["pick"],
                pick["opponent"],
            ),
            "season": int(pick["season"]),
            "week_number": int(pick["week_number"]),
            "week": pick["week"],
            "rank": int(pick["rank"]),
            "pick": pick["pick"],
            "opponent": pick["opponent"],
            "probability": float(pick["probability"]),
            "odds": float(pick["odds"]),
            "stake": float(stake),
            "recorded_at_utc": recorded_at,
            "status": "Pending",
            "winner": None,
            "result": None,
            "gross_return": None,
            "net_profit": None,
        })

    new_bets = pl.DataFrame(new_rows, schema=empty_ledger().schema)
    ledger = load_ledger()
    new_ids = new_bets["bet_id"].to_list()

    # A regenerated card replaces only matching pending bets. Settled history
    # is immutable and remains in the ledger.
    ledger = ledger.filter(
        ~(
            pl.col("bet_id").is_in(new_ids)
            & (pl.col("status") == "Pending")
        )
    )

    settled_duplicates = ledger.filter(
        pl.col("bet_id").is_in(new_ids)
        & (pl.col("status") == "Settled")
    )
    if not settled_duplicates.is_empty():
        settled_ids = set(settled_duplicates["bet_id"].to_list())
        new_bets = new_bets.filter(~pl.col("bet_id").is_in(settled_ids))

    return (
        pl.concat([ledger, new_bets], how="vertical")
        .sort(["season", "week_number", "rank"])
    )


def profit_for_win(stake: float, american_odds: float) -> float:
    if american_odds < 0:
        return stake * 100 / abs(american_odds)
    return stake * american_odds / 100


def settle_bets(ledger: pl.DataFrame) -> pl.DataFrame:
    pending = ledger.filter(pl.col("status") == "Pending")
    if pending.is_empty():
        print("There are no pending bets to settle.")
        return ledger

    seasons = sorted(set(pending["season"].to_list()))
    schedules = nfl.load_schedules(seasons)
    rows = []

    for bet in ledger.iter_rows(named=True):
        if bet["status"] == "Settled":
            rows.append(bet)
            continue

        game = schedules.filter(
            (pl.col("season") == bet["season"])
            & (pl.col("week") == bet["week_number"])
            & (
                (
                    (pl.col("home_team") == bet["pick"])
                    & (pl.col("away_team") == bet["opponent"])
                )
                | (
                    (pl.col("away_team") == bet["pick"])
                    & (pl.col("home_team") == bet["opponent"])
                )
            )
        )

        if game.is_empty():
            rows.append(bet)
            continue

        game_row = game.row(0, named=True)
        home_score = game_row["home_score"]
        away_score = game_row["away_score"]
        if home_score is None or away_score is None:
            rows.append(bet)
            continue

        if home_score > away_score:
            winner = game_row["home_team"]
        elif away_score > home_score:
            winner = game_row["away_team"]
        else:
            winner = None

        if winner is None:
            result = "Push"
            gross_return = float(bet["stake"])
            net_profit = 0.0
        elif winner == bet["pick"]:
            result = "Win"
            net_profit = profit_for_win(
                float(bet["stake"]),
                float(bet["odds"]),
            )
            gross_return = float(bet["stake"]) + net_profit
        else:
            result = "Loss"
            gross_return = 0.0
            net_profit = -float(bet["stake"])

        bet.update({
            "status": "Settled",
            "winner": winner,
            "result": result,
            "gross_return": round(gross_return, 2),
            "net_profit": round(net_profit, 2),
        })
        rows.append(bet)

    return pl.DataFrame(rows, schema=empty_ledger().schema)


def write_metrics(ledger: pl.DataFrame) -> None:
    settled = ledger.filter(pl.col("status") == "Settled")
    if settled.is_empty():
        empty_ledger().select(
            pl.lit("Overall").alias("period"),
            pl.lit(0).alias("bets"),
            pl.lit(0.0).alias("total_staked"),
            pl.lit(0.0).alias("net_profit"),
            pl.lit(0.0).alias("roi"),
        ).write_csv(SUMMARY_FILE)
        return

    settled = settled.with_columns(
        pl.col("net_profit").cum_sum().alias("cumulative_profit")
    )

    total_staked = float(settled["stake"].sum())
    net_profit = float(settled["net_profit"].sum())
    wins = settled.filter(pl.col("result") == "Win").height
    losses = settled.filter(pl.col("result") == "Loss").height
    decisions = wins + losses

    summary = pl.DataFrame([{
        "period": "Overall",
        "bets": settled.height,
        "wins": wins,
        "losses": losses,
        "pushes": settled.filter(pl.col("result") == "Push").height,
        "win_rate": round(wins / decisions * 100, 2) if decisions else 0.0,
        "total_staked": round(total_staked, 2),
        "gross_return": round(float(settled["gross_return"].sum()), 2),
        "net_profit": round(net_profit, 2),
        "roi": round(net_profit / total_staked * 100, 2),
    }])

    weekly = (
        settled
        .group_by(["season", "week_number", "week"])
        .agg(
            pl.len().alias("bets"),
            (pl.col("result") == "Win").sum().alias("wins"),
            pl.col("stake").sum().round(2).alias("total_staked"),
            pl.col("gross_return").sum().round(2).alias("gross_return"),
            pl.col("net_profit").sum().round(2).alias("net_profit"),
        )
        .with_columns(
            (pl.col("net_profit") / pl.col("total_staked") * 100)
            .round(2)
            .alias("roi")
        )
        .sort(["season", "week_number"])
        .with_columns(
            pl.col("net_profit").cum_sum().round(2)
            .alias("cumulative_profit")
        )
    )

    summary.write_csv(SUMMARY_FILE)
    weekly.write_csv(WEEKLY_FILE)


def main() -> None:
    args = parse_args()
    ledger = load_ledger()

    if args.command == "record":
        ledger = record_picks(args.stake)
        print(
            f"Recorded {ledger.filter(pl.col('status') == 'Pending').height} "
            f"pending paper bets at ${args.stake:.2f} each."
        )
    else:
        before = ledger.filter(pl.col("status") == "Settled").height
        ledger = settle_bets(ledger)
        after = ledger.filter(pl.col("status") == "Settled").height
        print(f"Settled {after - before} newly completed bets.")

    LEDGER_FILE.parent.mkdir(parents=True, exist_ok=True)
    ledger.select(LEDGER_COLUMNS).write_csv(LEDGER_FILE)
    write_metrics(ledger)

    with pl.Config(tbl_rows=25, tbl_width_chars=150):
        print("\nPAPER-BET LEDGER")
        print(
            ledger.select(
                "week", "rank", "pick", "opponent", "odds",
                "stake", "status", "result", "net_profit",
            )
        )

    print("\nSaved outputs/paper_bets.csv")


if __name__ == "__main__":
    main()
