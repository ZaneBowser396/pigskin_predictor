"""Paper-betting backtest for NFL moneyline favourites.

This is a simulation only. It applies the same fixed stake to each qualifying
favourite and tracks profit/loss using historical closing moneylines.

Examples:
    python models/12_favourite_betting_backtest.py
    python models/12_favourite_betting_backtest.py --stake 20
    python models/12_favourite_betting_backtest.py --stake 10 --scope all

The default ``pool`` scope uses Sunday and Monday US games. ``all`` includes
every completed regular-season game.

Outputs:
    outputs/betting_ledger.csv
    outputs/betting_summary.csv
    outputs/betting_by_season.csv
    outputs/betting_by_confidence_band.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nflreadpy as nfl
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"

DEFAULT_STAKE = 10.0
DEFAULT_START_SEASON = 2021
DEFAULT_END_SEASON = 2025
POOL_WEEKDAYS = ["Sunday", "Monday"]
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest fixed-stake bets on NFL moneyline favourites."
    )
    parser.add_argument(
        "--stake",
        type=float,
        default=DEFAULT_STAKE,
        help=f"Paper stake per game (default: ${DEFAULT_STAKE:.2f}).",
    )
    parser.add_argument(
        "--start-season",
        type=int,
        default=DEFAULT_START_SEASON,
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=DEFAULT_END_SEASON,
    )
    parser.add_argument(
        "--scope",
        choices=["pool", "all"],
        default="pool",
        help="Use Sunday/Monday pool games or all regular-season games.",
    )
    args = parser.parse_args()

    if args.stake <= 0:
        parser.error("--stake must be greater than zero.")
    if args.start_season > args.end_season:
        parser.error("--start-season cannot be after --end-season.")

    return args


def moneyline_probability(column: str) -> pl.Expr:
    odds = pl.col(column)
    return (
        pl.when(odds < 0)
        .then(-odds / (-odds + 100))
        .otherwise(100 / (odds + 100))
    )


def longest_losing_streak(results: list[str]) -> int:
    longest = 0
    current = 0

    for result in results:
        if result == "Loss":
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def add_running_profit(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame
        .with_columns(
            pl.col("net_profit").cum_sum().alias("cumulative_profit")
        )
        .with_columns(
            pl.max_horizontal(
                pl.col("cumulative_profit").cum_max(),
                pl.lit(0.0),
            ).alias("running_peak")
        )
        .with_columns(
            (pl.col("running_peak") - pl.col("cumulative_profit"))
            .alias("drawdown")
        )
    )


def summarise(frame: pl.DataFrame, label: str) -> dict:
    bets = frame.height
    wins = frame.filter(pl.col("result") == "Win").height
    losses = frame.filter(pl.col("result") == "Loss").height
    pushes = frame.filter(pl.col("result") == "Push").height
    decisions = wins + losses

    total_staked = float(frame["stake"].sum()) if bets else 0.0
    net_profit = float(frame["net_profit"].sum()) if bets else 0.0
    gross_return = float(frame["gross_return"].sum()) if bets else 0.0
    win_rate = wins / decisions * 100 if decisions else 0.0
    roi = net_profit / total_staked * 100 if total_staked else 0.0
    average_probability = (
        float(frame["favourite_probability"].mean() * 100)
        if bets
        else 0.0
    )

    running = add_running_profit(frame) if bets else frame
    max_drawdown = (
        float(running["drawdown"].max())
        if bets
        else 0.0
    )

    return {
        "strategy": label,
        "bets": bets,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": round(win_rate, 2),
        "average_favourite_probability": round(average_probability, 2),
        "total_staked": round(total_staked, 2),
        "gross_return": round(gross_return, 2),
        "net_profit": round(net_profit, 2),
        "roi": round(roi, 2),
        "profit_per_bet": round(net_profit / bets, 2) if bets else 0.0,
        "max_drawdown": round(max_drawdown, 2),
        "longest_losing_streak": longest_losing_streak(
            frame["result"].to_list()
        ) if bets else 0,
    }


def build_ledger(
    start_season: int,
    end_season: int,
    stake: float,
    scope: str,
) -> pl.DataFrame:
    games = nfl.load_schedules(
        list(range(start_season, end_season + 1))
    )

    games = (
        games
        .filter(
            (pl.col("game_type") == "REG")
            & pl.col("home_score").is_not_null()
            & pl.col("away_score").is_not_null()
            & pl.col("home_moneyline").is_not_null()
            & pl.col("away_moneyline").is_not_null()
            & (pl.col("home_moneyline") != pl.col("away_moneyline"))
        )
        .sort(["season", "week", "gameday", "gametime", "game_id"])
    )

    if scope == "pool":
        games = games.filter(pl.col("weekday").is_in(POOL_WEEKDAYS))

    games = games.with_columns(
        moneyline_probability("home_moneyline")
        .alias("home_raw_probability"),
        moneyline_probability("away_moneyline")
        .alias("away_raw_probability"),
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
        pl.when(pl.col("home_moneyline") < pl.col("away_moneyline"))
        .then(pl.col("home_team"))
        .otherwise(pl.col("away_team"))
        .alias("favourite"),
        pl.when(pl.col("home_moneyline") < pl.col("away_moneyline"))
        .then(pl.col("away_team"))
        .otherwise(pl.col("home_team"))
        .alias("opponent"),
        pl.when(pl.col("home_moneyline") < pl.col("away_moneyline"))
        .then(pl.col("home_moneyline"))
        .otherwise(pl.col("away_moneyline"))
        .alias("favourite_odds"),
        pl.max_horizontal("home_probability", "away_probability")
        .alias("favourite_probability"),
        pl.when(pl.col("home_score") > pl.col("away_score"))
        .then(pl.col("home_team"))
        .when(pl.col("away_score") > pl.col("home_score"))
        .then(pl.col("away_team"))
        .otherwise(None)
        .alias("winner"),
        pl.lit(stake).alias("stake"),
    )

    games = games.with_columns(
        pl.when(pl.col("winner").is_null())
        .then(pl.lit("Push"))
        .when(pl.col("favourite") == pl.col("winner"))
        .then(pl.lit("Win"))
        .otherwise(pl.lit("Loss"))
        .alias("result")
    )

    games = games.with_columns(
        pl.when(pl.col("result") == "Push")
        .then(0.0)
        .when(pl.col("result") == "Loss")
        .then(-pl.col("stake"))
        .when(pl.col("favourite_odds") < 0)
        .then(
            pl.col("stake") * 100 / -pl.col("favourite_odds")
        )
        .otherwise(
            pl.col("stake") * pl.col("favourite_odds") / 100
        )
        .alias("net_profit")
    )

    games = games.with_columns(
        pl.when(pl.col("result") == "Win")
        .then(pl.col("stake") + pl.col("net_profit"))
        .when(pl.col("result") == "Push")
        .then(pl.col("stake"))
        .otherwise(0.0)
        .alias("gross_return"),
        pl.when(pl.col("favourite_probability") < 0.55)
        .then(pl.lit("50-55%"))
        .when(pl.col("favourite_probability") < 0.60)
        .then(pl.lit("55-60%"))
        .when(pl.col("favourite_probability") < 0.65)
        .then(pl.lit("60-65%"))
        .when(pl.col("favourite_probability") < 0.70)
        .then(pl.lit("65-70%"))
        .when(pl.col("favourite_probability") < 0.75)
        .then(pl.lit("70-75%"))
        .otherwise(pl.lit("75%+"))
        .alias("confidence_band"),
    )

    ledger = (
        games
        .select(
            "season",
            "week",
            "gameday",
            "weekday",
            "gametime",
            "away_team",
            "home_team",
            "favourite",
            "opponent",
            "favourite_odds",
            (pl.col("favourite_probability") * 100)
            .round(2)
            .alias("favourite_probability"),
            "confidence_band",
            "stake",
            "winner",
            "result",
            pl.col("gross_return").round(2),
            pl.col("net_profit").round(2),
        )
        .with_row_index("bet_number", offset=1)
    )

    return add_running_profit(ledger).with_columns(
        pl.col("cumulative_profit").round(2),
        pl.col("running_peak").round(2),
        pl.col("drawdown").round(2),
    )


def main() -> None:
    args = parse_args()
    ledger = build_ledger(
        args.start_season,
        args.end_season,
        args.stake,
        args.scope,
    )

    # The ledger stores probability as a percentage for readability. Convert
    # it back to a proportion while calculating threshold summaries.
    analysis = ledger.with_columns(
        (pl.col("favourite_probability") / 100)
        .alias("favourite_probability")
    )

    summary_rows = []
    for threshold in THRESHOLDS:
        qualifying = analysis.filter(
            pl.col("favourite_probability") >= threshold
        )
        label = (
            "All favourites"
            if threshold == 0.50
            else f"Favourite probability >= {threshold:.0%}"
        )
        summary_rows.append(summarise(qualifying, label))

    summary = pl.DataFrame(summary_rows)

    season_rows = [
        {
            "season": season,
            **summarise(
                analysis.filter(pl.col("season") == season),
                "All favourites",
            ),
        }
        for season in range(args.start_season, args.end_season + 1)
    ]
    by_season = pl.DataFrame(season_rows).select(
        "season",
        pl.exclude("season"),
    )

    band_order = [
        "50-55%", "55-60%", "60-65%",
        "65-70%", "70-75%", "75%+",
    ]
    band_rows = [
        {
            "confidence_band": band,
            **summarise(
                analysis.filter(pl.col("confidence_band") == band),
                band,
            ),
        }
        for band in band_order
    ]
    by_band = pl.DataFrame(band_rows).select(
        "confidence_band",
        pl.exclude("confidence_band"),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ledger.write_csv(OUTPUT_DIR / "betting_ledger.csv")
    summary.write_csv(OUTPUT_DIR / "betting_summary.csv")
    by_season.write_csv(OUTPUT_DIR / "betting_by_season.csv")
    by_band.write_csv(OUTPUT_DIR / "betting_by_confidence_band.csv")

    with pl.Config(tbl_rows=20, tbl_cols=15, tbl_width_chars=190):
        print("\nFAVOURITE MONEYLINE PAPER-BETTING BACKTEST")
        print(
            f"Seasons: {args.start_season}-{args.end_season} | "
            f"Stake: ${args.stake:.2f} | Scope: {args.scope}"
        )
        print("\nBY STRATEGY")
        print(summary)
        print("\nBY SEASON — ALL FAVOURITES")
        print(by_season)

    print("\nSaved betting outputs in outputs/.")


if __name__ == "__main__":
    main()
