"""Build the static dashboard data file from model output CSVs.

Run from the repository root:
    python scripts/update_dashboard.py
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
DESTINATION = ROOT / "docs" / "data" / "dashboard-data.js"


def read_csv(name: str) -> list[dict[str, str]]:
    path = OUTPUTS / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def build_dashboard() -> dict:
    summaries = read_csv("model_summary.csv")
    picks = read_csv("weekly_picks.csv")
    paper_bets = read_csv("paper_bets.csv")
    betting_weekly = read_csv("paper_betting_by_week.csv")
    confidence_summaries = read_csv("confidence_pool_summary.csv")
    confidence_weekly = read_csv("confidence_pool_by_week.csv")

    strategies = [
        {"name": row["strategy"], "points": number(row, "confidence_points")}
        for row in summaries
    ]
    market = next((row for row in summaries if row["strategy"] == "Market"), {})
    full_model = next((row for row in summaries if row["strategy"] == "Full model"), {})
    best = max(summaries, key=lambda row: number(row, "confidence_points"), default={})

    week = picks[0].get("week", "Awaiting current schedule") if picks else "Awaiting current schedule"
    bet_lookup = {
        (row.get("week"), row.get("pick"), row.get("opponent")): row
        for row in paper_bets
    }
    rendered_picks = []
    for row in picks:
        bet = bet_lookup.get(
            (row.get("week"), row.get("pick"), row.get("opponent")),
            {},
        )
        rendered_picks.append({
            "rank": int(number(row, "rank")),
            "pick": row.get("pick", "TBD"),
            "opponent": row.get("opponent", "TBD"),
            "probability": number(row, "probability"),
            "confidence": number(row, "confidence"),
            "decimal_price": number(
                bet or row,
                "decimal_price",
            ),
            "stake": number(bet, "stake"),
            "potential_return": number(bet, "potential_return"),
            "potential_profit": number(bet, "potential_profit"),
            "status": bet.get("status", "Not recorded"),
            "result": bet.get("result") or None,
            "net_profit": (
                number(bet, "net_profit")
                if bet.get("net_profit") not in (None, "")
                else None
            ),
        })

    current_bets = [
        row for row in paper_bets
        if not picks or row.get("week") == week
    ]
    settled_bets = [
        row for row in paper_bets
        if row.get("status") == "Settled"
    ]
    total_staked = sum(number(row, "stake") for row in current_bets)
    potential_return = sum(
        number(row, "potential_return") for row in current_bets
    )
    settled_staked = sum(number(row, "stake") for row in settled_bets)
    net_profit = sum(number(row, "net_profit") for row in settled_bets)
    roi = net_profit / settled_staked * 100 if settled_staked else 0.0

    rendered_betting_weeks = [
        {
            "week": row.get("week", "Week"),
            "bets": int(number(row, "bets")),
            "staked": number(row, "total_staked"),
            "net_profit": number(row, "net_profit"),
            "roi": number(row, "roi"),
            "cumulative_profit": number(row, "cumulative_profit"),
        }
        for row in betting_weekly
    ]

    latest_confidence = max(
        confidence_summaries,
        key=lambda row: number(row, "season"),
        default={},
    )
    confidence_season = (
        int(number(latest_confidence, "season"))
        if latest_confidence
        else None
    )
    rendered_confidence_weeks = [
        {
            "week": row.get("week", f"Week {int(number(row, 'week_number'))}"),
            "week_number": int(number(row, "week_number")),
            "games": int(number(row, "games")),
            "correct_picks": int(number(row, "correct_picks")),
            "incorrect_picks": int(number(row, "incorrect_picks")),
            "ties": int(number(row, "ties")),
            "confidence_points": int(number(row, "confidence_points")),
            "max_points": int(number(row, "max_points")),
            "accuracy_percent": number(row, "accuracy_percent"),
            "points_percent": number(row, "points_percent"),
        }
        for row in confidence_weekly
        if confidence_season is None
        or int(number(row, "season")) == confidence_season
    ]
    rendered_confidence_weeks.sort(key=lambda row: row["week_number"])

    confidence_games = int(number(latest_confidence, "games"))
    confidence_correct = int(number(latest_confidence, "correct_picks"))
    confidence_points = int(number(latest_confidence, "confidence_points"))
    confidence_max = int(number(latest_confidence, "max_points"))

    return {
        "updated": date.today().isoformat(),
        "week": week,
        "metrics": [
            {"label": "Market winner accuracy", "value": number(market, "accuracy"), "suffix": "%", "detail": "historical baseline"},
            {"label": "Walk-forward accuracy", "value": number(full_model, "accuracy"), "suffix": "%", "detail": "out-of-sample"},
            {"label": "Best confidence points", "value": number(best, "confidence_points"), "suffix": "%", "detail": best.get("strategy", "best strategy")},
        ],
        "strategies": strategies,
        "picks": rendered_picks,
        "confidence_pool": {
            "season": confidence_season,
            "metrics": [
                {
                    "label": "Weeks completed",
                    "value": int(number(latest_confidence, "weeks")),
                    "format": "integer",
                    "detail": "settled cards",
                },
                {
                    "label": "Pick accuracy",
                    "value": number(latest_confidence, "accuracy_percent"),
                    "format": "percent",
                    "detail": f"{confidence_correct} of {confidence_games} correct",
                },
                {
                    "label": "Confidence points",
                    "value": confidence_points,
                    "format": "integer",
                    "detail": f"{confidence_max} available",
                },
                {
                    "label": "Points percentage",
                    "value": number(latest_confidence, "points_percent"),
                    "format": "percent",
                    "detail": "season efficiency",
                },
            ],
            "weekly": rendered_confidence_weeks,
        },
        "betting": {
            "metrics": [
                {
                    "label": "Weekly paper stake",
                    "value": round(total_staked, 2),
                    "format": "money",
                    "detail": f"{len(current_bets)} selected teams",
                },
                {
                    "label": "Potential return",
                    "value": round(potential_return, 2),
                    "format": "money",
                    "detail": "if every current pick wins",
                },
                {
                    "label": "Settled net P/L",
                    "value": round(net_profit, 2),
                    "format": "money",
                    "detail": f"{len(settled_bets)} settled bets",
                },
                {
                    "label": "Settled ROI",
                    "value": round(roi, 2),
                    "format": "percent",
                    "detail": "return on paper stake",
                },
            ],
            "weekly": rendered_betting_weeks,
        },
        "method": [
            {"title": "Read the market", "text": "Convert moneylines into implied win probabilities and remove the bookmaker margin."},
            {"title": "Add team strength", "text": "Update ELO ratings chronologically so every prediction uses only information available at the time."},
            {"title": "Test forward", "text": "Train on past seasons and score the next one, repeating the process without leaking future results."},
            {"title": "Rank the card", "text": "Order predicted winners by confidence and assign each weekly weight exactly once."},
        ],
    }


def main() -> None:
    payload = json.dumps(build_dashboard(), indent=2)
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(
        "// Generated by scripts/update_dashboard.py. Edit outputs/*.csv, not this file.\n"
        f"window.FOURTH_AND_DATA = {payload};\n",
        encoding="utf-8",
    )
    print(f"Updated {DESTINATION.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
