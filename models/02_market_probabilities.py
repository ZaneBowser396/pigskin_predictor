import nflreadpy as nfl
import polars as pl


# --------------------------------------------------
# 1. LOAD HISTORICAL NFL GAMES
# --------------------------------------------------

games = nfl.load_schedules([2021, 2022, 2023, 2024, 2025])


# --------------------------------------------------
# 2. KEEP COMPLETED REGULAR-SEASON GAMES WITH ODDS
# --------------------------------------------------

games = games.filter(
    (pl.col("game_type") == "REG")
    & pl.col("home_score").is_not_null()
    & pl.col("away_score").is_not_null()
    & pl.col("home_moneyline").is_not_null()
    & pl.col("away_moneyline").is_not_null()
)


# --------------------------------------------------
# 3. CONVERT AMERICAN MONEYLINE TO IMPLIED PROBABILITY
# --------------------------------------------------

games = games.with_columns(

    pl.when(pl.col("home_moneyline") < 0)
    .then(
        -pl.col("home_moneyline")
        / (-pl.col("home_moneyline") + 100)
    )
    .otherwise(
        100
        / (pl.col("home_moneyline") + 100)
    )
    .alias("home_raw_probability"),


    pl.when(pl.col("away_moneyline") < 0)
    .then(
        -pl.col("away_moneyline")
        / (-pl.col("away_moneyline") + 100)
    )
    .otherwise(
        100
        / (pl.col("away_moneyline") + 100)
    )
    .alias("away_raw_probability")

)


# --------------------------------------------------
# 4. REMOVE THE BOOKMAKER'S MARGIN
# --------------------------------------------------

games = games.with_columns(

    (
        pl.col("home_raw_probability")
        /
        (
            pl.col("home_raw_probability")
            + pl.col("away_raw_probability")
        )
    ).alias("home_win_probability"),

    (
        pl.col("away_raw_probability")
        /
        (
            pl.col("home_raw_probability")
            + pl.col("away_raw_probability")
        )
    ).alias("away_win_probability")

)


# --------------------------------------------------
# 5. DETERMINE MARKET PICK AND CONFIDENCE
# --------------------------------------------------

games = games.with_columns(

    pl.when(
        pl.col("home_win_probability")
        > pl.col("away_win_probability")
    )
    .then(pl.col("home_team"))
    .otherwise(pl.col("away_team"))
    .alias("market_pick"),


    pl.max_horizontal(
        "home_win_probability",
        "away_win_probability"
    ).alias("market_confidence")

)


# --------------------------------------------------
# 6. SHOW SOME EXAMPLES
# --------------------------------------------------

examples = (
    games
    .select(
        "season",
        "week",
        "away_team",
        "home_team",
        "away_moneyline",
        "home_moneyline",
        "market_pick",
        (
            pl.col("market_confidence") * 100
        )
        .round(1)
        .alias("confidence_percent")
    )
    .sort("confidence_percent", descending=True)
)

print("\nMOST CONFIDENT MARKET PICKS")
print(examples.head(20))


print("\nLEAST CONFIDENT MARKET PICKS")
print(examples.tail(20))