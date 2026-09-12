import nflreadpy as nfl
import polars as pl


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

games = nfl.load_schedules([2021, 2022, 2023, 2024, 2025])


# --------------------------------------------------
# 2. KEEP COMPLETED REGULAR-SEASON GAMES
# --------------------------------------------------

games = games.filter(
    (pl.col("game_type") == "REG")
    & pl.col("home_score").is_not_null()
    & pl.col("away_score").is_not_null()
    & pl.col("home_moneyline").is_not_null()
    & pl.col("away_moneyline").is_not_null()
)


# --------------------------------------------------
# 3. CONVERT MONEYLINES TO RAW PROBABILITIES
# --------------------------------------------------

games = games.with_columns(

    pl.when(pl.col("home_moneyline") < 0)
    .then(
        -pl.col("home_moneyline")
        / (-pl.col("home_moneyline") + 100)
    )
    .otherwise(
        100 / (pl.col("home_moneyline") + 100)
    )
    .alias("home_raw_probability"),

    pl.when(pl.col("away_moneyline") < 0)
    .then(
        -pl.col("away_moneyline")
        / (-pl.col("away_moneyline") + 100)
    )
    .otherwise(
        100 / (pl.col("away_moneyline") + 100)
    )
    .alias("away_raw_probability")

)


# --------------------------------------------------
# 4. REMOVE BOOKMAKER MARGIN
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
# 5. MARKET PICK + CONFIDENCE
# --------------------------------------------------

games = games.with_columns(

    pl.when(
        pl.col("home_win_probability")
        >= pl.col("away_win_probability")
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
# 6. ACTUAL WINNER
# --------------------------------------------------

games = games.with_columns(

    pl.when(pl.col("home_score") > pl.col("away_score"))
    .then(pl.col("home_team"))

    .when(pl.col("away_score") > pl.col("home_score"))
    .then(pl.col("away_team"))

    .otherwise(None)
    .alias("winner")

)


# --------------------------------------------------
# 7. DID OUR PICK WIN?
# --------------------------------------------------

games = games.with_columns(

    (
        pl.col("market_pick") == pl.col("winner")
    ).alias("correct")

)


# --------------------------------------------------
# 8. ASSIGN CONFIDENCE WEIGHTS
#
# Lowest probability = 1
# Highest probability = number of games that week
# --------------------------------------------------

games = games.with_columns(

    pl.col("market_confidence")
    .rank(method="ordinal")
    .over(["season", "week"])
    .cast(pl.Int32)
    .alias("weight")

)


# --------------------------------------------------
# 9. CALCULATE OFFICE POOL POINTS
# --------------------------------------------------

games = games.with_columns(

    pl.when(pl.col("correct"))
    .then(pl.col("weight"))
    .otherwise(0)
    .alias("confidence_points")

)


# --------------------------------------------------
# 10. WEEKLY RESULTS
# --------------------------------------------------

weekly = (
    games
    .group_by(["season", "week"])
    .agg(

        pl.len().alias("games"),

        pl.col("correct")
        .sum()
        .alias("correct_picks"),

        pl.col("confidence_points")
        .sum()
        .alias("points"),

        pl.col("weight")
        .sum()
        .alias("max_points")

    )
    .with_columns(

        (
            pl.col("correct_picks")
            / pl.col("games")
            * 100
        )
        .round(1)
        .alias("accuracy_percent"),

        (
            pl.col("points")
            / pl.col("max_points")
            * 100
        )
        .round(1)
        .alias("points_percent")

    )
    .sort(["season", "week"])
)


print("\nFIRST 20 WEEKS")
print(weekly.head(20))


# --------------------------------------------------
# 11. SEASON RESULTS
# --------------------------------------------------

season_results = (
    weekly
    .group_by("season")
    .agg(

        pl.col("games").sum().alias("games"),

        pl.col("correct_picks")
        .sum()
        .alias("correct_picks"),

        pl.col("points")
        .sum()
        .alias("confidence_points"),

        pl.col("max_points")
        .sum()
        .alias("max_points")

    )
    .with_columns(

        (
            pl.col("correct_picks")
            / pl.col("games")
            * 100
        )
        .round(1)
        .alias("accuracy_percent"),

        (
            pl.col("confidence_points")
            / pl.col("max_points")
            * 100
        )
        .round(1)
        .alias("points_percent")

    )
    .sort("season")
)


print("\nSEASON RESULTS")
print(season_results)


# --------------------------------------------------
# 12. ALL-TIME RESULTS
# --------------------------------------------------

overall = games.select(

    pl.len().alias("games"),

    pl.col("correct")
    .sum()
    .alias("correct_picks"),

    pl.col("confidence_points")
    .sum()
    .alias("confidence_points"),

    pl.col("weight")
    .sum()
    .alias("max_points"),

    (
        pl.col("correct").mean() * 100
    )
    .round(1)
    .alias("accuracy_percent"),

    (
        pl.col("confidence_points").sum()
        / pl.col("weight").sum()
        * 100
    )
    .round(1)
    .alias("points_percent")

)


print("\nOVERALL")
print(overall)