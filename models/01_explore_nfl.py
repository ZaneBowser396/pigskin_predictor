import nflreadpy as nfl
import polars as pl

# --------------------------------------------------
# 1. LOAD NFL DATA
# --------------------------------------------------

games = nfl.load_schedules([2021, 2022, 2023, 2024, 2025])


# --------------------------------------------------
# 2. KEEP ONLY COMPLETED REGULAR-SEASON GAMES
# --------------------------------------------------

games = games.filter(
    (pl.col("game_type") == "REG")
    & pl.col("home_score").is_not_null()
    & pl.col("away_score").is_not_null()
    & pl.col("home_moneyline").is_not_null()
    & pl.col("away_moneyline").is_not_null()
    & (pl.col("home_moneyline") != pl.col("away_moneyline"))
)


# --------------------------------------------------
# 3. IDENTIFY THE BETTING FAVOURITE
# --------------------------------------------------

games = games.with_columns(

    pl.when(
        pl.col("home_moneyline") < pl.col("away_moneyline")
    )
    .then(pl.col("home_team"))
    .otherwise(pl.col("away_team"))
    .alias("favorite")

)


# --------------------------------------------------
# 4. IDENTIFY WHO ACTUALLY WON
# --------------------------------------------------

games = games.with_columns(

    pl.when(
        pl.col("home_score") > pl.col("away_score")
    )
    .then(pl.col("home_team"))
    .otherwise(pl.col("away_team"))
    .alias("winner")

)


# --------------------------------------------------
# 5. DID THE FAVOURITE WIN?
# --------------------------------------------------

games = games.with_columns(

    (pl.col("favorite") == pl.col("winner"))
    .alias("favorite_won")

)


# --------------------------------------------------
# 6. OVERALL RESULTS
# --------------------------------------------------

overall = games.select(

    pl.len().alias("games"),

    pl.col("favorite_won")
    .sum()
    .alias("favorite_wins"),

    (
        pl.col("favorite_won")
        .mean() * 100
    )
    .round(1)
    .alias("favorite_win_percent")

)

print("\nOVERALL")
print(overall)


# --------------------------------------------------
# 7. RESULTS BY SEASON
# --------------------------------------------------

by_season = (
    games
    .group_by("season")
    .agg(

        pl.len().alias("games"),

        pl.col("favorite_won")
        .sum()
        .alias("favorite_wins"),

        (
            pl.col("favorite_won")
            .mean() * 100
        )
        .round(1)
        .alias("favorite_win_percent")

    )
    .sort("season")
)

print("\nBY SEASON")
print(by_season)