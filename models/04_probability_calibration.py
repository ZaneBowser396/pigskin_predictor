import nflreadpy as nfl
import polars as pl
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. LOAD THE DATA
# --------------------------------------------------

games = nfl.load_schedules([2021, 2022, 2023, 2024, 2025])

games = games.filter(
    (pl.col("game_type") == "REG")
    & pl.col("home_score").is_not_null()
    & pl.col("away_score").is_not_null()
    & pl.col("home_moneyline").is_not_null()
    & pl.col("away_moneyline").is_not_null()
)


# --------------------------------------------------
# 2. CONVERT MONEYLINES INTO RAW PROBABILITIES
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
# 3. REMOVE BOOKMAKER MARGIN
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
# 4. MARKET PICK + MARKET CONFIDENCE
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
# 5. ACTUAL WINNER
# --------------------------------------------------

games = games.with_columns(

    pl.when(pl.col("home_score") > pl.col("away_score"))
    .then(pl.col("home_team"))

    .when(pl.col("away_score") > pl.col("home_score"))
    .then(pl.col("away_team"))

    .otherwise(None)
    .alias("winner")
)


games = games.with_columns(

    (pl.col("market_pick") == pl.col("winner"))
    .alias("correct")
)


# --------------------------------------------------
# 6. PUT PICKS INTO CONFIDENCE BANDS
# --------------------------------------------------

games = games.with_columns(

    pl.when(pl.col("market_confidence") < 0.55)
    .then(pl.lit("50-55%"))

    .when(pl.col("market_confidence") < 0.60)
    .then(pl.lit("55-60%"))

    .when(pl.col("market_confidence") < 0.65)
    .then(pl.lit("60-65%"))

    .when(pl.col("market_confidence") < 0.70)
    .then(pl.lit("65-70%"))

    .when(pl.col("market_confidence") < 0.75)
    .then(pl.lit("70-75%"))

    .when(pl.col("market_confidence") < 0.80)
    .then(pl.lit("75-80%"))

    .when(pl.col("market_confidence") < 0.85)
    .then(pl.lit("80-85%"))

    .otherwise(pl.lit("85%+"))

    .alias("confidence_band")
)


# --------------------------------------------------
# 7. CALCULATE ACTUAL RESULTS
# --------------------------------------------------

calibration = (
    games
    .group_by("confidence_band")
    .agg(

        pl.len().alias("games"),

        (
            pl.col("market_confidence").mean() * 100
        )
        .round(1)
        .alias("average_predicted_percent"),

        (
            pl.col("correct").mean() * 100
        )
        .round(1)
        .alias("actual_win_percent")

    )
)


# Put bands into sensible order
band_order = {
    "50-55%": 1,
    "55-60%": 2,
    "60-65%": 3,
    "65-70%": 4,
    "70-75%": 5,
    "75-80%": 6,
    "80-85%": 7,
    "85%+": 8
}

calibration = calibration.with_columns(

    pl.col("confidence_band")
    .replace_strict(band_order)
    .alias("order")

).sort("order")


print("\nPROBABILITY CALIBRATION")
print(calibration)


# --------------------------------------------------
# 8. MAKE OUR FIRST GRAPH
# --------------------------------------------------

x = calibration["average_predicted_percent"].to_list()
y = calibration["actual_win_percent"].to_list()

plt.figure(figsize=(8, 6))

plt.scatter(x, y)

plt.plot(
    [50, 95],
    [50, 95],
    linestyle="--"
)

plt.xlabel("Market Predicted Win Probability (%)")
plt.ylabel("Actual Win Rate (%)")

plt.title(
    "NFL Market Probability Calibration\n"
    "Regular Seasons 2021-2025"
)

plt.grid(True)

plt.savefig(
    "probability_calibration.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()