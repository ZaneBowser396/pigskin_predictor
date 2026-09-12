import nflreadpy as nfl
import polars as pl


# ==================================================
# SETTINGS
# ==================================================

STARTING_ELO = 1500
K_FACTOR = 20
HOME_FIELD_ADVANTAGE = 55

# 2015-2017 warms up Elo
# 2018-2020 chooses our market/Elo blend
# 2021-2025 is untouched test data

WARMUP_START = 2015
VALIDATION_START = 2018
VALIDATION_END = 2020
TEST_START = 2021
TEST_END = 2025


# ==================================================
# 1. LOAD DATA
# ==================================================

games = nfl.load_schedules(
    list(range(WARMUP_START, TEST_END + 1))
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


# ==================================================
# 2. HELPER FUNCTIONS
# ==================================================

def expected_score(rating_a, rating_b):

    return 1 / (
        1 + 10 ** ((rating_b - rating_a) / 400)
    )


def moneyline_probability(moneyline):

    if moneyline < 0:

        return (
            -moneyline
            / (-moneyline + 100)
        )

    else:

        return (
            100
            / (moneyline + 100)
        )


# ==================================================
# 3. RUN ELO THROUGH HISTORY
# ==================================================

ratings = {}

predictions = []

previous_season = None


for game in games.iter_rows(named=True):

    season = game["season"]

    home_team = game["home_team"]
    away_team = game["away_team"]

    home_score = game["home_score"]
    away_score = game["away_score"]


    # ----------------------------------------------
    # OFFSEASON REGRESSION
    # ----------------------------------------------

    if (
        previous_season is not None
        and season != previous_season
    ):

        for team in ratings:

            ratings[team] = (
                STARTING_ELO
                + 0.75
                * (ratings[team] - STARTING_ELO)
            )


    previous_season = season


    # ----------------------------------------------
    # NEW TEAMS
    # ----------------------------------------------

    if home_team not in ratings:
        ratings[home_team] = STARTING_ELO

    if away_team not in ratings:
        ratings[away_team] = STARTING_ELO


    home_elo = ratings[home_team]
    away_elo = ratings[away_team]


    # ----------------------------------------------
    # ELO PROBABILITY
    # ----------------------------------------------

    elo_home_probability = expected_score(
        home_elo + HOME_FIELD_ADVANTAGE,
        away_elo
    )

    elo_away_probability = (
        1 - elo_home_probability
    )


    # ----------------------------------------------
    # MARKET PROBABILITY
    # ----------------------------------------------

    home_raw = moneyline_probability(
        game["home_moneyline"]
    )

    away_raw = moneyline_probability(
        game["away_moneyline"]
    )


    market_home_probability = (
        home_raw
        / (home_raw + away_raw)
    )

    market_away_probability = (
        away_raw
        / (home_raw + away_raw)
    )


    # ----------------------------------------------
    # ACTUAL WINNER
    # ----------------------------------------------

    if home_score > away_score:

        winner = home_team
        actual_home_result = 1.0

    elif away_score > home_score:

        winner = away_team
        actual_home_result = 0.0

    else:

        winner = None
        actual_home_result = 0.5


    # ----------------------------------------------
    # SAVE PREDICTIONS BEFORE UPDATE
    # ----------------------------------------------

    if season >= VALIDATION_START:

        predictions.append({

            "season": season,

            "week": game["week"],

            "home_team": home_team,
            "away_team": away_team,

            "winner": winner,

            "market_home_probability":
                market_home_probability,

            "elo_home_probability":
                elo_home_probability

        })


    # ----------------------------------------------
    # UPDATE ELO
    # ----------------------------------------------

    change = (
        K_FACTOR
        * (
            actual_home_result
            - elo_home_probability
        )
    )

    ratings[home_team] += change
    ratings[away_team] -= change


# ==================================================
# 4. CREATE DATAFRAME
# ==================================================

results = pl.DataFrame(predictions)


# ==================================================
# 5. FUNCTION TO TEST A BLEND
# ==================================================

def evaluate_model(data, market_weight):

    elo_weight = 1 - market_weight


    test = data.with_columns(

        (
            pl.col("market_home_probability")
            * market_weight

            +

            pl.col("elo_home_probability")
            * elo_weight

        ).alias("home_probability")

    )


    test = test.with_columns(

        pl.when(
            pl.col("home_probability") >= 0.5
        )
        .then(pl.col("home_team"))
        .otherwise(pl.col("away_team"))
        .alias("pick"),


        pl.max_horizontal(

            pl.col("home_probability"),

            1 - pl.col("home_probability")

        ).alias("confidence")

    )


    test = test.with_columns(

        (
            pl.col("pick")
            == pl.col("winner")
        ).alias("correct")

    )


    # Assign OfficePoolStop confidence weights

    test = test.with_columns(

        pl.col("confidence")
        .rank(method="ordinal")
        .over(["season", "week"])
        .cast(pl.Int32)
        .alias("weight")

    )


    test = test.with_columns(

        pl.when(pl.col("correct"))
        .then(pl.col("weight"))
        .otherwise(0)
        .alias("confidence_points")

    )


    summary = test.select(

        (
            pl.col("correct").mean()
            * 100
        )
        .alias("accuracy"),

        (
            pl.col("confidence_points").sum()
            / pl.col("weight").sum()
            * 100
        )
        .alias("points_percent")

    )


    return (
        summary["accuracy"][0],
        summary["points_percent"][0]
    )


# ==================================================
# 6. VALIDATION DATA
# ==================================================

validation = results.filter(

    (pl.col("season") >= VALIDATION_START)
    & (pl.col("season") <= VALIDATION_END)

)


# ==================================================
# 7. TRY DIFFERENT MARKET WEIGHTS
# ==================================================

weight_results = []


for market_weight_percent in range(0, 101, 5):

    market_weight = (
        market_weight_percent / 100
    )


    accuracy, points_percent = evaluate_model(

        validation,
        market_weight

    )


    weight_results.append({

        "market_weight":
            market_weight_percent,

        "elo_weight":
            100 - market_weight_percent,

        "accuracy_percent":
            round(accuracy, 2),

        "points_percent":
            round(points_percent, 2)

    })


weight_table = pl.DataFrame(weight_results)


print("\nVALIDATION RESULTS: 2018-2020")

print(
    weight_table.sort(
        "points_percent",
        descending=True
    )
)


# ==================================================
# 8. CHOOSE BEST BLEND
# ==================================================

best = (
    weight_table
    .sort(
        "points_percent",
        descending=True
    )
    .row(0, named=True)
)


best_market_weight = (
    best["market_weight"] / 100
)


print("\nBEST VALIDATION BLEND")

print(
    f"Market: {best['market_weight']}%"
)

print(
    f"Elo: {best['elo_weight']}%"
)


# ==================================================
# 9. TEST ON UNSEEN 2021-2025 DATA
# ==================================================

test_data = results.filter(

    (pl.col("season") >= TEST_START)
    & (pl.col("season") <= TEST_END)

)


test_accuracy, test_points = evaluate_model(

    test_data,
    best_market_weight

)


print("\nUNSEEN TEST RESULTS: 2021-2025")

print(
    f"Accuracy: {test_accuracy:.1f}%"
)

print(
    f"Confidence points: {test_points:.1f}%"
)


# ==================================================
# 10. COMPARE PURE MARKET AND PURE ELO
# ==================================================

market_accuracy, market_points = evaluate_model(

    test_data,
    1.0

)


elo_accuracy, elo_points = evaluate_model(

    test_data,
    0.0

)


comparison = pl.DataFrame({

    "model": [
        "Market",
        "Elo",
        "Optimised Blend"
    ],

    "accuracy_percent": [

        round(market_accuracy, 1),

        round(elo_accuracy, 1),

        round(test_accuracy, 1)

    ],

    "points_percent": [

        round(market_points, 1),

        round(elo_points, 1),

        round(test_points, 1)

    ]

})


print("\nMODEL COMPARISON")

print(comparison)
