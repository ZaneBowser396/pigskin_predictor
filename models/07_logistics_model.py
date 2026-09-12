import nflreadpy as nfl
import polars as pl

from collections import defaultdict, deque

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ==================================================
# SETTINGS
# ==================================================

STARTING_ELO = 1500
K_FACTOR = 20
HOME_FIELD_ADVANTAGE = 55

START_SEASON = 2015

TRAIN_START = 2018
TRAIN_END = 2020

TEST_START = 2021
TEST_END = 2025

RECENT_GAMES = 5


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def elo_probability(rating_a, rating_b):

    return 1 / (
        1 + 10 ** ((rating_b - rating_a) / 400)
    )


def moneyline_probability(moneyline):

    if moneyline < 0:

        return (
            -moneyline
            / (-moneyline + 100)
        )

    return (
        100
        / (moneyline + 100)
    )


def average(values):

    if len(values) == 0:
        return 0

    return sum(values) / len(values)


# ==================================================
# LOAD DATA
# ==================================================

games = nfl.load_schedules(
    list(range(START_SEASON, TEST_END + 1))
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
    .sort(
        [
            "season",
            "week",
            "gameday",
            "gametime"
        ]
    )
)


# ==================================================
# STORAGE
# ==================================================

ratings = {}

recent_margins = defaultdict(
    lambda: deque(maxlen=RECENT_GAMES)
)

rows = []

previous_season = None


# ==================================================
# BUILD PRE-GAME FEATURES
# ==================================================

for game in games.iter_rows(named=True):

    season = game["season"]

    home_team = game["home_team"]
    away_team = game["away_team"]

    home_score = game["home_score"]
    away_score = game["away_score"]


    # ----------------------------------------------
    # OFFSEASON
    # ----------------------------------------------

    if (
        previous_season is not None
        and season != previous_season
    ):

        # Regress Elo toward average
        for team in ratings:

            ratings[team] = (
                STARTING_ELO
                + 0.75
                * (
                    ratings[team]
                    - STARTING_ELO
                )
            )

        # Old recent-form data is much less useful
        # after an offseason
        recent_margins.clear()


    previous_season = season


    # ----------------------------------------------
    # INITIALISE TEAMS
    # ----------------------------------------------

    if home_team not in ratings:
        ratings[home_team] = STARTING_ELO

    if away_team not in ratings:
        ratings[away_team] = STARTING_ELO


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


    # ----------------------------------------------
    # ELO PROBABILITY
    # ----------------------------------------------

    home_elo = ratings[home_team]
    away_elo = ratings[away_team]


    elo_home_probability = elo_probability(

        home_elo + HOME_FIELD_ADVANTAGE,

        away_elo

    )


    # ----------------------------------------------
    # RECENT FORM
    # ----------------------------------------------

    home_recent_margin = average(
        recent_margins[home_team]
    )

    away_recent_margin = average(
        recent_margins[away_team]
    )


    recent_margin_difference = (
        home_recent_margin
        - away_recent_margin
    )


    # ----------------------------------------------
    # REST ADVANTAGE
    # ----------------------------------------------

    home_rest = game["home_rest"]
    away_rest = game["away_rest"]

    if home_rest is None:
        home_rest = 7

    if away_rest is None:
        away_rest = 7


    rest_difference = (
        home_rest
        - away_rest
    )


    # ----------------------------------------------
    # ACTUAL RESULT
    # ----------------------------------------------

    if home_score > away_score:

        home_win = 1
        actual_home_result = 1.0
        winner = home_team

    elif away_score > home_score:

        home_win = 0
        actual_home_result = 0.0
        winner = away_team

    else:

        # Ignore ties when training
        home_win = None
        actual_home_result = 0.5
        winner = None


    # ----------------------------------------------
    # SAVE FEATURES BEFORE UPDATING TEAM DATA
    # ----------------------------------------------

    if season >= TRAIN_START:

        rows.append({

            "season": season,

            "week": game["week"],

            "home_team": home_team,
            "away_team": away_team,

            "winner": winner,

            "home_win": home_win,

            "market_home_probability":
                market_home_probability,

            "elo_home_probability":
                elo_home_probability,

            "recent_margin_difference":
                recent_margin_difference,

            "rest_difference":
                rest_difference

        })


    # ----------------------------------------------
    # NOW UPDATE ELO
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


    # ----------------------------------------------
    # NOW UPDATE RECENT PERFORMANCE
    # ----------------------------------------------

    home_margin = (
        home_score - away_score
    )

    away_margin = (
        away_score - home_score
    )


    recent_margins[home_team].append(
        home_margin
    )

    recent_margins[away_team].append(
        away_margin
    )


# ==================================================
# DATAFRAME
# ==================================================

data = pl.DataFrame(rows)

data = data.filter(
    pl.col("home_win").is_not_null()
)


# ==================================================
# FEATURES
# ==================================================

feature_columns = [

    "market_home_probability",

    "elo_home_probability",

    "recent_margin_difference",

    "rest_difference"

]


# ==================================================
# TRAIN / TEST SPLIT
# ==================================================

train = data.filter(

    (pl.col("season") >= TRAIN_START)
    & (pl.col("season") <= TRAIN_END)

)


test = data.filter(

    (pl.col("season") >= TEST_START)
    & (pl.col("season") <= TEST_END)

)


X_train = train.select(
    feature_columns
).to_numpy()

y_train = train[
    "home_win"
].to_numpy()


X_test = test.select(
    feature_columns
).to_numpy()


# ==================================================
# TRAIN LOGISTIC REGRESSION
# ==================================================

model = make_pipeline(

    StandardScaler(),

    LogisticRegression(
        max_iter=1000
    )

)


model.fit(
    X_train,
    y_train
)


# ==================================================
# PREDICT TEST DATA
# ==================================================

home_probabilities = (
    model.predict_proba(X_test)[:, 1]
)


test = test.with_columns(

    pl.Series(
        "model_home_probability",
        home_probabilities
    )

)


# ==================================================
# MODEL PICK
# ==================================================

test = test.with_columns(

    pl.when(
        pl.col("model_home_probability")
        >= 0.5
    )
    .then(pl.col("home_team"))
    .otherwise(pl.col("away_team"))
    .alias("model_pick"),


    pl.max_horizontal(

        pl.col("model_home_probability"),

        1 - pl.col("model_home_probability")

    ).alias("model_confidence")

)


# ==================================================
# MARKET PICK
# ==================================================

test = test.with_columns(

    pl.when(
        pl.col("market_home_probability")
        >= 0.5
    )
    .then(pl.col("home_team"))
    .otherwise(pl.col("away_team"))
    .alias("market_pick"),


    pl.max_horizontal(

        pl.col("market_home_probability"),

        1 - pl.col("market_home_probability")

    ).alias("market_confidence")

)


# ==================================================
# CORRECT PICKS
# ==================================================

test = test.with_columns(

    (
        pl.col("model_pick")
        == pl.col("winner")
    ).alias("model_correct"),


    (
        pl.col("market_pick")
        == pl.col("winner")
    ).alias("market_correct")

)


# ==================================================
# CONFIDENCE WEIGHTS
# ==================================================

test = test.with_columns(

    pl.col("model_confidence")
    .rank(method="ordinal")
    .over(["season", "week"])
    .cast(pl.Int32)
    .alias("model_weight"),


    pl.col("market_confidence")
    .rank(method="ordinal")
    .over(["season", "week"])
    .cast(pl.Int32)
    .alias("market_weight")

)


# ==================================================
# POOL POINTS
# ==================================================

test = test.with_columns(

    pl.when(
        pl.col("model_correct")
    )
    .then(pl.col("model_weight"))
    .otherwise(0)
    .alias("model_points"),


    pl.when(
        pl.col("market_correct")
    )
    .then(pl.col("market_weight"))
    .otherwise(0)
    .alias("market_points")

)


# ==================================================
# OVERALL COMPARISON
# ==================================================

comparison = test.select(

    (
        pl.col("market_correct")
        .mean() * 100
    )
    .round(2)
    .alias("market_accuracy"),

    (
        pl.col("model_correct")
        .mean() * 100
    )
    .round(2)
    .alias("model_accuracy"),

    (
        pl.col("market_points").sum()
        / pl.col("market_weight").sum()
        * 100
    )
    .round(2)
    .alias("market_points_percent"),

    (
        pl.col("model_points").sum()
        / pl.col("model_weight").sum()
        * 100
    )
    .round(2)
    .alias("model_points_percent")

)


print("\nMODEL 3 — LOGISTIC REGRESSION")

print(comparison)


# ==================================================
# SEASON-BY-SEASON
# ==================================================

season_results = (

    test

    .group_by("season")

    .agg(

        (
            pl.col("market_correct")
            .mean() * 100
        )
        .round(1)
        .alias("market_accuracy"),

        (
            pl.col("model_correct")
            .mean() * 100
        )
        .round(1)
        .alias("model_accuracy"),

        (
            pl.col("market_points").sum()
            / pl.col("market_weight").sum()
            * 100
        )
        .round(1)
        .alias("market_points"),

        (
            pl.col("model_points").sum()
            / pl.col("model_weight").sum()
            * 100
        )
        .round(1)
        .alias("model_points")

    )

    .sort("season")

)


print("\nBY SEASON")

print(season_results)

# ==================================================
# MODEL COEFFICIENTS
# ==================================================

logistic_model = model.named_steps["logisticregression"]

coefficients = pl.DataFrame({

    "feature": feature_columns,

    "coefficient":
        logistic_model.coef_[0].tolist()

})

coefficients = (

    coefficients

    .with_columns(

        pl.col("coefficient")
        .abs()
        .alias("absolute_importance")

    )

    .sort(
        "absolute_importance",
        descending=True
    )

)

print("\nMODEL COEFFICIENTS")
print(coefficients)

# ==================================================
# MARKET VS MODEL DISAGREEMENTS
# ==================================================

disagreements = (

    test

    .filter(

        pl.col("model_pick")
        != pl.col("market_pick")

    )

    .with_columns(

        (
            pl.col("model_home_probability")
            - pl.col("market_home_probability")
        )
        .abs()
        .alias("probability_gap")

    )

)


disagreement_summary = disagreements.select(

    pl.len()
    .alias("games"),

    (
        pl.col("model_correct")
        .mean() * 100
    )
    .round(1)
    .alias("model_accuracy"),

    (
        pl.col("market_correct")
        .mean() * 100
    )
    .round(1)
    .alias("market_accuracy")

)


print("\nWHEN MODEL AND MARKET DISAGREE")
print(disagreement_summary)

# ==================================================
# DISAGREEMENTS BY STRENGTH
# ==================================================

disagreements = disagreements.with_columns(

    pl.when(
        pl.col("probability_gap") < 0.05
    )
    .then(pl.lit("<5%"))

    .when(
        pl.col("probability_gap") < 0.10
    )
    .then(pl.lit("5-10%"))

    .when(
        pl.col("probability_gap") < 0.15
    )
    .then(pl.lit("10-15%"))

    .otherwise(
        pl.lit("15%+")
    )

    .alias("gap_band")

)


gap_results = (

    disagreements

    .group_by("gap_band")

    .agg(

        pl.len()
        .alias("games"),

        (
            pl.col("model_correct")
            .mean() * 100
        )
        .round(1)
        .alias("model_accuracy"),

        (
            pl.col("market_correct")
            .mean() * 100
        )
        .round(1)
        .alias("market_accuracy")

    )

)


gap_order = {

    "<5%": 1,
    "5-10%": 2,
    "10-15%": 3,
    "15%+": 4

}


gap_results = (

    gap_results

    .with_columns(

        pl.col("gap_band")
        .replace_strict(gap_order)
        .alias("order")

    )

    .sort("order")

)


print("\nDISAGREEMENTS BY PROBABILITY GAP")
print(gap_results)