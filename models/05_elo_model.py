import nflreadpy as nfl
import polars as pl


# ==================================================
# SETTINGS
# ==================================================

STARTING_ELO = 1500
K_FACTOR = 20
HOME_FIELD_ADVANTAGE = 55

# We load earlier seasons to give Elo time to learn
# team strength before our real 2021-2025 test starts.
TRAINING_START = 2018
TEST_START = 2021
TEST_END = 2025


# ==================================================
# 1. LOAD GAMES
# ==================================================

games = nfl.load_schedules(
    list(range(TRAINING_START, TEST_END + 1))
)

games = (
    games
    .filter(
        (pl.col("game_type") == "REG")
        & pl.col("home_score").is_not_null()
        & pl.col("away_score").is_not_null()
    )
    .sort(["season", "week", "gameday", "gametime"])
)


# ==================================================
# 2. ELO HELPER FUNCTION
# ==================================================

def expected_score(rating_a, rating_b):
    """
    Calculate Team A's expected probability of winning.
    """

    return 1 / (
        1 + 10 ** ((rating_b - rating_a) / 400)
    )


# ==================================================
# 3. CREATE TEAM RATINGS
# ==================================================

ratings = {}

predictions = []

previous_season = None


# ==================================================
# 4. PROCESS EACH GAME IN CHRONOLOGICAL ORDER
# ==================================================

for game in games.iter_rows(named=True):

    season = game["season"]

    home_team = game["home_team"]
    away_team = game["away_team"]

    home_score = game["home_score"]
    away_score = game["away_score"]


    # ----------------------------------------------
    # NEW SEASON
    # ----------------------------------------------

    if previous_season is not None and season != previous_season:

        # Regress ratings slightly toward average.
        #
        # This reflects offseason uncertainty:
        # coaches change, players move, teams improve,
        # teams decline, etc.

        for team in ratings:

            ratings[team] = (
                STARTING_ELO
                + 0.75
                * (ratings[team] - STARTING_ELO)
            )


    previous_season = season


    # ----------------------------------------------
    # ADD NEW TEAMS IF NEEDED
    # ----------------------------------------------

    if home_team not in ratings:
        ratings[home_team] = STARTING_ELO

    if away_team not in ratings:
        ratings[away_team] = STARTING_ELO


    # ----------------------------------------------
    # RATINGS BEFORE THIS GAME
    # ----------------------------------------------

    home_elo = ratings[home_team]
    away_elo = ratings[away_team]


    # Give the home team a small advantage
    adjusted_home_elo = (
        home_elo + HOME_FIELD_ADVANTAGE
    )


    # ----------------------------------------------
    # PREDICT BEFORE SEEING RESULT
    # ----------------------------------------------

    home_probability = expected_score(
        adjusted_home_elo,
        away_elo
    )

    away_probability = 1 - home_probability


    if home_probability >= away_probability:

        elo_pick = home_team
        elo_confidence = home_probability

    else:

        elo_pick = away_team
        elo_confidence = away_probability


    # ----------------------------------------------
    # DETERMINE ACTUAL RESULT
    # ----------------------------------------------

    if home_score > away_score:

        actual_home_result = 1.0
        winner = home_team

    elif away_score > home_score:

        actual_home_result = 0.0
        winner = away_team

    else:

        actual_home_result = 0.5
        winner = None


    # ----------------------------------------------
    # SAVE TEST-SEASON PREDICTION
    # ----------------------------------------------

    if TEST_START <= season <= TEST_END:

        predictions.append({

            "season": season,

            "week": game["week"],

            "away_team": away_team,
            "home_team": home_team,

            "away_score": away_score,
            "home_score": home_score,

            "away_elo": round(away_elo, 1),
            "home_elo": round(home_elo, 1),

            "elo_pick": elo_pick,

            "elo_confidence": elo_confidence,

            "winner": winner,

            "correct": elo_pick == winner

        })


    # ----------------------------------------------
    # UPDATE ELO AFTER RESULT
    # ----------------------------------------------

    expected_home = home_probability

    change = (
        K_FACTOR
        * (actual_home_result - expected_home)
    )

    ratings[home_team] += change
    ratings[away_team] -= change


# ==================================================
# 5. TURN OUR RESULTS INTO A POLARS DATAFRAME
# ==================================================

results = pl.DataFrame(predictions)


# ==================================================
# 6. ASSIGN OFFICE POOL CONFIDENCE WEIGHTS
# ==================================================

results = results.with_columns(

    pl.col("elo_confidence")
    .rank(method="ordinal")
    .over(["season", "week"])
    .cast(pl.Int32)
    .alias("weight")

)


# ==================================================
# 7. CALCULATE CONFIDENCE POINTS
# ==================================================

results = results.with_columns(

    pl.when(pl.col("correct"))
    .then(pl.col("weight"))
    .otherwise(0)
    .alias("confidence_points")

)


# ==================================================
# 8. SEASON RESULTS
# ==================================================

season_results = (

    results
    .group_by("season")
    .agg(

        pl.len()
        .alias("games"),

        pl.col("correct")
        .sum()
        .alias("correct_picks"),

        pl.col("confidence_points")
        .sum()
        .alias("confidence_points"),

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
            pl.col("confidence_points")
            / pl.col("max_points")
            * 100
        )
        .round(1)
        .alias("points_percent")

    )

    .sort("season")
)


print("\nELO SEASON RESULTS")
print(season_results)


# ==================================================
# 9. OVERALL RESULT
# ==================================================

overall = results.select(

    pl.len()
    .alias("games"),

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
        pl.col("correct").mean()
        * 100
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


print("\nELO OVERALL")
print(overall)


# ==================================================
# 10. CURRENT TEAM RATINGS
# ==================================================

current_ratings = (

    pl.DataFrame({

        "team": list(ratings.keys()),

        "elo": [
            round(rating, 1)
            for rating in ratings.values()
        ]

    })

    .sort("elo", descending=True)

)


print("\nFINAL ELO RATINGS")
print(current_ratings)