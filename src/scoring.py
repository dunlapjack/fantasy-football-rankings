import json

def load_config(path="league_config.json"):
    with open(path) as f:
        return json.load(f)

def calculate_offensive_points(stat_row, config):
    scoring = config["scoring"]
    points = 0.0
    points += stat_row["passing_yards"] * scoring["pass_yard"]
    points += stat_row["passing_tds"] * scoring["pass_td"]
    points += stat_row["passing_interceptions"] * scoring["interception"]
    points += stat_row["rushing_yards"] * scoring["rush_yard"]
    points += stat_row["rushing_tds"] * scoring["rush_td"]
    points += stat_row["receptions"] * scoring["reception"]
    points += stat_row["receiving_yards"] * scoring["rec_yard"]
    points += stat_row["receiving_tds"] * scoring["rec_td"]

    fumbles_lost = (
        stat_row["rushing_fumbles_lost"]
        + stat_row["receiving_fumbles_lost"]
        + stat_row["sack_fumbles_lost"]
    )
    points += fumbles_lost * scoring["fumble_lost"]

    two_pt = (
        stat_row["passing_2pt_conversions"]
        + stat_row["rushing_2pt_conversions"]
        + stat_row["receiving_2pt_conversions"]
    )
    points += two_pt * scoring["two_point_conversion"]
    return round(points, 2)

def points_allowed_score(points_allowed, config):
    bands = config["scoring"]["defense_points_allowed"]
    if points_allowed == 0: return bands["0"]
    elif 1 <= points_allowed <= 6: return bands["1-6"]
    elif 7 <= points_allowed <= 13: return bands["7-13"]
    else: return 0

def yards_allowed_score(yards_allowed, config):
    bands = config["scoring"]["defense_yards_allowed"]
    if yards_allowed <= 99: return bands["0-99"]
    elif yards_allowed <= 199: return bands["100-199"]
    elif yards_allowed <= 299: return bands["200-299"]
    else: return 0

def calculate_dst_points(team_game_row, config):
    scoring = config["scoring"]
    points = 0.0
    points += team_game_row["def_sacks"] * scoring["defense_sack"]
    points += team_game_row["def_interceptions"] * scoring["defense_int"]
    points += team_game_row["fumble_recoveries"] * scoring["defense_fumble_recovery"]
    points += team_game_row["defensive_tds"] * scoring["defense_td"]
    points += points_allowed_score(team_game_row["points_allowed"], config)
    points += yards_allowed_score(team_game_row["yards_allowed"], config)
    return round(points, 2)