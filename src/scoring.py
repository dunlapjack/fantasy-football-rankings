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