import polars as pl

# load_players()'s latest_team uses a few nonstandard abbreviations that
# don't match team_stats/snap_counts/depth_charts. Add more here if
# other mismatches turn up later.
TEAM_ABBR_FIXES = {
    "AZ": "ARI",
    "ARZ": "ARI",
    "LAR": "LA",
}


# Franchises that changed city (and therefore code) inside the seasons
# this project reads. Maps the OLD code to the code the same franchise
# uses afterwards.
#
# Needed because anything comparing a team to its own prior season has
# to follow the franchise, not the abbreviation. The Chargers' 2017 row
# is "LAC" and their 2016 row is "SD", so a naive (season - 1, team)
# lookup finds nothing and silently returns null -- which is how
# playcaller_history's derived coaching-change flag came up 31 of 32
# for 2017 and 2020 instead of 32 of 32.
FRANCHISE_PREDECESSORS = {
    "LAC": "SD",    # San Diego -> Los Angeles, 2017
    "LV": "OAK",    # Oakland -> Las Vegas, 2020
    "LA": "STL",    # St. Louis -> Los Angeles, 2016
}

# Same relation the other way, for building a stable franchise key.
FRANCHISE_SUCCESSORS = {old: new for new, old in FRANCHISE_PREDECESSORS.items()}


def normalize_team_column(df, column="team"):
    return df.with_columns(pl.col(column).replace(TEAM_ABBR_FIXES).alias(column))


def franchise_key_column(df, column="team", alias="franchise"):
    """
    Adds a column holding one stable identifier per franchise across
    relocations, so year-over-year comparisons don't break at a move.
    Call after normalize_team_column.
    """
    return df.with_columns(
        pl.col(column).replace(FRANCHISE_SUCCESSORS).alias(alias)
    )