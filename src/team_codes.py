import polars as pl

# load_players()'s latest_team uses a few nonstandard abbreviations that
# don't match team_stats/snap_counts/depth_charts. Add more here if
# other mismatches turn up later.
TEAM_ABBR_FIXES = {
    "AZ": "ARI",
}


def normalize_team_column(df, column="team"):
    return df.with_columns(pl.col(column).replace(TEAM_ABBR_FIXES).alias(column))