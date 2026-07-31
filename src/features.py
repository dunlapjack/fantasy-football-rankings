import polars as pl
import nflreadpy as nfl
from src.scoring import load_config, calculate_offensive_points

from pathlib import Path

from src.team_codes import normalize_team_column

CONFIG_PATH = Path(__file__).resolve().parent.parent / "league_config.json"

RECENCY_WEIGHTS = [0.5, 0.3, 0.2]  # most-recent-first; applies to any 3-year window
OFFENSE_POSITIONS = ["QB", "RB", "WR", "TE"]

RAW_STAT_COLUMNS = [
    "passing_yards", "passing_tds", "passing_interceptions",
    "rushing_yards", "rushing_tds",
    "receptions", "targets", "receiving_yards", "receiving_tds",
    "fantasy_points",
]


def load_veteran_stats(seasons):
    raw = nfl.load_player_stats(seasons)
    raw = raw.filter(pl.col("season_type") == "REG")
    raw = raw.filter(pl.col("position").is_in(OFFENSE_POSITIONS))
    raw = raw.filter(pl.col("player_id").is_not_null())

    config = load_config(CONFIG_PATH)
    raw = raw.with_columns(
        pl.struct(raw.columns)
        .map_elements(lambda row: calculate_offensive_points(row, config), return_dtype=pl.Float64)
        .alias("fantasy_points")
    )
    return raw


def aggregate_season_stats(raw_stats):
    grouped = (
        raw_stats.sort("week")
        .group_by(["player_id", "season"], maintain_order=True)
        .agg(
            [pl.col(c).sum().alias(c) for c in RAW_STAT_COLUMNS]
            + [
                pl.len().alias("games_played"),
                pl.col("player_display_name").last().alias("player_name"),
                pl.col("position").last().alias("position"),
            ]
        )
    )

    per_game_exprs = [
        (pl.col(c) / pl.col("games_played")).alias(f"{c}_per_game")
        for c in RAW_STAT_COLUMNS
    ]
    grouped = grouped.with_columns(per_game_exprs)

    keep_cols = ["player_id", "player_name", "position", "season", "games_played"] + \
                [f"{c}_per_game" for c in RAW_STAT_COLUMNS]
    return grouped.select(keep_cols)


def get_recency_weights(seasons_present):
    """
    Maps a list of seasons to weights based on recency RANK, not
    absolute season number -- most recent season present = 50%,
    middle = 30%, oldest = 20%. This lets the same 3-year weighting
    scheme apply to any historical window (e.g. "entering 2024" uses
    2021/2022/2023 the same way "entering 2026" uses 2023/2024/2025),
    which the Phase 5 backtest needs.
    """
    sorted_desc = sorted(seasons_present, reverse=True)
    return {s: w for s, w in zip(sorted_desc, RECENCY_WEIGHTS)}


def apply_season_weighting(season_stats):
    per_game_cols = [c for c in season_stats.columns if c.endswith("_per_game")]
    weighted_rows = []

    for group_key, player_df in season_stats.group_by("player_id"):
        player_id = group_key[0]
        seasons_present = player_df["season"].to_list()
        raw_weights = get_recency_weights(seasons_present)
        total_weight = sum(raw_weights.values())
        normalized_weights = {s: w / total_weight for s, w in raw_weights.items()}
        weighted_stats = {}
        for col in per_game_cols:
            weighted_stats[col] = sum(
                player_df.filter(pl.col("season") == s)[col][0] * normalized_weights[s]
                for s in seasons_present
            )

        most_recent = player_df.sort("season").tail(1)
        weighted_stats["player_id"] = player_id
        weighted_stats["player_name"] = most_recent["player_name"][0]
        weighted_stats["position"] = most_recent["position"][0]
        weighted_stats["games_played"] = player_df["games_played"].sum()

        weighted_rows.append(weighted_stats)

    return pl.DataFrame(weighted_rows)


def build_veteran_feature_table(seasons=[2023, 2024, 2025]):
    raw = load_veteran_stats(seasons)
    season_stats = aggregate_season_stats(raw)
    return apply_season_weighting(season_stats)

def attach_current_team(veteran_features):
    """
    Joins each player's most current team onto the veteran feature table
    using nflreadpy's load_players() `latest_team` column. This resolves
    the mid-season-trade issue from Phase 2 (89 players had multi-team
    stats in 2025) for free, since latest_team already reflects each
    player's most recent team rather than their first-seen team.
    """
    players = nfl.load_players().select(["gsis_id", "latest_team"]).rename(
        {"gsis_id": "player_id", "latest_team": "team"}
    )
    players = normalize_team_column(players)
    return veteran_features.join(players, on="player_id", how="left")