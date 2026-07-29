from pathlib import Path
import polars as pl
import nflreadpy as nfl
from src.scoring import load_config, calculate_offensive_points
from src.team_codes import normalize_team_column

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "league_config.json"

OFFENSE_POSITIONS = ["QB", "RB", "WR", "TE"]
COHORT_SEASONS = [2021, 2022, 2023, 2024, 2025]
MIN_CONFIDENT_N = 5
CURRENT_ROOKIE_SEASON = 2026

QB_BASELINE_PATH = PROJECT_ROOT / "data" / "rookie_qb_baselines.csv"
SKILL_BASELINE_PATH = PROJECT_ROOT / "data" / "rookie_skill_baselines.csv"


# ===== Part A: historical cohort baselines =====

def load_rookie_season_stats(seasons=COHORT_SEASONS):
    draft_picks = nfl.load_draft_picks()
    player_stats = nfl.load_player_stats(seasons)

    rookie_stats = player_stats.join(
        draft_picks.select(["gsis_id", "season", "round"]),
        left_on="player_id", right_on="gsis_id",
    ).filter(pl.col("season") == pl.col("season_right"))

    rookie_stats = rookie_stats.filter(pl.col("position").is_in(OFFENSE_POSITIONS))

    config = load_config(CONFIG_PATH)
    rookie_stats = rookie_stats.with_columns(
        pl.struct(rookie_stats.columns)
        .map_elements(lambda row: calculate_offensive_points(row, config), return_dtype=pl.Float64)
        .alias("fantasy_points")
    )
    return rookie_stats


def aggregate_rookie_season(rookie_stats):
    return (
        rookie_stats.sort("week")
        .group_by(["player_id", "season", "round", "position", "team"], maintain_order=True)
        .agg([
            pl.col("fantasy_points").sum().alias("total_points"),
            pl.len().alias("games_played"),
        ])
        .with_columns(
            (pl.col("total_points") / pl.col("games_played")).alias("points_per_game")
        )
    )


def compute_primary_qb_by_team_season(seasons=COHORT_SEASONS):
    player_stats = nfl.load_player_stats(seasons)
    return (
        player_stats.filter((pl.col("position") == "QB") & (pl.col("season_type") == "REG"))
        .group_by(["team", "season", "player_id"])
        .agg(pl.col("attempts").sum().alias("attempts"))
        .sort("attempts", descending=True)
        .group_by(["team", "season"])
        .first()
        .select(["team", "season", "player_id"])
        .rename({"player_id": "primary_qb_id"})
    )


def flag_qb_starters(rookie_season_agg, seasons=COHORT_SEASONS):
    primary_qb = compute_primary_qb_by_team_season(seasons)
    return (
        rookie_season_agg.join(primary_qb, on=["team", "season"], how="left")
        .with_columns(
            (pl.col("player_id") == pl.col("primary_qb_id")).fill_null(False).alias("is_starter")
        )
        .drop("primary_qb_id")
    )


def build_rookie_cohort_baselines(seasons=COHORT_SEASONS):
    rookie_stats = load_rookie_season_stats(seasons)
    aggregated = aggregate_rookie_season(rookie_stats)
    flagged = flag_qb_starters(aggregated, seasons)

    flagged = flagged.with_columns(
        pl.when(pl.col("round") == 1).then(pl.lit("Round 1"))
        .when(pl.col("round").is_in([2, 3])).then(pl.lit("Day 2"))
        .otherwise(pl.lit("Day 3"))
        .alias("round_bucket")
    )

    qb_baselines = (
        flagged.filter(pl.col("position") == "QB")
        .group_by(["round_bucket", "is_starter"])
        .agg([
            pl.col("points_per_game").mean().alias("avg_points_per_game"),
            pl.len().alias("num_players"),
        ])
        .with_columns((pl.col("num_players") < MIN_CONFIDENT_N).alias("low_confidence"))
        .sort(["round_bucket", "is_starter"])
    )

    skill_baselines = (
        flagged.filter(pl.col("position") != "QB")
        .group_by(["position", "round"])
        .agg([
            pl.col("points_per_game").mean().alias("avg_points_per_game"),
            pl.len().alias("num_players"),
        ])
        .with_columns((pl.col("num_players") < MIN_CONFIDENT_N).alias("low_confidence"))
        .sort(["position", "round"])
    )

    return qb_baselines, skill_baselines


def save_rookie_cohort_baselines():
    qb_baselines, skill_baselines = build_rookie_cohort_baselines()

    qb_path = PROJECT_ROOT / "data" / "rookie_qb_baselines.csv"
    skill_path = PROJECT_ROOT / "data" / "rookie_skill_baselines.csv"
    qb_path.parent.mkdir(parents=True, exist_ok=True)

    qb_baselines.write_csv(qb_path)
    skill_baselines.write_csv(skill_path)

    print(f"QB baselines ({qb_baselines.shape[0]} rows) written to {qb_path}")
    print(f"Skill baselines ({skill_baselines.shape[0]} rows) written to {skill_path}")


# ===== Part B: applying baselines to the current rookie class =====

def get_current_rookie_class(season=CURRENT_ROOKIE_SEASON):
    """
    Identifies this year's rookie class (drafted + undrafted) using
    load_players()'s rookie_season. Draft round is matched two ways:
    first via pfr_id (most reliable early in the offseason), then a
    name+position fallback for anyone pfr_id misses -- gsis_id/pfr_id
    crosswalks aren't fully populated yet for a brand-new draft class.
    Remaining nulls are genuine undrafted free agents.
    """
    players = nfl.load_players().filter(
        (pl.col("rookie_season") == season) &
        (pl.col("position").is_in(OFFENSE_POSITIONS))
    ).select([
        pl.col("gsis_id").alias("player_id"),
        pl.col("display_name").alias("player_name"),
        "position",
        pl.col("latest_team").alias("team"),
        "pfr_id",
    ])

    draft_picks = nfl.load_draft_picks().filter(pl.col("season") == season)

    draft_by_id = draft_picks.select([
        pl.col("pfr_player_id").alias("pfr_id"),
        pl.col("round").alias("round_by_id"),
    ])
    draft_by_name = draft_picks.select([
        pl.col("pfr_player_name").alias("player_name"),
        "position",
        pl.col("round").alias("round_by_name"),
    ])

    rookie_class = (
        players.join(draft_by_id, on="pfr_id", how="left")
        .join(draft_by_name, on=["player_name", "position"], how="left")
        .with_columns(
            pl.coalesce(["round_by_id", "round_by_name"]).alias("round")
        )
        .drop(["pfr_id", "round_by_id", "round_by_name"])
    )

    return normalize_team_column(rookie_class)


def assign_round_bucket(rookie_class):
    """
    Round 1 / Day 2 (rounds 2-3) / Day 3 (rounds 4-7). Undrafted players
    (null round) fall into Day 3 as the floor.
    """
    return rookie_class.with_columns(
        pl.when(pl.col("round") == 1).then(pl.lit("Round 1"))
        .when(pl.col("round").is_in([2, 3])).then(pl.lit("Day 2"))
        .otherwise(pl.lit("Day 3"))
        .alias("round_bucket")
    )


def get_latest_depth_chart(season=CURRENT_ROOKIE_SEASON):
    """
    load_depth_charts() stores a new snapshot row every time it's
    scraped, so this keeps only the most recent snapshot per team. It
    also filters to offensive positions (QB/RB/WR/TE) only -- a single
    player can appear multiple times in the same snapshot under
    different pos_abb values for special-teams roles (e.g. a WR who
    also returns kicks/punts shows up as WR, KR, and PR). Without this
    filter those extra rows fan out through every downstream join that
    keys on player_id, compounding at each stage.
    """
    depth_charts = nfl.load_depth_charts(seasons=[season])
    latest_dt = depth_charts.group_by("team").agg(pl.col("dt").max().alias("latest_dt"))
    latest = (
        depth_charts.join(latest_dt, on="team")
        .filter(pl.col("dt") == pl.col("latest_dt"))
        .filter(pl.col("pos_abb").is_in(OFFENSE_POSITIONS))
        .select([pl.col("gsis_id").alias("player_id"), "pos_rank"])
        .unique(subset=["player_id"], keep="first")
    )
    return latest


def flag_current_starters(rookie_class):
    """
    is_starter = True if the rookie currently sits at pos_rank 1 on
    their team's depth chart.
    """
    latest_depth_chart = get_latest_depth_chart()
    joined = rookie_class.join(latest_depth_chart, on="player_id", how="left")
    return joined.with_columns(
        (pl.col("pos_rank") == 1).fill_null(False).alias("is_starter")
    )


def apply_rookie_baselines(rookie_class, qb_baselines, skill_baselines):
    """
    QBs matched on (round_bucket, is_starter). RB/WR/TE matched on
    (position, round), with undrafted players floored to round 7.
    """
    qb_rookies = (
        rookie_class.filter(pl.col("position") == "QB")
        .join(
            qb_baselines.select(["round_bucket", "is_starter", "avg_points_per_game", "low_confidence"]),
            on=["round_bucket", "is_starter"],
            how="left",
        )
    )

    skill_rookies = (
        rookie_class.filter(pl.col("position") != "QB")
        .with_columns(pl.col("round").fill_null(7).alias("round_for_lookup"))
        .join(
            skill_baselines.select(["position", "round", "avg_points_per_game", "low_confidence"])
            .rename({"round": "round_for_lookup"}),
            on=["position", "round_for_lookup"],
            how="left",
        )
        .drop("round_for_lookup")
    )

    combined = pl.concat([qb_rookies, skill_rookies], how="diagonal")
    return combined.rename({
        "avg_points_per_game": "fantasy_points_per_game",
        "low_confidence": "baseline_low_confidence",
    })


def build_rookie_feature_table(season=CURRENT_ROOKIE_SEASON):
    """
    Full Part B pipeline: current rookie class -> round bucket ->
    starter flag -> baseline lookup. Returns one row per current-year
    rookie with a projected fantasy_points_per_game, ready to merge
    into the master player table alongside veterans.
    """
    qb_baselines = pl.read_csv(QB_BASELINE_PATH)
    skill_baselines = pl.read_csv(SKILL_BASELINE_PATH)

    rookie_class = get_current_rookie_class(season)
    rookie_class = assign_round_bucket(rookie_class)
    rookie_class = flag_current_starters(rookie_class)
    rookie_features = apply_rookie_baselines(rookie_class, qb_baselines, skill_baselines)

    rookie_features = rookie_features.with_columns([
        pl.lit(True).alias("is_rookie"),
        pl.lit(0, dtype=pl.Int64).alias("games_played"),
    ])

    return rookie_features


if __name__ == "__main__":
    save_rookie_cohort_baselines()