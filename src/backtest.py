import argparse
from pathlib import Path
import polars as pl
import nflreadpy as nfl
from src.team_codes import normalize_team_column
from src.rookies import compute_primary_qb_by_team_season
from src.features import build_veteran_feature_table, load_veteran_stats, aggregate_season_stats
from src.situational import (
    compute_team_tendency,
    compute_coach_continuity,
    compute_oline_continuity,
    compute_position_competition,
    compute_recent_injury_flag,
    compute_workload_share,
    compute_experience,
    compute_age,
    compute_usage_trend,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_team_as_of_season(season):
    """
    Resolves each player's team as of the START of a historical season,
    using the earliest available regular-season week from
    load_rosters_weekly().

    This is the historical equivalent of load_players()'s `latest_team`
    field -- that field only reflects each player's CURRENT (2026) team,
    so it can't answer "what team was this player on entering 2024?"
    which the Phase 5 backtest needs in order to reconstruct situational
    features as they'd have looked at the start of a past season.
    """
    rosters = nfl.load_rosters_weekly(seasons=[season]).filter(
        pl.col("game_type") == "REG"
    )
    first_week = (
        rosters.sort("week")
        .group_by("gsis_id", maintain_order=True)
        .first()
        .select([
            pl.col("gsis_id").alias("player_id"),
            "team",
        ])
    )
    return normalize_team_column(first_week)


def compute_player_team_changed(target_season):
    """
    Player-level analog to qb_changed/coach_changed, which are team-level
    only and miss free-agent/trade moves entirely -- e.g. Kenneth Walker
    reads as "no continuity change" under the existing flags because
    neither Seattle's nor Kansas City's own coach/QB situation shifted,
    even though Walker himself is on a brand new roster, scheme, and QB.
    Flags whether a player's team-as-of-the-start-of-target_season differs
    from their team-as-of-the-start-of-the-prior season.
    Returns: player_id, team_changed
    """
    current = get_team_as_of_season(target_season).rename({"team": "current_team"})
    prior = get_team_as_of_season(target_season - 1).rename({"team": "prior_team"})
    combined = current.join(prior, on="player_id", how="left")
    return combined.with_columns(
        (pl.col("current_team") != pl.col("prior_team")).fill_null(True).alias("team_changed")
    ).select(["player_id", "team_changed"])


def resolve_oline_current_teams_historical(season):
    """
    Backtest equivalent of situational.py's resolve_oline_current_teams_live().
    Maps each offensive lineman's pfr_id to their team as of the START
    of `season`, by taking get_team_as_of_season()'s gsis_id-based
    result and re-keying it to pfr_player_id via load_players()'s
    gsis_id/pfr_id crosswalk (snap_counts identifies linemen by
    pfr_player_id, not gsis_id, so the join key has to switch).
    Returns: pfr_player_id, current_team
    """
    team_as_of = get_team_as_of_season(season)

    crosswalk = nfl.load_players().select([
        pl.col("gsis_id").alias("player_id"),
        "pfr_id",
    ])

    resolved = (
        team_as_of.join(crosswalk, on="player_id", how="left")
        .filter(pl.col("pfr_id").is_not_null())
        .select([
            pl.col("pfr_id").alias("pfr_player_id"),
            pl.col("team").alias("current_team"),
        ])
    )
    return normalize_team_column(resolved, column="current_team")


def compute_qb_continuity_historical(reference_season, target_season):
    """
    Backtest equivalent of situational.py's compute_qb_continuity().
    Unlike the live version -- which must predict the upcoming starter
    from a depth chart, since the season hasn't happened yet -- this
    already has real attempts data for both seasons, so it just
    compares each team's actual primary passer in reference_season
    against their actual primary passer in target_season directly.
    No prediction step needed.
    Returns: team, qb_changed
    """
    reference_qb = (
        compute_primary_qb_by_team_season([reference_season])
        .drop("season")
        .rename({"primary_qb_id": "reference_qb_id"})
    )
    target_qb = (
        compute_primary_qb_by_team_season([target_season])
        .drop("season")
        .rename({"primary_qb_id": "target_qb_id"})
    )

    combined = reference_qb.join(target_qb, on="team", how="left")
    combined = combined.with_columns(
        (pl.col("reference_qb_id") != pl.col("target_qb_id")).fill_null(True).alias("qb_changed")
    )
    return combined.select(["team", "qb_changed"])


def build_backtest_season(target_season):
    """
    Assembles one row per established veteran player for a single
    historical target season: their situational features as they'd
    have looked entering that season, their trailing baseline PPG
    (weighted average of the 3 prior seasons, same recency scheme as
    the live model), their actual PPG in the target season, and the
    delta between the two. Rookies in target_season are naturally
    excluded -- they have no baseline_ppg since there's no prior-season
    history for the join to find.

    Returns: player_id, player_name, position, team, baseline_ppg,
    actual_ppg, delta, pass_att_pg, rush_att_pg, qb_changed,
    coach_changed, returning_oline_starters, position_competition_ppg,
    experience, age, usage_trend_share, usage_trend_volume,
    usage_trend_relative, trend_seasons_used, trend_low_confidence,
    season
    """
    reference_season = target_season - 1
    baseline_seasons = [target_season - 3, target_season - 2, target_season - 1]

    baseline = (
        build_veteran_feature_table(baseline_seasons)
        .select(["player_id", "player_name", "position", "fantasy_points_per_game",
                  "carries_per_game", "targets_per_game"])
        .rename({"fantasy_points_per_game": "baseline_ppg"})
    )

    actual_raw = load_veteran_stats([target_season])
    actual = (
        aggregate_season_stats(actual_raw)
        .select(["player_id", "fantasy_points_per_game", "games_played"])
        .rename({"fantasy_points_per_game": "actual_ppg", "games_played": "actual_games_played"})
    )

    team = get_team_as_of_season(target_season)

    tendency = (
        compute_team_tendency([reference_season])
        .filter(pl.col("season") == reference_season)
        .drop("season")
    )
    qb_cont = compute_qb_continuity_historical(reference_season, target_season)
    coach_cont = compute_coach_continuity(target_season)
    oline_cont = compute_oline_continuity(
        [reference_season], resolve_oline_current_teams_historical(target_season)
    )

    team_features = (
        tendency.join(qb_cont, on="team", how="left")
        .join(coach_cont, on="team", how="left")
        .join(oline_cont, on="team", how="left")
    )

    baseline_with_team = baseline.join(team, on="player_id", how="left")
    position_competition_input = baseline_with_team.rename(
        {"baseline_ppg": "fantasy_points_per_game"}
    )
    position_competition = compute_position_competition(position_competition_input)
    team_changed = compute_player_team_changed(target_season)

    workload_share = compute_workload_share(baseline_with_team, tendency)
    injury = compute_recent_injury_flag([reference_season])
    experience = compute_experience(target_season)
    age = compute_age(target_season)

    # Trend is fitted over the SAME window the baseline is built from
    # (target-3 .. target-1), so it never sees the target season. If
    # these two windows ever drift apart, the trend starts leaking the
    # outcome into the predictor.
    usage_trend = compute_usage_trend(baseline_seasons)

    combined = (
        baseline_with_team.join(actual, on="player_id", how="left")
        .join(team_features, on="team", how="left")
        .join(position_competition, on="player_id", how="left")
        .join(team_changed, on="player_id", how="left")
        .join(workload_share, on="player_id", how="left")
        .join(injury, on="player_id", how="left")
        .join(experience, on="player_id", how="left")
        .join(age, on="player_id", how="left")
        .join(usage_trend, on="player_id", how="left")
        .with_columns([
            pl.col("team_changed").fill_null(True),
            pl.col("recent_major_injury").fill_null(False),
            pl.col("trend_seasons_used").fill_null(0),
            pl.col("trend_missing").fill_null(True),
            pl.col("trend_low_confidence").fill_null(False),
        ])
        .filter(pl.col("actual_ppg").is_not_null())
        .with_columns([
            (pl.col("actual_ppg") - pl.col("baseline_ppg")).alias("delta"),
            pl.lit(target_season).alias("season"),
        ])
    )
    return combined


# How far back the training set reaches.
#
# WHY 2021 AND NOT EARLIER (Aug 4)
# --------------------------------
# The binding constraint is playcaller_history.csv, which is maintained
# by hand and starts at 2021. compute_coach_continuity() reads the
# TARGET season's own row, so a 2021 target needs a 2021 row -- which
# exists. 2020 does not, so 2020 is the first target season that would
# require new manual research.
#
# Everything else reaches back much further: nflverse player stats,
# weekly rosters, and team stats all predate this comfortably, and
# baselines for a 2021 target come from 2018-2020, which is fine.
#
# Started at 3 seasons on the reasonable assumption that recent football
# predicts best. That logic applies to a PLAYER'S BASELINE -- how we
# project him -- but not to the training set, which is only estimating
# how much a coaching change or a year of age is worth. Those
# relationships are stable enough that more history is close to free
# accuracy, and small samples were the live constraint: the 3-season-only
# usage-trend test had n=113 and couldn't separate a real effect from
# zero. Phase 12's rookie model (5 draft classes) needs it more still.
#
# Caveat worth remembering: 2020 sits inside the baseline window for the
# 2021-2023 targets. No preseason, opt-outs, COVID absences. It was
# already there for 2023; widening the window gives it two more targets
# to influence.
DEFAULT_TARGET_SEASONS = [2021, 2022, 2023, 2024, 2025]

# Earliest season playcaller_history.csv covers. Targets before this
# would silently lose coach_changed rather than fail loudly.
EARLIEST_PLAYCALLER_SEASON = 2021


def build_backtest_dataset(target_seasons=None):
    """
    Runs build_backtest_season() for each target season and stacks the
    results into one table -- this is the training set for the
    per-position regression (delta ~ situational features) that
    produces the weights used in the live 2026 composite score.
    """
    if target_seasons is None:
        target_seasons = DEFAULT_TARGET_SEASONS

    too_early = [s for s in target_seasons if s < EARLIEST_PLAYCALLER_SEASON]
    if too_early:
        raise ValueError(
            f"Target seasons {too_early} precede playcaller_history.csv, which "
            f"starts at {EARLIEST_PLAYCALLER_SEASON}. compute_coach_continuity() "
            f"would return no rows for them and every player in those seasons "
            f"would silently get a null coach_changed -- a quiet hole in the "
            f"training set rather than an error. Extend playcaller_history.csv "
            f"first (Pro Football Reference has coordinator history), or drop "
            f"those seasons."
        )

    tables = [build_backtest_season(s) for s in target_seasons]
    return pl.concat(tables, how="vertical")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the backtest training set.")
    parser.add_argument(
        "--seasons", type=int, nargs="+", default=DEFAULT_TARGET_SEASONS,
        help=f"target seasons to train on (default: {DEFAULT_TARGET_SEASONS}). "
             f"Cannot go earlier than {EARLIEST_PLAYCALLER_SEASON} without "
             f"extending playcaller_history.csv.",
    )
    args = parser.parse_args()

    dataset = build_backtest_dataset(args.seasons)
    print(f"Backtest dataset: {dataset.shape[0]} player-seasons")
    print(dataset.group_by("season").len().sort("season"))

    output_path = PROJECT_ROOT / "data" / "backtest_features.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_csv(output_path)
    print(f"Wrote backtest dataset to {output_path}")