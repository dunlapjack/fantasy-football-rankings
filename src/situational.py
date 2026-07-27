from pathlib import Path
import polars as pl
import nflreadpy as nfl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLAYCALLER_PATH = PROJECT_ROOT / "playcaller_history.csv"

OL_POSITIONS = ["C", "G", "T", "OL"]


def compute_team_tendency(seasons):
    """
    Pass/rush attempts per game, by team and season.
    Returns: team, season, pass_att_pg, rush_att_pg
    """
    team_stats = nfl.load_team_stats(seasons=seasons)
    tendency = (
        team_stats.filter(pl.col("season_type") == "REG")
        .group_by(["team", "season"])
        .agg([
            pl.col("attempts").sum().alias("pass_att_season"),
            pl.col("carries").sum().alias("rush_att_season"),
            pl.col("week").n_unique().alias("games"),
        ])
        .with_columns([
            (pl.col("pass_att_season") / pl.col("games")).alias("pass_att_pg"),
            (pl.col("rush_att_season") / pl.col("games")).alias("rush_att_pg"),
        ])
        .select(["team", "season", "pass_att_pg", "rush_att_pg"])
    )
    return tendency


def compute_qb_continuity(seasons):
    """
    Flags whether each team's primary passer (most attempts that season)
    differs from the prior season's primary passer.
    Returns: team, season, qb_changed (bool)
    """
    player_stats = nfl.load_player_stats(seasons)
    primary_qb = (
        player_stats.filter((pl.col("position") == "QB") & (pl.col("season_type") == "REG"))
        .group_by(["team", "season", "player_id"])
        .agg(pl.col("attempts").sum().alias("attempts"))
        .sort("attempts", descending=True)
        .group_by(["team", "season"])
        .first()
        .select(["team", "season", "player_id"])
    )

    prior_qb = (
        primary_qb.with_columns((pl.col("season") + 1).alias("season"))
        .rename({"player_id": "prior_qb_id"})
    )

    qb_change = (
        primary_qb.join(prior_qb, on=["team", "season"], how="left")
        .with_columns(
            (pl.col("player_id") != pl.col("prior_qb_id")).fill_null(True).alias("qb_changed")
        )
        .select(["team", "season", "qb_changed"])
    )
    return qb_change


def compute_coach_continuity():
    """
    Reads playcaller_history.csv and cleans changed_from_prior_year into
    a real boolean (source data stores it as inconsistent-case text).
    Returns: team, season, coach_changed (bool)
    """
    playcallers = pl.read_csv(PLAYCALLER_PATH)
    coach_change = (
        playcallers.select(["team", "season", "changed_from_prior_year"])
        .with_columns(
            pl.col("changed_from_prior_year")
            .cast(pl.String)
            .str.to_lowercase()
            .eq("true")
            .alias("coach_changed")
        )
        .select(["team", "season", "coach_changed"])
    )
    return coach_change


def compute_continuity_score(seasons):
    """
    Combines QB and coach continuity into one 0/1/2 score:
    0 = both same as last season, 1 = one changed, 2 = both changed.
    Confirmed via research (see notebook) that this correlates with
    bigger swings in team pass/rush tendency.
    Returns: team, season, qb_changed, coach_changed, continuity_score
    """
    qb_change = compute_qb_continuity(seasons)
    coach_change = compute_coach_continuity()

    combined = (
        qb_change.join(coach_change, on=["team", "season"], how="left")
        .with_columns([
            pl.col("qb_changed").fill_null(False),
            pl.col("coach_changed").fill_null(False),
        ])
        .with_columns(
            (pl.col("qb_changed").cast(pl.Int8) + pl.col("coach_changed").cast(pl.Int8))
            .alias("continuity_score")
        )
    )
    return combined


def compute_oline_continuity(seasons):
    """
    For each team-season, finds the top 5 offensive linemen by total
    offensive snaps that season, then counts how many of those 5 were
    also top-5 starters for the same team the prior season.
    Returns: team, season, returning_oline_starters (int, 0-5)
    """
    snaps = nfl.load_snap_counts(seasons=seasons)
    ol = snaps.filter(pl.col("position").is_in(OL_POSITIONS))

    season_snaps = (
        ol.group_by(["team", "season", "pfr_player_id"])
        .agg(pl.col("offense_snaps").sum().alias("total_snaps"))
    )

    top5 = (
        season_snaps.sort("total_snaps", descending=True)
        .group_by(["team", "season"], maintain_order=True)
        .head(5)
    )

    prior_top5 = (
        top5.with_columns((pl.col("season") + 1).alias("season"))
        .rename({"pfr_player_id": "prior_pfr_id"})
        .select(["team", "season", "prior_pfr_id"])
    )

    matched = top5.join(prior_top5, on=["team", "season"], how="left").with_columns(
        (pl.col("pfr_player_id") == pl.col("prior_pfr_id")).fill_null(False).alias("is_returning")
    )

    returning_counts = (
        matched.group_by(["team", "season"])
        .agg(pl.col("is_returning").sum().alias("returning_oline_starters"))
    )
    return returning_counts


def compute_position_competition(player_team_table):
    """
    player_team_table must have columns: player_id, position, team,
    fantasy_points_per_game.

    For each player, computes the average fantasy_points_per_game of
    their teammates at the same position, excluding themselves --
    "how good is the competition I'm walking into for touches."

    Returns: player_id, position_competition_ppg
    """
    group_totals = (
        player_team_table.group_by(["team", "position"])
        .agg([
            pl.col("fantasy_points_per_game").sum().alias("group_sum"),
            pl.len().alias("group_n"),
        ])
    )

    joined = player_team_table.join(group_totals, on=["team", "position"], how="left")

    result = joined.with_columns(
        pl.when(pl.col("group_n") > 1)
        .then((pl.col("group_sum") - pl.col("fantasy_points_per_game")) / (pl.col("group_n") - 1))
        .otherwise(0.0)
        .alias("position_competition_ppg")
    ).select(["player_id", "position_competition_ppg"])

    return result


def build_situational_features(seasons, veteran_features):
    """
    Combines all situational features into one player-level table.
    veteran_features must already have a `team` column (call
    attach_current_team() from features.py before passing it in).

    Returns: player_id, team, position, pass_att_pg, rush_att_pg,
    qb_changed, coach_changed, continuity_score,
    returning_oline_starters, position_competition_ppg
    """
    most_recent_season = max(seasons)

    tendency = compute_team_tendency(seasons).filter(pl.col("season") == most_recent_season).drop("season")
    continuity = compute_continuity_score(seasons).filter(pl.col("season") == most_recent_season).drop("season")
    oline = compute_oline_continuity(seasons).filter(pl.col("season") == most_recent_season).drop("season")

    team_features = tendency.join(continuity, on="team", how="left").join(oline, on="team", how="left")

    position_competition = compute_position_competition(veteran_features)

    player_level = (
        veteran_features.select(["player_id", "team", "position"])
        .join(team_features, on="team", how="left")
        .join(position_competition, on="player_id", how="left")
    )
    return player_level