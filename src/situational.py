from datetime import date
from pathlib import Path
import polars as pl
import nflreadpy as nfl
from src.team_codes import normalize_team_column

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLAYCALLER_PATH = PROJECT_ROOT / "playcaller_history.csv"

OL_POSITIONS = ["C", "G", "T", "OL"]
UPCOMING_SEASON = 2026

# --- Phase 10 CP1: usage trend -------------------------------------------
# Positions with a meaningful opportunity share. QB is excluded for the
# same reason it is in workload_share.
TREND_POSITIONS = ["RB", "WR", "TE"]

# A season must have at least this many games to contribute a point to
# the slope. Below it you are measuring an injury, not a role.
MIN_TREND_GAMES = 4

# Seasons needed to fit any slope at all. Below this the trend columns
# are null and `trend_missing` fires.
MIN_TREND_SEASONS_MODEL = 2

# Mean share below which usage_trend_relative is nulled -- dividing a
# slope by a near-zero denominator manufactures huge numbers from noise.
RELATIVE_TREND_FLOOR = 0.02


def compute_team_tendency(seasons):
    """
    Pass/rush attempts per game, by team and season. This stays
    historical on purpose -- it's a trailing indicator of how a team
    actually played, not a "did something change" flag.
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


def get_league_qb_starters(season):
    """
    Latest depth chart snapshot per team, filtered to the presumptive
    starting QB (pos_rank == 1) for the given season.
    """
    depth_charts = nfl.load_depth_charts(seasons=[season])
    latest_dt = depth_charts.group_by("team").agg(pl.col("dt").max().alias("latest_dt"))
    latest = depth_charts.join(latest_dt, on="team").filter(pl.col("dt") == pl.col("latest_dt"))
    starters = latest.filter((pl.col("pos_abb") == "QB") & (pl.col("pos_rank") == 1))
    return starters.select(["team", pl.col("gsis_id").alias("current_qb_id")])


def compute_qb_continuity(seasons, upcoming_season=UPCOMING_SEASON):
    """
    Compares each team's actual primary passer from the most recent
    completed season against their CURRENT depth-chart starter for the
    upcoming season -- forward-looking, not a comparison of two past
    seasons against each other.
    Returns: team, qb_changed
    """
    most_recent_season = max(seasons)
    player_stats = nfl.load_player_stats(seasons)

    last_season_primary_qb = (
        player_stats.filter(
            (pl.col("position") == "QB")
            & (pl.col("season_type") == "REG")
            & (pl.col("season") == most_recent_season)
        )
        .group_by(["team", "player_id"])
        .agg(pl.col("attempts").sum().alias("attempts"))
        .sort("attempts", descending=True)
        .group_by("team")
        .first()
        .select(["team", "player_id"])
        .rename({"player_id": "last_season_qb_id"})
    )

    current_starters = get_league_qb_starters(upcoming_season)

    combined = last_season_primary_qb.join(current_starters, on="team", how="left")
    combined = combined.with_columns(
        (pl.col("last_season_qb_id") != pl.col("current_qb_id")).fill_null(True).alias("qb_changed")
    )
    return combined.select(["team", "qb_changed"])


def compute_coach_continuity(upcoming_season=UPCOMING_SEASON):
    """
    Reads playcaller_history.csv and returns whether each team's
    playcaller changed entering the upcoming season, using the
    manually-maintained changed_from_prior_year flag for that season's
    row specifically (not last season's row).
    Returns: team, coach_changed
    """
    playcallers = pl.read_csv(PLAYCALLER_PATH)
    playcallers = normalize_team_column(playcallers)
    coach_change = (
        playcallers.filter(pl.col("season") == upcoming_season)
        .select(["team", "changed_from_prior_year"])
        .with_columns(
            pl.col("changed_from_prior_year")
            .cast(pl.String)
            .str.to_lowercase()
            .eq("true")
            .alias("coach_changed")
        )
        .select(["team", "coach_changed"])
    )
    return coach_change


def compute_continuity_score(seasons, upcoming_season=UPCOMING_SEASON):
    """
    Combines QB and coach continuity into one 0/1/2 score, both
    forward-looking for the upcoming season.
    Returns: team, qb_changed, coach_changed, continuity_score
    """
    qb_change = compute_qb_continuity(seasons, upcoming_season)
    coach_change = compute_coach_continuity(upcoming_season)

    combined = (
        qb_change.join(coach_change, on="team", how="left")
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


def resolve_oline_current_teams_live():
    """
    Live path: maps each offensive lineman's pfr_id to their CURRENT
    (2026) team via load_players()'s latest_team. This is the "who's
    still here" half of o-line continuity for the live draft-day score.
    Returns: pfr_player_id, current_team
    """
    current_rosters = (
        nfl.load_players()
        .select(["pfr_id", "latest_team"])
        .rename({"pfr_id": "pfr_player_id", "latest_team": "current_team"})
    )
    return normalize_team_column(current_rosters, column="current_team")


def compute_oline_continuity(seasons, current_rosters):
    """
    Finds each team's top 5 offensive linemen by total offensive snaps
    in the most recent season within `seasons`, then checks how many of
    those 5 appear in `current_rosters` on the same team.

    `current_rosters` must supply columns [pfr_player_id, current_team]
    -- for the live 2026 score, pass resolve_oline_current_teams_live().
    For a historical backtest season, pass
    backtest.resolve_oline_current_teams_historical(season) instead --
    the matching logic below is identical either way; only how
    "current team" gets resolved differs between live and backtest.

    Returns: team, returning_oline_starters (int, 0-5)
    """
    most_recent_season = max(seasons)
    snaps = nfl.load_snap_counts(seasons=[most_recent_season])
    ol = snaps.filter(pl.col("position").is_in(OL_POSITIONS))

    season_snaps = (
        ol.group_by(["team", "pfr_player_id"])
        .agg(pl.col("offense_snaps").sum().alias("total_snaps"))
    )

    top5 = (
        season_snaps.sort("total_snaps", descending=True)
        .group_by("team", maintain_order=True)
        .head(5)
    )

    matched = top5.join(current_rosters, on="pfr_player_id", how="left").with_columns(
        (pl.col("team") == pl.col("current_team")).fill_null(False).alias("is_returning")
    )

    returning_counts = (
        matched.group_by("team")
        .agg(pl.col("is_returning").sum().alias("returning_oline_starters"))
    )
    return returning_counts


def compute_position_competition(player_team_table):
    """
    player_team_table must have columns: player_id, position, team,
    fantasy_points_per_game.

    For each player, computes the average fantasy_points_per_game of
    their teammates at the same position, excluding themselves.

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

def compute_workload_share(features_with_usage, tendency):
    """
    Player's own share of his team's opportunities -- rushing share for
    RB, target share for WR/TE. A more direct measure of "does this
    player have the job" than a teammate's raw PPG
    (position_competition_ppg tested flat, p=0.77, on the RB backtest).
    `features_with_usage` needs player_id, team, position,
    carries_per_game, targets_per_game. `tendency` needs team,
    pass_att_pg, rush_att_pg.
    Returns: player_id, workload_share
    """
    joined = features_with_usage.select(
        ["player_id", "team", "position", "carries_per_game", "targets_per_game"]
    ).join(tendency, on="team", how="left")
    return joined.with_columns(
        pl.when(pl.col("position") == "RB")
        .then(pl.col("carries_per_game") / pl.col("rush_att_pg"))
        .when(pl.col("position").is_in(["WR", "TE"]))
        .then(pl.col("targets_per_game") / pl.col("pass_att_pg"))
        .otherwise(None)
        .alias("workload_share")
    ).select(["player_id", "workload_share"])

def compute_recent_injury_flag(seasons):
    """
    Flags any player whose roster status was RES (reserve/injured) in
    their team's final regular-season week of the most recent completed
    season -- a simple, explicit signal that they ended last season on
    IR, rather than trying to parse free-text injury descriptions.

    This is intentionally a REFERENCE flag only, like ADP -- it doesn't
    feed into fantasy_points_per_game or any other feature. It just
    surfaces "this player ended last season hurt" so it's visible at
    the draft table.

    Returns: player_id, recent_major_injury (bool)
    """
    most_recent_season = max(seasons)
    rosters = nfl.load_rosters_weekly(seasons=[most_recent_season]).filter(
        pl.col("game_type") == "REG"
    )

    last_week_status = (
        rosters.sort("week")
        .group_by("gsis_id", maintain_order=True)
        .last()
        .select([
            pl.col("gsis_id").alias("player_id"),
            (pl.col("status") == "RES").alias("recent_major_injury"),
        ])
    )
    return last_week_status

def compute_experience(target_season):
    """
    Years since a player's rookie season, via load_players()'s
    rookie_season field (already used in rookies.py) -- an aging-curve
    proxy that needs no birthdate and works identically for a historical
    target_season or the live 2026 upcoming_season.
    Returns: player_id, experience
    """
    players = nfl.load_players().select([
        pl.col("gsis_id").alias("player_id"), "rookie_season",
    ])
    return players.with_columns(
        (pl.lit(target_season) - pl.col("rookie_season")).alias("experience")
    ).select(["player_id", "experience"])

def compute_age(target_season, as_of=(9, 1)):
    """
    Player age in years as of Sept 1 of target_season, from
    load_players()'s birth_date.

    WHY THIS EXISTS ALONGSIDE compute_experience()
    ----------------------------------------------
    `experience` (season - rookie_season) is an aging PROXY. It can't
    tell a 24-year-old fourth-year back from a 27-year-old one, and
    those are not the same asset. Phase 6's experience coefficient is
    the term memory flags as over-penalizing established veterans
    (Kamara took the largest negative adjustment in the drafted pool),
    so it's worth knowing whether the penalty is really about age or
    really about seasons logged. Phase 10 CP2 fits both and keeps
    whichever wins; this function only supplies the input.

    birth_date is a static biographical field, so -- exactly like
    rookie_season -- reading it from the CURRENT load_players() snapshot
    is valid for a historical target_season too. Nothing about a
    player's birthday gets revised after the fact. (This is NOT true of
    latest_team, which is why compute_live_team_changed exists
    separately.)

    Returns: player_id, age
    """
    players = nfl.load_players()
    if "birth_date" not in players.columns:
        raise KeyError(
            "load_players() has no 'birth_date' column -- available: "
            f"{sorted(players.columns)}. Phase 10 CP2 needs it; if nflreadpy "
            "renamed the field, update compute_age() rather than falling back "
            "to experience silently."
        )

    players = players.select([pl.col("gsis_id").alias("player_id"), "birth_date"])

    # birth_date arrives as either a Date or an ISO string depending on
    # the nflreadpy version. Handle both; strict=False so a malformed
    # value becomes null rather than killing the whole pipeline.
    if players.schema["birth_date"] == pl.Utf8:
        players = players.with_columns(
            pl.col("birth_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False)
        )
    else:
        players = players.with_columns(pl.col("birth_date").cast(pl.Date, strict=False))

    reference = date(target_season, as_of[0], as_of[1])

    return players.with_columns(
        ((pl.lit(reference) - pl.col("birth_date")).dt.total_days() / 365.25).alias("age")
    ).with_columns(
        # Guard against junk birthdates producing a 3-year-old or a
        # 60-year-old running back. Out-of-range becomes null, which
        # fit_weights imputes to the position mean.
        pl.when(pl.col("age").is_between(18.0, 50.0))
        .then(pl.col("age"))
        .otherwise(None)
        .alias("age")
    ).select(["player_id", "age"])


def _ols_slope(xs, ys):
    """
    Least-squares slope of ys on xs. Two points give the plain
    difference, which is what OLS reduces to at n=2 -- that's the
    intended behavior, not an accident, but it IS why
    trend_low_confidence exists.
    """
    n = len(xs)
    if n < 2:
        return None
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    sxx = sum((x - x_mean) ** 2 for x in xs)
    if sxx == 0:
        return None
    sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    return sxy / sxx


def _load_season_usage(seasons):
    """
    Per-player, per-season usage plus that season's team pace, which is
    what every trend variant below is built from.

    Deliberately does NOT go through features.load_veteran_stats():
    that function scores every row through calculate_offensive_points
    via map_elements, and the trend needs carries/targets/games only.

    Returns: player_id, season, position, team, games_played,
    usage_volume, usage_share
    """
    raw = nfl.load_player_stats(sorted(seasons)).filter(
        (pl.col("season_type") == "REG")
        & pl.col("player_id").is_not_null()
        & pl.col("position").is_in(TREND_POSITIONS)
    )

    team_column = "team" if "team" in raw.columns else "recent_team"

    per_season = (
        raw.sort("week")
        .group_by(["player_id", "season"], maintain_order=True)
        .agg([
            pl.col("carries").sum().alias("carries"),
            pl.col("targets").sum().alias("targets"),
            pl.len().alias("games_played"),
            pl.col("position").last().alias("position"),
            pl.col(team_column).last().alias("team"),
        ])
    )
    per_season = normalize_team_column(per_season, column="team")

    # Each season's share uses THAT season's team, so a player who moved
    # in the middle of the window still gets an apples-to-apples slope.
    # This is why usage trend does not need the team_changed null-out
    # that workload_share requires: workload_share divides a trailing
    # multi-year numerator by the CURRENT team's pace, which mixes
    # eras. The trend never does that.
    tendency = compute_team_tendency(seasons)
    tendency = normalize_team_column(tendency, column="team")

    joined = per_season.join(tendency, on=["team", "season"], how="left")

    return joined.with_columns([
        pl.when(pl.col("position") == "RB")
        .then(pl.col("carries") / pl.col("games_played"))
        .when(pl.col("position").is_in(["WR", "TE"]))
        .then(pl.col("targets") / pl.col("games_played"))
        .otherwise(None)
        .alias("usage_volume"),
    ]).with_columns([
        pl.when(pl.col("position") == "RB")
        .then(pl.col("usage_volume") / pl.col("rush_att_pg"))
        .when(pl.col("position").is_in(["WR", "TE"]))
        .then(pl.col("usage_volume") / pl.col("pass_att_pg"))
        .otherwise(None)
        .alias("usage_share"),
    ]).select([
        "player_id", "season", "position", "team", "games_played",
        "usage_volume", "usage_share",
    ])


def compute_usage_trend(seasons):
    """
    Direction of a player's usage across the baseline window -- the
    thing `workload_share` structurally cannot see. A 22% target share
    is the same number whether it came from 15% -> 18% -> 22% or from
    30% -> 26% -> 22%, and those two players should not be projected
    alike.

    Three variants are produced because Phase 10 CP1 tests all three
    rather than assuming which basis is right (Aug 4 decision):

      usage_trend_share     slope of share-of-team-volume per season.
                            Immune to team pace: a rising share on a
                            slowing offense still reads as rising.
                            This is the same basis as workload_share,
                            so trend and level are directly comparable
                            -- which also means they may be
                            collinear, checked at fit time.

      usage_trend_volume    slope of raw per-game carries/targets.
                            Confounds player role growth with team
                            volume, but it's what actually shows up in
                            a box score.

      usage_trend_relative  share slope divided by mean share. Separates
                            22%->26% (+18% relative) from 5%->9%
                            (+80% relative) -- absolute slope calls
                            those nearly identical. Null below a
                            RELATIVE_TREND_FLOOR mean share, since
                            dividing by a near-zero denominator
                            manufactures enormous slopes out of
                            rounding noise.

    Seasons in which the player appeared in fewer than MIN_TREND_GAMES
    games are dropped before fitting -- a 2-game injury season is a
    fact about availability, not about role, and it would dominate a
    3-point slope. Phase 11 handles availability properly.

    Players with only 2 usable seasons DO get a slope (Aug 4 decision:
    keep the coverage) but are flagged `trend_low_confidence`, so the
    fit can test whether a 2-point slope deserves the same weight as a
    3-point one -- either as its own term or interacted with the trend.
    Fewer than 2 usable seasons yields nulls, which fit_weights imputes
    to the position mean.

    QB is null throughout, matching workload_share.

    Returns: player_id, usage_trend_share, usage_trend_volume,
    usage_trend_relative, trend_seasons_used, trend_low_confidence
    """
    usage = _load_season_usage(seasons).filter(
        (pl.col("games_played") >= MIN_TREND_GAMES)
        & pl.col("usage_share").is_not_null()
        & pl.col("usage_volume").is_not_null()
    )

    grouped = (
        usage.sort(["player_id", "season"])
        .group_by("player_id", maintain_order=True)
        .agg([
            pl.col("season").alias("_seasons"),
            pl.col("usage_share").alias("_shares"),
            pl.col("usage_volume").alias("_volumes"),
            pl.len().alias("trend_seasons_used"),
        ])
    )

    def _slopes(row):
        xs = [float(s) for s in row["_seasons"]]
        shares = [float(v) for v in row["_shares"]]
        volumes = [float(v) for v in row["_volumes"]]

        share_slope = _ols_slope(xs, shares)
        volume_slope = _ols_slope(xs, volumes)

        mean_share = sum(shares) / len(shares) if shares else 0.0
        if share_slope is None or mean_share < RELATIVE_TREND_FLOOR:
            relative = None
        else:
            relative = share_slope / mean_share

        return {
            "usage_trend_share": share_slope,
            "usage_trend_volume": volume_slope,
            "usage_trend_relative": relative,
        }

    slope_struct = pl.Struct([
        pl.Field("usage_trend_share", pl.Float64),
        pl.Field("usage_trend_volume", pl.Float64),
        pl.Field("usage_trend_relative", pl.Float64),
    ])

    result = grouped.with_columns(
        pl.struct(["_seasons", "_shares", "_volumes"])
        .map_elements(_slopes, return_dtype=slope_struct)
        .alias("_slopes")
    ).unnest("_slopes")

    return result.with_columns([
        # MODEL TERM. No slope could be fitted at all. Paired with
        # mean-imputation in fit_weights, this is a standard missing
        # indicator -- it keeps these players in the sample instead of
        # dropping them, which matters because they are not a random
        # subset (Aug 4: mean delta +0.97 vs -0.79 for the rest).
        (pl.col("trend_seasons_used") < MIN_TREND_SEASONS_MODEL).alias("trend_missing"),
        # DISPLAY FLAG ONLY. Exactly 2 seasons of history.
        #
        # This started life as "discount these" and the data said the
        # opposite: the trend signal is CARRIED by 2-season players
        # (RB p=0.0013 including them, p=0.259 on 3-season-only, whose
        # 95% CI [-3.43, +12.86] still contains the 2-season estimate).
        # Explicit discount interactions tested p=0.57 (RB) and p=0.76
        # (TE). So it earns a column on the board and no weight in the
        # model.
        (pl.col("trend_seasons_used") == MIN_TREND_SEASONS_MODEL).alias("trend_low_confidence"),
    ]).select([
        "player_id", "usage_trend_share", "usage_trend_volume",
        "usage_trend_relative", "trend_seasons_used", "trend_missing",
        "trend_low_confidence",
    ])


def compute_live_team_changed(upcoming_season=UPCOMING_SEASON):
    """
    Live-path equivalent of backtest.py's compute_player_team_changed --
    that function only works for a completed historical target_season
    (it needs real roster data for both years). For the live 2026 score,
    "current team" instead comes from load_players()'s latest_team (the
    same source resolve_oline_current_teams_live() uses), compared
    against the player's team as of the start of the most recently
    completed season (2025), resolved from actual weekly roster data.

    This can't just import backtest.py's get_team_as_of_season() --
    backtest.py already imports from this file, so importing it back
    here would create a circular import -- so the prior-team lookup is
    duplicated here in miniature.

    Returns: player_id, team_changed
    """
    prior_season = upcoming_season - 1
    prior_rosters = nfl.load_rosters_weekly(seasons=[prior_season]).filter(
        pl.col("game_type") == "REG"
    )
    prior_team = (
        prior_rosters.sort("week")
        .group_by("gsis_id", maintain_order=True)
        .first()
        .select([
            pl.col("gsis_id").alias("player_id"),
            pl.col("team").alias("prior_team"),
        ])
    )
    prior_team = normalize_team_column(prior_team, column="prior_team")

    current_team = nfl.load_players().select([
        pl.col("gsis_id").alias("player_id"),
        pl.col("latest_team").alias("current_team"),
    ])
    current_team = normalize_team_column(current_team, column="current_team")

    combined = current_team.join(prior_team, on="player_id", how="left")
    return combined.with_columns(
        (pl.col("current_team") != pl.col("prior_team")).fill_null(True).alias("team_changed")
    ).select(["player_id", "team_changed"])


def build_situational_features(seasons, veteran_features, upcoming_season=UPCOMING_SEASON):
    """
    Combines all situational features into one player-level table.
    veteran_features must already have a `team` column.

    Team tendency is intentionally historical (last actual season).
    QB continuity, coach continuity, and o-line continuity are all
    forward-looking for the upcoming season.

    Returns: player_id, team, position, pass_att_pg, rush_att_pg,
    qb_changed, coach_changed, continuity_score,
    returning_oline_starters, position_competition_ppg,
    recent_major_injury, workload_share, experience, age,
    usage_trend_share, usage_trend_volume, usage_trend_relative,
    trend_seasons_used, trend_low_confidence, team_changed
    """
    most_recent_season = max(seasons)

    tendency = compute_team_tendency(seasons).filter(pl.col("season") == most_recent_season).drop("season")
    continuity = compute_continuity_score(seasons, upcoming_season)
    oline = compute_oline_continuity(seasons, resolve_oline_current_teams_live())

    team_features = tendency.join(continuity, on="team", how="left").join(oline, on="team", how="left")

    position_competition = compute_position_competition(veteran_features)
    injury_flag = compute_recent_injury_flag(seasons)

    workload_share = compute_workload_share(veteran_features, tendency)
    experience = compute_experience(upcoming_season)
    age = compute_age(upcoming_season)
    usage_trend = compute_usage_trend(seasons)
    team_changed = compute_live_team_changed(upcoming_season)

    player_level = (
        veteran_features.select(["player_id", "team", "position"])
        .join(team_features, on="team", how="left")
        .join(position_competition, on="player_id", how="left")
        .join(injury_flag, on="player_id", how="left")
        .join(workload_share, on="player_id", how="left")
        .join(experience, on="player_id", how="left")
        .join(age, on="player_id", how="left")
        .join(usage_trend, on="player_id", how="left")
        .join(team_changed, on="player_id", how="left")
        .with_columns([
            pl.col("recent_major_injury").fill_null(False),
            pl.col("team_changed").fill_null(True),
            # No usable seasons at all (rookies, and veterans whose every
            # season fell under MIN_TREND_GAMES) read as 0 seasons used
            # and trend_missing -- not as a silent null count.
            pl.col("trend_seasons_used").fill_null(0),
            pl.col("trend_missing").fill_null(True),
            pl.col("trend_low_confidence").fill_null(False),
        ])
        .with_columns(
            # workload_share is carries(or targets)/game measured against the
            # player's CURRENT team's pass/rush attempts per game -- but the
            # numerator (his own per-game rate) is a trailing average that,
            # for a player who just switched teams, was mostly or entirely
            # earned on his OLD team. Dividing old-team volume by new-team
            # pace is an apples-to-oranges number (e.g. David Montgomery:
            # DET-era carries/game over HOU's 2025 rush attempts/game), so
            # it's nulled out here rather than trusted -- fill_null(0) in
            # apply_situational_weights then treats it as "no opinion" for
            # these players instead of applying a possibly-wrong penalty.
            pl.when(pl.col("team_changed"))
            .then(None)
            .otherwise(pl.col("workload_share"))
            .alias("workload_share")
        )
    )
    return player_level