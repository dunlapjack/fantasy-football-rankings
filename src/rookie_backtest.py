"""
Phase 12 CP1. Builds the rookie training set: one row per drafted
offensive rookie from the 2021-2025 classes, with the situational
features that were knowable BEFORE that player took a snap, and the
delta between his cohort baseline and what he actually did.

WHY THIS FILE EXISTS
--------------------
Rookies are the biggest structural hole left in the model. They get a
flat cohort number -- one value per position/round bucket -- and no
situational adjustment at all. So Jeremiyah Love (ARI) and Jadarian
Price (SEA) both project 15.12 PPG and tie at model ranks 4-5, because
the model literally cannot see that they landed in different places.
Every veteran gets team context; rookies get none.

This is the rookie analog of backtest.py, and deliberately mirrors its
shape: build a per-season table of (baseline, actual, delta, features),
stack the seasons, hand the result to a fitter. What it does NOT mirror
is the FEATURE SET, and the reason is worth stating rather than
discovering later.

WHAT A ROOKIE CANNOT HAVE
-------------------------
Four of the veteran features are meaningless here, and three of them
would not merely be uninformative -- they would be actively wrong:

  `experience`, `age`      Experience is 0 for every rookie by
                           definition, so it is a constant, and a
                           constant is absorbed by the intercept.
                           Age survives as a candidate (draft classes
                           do vary, 21 to 25) but it is a much weaker
                           thing here than the veteran aging curve,
                           which measures decline. Left in the
                           candidate set and allowed to fail.

  `team_changed`           Compares this season's team to last
                           season's roster. A rookie was on NO NFL
                           roster last season, so this reads True for
                           essentially the entire population. It is a
                           rookie detector, not a feature. Phase 6
                           documented this; the plan excludes it
                           explicitly. Excluded.

  `workload_share`         Needs carries/targets per game from prior
                           seasons. There are none. Nulls everywhere.

  `usage_trend_share`      Needs two seasons of usage to fit a slope.
                           Same problem, worse.

  `qb_changed` /           These are team-continuity flags. A rookie
  `recent_major_injury`    has no continuity to break, and no NFL
                           injury history for the flag to read.

What remains is exactly the plan's CP2 list -- team pass/rush tendency,
position competition, O-line continuity, depth chart position -- plus
age as a tested candidate. That is a short list, which is honest: there
genuinely is not much the model can know about a player who has not
played yet.

TWO WAYS THIS COULD LEAK, AND WHAT STOPS THEM
---------------------------------------------
Both are the same mistake as fitting on the target season, wearing
different clothes, and both would inflate R^2 and look like success.

1. THE COHORT BASELINE. rookies.py computes cohort baselines by
   averaging PPG over ALL of 2021-2025. If the 2023 class's baseline
   includes 2023's own outcomes, then `delta` is measured against a
   number that already saw the answer, and it shrinks toward zero for
   reasons that have nothing to do with the features. Baselines here
   are therefore LEAVE-ONE-CLASS-OUT: the 2023 rows use a baseline
   built from 2021, 2022, 2024, 2025 only. See
   leave_one_out_baselines().

2. THE DEPTH CHART. rookies.py's live path reads the LATEST depth chart
   snapshot, which is correct for August 2026 -- it is the best current
   information about a season that has not started. Reading the latest
   snapshot of a season that HAS happened is a different thing
   entirely: by week 15 the depth chart has been rewritten by the very
   outcomes we are predicting, so a rookie who broke out reads
   `pos_rank 1` BECAUSE he broke out. This file takes the EARLIEST
   snapshot of the rookie's season instead. See week_one_depth_chart().

   Worth being clear that this is a compromise, not a fix. The live
   model reads an August depth chart; the earliest available historical
   snapshot is the closest analog nflverse offers, but if a team's first
   scrape lands mid-September the two are not quite measuring the same
   moment. The direction of the remaining error is toward MORE
   information than the live model has, so a pos_rank coefficient fitted
   here should be treated as an optimistic ceiling.

USAGE
-----
    python -m src.rookie_backtest

Writes data/rookie_backtest_features.csv. Then:

    python -m src.fit_rookie_weights
"""

import argparse
from pathlib import Path

import polars as pl
import nflreadpy as nfl

from src.team_codes import normalize_team_column
from src.scoring import load_config, calculate_offensive_points
from src.rookies import OFFENSE_POSITIONS, MIN_CONFIDENT_N
from src.situational import (
    compute_team_tendency,
    compute_oline_continuity,
    compute_position_competition,
    compute_age,
)
from src.backtest import (
    get_team_as_of_season,
    resolve_oline_current_teams_historical,
)
from src.fit_rookie_weights import SEASON_CONFOUND_SPREAD

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "league_config_lebronjames.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "rookie_backtest_features.csv"

# WIDENED 2021-2025 -> 2017-2025 (Aug 6), matching backtest.py.
#
# WHY THIS IS NOT FISHING, WHICH IS THE FIRST THING TO ASK
# --------------------------------------------------------
# The window is being changed after seeing that QB and TE failed, and
# that is exactly the shape of the move fit_weights warns about under ON
# MULTIPLE TESTING. It is defensible here for one specific reason, and
# the reason has to hold or this should be reverted:
#
# The 2021 floor was never a judgement about the right amount of
# history. It was a DATA constraint, stated at the time -- load_depth_
# charts() is thin before 2021, and `pos_rank` was the only feature that
# needed it. That constraint is now void, because pos_rank failed at
# every position (p = 0.16 to 0.54) and is being removed from the specs
# entirely. Removing a constraint whose stated justification no longer
# applies is not the same as widening until something passes.
#
# What keeps it honest: alpha stays at 0.10, the candidate list does not
# grow, and this is recorded as a SECOND look at the same hypotheses.
# A feature that clears alpha here and did not at five classes deserves
# more suspicion than one that cleared both times, not less.
#
# THE TRAP THIS OPENS, AND WHY pos_rank MUST GO
# ---------------------------------------------
# If pos_rank were kept while the window widened, `depth_chart_missing`
# would stop meaning "this rookie was buried" and start meaning "this
# season predates good depth chart coverage" -- a season indicator
# wearing a feature's name. It would likely test significant, because
# early classes differ from late ones for a hundred reasons, and the
# coefficient would be uninterpretable. fit_rookie_weights now refuses
# to use it when its missingness is concentrated by class; see
# SEASON_CONFOUND_SPREAD there. Do not re-add pos_rank without that
# guard passing.
#
# HONEST EXPECTATION: QB probably still will not fit. It is ~4 rookie
# quarterbacks per class clearing MIN_GAMES, so nine classes gives ~36
# against MIN_ROWS_TO_FIT=40. TE (~8.8/class -> ~79) is the position
# this actually stands to help.
COHORT_SEASONS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

# Same bar as fit_weights.MIN_GAMES, and for the same reason: a rookie
# who played three games and got hurt tells us about injury luck, not
# about whether his landing spot was good.
#
# It bites HARDER here than it does on veterans, and asymmetrically. A
# veteran who loses his job still dresses; a rookie who loses his job is
# inactive. So this filter removes disproportionately many Day 3 picks
# and busts, which means the surviving sample is a sample of rookies who
# were given a chance -- and the intercept is a statement about THEM,
# not about all rookies. That is the same selection artifact fit_weights
# documents for QB under SUPPRESS_LEVEL_SHIFT, and it is handled the
# same way there. See fit_rookie_weights.SUPPRESS_LEVEL_SHIFT.
MIN_GAMES = 8


def load_rookie_class(season):
    """
    Drafted offensive rookies for one class, with the team they were on
    at the START of their rookie season.

    Draft team is deliberately NOT used. A handful of players are traded
    between the draft and week 1, and for those the draft team is the
    wrong landing spot -- which is the entire thing this file is trying
    to measure. get_team_as_of_season() reads the first regular-season
    roster snapshot, same as the veteran backtest.

    Undrafted free agents are EXCLUDED here, unlike the live path which
    floors them to round 7. They are not a random sample of round-7-ish
    talent -- an UDFA who accumulates 8+ games is a genuine outlier, and
    including them would put a hand-picked group of overperformers into
    the training set under a round label they did not earn. The live
    path still projects them; it just does not get to claim this fit
    supports the number.
    """
    draft_picks = nfl.load_draft_picks().filter(pl.col("season") == season)

    rookies = draft_picks.select([
        pl.col("gsis_id").alias("player_id"),
        pl.col("pfr_player_name").alias("player_name"),
        "position",
        "round",
        "pick",
    ]).filter(
        pl.col("player_id").is_not_null()
        & pl.col("position").is_in(OFFENSE_POSITIONS)
    )

    team = get_team_as_of_season(season)
    rookies = rookies.join(team, on="player_id", how="inner")

    return rookies.with_columns(pl.lit(season).alias("season"))


def season_ppg(seasons):
    """
    Actual rookie-season PPG and games played, under the league's
    scoring. Regular season only -- playoff games are not part of any
    fantasy season this project builds a board for, and including them
    would reward players on good teams for reasons the features cannot
    see.
    """
    stats = nfl.load_player_stats(seasons).filter(pl.col("season_type") == "REG")
    config = load_config(CONFIG_PATH)

    stats = stats.with_columns(
        pl.struct(stats.columns)
        .map_elements(
            lambda row: calculate_offensive_points(row, config),
            return_dtype=pl.Float64,
        )
        .alias("fantasy_points")
    )

    return (
        stats.group_by(["player_id", "season"])
        .agg([
            pl.col("fantasy_points").sum().alias("total_points"),
            pl.len().alias("games_played"),
        ])
        .with_columns(
            (pl.col("total_points") / pl.col("games_played")).alias("actual_ppg")
        )
        .rename({"games_played": "actual_games_played"})
        .select(["player_id", "season", "actual_ppg", "actual_games_played"])
    )


def round_bucket(column="round"):
    """Round 1 / Day 2 (2-3) / Day 3 (4-7), matching rookies.py."""
    return (
        pl.when(pl.col(column) == 1).then(pl.lit("Round 1"))
        .when(pl.col(column).is_in([2, 3])).then(pl.lit("Day 2"))
        .otherwise(pl.lit("Day 3"))
        .alias("round_bucket")
    )


def leave_one_class_out_baselines(observed):
    """
    Cohort baseline for each class, computed WITHOUT that class.

    `observed` is one row per rookie with position, round, season, and
    actual_ppg. For target season S the baseline for a (position, round)
    cell is the mean actual_ppg of players in that cell from every class
    EXCEPT S.

    THE POINT. `delta` is what the regression predicts, and it is
    (actual - baseline). If the baseline for the 2023 class is an
    average that includes 2023, then every 2023 residual has already had
    part of itself subtracted out, and the more extreme a class was the
    harder its own baseline chases it. The features then get credit for
    variance that the baseline already absorbed. This is not a small
    effect at n=5: each class is a fifth of its own baseline.

    Implemented as (total - own) / (n - own_n) rather than by looping
    and refiltering, which is the same arithmetic and one pass.

    Cells that would be empty after removing the target class fall back
    to the POSITION's leave-one-out mean, and a `baseline_low_confidence`
    flag records where that happened -- a Day 3 tight end cell can be
    thin enough to vanish.
    """
    cell = ["position", "round"]

    totals = observed.group_by(cell).agg([
        pl.col("actual_ppg").sum().alias("cell_sum"),
        pl.len().alias("cell_n"),
    ])
    own = observed.group_by(cell + ["season"]).agg([
        pl.col("actual_ppg").sum().alias("own_sum"),
        pl.len().alias("own_n"),
    ])

    position_totals = observed.group_by("position").agg([
        pl.col("actual_ppg").sum().alias("pos_sum"),
        pl.len().alias("pos_n"),
    ])
    position_own = observed.group_by(["position", "season"]).agg([
        pl.col("actual_ppg").sum().alias("pos_own_sum"),
        pl.len().alias("pos_own_n"),
    ])

    loo = (
        own.join(totals, on=cell, how="left")
        .join(position_own, on=["position", "season"], how="left")
        .join(position_totals, on="position", how="left")
        .with_columns([
            (pl.col("cell_n") - pl.col("own_n")).alias("loo_n"),
            (pl.col("pos_n") - pl.col("pos_own_n")).alias("pos_loo_n"),
        ])
        .with_columns([
            pl.when(pl.col("loo_n") > 0)
            .then((pl.col("cell_sum") - pl.col("own_sum")) / pl.col("loo_n"))
            .otherwise(
                (pl.col("pos_sum") - pl.col("pos_own_sum")) / pl.col("pos_loo_n")
            )
            .alias("cohort_baseline_ppg"),
            (pl.col("loo_n") < MIN_CONFIDENT_N).alias("baseline_low_confidence"),
            pl.col("loo_n").alias("baseline_cohort_n"),
        ])
    )

    return loo.select(
        cell + ["season", "cohort_baseline_ppg", "baseline_low_confidence",
                "baseline_cohort_n"]
    )


def week_one_depth_chart(season):
    """
    Each player's depth-chart rank in the EARLIEST snapshot of `season`.

    LOAD_DEPTH_CHARTS RETURNS TWO DIFFERENT SCHEMAS (found Aug 6, by
    running it). This is not a version skew that will settle down --
    nflverse rebuilt the historical depth charts from a different source
    than the live scrape, so the two shapes are permanent and split
    right through the middle of anything that touches both:

        historical   season, club_code, week, game_type, depth_team,
                     position, depth_position, gsis_id, formation, ...
        live (2026)  team, dt, pos_abb, pos_rank, gsis_id, ...

    `rookies.get_latest_depth_chart()` reads the live shape and works,
    which is exactly why this went unnoticed: the live path is the only
    one the project had until Phase 12 asked for history. Both are
    handled here rather than in rookies.py, because that function is
    known-good against the shape it actually sees and a speculative
    rewrite of working code is how you break a pipeline that runs.

    THE SILVER LINING. The historical shape carries `week`, which is a
    better answer to leak #2 than `dt` ever was. `dt` is a scrape
    timestamp, so "earliest scrape" could be any time in the preseason
    and might sit later than the August snapshot the live model reads.
    Week 1 of the regular season is an actual, dated, pre-outcome
    moment. Where `week` exists this uses it.

    The offensive-position filter and the one-row-per-player de-dup are
    not optional in either shape: a player who also returns kicks
    appears in one snapshot as WR, KR and PR, and those extra rows fan
    out through every downstream join that keys on player_id.
    """
    depth_charts = nfl.load_depth_charts(seasons=[season])
    columns = set(depth_charts.columns)

    if {"club_code", "depth_team"} <= columns:
        # Historical shape. `depth_team` is the rank within the position
        # group and arrives as a string on some seasons, so it is cast
        # rather than assumed. Regular season only -- preseason depth
        # charts list everyone in camp and rank them by jersey number as
        # often as by role.
        frame = depth_charts.rename({"club_code": "team"})
        if "game_type" in columns:
            frame = frame.filter(pl.col("game_type") == "REG")
        frame = normalize_team_column(frame)

        position_column = "position" if "position" in columns else "depth_position"
        frame = frame.filter(pl.col(position_column).is_in(OFFENSE_POSITIONS))

        earliest = frame.group_by("team").agg(pl.col("week").min().alias("earliest"))
        return (
            frame.join(earliest, on="team")
            .filter(pl.col("week") == pl.col("earliest"))
            .select([
                pl.col("gsis_id").alias("player_id"),
                pl.col("depth_team").cast(pl.Float64, strict=False).alias("pos_rank"),
            ])
            .unique(subset=["player_id"], keep="first")
        )

    # Live shape -- same as rookies.get_latest_depth_chart(), with `dt`
    # minimized instead of maximized.
    earliest = depth_charts.group_by("team").agg(pl.col("dt").min().alias("earliest"))
    return (
        depth_charts.join(earliest, on="team")
        .filter(pl.col("dt") == pl.col("earliest"))
        .filter(pl.col("pos_abb").is_in(OFFENSE_POSITIONS))
        .select([
            pl.col("gsis_id").alias("player_id"),
            pl.col("pos_rank").cast(pl.Float64, strict=False),
        ])
        .unique(subset=["player_id"], keep="first")
    )


def veteran_ppg_for_competition(season):
    """
    Prior-season PPG for everyone on a roster at the start of `season`,
    keyed by the team they are on NOW -- the input
    compute_position_competition() expects.

    Note what this measures for a rookie: the average prior-season PPG
    of the players he has to beat out, on the roster he actually landed
    on. A rookie running back behind a back who averaged 16 PPG last
    year is in a materially different situation from one behind a
    committee that averaged 6, and this is the only feature in the set
    that says so directly.

    Players with no prior-season stats -- other rookies, mostly -- come
    through as 0.0 rather than null. That is the right value: an
    incumbent who has never produced is not competition, and nulling
    them would drop the entire row in the fit.
    """
    prior = season_ppg([season - 1]).select(
        ["player_id", pl.col("actual_ppg").alias("fantasy_points_per_game")]
    )

    roster = get_team_as_of_season(season)
    positions = nfl.load_players().select([
        pl.col("gsis_id").alias("player_id"), "position",
    ]).filter(pl.col("position").is_in(OFFENSE_POSITIONS))

    return (
        roster.join(positions, on="player_id", how="inner")
        .join(prior, on="player_id", how="left")
        .with_columns(pl.col("fantasy_points_per_game").fill_null(0.0))
    )


def build_rookie_backtest_season(season):
    """
    One class: rookies, their landing-spot features, and their outcome.

    Everything here is resolved as of the START of `season`. Team
    tendency and O-line snaps come from season-1; position competition
    reads season-1 production for the players on the season roster; the
    depth chart is the earliest snapshot of season itself.
    """
    reference_season = season - 1

    rookies = load_rookie_class(season)

    tendency = (
        compute_team_tendency([reference_season])
        .filter(pl.col("season") == reference_season)
        .drop("season")
    )
    oline = compute_oline_continuity(
        [reference_season], resolve_oline_current_teams_historical(season)
    )
    team_features = tendency.join(oline, on="team", how="left")

    competition_pool = veteran_ppg_for_competition(season)
    competition = compute_position_competition(competition_pool)

    depth = week_one_depth_chart(season)
    age = compute_age(season)

    combined = (
        rookies.join(team_features, on="team", how="left")
        .join(competition, on="player_id", how="left")
        .join(depth, on="player_id", how="left")
        .join(age, on="player_id", how="left")
        .with_columns([
            round_bucket(),
            # A rookie absent from his team's first depth chart snapshot
            # is not "rank 0" and not missing at random -- he is buried,
            # and being buried is exactly what we want the model to see.
            # `pos_rank` stays null and is mean-imputed at fit time; this
            # flag is its required companion, so "not listed" gets priced
            # separately from "listed 3rd" rather than the row being
            # dropped or the null silently reading as average.
            pl.col("pos_rank").is_null().alias("depth_chart_missing"),
        ])
    )

    return combined


def build_rookie_backtest(seasons=COHORT_SEASONS):
    """
    Stacks every class, attaches outcomes, and computes the
    leave-one-class-out baseline and `delta`.

    Order matters: the baseline is computed AFTER the MIN_GAMES filter,
    over exactly the rows that will be fitted. A baseline built on all
    rookies but applied to the 8+ game survivors would be systematically
    too low -- the players it was averaging over include everyone who
    never got on the field -- and the whole model would read as "rookies
    beat their cohort," which is a statement about the filter.
    """
    classes = [build_rookie_backtest_season(s) for s in seasons]
    rookies = pl.concat(classes, how="vertical")

    outcomes = season_ppg(seasons)
    rookies = rookies.join(outcomes, on=["player_id", "season"], how="inner")

    before = rookies.height
    rookies = rookies.filter(pl.col("actual_games_played") >= MIN_GAMES)
    print(f"MIN_GAMES={MIN_GAMES}: kept {rookies.height} of {before} "
          f"drafted offensive rookies with a snap")

    baselines = leave_one_class_out_baselines(
        rookies.select(["position", "round", "season", "actual_ppg"])
    )
    rookies = rookies.join(baselines, on=["position", "round", "season"], how="left")

    return rookies.with_columns(
        (pl.col("actual_ppg") - pl.col("cohort_baseline_ppg")).alias("delta")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the Phase 12 rookie training set."
    )
    parser.add_argument(
        "--seasons", type=int, nargs="+", default=COHORT_SEASONS,
        help=f"rookie classes to train on (default: {COHORT_SEASONS}). "
             f"Earlier classes are limited by load_depth_charts() coverage.",
    )
    args = parser.parse_args()

    dataset = build_rookie_backtest(args.seasons)

    print(f"\nRookie backtest: {dataset.height} rookie-seasons")
    print(dataset.group_by("season").len().sort("season"))
    print("\nBy position:")
    print(dataset.group_by("position").agg([
        pl.len().alias("n"),
        pl.col("delta").mean().round(2).alias("mean_delta"),
        pl.col("delta").std().round(2).alias("sd_delta"),
        pl.col("depth_chart_missing").sum().alias("no_depth_chart"),
    ]).sort("position"))

    # DEPTH CHART COVERAGE BY CLASS. Printed because the widened window
    # makes it the single most likely way to fool ourselves: if
    # `depth_chart_missing` is 90% in 2017 and 5% in 2024, then it is a
    # season label, not a feature, and any coefficient on it is measuring
    # the difference between eras. fit_rookie_weights refuses to use it
    # in that case, but seeing the numbers is what makes the refusal
    # legible rather than magic.
    coverage = (
        dataset.group_by("season")
        .agg([
            pl.len().alias("n"),
            pl.col("depth_chart_missing").mean().alias("missing_rate"),
        ])
        .sort("season")
    )
    print("\nDepth chart coverage by class (missing_rate near 0 is good):")
    print(coverage)
    spread = (coverage["missing_rate"].max() or 0.0) - (coverage["missing_rate"].min() or 0.0)
    if spread > SEASON_CONFOUND_SPREAD:
        print(f"  spread across classes: {spread:.2f} -- ABOVE the "
              f"{SEASON_CONFOUND_SPREAD:.2f} bar. pos_rank is behaving as a season "
              f"proxy and fit_rookie_weights will drop it.")
    else:
        print(f"  spread across classes: {spread:.2f} -- within the "
              f"{SEASON_CONFOUND_SPREAD:.2f} bar, so pos_rank is a real feature "
              f"here and will be fitted.")

    # The leave-one-class-out baseline should NOT reconcile to zero --
    # that is the difference between it and an in-sample one, and seeing
    # the gap is how you confirm the leakage guard is actually engaged.
    print(f"\nmean delta overall: {dataset['delta'].mean():+.3f}  "
          f"(an in-sample baseline would force this to ~0.000)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_csv(OUTPUT_PATH)
    print(f"\nWrote {OUTPUT_PATH}")
    print("Next: python -m src.fit_rookie_weights")
