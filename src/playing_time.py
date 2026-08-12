"""
Phase 13.5a. The playing-time model: what fraction of a drafted rookie's
season actually happens, and what that does to a number the board has
been treating as if it always does.

WHY THIS EXISTS, IN ONE PARAGRAPH
---------------------------------
`rookie_backtest.py` filters to `actual_games_played >= MIN_GAMES` (8),
and computes the leave-one-class-out cohort baseline AFTER that filter --
deliberately, and for a stated reason: a baseline built on all rookies
but applied to the survivors would read as "rookies beat their cohort."
That reasoning is correct for the thing it was protecting. What it also
did, silently, was define the shipped rookie projection as

    E[PPG | this rookie played at least half a season]

and then hand that number to `build_board`, which applies it to every
rookie on the board -- including the ones who will not play half a
season, which is most of them.

THE SIZE OF THE HOLE, MEASURED (Aug 12)
---------------------------------------
Against the files already in data/:

    rookie_backtest_features.csv   415 rows, P(>=8 games) = 100.0%
    backtest_features.csv         4177 rows, P(>=8 games) =  66.4%

Zero percent of the rookie training sample failed the availability test.
Thirty-four percent of the veteran sample did. The two baselines the
board compares against each other are therefore not the same kind of
number, and the difference runs entirely in the rookies' favour.

The filter's own comment says this out loud -- "the surviving sample is
a sample of rookies who were given a chance -- and the intercept is a
statement about THEM, not about all rookies." This module is the part
that was left undone: measuring how many rookies are not THEM.

    per rookie_backtest.build_rookie_backtest, the printed line reads
    "kept 415 of 600 drafted offensive rookies with a snap"

so 185 rookies WITH A SNAP were already excluded, and the rookies with
no snap at all never reached that 600 -- they are dropped one step
earlier, by the inner join to outcomes. Both groups are recovered here.

WHAT THIS MODULE DOES *NOT* DO
------------------------------
It does not fold availability into `adjusted_fantasy_points_per_game`.
`build_board.compute_expected_points` states the rule and the reason:
PPG is a RATE, a player who misses games is not worse per game, and
folding availability into the rate corrupts the one quantity the whole
project is fitted to predict. That rule is not relaxed here.

Instead this module produces two separable things, and they land in
different places on purpose:

  1. `expected_games` for rookies -- currently `fantasy_season_length`
     for every rookie, i.e. the model assumes a seventh-round tight end
     plays all 14 games. Replaced with a fitted expectation. This moves
     Exp Pts and NOTHING else, exactly like the PUP/NFI machinery in
     Phase 11 CP8 that it is deliberately built alongside.

  2. A MIN_GAMES sweep on the rookie cohort baseline itself -- the
     diagnostic that says whether the 8-game bar is choosing the
     projection. This one CAN move rank, so it ships nothing on its own.
     It writes a table, the holdout decides, and the decision is a
     separate commit.

Nothing in here is wired into a board until `--gate` has passed.

AND THE GATE IS THIS MODULE'S OWN, NOT holdout.py's. `holdout.py` tests
whether a FEATURE earns its slot by ablation; Phase 13.5 asks which
POPULATION a baseline should be estimated on, and there is no feature to
ablate. The Aug 12 `holdout --gate` run passed, and what it passed was
the veteran model and the rookie-TE model -- the two entries in
`holdout.MODELS`, neither of which knows this file exists. That run is
evidence v13 still holds. It is not evidence about anything here.
See `run_gate()`.

THE SELECTION CHAIN, STATED ONCE
--------------------------------
A drafted offensive rookie has to clear three filters to reach the
current training set, and each one is a place value leaks out:

    drafted                       load_rookie_class()
      -> appears on a week-1-ish roster snapshot   inner join to team
      -> recorded a stat line that season          inner join to outcomes
      -> played >= 8 games                         MIN_GAMES filter

`build_playing_time_universe()` keeps every drafted rookie and records
which filters he cleared, so the attrition is a column rather than an
absence. A player who cleared none of them is a real observation with
`actual_games_played = 0`, and he is the observation the board is
currently missing.

USAGE
-----
    python -m src.playing_time                 # build universe + fit
    python -m src.playing_time --sweep         # MIN_GAMES diagnostic only
    python -m src.playing_time --gate          # the Phase 13.5 gate
    python -m src.playing_time --no-write      # print, write nothing
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

# `src.rookie_backtest` is imported LAZILY, inside the functions that
# need it, and that is not a style preference.
#
# It pulls `nflreadpy`, which is a network-backed data client. `build_board`
# imports this module for `expected_games_for_rookies` -- two functions that
# need `json`, `polars` and arithmetic and nothing else. A module-level
# import here would put a data client into the import graph of every board
# build, every `sanity_top_n` run, and anything else that touches
# `build_board`, so a missing or broken nflreadpy would stop a draft board
# from rendering over a dependency it never uses.
#
# The board-side surface of this file is therefore dependency-free by
# construction. The analysis side pays for what it uses, where it uses it.
def _rookie_backtest():
    from src import rookie_backtest

    return rookie_backtest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNIVERSE_PATH = PROJECT_ROOT / "data" / "playing_time_universe.csv"
SWEEP_PATH = PROJECT_ROOT / "data" / "rookie_availability_sweep.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "playing_time.json"

# Deliberately NOT data/holdout_gate.json. Two gates answering two
# different questions must not share a file -- whichever ran last would
# silently claim to be the state of both.
GATE_PATH = PROJECT_ROOT / "data" / "playing_time_gate.json"

# Significance threshold for keeping a feature, matching fit_weights.ALPHA
# and fit_rookie_weights.ALPHA.
#
# CORRECTED (Aug 12): the first version of this comment called it a "ridge
# penalty" and said it was "the project's one regularisation constant." It
# is neither. `fit_weights.fit_position` calls plain `sm.OLS(...).fit()`
# and uses ALPHA as a p-value cutoff in its two-stage keep/drop -- there is
# no ridge anywhere in this project. The number was right and the sentence
# explaining it was invented. Recorded rather than quietly deleted,
# because a confident wrong comment is worse than no comment and the only
# defence against the next one is noticing this one.
ALPHA = 0.10

# Cells thinner than this are FLAGGED, and for availability they fall back
# to the fitted pick model. They no longer fall back to the position-wide
# rate, and the first run is why.
#
# THE FIRST RUN GOT ROUND 1 BACKWARDS (Aug 12)
# --------------------------------------------
# MIN_CELL_N was 15 and the fallback was the position mean. Round-1 RB and
# Round-1 TE are ~1.5 and ~1.2 players per class, so leave-one-class-out
# left ~12 and ~10 -- both under the bar -- and both fell back to their
# whole position. The result was printed with a straight face:
#
#     RB Round 1   P(>=8gm) 62.6%   E[games]  9.41   source=position
#     RB Day 2     P(>=8gm) 74.4%   E[games] 11.36   source=cell
#
# A first-round running back rated LESS available than a second-rounder.
# The same inversion hit TE (53.1% against Day 2's 65.0%). The tell was
# sitting in the sweep the whole time: RB Round 1's cohort baseline is
# 14.92 at EVERY threshold from 0 to 12, which can only happen if
# essentially every round-1 back plays twelve games. Their true
# availability is near the top of the range and they were assigned the
# bottom of it.
#
# Two changes, and the second matters more than the first:
#
#   1. The bar drops to 8. These are MEANS. A mean over twelve first-round
#      backs is thin; a mean over 168 backs of all rounds is a mean of the
#      wrong thing, and being precise about the wrong population is worse
#      than being imprecise about the right one.
#
#   2. Availability falls back to the FITTED PICK MODEL, not the position
#      mean. The fit is monotone in draft position by construction, so it
#      cannot produce "round 1 is worse than round 2" no matter how thin
#      the cell gets. It puts a pick-20 back at ~80% available share
#      against the position mean's 55%.
#
# Realisation ratios do NOT substitute at all -- a thin cell reports its
# own number with `thin: true` and its own real n. Silently swapping the
# population while keeping the label is the failure this comment exists
# to prevent, and it is the same class of error as the three
# `compute_replacement_ranks` bugs recorded on Aug 6.
MIN_CELL_N = 8

# The sweep's grid. 8 is what ships today; 0 keeps every rookie who
# recorded a snap. The interesting question is whether the baselines move
# smoothly across this or fall off a step somewhere, because a step means
# the threshold is choosing the answer.
#
# 1 WAS DROPPED after the first run printed it identical to 0 in all
# twelve cells. That is not a coincidence to note, it is arithmetic: the
# sweep operates on `took_a_snap`, so ">= 0 games" and ">= 1 game"
# describe the same population by construction. A grid point that cannot
# differ from its neighbour is a column of noise in a table people read
# for steps.
SWEEP_THRESHOLDS = [0, 2, 4, 6, 8, 10, 12]

# How far availability may run BACKWARDS across draft capital before the
# run stops rather than warns.
#
# Zero tolerance would be the obvious choice and it is the wrong one. A
# round-1 cell holds ~12 players over nine classes; two torn ACLs in that
# cell is a real 5-point dip that means nothing about draft capital, and
# halting a board build the night before a draft over sampling noise
# would make this check something to be disabled rather than trusted.
#
# 10 points is chosen to sit above that noise and well below the failure
# it exists to catch: the first run's inversion was RB Round 1 at 55.4%
# against Day 2's 66.8%, an 11.4-point reversal produced by substituting
# a different population. Noise wobbles; a wrong population steps.
MONOTONE_TOLERANCE = 0.10

# Which observed cell an undrafted rookie inherits.
#
# WAS `UDFA_EFFECTIVE_PICK = 245` AND IT PUT ELEVEN QUARTERBACKS AT ZERO
# (Aug 12, caught by the first v14 build)
# ---------------------------------------------------------------------
# The original idea was to reuse the fitted pick curve by giving a UDFA a
# round-7 pick number. `rookies.py` already floors UDFAs to round 7 for
# cohort baselines, so it looked like the same convention expressed once.
#
# It is not, because the curve is a LINEAR probability model and linear
# probability models go negative:
#
#     QB   slope -0.002785/pick   share = 0 at pick 230
#     TE   slope -0.003111/pick   share = 0 at pick 278
#     WR   slope -0.002479/pick   share = 0 at pick 349
#     RB   slope -0.002105/pick   share = 0 at pick 402
#
# **The quarterback curve crosses zero at pick 230, which is inside the
# draft and inside the fit's own training support.** At 245 it is
# negative, clipped to 0.000, and the board printed Exp Gm 0 / Exp Pts 0
# for eleven undrafted rookie QBs -- the value it reserves for OUT_SEASON.
# "This man will not play a single snap" is an assertion, and the data
# does not support it: the worst QB cell actually MEASURED is Day 3 at
# 12.1%, not zero.
#
# There were 152 pickless rookies of 232, not the six I saw when I only
# looked at the top of the 32-team board. A tail case that is two thirds
# of the population is not a tail case.
#
# THE FIX IS TO STOP EXTRAPOLATING AND START INHERITING. A UDFA is
# floored to round 7 for his cohort baseline, so he is floored to the
# round-7 -- Day 3 -- observed AVAILABILITY too. Same convention, both
# places, and it is a measurement rather than a line extended past where
# it means anything.
UDFA_ROUND_BUCKET = "Day 3"

# RETIRED in Phase 13.5b. This was the floor that stopped the linear model
# predicting negative availability, and the logit refit removed the thing
# it was defending against -- a logistic curve cannot reach 0 for any
# finite pick. The constant stays only so `availability` consumers that
# still read a floor get a harmless one; nothing in the board path uses
# it now. The comment it replaced said the real fix was a logit "in Phase
# 13.5b, not in a hotfix the night before a draft," and that is what
# happened.
CLIP_TO_OBSERVED_FLOOR = False

# Availability is a season-length-independent quantity here: it is a
# SHARE of the NFL regular season, not a game count, so that a board
# whose fantasy season is 12 or 14 weeks scales it correctly rather than
# inheriting a 17-game number. `build_board` multiplies it back out.
NFL_REGULAR_SEASON_GAMES = 17


def build_playing_time_universe(seasons=None):
    """
    Every drafted offensive rookie, whether or not he ever played.

    This is `rookie_backtest.build_rookie_backtest` with two joins
    loosened and nothing else changed. The features are built by the
    same function, from the same sources, as of the same instant -- so
    a row here and the corresponding row there agree exactly, and any
    difference in a fitted number is attributable to the population
    rather than to a second feature pipeline drifting from the first.

    THE TWO LOOSENED JOINS

      team snapshot   `load_rookie_class` inner-joins the week-1 roster,
                      so a rookie who never appeared on one vanishes.
                      He is exactly the player this module exists to
                      count. Recovered by re-reading the draft list and
                      left-joining the feature frame onto it.

      outcomes        joined LEFT, and `actual_games_played` filled with
                      0 rather than dropped. A null here means "no stat
                      line all season," which is a measurement, not a
                      missing value.

    `actual_ppg` is deliberately left NULL for a player with no games.
    Zero would be a lie -- he did not average zero points per game, he
    has no per-game rate at all -- and a zero would drag every mean that
    touches it. Availability and rate are kept apart here for the same
    reason `build_board` keeps them apart.
    """
    rb = _rookie_backtest()
    seasons = list(seasons or rb.COHORT_SEASONS)
    min_games = rb.MIN_GAMES

    drafted = pl.concat(
        [load_rookie_class_unfiltered(season) for season in seasons],
        how="vertical",
    )

    featured = pl.concat(
        [rb.build_rookie_backtest_season(season) for season in seasons],
        how="vertical",
    )

    feature_columns = [c for c in featured.columns if c not in drafted.columns]
    universe = drafted.join(
        featured.select(["player_id", "season", *feature_columns]),
        on=["player_id", "season"],
        how="left",
    )

    outcomes = rb.season_ppg(seasons)
    universe = universe.join(outcomes, on=["player_id", "season"], how="left")

    universe = universe.with_columns([
        rb.round_bucket(),
        pl.col("actual_games_played").fill_null(0).alias("actual_games_played"),
    ]).with_columns([
        (pl.col("actual_games_played") > 0).alias("took_a_snap"),
        (pl.col("actual_games_played") >= min_games).alias("in_training_set"),
        (pl.col("actual_games_played") / NFL_REGULAR_SEASON_GAMES)
        .clip(0.0, 1.0)
        .alias("available_share"),
    ])

    total = universe.height
    snaps = int(universe.select(pl.col("took_a_snap").sum()).item())
    trained = int(universe.select(pl.col("in_training_set").sum()).item())
    print(
        f"Playing-time universe: {total} drafted offensive rookies "
        f"{seasons[0]}-{seasons[-1]}\n"
        f"   took a snap        : {snaps:4d}  ({snaps / total:5.1%})\n"
        f"   >= {min_games} games (fitted): {trained:4d}  ({trained / total:5.1%})\n"
        f"   never played       : {total - snaps:4d}  ({1 - snaps / total:5.1%})\n"
        f"   The shipped rookie projection is estimated on the "
        f"{trained / total:.0%} and applied to the 100%."
    )
    return universe


def load_rookie_class_unfiltered(season):
    """
    `load_rookie_class` without the roster-snapshot inner join.

    Team is still attached where a snapshot exists, because the features
    need it; where it does not, `team` is null and every team feature
    for that row is null with it. That is correct -- a player who was
    never on a roster has no landing spot -- and it is why the
    availability model below is fitted on draft capital and position,
    which every drafted player has, rather than on landing-spot features
    that only the survivors have. Fitting availability on a feature only
    the available possess is the same bug one level down.
    """
    import nflreadpy as nfl

    from src.rookies import OFFENSE_POSITIONS

    draft_picks = nfl.load_draft_picks().filter(pl.col("season") == season)

    return draft_picks.select([
        pl.col("gsis_id").alias("player_id"),
        pl.col("pfr_player_name").alias("player_name"),
        "position",
        "round",
        "pick",
        pl.lit(season).alias("season"),
    ]).filter(
        pl.col("player_id").is_not_null()
        & pl.col("position").is_in(OFFENSE_POSITIONS)
    )


def availability_table(universe, model):
    """
    P(snap), P(>= MIN_GAMES) and E[games] per position x round bucket,
    each computed LEAVE-ONE-CLASS-OUT.

    Leave-one-class-out is not decoration. The 2026 class is being
    projected by this table, and a rate that included 2026 would be
    scoring a class against itself. Every other fitted quantity in this
    project holds a season out; so does this one.

    A cell always reports its OWN rates and its OWN n. What changes for a
    thin cell is `available_share`, which is taken from the fitted pick
    model instead of the cell mean -- see the MIN_CELL_N comment for the
    round-1 inversion that forced this. `source` says which happened and
    `thin` says whether the cell was under the bar, so a rate resting on
    ten players is never quoted as if it rested on a hundred.
    """
    rows = []
    seasons = sorted(universe.select("season").unique().to_series().to_list())

    for position in ["QB", "RB", "WR", "TE"]:
        pos_frame = universe.filter(pl.col("position") == position)
        if pos_frame.height == 0:
            continue

        for bucket in ["Round 1", "Day 2", "Day 3"]:
            cell = pos_frame.filter(pl.col("round_bucket") == bucket)
            for season in seasons:
                held_out = cell.filter(pl.col("season") != season)
                if held_out.height == 0:
                    continue

                thin = held_out.height < MIN_CELL_N
                observed_share = float(
                    held_out.select(pl.col("available_share").mean()).item()
                )

                # Thin cell: predict from the fitted pick model at this
                # cell's own mean pick. Monotone in draft position by
                # construction, so it cannot invert the round order.
                if thin and position in model:
                    mean_pick = held_out.select(pl.col("pick").mean()).item()
                    spec = model[position]
                    share = _logistic(
                        spec["intercept"]
                        + spec["weights"]["pick"] * (mean_pick - spec["centers"]["pick"])
                    )
                    source = "fitted"
                else:
                    share = observed_share
                    source = "cell"

                rows.append({
                    "position": position,
                    "round_bucket": bucket,
                    "season": season,
                    "n": held_out.height,
                    "thin": thin,
                    "source": source,
                    "p_snap": float(held_out.select(pl.col("took_a_snap").mean()).item()),
                    "p_min_games": float(
                        held_out.select(pl.col("in_training_set").mean()).item()
                    ),
                    "mean_games": float(
                        held_out.select(pl.col("actual_games_played").mean()).item()
                    ),
                    "observed_share": observed_share,
                    "available_share": share,
                })

    return pl.DataFrame(rows)


def realisation_ratio(universe, min_games=None):
    """
    The one number this whole module is for.

        realisation = E[season points | drafted]
                      -----------------------------------------
                      season_length x E[PPG | played >= min_games]

    Numerator: what a drafted rookie is actually worth over a season,
    counting the ones who never dressed as the zeros they were.
    Denominator: what the board currently pays for him.

    A ratio of 1.0 would mean the availability hole does not exist. Every
    point below 1.0 is the board's rookie premium, priced.

    Computed per position x round bucket and leave-one-class-out, so the
    2026 board's correction is not fitted on the 2026 board.

    NO SUBSTITUTION HERE, EVER. A thin cell reports its own ratio, its own
    n and `thin: true`. The first run substituted the position pool for
    round-1 RB and TE and printed the result under the round-1 label with
    n=168 and n=116 -- numbers larger than the entire Day 3 cell, which is
    how the substitution was caught. A ratio computed on a different
    population than its label claims is not a conservative estimate, it is
    a wrong one.
    """
    if min_games is None:
        min_games = _rookie_backtest().MIN_GAMES

    rows = []
    seasons = sorted(universe.select("season").unique().to_series().to_list())

    for position in ["QB", "RB", "WR", "TE"]:
        pos_frame = universe.filter(pl.col("position") == position)
        for bucket in ["Round 1", "Day 2", "Day 3"]:
            cell = pos_frame.filter(pl.col("round_bucket") == bucket)
            for season in seasons:
                fitted = cell.filter(pl.col("season") != season)
                if fitted.height == 0:
                    continue

                survivors = fitted.filter(pl.col("actual_games_played") >= min_games)
                if survivors.height == 0:
                    continue

                # What the board pays: the cohort rate, held for a full
                # NFL season. Season length cancels out of the ratio, so
                # the number is league-independent and a 12-week board and
                # a 14-week board can share it.
                paid = float(survivors.select(pl.col("actual_ppg").mean()).item())

                # What he is worth: total points over the whole drafted
                # cell, including the zeros, per season.
                earned = float(
                    fitted.select(
                        (pl.col("actual_ppg").fill_null(0.0)
                         * pl.col("actual_games_played")).mean()
                    ).item()
                ) / NFL_REGULAR_SEASON_GAMES

                rows.append({
                    "position": position,
                    "round_bucket": bucket,
                    "season": season,
                    "n_drafted": fitted.height,
                    "n_survivors": survivors.height,
                    "thin": fitted.height < MIN_CELL_N,
                    "paid_ppg": paid,
                    "earned_ppg_equivalent": earned,
                    "realisation": earned / paid if paid else float("nan"),
                })

    return pl.DataFrame(rows)


def decomposition_check(universe, sweep, availability, ratios):
    """
    Does rate x availability actually reproduce what rookies earned?

    The whole design rests on a decomposition -- keep PPG a rate, put
    availability in `expected_games`, multiply them in `build_board`. That
    is only legitimate if the product lands near the truth, and it cannot
    land exactly, because

        E[games x ppg]  !=  E[games] x E[ppg | played]

    whenever games and PPG are correlated, which they obviously are: the
    rookies who play more are the rookies who are good. The product is
    therefore BIASED LOW by the covariance, and the honest thing is to
    measure the bias rather than assert it is small.

    Hand-computed against the first run's numbers, over all twelve cells
    and a 14-game fantasy season, the product recovered a median of 0.89
    of the truth -- conservative by ~11%, one-directional, and roughly
    constant across cells (0.82 to 1.03). That is a far better error than
    the status quo, which overstates round-1 QB by 1.4x and day-3 QB by
    14x, and it errs in the safe direction.

    This function recomputes that check on live data so the 0.89 is a
    measurement each run rather than a number remembered from a run in
    August. If it drifts materially from 1.0 in either direction the
    covariance correction below is the knob, but it should be turned only
    with a holdout behind it.
    """
    rows = []
    for row in ratios.iter_rows(named=True):
        key = (row["position"], row["round_bucket"])
        rate_rows = sweep.filter(
            (pl.col("position") == key[0])
            & (pl.col("round_bucket") == key[1])
        )
        floor_rate = rate_rows.filter(pl.col("min_games") == 0)
        if floor_rate.height == 0:
            continue
        rate = float(floor_rate.select("cohort_baseline_ppg").item())

        share_rows = availability.filter(
            (pl.col("position") == key[0])
            & (pl.col("round_bucket") == key[1])
            & (pl.col("season") == row["season"])
        )
        if share_rows.height == 0:
            continue
        share = float(share_rows.select("available_share").item())

        proposed = rate * share
        truth = row["realisation"] * row["paid_ppg"]
        rows.append({
            "position": key[0],
            "round_bucket": key[1],
            "season": row["season"],
            "proposed_ppg_equivalent": proposed,
            "truth_ppg_equivalent": truth,
            "ratio": proposed / truth if truth else float("nan"),
        })

    check = pl.DataFrame(rows)
    if check.height:
        median = float(check.select(pl.col("ratio").median()).item())
        print(f"\nDECOMPOSITION CHECK: rate x availability recovers "
              f"{median:.2f} of actual rookie production (median over "
              f"{check.height} cell-seasons).")
        print("   Below 1.0 is the games/PPG covariance and is expected. "
              "It errs low, which is the safe direction.")
    return check


def min_games_sweep(seasons=None, thresholds=SWEEP_THRESHOLDS,
                    universe=None):
    """
    DIAGNOSTIC ONLY -- ships nothing.

    Rebuilds the leave-one-class-out cohort baseline at each candidate
    MIN_GAMES and reports how far the shipped projection moves. This is
    the rookie analogue of `baseline_weighting.py`'s threshold sweep, and
    it answers the same question that one does: is 8 a choice the data
    supports, or a constant that is choosing the answer.

    READ IT LIKE THIS. A baseline that slides smoothly as the bar drops
    is a population changing, which is expected and fine -- the number
    means something different at each threshold and 8 is as defensible as
    6. A baseline that steps is the threshold selecting a subgroup, and
    then the shipped number is an artifact of where the bar was put.

    Whichever it is, nothing changes on the strength of this table. It
    is evidence for the holdout to test, per Phase 13 CP2.

    FIRST RUN: IT STEPS, AND IT STEPS WHERE THE THEORY SAYS IT SHOULD
    ----------------------------------------------------------------
    Correlation between a cell's P(>=8 games) and the size of its step
    from threshold 0 to threshold 8 is **-0.81** across all twelve cells.
    The bar bites hardest exactly where fewest players clear it, which is
    the signature of a selection effect and not of a population that
    happens to differ.

        QB Day 3     P(>=8gm)  7.4%   baseline  7.97 -> 15.64   +96%
        QB Day 2     P(>=8gm) 33.3%   baseline  9.31 -> 12.34   +33%
        TE Day 3     P(>=8gm) 40.5%   baseline  3.97 ->  5.00   +26%
        RB Day 3     P(>=8gm) 55.6%   baseline  4.28 ->  5.30   +24%
        WR Round 1   P(>=8gm) 89.7%   baseline 10.15 -> 10.83    +7%
        RB Round 1   P(>=8gm) high    baseline 14.92 -> 14.92     0%

    Round 1 does not move at all, because nearly every first-rounder
    plays and there is no one for the filter to remove. Day 3 quarterback
    nearly doubles, because 92.6% of day-3 quarterbacks are removed and
    the survivors are Brock Purdy.

    THE ARGUMENT THIS SETTLES. MIN_GAMES=8 on the veteran side is applied
    to the BASELINE -- prior seasons, to get a stable estimate of a rate
    before predicting anything. On the rookie side the identical constant
    is applied to the OUTCOME. Those are not the same operation with the
    same justification; the second is conditioning on the dependent
    variable, and it has been hiding behind the fact that both are
    spelled `8`.
    """
    rb = _rookie_backtest()
    if universe is None:
        universe = build_playing_time_universe(seasons)
    with_snap = universe.filter(pl.col("took_a_snap"))

    rows = []
    for threshold in thresholds:
        kept = with_snap.filter(pl.col("actual_games_played") >= threshold)
        if kept.height == 0:
            continue

        baselines = rb.leave_one_class_out_baselines(
            kept.select(["position", "round", "season", "actual_ppg"])
        )
        joined = kept.join(baselines, on=["position", "round", "season"], how="left")

        for (position, bucket), cell in joined.group_by(["position", "round_bucket"]):
            rows.append({
                "min_games": threshold,
                "position": position,
                "round_bucket": bucket,
                "n": cell.height,
                "share_of_drafted": cell.height / universe.filter(
                    (pl.col("position") == position)
                    & (pl.col("round_bucket") == bucket)
                ).height,
                "cohort_baseline_ppg": float(
                    cell.select(pl.col("cohort_baseline_ppg").mean()).item()
                ),
                "mean_games": float(
                    cell.select(pl.col("actual_games_played").mean()).item()
                ),
            })

    sweep = pl.DataFrame(rows).sort(["position", "round_bucket", "min_games"])

    print("\nMIN_GAMES SWEEP -- cohort baseline PPG by threshold")
    print("(shipped threshold is 8; a STEP between rows means the bar is "
          "choosing the projection)")
    for (position, bucket), cell in sweep.group_by(
        ["position", "round_bucket"], maintain_order=True
    ):
        cell = cell.sort("min_games")
        line = "  ".join(
            f"{int(t):>2}:{v:5.2f}"
            for t, v in zip(
                cell.select("min_games").to_series().to_list(),
                cell.select("cohort_baseline_ppg").to_series().to_list(),
            )
        )
        print(f"   {position:2s} {bucket:8s}  {line}")

    return sweep


def _logit_fit(pick, games, season_length=NFL_REGULAR_SEASON_GAMES):
    """
    Binomial GLM with a logit link: games played out of a 17-game season.

    `pick` arrives already centred. Returns (intercept, slope) on the
    LOGIT scale, so a prediction is 1/(1+exp(-(a + b*(pick-centre)))).

    WHY BINOMIAL AND NOT A PROPORTION REGRESSION. `available_share` is
    games/17 -- a count out of a known denominator, which is what the
    binomial family is for. Passing endog as two columns (played, missed)
    lets statsmodels weight a player who appeared in 16 games more than
    one who appeared in 2, which a regression on the bare proportion
    treats as equally informative.
    """
    import statsmodels.api as sm

    played = np.asarray(games, dtype=float)
    missed = float(season_length) - played
    exog = sm.add_constant(np.asarray(pick, dtype=float).reshape(-1, 1))
    result = sm.GLM(
        np.column_stack([played, missed]),
        exog,
        family=sm.families.Binomial(),
    ).fit()
    return float(result.params[0]), float(result.params[1]), result


def fit_expected_games(universe, quiet=False):
    """
    Fits availability on draft capital. LOGIT link, leave-one-class-out.

    WAS A LINEAR PROBABILITY MODEL UNTIL PHASE 13.5b (Aug 12)
    --------------------------------------------------------
    And it was not merely inelegant. A linear fit on a [0,1] target
    predicts outside [0,1], and this one did so INSIDE its own training
    support: the quarterback curve crossed zero at pick 230, which is
    inside the draft. Eleven undrafted rookie QBs reached a board with
    Exp Gm 0 and Exp Pts 0 -- the value reserved for OUT_SEASON -- and a
    hardcoded floor was patched in to stop it. A logit cannot leave (0,1)
    for any input, so the failure mode is gone by construction rather
    than by clamp.

    IT ALSO FITS BETTER, AND ONLY WHERE IT WAS BROKEN. Leave-one-class-out
    predictive binomial deviance, 2017-2025, with 500-resample player
    bootstrap:

        pos    linear    logit      gain   95% CI          P(logit better)
        QB     1195.6    707.6    +488.0   [ +63, +823]    99.0%
        RB     1657.5   1653.9      +3.5   [  -6,  +20]    74.0%
        WR     2123.9   2122.9      +1.0   [ -19,  +27]    51.8%
        TE      864.7    887.3     -22.7   [ -43,   +3]     3.2%

    Read that honestly: **the logit repairs quarterback and is a coin
    flip everywhere else, and it is genuinely WORSE for tight end.** TE's
    -22.7 is not noise; the bootstrap puts it at 96.8% confidence that
    linear fits TE better.

    IT SHIPS FOR ALL FOUR POSITIONS ANYWAY, and the reason is not fit.

      - Admissibility beats deviance. A model that can output a negative
        probability is wrong whatever it scores. TE's linear form crosses
        zero at pick 278 -- outside a 262-pick draft, but by sixteen
        picks, and that margin is the only thing standing between it and
        the QB bug.
      - Keeping linear for TE means keeping the floor patch for TE. One
        form removes an entire class of failure; two forms retain it in
        one corner and add a per-position exception to defend.
      - The portfolio is overwhelmingly positive: TE gives up 0.17
        deviance per observation, QB gains 4.69.

    The cost is real and is recorded rather than rounded away. If TE ever
    earns features of its own, this is the first thing to re-test.

    THE FEATURE SET IS SHORT ON PURPOSE. `pick` and `position` are the
    only things every drafted rookie has -- including the ones who never
    reached a roster snapshot, who are the entire point. Every richer
    feature in `rookie_backtest_features.csv` (landing spot, position
    competition, depth chart, O-line) exists only for players who made a
    roster, so conditioning availability on them would re-introduce the
    survivorship this module was written to remove.

    Combine measurables and rookie-usage tendency are the exception, and
    that is exactly why they are the Phase 13.5b candidates: both exist
    for a drafted player who never took a snap.
    """
    seasons = sorted(universe.select("season").unique().to_series().to_list())
    model = {}

    for position in ["QB", "RB", "WR", "TE"]:
        frame = universe.filter(
            (pl.col("position") == position) & pl.col("pick").is_not_null()
        )
        if frame.height < MIN_CELL_N:
            if not quiet:
                print(f"   {position}: {frame.height} rows, below MIN_CELL_N -- "
                      f"no fit, cell rates only")
            continue

        pick = frame.select("pick").to_series().to_numpy().astype(float)
        games = frame.select("actual_games_played").to_series().to_numpy().astype(float)
        target = frame.select("available_share").to_series().to_numpy().astype(float)
        centre = float(pick.mean())

        intercept, slope, fitted = _logit_fit(pick - centre, games)

        per_class = {}
        for season in seasons:
            held = frame.filter(pl.col("season") != season)
            if held.height < MIN_CELL_N:
                continue
            h_pick = held.select("pick").to_series().to_numpy().astype(float)
            h_games = (held.select("actual_games_played").to_series()
                       .to_numpy().astype(float))
            h_intercept, h_slope, _ = _logit_fit(h_pick - centre, h_games)
            per_class[str(season)] = {"intercept": h_intercept, "pick": h_slope}

        slopes = [v["pick"] for v in per_class.values()]
        model[position] = {
            "link": "logit",
            "p_value_pick": float(fitted.pvalues[1]),
            "intercept": intercept,
            "weights": {"pick": slope},
            "centers": {"pick": centre},
            "n": frame.height,
            "mean_available_share": float(target.mean()),
            "stability_leave_one_class_out": per_class,
            "sign_flips": (
                ["pick"] if slopes and (min(slopes) < 0 < max(slopes)) else []
            ),
        }

        if quiet:
            continue
        print(
            f"   {position}: n={frame.height:4d}  "
            f"mean available share={target.mean():5.1%}  "
            f"logit slope={slope:+.6f}/pick  p={fitted.pvalues[1]:.2e}  "
            f"(pick 10 -> {_logistic(intercept + slope * (10 - centre)):5.1%}, "
            f"200 -> {_logistic(intercept + slope * (200 - centre)):5.1%}, "
            f"245 -> {_logistic(intercept + slope * (245 - centre)):5.1%})"
        )

    return model


def _logistic(eta):
    """Inverse logit. Never 0, never 1, never negative -- that is the point."""
    return float(1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0))))


# `_ridge_fit` and `_clip01` lived here until Phase 13.5b and are gone
# under the delete-dead-things rule. `_ridge_fit` fitted the linear
# probability model the logit replaced; `_clip01` existed to catch that
# model's out-of-range predictions. Neither has a caller now, and leaving
# a fitter on disk that nothing ships is how someone re-adopts it by
# autocomplete six weeks from now.


def load_playing_time_model(path=MODEL_PATH):
    """Shipped playing-time model, or None if the phase has not been run."""
    if not Path(path).exists():
        return None
    return json.loads(Path(path).read_text())


def expected_games_for_rookies(players, model, config):
    """
    The board-side hook, and the ONLY thing Phase 13.5 ships.

    The gate chose predictor B -- availability only, rate untouched. So
    `rookie_backtest.MIN_GAMES` stays at 8, no cohort baseline changes,
    `adjusted_fantasy_points_per_game` is not touched, and **no player's
    rank moves**. This function writes one column and that column reaches
    Exp Pts alone.

    Deliberately expressed as games MISSED, not games played, so it flows
    into the column `build_board.apply_injury_overrides` already
    populates and `compute_expected_points` already consumes. A rookie
    and a veteran on PUP end up in the same arithmetic, and there is one
    place where availability turns into Exp Pts rather than two.

    A rookie carrying BOTH a fitted availability and an injury override
    takes the LARGER absence rather than the sum. The two estimates are
    not independent -- the fitted number already averages over rookies
    who got hurt -- and adding them would charge a PUP rookie twice for
    the same four games.

    UNDRAFTED ROOKIES GET A PICK, NOT A POSITION MEAN. Six rookies on the
    32-team board have no round or pick at all. The first draft of this
    function fell back to a position-wide availability for them, which is
    the same substitution that produced the round-1 inversion -- a UDFA is
    not an average rookie, he is a worse-than-day-3 rookie. `rookies.py`
    already floors undrafted players to round 7 for cohort purposes, so
    this floors them to a round-7 PICK and runs them through the same
    fitted curve as everyone else. One rule, applied to everybody.
    """
    season_length = float(config.get("fantasy_season_length", 17))
    positions = model["positions"]

    # `pick` ARRIVES AS A STRING AND THE FIRST VERSION ASSUMED IT DID NOT
    # (Aug 12). player_features.csv is a CSV, so every column round-trips
    # as text -- which is why `prepare_board_frame` already has a loop
    # casting has_adp / is_rookie / recent_major_injury back to booleans a
    # few lines above this call. `pick` needed the same treatment and did
    # not get it, and the build died on the first rookie with
    # "unsupported operand type(s) for -: 'str' and 'float'".
    #
    # `strict=False` is doing real work rather than silencing an error.
    # An undrafted rookie has no pick, and depending on how the CSV was
    # written that is either an empty string or a null; both become null
    # here and both then take UDFA_EFFECTIVE_PICK. One path, not two.
    pick = pl.col("pick").cast(pl.Float64, strict=False)

    # Observed floors and the UDFA inheritance, both read from the shipped
    # `availability` block so there is no second copy of these numbers.
    availability = model.get("availability", {})
    floor = {}
    udfa = {}
    for key, values in availability.items():
        position, _, bucket = key.partition("|")
        share = values.get("available_share")
        if share is None:
            continue
        floor[position] = min(floor.get(position, 1.0), float(share))
        if bucket == UDFA_ROUND_BUCKET:
            udfa[position] = float(share)

    # NO map_elements. The original used a Python UDF over a struct, which
    # is how the string got as far as arithmetic in the first place -- a
    # UDF takes whatever the column holds and finds out at runtime, while
    # a native expression fails at cast time with the column named. It is
    # also several hundred times faster, though that is the lesser reason
    # on a 1088-row frame.
    share = pl.lit(None, dtype=pl.Float64)
    for position, spec in positions.items():
        # THE FLOOR PATCH IS GONE, and its removal is the point of the
        # logit refit rather than a side effect. A logistic curve cannot
        # reach 0 or 1 for any finite input, so there is nothing left to
        # clamp -- the guarantee is in the functional form instead of in
        # a constant somebody has to remember to keep correct.
        eta = (
            pl.lit(float(spec["intercept"]))
            + pl.lit(float(spec["weights"]["pick"]))
            * (pick - pl.lit(float(spec["centers"]["pick"])))
        )
        fitted = 1.0 / (1.0 + (-eta.clip(-30.0, 30.0)).exp())

        # A pickless rookie never touches the curve. He inherits the
        # measured Day 3 share, which is the same round-7 floor
        # `rookies.py` already applies to his cohort baseline.
        inherited = pl.lit(float(udfa.get(position, floor.get(position, 1.0))))

        share = (
            pl.when(pl.col("position") == position)
            .then(pl.when(pick.is_null()).then(inherited).otherwise(fitted))
            .otherwise(share)
        )

    # A position with no fitted model (K, DST, anything unmodelled) keeps a
    # full season. That is the v13 behaviour and it is the right default:
    # this phase only ever claimed to know about rookie skill players.
    share = share.clip(0.0, 1.0).fill_null(1.0)

    return players.with_columns(
        pl.when(pl.col("is_rookie"))
        .then(pl.lit(season_length) * (pl.lit(1.0) - share))
        .otherwise(pl.lit(0.0))
        .alias("rookie_games_missed")
    ).with_columns(
        pl.max_horizontal("expected_games_missed", "rookie_games_missed")
        .alias("expected_games_missed")
    ).drop("rookie_games_missed")


def selftest():
    """
    Exercises `expected_games_for_rookies` on a synthetic frame, in
    seconds, without touching nflverse or a board.

    IT EXISTS BECAUSE OF A DTYPE BUG, NOT A MATHS BUG (Aug 12). The
    availability arithmetic was right and the gate had already validated
    it on 719 players. What broke a board build was that `pick` arrives
    from a CSV as a string, so the first rookie hit
    `'str' - 'float'`. Every expensive check in this project ran clean and
    the cheap one that would have caught it did not exist.

    So this frame is built to be nasty in exactly the ways the real one
    is: picks as STRINGS, an undrafted rookie with an empty-string pick, a
    veteran who must come out at zero, a rookie already carrying a PUP
    absence, and a position with no fitted model.
    """
    # QB's slope is the real one from the Aug 12 fit, chosen so the curve
    # crosses zero at pick 230 -- the misspecification that put eleven
    # undrafted quarterbacks at Exp Pts 0 on the first v14 build. If the
    # floor or the UDFA inheritance ever regresses, these two rows fail.
    model = {
        "positions": {
            # LOGIT scale. Slopes chosen so the pick-250 QB lands at an eta
            # of about -3.4 -- deeply negative, exactly where the linear
            # model used to return a negative probability. The logistic
            # returns 3.2%: small, plausible, and crucially not zero.
            "RB": {"intercept": 0.40, "weights": {"pick": -0.009}, "centers": {"pick": 100.0}},
            "WR": {"intercept": 0.30, "weights": {"pick": -0.011}, "centers": {"pick": 100.0}},
            "QB": {"intercept": -0.70, "weights": {"pick": -0.018}, "centers": {"pick": 100.0}},
        },
        "availability": {
            "RB|Round 1": {"available_share": 0.778},
            "RB|Day 2": {"available_share": 0.668},
            "RB|Day 3": {"available_share": 0.492},
            "WR|Day 3": {"available_share": 0.403},
            "QB|Round 1": {"available_share": 0.689},
            "QB|Day 2": {"available_share": 0.327},
            "QB|Day 3": {"available_share": 0.121},
        },
    }
    players = pl.DataFrame({
        "player_name": ["early RB", "late WR", "undrafted RB", "veteran RB",
                        "PUP rookie RB", "unmodelled K",
                        "undrafted QB", "pick 250 QB"],
        "position": ["RB", "WR", "RB", "RB", "RB", "K", "QB", "QB"],
        "pick": ["10", "200", "", "5", "150", "20", None, "250"],
        "is_rookie": [True, True, True, False, True, True, True, True],
        "expected_games_missed": [0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
    })
    config = {"fantasy_season_length": 14}

    out = expected_games_for_rookies(players, model, config)
    got = dict(zip(
        out.select("player_name").to_series().to_list(),
        out.select("expected_games_missed").to_series().to_list(),
    ))

    def missed(share):
        return 14.0 * (1.0 - share)

    def logistic(intercept, slope, pick, centre):
        return 1.0 / (1.0 + np.exp(-(intercept + slope * (pick - centre))))

    expected = {
        # eta = 0.40 + -0.009*(10-100) = +1.21 -> 77.0%
        "early RB": missed(logistic(0.40, -0.009, 10, 100)),
        # eta = 0.30 + -0.011*(200-100) = -0.80 -> 31.0%
        "late WR": missed(logistic(0.30, -0.011, 200, 100)),
        # empty string -> null -> inherits the measured RB Day 3 share,
        # never the curve
        "undrafted RB": missed(0.492),
        # not a rookie: untouched
        "veteran RB": 0.0,
        # eta = 0.40 + -0.009*50 = -0.05 -> 48.8% -> 7.17 games missed,
        # which EXCEEDS the PUP 4.0, so the fitted number wins here.
        # Deliberately the opposite direction from the old linear fixture:
        # max() has to be tested from both sides or it passes as sum().
        "PUP rookie RB": missed(logistic(0.40, -0.009, 150, 100)),
        # no fitted model for K: full season
        "unmodelled K": 0.0,
        # the original bug: inherits QB Day 3 (12.1%), curve never consulted
        "undrafted QB": missed(0.121),
        # eta = -0.70 + -0.018*(250-100) = -3.40. The LINEAR model returned
        # a negative probability here and got clamped to zero. The logistic
        # returns 3.2% -- small, plausible, and not zero.
        "pick 250 QB": missed(logistic(-0.70, -0.018, 250, 100)),
    }

    failures = [
        f"{name}: expected {value:.4f}, got {got.get(name)}"
        for name, value in expected.items()
        if got.get(name) is None or abs(got[name] - value) > 1e-9
    ]
    for name, value in expected.items():
        status = "ok " if name not in " ".join(failures) else "FAIL"
        print(f"  {status} {name:16s} games missed = {got.get(name)}")

    # THE INVARIANT, checked separately because it is the one that matters
    # most and the one an expected-value table would not catch if someone
    # "fixed" the expectations to match a broken build. Exp Gm 0 is what
    # the board means by OUT_SEASON. No availability model is ever allowed
    # to say that about a healthy rookie.
    season = float(config["fantasy_season_length"])
    zeroed = [
        name for name, value in got.items()
        if value is not None and abs(value - season) < 1e-9
    ]
    if zeroed:
        failures.append(
            f"these rookies were given ZERO expected games, which is the "
            f"board's OUT_SEASON value: {zeroed}"
        )

    if failures:
        print("\nSELFTEST FAILED:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("\nSELFTEST PASSED: strings cast, UDFAs inherit the measured "
          "Day 3 share, the logistic keeps late picks inside (0,1) with no "
          "floor, veterans untouched, PUP takes the larger absence, "
          "unmodelled positions keep a full season, and no rookie was "
          "zeroed out.")
    return 0


def run_gate(seasons=None, universe=None, write=True):
    """
    The Phase 13.5 gate. A SEPARATE gate from `holdout.py`'s, on purpose.

    WHY IT CANNOT REUSE holdout.py
    ------------------------------
    `holdout.py` answers "does this FEATURE earn its slot," by ablation
    against a fitted model. Phase 13.5 asks a different question --
    "which POPULATION should the cohort baseline be estimated on, and
    should it be multiplied by an availability term" -- and there is no
    feature to ablate. Running it through the ablation machinery would
    require inventing a feature that is really a population choice, which
    is how you get a passing gate that tested nothing.

    Note carefully what the Aug 12 gate run did and did not do. It passed,
    and it passed the VETERAN model and the ROOKIE TE model -- the two
    entries in `holdout.MODELS`. Neither knows this module exists. That
    run is evidence that v13 still holds up. It is not evidence about
    anything in Phase 13.5.

    THE TEST
    --------
    Leave one rookie class out. Predict every drafted rookie in it -- the
    full population, zeros included, because that is the population the
    board applies these numbers to. Score on SEASON TOTAL POINTS rather
    than PPG, because total points is what a roster spot actually returns
    and because a player with no games has no PPG to score against but
    unambiguously scored zero points.

    Four predictors, a clean 2x2, so that the rate change and the
    availability change are each made to justify themselves separately
    rather than shipping as a bundle:

        A  rate@8  x  season length      <- what the board does today
        B  rate@8  x  fitted availability
        C  rate@0  x  season length
        D  rate@0  x  fitted availability   <- the proposal

    THE RULE, stated before the numbers are seen:
      - D must beat A, or the phase ships nothing.
      - B and C are each compared to A to say which half did the work.
        A change that does not beat A on its own does not ship on its own.
      - Every comparison is pooled across held-out classes. Single folds
        cannot resolve differences this size; see the note in main().

    THE RULE WAS INCOMPLETE, AND THE RESULT SHOWED WHERE (Aug 12)
    ------------------------------------------------------------
    It asked whether each half beat A. It did not ask whether the second
    half added anything ON TOP OF the first, and that is the comparison
    that decided the phase:

        A  rate@8 x full season      RMSE 102.01   bias +62.21
        B  rate@8 x availability      RMSE  59.41   bias  +1.30
        C  rate@0 x full season       RMSE  76.43   bias +39.72
        D  rate@0 x availability      RMSE  59.33   bias  -6.29

    D beats A, so the gate passes. But **D beats B by 0.08 RMSE, which is
    0.05 of one standard error**, while moving the mean bias from +1.30
    (0.6 SE from zero, unbiased) to -6.29 (2.8 SE from zero, biased low).
    Availability captures 99.8% of the total available improvement. The
    rate change captures 0.2% and costs the bias.

    WHY, AND THE PROJECT PREDICTED IT IN WRITING. From
    `build_board.compute_expected_points`, months before this module
    existed: folding availability into the rate "would silently
    double-count the moment a future phase models availability directly."
    That is exactly what rate@0 does. Lowering the games threshold pulls
    the rate down *because the players it admits played fewer games* --
    it is an availability correction wearing a rate's clothing. Multiply
    it by an explicit availability term and you have applied the same
    correction twice. The -6.29 is that double-count, measured.

    SO THE SHIPPING DECISION IS B, NOT D. `rookie_backtest.MIN_GAMES`
    stays at 8 and is not touched. The MIN_GAMES sweep remains a correct
    diagnostic of a real selection effect -- it just turns out the effect
    is fully absorbed by the availability term and does not want a second
    correction. And because only `expected_games` changes, **this phase
    moves no player's rank.** The design commitment made at the top of
    this file held, and the one place it was doubted is the one place the
    gate said no.

    The incremental test is now part of the rule, below, so the next
    person does not have to rediscover it.
    """
    rb = _rookie_backtest()
    seasons = list(seasons or rb.COHORT_SEASONS)
    current_min_games = rb.MIN_GAMES
    if universe is None:
        universe = build_playing_time_universe(seasons)

    season_length = float(NFL_REGULAR_SEASON_GAMES)
    rows = []

    for test_season in seasons:
        train = universe.filter(pl.col("season") != test_season)
        test = universe.filter(pl.col("season") == test_season)
        if train.height == 0 or test.height == 0:
            continue

        model = fit_expected_games(train, quiet=True)
        snap_train = train.filter(pl.col("took_a_snap"))

        rate_at = {}
        for threshold in (0, current_min_games):
            kept = snap_train.filter(pl.col("actual_games_played") >= threshold)
            rate_at[threshold] = {
                (row["position"], row["round_bucket"]): row["rate"]
                for row in kept.group_by(["position", "round_bucket"])
                .agg(pl.col("actual_ppg").mean().alias("rate"))
                .iter_rows(named=True)
            }

        for player in test.iter_rows(named=True):
            key = (player["position"], player["round_bucket"])
            rate_8 = rate_at[current_min_games].get(key)
            rate_0 = rate_at[0].get(key)
            if rate_8 is None or rate_0 is None:
                continue

            spec = model.get(player["position"])
            if spec is None or player["pick"] is None:
                continue
            share = _logistic(
                spec["intercept"]
                + spec["weights"]["pick"] * (player["pick"] - spec["centers"]["pick"])
            )

            actual = (player["actual_ppg"] or 0.0) * player["actual_games_played"]
            rows.append({
                "season": test_season,
                "position": player["position"],
                "A_rate8_full": rate_8 * season_length,
                "B_rate8_avail": rate_8 * season_length * share,
                "C_rate0_full": rate_0 * season_length,
                "D_rate0_avail": rate_0 * season_length * share,
                "actual": actual,
            })

    scored = pl.DataFrame(rows)
    if scored.height == 0:
        print("Gate: no scorable rows.")
        return None

    def rmse(column):
        return float(
            scored.select(
                ((pl.col(column) - pl.col("actual")) ** 2).mean().sqrt()
            ).item()
        )

    def bias(column):
        return float(
            scored.select((pl.col(column) - pl.col("actual")).mean()).item()
        )

    labels = {
        "A_rate8_full": "A  rate@8 x full season   (SHIPPED TODAY)",
        "B_rate8_avail": "B  rate@8 x availability",
        "C_rate0_full": "C  rate@0 x full season",
        "D_rate0_avail": "D  rate@0 x availability  (PROPOSAL)",
    }
    baseline = rmse("A_rate8_full")

    print(f"\n{'=' * 74}")
    print(f"PHASE 13.5 GATE -- {scored.height} held-out drafted rookies, "
          f"scored on season total points")
    print(f"{'=' * 74}")
    print(f"  {'predictor':44s} {'RMSE':>8s} {'vs A':>8s} {'mean bias':>10s}")
    results = {}
    for column, label in labels.items():
        r, b = rmse(column), bias(column)
        results[column] = {"rmse": r, "gain_vs_A": baseline - r, "mean_bias": b}
        print(f"  {label:44s} {r:8.2f} {baseline - r:+8.2f} {b:+10.2f}")

    passed = results["D_rate0_avail"]["gain_vs_A"] > 0
    print(f"\n  {'PASSED' if passed else 'FAILED'}: the proposal "
          f"{'beats' if passed else 'does not beat'} what ships today.")
    print(f"  Availability alone (B): {results['B_rate8_avail']['gain_vs_A']:+.2f}   "
          f"Rate alone (C): {results['C_rate0_full']['gain_vs_A']:+.2f}")
    print("  A positive mean bias is the board overpaying. Today's is the "
          "number this phase exists to remove.")

    # THE INCREMENTAL TEST. Beating A is necessary, not sufficient. The
    # question that decides what ships is whether the rate change adds
    # anything once availability is already handled -- and on Aug 12 it
    # did not, at 0.05 SE, while costing 7.6 points of bias.
    se_rmse = results["B_rate8_avail"]["rmse"] / np.sqrt(2 * scored.height)
    incremental = results["B_rate8_avail"]["rmse"] - results["D_rate0_avail"]["rmse"]
    bias_b = abs(results["B_rate8_avail"]["mean_bias"])
    bias_d = abs(results["D_rate0_avail"]["mean_bias"])
    ship = "D" if (incremental > se_rmse and bias_d <= bias_b) else "B"

    print(f"\n  INCREMENTAL -- does the rate change earn a slot on top of "
          f"availability?")
    print(f"    D over B: {incremental:+.2f} RMSE = {incremental / se_rmse:+.2f} SE "
          f"(1 SE = {se_rmse:.2f})")
    print(f"    |bias|:   B {bias_b:.2f} -> D {bias_d:.2f}")
    print(f"    SHIP {ship}: " + (
        "the rate change earns its slot."
        if ship == "D" else
        "availability only. Leave rookie_backtest.MIN_GAMES at 8 -- "
        "rate@0 is an availability correction in disguise and applying it "
        "alongside an explicit availability term double-counts, exactly as "
        "build_board.compute_expected_points warned."))
    if ship == "B":
        print("    Consequence: only expected_games changes. NO RANK MOVES.")

    results["incremental_D_over_B"] = {
        "rmse_gain": incremental,
        "se": se_rmse,
        "in_se": incremental / se_rmse,
        "abs_bias_B": bias_b,
        "abs_bias_D": bias_d,
        "ship": ship,
    }

    payload = {
        "passed": passed,
        "seasons": list(seasons),
        "n_held_out": scored.height,
        "scored_on": "season total fantasy points, full drafted population",
        "predictors": results,
        "ship": ship,
        "rule": "D must beat A, AND the second half must add more than 1 SE "
                "on top of the first without worsening bias. B and C are "
                "reported so the rate change and the availability change "
                "each justify themselves separately.",
        "note": "This is NOT holdout.py's gate. That one tests feature "
                "ablation on the veteran and rookie-TE models and knows "
                "nothing about playing time.",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    if write:
        GATE_PATH.write_text(json.dumps(payload, indent=2))
        print(f"\n  Wrote {GATE_PATH.relative_to(PROJECT_ROOT)}")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true",
                        help="run the MIN_GAMES diagnostic only")
    parser.add_argument("--gate", action="store_true",
                        help="run the Phase 13.5 gate only")
    parser.add_argument("--selftest", action="store_true",
                        help="check the board-side hook on a synthetic "
                             "frame; no nflverse, no board, seconds")
    parser.add_argument("--no-write", action="store_true",
                        help="print everything, write nothing")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(selftest())

    if args.gate:
        run_gate(write=not args.no_write)
        return

    if args.sweep:
        sweep = min_games_sweep()
        if not args.no_write:
            sweep.write_csv(SWEEP_PATH)
            print(f"\nWrote {SWEEP_PATH.relative_to(PROJECT_ROOT)}")
        return

    universe = build_playing_time_universe()

    # ORDER MATTERS. The fit comes first because availability_table falls
    # back to it for thin cells. Building the table first and the model
    # second is what produced the round-1 inversion on the first run --
    # the table had nothing monotone to lean on, so it leaned on the
    # position mean.
    print("\nFITTING expected games on draft capital")
    model = fit_expected_games(universe)

    print("\nAVAILABILITY BY POSITION AND DRAFT CAPITAL (leave-one-class-out)")
    table = availability_table(universe, model)
    summary = (
        table.group_by(["position", "round_bucket"])
        .agg([
            pl.col("p_snap").mean(),
            pl.col("p_min_games").mean(),
            pl.col("mean_games").mean(),
            pl.col("observed_share").mean(),
            pl.col("available_share").mean(),
            pl.col("n").mean().alias("cell_n"),
            pl.col("source").first(),
        ])
        .sort(["position", "round_bucket"])
    )
    print(f"   {'pos':3s} {'bucket':8s} {'cell n':>7s} {'P(snap)':>8s} "
          f"{'P(>=8gm)':>9s} {'E[games]':>9s} {'share used':>11s}  source")
    for row in summary.iter_rows(named=True):
        print(f"   {row['position']:3s} {row['round_bucket']:8s} "
              f"{row['cell_n']:7.0f} {row['p_snap']:7.1%} "
              f"{row['p_min_games']:8.1%} {row['mean_games']:9.2f} "
              f"{row['available_share']:10.1%}  {row['source']}")
    print("   `cell n` is the cell's OWN size, never a fallback pool's. "
          "source=fitted means the share came from the pick model.")

    # THE ROUND-1 SANITY CONDITION, ASSERTED RATHER THAN EYEBALLED.
    # A first-round rookie must not be assigned lower availability than a
    # day-2 rookie at the same position. This is the check that would have
    # caught the first run's inversion at the moment it happened, and it
    # is cheap enough that there is no reason it was not here already.
    for position in ["QB", "RB", "WR", "TE"]:
        cells = {
            row["round_bucket"]: row["available_share"]
            for row in summary.iter_rows(named=True)
            if row["position"] == position
        }
        ordered = [cells.get(b) for b in ["Round 1", "Day 2", "Day 3"]]
        present = [v for v in ordered if v is not None]
        inversion = max(
            (present[i + 1] - present[i] for i in range(len(present) - 1)),
            default=0.0,
        )
        if inversion > MONOTONE_TOLERANCE:
            raise ValueError(
                f"{position}: availability inverts by {inversion:.1%} across "
                f"draft capital -- Round 1 {cells.get('Round 1')}, "
                f"Day 2 {cells.get('Day 2')}, Day 3 {cells.get('Day 3')}.\n"
                f"Earlier picks play more. A violation this large means a "
                f"thin cell is borrowing from the wrong population again -- "
                f"check MIN_CELL_N and the fallback in availability_table()."
            )
        if inversion > 0:
            print(f"   NOTE: {position} availability inverts by "
                  f"{inversion:.1%}, inside the {MONOTONE_TOLERANCE:.0%} "
                  f"tolerance. Small and plausibly real; not blocking.")

    print("\nREALISATION -- what the board pays vs what a drafted rookie earns")
    ratios = realisation_ratio(universe)
    ratio_summary = (
        ratios.group_by(["position", "round_bucket"])
        .agg([
            pl.col("realisation").mean(),
            pl.col("n_drafted").mean().alias("n_drafted"),
            pl.col("thin").any(),
        ])
        .sort(["position", "round_bucket"])
    )
    for row in ratio_summary.iter_rows(named=True):
        flag = "  THIN" if row["thin"] else ""
        print(f"   {row['position']:3s} {row['round_bucket']:8s} "
              f"realisation={row['realisation']:5.2f}  "
              f"n={row['n_drafted']:5.0f}{flag}")
    print("   1.00 would mean no availability hole. Every point below is the "
          "board's rookie premium.")

    sweep = min_games_sweep(universe=universe)
    check = decomposition_check(universe, sweep, table, ratios)

    payload = {
        "_meta": {
            "source": "nflverse draft picks + player stats, all drafted "
                      "offensive rookies (no snap filter, no games filter)",
            "seasons": list(_rookie_backtest().COHORT_SEASONS),
            "alpha": ALPHA,
            "min_cell_n": MIN_CELL_N,
            "nfl_regular_season_games": NFL_REGULAR_SEASON_GAMES,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "note": "Phase 13.5a playing-time model. Generated by "
                    "src/playing_time.py -- do not hand-edit. Feeds "
                    "expected_games ONLY; it must never touch "
                    "adjusted_fantasy_points_per_game, which is a rate. "
                    "Not to be wired into a board until holdout.py has "
                    "gated it.",
        },
        "positions": model,
        "availability": {
            f"{row['position']}|{row['round_bucket']}": {
                "p_snap": row["p_snap"],
                "p_min_games": row["p_min_games"],
                "mean_games": row["mean_games"],
                "observed_share": row["observed_share"],
                "available_share": row["available_share"],
                "cell_n": row["cell_n"],
                "source": row["source"],
            }
            for row in summary.iter_rows(named=True)
        },
        "realisation": {
            f"{row['position']}|{row['round_bucket']}": {
                "realisation": row["realisation"],
                "n_drafted": row["n_drafted"],
                "thin": bool(row["thin"]),
            }
            for row in ratio_summary.iter_rows(named=True)
        },
        "decomposition_recovery_median": (
            float(check.select(pl.col("ratio").median()).item())
            if check.height else None
        ),
        "known_limitation": (
            "The universe is drafted players with a non-null nflverse "
            "gsis_id. A drafted player who never signed may have no id at "
            "all, so the 16.6% 'never played' share measured on the first "
            "run is a LOWER BOUND and every realisation ratio is "
            "correspondingly optimistic. 719 offensive rookies over nine "
            "classes is ~80/class against ~90 expected, so the gap is "
            "roughly 10 players a year -- all of them true zeros."
        ),
    }

    if args.no_write:
        print("\n--no-write: nothing written.")
        return

    universe.write_csv(UNIVERSE_PATH)
    sweep.write_csv(SWEEP_PATH)
    MODEL_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {UNIVERSE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {SWEEP_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print("\nNOTHING IS WIRED INTO A BOARD YET. Run the holdout gate next:")
    print("    python -m src.holdout --gate")


if __name__ == "__main__":
    main()
