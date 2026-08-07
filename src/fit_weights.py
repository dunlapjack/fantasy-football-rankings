"""
Fits the per-position situational weights used by ranking.py, and writes
them to data/situational_weights.json.

WHY THIS FILE EXISTS
--------------------
Through Phase 6 the weights in ranking.py were hand-copied out of a
notebook. That notebook isn't in the repo, so the fit wasn't
reproducible -- and the transcription silently dropped the regression's
constant term. Because `workload_share` and `experience` were
always-positive features carrying negative slopes, losing a positive
intercept forced EVERY player's adjustment negative. RB lost 3.59
points, WR 2.37, TE 1.58, uniformly, for no statistical reason.

Fitting and writing the coefficients in one step means the intercept
can't get separated from the slopes again. If a spec changes, re-run
this file; nothing gets retyped.

PHASE 10 CHANGES (Aug 4)
------------------------
Three structural fixes, all of them variations on "a coefficient is
only meaningful alongside the constants it was fitted with."

1. TWO-STAGE FIT. The old code fit the full spec, kept the significant
   subset of slopes, and shipped the FULL model's intercept alongside
   them. Those come from different models. At the feature means the
   dropped terms contribute `coef x mean`, which is not zero, so the
   applied adjustment no longer reconciles against the actual mean
   delta. It never bit before only because Phase 8 happened to drop
   nothing. TE's `trend_missing` (p=0.85) is the first real drop, so
   the survivors are now REFIT and that refit's intercept is what
   ships. Same family of bug as Phase 6, one level up.

2. CENTERING TRAVELS WITH THE COEFFICIENT. `age` enters centered at the
   position mean, which makes the intercept interpretable as "an
   average-aged player at this position." A centered coefficient
   applied to uncentered data is wrong by `coef x center` -- roughly
   -9.5 PPG for RB. The center is therefore written into the JSON under
   `centers` and ranking.py subtracts it before multiplying. Never ship
   one without the other. (See also: intercepts.)

3. IMPUTE AT FIT TIME, DON'T DROP. `usage_trend_share` is null for
   players with under two usable seasons. The old `drop_nulls` would
   have silently deleted 209 of 851 RB/WR/TE rows -- and those rows are
   emphatically not random: mean delta +0.97 against -0.79 for the rest.
   Dropping them would have pushed every intercept down for a
   non-statistical reason. They are mean-imputed and flagged with the
   `trend_missing` indicator instead, which is the standard treatment
   and keeps the sample honest.

USAGE
-----
    python -m src.fit_weights

Reads data/backtest_features.csv (build it with `python -m src.backtest`
first if it's missing or stale). Prints a full regression summary per
position, then writes data/situational_weights.json.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKTEST_PATH = PROJECT_ROOT / "data" / "backtest_features.csv"
WEIGHTS_PATH = PROJECT_ROOT / "data" / "situational_weights.json"

# Minimum games in the TARGET season for a player-season to count as a
# real observation. Someone who played 3 games and got hurt tells us
# about injury luck, not about whether a coaching change hurt him.
MIN_GAMES = 8

# Significance bar a feature must clear to keep a nonzero weight.
# Same discipline that dropped position_competition_ppg (p=0.77) and
# contract_year in Phase 6, and cut Phase 9 outright.
ALPHA = 0.10

# Loosest defensible bar for leave-one-season-out MAGNITUDE stability: a
# fold coefficient may sit between 1/3 and 3x the shipped value. Same
# constant and same reasoning as fit_rookie_weights, where it was
# introduced first. See the magnitude check in fit_position for the WR
# `pos_rank` case that forced it into this file too.
STABILITY_RATIO = 3.0

# Per-position candidate feature sets. Adding a feature here is how you
# test it -- fit, read the p-value, keep or cut.
#
# Phase 10 replaced `experience` with `age` at all three positions. Age
# won on both adjusted R^2 and AIC everywhere, and with both terms in
# the model experience went insignificant at every position (p = 0.38 to
# 0.85) while age held. Quadratic age was tested and is dead (p = 0.71
# to 0.94); binned age was competitive only at TE and non-monotonic
# there, which is noise rather than a curve.
#
# `usage_trend_share` ships at all three positions. It was tested
# against two other bases -- raw per-game volume and
# share-slope-over-mean-share -- and beat both.
#
# WR is a revision, and the reason is worth recording. On the original
# 3-season training set it tested p=0.23 and was cut. Widening the
# window to 2021-2025 (947 -> 1575 rows) moved it to p=0.034 with
# essentially the same coefficient, which is what a real-but-small
# effect looks like when it finally gets enough data, not a new finding.
# It is NOT a case of testing until something passes: alpha was fixed at
# 0.10 beforehand and the test is identical. But WR sits far closer to
# the line than RB (0.0016) or TE (0.0097), so it is the first
# coefficient that should fall if Phase 13 CP2's holdout disagrees.
#
# `trend_missing` is a required companion wherever usage_trend_share is
# mean-imputed -- imputing without a missing indicator invents data. It
# is listed at all three positions and lets alpha decide: it survives at
# RB (p=0.037) and is cut at WR (p=0.18) and TE (p=0.73).
#
# `position_competition_ppg` is a REINSTATEMENT. Phase 6 cut it at
# p=0.77 on three seasons; on five it returns at p=0.044 (RB) and
# p=0.035 (TE), negative both times -- better teammates at your position
# means fewer touches for you, which is what anyone would have guessed.
# WR stays out (p=0.17).
#
# ON MULTIPLE TESTING, since this feature and the QB entry below both
# came out of a sweep: re-running every previously-cut feature was
# roughly thirty tests, and at alpha=0.10 about three should clear the
# bar on luck alone. That makes "p < 0.10" weaker evidence here than
# when one pre-specified feature is tested. Both reinstatements below
# had a stated football reason to expect them BEFORE the sweep ran --
# competition for touches, and age already mattering everywhere else.
# `returning_oline_starters` also cleared the bar, at TE only (p=0.069),
# with no reason it should help tight ends and not backs or receivers.
# That is what a false positive looks like, and it was left out. Every
# reinstatement is on notice for Phase 13 CP2's holdout.
#
# `continuity_score` was RETIRED at RB in favour of `qb_changed` alone.
# On 9 seasons -- and with 2021's coach flags finally derived rather
# than defaulted to false -- the summed feature fell to p=0.059, and
# splitting it showed why: qb_changed carries it (-0.488, p=0.087) while
# coach_changed contributes noise (-0.256, p=0.36 at RB; nothing at TE;
# and at WR it comes out POSITIVE at p=0.082, i.e. a coaching change
# helping receivers, which is not a finding). This is Phase 13 CP3's
# question, answered early because the wider window forced it. Renamed
# rather than silently redefined, per the plan's own instruction.
# CUTS FROM PHASE 13 CP2'S HOLDOUT (Aug 6). Three features and one whole
# position were removed from these candidate lists after failing
# out-of-sample validation across three folds (2025, 2024, 2023). They
# are removed from the CANDIDATE LIST, not merely unshipped, because
# leaving them here means alpha can readmit them on the next refit --
# and alpha is precisely the bar that passed them in the first place.
#
#   WR `usage_trend_share`  Selected in 1 of 3 training folds, ablation
#   WR `trend_missing`      value -0.016 and -0.008 when selected. This
#                           file pre-registered usage_trend_share as
#                           "the first coefficient that should fall if
#                           Phase 13 CP2's holdout disagrees." It
#                           disagreed. The call was made before the
#                           evidence arrived, which is the only reason
#                           it is worth anything.
#
#   RB `qb_changed`         Same profile: 1 of 3 folds, -0.031 when
#                           selected. Consistency requires cutting it on
#                           the same rule that cut the WR pair.
#
# Everything retained beats a constant in all three folds. `age` is the
# most valuable feature in the model by ablation (+0.189 at RB, +0.161 at
# WR), and RB/WR/TE as a whole clear LEVEL by +0.313 / +0.363 / +0.119.
# PHASE 13 CP1 -- COMPETITION DEFINITION BAKE-OFF (Aug 6).
#
# `position_competition_ppg` averages EVERY teammate at the position,
# so a receiver on a team carrying six WRs -- three of them camp filler
# projecting near zero -- reads as facing less competition than one on a
# team carrying three real players. That is roster length, not football.
#
# The plan sized this on Aug 4 and the number is the reason it is being
# tested rather than argued about: the DEFINITION choice moves Gibbs by
# 1.3 PPG, while the roster move it was asked about (Montgomery out,
# Pacheco in) moves him 0.16. Eight times the football.
#
# THEY ARE NOT LISTED TOGETHER, AND THAT IS THE WHOLE METHOD.
#
# The obvious move is to put all four definitions in one spec and let
# alpha choose. That is wrong, and wrong in a way that would have looked
# like it worked: they are near-identical by construction, so they mask
# each other. Four collinear terms split one effect four ways, every
# p-value inflates, and the "winner" is whichever way the noise fell.
# A model containing all four is not a bake-off between them, it is a
# coin flip wearing a regression.
#
# Mutually exclusive DEFINITIONS of the same quantity get compared as
# separate models on held-out error -- see src/competition_bakeoff.py,
# which fits each position five times (one per definition, plus none)
# and scores all five on the 2023/2024/2025 folds.
#
# `position_competition_ppg` stays here as the incumbent until that
# bake-off names a replacement. Swap the name below; do not add to it.
COMPETITION_DEFINITION = "position_competition_ppg"

# PHASE 13 CP1 -- THE TUTEN TEST.
#
# The board ranks Bhayshul Tuten ~150th against an ADP of 51, with every
# driver on him POSITIVE. He ranks low because his BASELINE is thin
# production: the model prices demonstrated snaps, the market prices an
# expected 2026 role, and the model has no channel through which "he is
# the starter now" can reach a projection.
#
# Depth chart rank is the only pre-season, statistics-only signal that
# carries role information. FEATURE_SPEC.md ruled it "secondary
# tie-breaker only, not a primary weighted input" -- which was a
# DECISION and was never tested. It is tested here.
#
# It failed for ROOKIES on the holdout, and that says nothing about
# veterans: a rookie's chart position is a guess by a staff who have not
# seen him play, a veteran's is a summary of what they concluded last
# season. Different measurements sharing a name.
#
# `depth_chart_missing` is the required companion -- pos_rank is
# mean-imputed and imputing without a missing indicator invents data.
# Expect it to matter more here than it did for rookies: a veteran off
# his team's chart is often unsigned, which is a real signal.
# RESULT (Aug 6): IT WORKS AT RUNNING BACK AND NOWHERE ELSE.
#
#   RB   pos_rank ablation +0.080 mean over three folds, coefficient
#        -1.06 PPG per depth-chart place. Kept. This is the channel that
#        was missing -- being RB1 rather than RB3 is worth ~2 PPG, which
#        is exactly the "he is the starter now" information the market
#        prices and the model previously could not see.
#   WR   -0.224. Catastrophic, and instructive: every leave-one-season-
#        out fold sits near -0.30 except 2025, which lands at -0.886.
#        Every fold shares a sign, so the sign-flip check waved it
#        through, and the fitted coefficient then blew up the 2025
#        holdout by a quarter of a point of RMSE. Cut.
#   TE   -0.025. Cut.
#
# Why RB and not the others is a football answer rather than a
# statistical one: a running back depth chart is close to a declaration
# of touch allocation, while receivers rotate by package and personnel
# and a WR3 label says little about targets.
TUTEN_TEST = ["pos_rank", "depth_chart_missing"]

FEATURE_SPECS = {
    "RB": ["workload_share", "age", "usage_trend_share",
           "trend_missing", COMPETITION_DEFINITION, *TUTEN_TEST],
    "WR": ["team_changed", "workload_share", "recent_major_injury", "age"],
    "TE": ["workload_share", "age", "usage_trend_share", "trend_missing",
           COMPETITION_DEFINITION],
    # QB IS ABSENT, AND THAT IS A REVERSAL OF PHASE 10'S HEADLINE.
    #
    # "QB carries a weight for the first time in the project's history"
    # was the claim: age at -0.19 PPG per year, p=0.020 on 157
    # quarterback-seasons, sign-stable across all nine leave-one-season-
    # out folds. Every in-sample check this project had, passed.
    #
    # The holdout says 1 of 3 folds, mean -0.023 RMSE against a
    # constant. It does not predict. LOSO stability was never evidence
    # of prediction -- it asks whether a coefficient moves when a season
    # is withheld, not whether it forecasts a season it has not seen --
    # and this is the clearest demonstration in the project of the gap
    # between those two questions.
    #
    # With QB removed from this dict, apply_situational_weights() passes
    # quarterbacks through with a zero adjustment, exactly as it did from
    # Phase 5 through Phase 9. SUPPRESS_LEVEL_SHIFT below still names QB
    # and is now moot; it is left in place because if QB is ever refitted
    # the selection artifact it describes will still be there.
}

# Features entered as deviations from the position mean. The mean is
# written to the JSON so ranking.py can reproduce it.
CENTERED_FEATURES = {"age"}

# Features whose nulls are mean-imputed at FIT time rather than causing
# the row to be dropped. Only for features with a paired missing
# indicator in the spec -- otherwise imputation quietly invents data.
IMPUTED_FEATURES = {"usage_trend_share", "pos_rank"}

# Missing-indicator companions. If the key ships, the value ships with
# it -- ALWAYS, regardless of its own p-value.
#
# WHY THESE ARE NOT CANDIDATES (Aug 6). `trend_missing` failed the
# holdout gate at RB: it cleared alpha in one training fold out of three
# and contributed -0.0003 RMSE when it did. The obvious response is to
# cut it, and that would be wrong, because it is not in the model to
# predict anything.
#
# `usage_trend_share` is mean-imputed for 186 of 711 RB rows. The
# indicator exists so the model can tell "this player's trend is
# average" from "we do not know this player's trend." Without it the
# imputed coefficient attenuates toward zero and the intercept quietly
# absorbs the difference between the observed and missing groups.
# Judging it by predictive ablation is the wrong test in the same way
# ablating the intercept would be: it is machinery, not evidence.
#
# Treating it as a candidate also caused a subtler problem, which is
# what made this worth fixing rather than exempting. Subject to alpha,
# the companion flickered in and out across folds -- present in the 2025
# fit, absent in 2024 and 2023 -- so `usage_trend_share` did not mean
# the same thing in each fold, and the fold-to-fold comparison the whole
# gate rests on was not comparing like with like.
#
# IT ALSO SURFACED A LIVE BUG: TE ships `usage_trend_share` (152 rows
# imputed) while alpha cut `trend_missing` at p=0.68. TE has been
# imputing without an indicator since Phase 10, against this file's own
# stated rule, and nothing noticed because the rule lived in a comment
# rather than in the code.
IMPUTATION_COMPANIONS = {"usage_trend_share": "trend_missing"}

# Positions whose fitted intercept is a SELECTION artifact rather than a
# fact about the position, and so ships as a relative adjustment only.
#
# QB is the case. MIN_GAMES=8 filters far harder here than anywhere
# else, because quarterback is one-per-team: a mediocre receiver still
# plays eight games, a mediocre quarterback gets benched. Mean delta by
# games threshold makes it plain --
#
#   min games      QB      RB      WR      TE
#          0+   -0.93   -0.69   -0.47   -0.19
#          8+   +0.70   -0.08   -0.07   +0.12
#         12+   +1.16   +0.23   +0.18   +0.44
#
# QB swings from the worst position to the best as the filter tightens,
# roughly triple RB's movement. So +0.70 does not mean "quarterbacks beat
# their baseline"; it means "quarterbacks who keep their job beat their
# baseline," and on draft day you do not know which those are. Applied
# uniformly it lifts every QB against every other position and changes
# cross-position VOR -- it put Josh Allen in the top 10.
#
# The age SLOPE is unaffected by this: it is estimated within the
# sample and is stable across all nine leave-one-season-out folds
# (-0.15 to -0.23). Only the level is contaminated. So the level is
# removed and the ordering kept.
#
# The honest fix is a proper games-available model, which is Phase 11.
# Revisit this set then; if availability is modelled directly, the
# suppression should come back out rather than double-counting.
SUPPRESS_LEVEL_SHIFT = {"QB"}

BOOL_COLUMNS = [
    "qb_changed", "coach_changed", "team_changed",
    "recent_major_injury", "trend_missing", "trend_low_confidence",
    "depth_chart_missing",
]


def load_backtest():
    """
    Loads the backtest table and coerces the boolean-ish columns into
    0/1 integers so they can go straight into a design matrix.

    `continuity_score` is rebuilt here rather than read from the file --
    backtest_features.csv stores qb_changed and coach_changed separately
    and never materializes their sum, unlike player_features.csv which
    does. Same definition either way: 0, 1, or 2 changes.
    """
    df = pl.read_csv(BACKTEST_PATH)

    present = [c for c in BOOL_COLUMNS if c in df.columns]
    df = df.with_columns([
        pl.col(c).cast(pl.String).str.to_lowercase().eq("true").cast(pl.Int8).alias(c)
        for c in present
    ])

    df = df.with_columns(
        (pl.col("qb_changed") + pl.col("coach_changed")).alias("continuity_score")
    )

    return df.filter(pl.col("actual_games_played") >= MIN_GAMES)


def _design_matrix(subset, features, centers):
    """Builds X with centered features already shifted."""
    columns = []
    for f in features:
        values = subset.select(pl.col(f).cast(pl.Float64)).to_numpy().ravel()
        if f in centers:
            values = values - centers[f]
        columns.append(values)
    return np.column_stack(columns)


def fit_position(df, position, features, alpha=ALPHA):
    """
    Fits delta ~ features for one position, WITH an intercept, in two
    stages: fit everything, then refit on whatever cleared `alpha`.

    The target is `delta` = actual PPG minus the player's own trailing
    3-year baseline. So the model answers "given this situation, how far
    from his own history should we expect him to land?" -- and the
    intercept is the honest answer for a player at the position's mean
    age whose other features are zero. Dropping it isn't a neutral
    simplification; it's asserting such a player would exactly repeat
    his baseline, which the data does not say.

    Returns (result_dict, full_model, refit_model). result_dict is None
    if the position has too little usable data to fit.
    """
    subset = df.filter(pl.col("position") == position)

    # Nulls in a feature WITHOUT a missing indicator are still fatal to
    # the row -- imputing those would be inventing data. Nulls in an
    # imputed feature are filled with that feature's mean over the rows
    # that survive this drop.
    required = [f for f in features if f not in IMPUTED_FEATURES]
    subset = subset.drop_nulls(subset=required + ["delta"])

    if subset.height <= len(features) + 1:
        print(f"  {position}: only {subset.height} usable rows -- skipping fit")
        return None, None, None

    imputed_here = [f for f in features if f in IMPUTED_FEATURES]
    imputation_counts = {
        f: int(subset.select(pl.col(f).is_null().sum()).item()) for f in imputed_here
    }
    for f in imputed_here:
        mean_value = subset.select(pl.col(f).cast(pl.Float64).mean()).item()
        subset = subset.with_columns(pl.col(f).cast(pl.Float64).fill_null(mean_value))

    # Centers are computed AFTER imputation so they describe the rows
    # actually fitted.
    centers = {
        f: float(subset.select(pl.col(f).cast(pl.Float64).mean()).item())
        for f in features if f in CENTERED_FEATURES
    }

    y = subset.select("delta").to_numpy().ravel().astype(float)

    # --- stage 1: full spec, purely to read p-values ------------------
    X_full = _design_matrix(subset, features, centers)
    full_model = sm.OLS(y, sm.add_constant(X_full)).fit()

    p_values = {f: float(p) for f, p in zip(features, full_model.pvalues[1:])}
    full_coefficients = {f: float(c) for f, c in zip(features, full_model.params[1:])}
    survivors = [f for f in features if p_values[f] < alpha]

    # Force in the missing-indicator companion of any imputed feature
    # that survived. See IMPUTATION_COMPANIONS: these are machinery, not
    # candidates, and letting alpha decide them meant the imputed
    # feature's coefficient meant different things in different folds.
    forced_companions = []
    for imputed_feature, companion in IMPUTATION_COMPANIONS.items():
        if (imputed_feature in survivors and companion in features
                and companion not in survivors):
            forced_companions.append(companion)
    if forced_companions:
        # Rebuilt in spec order rather than appended, so the design
        # matrix column order always matches `features` and a
        # coefficient can never be read against the wrong name.
        survivors = [f for f in features
                     if f in survivors or f in forced_companions]

    # --- stage 2: refit on survivors ----------------------------------
    # This is the model that ships. Its intercept belongs to its own
    # slopes; borrowing stage 1's would bias every player at this
    # position by a fixed amount.
    if survivors:
        X_keep = _design_matrix(subset, survivors, centers)
        refit_model = sm.OLS(y, sm.add_constant(X_keep)).fit()
        intercept = float(refit_model.params[0])
        weights = {f: float(c) for f, c in zip(survivors, refit_model.params[1:])}
    else:
        refit_model = None
        intercept = float(np.mean(y))
        weights = {}

    # Means of the RAW (uncentered, post-imputation) columns over the
    # exact rows this model was fit on. ranking.py uses these to impute
    # missing values at apply time instead of filling with 0. That
    # distinction matters a lot with an intercept in play: filling
    # workload_share with 0 hands the player the full positive intercept
    # and none of the offsetting usage penalty, which would
    # systematically inflate exactly the players whose workload_share
    # gets nulled out (everyone who changed teams -- see situational.py).
    # Imputing the mean instead makes the fallback "an average player at
    # this position," which is what "no opinion" should mean.
    feature_means = {
        f: float(subset.select(pl.col(f).cast(pl.Float64).mean()).item())
        for f in features
    }

    # Leave-one-season-out refit. A coefficient that flips sign when one
    # season is withheld is not a finding, it is that season. This is a
    # stability probe, not the holdout validation -- Phase 13 CP2 is
    # still the first real out-of-sample test.
    stability = {}
    if survivors:
        for held_out in sorted(subset.select("season").unique().to_series().to_list()):
            fold = subset.filter(pl.col("season") != held_out)
            if fold.height <= len(survivors) + 1:
                continue
            X_fold = _design_matrix(fold, survivors, centers)
            y_fold = fold.select("delta").to_numpy().ravel().astype(float)
            fold_model = sm.OLS(y_fold, sm.add_constant(X_fold)).fit()
            stability[str(held_out)] = {
                f: float(c) for f, c in zip(survivors, fold_model.params[1:])
            }

    sign_flips = [
        f for f in survivors
        if any(np.sign(fold[f]) != np.sign(weights[f]) for fold in stability.values())
    ]

    # MAGNITUDE INSTABILITY -- PORTED FROM fit_rookie_weights (Aug 6),
    # after the veteran fit shipped something the rookie fit would have
    # caught.
    #
    # WR `pos_rank` came back sign-stable across all nine folds: -0.286,
    # -0.319, -0.299, -0.306, -0.280, -0.301, -0.350, -0.305, and then
    # -0.886 when 2025 was withheld. Every fold negative, so the check
    # above waved it through. That tripled coefficient then cost the 2025
    # holdout a quarter point of RMSE and took the whole WR model below a
    # constant.
    #
    # The rookie fitter has had this bar since RB `age` failed it there,
    # and the reason it was not here is simply that the veteran fit had
    # not yet met a feature that needed it. It has now. A coefficient one
    # season can move by 3x is not a measurement, whatever its sign.
    #
    # Bar is deliberately loose -- every fold shares eight ninths of its
    # rows with the shipped model, so anything failing this fails badly.
    # Companions are exempt, for the same reason the gate exempts them:
    # they are machinery, not predictions. TE `trend_missing` duly fired
    # this warning at 0.24-2.53x and the gate then passed it, which made
    # the warning's own text ("treat this as a prediction that the gate
    # will fail") false. A warning that cries wolf is how you train
    # someone to skim past the one that matters.
    #
    # Their instability is also expected rather than alarming: a
    # companion carries a small coefficient whose job is to absorb the
    # difference between the observed and imputed groups, and small
    # coefficients move around. It is not steering anyone's draft.
    companions = set(IMPUTATION_COMPANIONS.values())
    magnitude_unstable = []
    for f in survivors:
        if f in companions:
            continue
        values = [fold[f] for fold in stability.values() if f in fold]
        if not values or not weights[f]:
            continue
        ratios = [abs(v) / abs(weights[f]) for v in values]
        if max(ratios) > STABILITY_RATIO or min(ratios) < 1 / STABILITY_RATIO:
            magnitude_unstable.append(f)

    # A suppressed level shift shifts the whole position by a constant,
    # so it changes nothing about the order WITHIN the position and
    # everything about where the position sits against the others.
    # Recorded explicitly rather than folded into the intercept, so the
    # fitted value stays visible and verify_adjustments can check the
    # right identity.
    level_shift_removed = float(np.mean(y)) if position in SUPPRESS_LEVEL_SHIFT else 0.0
    shipped_intercept = intercept - level_shift_removed

    result = {
        "intercept": shipped_intercept,
        "intercept_fitted": intercept,
        "level_shift_removed": level_shift_removed,
        "weights": weights,
        "feature_means": feature_means,
        "centers": centers,
        "n": int(subset.height),
        "r_squared": float(refit_model.rsquared) if refit_model is not None else 0.0,
        "r_squared_full_spec": float(full_model.rsquared),
        "imputed_rows": imputation_counts,
        "diagnostics": {
            f: {
                "coef_full_spec": full_coefficients[f],
                "p_value": p_values[f],
                "kept": f in survivors,
                "coef_shipped": weights.get(f),
            }
            for f in features
        },
        "stability_leave_one_season_out": stability,
        "sign_flips": sign_flips,
        "magnitude_unstable": magnitude_unstable,
        # Pairwise correlations among SHIPPED features, strongest first.
        # Recorded so CP1's multicollinearity question has an answer on
        # file rather than an assurance.
        "collinearity": (
            sorted(
                (
                    ((a, b), float(np.corrcoef(
                        subset.select(pl.col(a).cast(pl.Float64)).to_numpy().ravel(),
                        subset.select(pl.col(b).cast(pl.Float64)).to_numpy().ravel(),
                    )[0, 1]))
                    for i, a in enumerate(survivors) for b in survivors[i + 1:]
                ),
                key=lambda pair: -abs(pair[1]),
            )[:5]
            if len(survivors) > 1 else []
        ),
        # The reconciliation that would have caught Phase 6. OLS with an
        # intercept forces mean(fitted) == mean(y) exactly, so any
        # meaningful gap here means the shipped numbers do not come from
        # the model that was fitted.
        "mean_fitted_adjustment": (
            float(np.mean(refit_model.fittedvalues)) if refit_model is not None
            else float(np.mean(y))
        ) - level_shift_removed,
        "mean_actual_delta": float(np.mean(y)),
    }
    return result, full_model, refit_model


def print_summary(position, result, full_model, features):
    """Human-readable regression report, so the numbers can be checked
    before they're trusted."""
    print(f"\n{'=' * 74}")
    print(f"{position}   n={result['n']}   R^2={result['r_squared']:.3f} "
          f"(full spec {result['r_squared_full_spec']:.3f})")
    print(f"{'=' * 74}")
    if result["imputed_rows"]:
        detail = ", ".join(f"{f}: {n}" for f, n in result["imputed_rows"].items())
        print(f"  mean-imputed at fit time -- {detail}  (kept, not dropped)")
    if result["centers"]:
        detail = ", ".join(f"{f} at {c:.2f}" for f, c in result["centers"].items())
        print(f"  centered -- {detail}")

    print(f"\n  {'feature':<24} {'coef':>10} {'std err':>10} {'p':>8}   kept   shipped")
    print(f"  {'-' * 68}")
    print(f"  {'(intercept)':<24} {result['intercept']:>10.4f} "
          f"{'':>10} {'':>8}   always {result['intercept']:>9.4f}")
    for i, f in enumerate(features):
        d = result["diagnostics"][f]
        mark = "yes" if d["kept"] else "NO"
        shipped = f"{d['coef_shipped']:.4f}" if d["coef_shipped"] is not None else "--"
        print(f"  {f:<24} {d['coef_full_spec']:>10.4f} {full_model.bse[i + 1]:>10.4f} "
              f"{d['p_value']:>8.4f}   {mark:<6} {shipped:>9}")

    shift = result["level_shift_removed"]
    expected = result["mean_actual_delta"] - shift
    gap = result["mean_fitted_adjustment"] - expected
    if shift:
        print(f"\n  LEVEL SHIFT SUPPRESSED: fitted intercept {result['intercept_fitted']:+.4f} "
              f"ships as {result['intercept']:+.4f}")
        print(f"  ({shift:+.4f} removed as a selection artifact of MIN_GAMES={MIN_GAMES} -- "
              f"see SUPPRESS_LEVEL_SHIFT.")
        print(f"   Ordering within {position} is unchanged; {position} no longer floats "
              f"against other positions.)")
    print(f"\n  mean applied adjustment {result['mean_fitted_adjustment']:+.4f}  "
          f"vs expected {expected:+.4f}   gap {gap:+.6f}")
    if abs(gap) > 1e-6:
        print("  *** RECONCILIATION FAILED -- shipped weights do not match the fit ***")

    if result.get("collinearity"):
        # CP1 was told to "check for multicollinearity." Asserting that
        # two correlated features are separable is not checking; this
        # prints the number so the claim can be read rather than trusted.
        #
        # A pair correlating strongly is NOT by itself a problem. It is a
        # problem when it makes coefficients unstable or inflates the
        # standard errors past the point of telling the two apart -- and
        # the LOSO table below plus the standard errors above are where
        # that shows up. RB `pos_rank` and `workload_share` are the live
        # case: correlated by construction ("does he have the job" vs
        # "how much of it has he already converted"), and separable.
        print(f"\n  strongest feature correlations:")
        for (a, b), r in result["collinearity"]:
            note = "   <-- worth a look" if abs(r) > 0.7 else ""
            print(f"    {a:<26} {b:<26} r={r:+.3f}{note}")

    if result["stability_leave_one_season_out"]:
        folds = result["stability_leave_one_season_out"]
        print(f"\n  leave-one-season-out (coefficient when that season is withheld):")
        header = "".join(f"{s:>11}" for s in folds)
        print(f"    {'feature':<24}{'shipped':>11}{header}")
        for f in result["weights"]:
            row = "".join(f"{folds[s][f]:>11.3f}" for s in folds)
            if f in result["sign_flips"]:
                flag = "   <-- SIGN FLIP"
            elif f in result.get("magnitude_unstable", []):
                folds_f = [v[f] for v in folds.values()]
                ratios = [abs(x) / abs(result["weights"][f]) for x in folds_f]
                flag = (f"   <-- MAGNITUDE {min(ratios):.2f}-{max(ratios):.2f}x "
                        f"(bar is {1 / STABILITY_RATIO:.2f}-{STABILITY_RATIO:.2f})")
            else:
                flag = ""
            print(f"    {f:<24}{result['weights'][f]:>11.3f}{row}{flag}")


def fit_all(alpha=ALPHA):
    """Fits every position in FEATURE_SPECS and returns the dict that
    gets serialized to JSON."""
    df = load_backtest()
    print(f"Backtest rows with {MIN_GAMES}+ games played: {df.height}")

    output = {
        "_meta": {
            "source": str(BACKTEST_PATH.name),
            "min_games": MIN_GAMES,
            "alpha": alpha,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "note": (
                "Generated by src/fit_weights.py. Do not hand-edit -- "
                "re-run the script. Intercepts AND centers are part of "
                "the model and must be applied alongside the weights."
            ),
        },
        "positions": {},
    }

    for position, features in FEATURE_SPECS.items():
        result, full_model, _ = fit_position(df, position, features, alpha)
        if result is None:
            continue
        print_summary(position, result, full_model, features)
        output["positions"][position] = result

    return output


if __name__ == "__main__":
    weights = fit_all()

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(WEIGHTS_PATH, "w") as f:
        json.dump(weights, f, indent=2)

    print(f"\nWrote weights to {WEIGHTS_PATH}")
    for position, result in weights["positions"].items():
        kept = ", ".join(result["weights"]) or "(none significant)"
        print(f"  {position}: intercept {result['intercept']:+.4f} | {kept}")
        if result["sign_flips"]:
            print(f"    WARNING sign-unstable across seasons: {result['sign_flips']}")
        if result.get("magnitude_unstable"):
            print(f"    WARNING magnitude-unstable across seasons "
                  f"(one season moves these by >{STABILITY_RATIO}x): "
                  f"{result['magnitude_unstable']}")
            print(f"    These still SHIP -- the veteran fit reports rather than "
                  f"drops, unlike the rookie fit. The holdout gate is what stops "
                  f"them. Treat a warning here as a prediction that the gate "
                  f"will fail.")

    print("\nNext: python -m src.pipeline, then python -m src.verify_adjustments")
