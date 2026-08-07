"""
Phase 12 CP2. Fits rookie-specific situational weights and writes them
to data/rookie_weights.json.

Reads data/rookie_backtest_features.csv (build it with
`python -m src.rookie_backtest` first). Deliberately a separate file
from fit_weights.py, and a separate JSON, for one reason: these
coefficients are estimated on a different population against a different
baseline, and the single worst outcome of Phase 12 would be a rookie
coefficient and a veteran coefficient ending up in the same dict where
someone later applies one to the other's population.

WHAT IS BEING PREDICTED, PRECISELY
----------------------------------
    delta = actual rookie PPG - leave-one-class-out cohort baseline

Not "how good is this rookie" but "given his landing spot, how far from
his draft cohort's average should we expect him to land." Draft capital
is therefore ALREADY IN THE BASELINE, not in the features -- the cohort
is a (position, round) cell. `pick` appears as a candidate anyway to
test whether within-round capital adds anything the round bucket
missed, which is a real question (pick 33 and pick 63 are both round 2
and are not the same asset).

THE HONEST PROBLEM WITH THIS WHOLE PHASE
----------------------------------------
n = 5 classes. Not 5 rows -- roughly 250-350 rows after MIN_GAMES -- but
the CLUSTERING is what matters and it is by class. A league-wide shift
in how rookies are used in one season moves every row in that class
together, so the effective sample for anything season-varying is much
closer to 5 than to 300, and ordinary standard errors will be too small.
The leave-one-class-out stability probe below is not a nicety here; with
five folds it is most of the evidence.

The plan's stated fallback is live: if coefficients are unstable, ship
the shrinkage haircut and say so, rather than a fragile model. That
decision is made by a human reading this output, but the run makes the
call easy to see -- see the UNSTABLE / THIN banners and the
`fit_is_trustworthy` flag in the JSON.

USAGE
-----
    python -m src.fit_rookie_weights
    python -m src.fit_rookie_weights --alpha 0.05   # stricter bar
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm

from src.fit_weights import _design_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKTEST_PATH = PROJECT_ROOT / "data" / "rookie_backtest_features.csv"
WEIGHTS_PATH = PROJECT_ROOT / "data" / "rookie_weights.json"

# Same bar as the veteran fit. Fixed BEFORE looking at any p-value, and
# not to be moved after -- the veteran file's note on multiple testing
# applies with more force here, because the candidate list is short and
# the temptation to keep sweeping is correspondingly larger.
ALPHA = 0.10

# Below this, a position does not get a fitted model at all and falls
# back to the flat cohort baseline. Chosen as roughly 10 rows per
# candidate feature -- under that, an OLS fit on five clustered classes
# is describing noise with confidence. QB is the position expected to
# trip this: about 15-20 quarterbacks are drafted across five classes
# and most never clear MIN_GAMES.
MIN_ROWS_TO_FIT = 40

# CANDIDATE FEATURES, and what each is actually asking.
#
# `pass_att_pg` / `rush_att_pg`   Last year's team volume. The
#     "opportunity" half of the landing spot. Pass volume for pass
#     catchers, rush volume for backs; the other is left out per
#     position rather than fitted and discarded, because with a sample
#     this thin every extra term costs real power.
#
# `position_competition_ppg`      Prior-season PPG of the players he has
#     to beat out, on the team he actually landed on. This is the
#     feature Phase 12 exists for: it is the ONLY candidate that
#     distinguishes two same-round rookies on different teams by
#     something other than team pace. If nothing else survives, this
#     one surviving is still a successful phase.
#
#     Expected NEGATIVE, same as the veteran fit -- better incumbents
#     means fewer touches. If it comes out positive and significant,
#     suspect that good teams draft skill players early and the feature
#     is proxying for offensive quality, not competition.
#
# `pos_rank` / `depth_chart_missing`   Where the team itself listed him
#     before the season. Paired, because pos_rank is imputed and
#     imputing without a missing indicator invents data -- the same rule
#     usage_trend_share/trend_missing follows in the veteran fit. Read
#     the caveat in rookie_backtest's docstring before trusting a large
#     coefficient here: the historical snapshot may sit later in the
#     preseason than the August snapshot the live model reads, which
#     would make this an optimistic ceiling rather than an estimate.
#
# `returning_oline_starters`      Cleared alpha at TE only in the
#     veteran sweep and was left out there as a likely false positive.
#     It has a better claim on rookies -- a rookie back behind a rebuilt
#     line is a real story -- so it is tested, at RB only, where that
#     story is strongest. Testing it at all four positions would be the
#     same fishing expedition fit_weights declined to run.
#
# `age`                           Draft classes span 21 to 25. Weak
#     prior, tested because it is free and it is the one veteran feature
#     that transfers at all. Expected to fail.
#
# `pick`                          Within-round draft capital. See the
#     header: the baseline is round-level, so this asks whether the
#     round bucket threw away signal. Expected NEGATIVE (a lower pick
#     number is a better player).
# GUTTED BY THE HOLDOUT, ONE DAY AFTER BEING BUILT (Aug 6).
#
# Phase 13 CP2 tested this model on three held-out classes and it mostly
# does not survive. The candidate lists below are what is left.
#
#   RB   REMOVED ENTIRELY. 0 of 3 folds, mean -0.099 RMSE against a
#        constant, and `position_competition_ppg` actively hurt
#        (-0.148 mean ablation). Rookie backs go back to the flat
#        cohort baseline.
#
#   WR   REMOVED ENTIRELY, and this is the strongest negative result in
#        the project. The feature failed to clear alpha in EVERY
#        training fold -- it is only significant when the test season is
#        inside the fit. That is the precise thing a holdout exists to
#        catch, and nothing weaker would have caught it.
#
#   QB   REMOVED. Never fitted anyway: ~4 rookie quarterbacks per class
#        clear MIN_GAMES, so 9 classes gives 35 rows against
#        MIN_ROWS_TO_FIT=40, and every holdout fold fell to ~30. Listing
#        candidates for a position that cannot be fitted only invites
#        someone to lower the threshold to make it fit.
#
#   TE   SURVIVES, ON `age` ALONE. 2 of 3 folds, mean +0.200. But read
#        WHICH feature carried it: `age` at +0.235 ablation value,
#        `pos_rank` at +0.055. That is the REVERSE of the in-sample fit,
#        where pos_rank had the larger coefficient (-1.03 vs -0.76) and
#        age looked secondary. The suspicion recorded when pos_rank
#        doubled its coefficient on the widened window was correct, and
#        pos_rank is cut.
#
#        THE CAVEAT TRAVELS WITH IT: rookie TE holdout folds are n = 10,
#        7 and 9. A +0.327 RMSE gain on seven players is not a finding.
#        This is the least-evidenced thing that ships anywhere in the
#        model and it survives on the consistency of the feature, not
#        the weight of the sample. First thing to cut if next season's
#        data disagrees.
#
# Cutting `pos_rank` here also retires the week-1-versus-August depth
# chart mismatch documented in rookie_backtest -- the model no longer
# depends on a feature measured at a different moment than it is applied.
FEATURE_SPECS = {
    "TE": ["age"],
}

CENTERED_FEATURES = {"age", "pick"}
IMPUTED_FEATURES = {"pos_rank"}

# Features that are only meaningful if their availability is roughly even
# across classes, paired with the indicator that gives them away.
#
# WHY THIS EXISTS (Aug 6). Widening COHORT_SEASONS to 2017 reaches back
# into seasons where load_depth_charts() coverage is thin. If coverage is
# 90% missing in 2017 and 5% missing in 2024, then `depth_chart_missing`
# has stopped meaning "this rookie was buried" and started meaning "this
# is an old season." It would very likely test significant -- early and
# late classes differ for a hundred reasons -- and the coefficient would
# be measuring the difference between eras while wearing a feature's
# name. That is a worse outcome than having no feature, because it looks
# like a finding.
#
# The window was widened precisely BECAUSE pos_rank failed and was no
# longer worth constraining the data for. Letting it back in through the
# side door as a season label would make the widening indefensible.
SEASON_CONFOUNDABLE = {"pos_rank": "depth_chart_missing"}

# Max acceptable spread (highest minus lowest class) in a feature's
# missing rate before it is treated as a season label. 0.35 is loose:
# real year-to-year variation in scraping completeness is a few points,
# not thirty-five.
SEASON_CONFOUND_SPREAD = 0.35

BOOL_COLUMNS = ["depth_chart_missing", "baseline_low_confidence"]

# EMPTY, DELIBERATELY, and this is a substantive claim rather than an
# oversight.
#
# fit_weights suppresses QB's intercept because MIN_GAMES selects hard
# at that position and the fitted level is a fact about quarterbacks who
# kept their job, not about quarterbacks. MIN_GAMES selects at least as
# hard on rookies -- a rookie who loses his job is inactive, not benched.
#
# The difference is WHERE THE BASELINE COMES FROM. The veteran baseline
# is a player's own trailing average, computed over everyone, so the
# 8-game filter shifts the fitted population without shifting the thing
# it is measured against, and the gap shows up in the intercept. The
# rookie cohort baseline is computed AFTER the same filter, over exactly
# the surviving rows (see rookie_backtest.build_rookie_backtest). Both
# sides move together, so the level is already netted out and the
# intercept here should sit near zero by construction.
#
# It will not be exactly zero, because the baseline is leave-one-class-
# out. A large intercept is therefore a SYMPTOM, not a finding: it means
# the filter and the baseline came apart somewhere. The banner below
# checks for it.
SUPPRESS_LEVEL_SHIFT = set()

# How far the fitted LEVEL can sit from zero before the run says
# something.
#
# THIS CHECKED THE RAW INTERCEPT UNTIL IT CRIED WOLF (Aug 6). RB came
# back with an intercept of +12.04 PPG and the banner fired, which looked
# alarming and was meaningless: `rush_att_pg` enters UNCENTERED with a
# mean near 26.5 and a coefficient of -0.444, so the intercept has to
# carry +11.8 just to cancel it. The two numbers are inseparable. An
# intercept is only interpretable as a level when every feature is
# centered, and here only `age` and `pick` are.
#
# The quantity that actually answers "did the filter and the baseline
# come apart" is the fitted value at the feature means, which OLS with
# an intercept forces to equal mean(delta). So that is what gets checked.
# RB's is +0.19, WR's +0.08 -- both fine, and both invisible behind the
# raw intercepts of +12.04 and +2.10.
#
# The lesson generalizes and is the same one this project keeps
# relearning: a coefficient means nothing apart from the constants it was
# fitted with. That applies to reading them, not just to shipping them.
LEVEL_SANITY_PPG = 1.5

# Loosest defensible bar for leave-one-class-out magnitude stability: a
# fold coefficient may sit anywhere between 1/3 and 3x the shipped value.
# Every fold shares 80% of its rows with the shipped model, so a
# coefficient that still moves by more than this is being carried by one
# class.
STABILITY_RATIO = 3.0


def load_rookie_backtest(path=BACKTEST_PATH):
    df = pl.read_csv(path)
    present = [c for c in BOOL_COLUMNS if c in df.columns]
    return df.with_columns([
        pl.col(c).cast(pl.String).str.to_lowercase().eq("true").cast(pl.Int8).alias(c)
        for c in present
    ])


def season_confounded_features(df):
    """
    Features whose MISSINGNESS varies so much by class that they are
    really season labels. See SEASON_CONFOUNDABLE.

    Returns {feature: explanation} for everything that should be removed
    from every spec before fitting. Removing rather than warning is
    deliberate: a warning about a feature that has already been fitted
    arrives after the number exists, and numbers that exist get used.
    """
    confounded = {}
    for feature, indicator in SEASON_CONFOUNDABLE.items():
        if indicator not in df.columns or "season" not in df.columns:
            continue
        by_season = (
            df.group_by("season")
            .agg(pl.col(indicator).cast(pl.Float64).mean().alias("rate"))
            .sort("season")
        )
        rates = by_season["rate"].to_list()
        if not rates:
            continue
        spread = max(rates) - min(rates)
        if spread > SEASON_CONFOUND_SPREAD:
            detail = ", ".join(
                f"{int(s)}:{r:.0%}"
                for s, r in zip(by_season["season"].to_list(), rates)
            )
            confounded[feature] = (
                f"missing-rate spread {spread:.0%} across classes "
                f"(> {SEASON_CONFOUND_SPREAD:.0%}) -- {detail}"
            )
            confounded[indicator] = (
                f"companion of {feature}, removed with it"
            )
    return confounded


def fit_position(df, position, features, alpha=ALPHA):
    """
    Fits delta ~ features for one position, WITH an intercept, in two
    stages -- full spec to read p-values, then refit on the survivors so
    the shipped intercept belongs to the shipped slopes.

    Identical in structure to fit_weights.fit_position() on purpose. The
    two files diverge in their feature sets and their population, not in
    their statistics, and a reader who has understood one should not
    have to re-derive the other.
    """
    subset = df.filter(pl.col("position") == position)

    required = [f for f in features if f not in IMPUTED_FEATURES]
    subset = subset.drop_nulls(subset=required + ["delta"])

    if subset.height < MIN_ROWS_TO_FIT:
        print(f"\n  {position}: {subset.height} rows, below MIN_ROWS_TO_FIT="
              f"{MIN_ROWS_TO_FIT} -- NOT FITTED. Falls back to the flat cohort "
              f"baseline, which is what this position already had.")
        return None, None

    imputation_counts = {}
    for f in [f for f in features if f in IMPUTED_FEATURES]:
        imputation_counts[f] = int(subset.select(pl.col(f).is_null().sum()).item())
        mean_value = subset.select(pl.col(f).cast(pl.Float64).mean()).item()
        subset = subset.with_columns(pl.col(f).cast(pl.Float64).fill_null(mean_value))

    centers = {
        f: float(subset.select(pl.col(f).cast(pl.Float64).mean()).item())
        for f in features if f in CENTERED_FEATURES
    }

    y = subset.select("delta").to_numpy().ravel().astype(float)

    X_full = _design_matrix(subset, features, centers)
    full_model = sm.OLS(y, sm.add_constant(X_full)).fit()

    p_values = {f: float(p) for f, p in zip(features, full_model.pvalues[1:])}
    full_coefficients = {f: float(c) for f, c in zip(features, full_model.params[1:])}
    survivors = [f for f in features if p_values[f] < alpha]

    if survivors:
        X_keep = _design_matrix(subset, survivors, centers)
        refit_model = sm.OLS(y, sm.add_constant(X_keep)).fit()
        intercept = float(refit_model.params[0])
        weights = {f: float(c) for f, c in zip(survivors, refit_model.params[1:])}
    else:
        refit_model = None
        intercept = float(np.mean(y))
        weights = {}

    feature_means = {
        f: float(subset.select(pl.col(f).cast(pl.Float64).mean()).item())
        for f in features
    }

    def leave_one_class_out(spec):
        """LOSO coefficients for a given feature list."""
        folds = {}
        for held_out in sorted(subset.select("season").unique().to_series().to_list()):
            fold = subset.filter(pl.col("season") != held_out)
            if fold.height <= len(spec) + 1:
                continue
            fold_model = sm.OLS(
                fold.select("delta").to_numpy().ravel().astype(float),
                sm.add_constant(_design_matrix(fold, spec, centers)),
            ).fit()
            folds[str(held_out)] = {
                f: float(c) for f, c in zip(spec, fold_model.params[1:])
            }
        return folds

    def unstable(spec, shipped, folds):
        """
        Features whose LOSO coefficients don't hold up.

        TWO WAYS TO FAIL, and the second was added after the first run
        let something through that should not have passed.

        A SIGN FLIP is the obvious one -- if withholding 2023 turns a
        positive coefficient negative, the coefficient is 2023.

        MAGNITUDE is the one the veteran fit never needed. RB `age` came
        back at +0.906 with no sign flip anywhere, which reads as stable
        until you look at the folds: 0.918, 1.126, 1.314, 0.739, 0.286.
        Withholding 2025 cuts it to under a third of the shipped value.
        A coefficient that a single class can move by 4.6x is not a
        measurement of anything, and shipping it because the sign held
        would be exactly the "talked myself into it" failure the phase
        plan warns about.

        This bar is deliberately loose (a factor of STABILITY_RATIO
        either way, on a 5-fold LOSO where every fold shares 80% of its
        data with the shipped model). Anything failing it is failing
        badly.
        """
        flagged = []
        for f in spec:
            values = [fold[f] for fold in folds.values()]
            if not values:
                continue
            if any(np.sign(v) != np.sign(shipped[f]) for v in values):
                flagged.append((f, "sign flip"))
                continue
            ratios = [abs(v) / abs(shipped[f]) for v in values if shipped[f]]
            if ratios and (max(ratios) > STABILITY_RATIO
                           or min(ratios) < 1 / STABILITY_RATIO):
                flagged.append((f, f"magnitude {min(ratios):.2f}-{max(ratios):.2f}x"))
        return flagged

    # LEAVE-ONE-CLASS-OUT. With five folds this carries more weight than
    # the p-values do -- see the module docstring on clustering.
    stability = leave_one_class_out(survivors) if survivors else {}
    flagged = unstable(survivors, weights, stability) if survivors else []

    # STAGE 3 -- DROP THE UNSTABLE AND REFIT. ONCE, NOT IN A LOOP.
    #
    # Same logic as stage 2 dropping insignificant terms: a coefficient
    # is only meaningful alongside the constants it was fitted with, so
    # removing a term means the intercept has to be re-estimated rather
    # than carried over. Doing it once is a decision; doing it until
    # everything passes would be fitting the fold structure, which at
    # five folds is fitting five numbers.
    #
    # This is what saves RB. `rush_att_pg` is solid across every fold
    # (-0.37 to -0.53) and would otherwise have been thrown out along
    # with `age`, because trustworthiness is judged per POSITION -- the
    # board takes a position's model or it doesn't.
    # Captured BEFORE the refit overwrites `flagged`. The first run
    # printed "DROPPED AFTER LEAVE-ONE-CLASS-OUT" followed by an empty
    # list, because the reasons being reported were the FINAL model's --
    # which is empty by construction whenever the refit worked. The two
    # are different questions: what got dropped, and whether what
    # remains holds up.
    dropped_reasons = dict(flagged)
    dropped_for_instability = [f for f, _ in flagged]
    if dropped_for_instability:
        survivors = [f for f in survivors if f not in dropped_for_instability]
        if survivors:
            refit_model = sm.OLS(
                y, sm.add_constant(_design_matrix(subset, survivors, centers))
            ).fit()
            intercept = float(refit_model.params[0])
            weights = {f: float(c) for f, c in zip(survivors, refit_model.params[1:])}
            stability = leave_one_class_out(survivors)
            flagged = unstable(survivors, weights, stability)
        else:
            refit_model = None
            intercept = float(np.mean(y))
            weights = {}
            stability = {}
            flagged = []

    sign_flips = [f for f, reason in flagged if reason == "sign flip"]

    result = {
        "intercept": intercept,
        "weights": weights,
        "feature_means": feature_means,
        "centers": centers,
        # The spec ACTUALLY fitted, after the season-confound filter.
        # verify_rookies rebuilds the fit sample to check reconciliation
        # and must drop nulls on the same columns this did -- reading
        # FEATURE_SPECS instead would use a different row set and the
        # identity would fail for a reason that has nothing to do with
        # the weights. Shipping the effective spec beats recomputing the
        # filter in two places and hoping they agree.
        "features_considered": list(features),
        "n": int(subset.height),
        "n_classes": int(subset["season"].n_unique()),
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
        "stability_leave_one_class_out": stability,
        "sign_flips": sign_flips,
        "dropped_for_instability": dropped_for_instability,
        # Why each dropped feature was dropped.
        "instability_flags": dropped_reasons,
        # Whether anything STILL fails after the refit. Non-empty here is
        # what sends a position to the fallback.
        "residual_instability": {f: reason for f, reason in flagged},
        "mean_fitted_adjustment": (
            float(np.mean(refit_model.fittedvalues)) if refit_model is not None
            else float(np.mean(y))
        ),
        "mean_actual_delta": float(np.mean(y)),
        # The fallback decision, made by rule rather than by mood.
        "fit_is_trustworthy": bool(survivors) and not flagged,
    }
    return result, full_model


def print_summary(position, result, full_model, features):
    print(f"\n{'=' * 74}")
    print(f"{position}   n={result['n']} over {result['n_classes']} classes   "
          f"R^2={result['r_squared']:.3f} (full spec {result['r_squared_full_spec']:.3f})")
    print(f"{'=' * 74}")

    if result["imputed_rows"]:
        detail = ", ".join(f"{f}: {n}" for f, n in result["imputed_rows"].items())
        print(f"  mean-imputed at fit time -- {detail}  (kept, not dropped)")
    if result["centers"]:
        detail = ", ".join(f"{f} at {c:.2f}" for f, c in result["centers"].items())
        print(f"  centered -- {detail}")

    print(f"\n  {'feature':<26} {'coef':>10} {'std err':>10} {'p':>8}   kept   shipped")
    print(f"  {'-' * 70}")
    print(f"  {'(intercept)':<26} {result['intercept']:>10.4f} "
          f"{'':>10} {'':>8}   always {result['intercept']:>9.4f}")
    for i, f in enumerate(features):
        d = result["diagnostics"][f]
        mark = "yes" if d["kept"] else "NO"
        shipped = f"{d['coef_shipped']:.4f}" if d["coef_shipped"] is not None else "--"
        print(f"  {f:<26} {d['coef_full_spec']:>10.4f} {full_model.bse[i + 1]:>10.4f} "
              f"{d['p_value']:>8.4f}   {mark:<6} {shipped:>9}")

    gap = result["mean_fitted_adjustment"] - result["mean_actual_delta"]
    print(f"\n  mean applied adjustment {result['mean_fitted_adjustment']:+.4f}  "
          f"vs mean delta {result['mean_actual_delta']:+.4f}   gap {gap:+.6f}")
    if abs(gap) > 1e-6:
        print("  *** RECONCILIATION FAILED -- shipped weights do not match the fit ***")

    if abs(result["mean_actual_delta"]) > LEVEL_SANITY_PPG:
        print(f"\n  *** LEVEL {result['mean_actual_delta']:+.2f} PPG is larger than "
              f"{LEVEL_SANITY_PPG} ***")
        print("  The rookie baseline is computed over the same filtered rows this")
        print("  model is fitted on, so the level should already be netted out.")
        print("  A large level means the filter and the baseline came apart --")
        print("  check MIN_GAMES is applied BEFORE leave_one_class_out_baselines().")

    if result["dropped_for_instability"]:
        print(f"\n  DROPPED AFTER LEAVE-ONE-CLASS-OUT, then refit:")
        for feature, reason in result["instability_flags"].items():
            print(f"    {feature:<26} {reason}")
        print("  These cleared alpha but not the folds. The intercept above is from")
        print("  the model WITHOUT them -- it was re-estimated, not carried over.")

    if result["stability_leave_one_class_out"]:
        folds = result["stability_leave_one_class_out"]
        print(f"\n  leave-one-CLASS-out (coefficient when that class is withheld):")
        header = "".join(f"{s:>11}" for s in folds)
        print(f"    {'feature':<26}{'shipped':>11}{header}")
        for f in result["weights"]:
            row = "".join(f"{folds[s][f]:>11.3f}" for s in folds)
            flag = "   <-- SIGN FLIP" if f in result["sign_flips"] else ""
            print(f"    {f:<26}{result['weights'][f]:>11.3f}{row}{flag}")

    if not result["fit_is_trustworthy"]:
        print(f"\n  >>> {position} DOES NOT MEET THE BAR.")
        if not result["weights"]:
            print("      Nothing survived. The flat cohort baseline stands.")
        else:
            print(f"      Still unstable after the refit: "
                  f"{result['residual_instability']}")
            print("      Per the Phase 12 risk note, prefer the shrinkage haircut")
            print("      over shipping this. Do not talk yourself into it.")


def fit_all(alpha=ALPHA):
    df = load_rookie_backtest()
    print(f"Rookie backtest rows: {df.height} "
          f"across {df['season'].n_unique()} classes")

    confounded = season_confounded_features(df)
    if confounded:
        print("\n  REMOVED FROM EVERY SPEC -- availability tracks the calendar, "
              "not the player:")
        for feature, reason in confounded.items():
            print(f"    {feature}")
            print(f"      {reason}")
        print("  Fitting these would measure the difference between eras under a "
              "feature's name.\n")

    specs = {
        position: [f for f in features if f not in confounded]
        for position, features in FEATURE_SPECS.items()
    }

    output = {
        "_meta": {
            "source": BACKTEST_PATH.name,
            "alpha": alpha,
            "min_rows_to_fit": MIN_ROWS_TO_FIT,
            "season_confounded_removed": confounded,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "note": (
                "Phase 12 rookie weights. Generated by src/fit_rookie_weights.py "
                "-- do not hand-edit. These are fitted against a leave-one-class-"
                "out COHORT baseline, not against a player's own history, and "
                "must never be applied to veterans. Intercepts and centers are "
                "part of the model and ship alongside the weights."
            ),
        },
        "positions": {},
    }

    for position, features in specs.items():
        if not features:
            print(f"\n  {position}: no candidate features left after the season-"
                  f"confound filter -- NOT FITTED.")
            continue
        result, full_model = fit_position(df, position, features, alpha)
        if result is None:
            continue
        print_summary(position, result, full_model, features)
        output["positions"][position] = result

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit Phase 12 rookie weights.")
    parser.add_argument("--alpha", type=float, default=ALPHA,
                        help=f"significance bar (default: {ALPHA})")
    args = parser.parse_args()

    weights = fit_all(args.alpha)

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(WEIGHTS_PATH, "w") as f:
        json.dump(weights, f, indent=2)

    print(f"\n{'=' * 74}")
    print(f"Wrote {WEIGHTS_PATH}")
    trustworthy = []
    for position, result in weights["positions"].items():
        kept = ", ".join(result["weights"]) or "(none significant)"
        print(f"  {position}: intercept {result['intercept']:+.4f} | {kept}")
        if result["fit_is_trustworthy"]:
            trustworthy.append(position)

    print(f"\nPositions meeting the bar: {trustworthy or 'NONE'}")
    if not trustworthy:
        print("\nThat is the documented Phase 12 failure mode, not a bug. The plan's")
        print("fallback is the shrinkage haircut. Record the result and move on --")
        print("a fragile rookie model is worse than the flat baseline, because it")
        print("looks like information.")
    print("\nNext: python -m src.pipeline, then python -m src.verify_rookies")
