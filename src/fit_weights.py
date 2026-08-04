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
# `usage_trend_share` ships at RB and TE. It was tested against two
# other bases -- raw per-game volume and share-slope-over-mean-share --
# and beat both. It FAILED at WR (p=0.23, and p=0.96 on 3-season-only
# players) and is cut there, though it still prints on the board as a
# reference column.
FEATURE_SPECS = {
    "RB": ["continuity_score", "workload_share", "age", "usage_trend_share", "trend_missing"],
    "WR": ["team_changed", "workload_share", "recent_major_injury", "age"],
    "TE": ["workload_share", "age", "usage_trend_share", "trend_missing"],
    # QB deliberately absent: nothing tested significant across every
    # specification tried in Phase 5-6, and Phase 10 retested it with
    # age -- the one genuinely new input QB had never seen -- at p=0.68
    # linear and p=0.52 quadratic. An empty entry here would write an
    # intercept-only model, which would shift every QB by a constant and
    # change nothing about their relative order. QBs pass through
    # unadjusted in ranking.py instead.
}

# Features entered as deviations from the position mean. The mean is
# written to the JSON so ranking.py can reproduce it.
CENTERED_FEATURES = {"age"}

# Features whose nulls are mean-imputed at FIT time rather than causing
# the row to be dropped. Only for features with a paired missing
# indicator in the spec -- otherwise imputation quietly invents data.
IMPUTED_FEATURES = {"usage_trend_share"}

BOOL_COLUMNS = [
    "qb_changed", "coach_changed", "team_changed",
    "recent_major_injury", "trend_missing", "trend_low_confidence",
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

    result = {
        "intercept": intercept,
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
        # The reconciliation that would have caught Phase 6. OLS with an
        # intercept forces mean(fitted) == mean(y) exactly, so any
        # meaningful gap here means the shipped numbers do not come from
        # the model that was fitted.
        "mean_fitted_adjustment": (
            float(np.mean(refit_model.fittedvalues)) if refit_model is not None
            else float(np.mean(y))
        ),
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

    gap = result["mean_fitted_adjustment"] - result["mean_actual_delta"]
    print(f"\n  mean fitted adjustment {result['mean_fitted_adjustment']:+.4f}  "
          f"vs mean actual delta {result['mean_actual_delta']:+.4f}   "
          f"gap {gap:+.6f}")
    if abs(gap) > 1e-6:
        print("  *** RECONCILIATION FAILED -- shipped weights do not match the fit ***")

    if result["stability_leave_one_season_out"]:
        folds = result["stability_leave_one_season_out"]
        print(f"\n  leave-one-season-out (coefficient when that season is withheld):")
        header = "".join(f"{s:>11}" for s in folds)
        print(f"    {'feature':<24}{'shipped':>11}{header}")
        for f in result["weights"]:
            row = "".join(f"{folds[s][f]:>11.3f}" for s in folds)
            flag = "   <-- SIGN FLIP" if f in result["sign_flips"] else ""
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

    print("\nNext: python -m src.pipeline, then python -m src.verify_adjustments")
