"""
Fits the per-position situational weights used by ranking.py, and writes
them to data/situational_weights.json.

WHY THIS FILE EXISTS
--------------------
Through Phase 6 the weights in ranking.py were hand-copied out of a
notebook. That notebook isn't in the repo, so the fit wasn't
reproducible -- and the transcription silently dropped the regression's
constant term. Because `workload_share` and `experience` are
always-positive features carrying negative slopes, losing a positive
intercept forced EVERY player's adjustment negative. RB lost 3.59
points, WR 2.37, TE 1.58, uniformly, for no statistical reason.

Fitting and writing the coefficients in one step means the intercept
can't get separated from the slopes again. If a spec changes, re-run
this file; nothing gets retyped.

USAGE
-----
    python -m src.fit_weights

Reads data/backtest_features.csv (build it with `python -m src.backtest`
first if it's missing or stale). Prints a full regression summary per
position, then writes data/situational_weights.json.
"""

import json
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
# contract_year in Phase 6.
ALPHA = 0.10

# Per-position candidate feature sets. These mirror the specs that
# survived Phase 6 testing. Adding a feature here is how you test it --
# fit, read the p-value, keep or cut.
FEATURE_SPECS = {
    "RB": ["continuity_score", "workload_share", "experience"],
    "WR": ["team_changed", "workload_share", "recent_major_injury", "experience"],
    "TE": ["workload_share", "experience"],
    # QB deliberately absent: nothing tested significant across every
    # specification tried in Phase 5-6. An empty entry here would write
    # an intercept-only model, which would shift every QB by a constant
    # and change nothing about their relative order -- pointless. QBs
    # pass through unadjusted in ranking.py instead.
}


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

    bool_columns = ["qb_changed", "coach_changed", "team_changed", "recent_major_injury"]
    df = df.with_columns([
        pl.col(c).cast(pl.String).str.to_lowercase().eq("true").cast(pl.Int8).alias(c)
        for c in bool_columns
    ])

    df = df.with_columns(
        (pl.col("qb_changed") + pl.col("coach_changed")).alias("continuity_score")
    )

    return df.filter(pl.col("actual_games_played") >= MIN_GAMES)


def fit_position(df, position, features, alpha=ALPHA):
    """
    Fits delta ~ features for one position, WITH an intercept.

    The target is `delta` = actual PPG minus the player's own trailing
    3-year baseline. So the model answers "given this situation, how far
    from his own history should we expect him to land?" -- and the
    intercept is the honest answer for a player whose features are all
    zero. Dropping it isn't a neutral simplification; it's asserting
    that such a player would exactly repeat his baseline, which the data
    does not say.

    Returns (result_dict, statsmodels_result). result_dict is None if
    the position has too little usable data to fit.
    """
    subset = df.filter(pl.col("position") == position).drop_nulls(
        subset=features + ["delta"]
    )

    if subset.height <= len(features) + 1:
        print(f"  {position}: only {subset.height} usable rows -- skipping fit")
        return None, None

    X = subset.select(features).to_numpy().astype(float)
    y = subset.select("delta").to_numpy().ravel().astype(float)

    model = sm.OLS(y, sm.add_constant(X)).fit()

    # model.params[0] is the constant; the rest align with `features`.
    intercept = float(model.params[0])
    coefficients = {f: float(c) for f, c in zip(features, model.params[1:])}
    p_values = {f: float(p) for f, p in zip(features, model.pvalues[1:])}

    significant = {f: c for f, c in coefficients.items() if p_values[f] < alpha}

    # Mean of each feature over the exact rows this model was fit on.
    # ranking.py uses these to impute missing values instead of filling
    # with 0. That distinction did not matter before -- with no
    # intercept, a zero row produced a zero adjustment, i.e. "no
    # opinion." With an intercept it matters a lot: filling
    # workload_share with 0 hands the player the full positive intercept
    # and none of the offsetting usage penalty, which would systematically
    # inflate exactly the players whose workload_share gets nulled out
    # (everyone who changed teams -- see situational.py's note on why it's
    # nulled). Imputing the mean instead makes the fallback "an average
    # player at this position," which is what "no opinion" should mean.
    feature_means = {
        f: float(subset.select(pl.col(f).cast(pl.Float64).mean()).item())
        for f in features
    }

    result = {
        "intercept": intercept,
        "weights": significant,
        "feature_means": feature_means,
        "n": int(subset.height),
        "r_squared": float(model.rsquared),
        "diagnostics": {
            f: {"coef": coefficients[f], "p_value": p_values[f], "kept": f in significant}
            for f in features
        },
    }
    return result, model


def print_summary(position, result, model, features):
    """Human-readable regression report, so the numbers can be checked
    before they're trusted."""
    print(f"\n{'=' * 68}")
    print(f"{position}   n={result['n']}   R^2={result['r_squared']:.3f}")
    print(f"{'=' * 68}")
    print(f"  {'feature':<26} {'coef':>10} {'std err':>10} {'p':>8}   kept")
    print(f"  {'-' * 62}")
    print(f"  {'(intercept)':<26} {result['intercept']:>10.4f} "
          f"{model.bse[0]:>10.4f} {model.pvalues[0]:>8.4f}   always")
    for i, f in enumerate(features):
        d = result["diagnostics"][f]
        mark = "yes" if d["kept"] else "NO"
        print(f"  {f:<26} {d['coef']:>10.4f} {model.bse[i + 1]:>10.4f} "
              f"{d['p_value']:>8.4f}   {mark}")

    # What the adjustment actually looks like in aggregate. If this mean
    # is wildly off the sample mean delta, something is wrong.
    fitted = model.fittedvalues
    print(f"\n  mean fitted adjustment: {np.mean(fitted):+.2f}  "
          f"(range {np.min(fitted):+.2f} to {np.max(fitted):+.2f})")
    print(f"  share positive: {100 * np.mean(fitted > 0):.0f}%   "
          f"share negative: {100 * np.mean(fitted < 0):.0f}%")


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
            "note": (
                "Generated by src/fit_weights.py. Do not hand-edit -- "
                "re-run the script. Intercepts are part of the model and "
                "must be applied alongside the weights."
            ),
        },
        "positions": {},
    }

    for position, features in FEATURE_SPECS.items():
        result, model = fit_position(df, position, features, alpha)
        if result is None:
            continue
        print_summary(position, result, model, features)
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
