"""
Phase 13 CP2. Out-of-sample validation for both models.

THE TEST THIS PROJECT HAS NEVER RUN
-----------------------------------
Every R^2 quoted anywhere in this repo is in-sample. Leave-one-season-out
in fit_weights is a STABILITY probe -- it asks whether a coefficient
survives withholding a season, which is a different and weaker question
than whether the model predicts a season it has never seen. Nine phases
of features have been adopted, reinstated and cut on in-sample evidence
alone.

So: fit on everything up to a cutoff, predict the season after it, and
score against what actually happened.

THREE PREDICTORS, NOT TWO, AND THE THIRD IS THE POINT
------------------------------------------------------
The plan asks "does the model beat the raw baseline out of sample?"
That question can be passed by a model whose every feature is noise, and
it took writing this file to notice.

    RAW       predict delta = 0. The player repeats his own shrunk
              baseline. This is the no-model null.

    LEVEL     predict delta = mean delta in the TRAINING data. No
              features at all -- just the observation that this
              population tends to land slightly above or below its
              baseline.

    MODEL     predict delta = intercept + sum(feature x weight), fitted
              on training seasons only.

MODEL beating RAW proves nothing on its own: the intercept alone can do
that, and the intercept is not a feature. **The number that matters is
MODEL vs LEVEL.** If the fitted features cannot beat a constant out of
sample, they are decoration, however many p-values they cleared. RAW is
kept because it is the plan's stated bar and because LEVEL beating RAW
is itself worth knowing.

WHAT LEAKS, AND WHAT STOPS IT
-----------------------------
Three channels, all of them silent:

1. WEIGHTS. Refit on training seasons only. The shipped
   situational_weights.json saw 2025 and must never be scored against
   it -- reusing it would be the entire bug.
2. IMPUTATION MEANS. A test row with a null feature is filled with the
   TRAINING mean, never the test mean. Filling with the test mean leaks
   the held-out distribution into the prediction.
3. CENTERS. Same rule, same reason. `age` is centered at the training
   mean when scoring test rows.

Feature SELECTION also happens inside the training fold -- alpha is
applied to training p-values, so the surviving spec can differ from the
shipped one. That is correct and worth watching: if the survivors differ,
the shipped spec is partly an artifact of having seen the test season.

ABLATION
--------
For each shipped feature, refit the training data without it and rescore.
A feature whose REMOVAL improves holdout error is actively hurting, and
is a cut candidate for CP1 regardless of its p-value.

USAGE
-----
    python -m src.holdout                    # both models, 2025 held out
    python -m src.holdout --test-season 2024
    python -m src.holdout --model rookie
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm

from src import fit_rookie_weights as rookie
from src import fit_weights as veteran
from src.ranking import _position_adjustment

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Held out by default. The most recent completed season is the honest
# choice -- it is the one most like the season the board is actually
# trying to predict.
DEFAULT_TEST_SEASON = 2025

# Improvement smaller than this is not a result. Delta standard
# deviations run 3-5 PPG, so a hundredth of a point of RMSE is noise
# dressed as a finding.
MEANINGFUL_RMSE_GAIN = 0.05


def _prep(subset, features, imputed, centered, means=None, centers=None):
    """
    Drops rows null in any non-imputed feature, then imputes and centers.

    `means` and `centers` are passed in when preparing TEST rows so that
    the training fold's constants are used. Computing them from the test
    rows would leak the held-out distribution -- quietly, and in the
    direction that flatters the model.
    """
    required = [f for f in features if f not in imputed]
    subset = subset.drop_nulls(subset=required + ["delta"])
    if subset.height == 0:
        return subset, {}, {}

    if means is None:
        means = {}
        for f in features:
            if f in imputed:
                means[f] = float(subset.select(pl.col(f).cast(pl.Float64).mean()).item())
    for f in features:
        if f in imputed and f in means:
            subset = subset.with_columns(
                pl.col(f).cast(pl.Float64).fill_null(means[f])
            )

    if centers is None:
        centers = {
            f: float(subset.select(pl.col(f).cast(pl.Float64).mean()).item())
            for f in features if f in centered
        }
    return subset, means, centers


def _matrix(subset, features, centers):
    columns = []
    for f in features:
        values = subset.select(pl.col(f).cast(pl.Float64)).to_numpy().ravel()
        if f in centers:
            values = values - centers[f]
        columns.append(values)
    return np.column_stack(columns) if columns else np.empty((subset.height, 0))


def _fit(subset, features, centers):
    """Plain OLS with an intercept on a GIVEN feature list -- no alpha
    selection. Used for ablation, where the spec is dictated rather than
    discovered."""
    y = subset.select("delta").to_numpy().ravel().astype(float)
    X = sm.add_constant(_matrix(subset, features, centers), has_constant="add")
    model = sm.OLS(y, X).fit()
    return float(model.params[0]), {
        f: float(c) for f, c in zip(features, model.params[1:])
    }


def _predict(subset, intercept, weights, centers, feature_means):
    """
    Scores test rows through `ranking._position_adjustment` -- the SAME
    expression builder the live pipeline uses.

    Not a convenience. verify_adjustments exists because Phase 6 shipped
    numbers that a restatement of the fit would have blessed and the real
    apply path would not, and the whole value of that file is that it
    runs the code that ships. A holdout that reimplemented the
    arithmetic could report a clean result for a model that the pipeline
    applies incorrectly, which is the same failure one level out.

    It also settles the constants question by construction: the
    intercept, centers and imputation means all travel inside one spec
    dict, so there is no way for a center to arrive without the
    coefficient it was fitted with.
    """
    spec = {
        "intercept": intercept,
        "weights": weights,
        "centers": centers,
        "feature_means": feature_means,
    }
    scored = subset.with_columns(_position_adjustment(spec).alias("_prediction"))
    return scored.select("_prediction").to_numpy().ravel().astype(float)


def _score(actual, predicted):
    error = actual - predicted
    return {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
    }


def run_holdout(df, position, features, spec, test_season, alpha):
    """
    One position, one held-out season.

    `spec` is the module (fit_weights or fit_rookie_weights) supplying
    IMPUTED_FEATURES, CENTERED_FEATURES and fit_position.
    """
    train = df.filter(pl.col("season") != test_season)
    test = df.filter(pl.col("season") == test_season)
    if train.height == 0 or test.height == 0:
        return None

    train_p = train.filter(pl.col("position") == position)
    test_p = test.filter(pl.col("position") == position)
    if train_p.height <= len(features) + 1 or test_p.height == 0:
        return None

    # Feature selection happens INSIDE the training fold.
    result, *_ = spec.fit_position(train_p, position, features, alpha)
    if result is None:
        return None
    survivors = list(result["weights"])

    imputed = spec.IMPUTED_FEATURES
    centered = spec.CENTERED_FEATURES

    # The fit's OWN constants, not recomputed ones. fit_position already
    # decided which rows it trained on and what the means and centers of
    # those rows were; deriving a second set here would risk them
    # drifting apart, and a coefficient applied with the wrong center is
    # precisely the bug Phase 10 nearly shipped.
    centers = result["centers"]
    means = result["feature_means"]

    train_prep, _, _ = _prep(train_p, features, imputed, centered, means, centers)
    test_prep, _, _ = _prep(test_p, features, imputed, centered, means, centers)
    if test_prep.height == 0:
        return None

    actual = test_prep.select("delta").to_numpy().ravel().astype(float)
    train_mean_delta = float(
        train_prep.select("delta").to_numpy().ravel().astype(float).mean()
    )

    # The QB level shift is suppressed in the SHIPPED weights because it
    # is a cross-position calibration artifact, not a prediction claim.
    # Here we are measuring prediction, so the fitted intercept is the
    # right one -- scoring the suppressed version would penalise QB for a
    # decision made about VOR, not about accuracy.
    intercept = result.get("intercept_fitted", result["intercept"])

    scores = {
        "RAW": _score(actual, np.zeros_like(actual)),
        "LEVEL": _score(actual, np.full_like(actual, train_mean_delta)),
        "MODEL": _score(actual, _predict(test_prep, intercept,
                                         result["weights"], centers, means)),
    }

    # ABLATION: drop one shipped feature, refit on train, rescore.
    ablation = {}
    for feature in survivors:
        reduced = [f for f in survivors if f != feature]
        ab_intercept, ab_weights = _fit(train_prep, reduced, centers)
        ab = _score(actual, _predict(test_prep, ab_intercept, ab_weights,
                                     centers, means))
        ablation[feature] = {
            "rmse_without": ab["rmse"],
            # Positive means the feature EARNS its slot out of sample.
            "rmse_gain_from_feature": ab["rmse"] - scores["MODEL"]["rmse"],
        }

    return {
        "position": position,
        "n_train": int(train_prep.height),
        "n_test": int(test_prep.height),
        "survivors_in_fold": survivors,
        "train_mean_delta": train_mean_delta,
        "scores": scores,
        "ablation": ablation,
    }


def print_result(result):
    scores = result["scores"]
    raw, level, model = scores["RAW"], scores["LEVEL"], scores["MODEL"]

    print(f"\n{'=' * 74}")
    print(f"{result['position']}   train n={result['n_train']}  "
          f"test n={result['n_test']}")
    print(f"{'=' * 74}")
    print(f"  survivors in this fold: {result['survivors_in_fold'] or '(none)'}")

    print(f"\n  {'predictor':<10}{'RMSE':>10}{'MAE':>10}   {'vs RAW':>10}{'vs LEVEL':>11}")
    print(f"  {'-' * 53}")
    print(f"  {'RAW':<10}{raw['rmse']:>10.4f}{raw['mae']:>10.4f}")
    print(f"  {'LEVEL':<10}{level['rmse']:>10.4f}{level['mae']:>10.4f}"
          f"{raw['rmse'] - level['rmse']:>+11.4f}")
    print(f"  {'MODEL':<10}{model['rmse']:>10.4f}{model['mae']:>10.4f}"
          f"{raw['rmse'] - model['rmse']:>+11.4f}"
          f"{level['rmse'] - model['rmse']:>+11.4f}")

    gain = level["rmse"] - model["rmse"]
    if gain > MEANINGFUL_RMSE_GAIN:
        print(f"\n  FEATURES EARN THEIR SLOT: {gain:+.4f} RMSE over a constant.")
    elif gain > 0:
        print(f"\n  Features beat a constant by {gain:+.4f} RMSE -- inside the "
              f"{MEANINGFUL_RMSE_GAIN} noise band. Treat as no evidence, not as a win.")
    else:
        print(f"\n  *** FEATURES DO NOT BEAT A CONSTANT OUT OF SAMPLE "
              f"({gain:+.4f} RMSE) ***")
        print("  Everything this position's features add is in-sample. The plan's")
        print("  instruction is explicit: they should be cut.")

    if result["ablation"]:
        print(f"\n  ablation (RMSE change when the feature is REMOVED):")
        print(f"    {'feature':<26}{'RMSE without':>14}{'feature worth':>15}")
        for feature, values in result["ablation"].items():
            worth = values["rmse_gain_from_feature"]
            flag = "" if worth > 0 else "   <-- HURTS out of sample"
            print(f"    {feature:<26}{values['rmse_without']:>14.4f}"
                  f"{worth:>+15.4f}{flag}")


MODELS = {
    "veteran": {
        "module": veteran,
        "path": veteran.BACKTEST_PATH,
        "loader": veteran.load_backtest,
        "specs": veteran.FEATURE_SPECS,
        "label": "VETERAN MODEL",
    },
    "rookie": {
        "module": rookie,
        "path": rookie.BACKTEST_PATH,
        "loader": rookie.load_rookie_backtest,
        "specs": rookie.FEATURE_SPECS,
        "label": "ROOKIE MODEL",
    },
}


def run_model(name, test_season, alpha):
    config = MODELS[name]
    if not config["path"].exists():
        print(f"\n{config['label']}: {config['path'].name} missing -- skipped.")
        return []

    df = config["loader"]()
    print(f"\n\n{'#' * 74}")
    print(f"# {config['label']}  --  fit on everything except {test_season}, "
          f"predict {test_season}")
    print(f"{'#' * 74}")

    if test_season not in df.select("season").unique().to_series().to_list():
        print(f"  {test_season} not present in this training set -- skipped.")
        return []

    specs = config["specs"]
    # The rookie model strips season-confounded features before fitting;
    # the holdout has to strip the same ones or it is validating a spec
    # that never ships.
    if name == "rookie":
        confounded = rookie.season_confounded_features(df)
        specs = {p: [f for f in fs if f not in confounded] for p, fs in specs.items()}

    results = []
    for position, features in specs.items():
        result = run_holdout(df, position, features, config["module"],
                             test_season, alpha)
        if result is None:
            print(f"\n  {position}: too few rows in one of the folds -- skipped.")
            continue
        print_result(result)
        results.append(result)
    return results


GATE_PATH = PROJECT_ROOT / "data" / "holdout_gate.json"

# Folds the gate runs. Three is the minimum that lets "failed once" be
# told apart from "fails".
GATE_SEASONS = [2025, 2024, 2023]


def run_gate(alpha):
    """
    Runs every fold and decides, by rule, whether the current feature
    specs are allowed to ship.

    WHY A GATE AND NOT A REPORT (Aug 6). Today the holdout cut four
    things that alpha had passed, two of which had ALSO passed
    leave-one-season-out, and one of which was a phase's headline
    finding. The evidence standard that let them through is the same one
    still guarding every future refit. A report has to be read and acted
    on; a gate cannot be forgotten at 11pm the night before a draft.

    THE RULE, stated so it cannot drift:
      - a shipped POSITION must beat a constant on average across folds
      - a shipped FEATURE must have non-negative mean ablation value
    Both are averages over folds, not unanimity -- one bad fold out of
    three is noise, and requiring 3/3 would cut features that are real.

    Writes data/holdout_gate.json. build_board refuses to build against
    weights newer than a passing gate.
    """
    import json

    everything = []
    for season in GATE_SEASONS:
        for name in ("veteran", "rookie"):
            everything.extend(
                (name, season, r) for r in run_model(name, season, alpha)
            )

    position_scores, feature_scores = {}, {}
    for name, _season, result in everything:
        key = f"{name}/{result['position']}"
        gain = result["scores"]["LEVEL"]["rmse"] - result["scores"]["MODEL"]["rmse"]
        position_scores.setdefault(key, []).append(gain)
        for feature, values in result["ablation"].items():
            feature_scores.setdefault(f"{key}/{feature}", []).append(
                values["rmse_gain_from_feature"]
            )

    failures = []
    for key, gains in sorted(position_scores.items()):
        mean = sum(gains) / len(gains)
        if mean <= 0:
            failures.append(f"{key}: features add nothing out of sample "
                            f"(mean {mean:+.4f} RMSE over {len(gains)} folds)")
    for key, gains in sorted(feature_scores.items()):
        mean = sum(gains) / len(gains)
        if mean < 0:
            failures.append(f"{key}: hurts out of sample "
                            f"(mean {mean:+.4f} RMSE over {len(gains)} folds)")

    payload = {
        "passed": not failures,
        "seasons": GATE_SEASONS,
        "alpha": alpha,
        "failures": failures,
        "position_mean_gain": {k: sum(v) / len(v) for k, v in position_scores.items()},
        "feature_mean_gain": {k: sum(v) / len(v) for k, v in feature_scores.items()},
    }
    GATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GATE_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n\n{'=' * 74}")
    print(f"GATE  --  folds {GATE_SEASONS}")
    print(f"{'=' * 74}")
    if failures:
        print(f"  FAILED. {len(failures)} item(s) have no out-of-sample value:\n")
        for failure in failures:
            print(f"    {failure}")
        print(f"\n  Remove them from FEATURE_SPECS and refit. Boards will refuse to")
        print(f"  build until this passes.")
    else:
        print("  PASSED. Every shipped position beats a constant and every shipped")
        print("  feature earns its slot, averaged across folds.")
    print(f"\n  Wrote {GATE_PATH}")
    return 0 if payload["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description="Phase 13 CP2 holdout validation.")
    parser.add_argument("--test-season", type=int, default=DEFAULT_TEST_SEASON)
    parser.add_argument("--model", choices=["veteran", "rookie", "both"],
                        default="both")
    parser.add_argument("--alpha", type=float, default=veteran.ALPHA)
    parser.add_argument("--gate", action="store_true",
                        help="run all folds and write data/holdout_gate.json; "
                             "exits non-zero if anything shipped fails")
    args = parser.parse_args()

    if args.gate:
        raise SystemExit(run_gate(args.alpha))

    names = ["veteran", "rookie"] if args.model == "both" else [args.model]
    everything = []
    for name in names:
        everything.extend(
            (name, r) for r in run_model(name, args.test_season, args.alpha)
        )

    print(f"\n\n{'=' * 74}")
    print(f"SUMMARY  --  held out {args.test_season}")
    print(f"{'=' * 74}")
    print(f"  {'model':<9}{'pos':<5}{'RMSE raw':>10}{'RMSE level':>12}"
          f"{'RMSE model':>12}{'features worth':>16}")
    cut_candidates = []
    for name, result in everything:
        scores = result["scores"]
        gain = scores["LEVEL"]["rmse"] - scores["MODEL"]["rmse"]
        print(f"  {name:<9}{result['position']:<5}"
              f"{scores['RAW']['rmse']:>10.4f}{scores['LEVEL']['rmse']:>12.4f}"
              f"{scores['MODEL']['rmse']:>12.4f}{gain:>+16.4f}")
        if gain <= 0:
            cut_candidates.append(f"{name}/{result['position']}")
        for feature, values in result["ablation"].items():
            if values["rmse_gain_from_feature"] <= 0:
                cut_candidates.append(f"{name}/{result['position']}/{feature}")

    print()
    if cut_candidates:
        print("  CUT CANDIDATES (no out-of-sample value):")
        for candidate in cut_candidates:
            print(f"    {candidate}")
        print("\n  A single held-out season is one draw, so this is evidence and not")
        print("  a verdict -- re-run with --test-season 2024 and 2023 before cutting")
        print("  anything. But a feature that fails all three has no defence left.")
    else:
        print("  Nothing failed out of sample. Re-run against 2024 and 2023 before")
        print("  believing that -- one season is one draw.")


if __name__ == "__main__":
    main()
