"""
Guard rail against the class of bug that produced Phase 7's complaint.

WHAT THIS IS FOR
----------------
Phase 6 shipped slopes without their intercept. Nothing in the codebase
noticed. The error surfaced as a vague human impression -- "every
adjustment is negative," "only 6 skill players clear 15 PPG" -- two
symptoms that took until Phase 8 to trace back to one dropped constant.

The lesson recorded at the time was "intercepts always ship with
coefficients." That's necessary but not sufficient, because the same
failure has more than one costume:

  - Phase 6: intercept fitted, never applied.
  - Phase 10 would have added: centered coefficient applied to
    uncentered data (wrong by coef x center -- about -9.5 PPG at RB).
  - Phase 10 would also have added: intercept taken from the full spec
    while only the significant slopes ship, so the two come from
    different models.

All three are the same mistake -- a coefficient separated from the
constants it was fitted with -- and all three are invisible to a human
reading a spreadsheet. So this file checks the thing they all break.

THE CENTRAL IDENTITY
--------------------
OLS with an intercept forces mean(fitted) == mean(y) exactly. The model
is fitted on `delta` = actual PPG - baseline PPG. So if the weights are
applied correctly, then over the rows the model was fitted on:

    mean(situational_adjustment) == mean(delta)

to floating-point precision. Not approximately. Any visible gap means
the numbers being applied did not come from the model that was fitted.
Under Phase 6's bug this gap was -3.43 at RB.

Crucially, this runs the REAL apply path -- ranking.apply_situational_weights,
the same function pipeline.py calls -- against the fit sample rebuilt
from fit_weights' own rules. It tests the code that ships, not a
restatement of the fit.

USAGE
-----
    python -m src.verify_adjustments

Exits non-zero if any hard check fails, so it can gate a commit.
"""

import json
import sys
from pathlib import Path

import polars as pl

from src.fit_weights import (
    BACKTEST_PATH,
    FEATURE_SPECS,
    IMPUTED_FEATURES,
    WEIGHTS_PATH,
    load_backtest,
)
from src.ranking import apply_situational_weights, load_situational_weights

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLAYER_FEATURES_PATH = PROJECT_ROOT / "data" / "player_features.csv"

# mean(fitted) == mean(y) is an algebraic identity, so the only slack
# needed is floating point.
RECONCILIATION_TOLERANCE = 1e-6

# The Phase 7 complaint, as a number: the raw baseline had 29 skill
# players over 15 PPG and the phantom penalty erased 23 of them.
HIGH_PPG_THRESHOLD = 15.0


class Check:
    """Collects pass/fail results so every check runs before exiting."""

    def __init__(self):
        self.failures = []
        self.warnings = []

    def hard(self, ok, label, detail=""):
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}" + (f"  --  {detail}" if detail else ""))
        if not ok:
            self.failures.append(label)

    def soft(self, ok, label, detail=""):
        status = "ok" if ok else "WARN"
        print(f"  [{status}] {label}" + (f"  --  {detail}" if detail else ""))
        if not ok:
            self.warnings.append(label)


def check_weights_file(check):
    """Structural integrity of situational_weights.json."""
    print("\n1. WEIGHTS FILE STRUCTURE")

    if not WEIGHTS_PATH.exists():
        check.hard(False, "situational_weights.json exists",
                   "run `python -m src.fit_weights`")
        return None

    with open(WEIGHTS_PATH) as f:
        payload = json.load(f)

    positions = payload.get("positions", {})
    check.hard(bool(positions), "weights file has fitted positions",
               f"{sorted(positions)}")

    for position, spec in positions.items():
        check.hard("intercept" in spec, f"{position}: intercept present")

        means = spec.get("feature_means", {})
        missing_means = [f for f in spec["weights"] if f not in means]
        check.hard(
            not missing_means,
            f"{position}: every shipped weight has a feature mean to impute with",
            f"missing: {missing_means}" if missing_means else "",
        )

        # A center without a weight is harmless; a weight that SHOULD be
        # centered but has no center is the -9.5 PPG bug.
        centers = spec.get("centers", {})
        uncentered = [f for f in spec["weights"] if f in {"age"} and f not in centers]
        check.hard(
            not uncentered,
            f"{position}: centered features ship with their center",
            f"missing center: {uncentered}" if uncentered else "",
        )

        # A suppressed level shift must be recorded, not just applied --
        # otherwise a later reader sees an intercept that doesn't match
        # the fit and has no way to tell deliberate from broken.
        if spec.get("level_shift_removed"):
            check.hard(
                "intercept_fitted" in spec,
                f"{position}: suppressed level shift records the fitted intercept",
                f"ships {spec['intercept']:+.4f}, fitted "
                f"{spec.get('intercept_fitted', float('nan')):+.4f}",
            )

        flips = spec.get("sign_flips", [])
        check.soft(
            not flips,
            f"{position}: no coefficient flips sign when a season is withheld",
            f"unstable: {flips}" if flips else "",
        )

    # A weights file older than the data it claims to describe is a
    # silent way to ship last week's model.
    if BACKTEST_PATH.exists():
        stale = WEIGHTS_PATH.stat().st_mtime < BACKTEST_PATH.stat().st_mtime
        check.hard(
            not stale,
            "weights are newer than backtest_features.csv",
            "weights are STALE -- re-run fit_weights" if stale else "",
        )

    return positions


def rebuild_fit_sample(df, position, features):
    """
    Reproduces exactly the rows fit_weights.fit_position() trained on:
    drop rows null in any non-imputed feature, then mean-impute the rest.
    """
    required = [f for f in features if f not in IMPUTED_FEATURES]
    subset = df.filter(pl.col("position") == position).drop_nulls(
        subset=required + ["delta"]
    )
    for f in features:
        if f in IMPUTED_FEATURES:
            mean_value = subset.select(pl.col(f).cast(pl.Float64).mean()).item()
            subset = subset.with_columns(pl.col(f).cast(pl.Float64).fill_null(mean_value))
    return subset


def check_reconciliation(check, positions):
    """
    THE test. Runs the live apply path over the fit sample and requires
    mean(adjustment) == mean(delta).
    """
    print("\n2. RECONCILIATION  --  mean applied adjustment vs mean actual delta")
    print("   (OLS identity: these are equal to floating point when weights are applied correctly)")

    if not BACKTEST_PATH.exists():
        check.hard(False, "backtest_features.csv exists",
                   "run `python -m src.backtest`")
        return

    df = load_backtest()
    weights = load_situational_weights()

    print(f"\n   {'pos':<5}{'n':>6}{'mean applied':>15}{'expected':>14}{'gap':>13}  note")
    for position, features in FEATURE_SPECS.items():
        if position not in positions:
            continue
        subset = rebuild_fit_sample(df, position, features)

        # apply_situational_weights needs these two columns; the
        # baseline is irrelevant to the adjustment itself, so zero it and
        # read the adjustment directly.
        scored = apply_situational_weights(
            subset.with_columns([
                pl.lit(0.0).alias("fantasy_points_per_game"),
                pl.lit(False).alias("is_rookie"),
            ]),
            weights,
        )

        applied = scored.select(pl.col("situational_adjustment").mean()).item()
        actual = subset.select(pl.col("delta").mean()).item()

        # A position whose level shift was deliberately suppressed (see
        # fit_weights.SUPPRESS_LEVEL_SHIFT) should reconcile to
        # mean(delta) MINUS that shift, not to mean(delta). Checking the
        # unadjusted identity here would fail QB for doing exactly what
        # it was told to do -- and, worse, would tempt someone to
        # loosen the tolerance, which is the one check in this file that
        # must stay exact.
        shift = float(positions[position].get("level_shift_removed", 0.0) or 0.0)
        expected = actual - shift
        gap = applied - expected
        note = f"level shift {shift:+.3f} suppressed" if shift else ""
        print(f"   {position:<5}{subset.height:>6}{applied:>15.6f}{expected:>14.6f}"
              f"{gap:>13.2e}  {note}")

        check.hard(
            abs(gap) < RECONCILIATION_TOLERANCE,
            f"{position}: applied adjustment reconciles with fitted model",
            f"gap {gap:+.4f} PPG on every {position}" if abs(gap) >= RECONCILIATION_TOLERANCE else "",
        )


def check_live_board(check, positions):
    """
    Symptom-level checks on the actual output -- the things a human
    noticed in Phase 7, now measured rather than eyeballed.
    """
    print("\n3. LIVE OUTPUT  --  data/player_features.csv")

    if not PLAYER_FEATURES_PATH.exists():
        check.soft(False, "player_features.csv exists",
                   "run `python -m src.pipeline`")
        return

    df = pl.read_csv(PLAYER_FEATURES_PATH, infer_schema_length=0).with_columns([
        pl.col("fantasy_points_per_game").cast(pl.Float64),
        pl.col("adjusted_fantasy_points_per_game").cast(pl.Float64),
        pl.col("situational_adjustment").cast(pl.Float64),
        pl.col("is_rookie").cast(pl.String).str.to_lowercase().eq("true"),
    ])

    veterans = df.filter(~pl.col("is_rookie"))

    print(f"\n   {'pos':<5}{'n':>6}{'mean adj':>11}{'min':>9}{'max':>9}{'% pos':>8}{'% neg':>8}")
    for position in positions:
        p = veterans.filter(pl.col("position") == position)
        if p.height == 0:
            continue
        adj = p.select("situational_adjustment").to_series()
        share_positive = float((adj > 0).mean())
        share_negative = float((adj < 0).mean())
        print(f"   {position:<5}{p.height:>6}{adj.mean():>11.3f}{adj.min():>9.3f}"
              f"{adj.max():>9.3f}{100 * share_positive:>7.0f}%{100 * share_negative:>7.0f}%")

        # The Phase 7 symptom, stated as a test: an adjustment that only
        # ever points one direction is a constant wearing a disguise.
        check.hard(
            share_positive > 0 and share_negative > 0,
            f"{position}: adjustments are two-sided",
            f"{100 * share_positive:.0f}% positive / {100 * share_negative:.0f}% negative",
        )

    skill = df.filter(pl.col("position").is_in(["RB", "WR", "TE"]))
    raw_high = skill.filter(pl.col("fantasy_points_per_game") > HIGH_PPG_THRESHOLD).height
    adj_high = skill.filter(
        pl.col("adjusted_fantasy_points_per_game") > HIGH_PPG_THRESHOLD
    ).height
    print(f"\n   skill players over {HIGH_PPG_THRESHOLD} PPG: raw {raw_high}, adjusted {adj_high}")
    check.soft(
        adj_high >= raw_high * 0.5,
        f"adjusted >{HIGH_PPG_THRESHOLD} PPG count is not decimated",
        f"raw {raw_high} -> adjusted {adj_high} "
        f"({100 * adj_high / raw_high:.0f}% retained)" if raw_high else "",
    )

    rookies = df.filter(pl.col("is_rookie"))
    if rookies.height:
        check.hard(
            float(rookies.select(pl.col("situational_adjustment").abs().max()).item()) == 0.0,
            "rookies take no situational adjustment",
        )


def main():
    print("=" * 74)
    print("VERIFYING SITUATIONAL ADJUSTMENTS")
    print("=" * 74)

    check = Check()
    positions = check_weights_file(check)
    if positions:
        check_reconciliation(check, positions)
        check_live_board(check, positions)

    print("\n" + "=" * 74)
    if check.failures:
        print(f"FAILED ({len(check.failures)}):")
        for f in check.failures:
            print(f"  - {f}")
        print("\nDo not build a board from this. Fix the fit or the apply path first.")
        return 1

    if check.warnings:
        print(f"PASSED with {len(check.warnings)} warning(s):")
        for w in check.warnings:
            print(f"  - {w}")
    else:
        print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
