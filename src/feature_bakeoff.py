"""
Phase 13 extended. Two questions, one instrument.

    python -m src.feature_bakeoff posrank   # level vs promotion vs both
    python -m src.feature_bakeoff audit     # re-test everything we cut
    python -m src.feature_bakeoff both

WHY ONE FILE
------------
Both questions have the same shape as the competition bake-off: take a
position's shipped spec, swap one thing, refit, and score on the three
holdout folds. Writing a third near-identical script would guarantee the
three drift apart. This generalizes the pattern and
`competition_bakeoff.py` remains as the worked example.

QUESTION 1 -- POSRANK: LEVEL, PROMOTION, OR BOTH
------------------------------------------------
The shipped feature is depth chart POSITION. The objection is sharp:
why should a back who was RB1 last year and is RB1 again be paid for it?
The model predicts change from a player's own baseline, and nothing about
him changed.

The level's defence is that the baseline is a THREE-YEAR weighted
average. A current RB1 whose 2023-24 were spent as RB2 has a baseline
that understates his role, and should beat it. That is information the
level carries and a promotion flag does not -- his rank did not change
this year, but it is above where his baseline was earned.

The promotion version states the hypothesis more directly and is easier
to defend to a human. Both are plausible; only one can be shipped without
the other unless both earn a slot. Four models per position: neither,
level only, change only, both.

QUESTION 2 -- THE AUDIT
-----------------------
Every feature currently in the model has been holdout-tested. **Nothing
we cut ever was.** Everything below was removed on in-sample p-values,
which QB `age` just demonstrated can be confidently wrong:

    coach_changed              cut Phase 13 CP3, p=0.36 at RB
    returning_oline_starters   cut as a suspected false positive
    experience                 replaced by age on in-sample AIC
    continuity_score           retired in favour of qb_changed alone
    age_squared                never adopted; "flat tail rests on 57
                               backs over 30"
    qb_changed                 cut Aug 6 on the holdout -- included here
                               as a CONTROL, since it should fail again

If in-sample evidence can keep something that does not predict, it can
also discard something that does. This tests the second direction.

READING THE OUTPUT
------------------
Same rule as the gate: mean MODEL-minus-LEVEL RMSE across three folds.
A candidate has to beat the shipped spec by more than DISPLACEMENT_MARGIN
to be worth changing anything, because churn on three folds is noise.

Nothing is applied automatically.
"""

import argparse

import polars as pl

from src import fit_weights as veteran
from src.holdout import GATE_SEASONS, run_holdout

DISPLACEMENT_MARGIN = 0.02

# Question 1. Each entry is what REPLACES the shipped pos_rank pair.
POSRANK_VARIANTS = {
    "neither": [],
    "level (shipped)": ["pos_rank", "depth_chart_missing"],
    "promotion": ["pos_rank_change", "pos_rank_change_missing"],
    "both": ["pos_rank", "depth_chart_missing",
             "pos_rank_change", "pos_rank_change_missing"],
}
POSRANK_FEATURES = {
    "pos_rank", "depth_chart_missing",
    "pos_rank_change", "pos_rank_change_missing",
}

# Question 2. Each is ADDED to the shipped spec, one at a time.
AUDIT_CANDIDATES = [
    "coach_changed",
    "returning_oline_starters",
    "experience",
    "continuity_score",
    "qb_changed",
    "age_squared",
]


def prepare(df):
    """Adds derived audit columns that aren't stored in the CSV."""
    if "age" in df.columns and "age_squared" not in df.columns:
        df = df.with_columns((pl.col("age") ** 2).alias("age_squared"))
    return df


def score(df, position, spec, alpha, seasons=None, detail=False):
    """
    Mean MODEL-minus-LEVEL RMSE over the fold seasons. None if no fold
    can be fitted. With `detail`, returns (mean, per-fold list) so a
    single number can be checked against the spread that produced it.
    """
    gains = []
    for season in (seasons or GATE_SEASONS):
        result = run_holdout(df, position, spec, veteran, season, alpha)
        if result is None:
            continue
        gains.append(
            result["scores"]["LEVEL"]["rmse"] - result["scores"]["MODEL"]["rmse"]
        )
    if not gains:
        return (None, []) if detail else None
    mean = sum(gains) / len(gains)
    return (mean, gains) if detail else mean


def probe(df, position, candidate, alpha, seasons):
    """
    Every fold, for ONE candidate, with its spread printed.

    WHY THIS EXISTS. The three-fold audit returned exactly one hit --
    `continuity_score` at RB, +0.0332 forced in. Against the 0.0121
    spread of the other seventeen tests that is 2.7 standard deviations,
    which sounds decisive until you remember eighteen tests were run: the
    family-wise chance of one noise draw that large is about 0.05. Right
    on the line, and the churn band it cleared had no multiple-testing
    correction in it.

    So: more folds. Nine estimates instead of three shows whether the
    number is a property of the feature or of the 2023-2025 window. A
    mean that survives with a spread narrower than itself is evidence; a
    mean carried by one fold is the same lesson QB `age` already taught.
    """
    features = veteran.FEATURE_SPECS[position]
    base_mean, base_folds = score(df, position, list(features), 1.0,
                                  seasons, detail=True)
    with_mean, with_folds = score(df, position, list(features) + [candidate],
                                  1.0, seasons, detail=True)
    if base_mean is None or with_mean is None:
        print(f"   {position}/{candidate}: unfittable")
        return

    deltas = [w - b for w, b in zip(with_folds, base_folds)]
    import numpy as np
    mean, sd = float(np.mean(deltas)), float(np.std(deltas, ddof=1))

    print(f"\n   {position} + {candidate}   ({len(deltas)} folds, forced in)")
    print("     " + "  ".join(f"{s}:{d:+.3f}" for s, d in zip(seasons, deltas)))
    print(f"     mean {mean:+.4f}   sd {sd:+.4f}   "
          f"mean/sd {mean / sd if sd else float('nan'):+.2f}")
    positive = sum(1 for d in deltas if d > 0)
    print(f"     positive in {positive} of {len(deltas)} folds")
    if mean > DISPLACEMENT_MARGIN and positive >= len(deltas) * 0.75:
        print("     -> HOLDS UP. Worth reinstating.")
    else:
        print("     -> DOES NOT HOLD. The three-fold result was the window,")
        print("        not the feature.")


def run_posrank(df, alpha):
    print(f"\n{'#' * 74}")
    print("# POSRANK -- is it the position, the promotion, or both?")
    print(f"{'#' * 74}")

    missing = [c for c in POSRANK_FEATURES if c not in df.columns]
    if missing:
        raise SystemExit(
            f"\n{missing} not in backtest_features.csv.\n"
            f"Rebuild it first:  python -m src.backtest\n"
        )

    for position, features in veteran.FEATURE_SPECS.items():
        base = [f for f in features if f not in POSRANK_FEATURES]
        print(f"\n{'=' * 74}")
        print(f"{position}   base spec: {base}")
        print(f"{'=' * 74}")
        print(f"   {'variant':<18}{'mean gain':>12}")

        scores = {}
        for name, extra in POSRANK_VARIANTS.items():
            value = score(df, position, base + extra, alpha)
            if value is None:
                continue
            scores[name] = value
            print(f"   {name:<18}{value:>+12.4f}")

        if not scores:
            continue
        best = max(scores, key=scores.get)
        shipped = "level (shipped)" if position == "RB" else "neither"
        margin = scores[best] - scores.get(shipped, float("-inf"))
        if best == shipped or margin <= DISPLACEMENT_MARGIN:
            print(f"\n   -> KEEP {shipped}"
                  + (f" ({best} leads by only {margin:+.4f}, inside the "
                     f"{DISPLACEMENT_MARGIN} churn band)" if best != shipped else ""))
        else:
            print(f"\n   -> SWITCH to {best} (+{margin:.4f} over {shipped})")


def run_audit(df, alpha):
    print(f"\n{'#' * 74}")
    print("# AUDIT -- do any of the features we CUT actually predict?")
    print(f"{'#' * 74}")
    print("# Every one was removed on in-sample p-values alone. QB `age`")
    print("# proved that evidence can keep something worthless; this asks")
    print("# whether it also discarded something real.")

    for position, features in veteran.FEATURE_SPECS.items():
        available = [c for c in AUDIT_CANDIDATES if c in df.columns
                     and c not in features]
        if not available:
            continue

        # TWO COLUMNS, AND THE SECOND IS THE ONE THAT ANSWERS THE
        # QUESTION (added Aug 7, after the first run half-failed).
        #
        # "alpha decides" adds the candidate and lets the two-stage fit
        # keep or drop it. That is what shipping would do -- but if alpha
        # rejects it in every fold the model never changes, the score
        # comes back identical, and a column of +0.0000 looks like "no
        # effect" while actually meaning "never tested." Four of the six
        # candidates did exactly that on the first run.
        #
        # That made the audit circular: it used alpha to decide whether
        # to test a feature, when alpha is the mechanism under suspicion.
        # QB `age` cleared alpha and did not predict; the reverse case is
        # a feature that predicts and never clears alpha, and only the
        # forced column can see it.
        #
        # "forced" runs at alpha=1.0, which keeps EVERY term in the spec.
        # Both arms are forced, so the difference still isolates the
        # candidate -- the baseline shifts, which is why the shipped
        # score is printed for each column separately.
        shipped_alpha = score(df, position, list(features), alpha)
        shipped_forced = score(df, position, list(features), 1.0)

        print(f"\n{'=' * 74}")
        print(f"{position}   shipped spec: {shipped_alpha:+.4f} (alpha decides), "
              f"{shipped_forced:+.4f} (all terms forced)")
        print(f"{'=' * 74}")
        print(f"   {'added feature':<28}{'alpha decides':>15}{'forced in':>12}")

        for candidate in available:
            spec = list(features) + [candidate]
            by_alpha = score(df, position, spec, alpha)
            forced = score(df, position, spec, 1.0)

            if by_alpha is None or forced is None:
                print(f"   {candidate:<28}{'--':>15}{'--':>12}   (unfittable)")
                continue

            delta_alpha = by_alpha - shipped_alpha
            delta_forced = forced - shipped_forced

            note = ""
            if abs(delta_alpha) < 1e-9:
                note = "   never selected"
            if delta_forced > DISPLACEMENT_MARGIN:
                note = "   <-- PREDICTS. investigate."
            print(f"   {candidate:<28}{delta_alpha:>+15.4f}{delta_forced:>+12.4f}{note}")

    print("\n   'alpha decides' = what shipping it would do. +0.0000 there means")
    print("   the fit rejected it in every fold, NOT that it has no effect.")
    print("   'forced in' = does it predict at all, with the significance bar")
    print("   removed. That is the column this audit exists for.")
    print("\n   `qb_changed` is a CONTROL -- it failed the gate on Aug 6, so if it")
    print("   looks good here the harness is flattering, not the feature.")


def main():
    parser = argparse.ArgumentParser(description="Phase 13 extended bake-offs.")
    parser.add_argument("mode", choices=["posrank", "audit", "both", "probe"],
                        default="both", nargs="?")
    parser.add_argument("--alpha", type=float, default=veteran.ALPHA)
    parser.add_argument("--candidate", type=str, default="continuity_score",
                        help="feature to probe across every fold (probe mode)")
    parser.add_argument("--position", type=str, default="RB")
    args = parser.parse_args()

    df = prepare(veteran.load_backtest())
    print(f"Backtest rows: {df.height}")

    if args.mode == "probe":
        seasons = sorted(df.select("season").unique().to_series().to_list())
        # The earliest season cannot be a test fold with anything left to
        # train on in the same direction, but run_holdout trains on
        # "everything except", so every season is usable.
        print(f"\n{'#' * 74}")
        print(f"# PROBE -- {args.candidate} at {args.position}, all "
              f"{len(seasons)} folds")
        print(f"{'#' * 74}")
        probe(df, args.position, args.candidate, args.alpha, seasons)
        return

    if args.mode in ("posrank", "both"):
        run_posrank(df, args.alpha)
    if args.mode in ("audit", "both"):
        run_audit(df, args.alpha)

    print("\n\nNothing was applied. Change FEATURE_SPECS only if something cleared")
    print("the churn band, then:")
    print("   python -m src.fit_weights && python -m src.holdout --gate")


if __name__ == "__main__":
    main()
