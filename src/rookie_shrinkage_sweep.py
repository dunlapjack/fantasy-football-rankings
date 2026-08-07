"""
Phase 12 fallback. Finds how hard to discount rookie cohort baselines,
by measuring rather than by picking a number.

THE PROBLEM
-----------
Every veteran's baseline is shrunk toward his position's 30th percentile
in proportion to how little evidence it rests on. Rookies are EXCLUDED
from that (`pipeline.apply_baseline_shrinkage(exclude=is_rookie)`), with
a comment saying "Phase 12 handles their confidence separately."

Phase 12 no longer does. Its RB/WR/QB models failed the holdout and were
cut, so those rookies now sit at a raw cohort average with nothing
pulling them down, while every veteran around them is discounted. That
is Phase 7's complaint #3 -- "rookies valued too high" -- returning
through a different door, and it shows up on the live board as the four
largest disagreements against the market all being rookies.

The Phase 12 plan named the remedy in advance: "if coefficients are
unstable, fall back to the shrinkage haircut and say so explicitly."
This is that haircut, sized by held-out error.

WHAT IS SWEPT
-------------
One parameter, applied to every rookie:

    projection = lambda x cohort_baseline + (1 - lambda) x anchor

`lambda = 1.0` is today's behaviour -- trust the cohort average
completely. `lambda = 0.0` ignores the cohort entirely and projects
every rookie at his position's anchor. The answer is somewhere between,
and the point is that it is measured on classes the anchor never saw.

WHY A SWEPT WEIGHT RATHER THAN AN "EFFECTIVE GAMES" NUMBER
----------------------------------------------------------
The veteran formula is `confidence = games / (games + K)`, which needs a
games count. A rookie has none, so any games number would be invented
and then laundered through a formula into looking principled. Sweeping
the weight directly is the same model with one fewer fiction in it.

THE HONEST REASON A HAIRCUT IS EXPECTED TO HELP: a cohort mean is a weak
predictor of an individual. CP1 measured the spread of rookie outcomes
around their own cohort baseline at 3.0-4.7 PPG standard deviation. A
number that noisy should not be trusted at face value against veterans
whose baselines rest on 30+ games.

LEAVE-ONE-CLASS-OUT, AND WHY THE ANCHOR MOVES WITH IT
-----------------------------------------------------
For each held-out class the anchor is recomputed from the OTHER classes
only. An anchor that has seen the test class is a smaller version of the
same leak the cohort baseline already guards against, and it would
flatter every lambda below 1.0 -- which is precisely the direction this
sweep is looking for a result in.

USAGE
-----
    python -m src.rookie_shrinkage_sweep

Prints RMSE by lambda and names a winner. Nothing is applied: put the
result into features.ROOKIE_SHRINKAGE_LAMBDA, then rebuild.
"""

import argparse

import numpy as np
import polars as pl

from src.fit_rookie_weights import BACKTEST_PATH, load_rookie_backtest

# The anchor rookies are pulled toward. Same quantile the veteran
# shrinkage uses, for the same reason: the 30th percentile of the
# position is roughly "a startable but unexciting player," which is the
# right prior for someone with no NFL record at all.
ANCHOR_QUANTILE = 0.30

LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# A lambda has to beat 1.0 (no haircut) by more than this to be worth
# the change. Same churn logic as the competition bake-off.
MEANINGFUL_GAIN = 0.02


def sweep(df):
    """
    For each held-out class, rebuild the anchor from the other classes,
    apply each lambda to that class's cohort baselines, and score against
    what those rookies actually did.
    """
    seasons = sorted(df.select("season").unique().to_series().to_list())
    per_lambda = {lam: [] for lam in LAMBDAS}

    for held_out in seasons:
        train = df.filter(pl.col("season") != held_out)
        test = df.filter(pl.col("season") == held_out)
        if train.height == 0 or test.height == 0:
            continue

        # Anchor per position, from training classes only.
        anchors = dict(
            train.group_by("position")
            .agg(pl.col("actual_ppg").quantile(ANCHOR_QUANTILE).alias("anchor"))
            .iter_rows()
        )

        scored = test.with_columns(
            pl.col("position").replace_strict(
                anchors, default=None, return_dtype=pl.Float64
            ).alias("anchor")
        ).drop_nulls(subset=["cohort_baseline_ppg", "actual_ppg", "anchor"])
        if scored.height == 0:
            continue

        baseline = scored.select("cohort_baseline_ppg").to_numpy().ravel()
        anchor = scored.select("anchor").to_numpy().ravel()
        actual = scored.select("actual_ppg").to_numpy().ravel()

        for lam in LAMBDAS:
            prediction = lam * baseline + (1 - lam) * anchor
            per_lambda[lam].append(
                float(np.sqrt(np.mean((actual - prediction) ** 2)))
            )

    return {lam: sum(v) / len(v) for lam, v in per_lambda.items() if v}


def relative_calibration(rookie_df):
    """
    THE QUESTION THE SWEEP ABOVE CANNOT ANSWER.

    The sweep scores rookie projections against rookie outcomes, and the
    cohort baseline IS the conditional mean of that exact population.
    Shrinking a conditional mean toward anything else must raise its own
    RMSE unless it buys more variance reduction than it costs in bias --
    and with 40-100 players per cohort cell there is little variance left
    to buy. That sweep could only ever return lambda=1.0. It was the
    wrong instrument, honestly built and pointed at the wrong target.

    The concern was never "is the cohort mean well-calibrated for rookies
    who play." It is "do rookies rank too high AGAINST VETERANS," which
    is a comparison BETWEEN two populations and needs a statistic that
    spans both.

    That statistic is the mean residual. If rookies land systematically
    further below their projection than veterans land below theirs, the
    gap is the haircut -- in PPG, directly, with no parameter to sweep.
    A gap near zero means the two populations are on the same scale and
    the board's rookie rankings are defensible however surprising they
    look.

    Both sides are measured on the SHRUNK/adjusted number the board
    actually shows, because that is the projection a drafter acts on.
    """
    from src import fit_weights as veteran  # noqa: PLC0415

    if not veteran.BACKTEST_PATH.exists():
        return None

    vets = veteran.load_backtest()

    # Veteran residual: what the board would have projected (shrunk
    # baseline, before the situational adjustment, which averages to zero
    # by construction) against what happened.
    vet_residual = (
        vets.select(
            (pl.col("actual_ppg") - pl.col("baseline_ppg_shrunk")).alias("residual")
        ).to_numpy().ravel()
    )

    rookie_residual = (
        rookie_df.drop_nulls(subset=["actual_ppg", "cohort_baseline_ppg"])
        .select(
            (pl.col("actual_ppg") - pl.col("cohort_baseline_ppg")).alias("residual")
        ).to_numpy().ravel()
    )

    gap = float(np.mean(rookie_residual) - np.mean(vet_residual))
    # Standard error of the difference in means. The first version of
    # this compared `gap` against a constant borrowed from the
    # competition bake-off's RMSE churn band -- a number chosen for a
    # different statistic in different units -- and duly declared a
    # 0.057 PPG gap meaningful. With residual SDs of 3-5 PPG this SE
    # lands near 0.2, so that gap was a quarter of one standard error.
    standard_error = float(np.sqrt(
        np.var(vet_residual, ddof=1) / len(vet_residual)
        + np.var(rookie_residual, ddof=1) / len(rookie_residual)
    ))
    return {
        "veteran_mean": float(np.mean(vet_residual)),
        "veteran_n": int(len(vet_residual)),
        "rookie_mean": float(np.mean(rookie_residual)),
        "rookie_n": int(len(rookie_residual)),
        "gap": gap,
        "standard_error": standard_error,
        "z": gap / standard_error if standard_error else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Rookie shrinkage sweep.")
    parser.parse_args()

    if not BACKTEST_PATH.exists():
        raise SystemExit(
            f"\n{BACKTEST_PATH.name} missing. Run:  python -m src.rookie_backtest\n"
        )

    df = load_rookie_backtest()
    print(f"Rookie rows: {df.height} across "
          f"{df['season'].n_unique()} classes\n")

    results = sweep(df)
    if not results:
        raise SystemExit("No usable folds.")

    best = min(results, key=results.get)
    no_haircut = results.get(1.0)

    print(f"   {'lambda':>8}{'held-out RMSE':>16}{'vs no haircut':>16}")
    for lam in LAMBDAS:
        if lam not in results:
            continue
        gain = no_haircut - results[lam] if no_haircut is not None else 0.0
        mark = "   <-- best" if lam == best else ""
        note = "   (today's behaviour)" if lam == 1.0 else ""
        print(f"   {lam:>8.1f}{results[lam]:>16.4f}{gain:>+16.4f}{mark}{note}")

    gain = (no_haircut - results[best]) if no_haircut is not None else 0.0
    print()
    if best == 1.0:
        print("   No haircut wins. Rookies should keep their raw cohort baseline,")
        print("   and the Phase 7 concern about rookies ranking too high is about")
        print("   the BASELINE being wrong, not about it being over-trusted.")
    elif gain > MEANINGFUL_GAIN:
        print(f"   lambda = {best} beats no haircut by {gain:.4f} RMSE.")
        print(f"   Rookie projections should sit {(1 - best) * 100:.0f}% of the way")
        print(f"   from their cohort average toward the position anchor.")
        print(f"\n   Set features.ROOKIE_SHRINKAGE_LAMBDA = {best}, rebuild, re-gate.")
    else:
        print(f"   lambda = {best} leads by only {gain:+.4f}, inside the "
              f"{MEANINGFUL_GAIN} churn band. Keep 1.0.")

    print("\n   NOTE: this sizes the haircut. It does NOT refit the rookie TE")
    print("   model, whose coefficient was fitted against UNSHRUNK deltas --")
    print("   applying it on top of a shrunk baseline without refitting is the")
    print("   Phase 11 CP5 bug. Rebuild the rookie backtest with the same lambda")
    print("   before refitting.")

    calibration = relative_calibration(df)
    if calibration is None:
        return

    print(f"\n\n{'=' * 66}")
    print("RELATIVE CALIBRATION  --  are rookies over-projected vs veterans?")
    print(f"{'=' * 66}")
    print("   The sweep above compares rookies to rookies, where the cohort mean")
    print("   is the conditional mean by construction and can only win. This is")
    print("   the comparison that actually bears on the board: mean residual")
    print("   (actual minus projected) for each population.\n")
    print(f"   {'population':<12}{'n':>8}{'mean residual':>16}")
    print(f"   {'veterans':<12}{calibration['veteran_n']:>8}"
          f"{calibration['veteran_mean']:>+16.3f}")
    print(f"   {'rookies':<12}{calibration['rookie_n']:>8}"
          f"{calibration['rookie_mean']:>+16.3f}")
    print(f"\n   gap: {calibration['gap']:+.3f} PPG   "
          f"(standard error {calibration['standard_error']:.3f}, "
          f"z = {calibration['z']:+.2f})")

    if abs(calibration["z"]) < 2:
        print("\n   INDISTINGUISHABLE FROM ZERO. No haircut is warranted by this")
        print("   measurement.")
    elif calibration["gap"] < 0:
        print(f"\n   Rookies land {abs(calibration['gap']):.2f} PPG further below their")
        print(f"   projection than veterans do, and the gap clears its own standard")
        print(f"   error. That is a haircut in points, with no parameter to sweep.")
    else:
        print("\n   Rookies BEAT their projection relative to veterans.")

    print("\n   READ THIS BEFORE ACTING ON THE NUMBER ABOVE.")
    print("   Both residuals are pinned near zero BY CONSTRUCTION. OLS with an")
    print("   intercept forces the veteran residual to equal mean(delta), which")
    print("   the situational adjustment then absorbs; the rookie cohort baseline")
    print("   is the mean of its own cell. Comparing them mostly compares two")
    print("   numbers their own fitting procedures already flattened, so a small")
    print("   gap here is close to uninformative rather than reassuring.")
    print()
    print("   THE TEST THAT WOULD ACTUALLY SETTLE IT, not yet built:")
    print("   both baselines are estimated on players who cleared MIN_GAMES=8,")
    print("   so each is 'expected PPG GIVEN you earn a role.' The board then")
    print("   applies them to everyone. That is far more forgiving to rookies,")
    print("   because a much larger share of them never earn the role at all --")
    print("   rookie_backtest kept 415 of 600 drafted rookies who took a snap,")
    print("   and rookies with no snap never entered that 600.")
    print()
    print("   So the real question is not 'is the cohort mean biased' but")
    print("   'what is P(earns a role) for a rookie versus a veteran, and does")
    print("   the board price that difference.' It does not currently. That is")
    print("   a playing-time model, adjacent to the expected_games machinery")
    print("   Phase 11 CP8 already built for PUP/NFI, and it is real work rather")
    print("   than a constant -- do not attempt it the week of a draft.")


if __name__ == "__main__":
    main()
