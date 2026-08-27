"""
Phase 15b, step 2 (scratch, not shipped).

The battery says the single best QB predictor of `delta` is
`baseline_ppg` itself: +0.58 RMSE over a constant, 3 folds of 3, and the
gain GROWS as MIN_GAMES rises (+1.21 at 12+ games), so it is not the
backup-QB spot-duty artifact Phase 11 CP1 found.

A negative coefficient on your own baseline is mean reversion, and there
are two reasons it can appear, only one of which is worth shipping:

  REAL      quarterbacks genuinely regress toward the position mean --
            last year's 22 PPG is not this year's expectation.
  ARTIFACT  `delta = actual - baseline`, so any noise in `baseline`
            enters `delta` with the opposite sign. Regressing delta on
            baseline finds that mechanically, even if nothing regresses.

The test that separates them is to stop scoring `delta` and score the
thing the board actually needs, `actual_ppg`, out of sample:

    P1  predict baseline_ppg                  (what the board does now)
    P2  predict w*baseline + (1-w)*anchor     (shrunk, w fitted on train)

The artifact cannot help P2. If noise in the baseline were the whole
story, pulling every quarterback toward an anchor would not reduce error
on the held-out season -- it would just add bias. If P2 beats P1 out of
sample, the reversion is real and QB's exclusion from shrinkage is
costing the board accuracy.

This matters beyond QB because features.SHRINKAGE_EXCLUDED_POSITIONS
excluded quarterbacks for a specific, correct reason -- the 30th
percentile of 16+ game QBs is a STARTER's number (12.52), so shrinking
toward it dragged clipboard-holders upward. That argument is about the
CHOICE OF ANCHOR, not about whether QBs revert. This script tests the
second question, which was never asked.

    python scratch_qb_reversion.py
"""
import numpy as np
import polars as pl

from src import fit_weights as veteran

FOLDS = [2023, 2024, 2025]
POSITIONS = ["QB", "RB", "WR", "TE"]


def anchor_candidates(train, position):
    """Anchors that do not have the 'played 16 games = was the starter'
    problem features.py documents for QB."""
    t = train.filter(pl.col("position") == position)
    b = t["baseline_ppg"].to_numpy()
    a = t["actual_ppg"].to_numpy()
    return {
        "actual_mean": float(a.mean()),
        "baseline_mean": float(b.mean()),
        "baseline_p30": float(np.percentile(b, 30)),
        "actual_p30": float(np.percentile(a, 30)),
    }


def fit_weight(train, position, anchor):
    """Least-squares w for actual ~= w*baseline + (1-w)*anchor."""
    t = train.filter(pl.col("position") == position)
    b = t["baseline_ppg"].to_numpy() - anchor
    a = t["actual_ppg"].to_numpy() - anchor
    denom = float((b * b).sum())
    return float((b * a).sum() / denom) if denom > 0 else 1.0


def rmse(x, y):
    return float(np.sqrt(((x - y) ** 2).mean()))


def main():
    df = veteran.load_backtest()
    print(f"rows after MIN_GAMES={veteran.MIN_GAMES}: {df.height}\n")

    for position in POSITIONS:
        print(f"=== {position} ===")
        p1_all, p2_all, w_all, n_all = [], [], [], []
        best_anchor_name = None
        for season in FOLDS:
            train = df.filter(pl.col("season") != season)
            test = df.filter((pl.col("season") == season) &
                             (pl.col("position") == position))
            if test.height == 0:
                continue
            anchors = anchor_candidates(train, position)

            # Anchor chosen INSIDE the training fold, by training-fold
            # error only. Picking it on the test season would be exactly
            # the leak this whole file exists to avoid.
            scored = {}
            for name, val in anchors.items():
                w = fit_weight(train, position, val)
                tr = train.filter(pl.col("position") == position)
                pred = w * tr["baseline_ppg"].to_numpy() + (1 - w) * val
                scored[name] = (rmse(pred, tr["actual_ppg"].to_numpy()), w, val)
            best_anchor_name = min(scored, key=lambda k: scored[k][0])
            _, w, anchor_val = scored[best_anchor_name]

            b = test["baseline_ppg"].to_numpy()
            a = test["actual_ppg"].to_numpy()
            p1_all.append(rmse(b, a))
            p2_all.append(rmse(w * b + (1 - w) * anchor_val, a))
            w_all.append(w)
            n_all.append(test.height)

        if not p1_all:
            continue
        p1, p2 = float(np.mean(p1_all)), float(np.mean(p2_all))
        print(f"  anchor chosen in-fold : {best_anchor_name}")
        print(f"  fitted w (own weight) : {np.mean(w_all):.3f}  "
              f"(1.0 = no reversion, what the board assumes at QB)")
        print(f"  P1 raw baseline  RMSE : {p1:.4f}   per fold {[round(x,3) for x in p1_all]}")
        print(f"  P2 shrunk        RMSE : {p2:.4f}   per fold {[round(x,3) for x in p2_all]}")
        gain = p1 - p2
        print(f"  gain                  : {gain:+.4f} PPG  "
              f"({'REAL -- reversion is not an artifact' if gain > 0 else 'no'})")
        print(f"  mean n_test           : {np.mean(n_all):.0f}\n")

    # ------------------------------------------------------------------
    # And the sanity check features.py demanded: does the winning anchor
    # drag backup quarterbacks upward the way the 30th-percentile anchor
    # did? Phase 11 CP5 rejected QB shrinkage because 59% of QBs moved UP.
    print("=== does the fitted reversion inflate backups? (2025 fold) ===")
    train = df.filter(pl.col("season") != 2025)
    test = df.filter((pl.col("season") == 2025) & (pl.col("position") == "QB"))
    anchors = anchor_candidates(train, "QB")
    for name, val in anchors.items():
        w = fit_weight(train, "QB", val)
        b = test["baseline_ppg"].to_numpy()
        pred = w * b + (1 - w) * val
        moved_up = float((pred > b).mean())
        print(f"  anchor={name:<15s} value={val:5.2f}  w={w:.3f}  "
              f"moved UP: {moved_up:.0%}  test RMSE={rmse(pred, test['actual_ppg'].to_numpy()):.3f}")


if __name__ == "__main__":
    main()
