"""
Phase 15d (scratch, not shipped).

"Games played acts as a confidence level, right? Is this the best way to
do this or should we explore other ways."

Today's answer is features.apply_baseline_shrinkage:

    confidence = games / (games + 2)
    shrunk     = confidence * baseline + (1 - confidence) * anchor

Games is a defensible unit and K=2 is deliberately gentle. But games
treats every appearance as one unit of evidence, and they are not. Eight
games as a team's WR1 is roughly 70 targets; eight games as a WR4 is
roughly 8. The first is a real estimate, the second is a rumour, and the
current formula calls them equally trustworthy.

Four estimators, scored the only way that settles it -- held-out RMSE on
`actual_ppg`, the number the board multiplies by expected games:

  1. NONE        raw baseline, no shrinkage             (K=0 incumbent)
  2. GAMES       games / (games + K)                    (shipped)
  3. OPPORTUNITY opportunities / (opportunities + K)     touches, not games
  4. GLOBAL      one fitted w per position               no per-player term
  5. EB          empirical Bayes: w = t2 / (t2 + s2/n)   variance-based

EB is the estimator the shrinkage formula is an approximation OF. It
sets each player's weight from two measured quantities -- how much
players at his position truly differ (between-player variance) and how
noisy one game is (within-player variance) -- instead of a hand-picked
K. If K=2 is right, EB should land near it and win nothing. If it wins,
K was a guess standing in for a measurement.

Everything is fitted inside the training fold; K is swept on training
error only.

    python scratch_confidence.py
"""
import numpy as np
import polars as pl

from src import fit_weights as veteran

FOLDS = [2023, 2024, 2025]
POSITIONS = ["RB", "WR", "TE", "QB"]
K_GRID = [0, 0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24]


def rmse(x, y):
    return float(np.sqrt(((x - y) ** 2).mean()))


def opportunities(frame):
    """Touches behind the baseline: carries + targets over the baseline
    window. For QB this is designed runs only, which is why QB is
    reported but not expected to benefit."""
    per_game = (frame["carries_per_game"].fill_null(0).to_numpy()
                + frame["targets_per_game"].fill_null(0).to_numpy())
    return per_game * frame["baseline_games"].to_numpy()


def shrink(baseline, weight, anchor):
    return weight * baseline + (1 - weight) * anchor


def main():
    df = veteran.load_backtest()
    print(f"rows after MIN_GAMES={veteran.MIN_GAMES}: {df.height}")
    print(f"shipped estimator: games/(games+{veteran_k()})\n")

    for position in POSITIONS:
        names = ["NONE", "GAMES", "OPPORTUNITY", "GLOBAL", "EB"]
        results = {name: [] for name in names}
        thin_results = {name: [] for name in names}
        chosen = {"GAMES": [], "OPPORTUNITY": [], "GLOBAL": [], "EB": []}
        thin_n = []

        for season in FOLDS:
            tr = df.filter((pl.col("season") != season) &
                           (pl.col("position") == position))
            te = df.filter((pl.col("season") == season) &
                           (pl.col("position") == position))
            if te.height == 0 or tr.height < 50:
                continue

            anchor = float(tr["actual_ppg"].to_numpy().mean())
            b_tr = tr["baseline_ppg"].to_numpy()
            a_tr = tr["actual_ppg"].to_numpy()
            b_te = te["baseline_ppg"].to_numpy()
            a_te = te["actual_ppg"].to_numpy()
            g_tr = tr["baseline_games"].to_numpy().astype(float)
            g_te = te["baseline_games"].to_numpy().astype(float)
            o_tr, o_te = opportunities(tr), opportunities(te)

            # The CP5 decision rule judged shrinkage on the LOW-CONFIDENCE
            # subgroup, not the full pool, and it was right to: 92% of
            # rows rest on 9+ baseline games, so a full-pool average
            # drowns exactly the players shrinkage exists for.
            thin = g_te <= 8

            results["NONE"].append(rmse(b_te, a_te))
            thin_results["NONE"].append(rmse(b_te[thin], a_te[thin])
                                        if thin.sum() else np.nan)

            # --- K swept on TRAINING error only, for each unit
            for name, unit_tr, unit_te in [("GAMES", g_tr, g_te),
                                           ("OPPORTUNITY", o_tr, o_te)]:
                grid = K_GRID if name == "GAMES" else [k * 20 for k in K_GRID]
                best_k, best_err = None, np.inf
                for k in grid:
                    w = unit_tr / (unit_tr + k) if k > 0 else np.ones_like(unit_tr)
                    err = rmse(shrink(b_tr, w, anchor), a_tr)
                    if err < best_err:
                        best_k, best_err = k, err
                w_te = (unit_te / (unit_te + best_k) if best_k > 0
                        else np.ones_like(unit_te))
                pred = shrink(b_te, w_te, anchor)
                results[name].append(rmse(pred, a_te))
                thin_results[name].append(rmse(pred[thin], a_te[thin])
                                          if thin.sum() else np.nan)
                chosen[name].append(best_k)

            # --- one global w, no per-player confidence at all
            d = b_tr - anchor
            w_global = float(((d * (a_tr - anchor)).sum() / (d * d).sum())
                             if (d * d).sum() > 0 else 1.0)
            pred_g = shrink(b_te, w_global, anchor)
            results["GLOBAL"].append(rmse(pred_g, a_te))
            thin_results["GLOBAL"].append(rmse(pred_g[thin], a_te[thin])
                                          if thin.sum() else np.nan)
            chosen["GLOBAL"].append(w_global)
            thin_n.append(int(thin.sum()))

            # --- empirical Bayes
            # s2 = within-player variance of a single game's fantasy
            # points, approximated from the spread of baselines built on
            # few games vs many. t2 = between-player variance of true
            # skill = total variance minus the average sampling variance.
            total_var = float(np.var(b_tr, ddof=1))
            # average sampling variance implied by the games each
            # baseline rests on, solved from total = t2 + mean(s2/n)
            inv_n = float(np.mean(1.0 / np.maximum(g_tr, 1)))
            # s2 estimated by regressing squared deviation on 1/n
            dev2 = (b_tr - b_tr.mean()) ** 2
            X = np.column_stack([np.ones(len(g_tr)), 1.0 / np.maximum(g_tr, 1)])
            coef, *_ = np.linalg.lstsq(X, dev2, rcond=None)
            t2 = max(coef[0], 1e-6)
            s2 = max(coef[1], 1e-6)
            w_eb_te = t2 / (t2 + s2 / np.maximum(g_te, 1))
            pred_eb = shrink(b_te, w_eb_te, anchor)
            results["EB"].append(rmse(pred_eb, a_te))
            thin_results["EB"].append(rmse(pred_eb[thin], a_te[thin])
                                      if thin.sum() else np.nan)
            chosen["EB"].append(s2 / t2)  # this is EB's implied K

        if not results["NONE"]:
            continue
        print(f"=== {position} ===")
        base = float(np.mean(results["NONE"]))
        for name in ["NONE", "GAMES", "OPPORTUNITY", "GLOBAL", "EB"]:
            if not results[name]:
                continue
            m = float(np.mean(results[name]))
            extra = ""
            if name in chosen and chosen[name]:
                if name == "GLOBAL":
                    extra = f"  (fitted w={np.mean(chosen[name]):.3f})"
                elif name == "EB":
                    extra = f"  (implied K={np.mean(chosen[name]):.1f} games)"
                else:
                    extra = f"  (K chosen in-fold={np.mean(chosen[name]):.1f})"
            print(f"  {name:<12s} RMSE={m:.4f}  vs no-shrinkage {base - m:+.4f}{extra}")

        tb = float(np.nanmean(thin_results["NONE"]))
        print(f"  -- LOW-CONFIDENCE subgroup only (baseline_games <= 8, "
              f"mean n={np.mean(thin_n):.0f}/fold) --")
        for name in names:
            vals = [v for v in thin_results[name] if not np.isnan(v)]
            if not vals:
                continue
            m = float(np.mean(vals))
            print(f"  {name:<12s} RMSE={m:.4f}  vs no-shrinkage {tb - m:+.4f}")
        print()


def veteran_k():
    from src.features import SHRINKAGE_K
    return SHRINKAGE_K


if __name__ == "__main__":
    main()
