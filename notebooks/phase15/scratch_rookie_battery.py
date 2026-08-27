"""
Phase 15c (scratch, not shipped).

The README's second known limitation, stated plainly:

    "Rookies who share a position/round bucket are the same player to
     this model. Two first-round running backs get the same cohort
     baseline before situational adjustment."

Only TE ships a rookie weight (`age`, n=69). RB, WR and QB rookies take
a zero adjustment, so the entire difference between the first pick of
the draft and the last pick of round 1 is nothing.

The obvious candidate is `pick` itself -- the exact draft slot, which the
round bucket throws away. It is in the rookie backtest table and has
never been run through the holdout at RB or WR. This runs it, and every
other pre-snap-knowable column, through the same instrument
(`holdout.run_holdout` with the rookie spec module), on the same folds.

Scoring: MODEL minus LEVEL RMSE, averaged over folds, where LEVEL is the
training-fold mean delta. A feature that cannot beat a constant on a
season it has not seen does not predict.

    python scratch_rookie_battery.py
"""
import numpy as np
import polars as pl

from src import fit_rookie_weights as rookie
from src import holdout

FOLDS = [2023, 2024, 2025]
ALPHA = rookie.ALPHA
POSITIONS = ["RB", "WR", "QB", "TE"]

CANDIDATES = [
    "pick",                       # exact slot -- the differentiator the bucket discards
    "round",
    "age",
    "pos_rank",                   # rookie's own depth-chart slot
    "position_competition_ppg",   # who is already there
    "pass_att_pg", "rush_att_pg",  # what the offence does
    "returning_oline_starters",
    "baseline_cohort_n",          # how thin his own cohort baseline is
]


def prepare(df):
    for c in ["depth_chart_missing", "baseline_low_confidence"]:
        if c in df.columns:
            df = df.with_columns(pl.col(c).cast(pl.Float64))
    return df


def score(df, position, features):
    gains, survivors, n_test = [], set(), []
    for season in FOLDS:
        try:
            r = holdout.run_holdout(df, position, features, rookie, season, ALPHA)
        except (ValueError, KeyError, np.linalg.LinAlgError):
            return None
        if r is None:
            continue
        s = r["scores"]
        gains.append(s["LEVEL"]["rmse"] - s["MODEL"]["rmse"])
        survivors.update(r["survivors_in_fold"])
        n_test.append(r["n_test"])
    if not gains:
        return None
    return {
        "mean_gain": float(np.mean(gains)),
        "folds_positive": int(sum(g > 0 for g in gains)),
        "n_folds": len(gains),
        "per_fold": [round(g, 3) for g in gains],
        "survivors": sorted(survivors),
        "mean_n_test": float(np.mean(n_test)),
    }


def main():
    df = prepare(rookie.load_rookie_backtest())
    print(f"rookie rows: {df.height}")
    print(df.group_by("position").len().sort("position"))
    print("\nNote the sample sizes. At QB there are 35 rookie-seasons in "
          "total\nand roughly 4 per held-out fold -- report it, do not "
          "believe it.\n")

    for position in POSITIONS:
        n = df.filter(pl.col("position") == position).height
        print(f"=== {position} (n={n}) ===")
        rows = []
        for feat in CANDIDATES:
            if feat not in df.columns:
                continue
            sub = df.filter(pl.col("position") == position)
            if sub[feat].null_count() / max(sub.height, 1) > 0.5:
                print(f"  {feat:<26s} -- mostly null at {position}, skipped")
                continue
            res = score(df, position, [feat])
            if res is None:
                print(f"  {feat:<26s} -- no usable fold")
                continue
            rows.append((feat, res))

        rows.sort(key=lambda r: r[1]["mean_gain"], reverse=True)
        for feat, r in rows:
            flag = "PASS" if (r["mean_gain"] > 0 and r["folds_positive"] >= 2
                              and r["survivors"]) else "    "
            kept = "kept" if r["survivors"] else "CUT by alpha every fold"
            print(f"  {flag} {feat:<26s} gain {r['mean_gain']:+.4f}  "
                  f"folds+ {r['folds_positive']}/{r['n_folds']}  "
                  f"{r['per_fold']}  n_test~{r['mean_n_test']:.0f}  {kept}")

        winners = [f for f, r in rows
                   if r["mean_gain"] > 0 and r["folds_positive"] >= 2 and r["survivors"]]
        if len(winners) > 1:
            combo = score(df, position, winners)
            if combo:
                print(f"  COMBINED {winners}: gain {combo['mean_gain']:+.4f}  "
                      f"folds+ {combo['folds_positive']}/{combo['n_folds']}  "
                      f"survivors {combo['survivors']}")
        elif not winners:
            print("  -> nothing beats a constant out of sample")
        print()


if __name__ == "__main__":
    main()
