"""
Phase 15b (scratch, not shipped).

"Make sure we've tested everything we have tested, on QBs."

Phase 10 fitted QB `age` and Phase 13 killed it on the holdout. Nothing
else has ever been run at quarterback on the WIDER window that now
exists (2017-2025, 582 QB-seasons in backtest_features.csv), and the
in-sample-vs-holdout gap that killed `age` means the old "nothing works
for QB" verdicts are not evidence either -- they were p-values.

So this is the full battery: every column in the backtest table that
could plausibly be a QB feature, run one at a time through the SAME
holdout instrument the shipped model is gated with
(`holdout.run_holdout`), on the same three folds.

Scoring, unchanged from the gate: MODEL minus LEVEL RMSE, averaged over
folds. LEVEL is the training-fold mean delta -- a constant. A feature
that cannot beat a constant out of sample does not predict, whatever its
p-value says.

    python scratch_qb_battery.py
"""
import numpy as np
import polars as pl

from src import fit_weights as veteran
from src import holdout

FOLDS = [2023, 2024, 2025]
ALPHA = veteran.ALPHA

CANDIDATES = [
    # team situation
    "pass_att_pg", "rush_att_pg", "qb_changed", "coach_changed",
    "returning_oline_starters", "team_changed",
    # player role
    "workload_share", "pos_rank", "position_competition_ppg",
    "position_competition_top1", "pos_rank_change",
    # QB-specific, never tested: designed rushing is the single biggest
    # separator in fantasy QB scoring and the model has never looked at it
    "carries_per_game",
    # ageing / experience
    "age", "age_squared", "experience",
    # trend
    "usage_trend_share", "usage_trend_volume", "usage_trend_relative",
    # evidence / confidence
    "baseline_games", "baseline_seasons", "baseline_confidence",
    # mean reversion
    "baseline_ppg",
    # injury
    "recent_major_injury",
]


def prepare(df):
    df = df.with_columns([
        pl.col(c).cast(pl.Float64) for c in veteran.BOOL_COLUMNS if c in df.columns
    ])
    if "age_squared" not in df.columns and "age" in df.columns:
        df = df.with_columns((pl.col("age") ** 2).alias("age_squared"))
    return df


def score_spec(df, features, label):
    gains, n_tests, survivors = [], [], set()
    for season in FOLDS:
        try:
            r = holdout.run_holdout(df, "QB", features, veteran, season, ALPHA)
        except (ValueError, KeyError, np.linalg.LinAlgError):
            return None
        if r is None:
            continue
        s = r["scores"]
        gains.append(s["LEVEL"]["rmse"] - s["MODEL"]["rmse"])
        n_tests.append(r["n_test"])
        survivors.update(r["survivors_in_fold"])
    if not gains:
        return None
    return {
        "label": label,
        "mean_gain": float(np.mean(gains)),
        "folds_positive": int(sum(g > 0 for g in gains)),
        "n_folds": len(gains),
        "per_fold": [round(g, 4) for g in gains],
        "survived_selection": sorted(survivors),
        "mean_n_test": float(np.mean(n_tests)),
    }


def main():
    df = prepare(veteran.load_backtest())
    qb = df.filter(pl.col("position") == "QB")
    print(f"QB rows after MIN_GAMES={veteran.MIN_GAMES} filter: {qb.height}")
    print(f"seasons: {sorted(qb['season'].unique().to_list())}")

    print("\n=== single-feature battery, QB, holdout folds 2023/24/25 ===")
    print("positive mean gain = beats a constant out of sample\n")
    rows = []
    for feat in CANDIDATES:
        if feat not in df.columns:
            print(f"  {feat:<28s} -- not in backtest table, skipped")
            continue
        # A feature that is null for most quarterbacks is not a QB
        # feature. workload_share is the case: it is a share of team
        # carries/targets, which is not a thing a passer has.
        coverage = 1 - qb[feat].null_count() / qb.height
        if coverage < 0.5:
            print(f"  {feat:<28s} -- only {coverage:.0%} non-null at QB, not a QB feature")
            continue
        res = score_spec(df, [feat], feat)
        if res is None:
            print(f"  {feat:<28s} -- no usable fold")
            continue
        rows.append(res)

    rows.sort(key=lambda r: r["mean_gain"], reverse=True)
    for r in rows:
        flag = "PASS" if (r["mean_gain"] > 0 and r["folds_positive"] >= 2) else "    "
        kept = "kept" if r["survived_selection"] else "CUT by alpha in every fold"
        print(f"  {flag} {r['label']:<28s} mean gain {r['mean_gain']:+.4f}  "
              f"folds+ {r['folds_positive']}/{r['n_folds']}  {r['per_fold']}  {kept}")

    winners = [r["label"] for r in rows
               if r["mean_gain"] > 0 and r["folds_positive"] >= 2
               and r["survived_selection"]]
    print(f"\nfeatures that beat a constant in 2+ folds: {winners or 'NONE'}")

    if winners:
        print("\n=== combined spec ===")
        combo = score_spec(df, winners, "combined")
        if combo:
            print(f"  combined {winners}")
            print(f"  mean gain {combo['mean_gain']:+.4f}  "
                  f"folds+ {combo['folds_positive']}/{combo['n_folds']}  "
                  f"{combo['per_fold']}")
            print(f"  survived selection: {combo['survived_selection']}")

    # ------------------------------------------------ MIN_GAMES sensitivity
    # fit_weights documents that MIN_GAMES=8 bites harder at QB than
    # anywhere else, because quarterback is one-per-team. If any of the
    # results above are an artifact of that filter, they should move when
    # the filter moves.
    print("\n=== MIN_GAMES sensitivity for the best single feature ===")
    if rows:
        best = rows[0]["label"]
        raw = prepare(pl.read_csv(veteran.BACKTEST_PATH))
        for mg in [4, 6, 8, 10, 12]:
            sub = raw.filter(pl.col("actual_games_played") >= mg)
            res = score_spec(sub, [best], f"{best}@{mg}")
            n = sub.filter(pl.col("position") == "QB").height
            if res:
                print(f"  MIN_GAMES={mg:<3d} n_QB={n:<4d} {best}: "
                      f"mean gain {res['mean_gain']:+.4f} "
                      f"folds+ {res['folds_positive']}/{res['n_folds']}")


if __name__ == "__main__":
    main()
