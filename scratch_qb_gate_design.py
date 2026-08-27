"""
Phase 15b, step 3 (scratch, not shipped).

The reversion finding is solid. The SHIPPING DESIGN is not, and this
script is about the gap between those two things.

Phase 11 CP5 rejected QB shrinkage for a concrete reason, and it is not
a reason the reversion result answers: applied to the LIVE pool, pulling
every quarterback toward a population anchor dragged clipboard-holders
upward -- Nathan Peterman -0.40 -> 8.21, and 59% of quarterbacks moved
UP. The 2025-fold check in scratch_qb_reversion.py reproduced exactly
that: 61% moved up under the mean anchor.

That is not a flaw in the measurement. It is a statement about SUPPORT.
w = 0.544 was estimated on quarterbacks who played 8+ games. A backup
with one game of history is outside that sample, and features.py already
says why it must be: "Shrinkage assumes a small sample is a noisy
estimate of the same quantity. For a backup QB it is a precise estimate
of a different one."

So the design under test is reversion WITH A SUPPORT GUARD: apply it
only to quarterbacks whose baseline rests on enough games to be the kind
of quarterback the coefficient was estimated from, and leave everyone
else exactly where they are today (zero adjustment). That guard can only
ever be a no-op or an improvement -- it cannot reintroduce the CP5
failure, because the players CP5 was about are never touched.

Three questions decide whether this ships:

  Q1  Is w stable across folds, or is 0.544 an average of noise?
  Q2  Does the gain survive on the guarded subgroup only?
  Q3  On the LIVE pool, does the guard actually keep backups out?

    python scratch_qb_gate_design.py
"""
import numpy as np
import polars as pl

from src import fit_weights as veteran
from src import features

FOLDS = [2023, 2024, 2025]
GUARDS = [0, 8, 12, 16, 20, 24, 32]


def rmse(x, y):
    return float(np.sqrt(((x - y) ** 2).mean()))


def fit_w(train, anchor):
    b = train["baseline_ppg"].to_numpy() - anchor
    a = train["actual_ppg"].to_numpy() - anchor
    d = float((b * b).sum())
    return float((b * a).sum() / d) if d > 0 else 1.0


def main():
    df = veteran.load_backtest().filter(pl.col("position") == "QB")
    print(f"QB rows: {df.height}  seasons {df['season'].min()}-{df['season'].max()}")

    # ---------------------------------------------------------- Q1
    print("\n=== Q1: is w stable across folds? ===")
    ws, anchors = [], []
    for season in FOLDS:
        tr = df.filter(pl.col("season") != season)
        anchor = float(tr["actual_ppg"].mean())
        w = fit_w(tr, anchor)
        ws.append(w)
        anchors.append(anchor)
        print(f"  fold {season}: anchor={anchor:5.2f}  w={w:.3f}")
    print(f"  spread: {min(ws):.3f} to {max(ws):.3f}  "
          f"({'STABLE' if max(ws) - min(ws) < 0.10 else 'MOVES -- do not ship'})")

    # ---------------------------------------------------------- Q2
    print("\n=== Q2: does the gain survive a support guard? ===")
    print("  guard = minimum baseline_games required to be reverted at all")
    print("  players below the guard keep today's behaviour (no change)\n")
    print(f"  {'guard':>6s} {'n reverted':>11s} {'n untouched':>12s} "
          f"{'raw RMSE':>9s} {'guarded':>9s} {'gain':>8s}")
    best = None
    for guard in GUARDS:
        raws, guards_rmse, n_rev, n_un = [], [], [], []
        for season, w, anchor in zip(FOLDS, ws, anchors):
            te = df.filter(pl.col("season") == season)
            b = te["baseline_ppg"].to_numpy()
            a = te["actual_ppg"].to_numpy()
            g = te["baseline_games"].to_numpy().astype(float)
            on = g >= guard
            pred = b.copy()
            pred[on] = w * b[on] + (1 - w) * anchor
            raws.append(rmse(b, a))
            guards_rmse.append(rmse(pred, a))
            n_rev.append(int(on.sum()))
            n_un.append(int((~on).sum()))
        raw_m, g_m = float(np.mean(raws)), float(np.mean(guards_rmse))
        gain = raw_m - g_m
        mark = ""
        if best is None or gain > best[1]:
            best = (guard, gain)
        print(f"  {guard:>6d} {np.mean(n_rev):>11.0f} {np.mean(n_un):>12.0f} "
              f"{raw_m:>9.4f} {g_m:>9.4f} {gain:>+8.4f}{mark}")
    print(f"\n  best guard on held-out data: {best[0]} games ({best[1]:+.4f})")

    # ---------------------------------------------------------- Q3
    print("\n=== Q3: the live pool -- does the guard keep backups out? ===")
    print("  building the 2023-2025 veteran table (no ADP, no board)...")
    live = features.build_veteran_feature_table([2023, 2024, 2025])
    live = features.attach_current_team(live)
    qb = live.filter(pl.col("position") == "QB").filter(
        pl.col("fantasy_points_per_game").is_not_null()
    )
    print(f"  live QB pool: {qb.height} quarterbacks")

    w = float(np.mean(ws))
    anchor = float(np.mean(anchors))
    print(f"  applying w={w:.3f} toward anchor={anchor:.2f}\n")

    for guard in [0, 12, 16, 20, 24]:
        b = qb["fantasy_points_per_game"].to_numpy()
        g = qb["games_played"].to_numpy().astype(float)
        on = g >= guard
        pred = b.copy()
        pred[on] = w * b[on] + (1 - w) * anchor
        moved_up = pred > b + 1e-9
        # The CP5 failure mode, named exactly: a quarterback with almost
        # no history who ends up projected like a starter.
        thin_inflated = int((moved_up & (g < 12)).sum())
        print(f"  guard={guard:>3d}: reverted {on.sum():>3d}/{qb.height}   "
              f"moved UP {moved_up.sum():>3d}   "
              f"thin (<12 gm) inflated: {thin_inflated}")

    # Name the ten biggest movers under the guard that Q2 chose, so the
    # change is a list of players rather than a delta column.
    guard = best[0] if best[0] > 0 else 16
    b = qb["fantasy_points_per_game"].to_numpy()
    g = qb["games_played"].to_numpy().astype(float)
    on = g >= guard
    pred = b.copy()
    pred[on] = w * b[on] + (1 - w) * anchor
    out = qb.with_columns([
        pl.Series("before", b), pl.Series("after", pred),
        pl.Series("delta", pred - b), pl.Series("reverted", on),
    ]).sort("before", descending=True).head(14)
    print(f"\n=== top 14 live QBs under guard={guard} ===")
    print(out.select(["player_name", "games_played", "before", "after", "delta",
                      "reverted"]))


if __name__ == "__main__":
    main()
