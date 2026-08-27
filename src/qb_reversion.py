"""
Phase 15b. Quarterback mean reversion, and its own gate.

    python -m src.qb_reversion              # fit + gate + write
    python -m src.qb_reversion --gate       # gate only
    python -m src.qb_reversion --no-write   # print, write nothing

WHY THIS EXISTS
---------------
The README's first stated limitation:

    "Quarterbacks have no situational features and no baseline
     shrinkage. A QB's projection is his own trailing average and
     nothing else."

Phase 15's feature battery ran all 23 plausible QB features through
`holdout.run_holdout` on the full 2017-2025 window. Twenty-one were cut
by alpha inside the training fold, confirming ten phases of "nothing
works for QB." The one that survived was not a situational feature at
all: it was the quarterback's OWN baseline, with a negative coefficient.
Mean reversion.

    baseline_ppg    mean RMSE gain +0.5805, 3 folds of 3
    baseline_games  mean RMSE gain +0.0817, 2 folds of 3
    everything else cut, including age (which Phase 13 already killed),
    experience, team tendency, playcaller change, O-line continuity,
    position competition, and QB rushing volume.

RULING OUT THE ARTIFACT -- THE REASON THIS ISN'T SHIPPED ON THAT ALONE
----------------------------------------------------------------------
`delta = actual_ppg - baseline`, so noise in the baseline enters delta
with the opposite sign automatically. A negative coefficient on your own
baseline is exactly what that arithmetic produces whether or not
anything reverts, which makes the battery result unshippable by itself.

So the test switched targets. Instead of scoring `delta`, score
`actual_ppg` -- the quantity the board multiplies by expected games --
comparing the raw baseline against a reverted one on a season the fit
has never seen. The artifact cannot help there: pulling every
quarterback toward an anchor can only add bias unless the reversion is
real.

    held-out RMSE on actual_ppg, pooled over three folds
      QB   raw 4.87   reverted 4.27   gain +0.60
      WR   raw 3.61   reverted 3.46   gain +0.15
      TE   raw 2.60   reverted 2.50   gain +0.10
      RB   raw 3.91   reverted 3.83   gain +0.08

Quarterbacks revert nearly twice as hard as any skill position, and QB
is the one position excluded from shrinkage entirely.

WHY THIS IS NOT A REVERSAL OF PHASE 11 CP5
------------------------------------------
`features.SHRINKAGE_EXCLUDED_POSITIONS` excludes QB, and that decision
was correct for the reason it gives: the 30th percentile of quarterbacks
with 16+ games is 12.52 PPG, which is a STARTER's number, because
quarterback is one-per-team. Shrinking toward it dragged clipboard
holders upward -- Nathan Peterman -0.40 -> 8.21, 59% of quarterbacks
moved UP.

That is an argument about the ANCHOR and about the POPULATION. It is not
an argument that quarterbacks do not revert, and nobody had asked.

THE SUPPORT GUARD, WHICH IS THE WHOLE SHIPPING DESIGN
-----------------------------------------------------
w was estimated from quarterbacks with a real starting history. A backup
with one game of history is not a noisy estimate of that same quantity
-- `features.py` says it best, "for a backup QB it is a precise estimate
of a different one." Applying w to him is applying a coefficient off its
support, and it reproduces CP5's failure exactly. Measured on the live
2023-25 pool, unguarded reversion inflated 50 quarterbacks with under 12
games of history.

So reversion applies ONLY to quarterbacks whose baseline rests on
SUPPORT_MIN_GAMES or more. Everyone else keeps today's behaviour
untouched: raw baseline, zero adjustment. The guard can therefore only
be a no-op or an improvement -- it cannot reintroduce the CP5 failure,
because the players CP5 was about are never modified.

    live pool, w = 0.544 toward anchor 18.99
      guard  0 -> 111 of 111 reverted, 50 thin quarterbacks inflated
      guard 12 ->  61 of 111 reverted,  0 thin quarterbacks inflated
      guard 16 ->  49 of 111 reverted,  0 thin quarterbacks inflated
      guard 20 ->  39 of 111 reverted,  0 thin quarterbacks inflated

SUPPORT_MIN_GAMES IS NOT TUNED ON THE TEST SET. It is set equal to
`features.SHRINKAGE_ANCHOR_MIN_GAMES`, the constant this project already
uses to mean "enough games to be a real observation." Held-out pooled
gain is +0.270 / +0.259 / +0.277 at guards 12 / 16 / 20, so the choice
is not load-bearing -- which is the point. Picking the guard by held-out
gain would have chosen 0 and shipped the CP5 bug.

WHAT THIS TOUCHES
-----------------
`fantasy_points_per_game_shrunk` for quarterbacks that clear the guard,
and nothing else. QB has no situational weights, so the reverted value
flows through `ranking.apply_situational_weights` unchanged and lands on
the board as Adj PPG. It DOES move rank and VOR -- compressing the QB
spread pulls the replacement quarterback up and elite quarterbacks down
-- so unlike the availability machinery this is a ranking change and has
to be diffed with `compare_boards.py` before a board is drafted from.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKTEST_PATH = PROJECT_ROOT / "data" / "backtest_features.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "qb_reversion.json"
GATE_PATH = PROJECT_ROOT / "data" / "qb_reversion_gate.json"

POSITION = "QB"

# Kept in sync with features.SHRINKAGE_ANCHOR_MIN_GAMES on purpose --
# see the module docstring. Imported rather than duplicated so the two
# cannot drift apart silently.
from src.features import SHRINKAGE_ANCHOR_MIN_GAMES as SUPPORT_MIN_GAMES

# The same three folds every other gate in this project uses.
GATE_FOLDS = [2023, 2024, 2025]

# The target-season games filter fit_weights applies. Reproduced here
# rather than imported through fit_weights.load_backtest so this module
# can state its own training population in one place.
MIN_GAMES = 8


def load_qb(path=BACKTEST_PATH):
    df = pl.read_csv(path)
    df = df.filter(pl.col("actual_games_played") >= MIN_GAMES)
    return df.filter(pl.col("position") == POSITION)


def fit(frame):
    """
    Least squares for  actual = w * baseline + (1 - w) * anchor.

    The anchor is the training frame's mean ACTUAL ppg, not a percentile
    of baselines. Phase 11 CP5's 30th percentile was chosen to avoid
    pulling part-time players up toward a starter's number; here the
    support guard does that job directly, so the anchor can be the
    honest centre of the distribution instead of a defensive one.
    """
    anchor = float(frame["actual_ppg"].mean())
    b = frame["baseline_ppg"].to_numpy() - anchor
    a = frame["actual_ppg"].to_numpy() - anchor
    denom = float((b * b).sum())
    w = float((b * a).sum() / denom) if denom > 0 else 1.0
    return w, anchor


def apply(baseline, games, w, anchor, support_min_games=SUPPORT_MIN_GAMES):
    """Vectorised reversion with the support guard. Returns (values, mask)."""
    baseline = np.asarray(baseline, dtype=float)
    games = np.asarray(games, dtype=float)
    on = games >= support_min_games
    out = baseline.copy()
    out[on] = w * baseline[on] + (1 - w) * anchor
    return out, on


def _rmse(x, y):
    return float(np.sqrt(((x - y) ** 2).mean()))


def _write_gate(gate):
    GATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GATE_PATH, "w") as fh:
        json.dump(gate, fh, indent=2)
    print(f"  wrote {GATE_PATH}")


def run_gate(write=True, support_min_games=SUPPORT_MIN_GAMES):
    """
    THE RULE, and it is holdout.run_gate's rule, not a new one:

      pooled over folds by test-set size, the reverted baseline must beat
      the raw baseline on held-out `actual_ppg`.

    Pooled, not averaged, for the reason holdout.py spells out at length
    -- `rmse^2 * n` is a fold's sum of squared errors, and summing those
    is the RMSE over every held-out player at once. And pooled rather
    than unanimous because that file settled that question too: "one bad
    fold out of three is noise, and requiring 3/3 would cut features that
    are real." The 2023 fold IS mildly negative here (-0.049) and is
    reported below rather than hidden.

    Two invariants are asserted alongside the score, because they are the
    properties that make this safe rather than merely accurate:

      1. No quarterback below the guard is modified. This is the CP5
         protection and it is checked, not assumed.
      2. w must be stable across folds. A w that moves is a w that was
         fitted to noise.
    """
    df = load_qb()
    if df.height == 0:
        raise SystemExit("no QB rows in the backtest -- run python -m src.backtest")

    folds, ws = [], []
    for season in GATE_FOLDS:
        train = df.filter(pl.col("season") != season)
        test = df.filter(pl.col("season") == season)
        if train.height == 0 or test.height == 0:
            continue
        w, anchor = fit(train)
        ws.append(w)

        b = test["baseline_ppg"].to_numpy()
        a = test["actual_ppg"].to_numpy()
        g = test["baseline_games"].to_numpy()
        reverted, on = apply(b, g, w, anchor, support_min_games)

        # INVARIANT 1, checked rather than trusted.
        untouched = ~on
        if untouched.any() and not np.allclose(reverted[untouched], b[untouched]):
            raise SystemExit(
                "GATE ABORTED: a quarterback below the support guard was "
                "modified. That is the Phase 11 CP5 failure and it must "
                "not be possible."
            )

        folds.append({
            "season": int(season),
            "n_test": int(test.height),
            "n_reverted": int(on.sum()),
            "n_untouched": int(untouched.sum()),
            "w": w,
            "anchor": anchor,
            "rmse_raw": _rmse(b, a),
            "rmse_reverted": _rmse(reverted, a),
            "gain": _rmse(b, a) - _rmse(reverted, a),
        })

    total = sum(f["n_test"] for f in folds)
    pooled_raw = float(np.sqrt(
        sum(f["n_test"] * f["rmse_raw"] ** 2 for f in folds) / total))
    pooled_rev = float(np.sqrt(
        sum(f["n_test"] * f["rmse_reverted"] ** 2 for f in folds) / total))
    pooled_gain = pooled_raw - pooled_rev

    w_spread = (max(ws) - min(ws)) if ws else 1.0
    w_stable = w_spread < 0.10          # INVARIANT 2
    passed = bool(pooled_gain > 0 and w_stable)

    gate = {
        "passed": passed,
        "position": POSITION,
        "support_min_games": int(support_min_games),
        "folds": folds,
        "n_held_out": total,
        "pooled_rmse_raw": pooled_raw,
        "pooled_rmse_reverted": pooled_rev,
        "pooled_gain": pooled_gain,
        "w_spread_across_folds": w_spread,
        "w_stable": w_stable,
        "scored_on": "actual_ppg, held-out season, MIN_GAMES=8 population",
        "rule": (
            "Pooled by test-set size (holdout.run_gate's arithmetic), the "
            "reverted baseline must beat the raw baseline on held-out "
            "actual_ppg, AND w must be stable across folds (<0.10 spread). "
            "Scored on actual_ppg rather than delta so the "
            "delta = actual - baseline artifact cannot produce the result."
        ),
        "note": (
            "This is NOT holdout.py's gate and NOT playing_time.py's. "
            "holdout.py tests feature ablation on the veteran and rookie "
            "models and knows nothing about this file. Unlike the "
            "playing-time model, this one MOVES RANK -- diff with "
            "compare_boards.py before drafting from a board built on it."
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }

    print(f"\n=== QB reversion gate (support guard {support_min_games} games) ===")
    for f in folds:
        flag = "+" if f["gain"] > 0 else "-"
        print(f"  {f['season']}  n={f['n_test']:<3d} reverted={f['n_reverted']:<3d} "
              f"untouched={f['n_untouched']:<2d}  w={f['w']:.3f}  "
              f"raw={f['rmse_raw']:.4f} reverted={f['rmse_reverted']:.4f}  "
              f"gain {flag}{abs(f['gain']):.4f}")
    print(f"\n  pooled over {total} held-out quarterbacks:")
    print(f"    raw      {pooled_raw:.4f}")
    print(f"    reverted {pooled_rev:.4f}")
    print(f"    gain     {pooled_gain:+.4f}")
    print(f"  w spread across folds: {w_spread:.4f} "
          f"({'stable' if w_stable else 'UNSTABLE'})")
    print(f"\n  GATE: {'PASSED' if passed else 'FAILED'}")

    if write:
        _write_gate(gate)
    return gate


def build(write=True):
    """Fits the shipped w and anchor on every season, after the gate."""
    df = load_qb()
    w, anchor = fit(df)
    model = {
        "_meta": {
            "source": "backtest_features.csv, QB only, MIN_GAMES=8",
            "seasons": sorted(df["season"].unique().to_list()),
            "n": int(df.height),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "note": (
                "Phase 15b QB mean reversion. Generated by "
                "src/qb_reversion.py -- do not hand-edit. Feeds "
                "fantasy_points_per_game_shrunk for quarterbacks at or "
                "above support_min_games and nothing else. Must not be "
                "wired into a board without a passing, newer "
                "qb_reversion_gate.json."
            ),
        },
        "position": POSITION,
        "w": w,
        "anchor_ppg": anchor,
        "support_min_games": int(SUPPORT_MIN_GAMES),
    }
    print(f"\n=== shipped QB reversion ===")
    print(f"  w      = {w:.4f}   (1.0 would be no reversion at all)")
    print(f"  anchor = {anchor:.4f} PPG")
    print(f"  guard  = {SUPPORT_MIN_GAMES} baseline games")
    print(f"  fitted on {df.height} quarterback-seasons")
    if write:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "w") as fh:
            json.dump(model, fh, indent=2)
        print(f"  wrote {MODEL_PATH}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Phase 15b QB mean reversion.")
    parser.add_argument("--gate", action="store_true",
                        help="run the gate only, do not refit the shipped model")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    write = not args.no_write

    # ORDERING, AND IT IS LOAD-BEARING (fixed Aug 27).
    #
    # `build_board.require_qb_reversion_gate` blocks the build when the
    # MODEL is newer than the GATE, on the rule that an artifact must not
    # postdate the validation that passed it. The first version of this
    # function wrote the gate and then the model, which made the model
    # newer every single time and blocked every build -- a staleness
    # check that fired on the one case that is never stale. It shipped
    # because the sandbox it was written in could not reach the ADP feed
    # and so never built a board.
    #
    # So: score first, write the MODEL, then write the GATE last. The
    # gate is now the newest file, which is what the check wants and what
    # it means -- this model has been validated, and nothing has happened
    # to it since.
    gate = run_gate(write=False)

    if args.gate:
        if write:
            _write_gate(gate)
        return 0 if gate["passed"] else 1

    if not gate["passed"]:
        if write:
            _write_gate(gate)
        print("\nGate failed -- the model is NOT written. Nothing changes.")
        return 1

    build(write=write)
    if write:
        _write_gate(gate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
