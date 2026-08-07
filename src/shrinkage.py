"""
Phase 11 B, checkpoints CP4 and CP5.

    python -m src.backtest              # REQUIRED FIRST -- adds baseline_games
    python -m src.shrinkage             # the sweep
    python -m src.shrinkage --trend-age # the off-support check

WHAT THIS ANSWERS
-----------------
A baseline built on 8 games is currently trusted exactly as much as one
built on 37. Cam Skattebo ranks 11th on 8 games, Omarion Hampton 18th on
9, Phil Mafah projects 13.4 PPG on a single game -- and Phase 11 A's CP1
found the problem is worse than the plan assumed, with backup
quarterbacks (Driskel 17.15 PPG on one game, Wentz, Mariota, Brissett)
sitting inside the model's top 100 on inflated spot-duty rates.

The fix under test is shrinkage toward the position anchor:

    weight  = games / (games + K)
    shrunk  = weight * baseline + (1 - weight) * anchor

K is a prior strength measured in games: at K games of history the
baseline gets half its own weight. K=0 is the incumbent, no shrinkage.

WHY THIS FITS trend_missing AT THE SAME TIME -- NOT NEGOTIABLE
--------------------------------------------------------------
The plan marks this MANDATORY and it is not a formality. Phase 10 shipped
`trend_missing` at RB knowing it overlaps this checkpoint, and the
overlap lands on exactly the players this file was written about: a
player with too little history to fit a usage slope is usually a player
with too little history to trust the baseline of. Phase 10 pays him for
the first. Shrinkage proposes to charge him for the second. Fit them
separately and the board double-counts, in opposite directions.

So every candidate K here is scored on the COMPOSITE prediction --
shrunk baseline plus a situational adjustment refitted against that
shrunk baseline -- never on the baseline alone. `delta` is recomputed and
every position is refit at each K. The `trend_missing` coefficient is
printed at each step, because the diagnostic that matters is whether it
decays toward zero as K rises. If it does, it was standing in for
shrinkage all along.

DECISION RULE for CP5 -- written before the numbers exist:

  Adopt a K > 0 only if ALL of:
    (a) composite paired dMAE on the LOW-CONFIDENCE subgroup beats K=0 by
        more than 2 standard errors,
    (b) composite MAE on the FULL pool does not get worse, and
    (c) the chosen K is not at the edge of the swept range -- an optimum
        at the boundary means the range was wrong, not that the answer is
        the boundary.

  Report the trend_missing decay either way. If it collapses, that is a
  finding about Phase 10 regardless of whether shrinkage ships.
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from src.fit_weights import (
    ALPHA,
    BOOL_COLUMNS,
    FEATURE_SPECS,
    MIN_GAMES,
    fit_position,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKTEST_PATH = PROJECT_ROOT / "data" / "backtest_features.csv"

# Prior strength in games. 0 is the incumbent. The range deliberately
# runs past where anyone expects the answer so clause (c) can bite.
K_VALUES = [0, 4, 8, 12, 16, 24, 32, 48, 64]

# A baseline resting on less than a full season of football. This is the
# population shrinkage is FOR, and the one clause (a) is scored on.
LOW_CONFIDENCE_GAMES = 17

# What a shrunk baseline is pulled TOWARD. Every anchor is computed per
# (season, position) from the baseline table only -- never from the
# target season -- so none of them leak the outcome.
ANCHORS = ["mean_all", "mean_qualified", "median_qualified", "p30_qualified"]
DEFAULT_ANCHORS = ["mean_qualified", "p30_qualified"]
ANCHOR_QUALIFIED_GAMES = 16

# Two shrinkage FORMS, and the second one exists because a hand-check of
# the first found a problem before the backtest ran.
#
#   james_stein  w = g / (g + K)
#   capped       w = min(1, (g/(g+K)) / (REF/(REF+K)))
#
# The textbook form shrinks EVERYONE. At K=8 it pulls Jahmyr Gibbs from
# 20.42 to 18.82 on 49 games of history, and Christian McCaffrey from
# 20.73 to 18.64 on 37. That is not a confidence correction, it is a
# global recalibration wearing one, and it would show up in the sweep as
# full-pool MAE degrading -- tripping clause (b) and killing shrinkage for
# a reason that has nothing to do with the small-sample players the phase
# is about.
#
# `capped` renormalizes so a player with REF games gets weight exactly 1
# and is left alone entirely. Only players below REF are pulled toward the
# anchor. Both are swept; if they agree, the distinction did not matter
# and that is worth knowing too.
SHRINK_FORMS = ["james_stein", "capped"]
SHRINK_REFERENCE_GAMES = 34  # two full seasons = "enough history"


def load_backtest_with_games():
    """
    Same coercion as fit_weights.load_backtest(), but keeps the whole
    frame -- the MIN_GAMES filter on the TARGET season is applied here so
    it is visible, and `baseline_games` must survive it.
    """
    if not BACKTEST_PATH.exists():
        raise SystemExit(
            f"{BACKTEST_PATH} not found. Run `python -m src.backtest` first."
        )

    df = pl.read_csv(BACKTEST_PATH)

    missing = [c for c in ("baseline_games", "baseline_seasons") if c not in df.columns]
    if missing:
        raise SystemExit(
            f"backtest_features.csv is missing {missing}. This file predates "
            f"Phase 11 B -- re-run `python -m src.backtest` to regenerate it "
            f"with the baseline-window game counts."
        )

    present = [c for c in BOOL_COLUMNS if c in df.columns]
    df = df.with_columns([
        pl.col(c).cast(pl.String).str.to_lowercase().eq("true").cast(pl.Int8).alias(c)
        for c in present
    ])
    df = df.with_columns(
        (pl.col("qb_changed") + pl.col("coach_changed")).alias("continuity_score")
    )
    return df.filter(pl.col("actual_games_played") >= MIN_GAMES)


def attach_anchor(df, anchor=ANCHORS[1]):
    """
    Adds `anchor_ppg` per (season, position).

    The anchor choice is not cosmetic and is swept alongside K. Shrinking
    toward the mean of ALL players at a position pulls toward a number
    dominated by fringe roster filler -- there are several hundred
    receivers and most of them are nobody. `mean_qualified` restricts to
    players whose own baseline rests on a real sample, which is a more
    honest statement of "what a typical player at this position does."
    """
    qualified = pl.col("baseline_games") >= ANCHOR_QUALIFIED_GAMES

    if anchor == "mean_all":
        expr = pl.col("baseline_ppg").mean()
    elif anchor == "mean_qualified":
        expr = pl.col("baseline_ppg").filter(qualified).mean()
    elif anchor == "median_qualified":
        expr = pl.col("baseline_ppg").filter(qualified).median()
    elif anchor == "p30_qualified":
        expr = pl.col("baseline_ppg").filter(qualified).quantile(0.30)
    else:
        raise ValueError(f"unknown anchor {anchor!r}; expected one of {ANCHORS}")

    return df.with_columns(
        expr.over(["season", "position"]).alias("anchor_ppg")
    ).with_columns(
        # A (season, position) cell with nobody qualified would produce a
        # null anchor and silently drop every row in it. Fall back to the
        # cell's own mean rather than losing the season.
        pl.col("anchor_ppg").fill_null(
            pl.col("baseline_ppg").mean().over(["season", "position"])
        )
    )


def apply_shrinkage(df, k, form="james_stein"):
    """
    Adds `shrunk_baseline` and recomputes `delta` against it.

    K=0 is an exact passthrough under either form -- weight is
    games/(games+0) = 1 -- so the incumbent is reproduced by the same code
    path as every challenger rather than by a separate branch that could
    drift from it.
    """
    games = pl.col("baseline_games")

    if k == 0:
        weight = pl.lit(1.0)
    elif form == "james_stein":
        weight = games / (games + k)
    elif form == "capped":
        reference = SHRINK_REFERENCE_GAMES / (SHRINK_REFERENCE_GAMES + k)
        weight = ((games / (games + k)) / reference).clip(0.0, 1.0)
    else:
        raise ValueError(f"unknown shrinkage form {form!r}; expected one of {SHRINK_FORMS}")

    return df.with_columns(
        (weight * pl.col("baseline_ppg") + (1 - weight) * pl.col("anchor_ppg"))
        .alias("shrunk_baseline")
    ).with_columns(
        (pl.col("actual_ppg") - pl.col("shrunk_baseline")).alias("delta")
    )


def fit_and_predict(df, alpha=ALPHA):
    """
    Refits every position against the CURRENT `delta` and returns the
    frame with a `predicted_ppg` column plus the fitted specs.

    This is the joint part. The situational weights are re-estimated at
    every K rather than carried over, because `delta` is a different
    target once the baseline moves -- reusing Phase 10's coefficients
    would score shrinkage against a model fitted to a baseline that no
    longer exists.
    """
    frames, specs = [], {}

    for position, features in FEATURE_SPECS.items():
        subset = df.filter(pl.col("position") == position)
        if subset.height == 0:
            continue

        result, _, _ = fit_position(subset, position, features, alpha=alpha)
        if result is None:
            frames.append(subset.with_columns(
                pl.col("shrunk_baseline").alias("predicted_ppg")
            ))
            continue

        specs[position] = result

        # `intercept_fitted`, NOT the shipped `intercept`. This is a
        # predictive-accuracy test, and the shipped QB value has the
        # Phase 10 level shift removed -- a deliberate selection-bias
        # correction for draft-day use, but here it would just bias every
        # QB prediction low by a constant 0.70 and inflate MAE for reasons
        # that have nothing to do with shrinkage. The shift is identical
        # at every K, so it cannot change which K wins either way; using
        # the fitted value simply keeps the absolute numbers meaningful.
        adjustment = pl.lit(float(result["intercept_fitted"]))
        centers = result.get("centers", {})
        for feature, coefficient in result["weights"].items():
            value = pl.col(feature).cast(pl.Float64).fill_null(
                float(result["feature_means"].get(feature, 0.0))
            )
            if feature in centers:
                value = value - float(centers[feature])
            adjustment = adjustment + value * coefficient

        frames.append(subset.with_columns(
            (pl.col("shrunk_baseline") + adjustment).alias("predicted_ppg")
        ))

    return pl.concat(frames, how="vertical"), specs


def score(frame):
    errors = (frame.select("predicted_ppg").to_series()
              - frame.select("actual_ppg").to_series()).abs()
    rho = frame.select(pl.corr("predicted_ppg", "actual_ppg", method="spearman")).item()
    return {
        "n": frame.height,
        "mae": float(errors.mean()),
        "rmse": float((errors ** 2).mean() ** 0.5),
        "rho": float(rho) if rho is not None else float("nan"),
    }


def paired_delta(challenger, incumbent):
    """
    Mean paired improvement in absolute error, and its SE, over the rows
    the challenger actually moves. Same design as Phase 11 A's CP2 --
    paired differences strip out the player-to-player variance that
    otherwise swamps a 0.05 PPG effect.
    """
    joined = challenger.select([
        "player_id", "season",
        (pl.col("predicted_ppg") - pl.col("actual_ppg")).abs().alias("challenger_error"),
    ]).join(
        incumbent.select([
            "player_id", "season",
            (pl.col("predicted_ppg") - pl.col("actual_ppg")).abs().alias("incumbent_error"),
        ]),
        on=["player_id", "season"], how="inner",
    ).with_columns(
        (pl.col("incumbent_error") - pl.col("challenger_error")).alias("improvement")
    ).filter(pl.col("improvement").abs() > 1e-9)

    if joined.height < 2:
        return 0.0, 0.0, joined.height

    improvement = joined.select("improvement").to_series()
    return (float(improvement.mean()),
            float(improvement.std() / (joined.height ** 0.5)),
            joined.height)


def run_sweep(df, anchor, k_values=None, alpha=ALPHA, form="james_stein"):
    if k_values is None:
        k_values = K_VALUES

    anchored = attach_anchor(df, anchor)
    low_confidence = pl.col("baseline_games") < LOW_CONFIDENCE_GAMES

    print("\n" + "=" * 88)
    print(f"ANCHOR = {anchor}   FORM = {form}"
          + (f" (untouched at >= {SHRINK_REFERENCE_GAMES} games)"
             if form == "capped" else " (shrinks every player)"))
    print("=" * 88)
    print(f"   {'K':>4}{'MAE all':>10}{'MAE low':>10}{'rho all':>9}"
          f"{'paired dMAE (low)':>20}{'SE':>8}{'n moved':>9}   trend_missing")

    results, predictions = [], {}
    for k in k_values:
        shrunk = apply_shrinkage(anchored, k, form)
        predicted, specs = fit_and_predict(shrunk, alpha=alpha)
        predictions[k] = predicted

        overall = score(predicted)
        low = score(predicted.filter(low_confidence))

        if k == 0:
            delta, se, moved = 0.0, 0.0, 0
            delta_text, se_text, moved_text = "--", "--", "--"
        else:
            delta, se, moved = paired_delta(
                predicted.filter(low_confidence),
                predictions[0].filter(low_confidence),
            )
            delta_text, se_text, moved_text = f"{delta:+.4f}", f"{se:.4f}", str(moved)

        trend = ", ".join(
            f"{position} {specs[position]['weights']['trend_missing']:+.3f}"
            for position in ("RB", "WR", "TE")
            if position in specs and "trend_missing" in specs[position]["weights"]
        ) or "dropped everywhere"

        print(f"   {k:>4}{overall['mae']:>10.4f}{low['mae']:>10.4f}{overall['rho']:>9.4f}"
              f"{delta_text:>20}{se_text:>8}{moved_text:>9}   {trend}")

        # The paired statistics are what the decision rule is actually
        # read against, so they belong in the file the charts are drawn
        # from. Printing them and dropping them meant any chart built
        # later would have to re-derive them and could disagree.
        results.append({"k": k, "anchor": anchor, "form": form, **overall,
                        "mae_low": low["mae"], "n_low": low["n"],
                        "paired_dmae_low": delta, "paired_se_low": se,
                        "n_moved": moved})

    return results, predictions


def report_trend_age_interaction(df, anchor="mean_qualified", k=0):
    """
    The plan's second, narrower problem with `trend_missing`: it was
    estimated on RBs averaging 24.5 years old with only 2% aged 29+, and
    is applied to a live pool averaging 27.2 with 24% aged 29+. A
    washed-up 34-year-old fails MIN_TREND_GAMES for the same mechanical
    reason an ascending rookie does, and collects the same bonus.

    Currently harmless -- of the 12 trend_missing RBs above 8 PPG,
    exactly one is 29+ -- but the plan notes it stops being harmless the
    moment shrinkage moves those players, which is what this file does.

    Tests whether the bonus should be age-conditional by adding a
    trend_missing x age interaction to the RB spec.
    """
    print("\n" + "=" * 88)
    print(f"trend_missing OFF-SUPPORT CHECK  (anchor={anchor}, K={k})")
    print("=" * 88)

    shrunk = apply_shrinkage(attach_anchor(df, anchor), k)

    for position in ("RB", "WR", "TE"):
        features = FEATURE_SPECS.get(position, [])
        if "trend_missing" not in features:
            continue

        subset = shrunk.filter(pl.col("position") == position)
        centered_age = (pl.col("age").cast(pl.Float64)
                        - pl.col("age").cast(pl.Float64).mean())
        subset = subset.with_columns(
            (pl.col("trend_missing").cast(pl.Float64) * centered_age)
            .alias("trend_missing_x_age")
        )

        result, full_model, _ = fit_position(
            subset, position, features + ["trend_missing_x_age"]
        )
        if result is None:
            continue

        diagnostics = result["diagnostics"]
        interaction_p = diagnostics["trend_missing_x_age"]["p_value"]
        interaction_c = diagnostics["trend_missing_x_age"]["coef_full_spec"]
        base_c = diagnostics["trend_missing"]["coef_full_spec"]

        print(f"\n   {position}:  trend_missing {base_c:+.4f}   "
              f"x age {interaction_c:+.4f} (p={interaction_p:.3f})")

        flagged = subset.filter(pl.col("trend_missing") == 1)
        if flagged.height:
            ages = flagged.select("age").to_series()
            print(f"      fitted on {flagged.height} flagged rows, "
                  f"mean age {float(ages.mean()):.1f}, "
                  f"{100 * float((ages >= 29).mean()):.0f}% aged 29+")

    print("\n   Read: a significant NEGATIVE interaction means the bonus should")
    print("   shrink or reverse with age -- i.e. it is a young-player effect being")
    print("   handed to veterans by a mechanical missing-data rule. A null means")
    print("   the flag means the same thing at every age and can stay as-is.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 11 B: backtest baseline shrinkage jointly with trend_missing."
    )
    parser.add_argument("--anchors", nargs="+", default=DEFAULT_ANCHORS,
                        help=f"anchors to sweep (default: {DEFAULT_ANCHORS}; "
                             f"all available: {ANCHORS})")
    parser.add_argument("--forms", nargs="+", default=SHRINK_FORMS,
                        help=f"shrinkage forms (default: {SHRINK_FORMS})")
    parser.add_argument("--k", type=int, nargs="+", default=K_VALUES,
                        help=f"prior strengths in games (default: {K_VALUES})")
    parser.add_argument("--trend-age", action="store_true",
                        help="run the trend_missing x age off-support check")
    parser.add_argument("--alpha", type=float, default=ALPHA)
    args = parser.parse_args()

    data = load_backtest_with_games()
    print(f"Backtest rows (target season >= {MIN_GAMES} games): {data.height}")
    low = data.filter(pl.col("baseline_games") < LOW_CONFIDENCE_GAMES)
    print(f"Low-confidence rows (baseline under {LOW_CONFIDENCE_GAMES} games): "
          f"{low.height} ({100 * low.height / data.height:.1f}%)")
    print(data.group_by("position").agg([
        pl.len().alias("n"),
        pl.col("baseline_games").median().alias("median_baseline_games"),
        (pl.col("baseline_games") < LOW_CONFIDENCE_GAMES).mean().alias("share_low"),
    ]).sort("position"))

    all_results = []
    for form_name in args.forms:
        for anchor_name in args.anchors:
            rows, _ = run_sweep(data, anchor_name, args.k, args.alpha, form_name)
            all_results.extend(rows)

    if args.trend_age:
        report_trend_age_interaction(data)

    summary_path = PROJECT_ROOT / "data" / "shrinkage_sweep.csv"
    pl.DataFrame(all_results).write_csv(summary_path)
    print(f"\nWrote sweep summary to {summary_path}")

    print("""
DECISION RULE for CP5 -- pre-committed, see module docstring:

  Adopt K > 0 only if ALL of:
    (a) paired dMAE on the LOW-CONFIDENCE subgroup beats K=0 by > 2 SE,
    (b) MAE on the FULL pool does not get worse, and
    (c) the winning K is not at the edge of the swept range.

  Report the trend_missing decay regardless. If the coefficient collapses
  as K rises, it was standing in for shrinkage, and that is a finding about
  Phase 10 whether or not shrinkage ships.""")
