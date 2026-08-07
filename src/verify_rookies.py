"""
Phase 12 CP3. Checks that the rookie model does the thing Phase 12
exists to do, and that it did not break anything on the way.

Companion to verify_adjustments.py, which guards the veteran path. Same
Check class, same exit-non-zero-on-hard-failure contract, so both can
gate a commit.

THE FOUR QUESTIONS
------------------
1. RECONCILIATION. Same OLS identity verify_adjustments checks for
   veterans: mean(applied adjustment) == mean(delta) over the fit
   sample, to floating point. This is the check that would have caught
   Phase 6, and it catches the rookie version of the same bug -- an
   intercept fitted but not applied, or a center applied without its
   coefficient.

2. SEPARATION. The plan's actual CP3: do two same-round rookies on
   different teams now get different numbers? Before Phase 12 the answer
   was structurally no -- Jeremiyah Love (ARI) and Jadarian Price (SEA)
   both projected 15.12 PPG and tied at model ranks 4-5, because the
   model had no channel through which a landing spot could reach a
   rookie's projection. Ties within a (position, round) cell are
   therefore the direct measurement of whether the phase worked.

3. DEFENSIBILITY AGAINST VETERANS. Phase 7's complaint was that rookies
   ranked too high, and part of that was mechanical: veterans were
   eating a phantom -3.4 penalty while rookies floated up untouched.
   That is fixed, but Phase 12 introduces the opposite risk -- a rookie
   model with a large positive intercept would re-inflate the whole
   class in one step. This compares the rookie and veteran pools at
   each position and flags a rookie share of the top ranks that no
   draft class has ever earned.

4. NO CROSS-CONTAMINATION. A rookie must never pick up a veteran
   coefficient and vice versa. Checked by asserting that the feature
   sets are disjoint where they must be, and that every rookie's
   adjustment is reproducible from the rookie weights alone.

USAGE
-----
    python -m src.verify_rookies

Requires data/rookie_backtest_features.csv, data/rookie_weights.json,
and data/player_features.csv. Skips individual sections with a message
when an input is missing rather than failing the run -- an unfitted
rookie model is a legitimate documented outcome, not an error.
"""

import sys
from pathlib import Path

import polars as pl

from src.fit_rookie_weights import (
    BACKTEST_PATH as ROOKIE_BACKTEST_PATH,
    FEATURE_SPECS as ROOKIE_FEATURE_SPECS,
    IMPUTED_FEATURES as ROOKIE_IMPUTED,
    WEIGHTS_PATH as ROOKIE_WEIGHTS_PATH,
    load_rookie_backtest,
)
from src.fit_weights import FEATURE_SPECS as VETERAN_FEATURE_SPECS
from src.ranking import _position_adjustment, load_rookie_weights
from src.verify_adjustments import Check

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLAYER_FEATURES_PATH = PROJECT_ROOT / "data" / "player_features.csv"

RECONCILIATION_TOLERANCE = 1e-6

# A rookie class has never supplied more than a small minority of the
# genuinely startable players at a position. This is a smell test, not a
# law -- it is deliberately loose, because the point is to catch a
# structural re-inflation (every rookie moving up together) rather than
# to referee any individual player.
TOP_N = 36
MAX_ROOKIE_SHARE = 0.40


def check_reconciliation(check):
    """mean(applied adjustment) == mean(delta), per position."""
    print("\n1. RECONCILIATION  --  mean applied adjustment vs mean actual delta")

    if not ROOKIE_BACKTEST_PATH.exists() or not ROOKIE_WEIGHTS_PATH.exists():
        check.soft(False, "rookie backtest and weights both present",
                   "run src.rookie_backtest then src.fit_rookie_weights")
        return

    df = load_rookie_backtest()
    weights = load_rookie_weights()

    if not weights:
        check.soft(False, "at least one rookie position met the stability bar",
                   "all positions fell back to the flat cohort baseline -- "
                   "this is the documented Phase 12 fallback, not a bug")
        return

    print(f"\n   {'pos':<5}{'n':>6}{'mean applied':>15}{'mean delta':>14}{'gap':>13}")
    for position, spec in weights.items():
        # The spec the fit ACTUALLY used, which is not necessarily
        # ROOKIE_FEATURE_SPECS -- the season-confound filter can remove
        # features before fitting. Falling back to the module constant
        # keeps older weights files readable, but a mismatch would change
        # which rows survive drop_nulls and break the identity below for
        # reasons unrelated to the weights.
        features = spec.get("features_considered") or ROOKIE_FEATURE_SPECS[position]

        # Reproduce exactly the rows fit_position() trained on.
        required = [f for f in features if f not in ROOKIE_IMPUTED]
        subset = df.filter(pl.col("position") == position).drop_nulls(
            subset=required + ["delta"]
        )
        for f in features:
            if f in ROOKIE_IMPUTED:
                mean_value = subset.select(pl.col(f).cast(pl.Float64).mean()).item()
                subset = subset.with_columns(
                    pl.col(f).cast(pl.Float64).fill_null(mean_value)
                )

        applied = subset.with_columns(
            _position_adjustment(spec).alias("adjustment")
        )
        mean_applied = applied.select(pl.col("adjustment").mean()).item()
        mean_delta = applied.select(pl.col("delta").mean()).item()
        gap = mean_applied - mean_delta

        print(f"   {position:<5}{subset.height:>6}{mean_applied:>15.4f}"
              f"{mean_delta:>14.4f}{gap:>13.6f}")
        check.hard(
            abs(gap) < RECONCILIATION_TOLERANCE,
            f"{position}: applied adjustment reconciles to the fit",
            f"gap {gap:+.8f}",
        )


def check_separation(check):
    """
    THE Phase 12 checkpoint. Two same-round rookies at the same position
    must no longer tie.
    """
    print("\n2. SEPARATION  --  do same-round rookies on different teams differ?")

    if not PLAYER_FEATURES_PATH.exists():
        check.soft(False, "player_features.csv present", "run src.pipeline")
        return

    players = pl.read_csv(PLAYER_FEATURES_PATH).with_columns(
        pl.col("is_rookie").cast(pl.String).str.to_lowercase().eq("true")
    )
    rookies = players.filter(pl.col("is_rookie"))

    if rookies.height == 0:
        check.soft(False, "rookies present in player_features.csv")
        return

    # SEPARATION IS A HARD CHECK ONLY WHEN THERE IS SOMETHING TO SEPARATE
    # BY (fixed Aug 6, after it failed for the right reason).
    #
    # With no trusted rookie weights, every player in a (position, round)
    # cell SHOULD tie -- that is the flat cohort baseline doing exactly
    # what it has always done, and it is the documented Phase 12
    # fallback. Failing the build for it would mean the fallback path can
    # never pass its own verification, which turns a designed outcome
    # into a broken one and makes the check useless as a gate.
    #
    # So: ties are a FAILURE when weights are active and expected to
    # break them, and a NOTE when there are none. The distinction is the
    # whole point of the check.
    active = load_rookie_weights()
    positions_with_weights = set(active)
    if not positions_with_weights:
        print("\n   No trusted rookie weights are active -- every cell is EXPECTED")
        print("   to tie. Reporting the pre-Phase-12 state, not failing on it.")

    # A "cell" is one (position, round) bucket -- exactly the granularity
    # the flat cohort baseline operates at, so every player inside a cell
    # was identical before Phase 12 by construction.
    # `round` comes back from CSV as a string whenever the column has
    # nulls in it -- undrafted rookies -- so it is cast rather than
    # trusted. Read as a string it still GROUPS correctly, which is why
    # this only surfaced at the point of formatting a label, three lines
    # after the number had already been used.
    # THE TARGET IS `teams`, NOT `n` (corrected Aug 6).
    #
    # CP3 asks whether two same-round rookies on DIFFERENT TEAMS
    # separate. Two rookies on the SAME team at the same position in the
    # same round are identical in every feature the rookie model has --
    # same team pace, same position competition, same O-line -- so they
    # SHOULD tie, and the model would have to be inventing something to
    # tell them apart.
    #
    # The first run failed this check on exactly two cells, WR round 3
    # (9 players, 8 distinct) and WR round 6 (7 players, 6 distinct):
    # one duplicate pair each, both same-team. Counting distinct teams
    # instead of distinct players asks the question the plan actually
    # asked, and stops the check from demanding a distinction the data
    # does not contain.
    cells = (
        rookies.with_columns(pl.col("round").cast(pl.Float64, strict=False))
        .filter(pl.col("round").is_not_null())
        .group_by(["position", "round"])
        .agg([
            pl.len().alias("n"),
            pl.col("team").n_unique().alias("teams"),
            pl.col("adjusted_fantasy_points_per_game").n_unique().alias("distinct"),
            pl.col("adjusted_fantasy_points_per_game").std().alias("spread"),
        ])
        .filter(pl.col("n") > 1)
        .sort(["position", "round"])
    )

    print(f"\n   {'cell':<12}{'n':>5}{'teams':>7}{'distinct':>10}{'sd PPG':>10}")
    for row in cells.iter_rows(named=True):
        cell = f"{row['position']} rd{row['round']:.0f}"
        spread = row["spread"] if row["spread"] is not None else 0.0
        mark = "" if row["distinct"] >= row["teams"] else "   <-- TIE ACROSS TEAMS"
        print(f"   {cell:<12}{row['n']:>5}{row['teams']:>7}{row['distinct']:>10}"
              f"{spread:>10.2f}{mark}")

    if not positions_with_weights:
        check.soft(
            True,
            "separation not applicable -- no rookie position met the stability bar",
            f"{cells.height} cells tie, as the flat cohort baseline implies",
        )
        return

    # Only judge the positions the rookie model actually fitted. A cell at
    # an unfitted position ties by design and must not drag down a check
    # about the positions that did fit.
    judged = cells.filter(pl.col("position").is_in(list(positions_with_weights)))
    fully_separated = judged.filter(pl.col("distinct") >= pl.col("teams")).height
    total_cells = judged.height

    check.hard(
        fully_separated == total_cells,
        f"same-round rookies on different teams separate at "
        f"{sorted(positions_with_weights)}",
        f"{fully_separated} of {total_cells} cells separate by team",
    )

    # A cell can separate by a rounding error and still be useless. The
    # spread has to be big enough to reorder a draft board.
    mean_spread = judged.select(pl.col("spread").mean()).item() or 0.0
    check.soft(
        mean_spread >= 0.5,
        "separation is large enough to matter (mean within-cell sd >= 0.5 PPG)",
        f"mean sd {mean_spread:.2f} PPG",
    )


def check_defensibility(check):
    """Rookies must not re-inflate against the veteran pool."""
    print("\n3. DEFENSIBILITY  --  rookie share of the top of each position")

    if not PLAYER_FEATURES_PATH.exists():
        return

    players = pl.read_csv(PLAYER_FEATURES_PATH).with_columns(
        pl.col("is_rookie").cast(pl.String).str.to_lowercase().eq("true")
    )

    print(f"\n   {'pos':<5}{'rookies in top ' + str(TOP_N):>22}{'share':>9}")
    for position in ["QB", "RB", "WR", "TE"]:
        pool = (
            players.filter(pl.col("position") == position)
            .sort("adjusted_fantasy_points_per_game", descending=True, nulls_last=True)
            .head(TOP_N)
        )
        if pool.height == 0:
            continue
        rookie_count = pool.filter(pl.col("is_rookie")).height
        share = rookie_count / pool.height

        print(f"   {position:<5}{rookie_count:>22}{share:>9.1%}")
        check.soft(
            share <= MAX_ROOKIE_SHARE,
            f"{position}: rookies are under {MAX_ROOKIE_SHARE:.0%} of the top {TOP_N}",
            f"{rookie_count}/{pool.height} = {share:.1%}",
        )


def check_no_contamination(check):
    """
    The rookie and veteran models must not share features that mean
    different things, and no rookie-only feature may appear in a veteran
    spec.
    """
    print("\n4. CONTAMINATION  --  rookie and veteran specs stay separate")

    # These are the features whose VALUE is a rookie artifact rather than
    # a measurement. If one ever appears in a rookie spec, the fit is
    # reading a constant or a rookie-detector as if it were signal.
    forbidden_for_rookies = {"team_changed", "experience", "workload_share",
                             "usage_trend_share", "trend_missing",
                             "recent_major_injury", "qb_changed"}

    for position, features in ROOKIE_FEATURE_SPECS.items():
        leaked = sorted(set(features) & forbidden_for_rookies)
        check.hard(
            not leaked,
            f"{position}: rookie spec excludes veteran-only features",
            f"LEAKED: {leaked}" if leaked else "",
        )

    # Rookie-only features must not be in any veteran spec. `pick` and
    # `pos_rank` are null for every veteran, so a veteran coefficient on
    # them would be fitted entirely on imputed values.
    rookie_only = {"pick", "pos_rank", "depth_chart_missing"}
    for position, features in VETERAN_FEATURE_SPECS.items():
        leaked = sorted(set(features) & rookie_only)
        check.hard(
            not leaked,
            f"{position}: veteran spec excludes rookie-only features",
            f"LEAKED: {leaked}" if leaked else "",
        )

    # And the weights files must not have grown into each other.
    rookie_weights = load_rookie_weights()
    for position, spec in rookie_weights.items():
        shipped = set(spec["weights"])
        leaked = sorted(shipped & forbidden_for_rookies)
        check.hard(
            not leaked,
            f"{position}: shipped rookie weights carry no veteran feature",
            f"LEAKED: {leaked}" if leaked else "",
        )


def main():
    print("=" * 74)
    print("VERIFYING THE PHASE 12 ROOKIE MODEL")
    print("=" * 74)

    check = Check()
    check_reconciliation(check)
    check_separation(check)
    check_defensibility(check)
    check_no_contamination(check)

    print("\n" + "=" * 74)
    if check.failures:
        print(f"{len(check.failures)} HARD FAILURE(S): {check.failures}")
        sys.exit(1)
    if check.warnings:
        print(f"All hard checks passed. {len(check.warnings)} warning(s): "
              f"{check.warnings}")
    else:
        print("All checks passed.")


if __name__ == "__main__":
    main()
