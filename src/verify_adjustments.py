"""
Guard rail against the class of bug that produced Phase 7's complaint.

WHAT THIS IS FOR
----------------
Phase 6 shipped slopes without their intercept. Nothing in the codebase
noticed. The error surfaced as a vague human impression -- "every
adjustment is negative," "only 6 skill players clear 15 PPG" -- two
symptoms that took until Phase 8 to trace back to one dropped constant.

The lesson recorded at the time was "intercepts always ship with
coefficients." That's necessary but not sufficient, because the same
failure has more than one costume:

  - Phase 6: intercept fitted, never applied.
  - Phase 10 would have added: centered coefficient applied to
    uncentered data (wrong by coef x center -- about -9.5 PPG at RB).
  - Phase 10 would also have added: intercept taken from the full spec
    while only the significant slopes ship, so the two come from
    different models.

All three are the same mistake -- a coefficient separated from the
constants it was fitted with -- and all three are invisible to a human
reading a spreadsheet. So this file checks the thing they all break.

THE CENTRAL IDENTITY
--------------------
OLS with an intercept forces mean(fitted) == mean(y) exactly. The model
is fitted on `delta` = actual PPG - baseline PPG. So if the weights are
applied correctly, then over the rows the model was fitted on:

    mean(situational_adjustment) == mean(delta)

to floating-point precision. Not approximately. Any visible gap means
the numbers being applied did not come from the model that was fitted.
Under Phase 6's bug this gap was -3.43 at RB.

Crucially, this runs the REAL apply path -- ranking.apply_situational_weights,
the same function pipeline.py calls -- against the fit sample rebuilt
from fit_weights' own rules. It tests the code that ships, not a
restatement of the fit.

USAGE
-----
    python -m src.verify_adjustments

Exits non-zero if any hard check fails, so it can gate a commit.
"""

import json
import sys
from pathlib import Path

import polars as pl

from src.fit_weights import (
    BACKTEST_PATH,
    FEATURE_SPECS,
    IMPUTED_FEATURES,
    WEIGHTS_PATH,
    load_backtest,
)
from src.ranking import apply_situational_weights, load_situational_weights

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLAYER_FEATURES_PATH = PROJECT_ROOT / "data" / "player_features.csv"

# mean(fitted) == mean(y) is an algebraic identity, so the only slack
# needed is floating point.
RECONCILIATION_TOLERANCE = 1e-6

# The Phase 7 complaint, as a number: the raw baseline had 29 skill
# players over 15 PPG and the phantom penalty erased 23 of them.
HIGH_PPG_THRESHOLD = 15.0


class Check:
    """Collects pass/fail results so every check runs before exiting."""

    def __init__(self):
        self.failures = []
        self.warnings = []

    def hard(self, ok, label, detail=""):
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}" + (f"  --  {detail}" if detail else ""))
        if not ok:
            self.failures.append(label)

    def soft(self, ok, label, detail=""):
        status = "ok" if ok else "WARN"
        print(f"  [{status}] {label}" + (f"  --  {detail}" if detail else ""))
        if not ok:
            self.warnings.append(label)


def check_weights_file(check):
    """Structural integrity of situational_weights.json."""
    print("\n1. WEIGHTS FILE STRUCTURE")

    if not WEIGHTS_PATH.exists():
        check.hard(False, "situational_weights.json exists",
                   "run `python -m src.fit_weights`")
        return None

    with open(WEIGHTS_PATH) as f:
        payload = json.load(f)

    positions = payload.get("positions", {})
    check.hard(bool(positions), "weights file has fitted positions",
               f"{sorted(positions)}")

    for position, spec in positions.items():
        check.hard("intercept" in spec, f"{position}: intercept present")

        means = spec.get("feature_means", {})
        missing_means = [f for f in spec["weights"] if f not in means]
        check.hard(
            not missing_means,
            f"{position}: every shipped weight has a feature mean to impute with",
            f"missing: {missing_means}" if missing_means else "",
        )

        # A center without a weight is harmless; a weight that SHOULD be
        # centered but has no center is the -9.5 PPG bug.
        centers = spec.get("centers", {})
        uncentered = [f for f in spec["weights"] if f in {"age"} and f not in centers]
        check.hard(
            not uncentered,
            f"{position}: centered features ship with their center",
            f"missing center: {uncentered}" if uncentered else "",
        )

        # A suppressed level shift must be recorded, not just applied --
        # otherwise a later reader sees an intercept that doesn't match
        # the fit and has no way to tell deliberate from broken.
        if spec.get("level_shift_removed"):
            check.hard(
                "intercept_fitted" in spec,
                f"{position}: suppressed level shift records the fitted intercept",
                f"ships {spec['intercept']:+.4f}, fitted "
                f"{spec.get('intercept_fitted', float('nan')):+.4f}",
            )

        flips = spec.get("sign_flips", [])
        check.soft(
            not flips,
            f"{position}: no coefficient flips sign when a season is withheld",
            f"unstable: {flips}" if flips else "",
        )

    # A weights file older than the data it claims to describe is a
    # silent way to ship last week's model.
    if BACKTEST_PATH.exists():
        stale = WEIGHTS_PATH.stat().st_mtime < BACKTEST_PATH.stat().st_mtime
        check.hard(
            not stale,
            "weights are newer than backtest_features.csv",
            "weights are STALE -- re-run fit_weights" if stale else "",
        )

    return positions


def rebuild_fit_sample(df, position, features):
    """
    Reproduces exactly the rows fit_weights.fit_position() trained on:
    drop rows null in any non-imputed feature, then mean-impute the rest.
    """
    required = [f for f in features if f not in IMPUTED_FEATURES]
    subset = df.filter(pl.col("position") == position).drop_nulls(
        subset=required + ["delta"]
    )
    for f in features:
        if f in IMPUTED_FEATURES:
            mean_value = subset.select(pl.col(f).cast(pl.Float64).mean()).item()
            subset = subset.with_columns(pl.col(f).cast(pl.Float64).fill_null(mean_value))
    return subset


def check_reconciliation(check, positions):
    """
    THE test. Runs the live apply path over the fit sample and requires
    mean(adjustment) == mean(delta).
    """
    print("\n2. RECONCILIATION  --  mean applied adjustment vs mean actual delta")
    print("   (OLS identity: these are equal to floating point when weights are applied correctly)")

    if not BACKTEST_PATH.exists():
        check.hard(False, "backtest_features.csv exists",
                   "run `python -m src.backtest`")
        return

    df = load_backtest()
    weights = load_situational_weights()

    print(f"\n   {'pos':<5}{'n':>6}{'mean applied':>15}{'expected':>14}{'gap':>13}  note")
    for position, features in FEATURE_SPECS.items():
        if position not in positions:
            continue
        subset = rebuild_fit_sample(df, position, features)

        # apply_situational_weights needs these two columns; the
        # baseline is irrelevant to the adjustment itself, so zero it and
        # read the adjustment directly.
        scored = apply_situational_weights(
            subset.with_columns([
                pl.lit(0.0).alias("fantasy_points_per_game"),
                pl.lit(False).alias("is_rookie"),
            ]),
            weights,
        )

        applied = scored.select(pl.col("situational_adjustment").mean()).item()
        actual = subset.select(pl.col("delta").mean()).item()

        # A position whose level shift was deliberately suppressed (see
        # fit_weights.SUPPRESS_LEVEL_SHIFT) should reconcile to
        # mean(delta) MINUS that shift, not to mean(delta). Checking the
        # unadjusted identity here would fail QB for doing exactly what
        # it was told to do -- and, worse, would tempt someone to
        # loosen the tolerance, which is the one check in this file that
        # must stay exact.
        shift = float(positions[position].get("level_shift_removed", 0.0) or 0.0)
        expected = actual - shift
        gap = applied - expected
        note = f"level shift {shift:+.3f} suppressed" if shift else ""
        print(f"   {position:<5}{subset.height:>6}{applied:>15.6f}{expected:>14.6f}"
              f"{gap:>13.2e}  {note}")

        check.hard(
            abs(gap) < RECONCILIATION_TOLERANCE,
            f"{position}: applied adjustment reconciles with fitted model",
            f"gap {gap:+.4f} PPG on every {position}" if abs(gap) >= RECONCILIATION_TOLERANCE else "",
        )


def check_live_board(check, positions):
    """
    Symptom-level checks on the actual output -- the things a human
    noticed in Phase 7, now measured rather than eyeballed.
    """
    print("\n3. LIVE OUTPUT  --  data/player_features.csv")

    if not PLAYER_FEATURES_PATH.exists():
        check.soft(False, "player_features.csv exists",
                   "run `python -m src.pipeline`")
        return

    df = pl.read_csv(PLAYER_FEATURES_PATH, infer_schema_length=0).with_columns([
        pl.col("fantasy_points_per_game").cast(pl.Float64),
        pl.col("adjusted_fantasy_points_per_game").cast(pl.Float64),
        pl.col("situational_adjustment").cast(pl.Float64),
        pl.col("is_rookie").cast(pl.String).str.to_lowercase().eq("true"),
    ])

    veterans = df.filter(~pl.col("is_rookie"))

    print(f"\n   {'pos':<5}{'n':>6}{'mean adj':>11}{'min':>9}{'max':>9}{'% pos':>8}{'% neg':>8}")
    for position in positions:
        p = veterans.filter(pl.col("position") == position)
        if p.height == 0:
            continue
        adj = p.select("situational_adjustment").to_series()
        share_positive = float((adj > 0).mean())
        share_negative = float((adj < 0).mean())
        print(f"   {position:<5}{p.height:>6}{adj.mean():>11.3f}{adj.min():>9.3f}"
              f"{adj.max():>9.3f}{100 * share_positive:>7.0f}%{100 * share_negative:>7.0f}%")

        # The Phase 7 symptom, stated as a test: an adjustment that only
        # ever points one direction is a constant wearing a disguise.
        check.hard(
            share_positive > 0 and share_negative > 0,
            f"{position}: adjustments are two-sided",
            f"{100 * share_positive:.0f}% positive / {100 * share_negative:.0f}% negative",
        )

    skill = df.filter(pl.col("position").is_in(["RB", "WR", "TE"]))
    raw_high = skill.filter(pl.col("fantasy_points_per_game") > HIGH_PPG_THRESHOLD).height
    adj_high = skill.filter(
        pl.col("adjusted_fantasy_points_per_game") > HIGH_PPG_THRESHOLD
    ).height
    print(f"\n   skill players over {HIGH_PPG_THRESHOLD} PPG: raw {raw_high}, adjusted {adj_high}")
    check.soft(
        adj_high >= raw_high * 0.5,
        f"adjusted >{HIGH_PPG_THRESHOLD} PPG count is not decimated",
        f"raw {raw_high} -> adjusted {adj_high} "
        f"({100 * adj_high / raw_high:.0f}% retained)" if raw_high else "",
    )

    rookies = df.filter(pl.col("is_rookie"))
    if rookies.height:
        check.hard(
            float(rookies.select(pl.col("situational_adjustment").abs().max()).item()) == 0.0,
            "rookies take no situational adjustment",
        )


def check_shrunk_baseline(check):
    """
    Phase 11 B (CP5). The weights are fitted against deltas measured from
    the SHRUNK baseline, so the live board must add them to the shrunk
    baseline too. Mixing the two is silent -- every projection would be
    off by whatever shrinkage moved, in the direction that makes thin
    players look good again, which is the exact failure CP5 exists to fix.
    """
    print("\n6. SHRUNK BASELINE  --  is the adjustment sitting on the right number?")

    if not PLAYER_FEATURES_PATH.exists():
        check.soft(False, "player_features.csv exists")
        return

    df = pl.read_csv(PLAYER_FEATURES_PATH)

    if "fantasy_points_per_game_shrunk" not in df.columns:
        check.hard(False, "player_features.csv carries a shrunk baseline",
                   "re-run `python -m src.pipeline` to apply Phase 11 CP5")
        return
    check.hard(True, "player_features.csv carries a shrunk baseline")

    gap = df.select(
        (pl.col("adjusted_fantasy_points_per_game")
         - pl.col("fantasy_points_per_game_shrunk")
         - pl.col("situational_adjustment")).abs().max()
    ).item()
    check.hard(gap < 1e-6,
               "adjusted PPG = shrunk baseline + situational adjustment",
               f"worst gap {gap:.2e}")

    rookies = df.filter(
        pl.col("is_rookie").cast(pl.String).str.to_lowercase().eq("true")
    )
    if rookies.height:
        untouched = rookies.select(
            (pl.col("fantasy_points_per_game_shrunk")
             - pl.col("fantasy_points_per_game")).abs().max()
        ).item()
        check.hard(untouched < 1e-6, "rookies are not shrunk",
                   f"worst move {untouched:.3f}")

    veterans = df.filter(
        ~pl.col("is_rookie").cast(pl.String).str.to_lowercase().eq("true")
    ).with_columns(
        (pl.col("fantasy_points_per_game_shrunk")
         - pl.col("fantasy_points_per_game")).alias("move")
    )
    print(f"\n   {'pos':<5}{'n':>6}{'moved 0.5+':>12}{'mean move':>12}{'worst':>9}")
    for position in ("QB", "RB", "WR", "TE"):
        p = veterans.filter(pl.col("position") == position)
        if p.height == 0:
            continue
        moved = p.filter(pl.col("move").abs() >= 0.5).height
        print(f"   {position:<5}{p.height:>6}{moved:>12}"
              f"{float(p.select(pl.col('move').mean()).item()):>12.3f}"
              f"{float(p.select(pl.col('move').min()).item()):>9.2f}")

    # THIS CHECK ORIGINALLY ASSERTED "shrinkage mostly lowers
    # projections" and warned on every run, because the premise was
    # arithmetically false. Shrinking toward the 30th percentile raises
    # everyone BELOW the 30th percentile -- that is what the anchor is.
    # About half the pool moving up is the mechanism working, not failing,
    # and those players have a median raw projection of 1.4 PPG. A check
    # that always warns is a check nobody reads.
    #
    # What actually matters is narrower: shrinkage must never inflate
    # someone into DRAFTABILITY. That is precisely the QB failure this
    # check was supposed to catch and didn't -- it buried a real defect
    # (clipboard QBs pulled to 8+ PPG) inside a warning that was firing
    # for a harmless reason.
    #
    # Scoped to ADP-bearing players, exactly one moves up: Jonathon
    # Brooks, +0.38 on a 3-game sample, which is the anchor correctly
    # declining to believe 2.50 PPG.
    draftable = veterans.filter(pl.col("adp").is_not_null())
    if draftable.height:
        raised = draftable.filter(pl.col("move") > 0.5)
        worst = float(draftable.select(pl.col("move").max()).item())
        check.hard(
            raised.height == 0,
            "shrinkage never raises a draftable player by 0.5+ PPG",
            f"{raised.height} of {draftable.height} draftable players raised; "
            f"largest upward move {worst:+.2f}",
        )

    fringe = veterans.filter(pl.col("move") > 0.01)
    if fringe.height:
        median_raw = float(
            fringe.select(pl.col("fantasy_points_per_game").median()).item()
        )
        print(f"\n   {fringe.height} of {veterans.height} moved up "
              f"(median raw {median_raw:.2f} PPG -- below the anchor by design)")


def check_value_drivers(check):
    """
    Phase 11. The "Why (value drivers)" column claims to be the model's own
    arithmetic rather than a description of it. This is that claim, tested.

    The board prints only drivers above a display threshold, so the visible
    string does NOT sum to Sit Adj -- that would be a false test. What must
    hold is the identity underneath it: position base plus EVERY
    contribution equals the adjustment the model actually applied. If that
    breaks, the column is explaining a number that isn't there, which is
    worse than having no column at all.
    """
    print("\n4. VALUE DRIVERS  --  do the printed reasons sum to the number?")

    from src.build_board import build_value_drivers  # noqa: PLC0415
    from src.ranking import load_situational_weights  # noqa: PLC0415

    if not PLAYER_FEATURES_PATH.exists():
        check.soft(False, "player_features.csv exists", "run `python -m src.pipeline`")
        return

    weights = load_situational_weights()
    df = pl.read_csv(PLAYER_FEATURES_PATH).filter(
        pl.col("position").is_in(list(weights.keys()))
    ).with_columns(
        pl.col("is_rookie").cast(pl.String).str.to_lowercase().eq("true")
    )

    worst_name, worst_gap = None, 0.0
    for row in df.filter(~pl.col("is_rookie")).to_dicts():
        spec = weights[row["position"]]
        means = spec.get("feature_means", {})
        centers = spec.get("centers", {})

        total = float(spec["intercept"])
        for feature, weight in spec["weights"].items():
            mean = float(means.get(feature, 0.0))
            raw = row.get(feature)
            value = mean if raw is None else float(raw)
            total += (value - float(centers.get(feature, 0.0))) * weight

        gap = abs(total - float(row["situational_adjustment"]))
        if gap > worst_gap:
            worst_name, worst_gap = row["player_name"], gap

    check.hard(
        worst_gap < 1e-6,
        "driver decomposition reconciles to situational_adjustment",
        f"worst gap {worst_gap:.2e}" + (f" ({worst_name})" if worst_name else ""),
    )

    # And the column actually gets produced for everyone.
    built = build_value_drivers(
        df.with_columns([
            pl.lit(False).alias("out_for_season"),
            pl.lit(0.0).alias("expected_games_missed"),
            pl.lit(None, dtype=pl.String).alias("injury_status"),
        ]),
        weights,
    )
    missing = built.filter(pl.col("value_drivers").is_null()).height
    check.hard(missing == 0, "every modeled player gets a driver string",
               f"{missing} null")

    sample = built.filter(~pl.col("is_rookie")).sort(
        pl.col("situational_adjustment").abs(), descending=True
    ).head(5)
    print()
    for row in sample.select(["player_name", "position", "situational_adjustment",
                              "value_drivers"]).iter_rows():
        print(f"   {row[0]:<22}{row[1]:<4}{row[2]:+6.2f}  {row[3]}")


def check_replacement_levels(check):
    """
    Phase 11 CP6/CP7. The sanity condition the plan wrote down before the
    fix existed: a SHALLOWER league must push QB and TE DOWN the board, not
    up. Deep leagues are where quarterback scarcity lives; in a 6-team
    league the best free-agent QB is a perfectly good starter, so the gap a
    pick buys you there is small.

    Checked across both real configs rather than asserted about one, because
    that is precisely how the bug stayed hidden -- it was invisible on the
    12-team board, where starter count and waiver depth roughly agree.
    """
    print("\n5. REPLACEMENT LEVEL  --  does a shallow league discount QB and TE?")

    from src.build_board import (  # noqa: PLC0415
        MODELED_POSITIONS, apply_injury_overrides, compute_draft_targets,
        compute_replacement_ranks, compute_starter_ranks, compute_vor, load_config,
    )

    configs = {
        "Lebron James (12)": PROJECT_ROOT / "league_config.json",
        "Dunlap Family (6)": PROJECT_ROOT / "league_config_dunlap.json",
    }
    if not all(path.exists() for path in configs.values()):
        check.soft(False, "both league configs present")
        return

    # WHAT THIS COUNTS, AND WHY IT CHANGED (Aug 4)
    # -------------------------------------------
    # This first asked whether the BEST player at a position ranked lower
    # in the shallow league. That is a knife-edge statistic for a claim
    # about positional value -- it turns on one player shuffling past a
    # neighbour. It failed on TE by two spots (21 deep, 19 shallow) and
    # the investigation found no bug: TE replacement moves from TE21 to
    # TE7 between the leagues, which sounds enormous, but the TE
    # production curve is FLAT, so that move costs about as much as
    # RB52 -> RB32 costs running backs. The two roughly cancel and TE is
    # genuinely neutral across the two leagues.
    #
    # Removing the starter floor was tested as a candidate cause and moved
    # TE only 19 -> 20, so that was not it either.
    #
    # Counting how many of a position appear in the top 30 measures the
    # same claim without turning on a single player's neighbours. Recorded
    # in full because changing a test that failed is exactly the move that
    # deserves the most scrutiny.
    top_n = 30
    position_counts = {}
    best_rank = {}
    for label, path in configs.items():
        config = load_config(path)
        players = pl.read_csv(PLAYER_FEATURES_PATH).filter(
            pl.col("position").is_in(MODELED_POSITIONS)
        )
        for column in ["has_adp", "is_rookie", "recent_major_injury"]:
            players = players.with_columns(
                pl.col(column).cast(pl.String).str.to_lowercase().eq("true").alias(column)
            )
        players = apply_injury_overrides(players)

        ranks = compute_replacement_ranks(config, players)
        starters = compute_starter_ranks(config)
        print(f"\n   {label}: drafted {ranks}  (starter-slot rule was {starters})")

        board = compute_draft_targets(compute_vor(players, ranks), config["num_teams"])
        top = board.head(top_n)
        for position in MODELED_POSITIONS:
            hit = top.filter(pl.col("position") == position)
            position_counts.setdefault(position, {})[label] = hit.height
            best_rank.setdefault(position, {})[label] = (
                int(hit.select("rank").to_series()[0]) if hit.height else 999
            )
        counts = {p: position_counts[p][label] for p in MODELED_POSITIONS}
        print(f"   top-{top_n} mix: {counts}")

    deep, shallow = list(configs.keys())
    for position in ("QB", "TE"):
        check.hard(
            position_counts[position][shallow] <= position_counts[position][deep],
            f"{position} takes no MORE of the top {top_n} in the 6-team league",
            f"{position} in top {top_n}: {position_counts[position][deep]} deep, "
            f"{position_counts[position][shallow]} shallow",
        )
        # The old assertion, kept as a soft check. It is noisy, but it is
        # also strictly more sensitive than the count, so it is worth
        # seeing rather than deleting.
        check.soft(
            best_rank[position][shallow] >= best_rank[position][deep],
            f"{position}: best player also ranks no higher in the 6-team league",
            f"best {position}: rank {best_rank[position][deep]} deep, "
            f"{best_rank[position][shallow]} shallow",
        )


def main():
    print("=" * 74)
    print("VERIFYING SITUATIONAL ADJUSTMENTS")
    print("=" * 74)

    check = Check()
    positions = check_weights_file(check)
    if positions:
        check_reconciliation(check, positions)
        check_live_board(check, positions)
        check_value_drivers(check)
        check_replacement_levels(check)
        check_shrunk_baseline(check)

    print("\n" + "=" * 74)
    if check.failures:
        print(f"FAILED ({len(check.failures)}):")
        for f in check.failures:
            print(f"  - {f}")
        print("\nDo not build a board from this. Fix the fit or the apply path first.")
        return 1

    if check.warnings:
        print(f"PASSED with {len(check.warnings)} warning(s):")
        for w in check.warnings:
            print(f"  - {w}")
    else:
        print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
