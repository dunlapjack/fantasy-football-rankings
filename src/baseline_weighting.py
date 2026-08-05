"""
Phase 11 A, checkpoints CP1 and CP2.

    python -m src.baseline_weighting            # both reports
    python -m src.baseline_weighting --cp1      # just the quantification
    python -m src.baseline_weighting --cp2      # just the backtest
    python -m src.baseline_weighting --threshold-sweep

WHAT THIS ANSWERS
-----------------
The 3-year baseline weights a season by WHEN it happened and nothing
else. `apply_season_weighting()` gives a 4-game 2024 the same 30% as a
17-game 2024, so an injury season enters a player's projection at full
weight carrying a per-game rate measured while he was hurt, working back,
or splitting a role he had lost.

CP1 measures how much of the draftable pool this touches.
CP2 asks whether any of the candidate fixes actually predicts better.

It changes nothing. `features.DEFAULT_SCHEME` stays "recency" until CP3
picks a winner off these numbers.

READING THE OUTPUT
------------------
The pooled MAE column is the least interesting number on the page and
will look nearly flat across schemes -- most players never had a thin
season, so most rows are identical under every scheme and dilute the
comparison. The AFFECTED subgroup is where the schemes actually differ,
and the paired column is the decisive one: same player, same target
season, difference in absolute error against the incumbent. Paired
differences remove the player-to-player variance that swamps the pooled
means.

Two honesty notes on the design:

1. The primary target filter is `actual_games_played >= MIN_TARGET_GAMES`.
   That conditions on the outcome, which is a real cost: it asks "given
   he plays, do we get his RATE right," and deliberately does not ask
   "will he play." The second question is the whole point of the Exp Pts
   column and of Phase 11 D, and mixing them here would let a scheme that
   simply projects everyone lower win by accident. The unfiltered numbers
   print underneath so the choice is visible rather than buried.

2. Every scheme here is evaluated in-sample in the sense that matters --
   these are the same seasons the weights were fitted on. Phase 13 CP2's
   holdout is still the only test that settles anything. A scheme that
   wins by a hair should not be adopted on that basis; see the decision
   rule printed at the bottom.
"""

import argparse
from pathlib import Path

import polars as pl

from src.backtest import DEFAULT_TARGET_SEASONS
from src.features import (
    MIN_GAMES_THRESHOLD,
    SCHEMES,
    aggregate_season_stats,
    apply_season_weighting,
    load_veteran_stats,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLAYER_FEATURES_PATH = PROJECT_ROOT / "data" / "player_features.csv"

# IMPORTED, not redeclared. This file originally carried its own copy of
# [2021..2025] and that copy was wrong for exactly as long as
# backtest.py's was -- the first CP2 run silently scored five seasons
# while reporting nothing unusual, and the re-run after backtest.py was
# fixed produced byte-identical output, which is what gave it away.
#
# Two constants naming the same thing is one constant and one bug waiting
# for someone to update the other. There is now a single source of truth.
TARGET_SEASONS = DEFAULT_TARGET_SEASONS

# The live window, for CP1.
LIVE_BASELINE_SEASONS = [2023, 2024, 2025]

# A target season needs enough games for `actual_ppg` to mean anything.
# A player who tore an ACL in week 2 of the target season has an actual
# PPG that no baseline scheme could have predicted, and including him
# rewards whichever scheme happens to project lowest.
MIN_TARGET_GAMES = 8

MODELED_POSITIONS = ["QB", "RB", "WR", "TE"]


# ---------------------------------------------------------------------
# CP1 -- how many draftable players carry a thin season?
# ---------------------------------------------------------------------

def report_cp1(top_n=100):
    """
    Quantifies the exposure: among the players you would actually draft,
    how many have a season inside the 3-year window with materially
    reduced games played, and how much baseline weight is currently
    resting on those seasons?

    Reported against two definitions of "top 100" on purpose. The MARKET
    top 100 (by ADP) is the pool you draft from and is independent of the
    model, so it cannot flatter the model. The MODEL top 100 is the pool
    the board recommends. If the two disagree sharply that is itself
    worth seeing.
    """
    print("=" * 78)
    print(f"CP1  --  thin seasons inside the live baseline window "
          f"{LIVE_BASELINE_SEASONS}")
    print("=" * 78)

    season_stats = aggregate_season_stats(load_veteran_stats(LIVE_BASELINE_SEASONS))

    per_player = (
        season_stats
        .group_by("player_id")
        .agg([
            pl.col("player_name").last().alias("player_name"),
            pl.col("position").last().alias("position"),
            pl.len().alias("seasons_present"),
            pl.col("games_played").sum().alias("total_games"),
            pl.col("games_played").min().alias("min_season_games"),
            (pl.col("games_played") < MIN_GAMES_THRESHOLD).sum().alias("thin_seasons"),
        ])
    )

    if not PLAYER_FEATURES_PATH.exists():
        print(f"\n{PLAYER_FEATURES_PATH.name} not found -- run `python -m src.pipeline` "
              f"first. Skipping the top-{top_n} breakdown.")
        return

    board = pl.read_csv(PLAYER_FEATURES_PATH).filter(
        pl.col("position").is_in(MODELED_POSITIONS)
    ).with_columns(
        pl.col("is_rookie").cast(pl.String).str.to_lowercase().eq("true")
    ).filter(~pl.col("is_rookie"))

    pools = {
        f"market top {top_n} (by ADP)":
            board.filter(pl.col("adp").is_not_null()).sort("adp").head(top_n),
        f"model top {top_n} (by adj PPG)":
            board.sort("adjusted_fantasy_points_per_game", descending=True).head(top_n),
    }

    for label, pool in pools.items():
        joined = pool.select(["player_id"]).join(per_player, on="player_id", how="inner")
        if joined.height == 0:
            print(f"\n{label}: no overlap with the stats window.")
            continue

        thin = joined.filter(pl.col("thin_seasons") > 0)
        very_thin = joined.filter(pl.col("min_season_games") < 5)
        partial_window = joined.filter(pl.col("seasons_present") < 3)

        print(f"\n{label}  (n={joined.height} matched)")
        print(f"   with >=1 season under {MIN_GAMES_THRESHOLD} games : "
              f"{thin.height:>3}  ({100 * thin.height / joined.height:.0f}%)")
        print(f"   with >=1 season under 5 games        : "
              f"{very_thin.height:>3}  ({100 * very_thin.height / joined.height:.0f}%)")
        print(f"   with fewer than 3 seasons of history : "
              f"{partial_window.height:>3}  ({100 * partial_window.height / joined.height:.0f}%)")

        if thin.height:
            print(f"\n   The 15 worst-exposed (weight currently resting on a thin season):")
            detail = (
                thin.join(
                    board.select(["player_id", "adjusted_fantasy_points_per_game", "adp"]),
                    on="player_id", how="left",
                )
                .sort("min_season_games")
                .head(15)
            )
            print(f"   {'player':<24}{'pos':<5}{'seasons':>8}{'min gm':>8}"
                  f"{'thin':>6}{'adj PPG':>9}{'ADP':>8}")
            for row in detail.select([
                "player_name", "position", "seasons_present", "min_season_games",
                "thin_seasons", "adjusted_fantasy_points_per_game", "adp",
            ]).iter_rows():
                adp = f"{row[6]:.1f}" if row[6] is not None else "--"
                print(f"   {row[0]:<24}{row[1]:<5}{row[2]:>8}{row[3]:>8}"
                      f"{row[4]:>6}{row[5]:>9.2f}{adp:>8}")

    print(f"\n   Note: `thin` counts seasons the player APPEARED in with fewer than "
          f"{MIN_GAMES_THRESHOLD}\n   games. A season he missed entirely is a different "
          f"problem -- it shows up as\n   `seasons_present < 3`, and the recency scheme "
          f"renormalizes over what is there,\n   which is already the sane behavior.")


# ---------------------------------------------------------------------
# CP2 -- does any candidate scheme predict next season better?
# ---------------------------------------------------------------------

def build_scheme_comparison(target_seasons=None, schemes=None):
    """
    For each target season, builds every candidate baseline from the same
    three prior seasons and joins the player's ACTUAL PPG in the target
    season. One row per player-season-scheme.

    The raw stat pull happens once per season window and is re-weighted
    per scheme rather than reloaded, which is the difference between this
    running in a minute and running in ten.
    """
    if target_seasons is None:
        target_seasons = TARGET_SEASONS
    if schemes is None:
        schemes = SCHEMES

    frames = []
    for target in target_seasons:
        baseline_seasons = [target - 3, target - 2, target - 1]
        print(f"   building {target} (baseline {baseline_seasons[0]}-{baseline_seasons[2]})...")

        season_stats = aggregate_season_stats(load_veteran_stats(baseline_seasons))

        window = (
            season_stats
            .group_by("player_id")
            .agg([
                pl.len().alias("seasons_present"),
                pl.col("games_played").min().alias("min_season_games"),
                (pl.col("games_played") < MIN_GAMES_THRESHOLD).sum().alias("thin_seasons"),
            ])
        )

        actual = (
            aggregate_season_stats(load_veteran_stats([target]))
            .select(["player_id", "fantasy_points_per_game", "games_played"])
            .rename({
                "fantasy_points_per_game": "actual_ppg",
                "games_played": "actual_games_played",
            })
        )

        for scheme in schemes:
            baseline = (
                apply_season_weighting(season_stats, scheme)
                .select(["player_id", "player_name", "position",
                         "fantasy_points_per_game"])
                .rename({"fantasy_points_per_game": "baseline_ppg"})
            )
            frames.append(
                baseline
                .join(window, on="player_id", how="left")
                .join(actual, on="player_id", how="inner")
                .with_columns([
                    pl.lit(target).alias("season"),
                    pl.lit(scheme).alias("scheme"),
                    (pl.col("baseline_ppg") - pl.col("actual_ppg")).abs().alias("abs_error"),
                ])
            )

    return pl.concat(frames, how="vertical")


def summarize(frame, label, incumbent="recency"):
    """
    Prints MAE / RMSE / rank correlation per scheme, plus the paired
    improvement against the incumbent, which is the number that decides.
    """
    if frame.height == 0:
        print(f"\n{label}: no rows.")
        return None

    print(f"\n{label}")
    print(f"   {'scheme':<24}{'n':>7}{'MAE':>9}{'RMSE':>9}{'rho':>8}"
          f"{'paired dMAE':>13}{'SE':>8}")

    base = (
        frame.filter(pl.col("scheme") == incumbent)
        .select(["player_id", "season", pl.col("abs_error").alias("base_error")])
    )

    rows = []
    for scheme in frame.select("scheme").unique().to_series().to_list():
        part = frame.filter(pl.col("scheme") == scheme)
        errors = part.select("abs_error").to_series()
        mae = float(errors.mean())
        rmse = float((errors ** 2).mean() ** 0.5)

        rho = part.select(
            pl.corr("baseline_ppg", "actual_ppg", method="spearman")
        ).item()
        rho = float(rho) if rho is not None else float("nan")

        paired = part.join(base, on=["player_id", "season"], how="inner").with_columns(
            (pl.col("base_error") - pl.col("abs_error")).alias("improvement")
        )
        # Positive = this scheme beats the incumbent. Only rows the scheme
        # actually changes carry information; identical rows contribute a
        # hard zero and would shrink the SE toward a false confidence.
        moved = paired.filter(pl.col("improvement").abs() > 1e-9)
        if moved.height > 1:
            improvement = moved.select("improvement").to_series()
            delta = float(improvement.mean())
            se = float(improvement.std() / (moved.height ** 0.5))
            paired_text = f"{delta:+.4f}"
            se_text = f"{se:.4f}"
        else:
            delta, se = 0.0, 0.0
            paired_text, se_text = "--", "--"

        rows.append((scheme, part.height, mae, rmse, rho, delta, se, moved.height))
        marker = "  <- incumbent" if scheme == incumbent else ""
        print(f"   {scheme:<24}{part.height:>7}{mae:>9.3f}{rmse:>9.3f}{rho:>8.3f}"
              f"{paired_text:>13}{se_text:>8}{marker}")

    print(f"   (paired dMAE is measured only on the rows a scheme actually moves; "
          f"n moved shown below)")
    for scheme, _, _, _, _, _, _, n_moved in rows:
        if scheme != incumbent:
            print(f"      {scheme:<24} moved {n_moved} rows")
    return rows


def report_cp2(target_seasons=None):
    print("\n" + "=" * 78)
    print("CP2  --  do any of the candidate schemes predict next season better?")
    print("=" * 78)

    comparison = build_scheme_comparison(target_seasons)

    playable = comparison.filter(pl.col("actual_games_played") >= MIN_TARGET_GAMES)
    affected = playable.filter(
        (pl.col("thin_seasons") > 0) | (pl.col("seasons_present") < 3)
    )

    summarize(playable, f"ALL players, target season >= {MIN_TARGET_GAMES} games")
    summarize(affected,
              f"AFFECTED only (a thin season or a short window) -- the decisive split")

    for position in MODELED_POSITIONS:
        summarize(affected.filter(pl.col("position") == position),
                  f"AFFECTED, {position} only")

    summarize(comparison, "ALL players, NO target-games filter (secondary, see docstring)")

    output = PROJECT_ROOT / "data" / "baseline_scheme_comparison.csv"
    comparison.write_csv(output)
    print(f"\nWrote per-row detail to {output}")

    print("""
DECISION RULE for CP3 -- write the answer down before reading the numbers:

  Adopt a challenger only if BOTH hold on the AFFECTED subgroup:
    (a) paired dMAE is positive by more than 2 standard errors, and
    (b) it does not go the wrong way at any individual position.

  A scheme that wins pooled but loses at one position is picking up
  something positional, not something about injuries, and belongs in the
  per-position weights instead of the baseline.

  If nothing clears the bar, the finding is that recency-only weighting is
  fine and the phase's premise was wrong. Record that and move to B --
  small-sample confidence is a separate mechanism and is not tested here.""")


def report_threshold_sweep(target_seasons=None):
    """
    MIN_GAMES_THRESHOLD = 8 was picked because it reads as "missed half
    the year," which is a sentence, not evidence. This sweeps it so the
    choice is made on the backtest like everything else.
    """
    print("\n" + "=" * 78)
    print("THRESHOLD SWEEP  --  where should 'too thin' actually sit?")
    print("=" * 78)

    import src.features as features  # noqa: PLC0415

    original = features.MIN_GAMES_THRESHOLD
    try:
        for threshold in (4, 6, 8, 10, 12):
            features.MIN_GAMES_THRESHOLD = threshold
            comparison = build_scheme_comparison(
                target_seasons, schemes=["recency", "discount_thin", "drop_thin"]
            )
            playable = comparison.filter(
                pl.col("actual_games_played") >= MIN_TARGET_GAMES
            )
            summarize(playable, f"threshold = {threshold} games")
    finally:
        features.MIN_GAMES_THRESHOLD = original


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 11 A: quantify and backtest baseline weighting schemes."
    )
    parser.add_argument("--cp1", action="store_true", help="only the CP1 quantification")
    parser.add_argument("--cp2", action="store_true", help="only the CP2 backtest")
    parser.add_argument("--threshold-sweep", action="store_true",
                        help="sweep MIN_GAMES_THRESHOLD")
    parser.add_argument("--seasons", type=int, nargs="+", default=TARGET_SEASONS,
                        help=f"target seasons for CP2 (default: {TARGET_SEASONS})")
    parser.add_argument("--top", type=int, default=100, help="pool size for CP1")
    args = parser.parse_args()

    run_all = not (args.cp1 or args.cp2 or args.threshold_sweep)

    if args.cp1 or run_all:
        report_cp1(args.top)
    if args.cp2 or run_all:
        report_cp2(args.seasons)
    if args.threshold_sweep:
        report_threshold_sweep(args.seasons)
