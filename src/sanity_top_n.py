"""
Phase 13 CP5. Screens the top of every board for the errors a human
reader cannot reliably catch, so the human read can be about football.

WHY BOTH, AND WHY THIS ONE FIRST
--------------------------------
CP5 says "sanity-read the top 60 of each board by hand," and that is the
right instruction -- a person spots "that back retired" and "nobody has
played for that team since 2023" in a way no assertion will. But a person
reading 180 rows across three boards will NOT reliably notice that one
player appears twice, or that a driver string references a feature the
model no longer ships. Those are mechanical and they are exactly what
gets missed at row 47 of 60.

So this file takes the mechanical half. What it deliberately does NOT do
is judge whether a ranking is *right*. Every genuinely surprising thing
this model produces -- a 30-year-old star buried, a workhorse back
ranked below his ADP -- is the model working as designed, and a check
that flagged those would be re-implementing an opinion. Those go in the
JUDGEMENT section as questions for the reader, not failures.

WHAT IS A BUG VERSUS WHAT IS A DISAGREEMENT
-------------------------------------------
    BUG          duplicate rows, nulls, a retired player, a player on a
                 team he left, a driver naming a dead feature, K/DST
                 leaking into a skill board
    DISAGREEMENT the board likes someone the market doesn't, or hates
                 someone it loves

The first list should be empty. The second list is the point of drafting
off a model at all, and its job here is to be SHORT and CHECKABLE -- the
15 biggest gaps against ADP, so a human can ask "do I believe this?"
fifteen times instead of sixty.

USAGE
-----
    python -m src.sanity_top_n
    python -m src.sanity_top_n --top 60 --disagreements 20
"""

import argparse
from pathlib import Path

import polars as pl

from src.build_board import (
    MODELED_POSITIONS,
    load_config,
    prepare_board_frame,
    FEATURES_PATH,
)
from src.ranking import load_rookie_weights, load_situational_weights

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BOARDS = {
    "12-team": PROJECT_ROOT / "league_config_lebronjames.json",
    "6-team": PROJECT_ROOT / "league_config_dunlap.json",
    "32-team SF": PROJECT_ROOT / "league_config_32team.json",
}

DEFAULT_TOP = 60
DEFAULT_DISAGREEMENTS = 15

# Tripwire for the KNOWN nflverse issue the plan records: retired players
# keep a live `latest_team`, so a retiree can walk onto a board looking
# active. The model already prices age -- this is not an age-curve
# opinion, it is an is-this-person-still-playing check.
#
# FIRST VERSION WAS A FLAT 36 AND IT FLAGGED MATTHEW STAFFORD (Aug 6),
# who is 39 and starting. That is the check being wrong, not the board,
# and the fix is not to raise the number until the failure goes away --
# that is the move this whole project exists to avoid. It is to say what
# was actually meant.
#
# TWO CONDITIONS, BOTH REQUIRED:
#
# 1. Old FOR THE POSITION. Quarterback longevity is a fact about
#    football, not a fudge factor: Brady played to 45, Rodgers and
#    Stafford into their forties, while a 34-year-old running back is
#    already an anomaly. One threshold cannot express that.
# 2. NO ADP. This is what actually separates Stafford from an artifact.
#    A retired player carried in by a stale `latest_team` is not in
#    anyone's mock drafts, so FFC has never heard of him. An old player
#    the market is still drafting is simply an old player.
#
# Condition 2 does most of the work; condition 1 keeps the check from
# firing on every un-drafted deep-bench name.
IMPLAUSIBLE_AGE = {"QB": 41, "RB": 33, "WR": 36, "TE": 37}


def screen_board(label, config, top_n, check):
    board, replacement, _ = prepare_board_frame(FEATURES_PATH, config, quiet=True)
    top = board.head(top_n)

    print(f"\n{'=' * 74}")
    print(f"{label}  --  top {top_n}   (replacement {replacement})")
    print(f"{'=' * 74}")

    mix = dict(top.group_by("position").len().iter_rows())
    print("   position mix: " + "  ".join(
        f"{p} {mix.get(p, 0)}" for p in MODELED_POSITIONS
    ))

    # --- mechanical failures -----------------------------------------
    dupes = (
        top.group_by("player_id").len().filter(pl.col("len") > 1)
    )
    check(label, dupes.height == 0, "no duplicate players",
          f"{dupes.height} player_id(s) appear twice")

    unmodeled = top.filter(~pl.col("position").is_in(MODELED_POSITIONS))
    check(label, unmodeled.height == 0, "no unmodeled positions leaked in",
          f"{unmodeled.height} rows")

    for column in ["player_name", "position", "adjusted_fantasy_points_per_game", "vor"]:
        if column not in top.columns:
            continue
        nulls = top.filter(pl.col(column).is_null()).height
        check(label, nulls == 0, f"no null {column}", f"{nulls} null")

    out = top.filter(pl.col("out_for_season"))
    check(label, out.height == 0, "nobody OUT for the season is in the top",
          ", ".join(out.select("player_name").to_series().to_list()))

    if "team" in top.columns:
        teamless = top.filter(pl.col("team").is_null())
        check(label, teamless.height == 0, "everyone has a team",
              ", ".join(teamless.select("player_name").to_series().to_list()))

    if "age" in top.columns:
        limit = pl.coalesce([
            pl.when(pl.col("position") == position).then(pl.lit(float(cutoff)))
            for position, cutoff in IMPLAUSIBLE_AGE.items()
        ] + [pl.lit(99.0)])
        ancient = top.filter((pl.col("age") > limit) & ~pl.col("has_adp"))
        check(
            label, ancient.height == 0,
            "no old player without an ADP (retired-with-live-team tripwire)",
            ", ".join(
                f"{r['player_name']} {r['position']} {r['age']:.0f}"
                for r in ancient.select(
                    ["player_name", "position", "age"]
                ).iter_rows(named=True)
            ),
        )

        # Old but drafted is not an error. Printed anyway, because the
        # model's largest single coefficient is age and a 39-year-old
        # inside a top 60 is worth one deliberate look.
        old_but_drafted = top.filter((pl.col("age") > limit) & pl.col("has_adp"))
        if old_but_drafted.height:
            names = ", ".join(
                f"{r['player_name']} ({r['position']} {r['age']:.0f}, ADP {r['adp']:.0f})"
                for r in old_but_drafted.select(
                    ["player_name", "position", "age", "adp"]
                ).iter_rows(named=True)
            )
            print(f"   [note] old for the position but genuinely drafted: {names}")

    # Driver strings must only name features the model actually ships.
    # A driver referencing a cut feature means the board is explaining
    # itself with a reason that no longer exists.
    shipped = set()
    for spec in load_situational_weights().values():
        shipped |= set(spec.get("weights", {}))
    for spec in load_rookie_weights().values():
        shipped |= set(spec.get("weights", {}))
    retired_words = {
        "usage_trend_share": "role trend",
        "qb_changed": "new QB",
        "trend_missing": "thin usage history",
        "pos_rank": "depth chart",
    }
    stale_phrases = [
        phrase for feature, phrase in retired_words.items() if feature not in shipped
    ]
    if stale_phrases and "value_drivers" in top.columns:
        offenders = top.filter(
            pl.any_horizontal([
                pl.col("value_drivers").str.contains(p, literal=True)
                for p in stale_phrases
            ])
        )
        check(label, offenders.height == 0,
              "no driver cites a feature the model no longer ships",
              f"{offenders.height} rows mention {stale_phrases}")

    return board


def report_disagreements(label, board, top_n, count):
    """
    The biggest board-versus-market gaps. Not a check -- the reading list.
    """
    if "adp" not in board.columns:
        return
    ranked = board.head(top_n * 3).filter(pl.col("has_adp") & pl.col("adp").is_not_null())
    if ranked.height == 0:
        return

    ranked = ranked.with_columns(
        (pl.col("adp").rank("ordinal") - pl.col("rank").cast(pl.Float64)).alias("gap")
    )

    print(f"\n   board LIKES more than the market  (top {count // 2}):")
    for row in ranked.sort("gap", descending=True).head(count // 2).iter_rows(named=True):
        print(f"     #{row['rank']:<4}{row['player_name']:<22}{row['position']:<4}"
              f"ADP {row['adp']:>6.1f}   {row.get('value_drivers') or ''}")

    print(f"\n   board FADES against the market  (top {count // 2}):")
    for row in ranked.sort("gap").head(count // 2).iter_rows(named=True):
        print(f"     #{row['rank']:<4}{row['player_name']:<22}{row['position']:<4}"
              f"ADP {row['adp']:>6.1f}   {row.get('value_drivers') or ''}")


def compare_boards(frames, top_n):
    """
    CP5's actual instruction: the boards must visibly diverge, or the
    league-awareness work did not land.
    """
    print(f"\n\n{'=' * 74}")
    print(f"CROSS-BOARD  --  position mix of each top {top_n}")
    print(f"{'=' * 74}")
    print(f"   {'board':<14}" + "".join(f"{p:>7}" for p in MODELED_POSITIONS))
    mixes = {}
    for label, board in frames.items():
        mix = dict(board.head(top_n).group_by("position").len().iter_rows())
        mixes[label] = mix
        print(f"   {label:<14}" + "".join(
            f"{mix.get(p, 0):>7}" for p in MODELED_POSITIONS
        ))

    if {"12-team", "6-team"} <= set(mixes):
        deep, shallow = mixes["12-team"], mixes["6-team"]
        same = all(deep.get(p, 0) == shallow.get(p, 0) for p in MODELED_POSITIONS)
        print()
        if same:
            print("   *** 12-team and 6-team have IDENTICAL mixes. CP5 says that means")
            print("   the league-awareness work did not land. Investigate before drafting.")
        else:
            print("   12-team and 6-team diverge, as CP5 requires.")


def main():
    parser = argparse.ArgumentParser(description="Phase 13 CP5 mechanical screen.")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP)
    parser.add_argument("--disagreements", type=int, default=DEFAULT_DISAGREEMENTS)
    args = parser.parse_args()

    failures = []

    def check(label, ok, name, detail=""):
        status = "PASS" if ok else "FAIL"
        print(f"   [{status}] {name}" + (f"  --  {detail}" if detail and not ok else ""))
        if not ok:
            failures.append(f"{label}: {name}")

    frames = {}
    for label, path in BOARDS.items():
        if not path.exists():
            print(f"\n{label}: {path.name} missing -- skipped.")
            continue
        board = screen_board(label, load_config(path), args.top, check)
        report_disagreements(label, board, args.top, args.disagreements)
        frames[label] = board

    compare_boards(frames, args.top)

    print(f"\n\n{'=' * 74}")
    if failures:
        print(f"{len(failures)} MECHANICAL FAILURE(S):")
        for failure in failures:
            print(f"   {failure}")
        raise SystemExit(1)

    print("No mechanical problems. What is left is judgement, and it is yours:")
    print("   1. Read the two disagreement lists above. Fifteen names, not sixty.")
    print("      For each: do you believe the board, or the market?")
    print("   2. Anyone you know something the model cannot -- holdout, suspension,")
    print("      camp report, a trade it has not seen. That is what")
    print("      injury_overrides.csv is for.")
    print("   3. Any player whose TEAM looks wrong. The model reads latest_team and")
    print("      nflverse lags on late signings.")
    print("   4. The 32-team board should look nothing like the 6-team one. If they")
    print("      rhyme, something collapsed.")
    print("\nSurprising is not the same as wrong. An old star buried is the age")
    print("coefficient -- the most validated thing in the model -- doing its job.")


if __name__ == "__main__":
    main()
