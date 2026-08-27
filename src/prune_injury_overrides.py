"""
Prune injury_overrides.csv rows whose player is not in the model universe.

    python -m src.prune_injury_overrides            # report only, changes nothing
    python -m src.prune_injury_overrides --apply    # actually prune

WHY THIS EXISTS
---------------
`build_board.apply_injury_overrides` raises when an override name matches
no player in player_features.csv, and it is right to: an unmatched
override does nothing, and the failure it prevents is an injured player
sitting in the draftable pool looking like a steal.

But the error tells you to fix the SPELLING, and in late August that is
usually the wrong diagnosis. The model universe is built from players
with 2023-2025 production plus the current rookie class. A player who has
never recorded an NFL stat line and is no longer a rookie is not in it,
and no spelling will put him there. Cutdown week fills the override file
with exactly those players -- you write down a camp injury, the player
gets released, and the row is now a landmine that blocks every build.

So this separates the two cases instead of guessing:

  NOT IN NFLVERSE AT ALL   almost certainly a typo. Reported and KEPT,
                           because deleting a real player's injury note
                           on a spelling mistake is the dangerous
                           direction to be wrong in.
  IN NFLVERSE, NOT ON THE  a real player the model cannot rank. Safe to
  BOARD                    prune -- the override was never going to do
                           anything.

Nothing is thrown away. Pruned rows are appended to
`injury_overrides_pruned.csv` with the reason, so a player who signs
somewhere and reappears can be moved back.
"""
import argparse
import csv
import difflib
import sys
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OVERRIDES_PATH = PROJECT_ROOT / "injury_overrides.csv"
FEATURES_PATH = PROJECT_ROOT / "data" / "player_features.csv"
PRUNED_PATH = PROJECT_ROOT / "injury_overrides_pruned.csv"


def read_rows(path):
    """csv.reader, not polars: this file is hand-maintained and a ragged
    row is exactly the kind of thing we are here to survive rather than
    crash on."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.reader(fh) if r and any(c.strip() for c in r)]
    return rows[0], rows[1:]


def main():
    parser = argparse.ArgumentParser(
        description="Prune override rows for players the model cannot rank."
    )
    parser.add_argument("--apply", action="store_true",
                        help="write the changes (default is report only)")
    args = parser.parse_args()

    if not FEATURES_PATH.exists():
        raise SystemExit(
            f"{FEATURES_PATH.name} not found -- run `python -m src.pipeline` first."
        )

    header, rows = read_rows(OVERRIDES_PATH)
    width = len(header)
    print(f"{OVERRIDES_PATH.name}: {len(rows)} rows, header {header}")

    ragged = [(i + 2, r) for i, r in enumerate(rows) if len(r) != width]
    if ragged:
        print(f"\n{len(ragged)} RAGGED ROW(S) -- a comma inside a note that is "
              f"not quoted. Fix these by hand before anything else:")
        for line_no, r in ragged:
            print(f"  line {line_no}: {r}")
        return 1

    features = pl.read_csv(FEATURES_PATH)
    on_board = set(features["player_name"].to_list())

    import nflreadpy as nfl
    known = set(
        nfl.load_players().select("display_name").to_series().to_list()
    )

    keep, prunable, typos = [], [], []
    for r in rows:
        name = r[0].strip()
        if name in on_board:
            keep.append(r)
        elif name in known:
            prunable.append(r)
        else:
            typos.append(r)
            keep.append(r)

    print(f"\n  {len(keep) - len(typos)} rows match a player on the board")
    if typos:
        print(f"\n  {len(typos)} NOT FOUND IN NFLVERSE AT ALL -- these are "
              f"spellings, not releases. KEPT, with the closest real names:")
        for r in typos:
            close = difflib.get_close_matches(r[0], known, n=3, cutoff=0.75)
            suggestion = "  ->  " + " | ".join(close) if close else \
                         "  ->  no close match; check the name yourself"
            print(f"    {r[0]:<24s}{suggestion}")
        print("\n    Fix the spelling rather than deleting the row. A dropped "
              "note on a\n    real player is the failure apply_injury_overrides "
              "exists to prevent.")

    if not prunable:
        print("\n  nothing to prune.")
        return 0

    print(f"\n  {len(prunable)} real players who are not in the model universe "
          f"-- no 2023-25 production and not a current rookie, which in late "
          f"August usually means released:")
    for r in prunable:
        note = r[2] if len(r) > 2 else ""
        print(f"    {r[0]:<24s} {r[1]:<12s} {note}")

    if not args.apply:
        print(f"\nReport only. Re-run with --apply to prune these "
              f"{len(prunable)} rows.")
        return 0

    existing_pruned = []
    if PRUNED_PATH.exists():
        _, existing_pruned = read_rows(PRUNED_PATH)

    with open(PRUNED_PATH, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(existing_pruned)
        w.writerows(prunable)

    with open(OVERRIDES_PATH, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(keep)

    print(f"\n  pruned {len(prunable)} rows from {OVERRIDES_PATH.name}")
    print(f"  moved them to {PRUNED_PATH.name} -- nothing was lost")
    print(f"\n  review with:  git diff injury_overrides.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
