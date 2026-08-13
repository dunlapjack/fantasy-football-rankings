"""
Turn two real 32-team mock drafts into an ADP feed. A 32-team-only tool.

WHY THIS EXISTS
---------------
Every other board here reads Fantasy Football Calculator, whose deepest 2026
2QB data stops around pick 190. A 32-team draft is 384 picks long, so from
round 7 on the FFC feed had nothing to say: every remaining player fell into
`has_adp = false`, got hard-capped below every ADP-bearing player and shaded
pink. The board went dead in the half of the draft it exists to help with.

Two full 32-team mocks cover all twelve rounds, and for this league they are
better evidence than FFC besides -- they are 32-team superflex rooms, not
12-team 2QB rooms rescaled and hoped over.

WHAT IT DOES NOT FIX
--------------------
Coverage is 312 players, not 384. The gap is players the MODEL has never
heard of: Travis Hunter (a CB in nflreadpy's player table), Jarquez Hunter,
Will Howard, Zack Kuntz and about seventy round 9-12 fliers with no NFL
snaps. They were undraftable on this board before this change and they still
are. What changed is that the ~120 real players taken between picks 190 and
384 now have a draft slot instead of a pink row.

THE PIPELINE, IN ONE PASS
-------------------------
    data/mock_boards/*.txt        raw transcription, one line per pick
      -> resolve()                board names -> player_ids
      -> data/mock_picks_32team.csv
      -> average()
      -> data/mock_adp_32team.csv   what build_board.py reads

    python -m src.mock_adp
"""
import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "player_features.csv"
BOARD_DIR = PROJECT_ROOT / "data" / "mock_boards"
PICKS_OUT = PROJECT_ROOT / "data" / "mock_picks_32team.csv"
ADP_OUT = PROJECT_ROOT / "data" / "mock_adp_32team.csv"

TEAMS = 32
# (label, file, rounds). Mock A ran before the league added a fifth bench
# spot, which is why it is one round shorter and has to be rescaled below.
MOCKS = [
    ("A", "mockA_11round_picks.txt", 11),
    ("B", "mockB_12round_picks.txt", 12),
]
SCALE_ROUNDS = 12                       # the league's current length
SCALE_TO = TEAMS * SCALE_ROUNDS         # 384
UNDRAFTED = SCALE_TO + 1                # censoring point, not a real pick
SQRT2 = 2 ** 0.5

# The boards print Sleeper's abbreviations; the model stores nflverse's.
# They agree except for the Rams, and that one disagreement silently
# disabled the team check for nine players -- Nacua, Kyren Williams,
# Stafford and six more fell through to the position fallback and only
# resolved because no same-surname rival happened to exist.
#
# Copied from team_codes rather than imported, because that module pulls
# in polars and this one is deliberately stdlib-only: transcribing a mock
# and checking it should not require the model's environment. `_check_
# team_codes()` below asserts the copy still matches whenever polars IS
# available, so the duplication cannot drift unnoticed.
TEAM_ABBR_FIXES = {"AZ": "ARI", "ARZ": "ARI", "LAR": "LA"}

MODELED = ("QB", "RB", "WR", "TE")
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _check_team_codes():
    """No-op without polars; asserts the copied mapping otherwise."""
    try:
        from src.team_codes import TEAM_ABBR_FIXES as canonical
    except ImportError:
        return
    if TEAM_ABBR_FIXES != canonical:
        raise ValueError(
            f"mock_adp.TEAM_ABBR_FIXES has drifted from team_codes: "
            f"{TEAM_ABBR_FIXES} vs {canonical}. Copy the canonical one over."
        )


def normalize(text):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[.'\-’…]", "", text).lower().strip()


def load_universe(features_path=FEATURES_PATH):
    universe = []
    for row in csv.DictReader(open(features_path)):
        if row["position"] not in MODELED:
            continue
        tokens = [t for t in normalize(row["player_name"]).split() if t not in SUFFIXES]
        universe.append({
            "id": row["player_id"], "name": row["player_name"],
            "pos": row["position"], "team": row["team"],
            "first": tokens[0], "last": "".join(tokens[1:]),
            # Sorting key for ties: whoever the market drafts earlier is who
            # the earlier pick meant. Falls back to production when neither
            # has ADP, which is the case for most same-name pairs.
            "adp": float(row["adp_2qb"] or row["adp"] or 9999),
            "ppg": float(row["fantasy_points_per_game"] or 0),
        })
    return universe


def load_picks(path, rounds):
    picks = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pick, name, pos, team = (line.split("|") + [""])[:4]
        round_number, pick_in_round = (int(x) for x in pick.split("."))
        initial, _, last = name.partition(".")
        picks.append({
            "overall": (round_number - 1) * TEAMS + pick_in_round,
            "rdpk": pick, "board": name, "pos": pos,
            "team": TEAM_ABBR_FIXES.get(team, team),
            "initial": normalize(initial)[:1],
            "last": normalize(last).replace(" ", ""),
        })
    expected = TEAMS * rounds
    if len(picks) != expected:
        raise ValueError(
            f"{Path(path).name} has {len(picks)} picks; a {TEAMS}-team "
            f"{rounds}-round draft has {expected}. A transcription that is "
            f"short or long moves every ADP after the gap."
        )
    return picks


def candidates(pick, universe):
    """
    Board cells render "F. Lastname" and truncate ("T. Henders..."), so
    matching is first-initial + surname PREFIX, narrowed by what the board
    itself prints next to the name.
    """
    def matches(player):
        if not player["first"].startswith(pick["initial"]):
            return False
        universe_last, board_last = player["last"], pick["last"]
        if universe_last == board_last:
            return True
        # Prefix matching exists ONLY to survive the board's truncation, and
        # truncation can only make the board's name SHORTER. So the board's
        # surname may be a prefix of the universe's ("Henders" -> Henderson)
        # and never the reverse. That asymmetry is load-bearing: the
        # symmetric version matched the board's "J. Lovett" to Jeremiyah
        # LOVE, who was already gone at 1.31, and handed Love an ADP of 329.
        # Four characters is the floor, so "Nix" cannot reach Nixon.
        return len(board_last) >= 4 and universe_last.startswith(board_last)

    pool = [p for p in universe if matches(p)]
    if pick["team"]:
        # The team is the strongest signal on the cell: it separates Bijan
        # from Brian Robinson, and it survives a position the model
        # disagrees with (Justin Shorter is a TE in nflreadpy, a WR here).
        same_team = [p for p in pool if p["team"] == pick["team"]]
        if same_team:
            return [p for p in same_team if p["pos"] == pick["pos"]] or same_team
    # Without a team agreeing, POSITION is not negotiable. Dropping this
    # guard let a cell reading "A. Brown / QB" match A.J. Brown, the WR.
    return [p for p in pool if p["pos"] == pick["pos"]]


def _team_confirmed(pick, universe):
    """True when the board's own team matches a candidate's team."""
    return bool(pick["team"]) and any(
        p["team"] == pick["team"] for p in candidates(pick, universe))


def resolve(picks, label, universe):
    """
    Board names -> player_ids, in pick order.

    Order matters: a player already taken cannot be taken again, so walking
    the draft forwards makes the duplicate-surname case resolve itself. Both
    mocks contain "B. Robinson / RB / ATL" twice; the round-1 one is Bijan
    because Bijan was still on the board, and the round-4 one is Brian.
    """
    used, rows, unmatched = set(), [], []

    # TWO PASSES, and the order is the point. A cell that names a team is
    # strictly better evidence than one that doesn't, and pick order alone
    # gets this wrong: both mocks contain a "T. Scott / WR" with no team a
    # few picks BEFORE "T. Scott / WR / LAR", and first-come handed the
    # only Tyler Scott in the universe (a Ram) to the anonymous one. The
    # team-confirmed cells claim their players first; everything else takes
    # what is left, still in pick order.
    confirmed = [p for p in picks if _team_confirmed(p, universe)]
    rest = [p for p in picks if not _team_confirmed(p, universe)]

    for pick in confirmed + rest:
        pool = candidates(pick, universe)
        # A player already taken in this draft cannot be taken again, and
        # there is no "best guess" worth making when every candidate is
        # gone -- the old `or pool` fallback here silently assigned A.J.
        # Brown a SECOND pick at 11.16, and since a player's ADP is stored
        # per mock, the second write erased the first. He went from pick 27
        # to pick 336 in mock A and showed up in round 7 of the board.
        # An unresolved cell costs one player's ADP; a wrong one corrupts a
        # player who was correctly resolved 300 picks earlier.
        free = [p for p in pool if p["id"] not in used]
        if not free:
            unmatched.append(pick)
            continue
        best = sorted(free, key=lambda p: (p["adp"], -p["ppg"]))[0]
        used.add(best["id"])
        rows.append({
            "mock": label, "overall": pick["overall"], "rdpk": pick["rdpk"],
            "board_name": pick["board"], "board_pos": pick["pos"],
            "board_team": pick["team"], "player_id": best["id"],
            "player_name": best["name"], "univ_pos": best["pos"],
            "univ_team": best["team"],
        })
    rows.sort(key=lambda r: r["overall"])
    unmatched.sort(key=lambda p: p["overall"])
    return rows, unmatched


def audit(resolved_rows, raise_on_duplicate=True):
    """
    Checks the resolution can't have corrupted anyone, and returns the
    disagreements a human should eyeball.

    THE CHECK THAT WAS MISSING. `average()` stores one pick per player per
    mock in a dict, so a player resolved TWICE in one mock loses his first
    pick silently -- the second write wins and no count changes. The pick
    totals still summed to 352 and 384, the position counts still matched
    the live boards, and A.J. Brown still moved from pick 27 to pick 336
    with every checksum green. Jack caught it by reading the board.

    Nothing about pick COUNTS can catch this, because no pick is lost --
    one player is credited with two of them. The invariant that does catch
    it is the one the draft itself enforces: within a mock, a player
    appears at most once.
    """
    by_player = {}
    duplicates = []
    for row in resolved_rows:
        key = (row["mock"], row["player_id"])
        if key in by_player:
            duplicates.append((row["mock"], row["player_name"],
                               by_player[key]["rdpk"], by_player[key]["board_name"],
                               row["rdpk"], row["board_name"]))
        else:
            by_player[key] = row

    if duplicates and raise_on_duplicate:
        detail = "\n".join(
            f"    mock {m}: {name} resolved at BOTH {p1} ({n1}) and {p2} ({n2})"
            for m, name, p1, n1, p2, n2 in duplicates)
        raise ValueError(
            "mock_adp: a player was resolved more than once inside a single "
            "mock. One of the two cells is a different player the matcher "
            "could not see, and whichever pick came second has already "
            "overwritten the first:\n" + detail
        )
    return duplicates


def disagreements(adp_rows, threshold=96):
    """
    Players whose two mocks disagree by more than `threshold` picks (three
    rounds). Not an error -- two rooms genuinely differ, and a player taken
    in only one mock is censored at 385 by design -- but a mis-resolution
    shows up here first, because a wrong name is usually wrong by hundreds
    of picks rather than dozens.
    """
    flagged = []
    for row in adp_rows:
        if row["times_drafted"] != 2:
            continue
        gap = abs(row["mock_a_pick"] - row["mock_b_pick"])
        if gap > threshold:
            flagged.append((gap, row))
    return sorted(flagged, key=lambda pair: -pair[0])


def average(resolved_rows):
    """
    One ADP per player, from up to two observations.

    THE MOCKS ARE DIFFERENT LENGTHS. Averaging raw pick numbers would treat
    mock A's last pick as 32 picks earlier than mock B's last pick when both
    mean "the end of the draft," so A is put on B's 384-pick scale first.

    A PLAYER TAKEN IN ONLY ONE MOCK is not a player with one observation --
    he is a player one room passed on entirely, and in a 32-team draft that
    is real evidence. The missing observation is censored at pick 385, the
    first pick that did not happen. He therefore sorts behind a player taken
    at the same depth in both, and his stdev blows up to say how little is
    known, which widens his Draft Target cushion instead of faking precision
    two drafts cannot support.
    """
    lengths = {label: TEAMS * rounds for label, _, rounds in MOCKS}
    observed, names = {}, {}
    for row in resolved_rows:
        scaled = int(row["overall"]) * SCALE_TO / lengths[row["mock"]]
        observed.setdefault(row["player_id"], {})[row["mock"]] = scaled
        names[row["player_id"]] = row["player_name"]

    rows = []
    for player_id, picks in observed.items():
        seen = sorted(picks.values())
        padded = seen + [UNDRAFTED] * (2 - len(seen))
        rows.append({
            "player_id": player_id,
            "player_name": names[player_id],
            "adp": round(sum(padded) / 2, 1),
            "adp_high": round(min(seen)),        # earliest he actually went
            "adp_low": round(max(padded)),       # latest, or the censor point
            "adp_stdev": round(abs(padded[0] - padded[1]) / SQRT2, 1),
            "times_drafted": len(seen),
            "mock_a_pick": round(picks["A"]) if "A" in picks else "",
            "mock_b_pick": round(picks["B"]) if "B" in picks else "",
        })
    rows.sort(key=lambda r: r["adp"])
    return rows


def write(rows, path, fieldnames=None):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build():
    _check_team_codes()
    universe = load_universe()
    resolved, report = [], []
    for label, filename, rounds in MOCKS:
        picks = load_picks(BOARD_DIR / filename, rounds)
        counts = Counter(p["pos"] for p in picks)
        rows, unmatched = resolve(picks, label, universe)
        resolved += rows
        report.append((label, rounds, len(picks), counts, len(rows), unmatched))

    audit(resolved)
    write(resolved, PICKS_OUT)
    adp = average(resolved)
    write(adp, ADP_OUT)
    return resolved, adp, report


if __name__ == "__main__":
    resolved, adp, report = build()

    for label, rounds, total, counts, matched, unmatched in report:
        mix = "  ".join(f"{p} {counts[p]}" for p in MODELED)
        print(f"mock {label}: {rounds} rounds, {total} picks  ({mix})")
        print(f"         resolved {matched}, unresolved {len(unmatched)} "
              f"(deepest resolved pick "
              f"{max(int(r['overall']) for r in resolved if r['mock'] == label)})")
        named = [p for p in unmatched if p["team"]]
        if named:
            print("         not in the model's universe: "
                  + ", ".join(f"{p['board']} ({p['team']})" for p in named))

    flagged = disagreements(adp)
    if flagged:
        print(f"\nthe two rooms disagree by 3+ rounds on {len(flagged)} players. "
              f"Not an error -- check the names, not the gaps:")
        for gap, row in flagged[:12]:
            print(f"  {gap:>4} picks  {row['player_name']:24s} "
                  f"A={row['mock_a_pick']:>4}  B={row['mock_b_pick']:>4}")

    both = sum(1 for r in adp if r["times_drafted"] == 2)
    print(f"\n{ADP_OUT.name}: {len(adp)} players, {both} in both mocks, "
          f"{len(adp) - both} in one.")
    print(f"Coverage runs to pick {adp[-1]['adp']:.0f} of {SCALE_TO} "
          f"-- {len(adp) / TEAMS:.1f} rounds deep, against 5.9 from the FFC feed.")
    print("\ntop 12:")
    for row in adp[:12]:
        print(f"  {row['adp']:6.1f}  {row['player_name']:24s}"
              f"  A={row['mock_a_pick'] or '-':>4}  B={row['mock_b_pick'] or '-':>4}"
              f"  sd={row['adp_stdev']}")
