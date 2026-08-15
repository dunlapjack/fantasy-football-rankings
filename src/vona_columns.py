"""
Phase 14 board columns: the two lookups VONA needs on the clock.

Pure Python on a list of dicts, deliberately. These get called from
build_board.py with `board.to_dicts()` and attached back as pl.Series --
the same pattern build_value_drivers() already uses -- so the logic can be
unit-tested without standing up a polars frame, and so this file can be
checked against a board that already shipped.
"""

MODELED_POSITIONS = ["QB", "RB", "WR", "TE"]


def snake_gaps(teams, slot):
    """
    How many picks pass between your turns in a snake draft, from a given
    1-based slot.

    A snake gives you exactly two gaps, alternating. From an odd-round pick
    you wait 2*(teams - slot) + 1; from an even-round pick, 2*slot - 1.
    Slot 3 of 12 gets 19 and 5. Slot 1 gets 23 and 1 -- the turn.

    Returned smallest first, because the short gap is the one that makes
    most of the board read +0.0 and is therefore the one that most changes
    how the sheet is used.
    """
    long_gap = 2 * (teams - slot) + 1
    short_gap = 2 * slot - 1
    return tuple(sorted({max(1, short_gap), max(1, long_gap)}))


def default_slot(teams):
    """Middle of the room, for a league whose draft order isn't known yet."""
    return (teams + 1) // 2


def compute_ppg_pos_rank(rows, ppg_key="adjusted_fantasy_points_per_game"):
    """
    Rank within position by projected points, best = 1.

    WHY THIS EARNS A COLUMN. The board is sorted by VOR, which is a global
    order. VONA asks a positional question -- "who is the best tight end
    left" -- and answering it off a globally-sorted sheet means filtering
    and scanning on every single pick. This turns that into a glance.

    It is NOT the same as the existing `pos_rank` feature, which is a depth
    chart position on an NFL roster. This one is the model's own ordering.

    Out-for-season players are ranked anyway rather than skipped: they are
    still visible on the sheet, and a blank there would read as missing
    data. They are excluded from being anyone's SURVIVOR in
    compute_wait_cost, which is where it actually matters.
    """
    out = {}
    for pos in MODELED_POSITIONS:
        group = [r for r in rows if r.get("position") == pos
                 and r.get(ppg_key) is not None]
        group.sort(key=lambda r: -float(r[ppg_key]))
        for i, r in enumerate(group, start=1):
            out[id(r)] = i
    return [out.get(id(r)) for r in rows]


def compute_wait_cost(rows, gaps, ppg_key="adjusted_fantasy_points_per_game",
                      last_pick=None):
    """
    VONA, precomputed. For each player and each gap N:

        wait_N(p) = ppg(p) - best ppg at p's position with ADP > adp(p) + N

    Read it as: "if I'm on the clock around where this man goes, and my next
    pick is N picks later, this is what passing on him costs me."

    WHY IT IS ANCHORED TO HIS OWN ADP rather than to a pick number. The
    board does not know which pick you are holding when you look at a row,
    but a row is only a live decision when you are on the clock somewhere
    near that player's ADP -- earlier and he is a reach, later and he is
    gone. So his own ADP is the only defensible anchor, and it makes the
    column mean the same thing on every row.

    WHAT IT CANNOT DO, and this is the honest limit. The survivor is found
    in ADP order, frozen before the draft starts. The moment the room
    deviates -- a run on tight ends, an injury scratch -- the real survivor
    is someone else. The number degrades as the draft goes on and it
    degrades fastest at exactly the positions having a run, which is when
    you most want it. The correction is mechanical and worth knowing:
    if K more players at a position have gone than ADP expected, read the
    survivor K rows further down the position list. That is what the
    "PPG@Pos" column is for.

    Players with no real ADP are skipped entirely -- they have no anchor,
    so there is no pick at which the question is live. Out-for-season
    players cannot be a survivor: nobody can start them, so they are not
    the alternative this measures.

    `last_pick` BLANKS everyone drafted after the end of the draft, and it
    is not tidiness -- it is a correctness fix found by testing this column
    against itself. Because the number is anchored to a player's own ADP,
    a man deep in the tail is measured against the people behind HIM, and
    the people behind him are worthless. Joe Mixon at ADP 187 scored +5.6,
    higher than Jahmyr Gibbs at +5.3, which would read as "Mixon is the
    bigger cliff on the board." He is not. He is standing at the edge of a
    cliff that nobody is ever going to fall off.

    So the column is only ever comparable BETWEEN PLAYERS AT SIMILAR ADP,
    and sorting it globally is meaningless. Blanking past the last pick
    removes the worst of the illusion; the rest is a caveat that has to be
    read, and it is in the sheet's notes block.
    """
    results = {gap: [] for gap in gaps}

    # One sorted list per position, by ADP, of players who could actually be
    # the alternative.
    pool = {}
    for pos in MODELED_POSITIONS:
        pool[pos] = sorted(
            (r for r in rows
             if r.get("position") == pos
             and r.get(ppg_key) is not None
             and r.get("adp") is not None
             and r.get("has_adp")
             and not r.get("out_for_season")),
            key=lambda r: float(r["adp"]))

    for r in rows:
        pos = r.get("position")
        ppg = r.get(ppg_key)
        adp = r.get("adp")
        usable = (pos in pool and ppg is not None and adp is not None
                  and r.get("has_adp")
                  and (last_pick is None or float(adp) <= last_pick))
        for gap in gaps:
            if not usable:
                results[gap].append(None)
                continue
            cutoff = float(adp) + gap
            best = None
            for q in pool[pos]:
                if float(q["adp"]) <= cutoff:
                    continue
                v = float(q[ppg_key])
                if best is None or v > best:
                    best = v
            # Nobody left at the position at all. That is a real state deep
            # in the tail, and reporting it as a huge positive gap would be
            # a lie of scale, so it goes blank.
            results[gap].append(None if best is None
                                else float(ppg) - best)
    return results
