"""
Hand-maintained position corrections for nflverse (Phase 13.7).

WHY THIS EXISTS
---------------
The model's universe is every player whose nflverse `position` is QB, RB, WR
or TE -- `features.load_veteran_stats` filters on it before a single stat is
read. That filter is right almost always and silently catastrophic when
nflverse's position is not the position a player is DRAFTED at.

Travis Hunter is the case that found it. nflverse carries him as a CB, so
every one of his 2025 receiving rows was dropped at the source and he never
reached the board at all -- not ranked low, absent. He went in round 6 of both
32-team mocks. A board that cannot express an opinion about a player the room
is drafting is worse than one that rates him badly, because there is nothing
on the sheet to argue with.

The fix has to live upstream of the filter and it cannot live in one place:
`features.py`, `situational.py` and the depth-chart readers each pull their
own frame from nflverse and each apply their own position filter. Hence a
shared helper rather than a patch, and hence a FILE rather than a constant --
the next miscoded player should be a one-line edit by whoever notices, not a
code change.

WHAT IT DOES NOT DO
-------------------
It does not invent players. A name nflverse has never recorded an offensive
snap for cannot be rescued by relabelling him, and this module raises rather
than pretend otherwise (see `apply`'s `strict` argument, and the four names in
`data/mock_boards/` that this file deliberately does not list).

It also does not make the model two-way-aware. Hunter is scored as a WR on his
receiving line, full stop -- same shrinkage, same situational terms, same
everything. That is a deliberate choice over a special case: a one-player
exception to the ranking logic is exactly what the holdout gate exists to
catch, and there is no test that could justify one.

FILE FORMAT -- player_name,position,note
    Travis Hunter,WR,"nflverse lists him CB; drafted as a WR"

`player_name` must match nflverse's display name exactly. A name that matches
nothing raises: a typo that silently did nothing would leave the player
missing from the board, which is the failure this file exists to prevent.
"""
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OVERRIDES_PATH = PROJECT_ROOT / "position_overrides.csv"

VALID_POSITIONS = {"QB", "RB", "WR", "TE"}


def load(path=OVERRIDES_PATH):
    """
    Returns {player_name: position}. Missing or empty file returns {}, which
    makes every call site below a no-op -- the same shape `apply_injury_
    overrides` uses, and for the same reason: an optional hand-maintained
    file must not be able to break a build by not existing.
    """
    path = Path(path)
    if not path.exists():
        return {}

    frame = pl.read_csv(path)
    if frame.height == 0:
        return {}

    missing = {"player_name", "position"} - set(frame.columns)
    if missing:
        raise ValueError(
            f"{path.name} needs columns player_name and position; missing {sorted(missing)}."
        )

    overrides = {}
    for name, position in frame.select(["player_name", "position"]).iter_rows():
        position = str(position).strip().upper()
        if position not in VALID_POSITIONS:
            raise ValueError(
                f"{path.name}: {name!r} is mapped to {position!r}, which the model "
                f"does not rank. Valid: {sorted(VALID_POSITIONS)}."
            )
        overrides[str(name).strip()] = position
    return overrides


def apply(frame, name_column, position_column="position", overrides=None,
          strict=False, label=None):
    """
    Rewrites `position_column` for any row whose `name_column` is listed.

    Call this BEFORE the OFFENSE_POSITIONS filter -- after it, the rows this
    is meant to rescue are already gone.

    `strict` raises when a listed name appears nowhere in the frame. It is on
    for the veteran stats table, which is the one frame where absence means
    the override did nothing and the player silently stays off the board. It
    is OFF for depth charts and roster snapshots, where a listed player
    legitimately may not appear (Hunter has no offensive depth-chart row) and
    raising would turn a hand-maintained note into a build failure.
    """
    overrides = load() if overrides is None else overrides
    if not overrides:
        return frame

    if name_column not in frame.columns or position_column not in frame.columns:
        return frame

    present = set(frame.select(name_column).to_series().to_list())
    missing = [name for name in overrides if name not in present]
    if missing and strict:
        raise ValueError(
            f"position_overrides.csv: {missing} appear nowhere in "
            f"{label or name_column}. nflverse has no rows for that name, so "
            f"relabelling it changes nothing and the player stays off the "
            f"board. Check the spelling against nflverse's display name -- and "
            f"if it is right, the player has no offensive snaps in the window "
            f"and this file is the wrong tool for him."
        )

    applied = {name: pos for name, pos in overrides.items() if name in present}
    if not applied:
        return frame

    mapping = pl.DataFrame({
        name_column: list(applied.keys()),
        "_override_position": list(applied.values()),
    })
    frame = frame.join(mapping, on=name_column, how="left").with_columns(
        pl.coalesce([pl.col("_override_position"), pl.col(position_column)])
          .alias(position_column)
    ).drop("_override_position")

    if label:
        print(f"Position overrides ({label}): " + ", ".join(
            f"{name} -> {pos}" for name, pos in sorted(applied.items())))
    return frame


def selftest():
    """Runs without network. `python -m src.position_overrides`."""
    overrides = {"Travis Hunter": "WR"}
    frame = pl.DataFrame({
        "player_display_name": ["Travis Hunter", "Travis Hunter", "Puka Nacua"],
        "position": ["CB", "CB", "WR"],
        "receiving_yards": [61.0, 44.0, 90.0],
    })
    out = apply(frame, "player_display_name", overrides=overrides, label="selftest")
    assert out.filter(pl.col("player_display_name") == "Travis Hunter")["position"].to_list() == ["WR", "WR"]
    assert out.filter(pl.col("player_display_name") == "Puka Nacua")["position"].to_list() == ["WR"]
    assert out.columns == frame.columns, "override must not add or reorder columns"
    assert out.height == frame.height, "override must not duplicate rows"

    try:
        apply(frame, "player_display_name", overrides={"Nobody At All": "WR"},
              strict=True, label="selftest")
    except ValueError:
        pass
    else:
        raise AssertionError("strict=True must raise on a name that matches nothing")

    assert apply(frame, "player_display_name", overrides={}).equals(frame)
    print("position_overrides selftest: PASSED")


if __name__ == "__main__":
    selftest()
    live = load()
    print(f"{OVERRIDES_PATH.name}: {len(live)} entries" +
          (" -- " + ", ".join(f"{n} -> {p}" for n, p in sorted(live.items())) if live else ""))
