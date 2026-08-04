"""
Writes a pre-filled skeleton for extending playcaller_history.csv, and
validates the result once it's filled in.

WHY THIS EXISTS RATHER THAN "JUST ADD ROWS"
-------------------------------------------
playcaller_history.csv joins to the rest of the pipeline on
(season, team). Team codes are not stable across the seasons being
added: San Diego became the LA Chargers in 2017, the Rams moved in 2016,
and Oakland became Las Vegas in 2020. Hand-typing a code that nflverse
spells differently produces no error -- `compute_coach_continuity`
returns a left join, so the mismatch shows up as a silently null
`coach_changed` for all ~50 players on that team, in that season, in the
training set. A quiet hole, not a crash.

So the team column is generated FROM nflverse for the exact seasons
requested, normalized through team_codes, rather than typed. Only the
human-knowledge columns are left blank.

USAGE
-----
Generate a skeleton to fill in (writes alongside the real file, never
over it):

    python -m src.make_playcaller_template --seasons 2016 2017 2018 2019 2020

Then fill in `playcaller` and `playcaller_role` in the generated file,
append those rows to playcaller_history.csv, and check the result:

    python -m src.make_playcaller_template --validate

WHAT TO FILL IN
---------------
- `playcaller`      Whoever actually called the offensive plays. Usually
                    the OC, but a head coach who calls his own plays
                    goes here instead -- that is the whole point of the
                    column. Match the naming style already in the file.
- `playcaller_role` "Head Coach" or "Offensive Coordinator".

There is no `changed_from_prior_year` column to fill in any more -- it is
derived from the playcaller column by
situational.load_playcaller_history(). That is still why you research one
season BEFORE the earliest you want to train on: 2017's flag is computed
by comparing against the 2016 row.

Pro Football Reference team pages list coordinator history by season.

MID-SEASON CHANGES
------------------
One row per team-season, so a coordinator fired in week 10 has to
resolve to a single name. Use whoever called plays for most of the
season. The existing 2021-2026 rows follow that convention; matching it
matters more than getting any single ambiguous case right.

Because the flag is derived from the name, a wrong name costs twice --
once for that season, and once for the next season's flag. Spelling
matters for the same reason: "Mike McDaniel" one year and "Mike Mcdaniel"
the next reads as a coaching change.
"""

import argparse
from pathlib import Path

import polars as pl

from src.team_codes import normalize_team_column

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLAYCALLER_PATH = PROJECT_ROOT / "playcaller_history.csv"
TEMPLATE_PATH = PROJECT_ROOT / "playcaller_history_template.csv"

COLUMNS = ["season", "team", "playcaller", "playcaller_role"]

# A season where almost nothing changed, or almost everything did, is
# usually inconsistent name spelling rather than a real churn year.
PLAUSIBLE_CHANGES = (3, 20)

# Most recent season with roster data. The file intentionally carries a
# row for the upcoming season, which cannot be checked against nflverse.
LATEST_PLAYED_SEASON = 2025


def teams_by_season(seasons):
    """
    Teams that actually existed and played in each season, with the team
    code spelled the way the rest of the pipeline will spell it.

    Sourced from weekly rosters because that is what
    backtest.get_team_as_of_season() uses, so the codes are guaranteed
    to match the join they will eventually be part of.
    """
    import nflreadpy as nfl

    rosters = nfl.load_rosters_weekly(seasons=sorted(seasons)).filter(
        pl.col("game_type") == "REG"
    )
    pairs = rosters.select(["season", "team"]).unique().drop_nulls()
    pairs = normalize_team_column(pairs, column="team")
    return pairs.unique().sort(["season", "team"])


def write_template(seasons):
    pairs = teams_by_season(seasons)

    template = pairs.with_columns([
        pl.lit(None, dtype=pl.String).alias("playcaller"),
        pl.lit(None, dtype=pl.String).alias("playcaller_role"),
    ]).select(COLUMNS)

    template.write_csv(TEMPLATE_PATH)

    print(f"Wrote {template.height} blank rows to {TEMPLATE_PATH.name}")
    for row in pairs.group_by("season").len().sort("season").iter_rows(named=True):
        print(f"  {row['season']}: {row['len']} teams")

    existing = pl.read_csv(PLAYCALLER_PATH, infer_schema_length=0)
    existing_min = int(min(int(s) for s in existing.select("season").to_series()))
    print(f"\nplaycaller_history.csv currently starts at {existing_min}.")
    print(f"Filling this in and appending unlocks target seasons "
          f"{min(seasons) + 1} onward in src/backtest.py.")
    print("\nRelocation codes to expect (generated, not typed -- listed so they "
          "don't look like typos):")
    for code in ["SD", "LAC", "STL", "LA", "LAR", "OAK", "LV"]:
        hits = pairs.filter(pl.col("team") == code)
        if hits.height:
            years = sorted(hits.select("season").to_series().to_list())
            print(f"  {code}: {years[0]}-{years[-1]}")


def validate():
    """
    Checks the filled-in file the way the pipeline will consume it.
    Every failure here is a silent null downstream, not an error.
    """
    from src.situational import load_playcaller_history

    history = pl.read_csv(PLAYCALLER_PATH, infer_schema_length=0)
    history = normalize_team_column(history)
    history = history.filter(pl.col("season").is_not_null())

    failures = []
    seasons = sorted({int(s) for s in history.select("season").to_series().to_list()})
    print(f"playcaller_history.csv: {history.height} rows, "
          f"seasons {seasons[0]}-{seasons[-1]}")

    # The file carries a row for the UPCOMING season, which by definition
    # has no roster data yet -- nflreadpy errors rather than returning
    # empty. Coverage is only checkable for seasons that have been played.
    playable = [s for s in seasons if s <= LATEST_PLAYED_SEASON]
    upcoming = [s for s in seasons if s > LATEST_PLAYED_SEASON]
    if upcoming:
        print(f"  (skipping team-coverage check for {upcoming} -- not played yet, "
              f"so there is no roster data to check against)")
    print()

    # 1. Team coverage against nflverse, per season.
    actual = teams_by_season(playable).with_columns(pl.col("season").cast(pl.String))
    history_playable = history.filter(
        pl.col("season").cast(pl.Int64) <= LATEST_PLAYED_SEASON
    )
    merged = actual.join(
        history_playable.select(["season", "team"]).with_columns(pl.lit(True).alias("_have")),
        on=["season", "team"], how="left",
    )
    missing = merged.filter(pl.col("_have").is_null())
    if missing.height:
        failures.append(f"{missing.height} team-seasons missing from the file")
        print(f"  [FAIL] {missing.height} team-seasons missing -- these teams would "
              f"get a null coach_changed for every player:")
        for row in missing.head(40).iter_rows(named=True):
            print(f"      {row['season']} {row['team']}")
    else:
        print("  [PASS] every nflverse team-season has a row")

    # 2. Rows referring to teams that didn't play that season.
    extra = history_playable.select(["season", "team"]).join(
        actual.with_columns(pl.lit(True).alias("_real")),
        on=["season", "team"], how="left",
    ).filter(pl.col("_real").is_null())
    if extra.height:
        failures.append(f"{extra.height} rows reference a team that did not play")
        print(f"  [FAIL] {extra.height} rows reference a nonexistent team-season "
              f"-- likely a relocation code typo:")
        for row in extra.head(40).iter_rows(named=True):
            print(f"      {row['season']} {row['team']}")
    else:
        print("  [PASS] no rows reference a nonexistent team-season")

    # 3. Blank required fields.
    for column in ["playcaller", "playcaller_role"]:
        blanks = history.filter(
            pl.col(column).is_null() | (pl.col(column).str.strip_chars() == "")
        )
        if blanks.height:
            failures.append(f"{blanks.height} rows have a blank {column}")
            print(f"  [FAIL] {blanks.height} rows have a blank {column}")
        else:
            print(f"  [PASS] {column} filled everywhere")

    # 4. Duplicates would silently fan out the join.
    dupes = history.group_by(["season", "team"]).len().filter(pl.col("len") > 1)
    if dupes.height:
        failures.append(f"{dupes.height} duplicated season/team pairs")
        print(f"  [FAIL] {dupes.height} duplicate season/team rows -- these fan out "
              f"the join and double-count players")
    else:
        print("  [PASS] one row per season/team")

    # 5. Report the DERIVED flag rather than checking a typed one.
    derived = load_playcaller_history()
    rates = (
        derived.filter(pl.col("changed_from_prior_year").is_not_null())
        .group_by("season")
        .agg(pl.col("changed_from_prior_year").sum().alias("changed"))
        .sort("season")
    )
    low, high = PLAUSIBLE_CHANGES
    print(f"\n  derived coaching changes per season (expect roughly {low}-14 of 32):")
    for row in rates.iter_rows(named=True):
        n = int(row["changed"])
        odd = n < low or n > high
        print(f"    {row['season']}: {n:2d}"
              + ("   <-- implausible, check name spellings" if odd else ""))
        if odd:
            failures.append(f"{row['season']} has {n} derived coaching changes")
    print(f"    {seasons[0]}: n/a (earliest season, nothing to compare against)")

    print()
    if failures:
        print("VALIDATION FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"ALL CHECKS PASSED -- target seasons {seasons[0] + 1} onward are usable.")
    print("Next: python -m src.backtest --seasons "
          f"{' '.join(str(s) for s in range(seasons[0] + 1, 2026))}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate or validate playcaller_history.csv coverage.")
    parser.add_argument("--seasons", type=int, nargs="+",
                        help="seasons to generate blank rows for")
    parser.add_argument("--validate", action="store_true",
                        help="check the filled-in playcaller_history.csv")
    args = parser.parse_args()

    if args.validate:
        raise SystemExit(validate())
    if not args.seasons:
        parser.error("pass --seasons or --validate")
    write_template(args.seasons)
