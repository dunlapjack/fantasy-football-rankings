"""
Phase 9 CP1 -- playcaller position-PPG.

Today `coach_changed` is a binary "did the playcaller change entering this
season." That throws away everything in playcaller_history.csv except the
diff: it can tell you Arizona has a new playcaller, but not that the new one
has fed running backs 24 PPG for four years while the old one fed them 17.

This module builds the lookup table that makes "which playcaller" answerable:
for each (playcaller, role, position), the average points per team game that
position group produced under them, 2021-2025.

Definition note -- the unit here is the POSITION ROOM, not the player. If
Detroit's running backs combined for 400 points over 17 games, that's 23.5
RB PPG for that team-season, regardless of how it split between Gibbs and
Montgomery. That's the right unit for a scheme signal: it measures how much
the offense feeds a position, not how good the individuals were. A per-player
average would just re-measure talent, which the baseline already covers.

Output is descriptive only. Nothing here is shrunk (CP2) and nothing is fit
or joined into the model (CP3). Read the summary before either of those --
if the raw spread across playcallers is small relative to the noise, the
feature is dead on arrival and CP2/CP3 are wasted work.

Run:  python -m src.playcaller_ppg
"""

from pathlib import Path

import polars as pl

from src.features import load_veteran_stats
from src.team_codes import normalize_team_column

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLAYCALLER_PATH = PROJECT_ROOT / "playcaller_history.csv"
DATA_DIR = PROJECT_ROOT / "data"

TEAM_SEASON_OUTPUT = DATA_DIR / "playcaller_team_season_ppg.csv"
AGGREGATE_OUTPUT = DATA_DIR / "playcaller_position_ppg.csv"

HISTORY_SEASONS = [2021, 2022, 2023, 2024, 2025]
UPCOMING_SEASON = 2026
POSITIONS = ["QB", "RB", "WR", "TE"]


def load_playcaller_history():
    """
    Reads playcaller_history.csv defensively.

    infer_schema_length=0 forces every column to string on read. This file is
    hand-maintained and has whitespace-padded values ("Head Coach  ", "True  ")
    -- Polars' automatic boolean inference turns those into silent nulls rather
    than erroring, which is exactly the trap that bit this project before.
    Read raw, strip, then cast deliberately.

    Returns: season, team, playcaller, playcaller_role, changed_from_prior_year
    """
    raw = pl.read_csv(PLAYCALLER_PATH, infer_schema_length=0)

    cleaned = raw.with_columns(
        [pl.col(c).str.strip_chars() for c in raw.columns]
    ).with_columns([
        pl.col("season").cast(pl.Int64),
        (pl.col("changed_from_prior_year").str.to_lowercase() == "true")
        .alias("changed_from_prior_year"),
    ])

    # The file has a blank separator line between each season block (lines 34,
    # 67, 100, 133, 166) purely for readability. Polars reads those as all-null
    # rows. Drop them before validating, or they masquerade as both "missing
    # playcaller" and "duplicate team-season" errors.
    cleaned = cleaned.filter(pl.col("season").is_not_null())

    cleaned = normalize_team_column(cleaned)

    expected = 32
    per_season = (
        cleaned.group_by("season").len().filter(pl.col("len") != expected)
    )
    if per_season.height:
        print(f"WARNING: seasons without exactly {expected} teams:")
        print(per_season.sort("season"))

    null_calls = cleaned.filter(
        pl.col("playcaller").is_null() | (pl.col("playcaller") == "")
    )
    if null_calls.height:
        print(f"WARNING: {null_calls.height} rows with no playcaller name:")
        print(null_calls)

    dupes = (
        cleaned.group_by(["season", "team"])
        .len()
        .filter(pl.col("len") > 1)
        .sort(["season", "team"])
    )
    if dupes.height:
        print(f"WARNING: {dupes.height} duplicate team-season rows "
              f"(a team can only have one playcaller of record per season):")
        print(dupes)

    return cleaned


def compute_team_position_ppg(seasons=HISTORY_SEASONS):
    """
    Position-room points per team game, by team and season.

    Uses load_veteran_stats(), so the points are scored under OUR league
    config (6-pt pass TD, full PPR) rather than nflverse's default -- a
    playcaller who feeds tight ends is worth more in this league than in a
    half-PPR one, and the table should reflect that.

    Games are counted per team-season (distinct weeks the team played), not
    per position. Counting per position would divide by fewer games whenever
    a position group had a zero-snap week, silently inflating its PPG.

    Returns: season, team, position, position_points, position_ppg,
             team_games, n_players
    """
    stats = load_veteran_stats(seasons)

    # nflreadpy renamed recent_team -> team; tolerate either so this doesn't
    # break on a version bump.
    if "team" not in stats.columns:
        if "recent_team" in stats.columns:
            stats = stats.rename({"recent_team": "team"})
        else:
            raise KeyError(
                "No team column in load_player_stats output. "
                f"Available: {stats.columns}"
            )

    stats = normalize_team_column(stats)

    team_games = (
        stats.group_by(["season", "team"])
        .agg(pl.col("week").n_unique().alias("team_games"))
    )

    by_position = (
        stats.filter(pl.col("position").is_in(POSITIONS))
        .group_by(["season", "team", "position"])
        .agg([
            pl.col("fantasy_points").sum().alias("position_points"),
            pl.col("player_id").n_unique().alias("n_players"),
        ])
        .join(team_games, on=["season", "team"], how="left")
        .with_columns(
            (pl.col("position_points") / pl.col("team_games")).alias("position_ppg")
        )
    )

    return by_position.sort(["season", "team", "position"])


def build_team_season_table(seasons=HISTORY_SEASONS):
    """
    Joins position-room PPG to the playcaller of record for that team-season.
    This is the row-level table CP2 will shrink and CP3 will fit against;
    keeping it on disk means neither has to re-derive it.

    Returns: season, team, playcaller, playcaller_role,
             changed_from_prior_year, position, position_ppg, ...
    """
    team_pos = compute_team_position_ppg(seasons)
    history = load_playcaller_history().filter(pl.col("season").is_in(seasons))

    joined = team_pos.join(
        history.select([
            "season", "team", "playcaller", "playcaller_role",
            "changed_from_prior_year",
        ]),
        on=["season", "team"],
        how="left",
    )

    unmatched = (
        joined.filter(pl.col("playcaller").is_null())
        .select(["season", "team"])
        .unique()
        .sort(["season", "team"])
    )
    if unmatched.height:
        print(f"WARNING: {unmatched.height} team-seasons have stats but no "
              f"playcaller_history row -- these drop out of the aggregate:")
        print(unmatched)

    return joined.select([
        "season", "team", "playcaller", "playcaller_role",
        "changed_from_prior_year", "position", "position_ppg",
        "position_points", "team_games", "n_players",
    ]).sort(["playcaller", "position", "season"])


def aggregate_by_playcaller(team_season_table):
    """
    Collapses team-seasons into one row per (playcaller, role, position).

    Role is part of the key on purpose: the plan calls for treating a
    head-coach-as-playcaller as potentially different from an OC, and someone
    promoted mid-window (OC in 2022, HC in 2024) should not have those seasons
    silently pooled. The cost is splitting an already-thin sample, which is
    exactly what CP2's shrinkage exists to handle -- so a two-season split
    into 1+1 is fine here and gets absorbed downstream.

    Returns: playcaller, playcaller_role, position, seasons_n, n_teams,
             first_season, last_season, raw_ppg, ppg_sd
    """
    return (
        team_season_table.filter(pl.col("playcaller").is_not_null())
        .group_by(["playcaller", "playcaller_role", "position"])
        .agg([
            pl.len().alias("seasons_n"),
            pl.col("team").n_unique().alias("n_teams"),
            pl.col("season").min().alias("first_season"),
            pl.col("season").max().alias("last_season"),
            pl.col("position_ppg").mean().alias("raw_ppg"),
            pl.col("position_ppg").std().alias("ppg_sd"),
        ])
        .sort(["position", "raw_ppg"], descending=[False, True])
    )


def league_baselines(team_season_table):
    """
    League mean and spread per position, and per (role, position).

    Two jobs. CP2 needs the league mean as the shrinkage target. CP1 needs the
    standard deviation as a reality check: if between-playcaller spread isn't
    meaningfully larger than within-playcaller season-to-season noise, there's
    no signal here to extract.

    Returns: (by_position, by_role)
    """
    by_position = (
        team_season_table.group_by("position")
        .agg([
            pl.len().alias("team_seasons"),
            pl.col("position_ppg").mean().alias("league_mean_ppg"),
            pl.col("position_ppg").std().alias("league_sd_ppg"),
        ])
        .sort("position")
    )

    by_role = (
        team_season_table.filter(pl.col("playcaller_role").is_not_null())
        .group_by(["playcaller_role", "position"])
        .agg([
            pl.len().alias("team_seasons"),
            pl.col("position_ppg").mean().alias("role_mean_ppg"),
            pl.col("position_ppg").std().alias("role_sd_ppg"),
        ])
        .sort(["position", "playcaller_role"])
    )

    return by_position, by_role


def check_2026_coverage(aggregate, upcoming_season=UPCOMING_SEASON):
    """
    The Phase 9 risk, quantified before it costs anything.

    Every 2026 playcaller with no 2021-25 history is a player pool the feature
    cannot score. If that's four teams, a fallback is a detail. If it's twelve,
    the feature is mostly imputation and CP3 should know that going in.

    Reports coverage three ways, because the fallback ladder should follow it:
    exact (playcaller + role) -> name-only (role changed, e.g. promoted to HC)
    -> none (needs the league/role mean).
    """
    history = load_playcaller_history()
    upcoming = history.filter(pl.col("season") == upcoming_season).select(
        ["team", "playcaller", "playcaller_role"]
    )

    if upcoming.height == 0:
        print(f"No {upcoming_season} rows in playcaller_history.csv -- "
              f"coverage check skipped.")
        return None

    exact_keys = aggregate.select(["playcaller", "playcaller_role"]).unique()
    name_keys = aggregate.select("playcaller").unique().with_columns(
        pl.lit(True).alias("has_name_history")
    )

    coverage = (
        upcoming.join(
            exact_keys.with_columns(pl.lit(True).alias("has_exact_history")),
            on=["playcaller", "playcaller_role"],
            how="left",
        )
        .join(name_keys, on="playcaller", how="left")
        .with_columns([
            pl.col("has_exact_history").fill_null(False),
            pl.col("has_name_history").fill_null(False),
        ])
        .with_columns(
            pl.when(pl.col("has_exact_history")).then(pl.lit("exact"))
            .when(pl.col("has_name_history")).then(pl.lit("name_only_role_changed"))
            .otherwise(pl.lit("no_history"))
            .alias("coverage")
        )
        .sort(["coverage", "team"])
    )

    return coverage


def print_summary(team_season, aggregate, by_position, by_role, coverage):
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_width_chars(160)

    print("\n" + "=" * 72)
    print("PHASE 9 CP1 -- PLAYCALLER POSITION-PPG")
    print("=" * 72)

    print(f"\nTeam-seasons with stats and a playcaller: "
          f"{team_season.filter(pl.col('playcaller').is_not_null()).height // len(POSITIONS)}"
          f"  (expect ~{32 * len(HISTORY_SEASONS)})")
    print(f"Distinct playcaller/role combinations: "
          f"{aggregate.select(['playcaller', 'playcaller_role']).unique().height}")

    print("\n--- League baselines by position ---")
    print("If league_sd_ppg is small, there is little between-team variation to")
    print("attribute to anyone, and this feature cannot help.")
    print(by_position)

    print("\n--- Baselines by role ---")
    print(by_role)

    print("\n--- Sample-size reality check ---")
    seasons_dist = (
        aggregate.filter(pl.col("position") == "RB")
        .group_by("seasons_n")
        .agg(pl.len().alias("n_playcallers"))
        .sort("seasons_n")
    )
    print("Seasons per playcaller/role (RB rows; same shape for every position):")
    print(seasons_dist)
    print("This is the CP2 argument in one table. Most playcallers will sit at")
    print("1-3 seasons, where a raw mean is mostly noise.")

    print("\n--- Top and bottom 8 by position (>= 3 seasons only) ---")
    print("Restricted to 3+ seasons so the extremes shown are not just the")
    print("smallest samples, which is what an unfiltered sort would surface.")
    stable = aggregate.filter(pl.col("seasons_n") >= 3)
    for position in POSITIONS:
        subset = stable.filter(pl.col("position") == position)
        if subset.height == 0:
            print(f"\n{position}: no playcaller with 3+ seasons.")
            continue
        print(f"\n{position} -- {subset.height} playcallers with 3+ seasons")
        print(subset.head(8).select([
            "playcaller", "playcaller_role", "seasons_n", "n_teams",
            "raw_ppg", "ppg_sd",
        ]))
        print(subset.tail(8).select([
            "playcaller", "playcaller_role", "seasons_n", "n_teams",
            "raw_ppg", "ppg_sd",
        ]))

    print("\n--- Within-playcaller vs between-playcaller spread ---")
    print("The honest test of whether 'which playcaller' means anything. If the")
    print("between figure is not clearly larger than the within figure, a")
    print("playcaller's history does not predict their next season and CP3")
    print("will not find significance.")
    spread = (
        stable.group_by("position")
        .agg([
            pl.col("raw_ppg").std().alias("between_playcaller_sd"),
            pl.col("ppg_sd").mean().alias("mean_within_playcaller_sd"),
        ])
        .sort("position")
    )
    print(spread)

    if coverage is not None:
        print(f"\n--- {UPCOMING_SEASON} coverage (the fallback question) ---")
        summary = (
            coverage.group_by("coverage")
            .agg(pl.len().alias("teams"))
            .sort("teams", descending=True)
        )
        print(summary)
        gaps = coverage.filter(pl.col("coverage") != "exact")
        if gaps.height:
            print("\nTeams needing a fallback:")
            print(gaps.select(["team", "playcaller", "playcaller_role", "coverage"]))

    print("\n" + "=" * 72)
    print(f"Wrote {TEAM_SEASON_OUTPUT.name} and {AGGREGATE_OUTPUT.name} to data/")
    print("=" * 72 + "\n")


def main():
    DATA_DIR.mkdir(exist_ok=True)

    team_season = build_team_season_table()
    aggregate = aggregate_by_playcaller(team_season)
    by_position, by_role = league_baselines(team_season)
    coverage = check_2026_coverage(aggregate)

    team_season.write_csv(TEAM_SEASON_OUTPUT)
    aggregate.write_csv(AGGREGATE_OUTPUT)

    print_summary(team_season, aggregate, by_position, by_role, coverage)


if __name__ == "__main__":
    main()
