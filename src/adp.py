from pathlib import Path
import requests
import polars as pl
import nflreadpy as nfl

from src.team_codes import normalize_team_column   # <-- this line needs to be there

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FFC_TEAMS = 12
FFC_YEAR = 2026
FFC_FORMAT = "ppr"  # PPR, non-superflex -- matches league_config.json
OFFENSE_POSITIONS = ["QB", "RB", "WR", "TE"]

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name):
    """
    Lowercases, strips periods/apostrophes, and drops trailing suffixes
    (Jr, Sr, II, III, IV, V) so names compare cleanly across sources that
    format suffixes inconsistently (e.g. FFC's "James Cook III" vs
    nflreadpy's "James Cook").
    """
    cleaned = name.lower().replace(".", "").replace("'", "").strip()
    parts = [p for p in cleaned.split() if p not in SUFFIXES]
    return " ".join(parts)


def fetch_ffc_adp(teams=FFC_TEAMS, year=FFC_YEAR, adp_format=FFC_FORMAT):
    """
    Pulls PPR, non-superflex ADP from the Fantasy Football Calculator
    REST API -- real average draft position from live human mock drafts.
    (Confirmed as of July 2026: Sleeper's and FantasyCalc's free tiers
    don't return usable ADP data, so this is the sole source.)
    """
    resp = requests.get(
        f"https://fantasyfootballcalculator.com/api/v1/adp/{adp_format}",
        params={"teams": teams, "year": year, "position": "all"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    players = pl.DataFrame(data["players"])
    players = players.filter(pl.col("position").is_in(OFFENSE_POSITIONS))
    players = players.with_columns(
        pl.col("name").map_elements(normalize_name, return_dtype=pl.String).alias("name_key")
    )
    return players.select([
        "name_key", "name", "position", "team", "adp",
        pl.col("adp_formatted"), "times_drafted",
        pl.col("high").alias("adp_high"),
        pl.col("low").alias("adp_low"),
        pl.col("stdev").alias("adp_stdev"),
        "bye",
    ]).rename({"name": "adp_source_name", "team": "adp_source_team"})


def build_gsis_lookup():
    """
    nflreadpy's load_players() gives a gsis_id <-> display_name mapping
    for every player. Matching is name-first: FFC and nflreadpy don't
    always agree on a player's listed `position` (e.g. Travis Hunter is
    WR in FFC's redraft ADP but CB in nflreadpy's player table), so we
    only fall back to team as a tiebreaker when a name_key maps to more
    than one gsis_id (e.g. two different players both named
    "Marvin Harrison").
    """
    players = nfl.load_players().select([
        pl.col("gsis_id").alias("player_id"),
        "display_name",
        "position",
        pl.col("latest_team").alias("team"),
    ])
    players = normalize_team_column(players)
    players = players.with_columns(
        pl.col("display_name").map_elements(normalize_name, return_dtype=pl.String).alias("name_key")
    )
    return players


def attach_adp(player_features):
    ffc_adp = fetch_ffc_adp()
    gsis_lookup = build_gsis_lookup()

    adp_cols = ["player_id", "adp", "adp_formatted", "times_drafted",
                "adp_high", "adp_low", "adp_stdev", "bye"]

    name_counts = gsis_lookup.group_by("name_key").agg(pl.len().alias("n"))
    unique_names = name_counts.filter(pl.col("n") == 1).select("name_key")
    ambiguous_names = name_counts.filter(pl.col("n") > 1).select("name_key")

    # Case 1: name_key is globally unique in nflreadpy -- match on name
    # alone, ignoring position disagreements between sources.
    unique_lookup = gsis_lookup.join(unique_names, on="name_key", how="semi")
    clean_ffc = ffc_adp.join(unique_names, on="name_key", how="semi")
    clean_matches = clean_ffc.join(
        unique_lookup.select(["player_id", "name_key"]), on="name_key", how="inner"
    ).select(adp_cols)

    # Case 2: name_key collides across multiple real people -- use team
    # as a tiebreaker.
    ambiguous_lookup = gsis_lookup.join(ambiguous_names, on="name_key", how="semi")
    ambiguous_ffc = ffc_adp.join(ambiguous_names, on="name_key", how="semi")

    team_matches = ambiguous_ffc.join(
        ambiguous_lookup.select(["player_id", "name_key", "team"]),
        left_on=["name_key", "adp_source_team"],
        right_on=["name_key", "team"],
        how="inner",
    )
    dupe_after_tiebreak = (
        team_matches.group_by("name_key").agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    )
    if dupe_after_tiebreak.height > 0:
        print(f"WARNING: {dupe_after_tiebreak.height} name+team combos still ambiguous, dropping:")
        print(dupe_after_tiebreak)
        team_matches = team_matches.join(dupe_after_tiebreak.select("name_key"), on="name_key", how="anti")
    team_matches = team_matches.select(adp_cols)

    resolved_keys = pl.concat(
        [clean_ffc.select("name_key"), team_matches.join(gsis_lookup.select(["player_id","name_key"]), on="player_id", how="left").select("name_key")],
        how="vertical",
    ).unique()
    unresolved = ffc_adp.join(resolved_keys, on="name_key", how="anti")
    if unresolved.height > 0:
        print(f"\n{unresolved.height} FFC entries had no confident match, dropped:")
        print(unresolved.select(["adp_source_name", "position", "adp_source_team"]))

    all_matches = pl.concat([clean_matches, team_matches], how="vertical")
    all_matches = all_matches.unique(subset=["player_id"], keep="first")

    print(f"\nMatched {all_matches.height} / {ffc_adp.height} FFC ADP entries to a gsis_id")

    result = player_features.join(all_matches, on="player_id", how="left")
    result = result.with_columns(pl.col("adp").is_not_null().alias("has_adp"))
    return result


if __name__ == "__main__":
    existing = pl.read_csv(PROJECT_ROOT / "data" / "player_features.csv")
    updated = attach_adp(existing)

    print("\nSpot check -- top 5 by ADP:")
    print(
        updated.filter(pl.col("has_adp"))
        .sort("adp")
        .select(["player_name", "position", "team", "adp", "adp_formatted"])
        .head(5)
    )

    print(f"\nPlayers with no ADP match: {updated.filter(~pl.col('has_adp')).height} / {updated.height}")