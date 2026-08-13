from pathlib import Path
import requests
import polars as pl
import nflreadpy as nfl

from src.team_codes import normalize_team_column   # <-- this line needs to be there

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# FFC IGNORES `teams` (verified Aug 13, ppr/2026). This constant, and the
# `teams` parameter on fetch_ffc_adp() below, describe a choice the API does
# not actually offer: teams=8 and teams=12 return byte-identical payloads,
# all 256 players. teams=6 is a 400.
#
# So there is ONE FFC feed, and every board reads it. `adp_format` is a real
# lever; `teams` is not. Do not read the 12 below as "we chose 12-team ADP" --
# nothing was chosen, and a future attempt to select an 8-team feed will
# appear to work while changing nothing at all.
#
# The consequence worth remembering is in compute_replacement_ranks(), which
# takes the position mix of the first `skill_picks` of this feed as
# replacement level. For a shallow league that reads a MID-draft mix (pick 112
# is round 9 of a 12-team room) and applies it to a LATE-draft moment (round
# 14 of an 8-team room). That conflation is UNTESTED, not refuted -- the only
# feed that could have tested it does not exist. See Phase 13.8.
FFC_TEAMS = 12
FFC_YEAR = 2026
FFC_FORMAT = "ppr"  # PPR, non-superflex -- matches league_config_lebronjames.json
OFFENSE_POSITIONS = ["QB", "RB", "WR", "TE"]

# ADP variants pulled on every run, and the column suffix each gets.
#
# WHY A SECOND FORMAT EXISTS (Aug 6)
# ----------------------------------
# The 32-team league starts a SUPERFLEX, and `ppr` ADP comes from
# one-quarterback mocks. That is not a small-sample problem, it is the
# wrong question: in a superflex room quarterbacks come off the board
# earlier and roughly twice as deep, so the `ppr` position mix
# understates QB demand by a wide, KNOWN margin. Feeding it to
# compute_replacement_ranks() sets QB replacement far too shallow and
# systematically undervalues every quarterback on that board.
#
# FFC has no `superflex` endpoint. It has `2qb`, which is live for 2026,
# and that is the closest honest proxy -- with one caveat that has to
# stay attached to the number: 2QB REQUIRES two starting quarterbacks
# while superflex merely PERMITS a second, so `2qb` ADP overstates QB
# demand somewhat. The bias runs opposite to `ppr`'s and is much
# smaller. Bracketing the truth between two feeds is worth more than
# picking one and forgetting which way it leans.
#
# Both are attached to every player in one pass so that all boards still
# ship from a single model run (Phase 13 CP4). The board picks its column
# via `adp_format` in the league config; nothing here knows about
# leagues.
ADP_VARIANTS = {
    "ppr": "",       # canonical, unsuffixed -- what every existing board reads
    "2qb": "_2qb",   # superflex proxy
}

# Columns that describe the DRAFT and therefore differ by format.
FORMAT_SPECIFIC_COLUMNS = [
    "adp", "adp_formatted", "times_drafted", "adp_high", "adp_low", "adp_stdev",
]
# Columns that describe the PLAYER and are identical across formats. Taken
# from the canonical variant only, so a second pull can't create a second
# copy of the bye week that later disagrees with the first.
FORMAT_INVARIANT_COLUMNS = ["bye"]

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


def match_adp_to_gsis(ffc_adp, gsis_lookup, adp_cols):
    """
    Resolves one FFC ADP pull to gsis_ids.

    Extracted from attach_adp() unchanged (Aug 6) so that a second ADP
    format reuses the SAME matching rules rather than growing a parallel
    copy of them. The name normalization, the unique-name fast path, and
    the team tiebreak are all subtle enough that two implementations
    would drift, and the drift would show up as one format silently
    matching fewer players than the other.
    """
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

    print(f"  matched {all_matches.height} / {ffc_adp.height} FFC entries to a gsis_id")
    return all_matches


def attach_adp(player_features, variants=None):
    """
    Attaches one ADP column set per format in ADP_VARIANTS.

    The canonical `ppr` variant keeps the unsuffixed column names every
    existing board and chart already reads, so nothing downstream changes
    unless it asks for a suffix. `2qb` arrives alongside as `adp_2qb`,
    `has_adp_2qb`, and so on, for the superflex board to select via its
    config.

    A variant that fails to fetch is WARNED about and skipped rather than
    killing the run -- losing the superflex proxy should not cost you the
    two boards that don't use it. But it is never silently absent: if the
    canonical variant is what failed, that raises, because every board
    depends on it.
    """
    variants = variants or ADP_VARIANTS
    gsis_lookup = build_gsis_lookup()

    result = player_features
    for adp_format, suffix in variants.items():
        print(f"\nADP pull: format={adp_format!r} teams={FFC_TEAMS} year={FFC_YEAR}")
        try:
            ffc_adp = fetch_ffc_adp(adp_format=adp_format)
        except Exception as exc:  # noqa: BLE001 -- network/feed shape, both non-fatal
            if suffix == "":
                raise
            print(f"  WARNING: {adp_format!r} pull failed ({exc}). Skipping.")
            print(f"  Boards configured with adp_format={adp_format!r} will have NO ADP "
                  f"and their replacement levels will fall back to starter slots.")
            continue

        # `bye` only comes along with the canonical pull -- see
        # FORMAT_INVARIANT_COLUMNS.
        wanted = FORMAT_SPECIFIC_COLUMNS + (FORMAT_INVARIANT_COLUMNS if suffix == "" else [])
        matches = match_adp_to_gsis(ffc_adp, gsis_lookup, ["player_id"] + wanted)

        if suffix:
            matches = matches.rename({c: f"{c}{suffix}" for c in wanted})

        result = result.join(matches, on="player_id", how="left")
        result = result.with_columns(
            pl.col(f"adp{suffix}").is_not_null().alias(f"has_adp{suffix}")
        )

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