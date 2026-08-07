from pathlib import Path
import polars as pl

from src.adp import normalize_name

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLAYER_FEATURES_PATH = PROJECT_ROOT / "data" / "player_features.csv"
KEEPER_HISTORY_PATH = PROJECT_ROOT / "keeper_history.csv"
DRAFTABLE_OUTPUT_PATH = PROJECT_ROOT / "data" / "draftable_players.csv"


def load_keeper_history(path=KEEPER_HISTORY_PATH):
    """
    Reads keeper_history.csv, which Jack fills in by hand on draft
    morning as opposing keepers get announced (same pattern as
    playcaller_history.csv). Two defensive steps, both because this
    file gets hand-edited under time pressure:
      1. Header names get stripped -- the raw file has a stray space
         after every comma (" draft_round_last_year", etc.).
      2. Every column gets cast to string and stripped BEFORE any
         numeric/boolean casting, since a hand-typed " 3" would
         otherwise fail a direct cast to Int64.
    Returns only rows where keeping_this_year is true.
    """
    # infer_schema_length=0 forces every column to read as raw String.
    # Without it, Polars' type inference can silently mis-parse a
    # boolean-looking column with a leading space (" true") as Boolean
    # with a null value instead of falling back to String -- and that
    # null then gets filtered out same as False, with no error raised.
    raw = pl.read_csv(path, infer_schema_length=0)
    raw = raw.rename({c: c.strip() for c in raw.columns})

    if raw.height == 0:
        return raw

    raw = raw.with_columns([
        pl.col(c).str.strip_chars().alias(c)
        for c in raw.columns
    ])

    raw = raw.with_columns([
        pl.col("draft_round_last_year").cast(pl.Int64, strict=False),
        pl.col("time_kept_consecutively").cast(pl.Int64, strict=False),
        pl.col("keeping_this_year").str.to_lowercase().eq("true"),
    ])
    return raw.filter(pl.col("keeping_this_year"))


def compute_keeper_round(keepers):
    """
    Applies the league's keeper cost rule (league_config_lebronjames.json ->
    keeper_rule):
      - draft_round_last_year is null (undrafted / waiver add last
        year) -> free, no round cost
      - time_kept_consecutively >= 1 (already kept at least once in a
        row before this year) -> escalates to a Round 1 cost
      - otherwise -> costs the same round they were drafted last year
    Adds a `keeper_round` column (nullable Int64).
    """
    return keepers.with_columns(
        pl.when(pl.col("draft_round_last_year").is_null())
        .then(pl.lit(None, dtype=pl.Int64))
        .when(pl.col("time_kept_consecutively").fill_null(0) >= 1)
        .then(pl.lit(1, dtype=pl.Int64))
        .otherwise(pl.col("draft_round_last_year"))
        .alias("keeper_round")
    )


def match_keepers_to_players(keepers, player_features):
    """
    Name-matches keeper_history.csv rows to player_id using the same
    normalize_name() scheme as adp.py (lowercase, strip periods/
    apostrophes, drop Jr/Sr/II/III/IV/V). Unlike adp.py, this does NOT
    fall back to a team tiebreaker for ambiguous names -- keeper lists
    are short (well under 50 rows league-wide), so an ambiguous or
    unmatched name just gets printed as a warning for you to eyeball,
    rather than guessed at automatically.
    """
    keepers = keepers.with_columns(
        pl.col("player_name").map_elements(normalize_name, return_dtype=pl.String).alias("name_key")
    )
    players = player_features.select(["player_id", "player_name"]).with_columns(
        pl.col("player_name").map_elements(normalize_name, return_dtype=pl.String).alias("name_key")
    )

    dupes = players.group_by("name_key").agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    if dupes.height > 0:
        print(f"WARNING: {dupes.height} name_key(s) in player_features.csv are ambiguous "
              f"(shared by multiple players) -- double check any keeper matches below:")
        print(dupes)

    matched = keepers.join(players.select(["player_id", "name_key"]), on="name_key", how="left")

    unmatched = matched.filter(pl.col("player_id").is_null())
    if unmatched.height > 0:
        print(f"\nWARNING: {unmatched.height} keeper(s) had no match in player_features.csv "
              f"-- check spelling in keeper_history.csv:")
        print(unmatched.select(["player_name", "kept_by_team"]))

    return matched.filter(pl.col("player_id").is_not_null())


def build_draftable_pool(
    player_features_path=PLAYER_FEATURES_PATH,
    keeper_history_path=KEEPER_HISTORY_PATH,
    output_path=DRAFTABLE_OUTPUT_PATH,
):
    """
    The fast, draft-day-repeatable keeper step. Reads the already-built
    player_features.csv (slow, run once ahead of time) plus
    keeper_history.csv (edited by hand as opposing keepers get
    announced), and writes draftable_players.csv -- the full player
    table with every kept player removed entirely, since they're off
    the board regardless of which round their keeper cost lands in.

    Re-run this any time keeper_history.csv changes. It never touches
    player_features.csv and never calls nflreadpy.
    """
    player_features = pl.read_csv(player_features_path)
    keepers = load_keeper_history(keeper_history_path)

    if keepers.height == 0:
        print("No keepers marked in keeper_history.csv yet -- writing full player pool as draftable.")
        player_features.write_csv(output_path)
        return player_features

    keepers = compute_keeper_round(keepers)
    matched = match_keepers_to_players(keepers, player_features)

    draftable = player_features.join(matched.select("player_id"), on="player_id", how="anti")

    print(f"\n{matched.height} kept player(s) removed from the draft pool:")
    print(
        matched.join(player_features.select(["player_id", "player_name", "position"]), on="player_id")
        .select(["player_name", "position", "draft_round_last_year",
                  "time_kept_consecutively", "keeper_round", "kept_by_team"])
        .sort("keeper_round")
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    draftable.write_csv(output_path)
    print(f"\nWrote {draftable.height} / {player_features.height} draftable players to {output_path}")
    return draftable


if __name__ == "__main__":
    build_draftable_pool()