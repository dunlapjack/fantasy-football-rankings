"""
Phase 15a exploration, step 3 (scratch, not shipped).

Step 2 killed the idea that a player's INJURY HISTORY predicts next
season's missed games. It does not: injury-caused missed games persist
year over year at r = 0.06 to 0.14, and adding injury history on top of
role made held-out RMSE worse, not better.

That result does not touch the other half of the question, and the two
get confused constantly. "Is this player injury-prone?" is unanswerable.
"This player is on the PUP list on September 1 -- how many games does
that historically cost?" is a completely different question, and it is
answerable, because it is a question about a CURRENT designation rather
than a past pattern.

That second question is exactly what injury_overrides.csv is guessing at
by hand right now, with a blank `games_missed` column for most rows.
This script measures the answer from nflverse instead:

    for every player carrying designation X at the start of season T,
    what share of his team's games did he actually go on to play?

Output is a lookup table: designation -> expected games missed, with the
sample size behind each cell so a thin one can be refused.
"""
import numpy as np
import polars as pl
import nflreadpy as nfl

SEASONS = list(range(2018, 2026))
OFFENSE = ["QB", "RB", "WR", "TE"]
SEASON_GAMES = {s: (16 if s <= 2020 else 17) for s in range(2015, 2027)}

print("loading...")
stats = nfl.load_player_stats(SEASONS).filter(pl.col("season_type") == "REG")
rw = nfl.load_rosters_weekly(SEASONS).filter(pl.col("game_type") == "REG")
inj = nfl.load_injuries(SEASONS).filter(pl.col("game_type") == "REG")

games = (
    stats.filter(pl.col("player_id").is_not_null())
    .group_by(["player_id", "season"])
    .agg(pl.col("week").n_unique().alias("games_played"))
    .with_columns(pl.col("season").cast(pl.Int32))
)

print("\n=== status_description_abbr, week 1, offensive players ===")
wk1 = (
    rw.filter(pl.col("week") == 1)
    .filter(pl.col("position").is_in(OFFENSE))
    .filter(pl.col("gsis_id").is_not_null())
    .select(["season", "gsis_id", "position", "full_name", "status",
             "status_description_abbr"])
    .unique(subset=["season", "gsis_id"], keep="first")
    .rename({"gsis_id": "player_id"})
    .with_columns(pl.col("season").cast(pl.Int32))
)
print(wk1["status_description_abbr"].value_counts().sort("count", descending=True))

wk1 = wk1.join(games, on=["player_id", "season"], how="left").with_columns(
    pl.col("games_played").fill_null(0)
).with_columns(
    pl.col("season").replace_strict(SEASON_GAMES, default=17).alias("team_games")
).with_columns(
    (pl.col("games_played") / pl.col("team_games")).clip(0, 1).alias("avail_share")
)

print("\n=== expected availability by week-1 roster designation ===")
tbl = (
    wk1.group_by(["status", "status_description_abbr"])
    .agg([
        pl.len().alias("n"),
        pl.col("avail_share").mean().alias("mean_avail"),
        pl.col("games_played").mean().alias("mean_games"),
        (pl.col("games_played") == 0).mean().alias("share_zero"),
    ])
    .filter(pl.col("n") >= 20)
    .sort("mean_avail", descending=True)
)
print(tbl)

# ------------------------------------------------- week-1 injury report
# The other half of what is knowable at the end of August: the player is
# healthy enough to be on the active roster but is carrying a listed
# injury. Split by body part, because a hamstring and an ACL are not the
# same bet even though the override file writes them the same way.
SOFT = ["hamstring", "groin", "quad", "calf", "hip", "adductor", "abdomen", "oblique"]
KNEE = ["knee", "acl", "mcl", "meniscus"]
ACHILLES = ["achilles"]
FOOT = ["foot", "ankle", "toe", "lisfranc", "heel"]
SHOULDER = ["shoulder", "pec", "clavicle", "collarbone"]
BACK = ["back", "spine", "neck"]
CONCUSSION = ["concussion", "head"]

BUCKETS = {
    "soft_tissue": SOFT, "knee": KNEE, "achilles": ACHILLES, "foot_ankle": FOOT,
    "shoulder": SHOULDER, "back_neck": BACK, "concussion": CONCUSSION,
}


def bucket_expr():
    e = pl.lit("other")
    for name, words in reversed(list(BUCKETS.items())):
        cond = pl.lit(False)
        for w in words:
            cond = cond | pl.col("report_primary_injury").str.to_lowercase() \
                            .str.contains(w, literal=True).fill_null(False)
        e = pl.when(cond).then(pl.lit(name)).otherwise(e)
    return e


inj_wk1 = (
    inj.filter(pl.col("week") == 1)
    .filter(pl.col("position").is_in(OFFENSE))
    .filter(pl.col("gsis_id").is_not_null())
    .with_columns(bucket_expr().alias("injury_bucket"))
    .select(["season", "gsis_id", "injury_bucket", "report_status", "practice_status"])
    .unique(subset=["season", "gsis_id"], keep="first")
    .rename({"gsis_id": "player_id"})
    .with_columns(pl.col("season").cast(pl.Int32))
)

j = wk1.join(inj_wk1, on=["player_id", "season"], how="left").with_columns(
    pl.col("injury_bucket").fill_null("none")
)

print("\n=== expected availability by week-1 injury bucket (active roster only) ===")
active = j.filter(pl.col("status") == "ACT")
print(
    active.group_by("injury_bucket")
    .agg([
        pl.len().alias("n"),
        pl.col("avail_share").mean().alias("mean_avail"),
        pl.col("games_played").mean().alias("mean_games"),
    ])
    .filter(pl.col("n") >= 25)
    .sort("mean_avail", descending=True)
)

print("\n=== ... and by week-1 report status ===")
print(
    active.filter(pl.col("report_status").is_not_null())
    .group_by("report_status")
    .agg([
        pl.len().alias("n"),
        pl.col("avail_share").mean().alias("mean_avail"),
        pl.col("games_played").mean().alias("mean_games"),
    ])
    .filter(pl.col("n") >= 20)
    .sort("mean_avail", descending=True)
)

# ------------------------------------------------------ the comparison
# The number that matters: how much better is the designation table than
# what the board does today, which is "everyone plays every game unless
# hand-typed"?
base = active.filter(pl.col("injury_bucket") == "none")["avail_share"].mean()
print(f"\nreference: a week-1 active player with NO listed injury plays "
      f"{base:.1%} of his team's games")

j.write_csv("data/scratch_designation.csv")
print("wrote data/scratch_designation.csv")
