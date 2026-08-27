"""
Phase 15a exploration (scratch, not shipped).

Question: can next-season GAMES PLAYED be predicted from injury news and
prior injury history, for veterans, using nothing but NFL data?

Step 1 here is not a model. It is the two facts that decide whether a
model is worth building:

  1. How big is the hole? The current veteran training set
     (backtest_features.csv) drops every player who missed the whole
     target season -- the same selection hole playing_time.py found for
     rookies. Measure it.
  2. Does injury history carry ANY year-over-year signal, or is
     availability a coin flip? Measure the raw correlations before
     fitting anything.
"""
import polars as pl
import nflreadpy as nfl
import numpy as np

SEASONS = list(range(2016, 2026))
OFFENSE = ["QB", "RB", "WR", "TE"]

print("loading nflverse tables...")
stats = nfl.load_player_stats(SEASONS).filter(pl.col("season_type") == "REG")
rosters = nfl.load_rosters(SEASONS)
inj = nfl.load_injuries(SEASONS).filter(pl.col("game_type") == "REG")
sched = nfl.load_schedules()

# ---------------------------------------------------------------- games
# Games played = distinct weeks with a stat row. This is the same
# definition features.aggregate_season_stats uses.
games = (
    stats.filter(pl.col("player_id").is_not_null())
    .group_by(["player_id", "season"])
    .agg(pl.col("week").n_unique().alias("games_played"))
)

# ------------------------------------------------------------- universe
# Every offensive player who held a roster spot in season T. This is the
# population the board applies expected_games to, and it INCLUDES the
# players who never took a snap -- the ones the current training set
# silently drops.
uni = (
    rosters.filter(pl.col("position").is_in(OFFENSE))
    .filter(pl.col("gsis_id").is_not_null())
    .select(["season", "gsis_id", "position", "full_name", "team"])
    .unique(subset=["season", "gsis_id"], keep="first")
    .rename({"gsis_id": "player_id"})
)
uni = uni.join(games, on=["player_id", "season"], how="left").with_columns(
    pl.col("games_played").fill_null(0)
)

print(f"\nroster universe: {uni.height} player-seasons")
print(uni.group_by("position").agg(
    pl.len().alias("n"),
    (pl.col("games_played") == 0).mean().alias("share_zero_games"),
    pl.col("games_played").mean().alias("mean_games"),
).sort("position"))

# --------------------------------------------------- THE HOLE, MEASURED
bt = pl.read_csv("data/backtest_features.csv")
print("\n--- selection hole, veteran training set ---")
print(f"backtest_features.csv rows           : {bt.height}")
print(f"min actual_games_played in that file : {bt['actual_games_played'].min()}")
uni_recent = uni.filter(pl.col("season") >= 2017)
print(f"roster universe 2017-25              : {uni_recent.height}")
print(f"  of which zero-game seasons         : {(uni_recent['games_played'] == 0).sum()}"
      f"  ({(uni_recent['games_played'] == 0).mean():.1%})")

# ------------------------------------------------------ injury features
# All computed from season T-1 and earlier, so nothing leaks.
STATUS_OUT = ["Out", "Doubtful"]
SOFT_TISSUE = ["hamstring", "groin", "quad", "calf", "hip flexor", "adductor"]
MAJOR = ["acl", "achilles", "knee", "lisfranc", "fibula", "tibia", "neck",
         "back", "spine", "torn"]


def _has(col, words):
    e = pl.lit(False)
    for w in words:
        e = e | pl.col(col).str.to_lowercase().str.contains(w, literal=True)
    return e.fill_null(False)


inj_season = (
    inj.filter(pl.col("gsis_id").is_not_null())
    .with_columns([
        _has("report_primary_injury", SOFT_TISSUE).alias("_soft"),
        _has("report_primary_injury", MAJOR).alias("_major"),
        pl.col("report_status").is_in(STATUS_OUT).fill_null(False).alias("_out"),
        pl.col("practice_status").str.to_lowercase()
          .str.contains("did not participate").fill_null(False).alias("_dnp"),
    ])
    .group_by(["gsis_id", "season"])
    .agg([
        pl.col("week").n_unique().alias("inj_weeks_listed"),
        pl.col("_out").sum().alias("inj_weeks_out"),
        pl.col("_dnp").sum().alias("inj_weeks_dnp"),
        pl.col("_soft").sum().alias("inj_weeks_soft"),
        pl.col("_major").sum().alias("inj_weeks_major"),
    ])
    .rename({"gsis_id": "player_id"})
)

# ------------------------------------------------- assemble prior-year
games = games.with_columns(pl.col("season").cast(pl.Int32))
uni = uni.with_columns(pl.col("season").cast(pl.Int32))
inj_season = inj_season.with_columns(pl.col("season").cast(pl.Int32))

prior_games = games.with_columns((pl.col("season") + 1).cast(pl.Int32).alias("season")).rename(
    {"games_played": "prior_games"}
)
prior2_games = games.with_columns((pl.col("season") + 2).cast(pl.Int32).alias("season")).rename(
    {"games_played": "prior2_games"}
)
prior_inj = inj_season.with_columns((pl.col("season") + 1).cast(pl.Int32).alias("season"))

df = (
    uni.filter(pl.col("season") >= 2018)
    .join(prior_games, on=["player_id", "season"], how="left")
    .join(prior2_games, on=["player_id", "season"], how="left")
    .join(prior_inj, on=["player_id", "season"], how="left")
)

# A player with no prior-season row was not in the league; drop him here,
# he is the rookie/returnee case and playing_time.py owns it.
df = df.filter(pl.col("prior_games").is_not_null()).with_columns([
    pl.col("inj_weeks_listed").fill_null(0),
    pl.col("inj_weeks_out").fill_null(0),
    pl.col("inj_weeks_dnp").fill_null(0),
    pl.col("inj_weeks_soft").fill_null(0),
    pl.col("inj_weeks_major").fill_null(0),
    pl.col("prior2_games").fill_null(0),
])

season_len = {s: (16 if s <= 2020 else 17) for s in range(2016, 2027)}
df = df.with_columns(
    pl.col("season").replace_strict(season_len, default=17).alias("season_games")
).with_columns([
    (pl.col("games_played") / pl.col("season_games")).clip(0, 1).alias("avail_share"),
    (pl.col("prior_games") /
     pl.col("season").map_elements(lambda s: season_len[s - 1], return_dtype=pl.Int64)
     ).clip(0, 1).alias("prior_avail_share"),
])

print(f"\nmodelling frame: {df.height} player-seasons, {df['season'].min()}-{df['season'].max()}")

# ------------------------------------------------------- raw signal check
print("\n--- correlation with THIS season's availability share ---")
preds = ["prior_avail_share", "prior2_games", "inj_weeks_listed", "inj_weeks_out",
         "inj_weeks_dnp", "inj_weeks_soft", "inj_weeks_major"]
for p in preds:
    for pos in ["ALL"] + OFFENSE:
        sub = df if pos == "ALL" else df.filter(pl.col("position") == pos)
        x = sub[p].cast(pl.Float64).to_numpy()
        y = sub["avail_share"].to_numpy()
        r = np.corrcoef(x, y)[0, 1]
        end = "\n" if pos == "TE" else "   "
        print(f"{p:>20s} {pos:>3s} r={r:+.3f} (n={len(x)})", end=end)

# ---------------------------------- persistence: is "injury prone" real?
print("\n--- year-over-year persistence of games missed ---")
for pos in OFFENSE:
    sub = df.filter(pl.col("position") == pos)
    miss_prior = 1 - sub["prior_avail_share"].to_numpy()
    miss_now = 1 - sub["avail_share"].to_numpy()
    r = np.corrcoef(miss_prior, miss_now)[0, 1]
    print(f"  {pos}: r(missed T-1, missed T) = {r:+.3f}   n={sub.height}")

df.write_csv("data/scratch_availability.csv")
print("\nwrote data/scratch_availability.csv")
