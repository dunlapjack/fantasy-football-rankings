"""
Phase 15a exploration, step 4 (scratch, not shipped).

Step 3's table is right but it is answering about the wrong people. The
week-1 designations are dominated by fringe players -- 573 practice-squad
PUP cases -- and the board never asks about them. It asks about the
~60 players a year who are draftable AND carrying a designation, which is
exactly the population injury_overrides.csv is hand-maintained for.

So: rebuild the same table restricted to players who were fantasy-relevant
going INTO the season (prior-season top-N at their position, or drafted in
the first three rounds as a rookie), and report the number the override
file actually needs -- expected games missed, denominated in team games.
"""
import numpy as np
import polars as pl
import nflreadpy as nfl

SEASONS = list(range(2018, 2026))
OFFENSE = ["QB", "RB", "WR", "TE"]
SEASON_GAMES = {s: (16 if s <= 2020 else 17) for s in range(2015, 2027)}
TOP_N = {"QB": 32, "RB": 60, "WR": 80, "TE": 32}

stats = nfl.load_player_stats(SEASONS + [2017]).filter(pl.col("season_type") == "REG")
rw = nfl.load_rosters_weekly(SEASONS).filter(pl.col("game_type") == "REG")
draft = nfl.load_draft_picks()

season_pts = (
    stats.filter(pl.col("player_id").is_not_null())
    .group_by(["player_id", "season"])
    .agg(pl.col("week").n_unique().alias("games_played"),
         pl.col("fantasy_points").sum().alias("season_points"))
    .with_columns(pl.col("season").cast(pl.Int32))
)

wk1 = (
    rw.filter(pl.col("week") == 1)
    .filter(pl.col("position").is_in(OFFENSE))
    .filter(pl.col("gsis_id").is_not_null())
    .select(["season", "gsis_id", "position", "full_name", "status",
             "status_description_abbr"])
    .unique(subset=["season", "gsis_id"], keep="first")
    .rename({"gsis_id": "player_id"})
    .with_columns(pl.col("season").cast(pl.Int32))
    .join(season_pts, on=["player_id", "season"], how="left")
    .with_columns([pl.col("games_played").fill_null(0),
                   pl.col("season_points").fill_null(0.0)])
    .with_columns(pl.col("season").replace_strict(SEASON_GAMES, default=17)
                    .alias("team_games"))
    .with_columns((pl.col("games_played") / pl.col("team_games")).clip(0, 1)
                    .alias("avail_share"))
)

prior = (
    season_pts.select(["player_id", "season", "season_points"])
    .with_columns((pl.col("season") + 1).cast(pl.Int32).alias("season"))
    .rename({"season_points": "prior_points"})
)
early = (
    draft.filter(pl.col("round") <= 3)
    .filter(pl.col("gsis_id").is_not_null())
    .select(["gsis_id", "season"])
    .rename({"gsis_id": "player_id"})
    .with_columns([pl.col("season").cast(pl.Int32), pl.lit(True).alias("early_rookie")])
    .unique()
)

wk1 = (
    wk1.join(prior, on=["player_id", "season"], how="left")
    .join(early, on=["player_id", "season"], how="left")
    .with_columns([pl.col("prior_points").fill_null(-1.0),
                   pl.col("early_rookie").fill_null(False)])
)
wk1 = wk1.with_columns(
    pl.col("prior_points").rank("ordinal", descending=True)
      .over(["season", "position"]).alias("_rank")
).with_columns(
    ((pl.col("prior_points") > 0) &
     (pl.col("_rank") <= pl.col("position").replace_strict(TOP_N)) |
     pl.col("early_rookie")).alias("relevant")
)

rel = wk1.filter(pl.col("relevant"))
print(f"fantasy-relevant week-1 population: {rel.height} player-seasons "
      f"({rel.height / len(SEASONS):.0f}/yr)")

# Map the raw nflverse codes onto the categories injury_overrides.csv uses.
CATEGORY = {
    ("RES", "R01"): "IR at week 1",
    ("DEV", "P01"): "practice squad",
    ("RES", "R04"): "PUP (reserve)",
    ("RES", "R05"): "NFI (reserve)",
    ("INA", "A01"): "inactive week 1",
    ("ACT", "A01"): "active, no designation",
}

print("\n=== fantasy-relevant players by week-1 designation ===")
t = (
    rel.group_by(["status", "status_description_abbr"])
    .agg([
        pl.len().alias("n"),
        pl.col("avail_share").mean().alias("mean_avail"),
        pl.col("games_played").mean().alias("mean_games"),
        pl.col("games_played").median().alias("median_games"),
        (pl.col("games_played") == 0).mean().alias("share_zero"),
    ])
    .filter(pl.col("n") >= 8)
    .sort("mean_avail", descending=True)
)
pl.Config.set_tbl_rows(30)
print(t)

base = rel.filter((pl.col("status") == "ACT"))["avail_share"].mean()
season_len_2026 = 17
print(f"\nfantasy-relevant, week-1 ACTIVE baseline: {base:.3f} of team games "
      f"= {base * season_len_2026:.1f} of 17")

print("\n=== what the board should put in injury_overrides.games_missed ===")
for (status, abbr), label in CATEGORY.items():
    sub = rel.filter((pl.col("status") == status) &
                     (pl.col("status_description_abbr") == abbr))
    if sub.height < 8:
        print(f"  {label:<26s} n={sub.height:<4d} -- too thin to ship a number")
        continue
    share = sub["avail_share"].mean()
    missed_vs_base = (base - share) * season_len_2026
    se = sub["avail_share"].std(ddof=1) / np.sqrt(sub.height) * season_len_2026
    print(f"  {label:<26s} n={sub.height:<4d} plays {share * season_len_2026:5.1f}/17"
          f"   -> games_missed vs a clean player = {missed_vs_base:4.1f}  (SE {se:.1f})")

rel.write_csv("data/scratch_designation_relevant.csv")
print("\nwrote data/scratch_designation_relevant.csv")
