"""
Phase 15a, step 5 (scratch, not shipped).

Before anything goes in the board, the PUP/NFI numbers have to survive
the objection I would raise against them: n = 11 and n = 8.

`build_board.apply_injury_overrides` currently charges a flat 4 games for
PUP and NFI. That number is not measured -- it is the roster RULE (a
player who opens the season on either list cannot play the first four
games) read as if returning to the list's minimum were the same as
returning to full availability.

Step 4 measured 6.2 and 9.4 instead, on 11 and 8 players. This script
asks whether that survives three things:

  1. A WIDER WINDOW -- 2016 instead of 2018.
  2. A WIDER POPULATION -- the TOP_N cut was one judgement call. Redo it
     three ways and see if the answer moves.
  3. LEAVE-ONE-SEASON-OUT -- if one season carries the whole result, the
     number is an anecdote with a decimal point.

The bar: the estimate has to sit clearly above the flat 4 under every
population definition, and no single season may be responsible for it.
If it wobbles, the honest ship is nothing.
"""
import numpy as np
import polars as pl
import nflreadpy as nfl

SEASONS = list(range(2016, 2026))
OFFENSE = ["QB", "RB", "WR", "TE"]
SEASON_GAMES = {s: (16 if s <= 2020 else 17) for s in range(2015, 2027)}
TOP_N = {"QB": 32, "RB": 60, "WR": 80, "TE": 32}
TARGET_SEASON_GAMES = 17

DESIGNATIONS = {
    "PUP (reserve)": ("RES", "R04"),
    "NFI (reserve)": ("RES", "R05"),
    "IR at week 1": ("RES", "R01"),
}

print("loading...")
stats = nfl.load_player_stats(SEASONS).filter(pl.col("season_type") == "REG")
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
    season_pts.select(["player_id", "season", "season_points", "games_played"])
    .with_columns((pl.col("season") + 1).cast(pl.Int32).alias("season"))
    .rename({"season_points": "prior_points", "games_played": "prior_games"})
)
early = (
    draft.filter(pl.col("round") <= 3)
    .filter(pl.col("gsis_id").is_not_null())
    .select(["gsis_id", "season"]).rename({"gsis_id": "player_id"})
    .with_columns([pl.col("season").cast(pl.Int32), pl.lit(True).alias("early_rookie")])
    .unique()
)

wk1 = (
    wk1.join(prior, on=["player_id", "season"], how="left")
    .join(early, on=["player_id", "season"], how="left")
    .with_columns([pl.col("prior_points").fill_null(-1.0),
                   pl.col("prior_games").fill_null(0),
                   pl.col("early_rookie").fill_null(False)])
    .with_columns(pl.col("prior_points").rank("ordinal", descending=True)
                    .over(["season", "position"]).alias("_rank"))
)

POPULATIONS = {
    "A top-N by prior points (step 4's cut)":
        ((pl.col("prior_points") > 0) &
         (pl.col("_rank") <= pl.col("position").replace_strict(TOP_N))) |
        pl.col("early_rookie"),
    "B played 8+ games last season":
        pl.col("prior_games") >= 8,
    "C any prior NFL production at all":
        pl.col("prior_points") > 0,
}


def summarise(frame, label):
    base = frame.filter(pl.col("status") == "ACT")["avail_share"].mean()
    print(f"\n--- {label} ---")
    print(f"  n={frame.height}   week-1 ACTIVE baseline = "
          f"{base * TARGET_SEASON_GAMES:.1f}/17 games")
    for name, (status, abbr) in DESIGNATIONS.items():
        sub = frame.filter((pl.col("status") == status) &
                           (pl.col("status_description_abbr") == abbr))
        if sub.height < 5:
            print(f"  {name:<16s} n={sub.height:<4d} too thin")
            continue
        share = sub["avail_share"].mean()
        missed = (base - share) * TARGET_SEASON_GAMES
        se = sub["avail_share"].std(ddof=1) / np.sqrt(sub.height) * TARGET_SEASON_GAMES
        # Distance from the flat 4 the board uses today, in SEs.
        z = (missed - 4.0) / se if se > 0 else float("nan")
        verdict = "clears 4.0 by 2+ SE" if z >= 2 else (
            "above 4.0 but inside 2 SE" if missed > 4 else "does NOT clear 4.0")
        print(f"  {name:<16s} n={sub.height:<4d} missed={missed:5.1f}  "
              f"SE={se:4.1f}  z vs 4.0 = {z:+5.2f}   {verdict}")
    return base


for label, expr in POPULATIONS.items():
    summarise(wk1.filter(expr), label)

# ------------------------------------------------------ leave-one-season-out
print("\n\n=== leave-one-season-out, population A ===")
print("if one season carries the result, the number is an anecdote\n")
popA = wk1.filter(POPULATIONS["A top-N by prior points (step 4's cut)"])
for name, (status, abbr) in DESIGNATIONS.items():
    ests = []
    for drop in sorted(popA["season"].unique().to_list()):
        f = popA.filter(pl.col("season") != drop)
        base = f.filter(pl.col("status") == "ACT")["avail_share"].mean()
        sub = f.filter((pl.col("status") == status) &
                       (pl.col("status_description_abbr") == abbr))
        if sub.height < 5:
            continue
        ests.append((base - sub["avail_share"].mean()) * TARGET_SEASON_GAMES)
    if not ests:
        print(f"  {name:<16s} -- not enough rows in any fold")
        continue
    print(f"  {name:<16s} range {min(ests):.1f} to {max(ests):.1f}   "
          f"(spread {max(ests) - min(ests):.1f} games over {len(ests)} folds)"
          f"   {'STABLE' if min(ests) > 4.0 else 'crosses the current default'}")

# --------------------------------------------------------- per-season counts
print("\n=== per-season counts, population A ===")
counts = (
    popA.filter(pl.col("status") == "RES")
    .filter(pl.col("status_description_abbr").is_in(["R04", "R05", "R01"]))
    .group_by(["season", "status_description_abbr"]).len()
    .pivot(on="status_description_abbr", index="season", values="len")
    .sort("season")
)
print(counts)
