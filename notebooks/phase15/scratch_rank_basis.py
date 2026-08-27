"""
Phase 15e (scratch, not shipped).

THE QUESTION
------------
`build_board.compute_vor` ranks on `adjusted_fantasy_points_per_game` --
a RATE. Expected games feeds Exp Pts and nothing else, and build_board
says why in its own words: "PPG is a RATE, a player who misses games is
not worse per game, and folding availability into the rate corrupts the
one quantity the whole project is fitted to predict."

That reasoning is correct about the RATE. It is not an argument about
what to RANK on, and the two got merged. A receiver projected for 11
games and one projected for 17 rank identically at the same rate, and in
a redraft league you keep the one who plays.

So: which quantity, ranked, gets you more points?

    RATE          adjusted PPG
    EXPECTED      adjusted PPG * E[games]

THE POPULATION IS THE WHOLE POINT
---------------------------------
This cannot be run on backtest_features.csv. That file drops every
player who missed his entire target season -- minimum actual_games_played
is 1, never 0 -- which is precisely the population where rate-only
ranking fails. Testing availability on a sample with no absences would
be like testing a smoke alarm in a room with no fire.

So the frame is rebuilt from scratch: every player with a real baseline
going into season T, including the ones who then played zero games, with
their actual SEASON TOTAL as the target rather than their rate.

HOW IT IS SCORED
----------------
Two ways, because they answer different questions.

  1. Spearman rank correlation against actual season points. Does the
     ordering match reality?
  2. DRAFT UTILITY -- take the top N by each ranking and sum what those
     players actually scored. This is the number that matters: if you
     had drafted by this ranking, how many points would you have got?
     Reported at N = 24, 50, 100, 150.

E[games] is fitted INSIDE each training fold from prior-season
availability and age -- the block that cleared the Phase 15a gate at
+0.0120 against a 1-SE bar of 0.0023. Nothing about the target season
enters it.

    python scratch_rank_basis.py
"""
import numpy as np
import polars as pl
import nflreadpy as nfl

from src import features

TARGET_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
OFFENSE = ["QB", "RB", "WR", "TE"]
SEASON_GAMES = {s: (16 if s <= 2020 else 17) for s in range(2014, 2027)}
TOP_N_REPORT = [24, 50, 100, 150]

# The pool the board actually ranks. A board does not rank 900 players;
# it ranks the ones a draft could plausibly reach.
POOL_SIZE = 200


def season_totals(seasons):
    """Actual season fantasy points, scored under the league config, for
    every player-season -- including nobody, which is the point: a player
    absent here scores zero, and the join below makes that explicit."""
    raw = features.load_veteran_stats(seasons, strict_overrides=False)
    return (
        raw.group_by(["player_id", "season"])
        .agg([
            pl.col("fantasy_points").sum().alias("actual_season_points"),
            pl.col("week").n_unique().alias("actual_games"),
        ])
        .with_columns(pl.col("season").cast(pl.Int32))
    )


def build_frame():
    """One row per (player, target season) for every player with a real
    three-year baseline going in. Zero-game seasons included."""
    rows = []
    totals_cache = {}

    rosters = nfl.load_rosters(TARGET_SEASONS).filter(
        pl.col("position").is_in(OFFENSE)
    ).filter(pl.col("gsis_id").is_not_null()).select(
        ["season", "gsis_id", "birth_date"]
    ).unique(subset=["season", "gsis_id"], keep="first").rename(
        {"gsis_id": "player_id"}
    ).with_columns(pl.col("season").cast(pl.Int32))

    for target in TARGET_SEASONS:
        window = [target - 3, target - 2, target - 1]
        print(f"  building {target} (baseline {window[0]}-{window[2]})...")

        base = features.build_veteran_feature_table(window, strict_overrides=False)
        base = base.select([
            "player_id", "player_name", "position",
            "fantasy_points_per_game", "games_played",
        ]).rename({
            "fantasy_points_per_game": "baseline_ppg",
            "games_played": "baseline_games",
        })

        # Shrinkage, exactly as the live pipeline applies it, so the RATE
        # being tested is the rate the board would actually carry.
        base = base.with_columns(pl.lit(False).alias("is_rookie"))
        base = features.apply_baseline_shrinkage(
            base, exclude=pl.col("is_rookie"),
            value_column="baseline_ppg", games_column="baseline_games",
        )
        base = features.apply_qb_reversion(
            base, value_column="baseline_ppg", games_column="baseline_games",
        )

        for s in window + [target]:
            if s not in totals_cache:
                totals_cache[s] = season_totals([s])

        prior = totals_cache[target - 1].select(
            ["player_id", "actual_games"]
        ).rename({"actual_games": "prior_games"})
        prior2 = totals_cache[target - 2].select(
            ["player_id", "actual_games"]
        ).rename({"actual_games": "prior2_games"})
        outcome = totals_cache[target].select(
            ["player_id", "actual_season_points", "actual_games"]
        )

        frame = (
            base.join(prior, on="player_id", how="left")
            .join(prior2, on="player_id", how="left")
            .join(outcome, on="player_id", how="left")
            .with_columns([
                pl.lit(target).cast(pl.Int32).alias("season"),
                pl.col("prior_games").fill_null(0),
                pl.col("prior2_games").fill_null(0),
                # THE LINE THIS WHOLE SCRIPT EXISTS FOR. A player with no
                # outcome row did not play. That is a zero, not a missing
                # value, and treating it as missing is the selection hole.
                pl.col("actual_season_points").fill_null(0.0),
                pl.col("actual_games").fill_null(0),
            ])
        )
        rows.append(frame)

    df = pl.concat(rows, how="diagonal")
    df = df.join(rosters, on=["player_id", "season"], how="left")
    df = df.with_columns(
        (pl.col("season") - pl.col("birth_date").cast(pl.Date, strict=False).dt.year())
        .cast(pl.Float64).alias("age")
    )
    return df.with_columns(pl.col("age").fill_null(pl.col("age").median()))


def fit_expected_games(train, test):
    """E[games] per position from prior-season availability and age.

    This is Phase 15a's block B (+ age), the one that cleared the gate.
    Fitted on the training seasons only and applied to the held-out one.
    """
    pred = np.full(test.height, np.nan)
    for pos in OFFENSE:
        tr = train.filter(pl.col("position") == pos)
        mask = (test["position"] == pos).to_numpy()
        if mask.sum() == 0 or tr.height < 50:
            continue
        cols = ["prior_games", "prior2_games", "age"]
        Xtr = tr.select(cols).to_numpy().astype(float)
        ytr = tr["actual_games"].to_numpy().astype(float)
        Xte = test.filter(pl.col("position") == pos).select(cols).to_numpy().astype(float)
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1.0
        A = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / sd])
        B = np.column_stack([np.ones(len(Xte)), (Xte - mu) / sd])
        reg = np.eye(A.shape[1])
        reg[0, 0] = 0
        beta = np.linalg.solve(A.T @ A + reg, A.T @ ytr)
        pred[mask] = np.clip(B @ beta, 0, 17)
    # A position with too little training data keeps today's behaviour:
    # a full season for everyone.
    return np.where(np.isnan(pred), 17.0, pred)


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    print("building the frame (this walks seven baseline windows)...")
    df = build_frame()
    print(f"\nframe: {df.height} player-seasons "
          f"{df['season'].min()}-{df['season'].max()}")
    zero = (df["actual_games"] == 0).sum()
    print(f"  of which zero-game seasons: {zero} ({zero / df.height:.1%})")
    print("  backtest_features.csv contains NONE of these.\n")

    results = {"RATE": [], "EXPECTED": []}
    utility = {n: {"RATE": [], "EXPECTED": [], "PERFECT": []} for n in TOP_N_REPORT}

    for season in TARGET_SEASONS:
        train = df.filter(pl.col("season") != season)
        test = df.filter(pl.col("season") == season)
        if test.height == 0:
            continue

        exp_games = fit_expected_games(train, test)
        rate = test["baseline_ppg_shrunk"].to_numpy()
        actual = test["actual_season_points"].to_numpy()

        # The pool a board would carry: top POOL_SIZE by the incumbent
        # ranking. Both rankings are then judged on the SAME pool, so the
        # comparison is about ordering, not about who got included.
        pool = np.argsort(-rate)[:POOL_SIZE]
        r_rate = rate[pool]
        r_exp = rate[pool] * exp_games[pool]
        r_actual = actual[pool]

        results["RATE"].append(spearman(r_rate, r_actual))
        results["EXPECTED"].append(spearman(r_exp, r_actual))

        for n in TOP_N_REPORT:
            top_rate = np.argsort(-r_rate)[:n]
            top_exp = np.argsort(-r_exp)[:n]
            top_perfect = np.argsort(-r_actual)[:n]
            utility[n]["RATE"].append(float(r_actual[top_rate].sum()))
            utility[n]["EXPECTED"].append(float(r_actual[top_exp].sum()))
            utility[n]["PERFECT"].append(float(r_actual[top_perfect].sum()))

    print("=== 1. rank correlation with actual season points ===")
    print(f"  {'season':>8s} {'RATE':>8s} {'EXPECTED':>10s} {'diff':>8s}")
    for i, season in enumerate(TARGET_SEASONS):
        a, b = results["RATE"][i], results["EXPECTED"][i]
        print(f"  {season:>8d} {a:>8.4f} {b:>10.4f} {b - a:>+8.4f}")
    ma, mb = np.mean(results["RATE"]), np.mean(results["EXPECTED"])
    wins = sum(b > a for a, b in zip(results["RATE"], results["EXPECTED"]))
    print(f"  {'mean':>8s} {ma:>8.4f} {mb:>10.4f} {mb - ma:>+8.4f}"
          f"   ({wins}/{len(results['RATE'])} seasons)")

    print("\n=== 2. draft utility -- points actually scored by the top N ===")
    print("  (PERFECT is hindsight, the ceiling; it is not achievable)")
    for n in TOP_N_REPORT:
        a = float(np.mean(utility[n]["RATE"]))
        b = float(np.mean(utility[n]["EXPECTED"]))
        p = float(np.mean(utility[n]["PERFECT"]))
        seasons_won = sum(y > x for x, y in
                          zip(utility[n]["RATE"], utility[n]["EXPECTED"]))
        gap_closed = (b - a) / (p - a) if p > a else 0.0
        print(f"  top {n:>3d}: RATE {a:>9.0f}   EXPECTED {b:>9.0f}   "
              f"{b - a:>+7.0f} pts ({(b - a) / a:+.2%})   "
              f"won {seasons_won}/{len(utility[n]['RATE'])}   "
              f"closes {gap_closed:.1%} of the gap to perfect")

    df.write_csv("data/scratch_rank_basis.csv")
    print("\nwrote data/scratch_rank_basis.csv")


if __name__ == "__main__":
    main()
