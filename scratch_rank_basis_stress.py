"""
Phase 15e, step 2 (scratch, not shipped).

Step 1 said ranking on rate * E[games] beats ranking on rate in 7 of 7
seasons, by +0.27 Spearman and +13% of actual points in the top 100.
That is a big enough result to be suspicious of, and there are three
specific things it could be instead of a real improvement.

  OBJECTION 1 -- "it is just a not-in-the-NFL detector."
      46% of the frame played zero games. If the whole gain is demoting
      people who were never going to be drafted, it will vanish when the
      pool is restricted to players with a real role last season.

  OBJECTION 2 -- "it re-imports the thing 15a disproved."
      E[games] is fitted on prior_games. A star who missed ten games to
      injury looks identical to a bench player who dressed for seven. But
      15a established that injury-caused absence does NOT persist
      (r = 0.06-0.14) while ROLE does (r = 0.60-0.74). Ranking Christian
      McCaffrey down because he got hurt last year is exactly the mistake
      15a says not to make.

      The fix follows from that finding rather than from taste: rebuild
      prior availability with injury weeks ADDED BACK, so it measures the
      role the player held rather than the games his body allowed. A
      player designated Out for six weeks had a starter's role for those
      six weeks.

  OBJECTION 3 -- "it helps on average and hurts where it matters."
      Average utility can improve while the ranking gets worse for exactly
      the players you spend a first-round pick on. Scored separately.

    python scratch_rank_basis_stress.py
"""
import numpy as np
import polars as pl
import nflreadpy as nfl

TARGET_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
OFFENSE = ["QB", "RB", "WR", "TE"]
SEASON_GAMES = {s: (16 if s <= 2020 else 17) for s in range(2014, 2027)}
POOL_SIZE = 200
FRAME = "data/scratch_rank_basis.csv"


def injury_weeks(seasons):
    """Weeks a player was designated Out or Doubtful. 15a's health-only
    measure -- the part of an absence that is NOT role."""
    inj = nfl.load_injuries(seasons).filter(pl.col("game_type") == "REG")
    return (
        inj.filter(pl.col("gsis_id").is_not_null())
        .with_columns(
            pl.col("report_status").is_in(["Out", "Doubtful"]).fill_null(False)
              .alias("_out")
        )
        .group_by(["gsis_id", "season"])
        .agg(pl.col("_out").sum().alias("weeks_out"))
        .rename({"gsis_id": "player_id"})
        .with_columns(pl.col("season").cast(pl.Int32))
    )


def fit_expected_games(train, test, feature_cols):
    pred = np.full(test.height, np.nan)
    for pos in OFFENSE:
        tr = train.filter(pl.col("position") == pos)
        mask = (test["position"] == pos).to_numpy()
        if mask.sum() == 0 or tr.height < 50:
            continue
        Xtr = tr.select(feature_cols).to_numpy().astype(float)
        ytr = tr["actual_games"].to_numpy().astype(float)
        Xte = test.filter(pl.col("position") == pos).select(
            feature_cols).to_numpy().astype(float)
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1.0
        A = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / sd])
        B = np.column_stack([np.ones(len(Xte)), (Xte - mu) / sd])
        reg = np.eye(A.shape[1])
        reg[0, 0] = 0
        beta = np.linalg.solve(A.T @ A + reg, A.T @ ytr)
        pred[mask] = np.clip(B @ beta, 0, 17)
    return np.where(np.isnan(pred), 17.0, pred)


def spearman(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)),
                             np.argsort(np.argsort(b)))[0, 1])


def main():
    df = pl.read_csv(FRAME).with_columns(pl.col("season").cast(pl.Int32))
    print(f"frame: {df.height} player-seasons")

    # ---- OBJECTION 2's fix: role-based prior availability
    inj = injury_weeks(list(range(2015, 2026)))
    prior_inj = inj.with_columns(
        (pl.col("season") + 1).cast(pl.Int32).alias("season")
    ).rename({"weeks_out": "prior_weeks_out"})
    prior2_inj = inj.with_columns(
        (pl.col("season") + 2).cast(pl.Int32).alias("season")
    ).rename({"weeks_out": "prior2_weeks_out"})

    df = (
        df.join(prior_inj, on=["player_id", "season"], how="left")
        .join(prior2_inj, on=["player_id", "season"], how="left")
        .with_columns([pl.col("prior_weeks_out").fill_null(0),
                       pl.col("prior2_weeks_out").fill_null(0)])
    )
    df = df.with_columns([
        # Role = games he was on the field for, PLUS the games his role
        # was his but his body was not. Capped at a full season.
        (pl.col("prior_games") + pl.col("prior_weeks_out")).clip(0, 17)
          .alias("prior_role_games"),
        (pl.col("prior2_games") + pl.col("prior2_weeks_out")).clip(0, 17)
          .alias("prior2_role_games"),
    ])

    print("  how much do the two priors differ?")
    diff = df.filter(pl.col("prior_role_games") > pl.col("prior_games"))
    print(f"    {diff.height} player-seasons where injury weeks are added back "
          f"(mean +{(diff['prior_role_games'] - diff['prior_games']).mean():.1f} games)\n")

    MODELS = {
        "EXPECTED_raw":  ["prior_games", "prior2_games", "age"],
        "EXPECTED_role": ["prior_role_games", "prior2_role_games", "age"],
    }

    POOLS = {
        "all top-200 by rate": None,
        "had a real role last yr (prior_role_games>=8)": pl.col("prior_role_games") >= 8,
        "produced last yr (prior_games>=8)": pl.col("prior_games") >= 8,
    }

    for pool_label, pool_expr in POOLS.items():
        print(f"=== POOL: {pool_label} ===")
        sp = {k: [] for k in ["RATE", *MODELS]}
        util = {k: [] for k in ["RATE", *MODELS, "PERFECT"]}
        elite = {k: [] for k in ["RATE", *MODELS]}

        for season in TARGET_SEASONS:
            train = df.filter(pl.col("season") != season)
            test_full = df.filter(pl.col("season") == season)
            if pool_expr is not None:
                test_full = test_full.filter(pool_expr)
            if test_full.height < 50:
                continue

            rate_full = test_full["baseline_ppg_shrunk"].to_numpy()
            keep = np.argsort(-rate_full)[:POOL_SIZE]
            test = test_full[keep.tolist()]

            rate = test["baseline_ppg_shrunk"].to_numpy()
            actual = test["actual_season_points"].to_numpy()
            n = min(100, test.height)

            rankings = {"RATE": rate}
            for name, cols in MODELS.items():
                eg = fit_expected_games(train, test, cols)
                rankings[name] = rate * eg

            for name, score in rankings.items():
                sp[name].append(spearman(score, actual))
                top = np.argsort(-score)[:n]
                util[name].append(float(actual[top].sum()))
                # OBJECTION 3: the players a first-round pick goes on.
                top24 = np.argsort(-score)[:24]
                elite[name].append(float(actual[top24].sum()))
            util["PERFECT"].append(float(actual[np.argsort(-actual)[:n]].sum()))

        base_sp = np.mean(sp["RATE"])
        base_ut = np.mean(util["RATE"])
        base_el = np.mean(elite["RATE"])
        print(f"  {'ranking':<16s} {'spearman':>9s} {'top-100 pts':>12s} "
              f"{'vs RATE':>9s} {'top-24 pts':>11s} {'vs RATE':>9s}")
        for name in ["RATE", *MODELS]:
            print(f"  {name:<16s} {np.mean(sp[name]):>9.4f} "
                  f"{np.mean(util[name]):>12.0f} "
                  f"{np.mean(util[name]) - base_ut:>+9.0f} "
                  f"{np.mean(elite[name]):>11.0f} "
                  f"{np.mean(elite[name]) - base_el:>+9.0f}")
        for name in MODELS:
            w = sum(b > a for a, b in zip(util["RATE"], util[name]))
            we = sum(b > a for a, b in zip(elite["RATE"], elite[name]))
            print(f"    {name}: top-100 won {w}/{len(util['RATE'])} seasons, "
                  f"top-24 won {we}/{len(elite['RATE'])}")
        print()

    # ------------------------------------------------- the McCaffrey case
    print("=== the case 15a says to worry about ===")
    print("established producers who missed most of last season to injury")
    print("(high baseline rate, prior_games <= 8, but weeks_out >= 4)\n")
    hurt = df.filter(
        (pl.col("baseline_ppg_shrunk") >= 10)
        & (pl.col("prior_games") <= 8)
        & (pl.col("prior_weeks_out") >= 4)
    )
    print(f"  n = {hurt.height} player-seasons")
    print(f"  they went on to play a mean of {hurt['actual_games'].mean():.1f} games")
    others = df.filter(
        (pl.col("baseline_ppg_shrunk") >= 10) & (pl.col("prior_games") >= 14)
    )
    print(f"  comparable healthy producers played {others['actual_games'].mean():.1f}")
    print(f"\n  raw prior would predict them low on {hurt['prior_games'].mean():.1f} "
          f"games; role-adjusted uses {hurt['prior_role_games'].mean():.1f}")
    print(hurt.sort("baseline_ppg_shrunk", descending=True)
          .select(["season", "player_name", "position", "baseline_ppg_shrunk",
                   "prior_games", "prior_weeks_out", "prior_role_games",
                   "actual_games"]).head(10))


if __name__ == "__main__":
    main()
