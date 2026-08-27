"""
Phase 15a exploration, step 2 (scratch, not shipped).

Step 1 found that every injury-report feature correlates POSITIVELY with
next-season availability. That is not a finding about health, it is a
finding about role: only players who matter get listed on an injury
report. A practice-squad tight end never appears on one and also never
plays.

So the question has to be asked in two pieces, and they are different
questions:

  ROLE   -- will this player be on the field at all?
  HEALTH -- given a role, how many games does he lose to injury?

This script separates them:

  * `avail_share`      = games played / team games. Role AND health.
                         This is what expected_games actually needs.
  * `inj_missed_share` = share of his team's games he was designated
                         Out/Doubtful for. Health only, near enough.

Then it runs a nested, leave-one-season-out comparison to answer the
thing that decides whether Phase 15a ships anything at all:

    does injury history add predictive power ON TOP OF role?

Decision rule, written before the numbers exist, and copied from
playing_time.run_gate() on purpose: a block of features ships only if it
beats the previous block by more than 1 standard error of the paired
RMSE difference, on held-out seasons, without worsening bias.
"""
import numpy as np
import polars as pl
import nflreadpy as nfl

SEASONS = list(range(2016, 2026))
OFFENSE = ["QB", "RB", "WR", "TE"]
SEASON_GAMES = {s: (16 if s <= 2020 else 17) for s in range(2015, 2027)}

print("loading...")
stats = nfl.load_player_stats(SEASONS).filter(pl.col("season_type") == "REG")
rosters = nfl.load_rosters(SEASONS)
inj = nfl.load_injuries(SEASONS).filter(pl.col("game_type") == "REG")

games = (
    stats.filter(pl.col("player_id").is_not_null())
    .group_by(["player_id", "season"])
    .agg(
        pl.col("week").n_unique().alias("games_played"),
        pl.col("fantasy_points").sum().alias("season_points"),
    )
    .with_columns(pl.col("season").cast(pl.Int32))
)

uni = (
    rosters.filter(pl.col("position").is_in(OFFENSE))
    .filter(pl.col("gsis_id").is_not_null())
    .select(["season", "gsis_id", "position", "full_name"])
    .unique(subset=["season", "gsis_id"], keep="first")
    .rename({"gsis_id": "player_id"})
    .with_columns(pl.col("season").cast(pl.Int32))
    .join(games, on=["player_id", "season"], how="left")
    .with_columns([
        pl.col("games_played").fill_null(0),
        pl.col("season_points").fill_null(0.0),
    ])
)

# ------------------------------------------------------- health target
# Weeks the player was designated Out or Doubtful. This is the closest
# thing nflverse has to "missed this game because he was hurt," and it
# is deliberately NOT "games not played," which counts healthy scratches
# and depth-chart decisions as injuries.
SOFT = ["hamstring", "groin", "quad", "calf", "hip flexor", "adductor", "abdomen"]
MAJOR = ["acl", "achilles", "lisfranc", "fibula", "tibia", "torn", "spine",
         "neck", "knee", "back"]


def _has(col, words):
    e = pl.lit(False)
    for w in words:
        e = e | pl.col(col).str.to_lowercase().str.contains(w, literal=True)
    return e.fill_null(False)


inj_season = (
    inj.filter(pl.col("gsis_id").is_not_null())
    .with_columns([
        pl.col("report_status").is_in(["Out", "Doubtful"]).fill_null(False).alias("_out"),
        pl.col("practice_status").str.to_lowercase()
          .str.contains("did not participate").fill_null(False).alias("_dnp"),
        _has("report_primary_injury", SOFT).alias("_soft"),
        _has("report_primary_injury", MAJOR).alias("_major"),
    ])
    .group_by(["gsis_id", "season"])
    .agg([
        pl.col("week").n_unique().alias("weeks_listed"),
        pl.col("_out").sum().alias("weeks_out"),
        pl.col("_dnp").sum().alias("weeks_dnp"),
        pl.col("_soft").sum().alias("weeks_soft"),
        pl.col("_major").sum().alias("weeks_major"),
    ])
    .rename({"gsis_id": "player_id"})
    .with_columns(pl.col("season").cast(pl.Int32))
)

uni = uni.join(inj_season, on=["player_id", "season"], how="left").with_columns(
    [pl.col(c).fill_null(0) for c in
     ["weeks_listed", "weeks_out", "weeks_dnp", "weeks_soft", "weeks_major"]]
)
uni = uni.with_columns(
    pl.col("season").replace_strict(SEASON_GAMES, default=17).alias("team_games")
).with_columns([
    (pl.col("games_played") / pl.col("team_games")).clip(0, 1).alias("avail_share"),
    (pl.col("weeks_out") / pl.col("team_games")).clip(0, 1).alias("inj_missed_share"),
])

# ------------------------------------------------------ lag the features
lag = uni.select([
    "player_id", "season", "avail_share", "inj_missed_share", "season_points",
    "weeks_listed", "weeks_out", "weeks_dnp", "weeks_soft", "weeks_major",
]).with_columns((pl.col("season") + 1).cast(pl.Int32).alias("season"))
lag = lag.rename({c: f"p1_{c}" for c in lag.columns if c not in ("player_id", "season")})

lag2 = uni.select(["player_id", "season", "avail_share", "inj_missed_share"]).with_columns(
    (pl.col("season") + 2).cast(pl.Int32).alias("season")
)
lag2 = lag2.rename({c: f"p2_{c}" for c in lag2.columns if c not in ("player_id", "season")})

birth = (
    nfl.load_players().select(["gsis_id", "birth_date"]).rename({"gsis_id": "player_id"})
)
df = (
    uni.filter(pl.col("season") >= 2018)
    .join(lag, on=["player_id", "season"], how="inner")
    .join(lag2, on=["player_id", "season"], how="left")
    .join(birth, on="player_id", how="left")
)
df = df.with_columns([
    pl.col("p2_avail_share").fill_null(0.0),
    pl.col("p2_inj_missed_share").fill_null(0.0),
    (pl.col("season") - pl.col("birth_date").cast(pl.Date, strict=False).dt.year())
    .cast(pl.Float64).alias("age"),
])
df = df.with_columns(pl.col("age").fill_null(pl.col("age").median()))

print(f"frame: {df.height} player-seasons {df['season'].min()}-{df['season'].max()}")

# ------------------------------------------ persistence, done honestly
print("\n=== how persistent is each thing, year over year? ===")
for pos in OFFENSE:
    s = df.filter(pl.col("position") == pos)
    r_role = np.corrcoef(s["p1_avail_share"], s["avail_share"])[0, 1]
    r_inj = np.corrcoef(s["p1_inj_missed_share"], s["inj_missed_share"])[0, 1]
    print(f"  {pos}: availability r={r_role:+.3f}   injury-missed r={r_inj:+.3f}   n={s.height}")

# ---------------------------------------------- fantasy-relevant subset
# The board never applies expected_games to a practice-squad tight end.
# Everything below is also reported on the players who were actually
# rankable the prior season: top 40 QB / 60 RB / 80 WR / 40 TE by prior
# season points. Same cut the board's replacement level roughly implies.
TOP_N = {"QB": 40, "RB": 60, "WR": 80, "TE": 40}
df = df.with_columns(
    pl.col("p1_season_points").rank("ordinal", descending=True)
      .over(["season", "position"]).alias("_prior_rank")
).with_columns(
    (pl.col("_prior_rank") <= pl.col("position").replace_strict(TOP_N)).alias("relevant")
)
print(f"\nfantasy-relevant subset: {df['relevant'].sum()} of {df.height}")


# ------------------------------------------------------- nested models
def ridge_fit(X, y, alpha=1.0):
    Xb = np.column_stack([np.ones(len(X)), X])
    A = Xb.T @ Xb + alpha * np.eye(Xb.shape[1])
    A[0, 0] -= alpha
    return np.linalg.solve(A, Xb.T @ y)


def ridge_pred(beta, X):
    return np.column_stack([np.ones(len(X)), X]) @ beta


BLOCKS = {
    "A_position_mean": [],
    "B_role": ["p1_avail_share", "p2_avail_share"],
    "C_role_plus_injury": ["p1_avail_share", "p2_avail_share", "p1_weeks_out",
                           "p1_weeks_dnp", "p1_weeks_soft", "p1_weeks_major",
                           "p1_weeks_listed"],
    "D_role_injury_age": ["p1_avail_share", "p2_avail_share", "p1_weeks_out",
                          "p1_weeks_dnp", "p1_weeks_soft", "p1_weeks_major",
                          "p1_weeks_listed", "age"],
}


def evaluate(frame, target, label):
    print(f"\n=== {label}: predicting `{target}`, leave-one-season-out ===")
    seasons = sorted(frame["season"].unique().to_list())
    errs = {}
    for name, cols in BLOCKS.items():
        resid = []
        for s in seasons:
            tr = frame.filter(pl.col("season") != s)
            te = frame.filter(pl.col("season") == s)
            pred = np.zeros(te.height)
            for pos in OFFENSE:
                m_tr = tr.filter(pl.col("position") == pos)
                m_te_idx = (te["position"] == pos).to_numpy()
                if m_te_idx.sum() == 0 or m_tr.height < 30:
                    continue
                ytr = m_tr[target].to_numpy()
                if not cols:
                    pred[m_te_idx] = ytr.mean()
                    continue
                Xtr = m_tr.select(cols).to_numpy().astype(float)
                Xte = te.filter(pl.col("position") == pos).select(cols).to_numpy().astype(float)
                mu, sd = Xtr.mean(0), Xtr.std(0)
                sd[sd == 0] = 1.0
                beta = ridge_fit((Xtr - mu) / sd, ytr)
                pred[m_te_idx] = np.clip(ridge_pred(beta, (Xte - mu) / sd), 0, 1)
            resid.append(te[target].to_numpy() - pred)
        resid = np.concatenate(resid)
        errs[name] = resid
        print(f"  {name:>20s} RMSE={np.sqrt((resid**2).mean()):.4f}  "
              f"MAE={np.abs(resid).mean():.4f}  bias={resid.mean():+.4f}")

    order = list(BLOCKS)
    for prev, cur in zip(order, order[1:]):
        d = errs[prev] ** 2 - errs[cur] ** 2
        gain = np.sqrt((errs[prev] ** 2).mean()) - np.sqrt((errs[cur] ** 2).mean())
        se = d.std(ddof=1) / np.sqrt(len(d)) / (2 * np.sqrt((errs[cur] ** 2).mean()))
        verdict = "SHIP" if gain > se else "no"
        print(f"  {cur} vs {prev}: RMSE gain={gain:+.4f}  1SE={se:.4f}  -> {verdict}")


evaluate(df, "avail_share", "ALL ROSTERED")
evaluate(df.filter(pl.col("relevant")), "avail_share", "FANTASY-RELEVANT")
evaluate(df.filter(pl.col("relevant")), "inj_missed_share", "FANTASY-RELEVANT, health only")

df.write_csv("data/scratch_availability_model.csv")
print("\nwrote data/scratch_availability_model.csv")
