import polars as pl
import nflreadpy as nfl
from src.scoring import load_config, calculate_offensive_points

from pathlib import Path

from src.team_codes import normalize_team_column

CONFIG_PATH = Path(__file__).resolve().parent.parent / "league_config_lebronjames.json"

RECENCY_WEIGHTS = [0.5, 0.3, 0.2]  # most-recent-first; applies to any 3-year window
OFFENSE_POSITIONS = ["QB", "RB", "WR", "TE"]

# ---------------------------------------------------------------------
# Phase 11 A -- candidate baseline weighting schemes (CP2).
#
# The incumbent ("recency") weights a season by WHEN it happened and
# nothing else. A 4-game 2024 counts 30% exactly like a 17-game 2024.
# That is the bug this phase exists to test: an injury season enters the
# baseline at full weight carrying a per-game rate measured while the
# player was hurt, coming back, or splitting a role he no longer had.
#
# Nothing here changes the default. `recency` reproduces the pre-Phase-11
# baseline exactly, so importing this module cannot move a number until
# CP3 picks a winner and changes DEFAULT_SCHEME deliberately.
# ---------------------------------------------------------------------
# ADOPTED at CP3 on the nine-season training set (Aug 4).
#
# `discount_thin` halves a season's recency weight if the player appeared
# in fewer than MIN_GAMES_THRESHOLD games that year. It cleared both
# pre-committed clauses on the AFFECTED subgroup (n=1617): paired ΔMAE
# +0.0619 ± 0.0289, which is 2.14 SE, and positive at all four positions.
#
# Worth recording that this REVERSES the verdict reached a few hours
# earlier, and why. That run scored a silently-corrupted five-season
# training set (see the DEFAULT_TARGET_SEASONS incident in
# PHASE_8-14_PLAN.md) where the same scheme measured +0.0702 ± 0.0378 = 1.86
# SE and went negative at RB. The point estimate barely moved -- 0.070 to
# 0.062 -- while the standard error fell from 0.0378 to 0.0289. That is
# what more data is supposed to do to a real effect, and it is the reason
# to believe this rather than the reason to be suspicious of it.
#
# It is still a +0.06 PPG effect, carried substantially by QB (+0.426),
# with RB at +0.0005, i.e. nothing. In-sample. Phase 13 CP2's holdout is
# still the arbiter.
DEFAULT_SCHEME = "discount_thin"

# What counts as a season too thin to trust at face value. 8 of 17 is
# "missed half the year"; it is a starting point to be backtested, not a
# finding. CP2 sweeps it.
MIN_GAMES_THRESHOLD = 8
THIN_SEASON_DISCOUNT = 0.5
FULL_SEASON_GAMES = 17

SCHEMES = [
    "recency",          # incumbent: 50/30/20 by recency rank
    "games",            # availability only, recency ignored
    "recency_x_games",  # recency rank x games played
    "discount_thin",    # recency, halved for seasons under the threshold
    "drop_thin",        # recency, zeroed for seasons under the threshold
]

# Deliberately NOT a sixth scheme: "recency x (games / 17)". Dividing by a
# constant cancels in the normalization, so it produces weights identical
# to `recency_x_games` for every player -- a player cannot exceed 17
# regular-season games, so the cap never binds either. It was in the first
# draft of this list and came out when a hand-check showed the two columns
# matching to three decimals on all four test cases.


def season_weights(seasons_present, games_by_season, scheme=DEFAULT_SCHEME):
    """
    Returns {season: normalized weight} for one player's baseline window.

    A note on the `drop_thin` fallback, which is the only subtle line
    here. A player whose ENTIRE window is thin -- Phil Mafah, one game --
    has every season zeroed and would end up with no baseline at all, or
    worse, a divide-by-zero that silently drops him off the board. He
    falls back to plain recency instead.

    That is deliberate and it is a scoping decision, not a patch: a
    player with too little history to weight is Phase 11 B's problem, and
    B shrinks him toward the position mean. A should not quietly delete
    the players B was written to handle. Keeping him here with an honest,
    thin baseline is what lets B measure and fix it.
    """
    recency = get_recency_weights(seasons_present)

    def games(season):
        return float(games_by_season.get(season, 0))

    if scheme == "recency":
        raw = dict(recency)
    elif scheme == "games":
        raw = {s: games(s) for s in seasons_present}
    elif scheme == "recency_x_games":
        raw = {s: recency[s] * min(games(s), FULL_SEASON_GAMES) for s in seasons_present}
    elif scheme == "discount_thin":
        raw = {s: recency[s] * (1.0 if games(s) >= MIN_GAMES_THRESHOLD
                                else THIN_SEASON_DISCOUNT)
               for s in seasons_present}
    elif scheme == "drop_thin":
        raw = {s: (recency[s] if games(s) >= MIN_GAMES_THRESHOLD else 0.0)
               for s in seasons_present}
    else:
        raise ValueError(f"unknown weighting scheme {scheme!r}; expected one of {SCHEMES}")

    total = sum(raw.values())
    if total <= 0:
        raw = dict(recency)
        total = sum(raw.values())

    return {s: w / total for s, w in raw.items()}

RAW_STAT_COLUMNS = [
    "passing_yards", "passing_tds", "passing_interceptions",
    "carries", "rushing_yards", "rushing_tds",
    "receptions", "targets", "receiving_yards", "receiving_tds",
    "fantasy_points",
]


def load_veteran_stats(seasons):
    raw = nfl.load_player_stats(seasons)
    raw = raw.filter(pl.col("season_type") == "REG")
    raw = raw.filter(pl.col("position").is_in(OFFENSE_POSITIONS))
    raw = raw.filter(pl.col("player_id").is_not_null())

    config = load_config(CONFIG_PATH)
    raw = raw.with_columns(
        pl.struct(raw.columns)
        .map_elements(lambda row: calculate_offensive_points(row, config), return_dtype=pl.Float64)
        .alias("fantasy_points")
    )
    return raw


def aggregate_season_stats(raw_stats):
    grouped = (
        raw_stats.sort("week")
        .group_by(["player_id", "season"], maintain_order=True)
        .agg(
            [pl.col(c).sum().alias(c) for c in RAW_STAT_COLUMNS]
            + [
                pl.len().alias("games_played"),
                pl.col("player_display_name").last().alias("player_name"),
                pl.col("position").last().alias("position"),
            ]
        )
    )

    per_game_exprs = [
        (pl.col(c) / pl.col("games_played")).alias(f"{c}_per_game")
        for c in RAW_STAT_COLUMNS
    ]
    grouped = grouped.with_columns(per_game_exprs)

    keep_cols = ["player_id", "player_name", "position", "season", "games_played"] + \
                [f"{c}_per_game" for c in RAW_STAT_COLUMNS]
    return grouped.select(keep_cols)


def get_recency_weights(seasons_present):
    """
    Maps a list of seasons to weights based on recency RANK, not
    absolute season number -- most recent season present = 50%,
    middle = 30%, oldest = 20%. This lets the same 3-year weighting
    scheme apply to any historical window (e.g. "entering 2024" uses
    2021/2022/2023 the same way "entering 2026" uses 2023/2024/2025),
    which the Phase 5 backtest needs.
    """
    sorted_desc = sorted(seasons_present, reverse=True)
    return {s: w for s, w in zip(sorted_desc, RECENCY_WEIGHTS)}


def apply_season_weighting(season_stats, scheme=DEFAULT_SCHEME):
    per_game_cols = [c for c in season_stats.columns if c.endswith("_per_game")]
    weighted_rows = []

    for group_key, player_df in season_stats.group_by("player_id"):
        player_id = group_key[0]
        seasons_present = player_df["season"].to_list()
        games_by_season = dict(
            zip(player_df["season"].to_list(), player_df["games_played"].to_list())
        )
        normalized_weights = season_weights(seasons_present, games_by_season, scheme)
        weighted_stats = {}
        for col in per_game_cols:
            weighted_stats[col] = sum(
                player_df.filter(pl.col("season") == s)[col][0] * normalized_weights[s]
                for s in seasons_present
            )

        most_recent = player_df.sort("season").tail(1)
        weighted_stats["player_id"] = player_id
        weighted_stats["player_name"] = most_recent["player_name"][0]
        weighted_stats["position"] = most_recent["position"][0]
        weighted_stats["games_played"] = player_df["games_played"].sum()
        # Phase 11 B (CP4). How many of the window's seasons the player
        # actually appeared in. `games_played` alone can't tell 3 seasons
        # of 5 games from 1 season of 15 -- same sample size, different
        # amounts of evidence about a stable role.
        weighted_stats["seasons_used"] = len(seasons_present)

        weighted_rows.append(weighted_stats)

    return pl.DataFrame(weighted_rows)


# ---------------------------------------------------------------------
# Phase 11 B (CP5) -- baseline shrinkage. ADOPTED Aug 4.
#
#     confidence = games / (games + K)
#     shrunk     = confidence * baseline + (1 - confidence) * anchor
#
# K=2 on the nine-season set: paired dMAE +0.0919 +/- 0.0334 on the
# low-confidence subgroup (2.75 SE), full-pool MAE improves at every K,
# and K=2 is the interior argmax of a 0-8 sweep. See src/shrinkage.py for
# the sweep and the pre-committed decision rule.
#
# WHY THE 30th PERCENTILE AND NOT THE MEAN. Swept alongside K, and the
# anchor mattered more than K did: every mean-anchored variant was
# NEGATIVE at every K. Shrinking a fringe player toward the position mean
# pulls him UP, which is backwards -- the whole point is that a projection
# resting on one game should fall back to something like what you could
# get for free, not to the average starter. The 30th percentile of
# players with a real sample behaves like that; the mean does not.
#
# K=2 is deliberately gentle. A player with a full season already sits at
# 17/19 = 0.89 confidence and barely moves; the correction is aimed at
# the one-game and eight-game projections, which is where CP1 found the
# damage.
# ---------------------------------------------------------------------
SHRINKAGE_K = 2
SHRINKAGE_ANCHOR_QUANTILE = 0.30
SHRINKAGE_ANCHOR_MIN_GAMES = 16

# QB IS EXCLUDED, and the reason is one this project has already met.
#
# The anchor is the 30th percentile of players with 16+ games. At
# RB/WR/TE that lands at 3.44 / 3.86 / 2.88 -- plenty of part-time
# players accumulate games, so the qualified pool still contains ordinary
# ones. At QB it lands at 12.52, because quarterback is one-per-team:
# "played 16 games" IS "was the starter." fit_weights.SUPPRESS_LEVEL_SHIFT
# documents the identical effect in its own words -- "a mediocre receiver
# still plays eight games, a mediocre quarterback gets benched."
#
# Applied to the live pool, that pulled every clipboard-holder up toward a
# starter's production: Nathan Peterman -0.40 -> 8.21, Logan Woodside
# -0.32 -> 8.24, 59% of quarterbacks moved UP. Shrinkage assumes a small
# sample is a noisy estimate of the same quantity. For a backup QB it is
# a precise estimate of a different one.
#
# The sweep never saw this because it scored only players who went on to
# play 8+ games in the target season -- Peterman was not in the
# population the anchor was fitted on, but he is in the population it
# gets applied to.
#
# `discount_thin` (CP3) still applies at QB and its QB-driven benefit
# stands: re-weighting a player's OWN seasons cannot inflate him toward
# anyone else's number. It is specifically the pull-toward-a-population
# -anchor mechanism that fails here.
SHRINKAGE_EXCLUDED_POSITIONS = {"QB"}


def apply_baseline_shrinkage(table, k=SHRINKAGE_K, group_by=("position",),
                             value_column="fantasy_points_per_game",
                             games_column="games_played",
                             exclude=None):
    """
    Adds `baseline_anchor_ppg`, `baseline_confidence`, and
    `<value_column>_shrunk`.

    `group_by` is ("position",) live and ("season", "position") in the
    backtest, because the anchor has to be computed from information
    available at the time -- a 2019 player must not be shrunk toward a
    number that knows about 2024.

    `exclude` is a boolean expression for rows that must NOT be shrunk
    and must NOT contribute to the anchor. Rookies are the case: their
    baseline is a cohort projection rather than personal history, so
    `games_played` says nothing about how much evidence it rests on, and
    feeding cohort numbers into the anchor would contaminate it. They
    pass through at confidence 1.0.
    """
    group_by = list(group_by)
    excluded = pl.lit(False) if exclude is None else exclude

    # Positional exclusions ride along with the caller's own, so an
    # excluded position is left out of BOTH the shrinking and the anchor.
    if SHRINKAGE_EXCLUDED_POSITIONS and "position" in table.columns:
        excluded = excluded | pl.col("position").is_in(
            list(SHRINKAGE_EXCLUDED_POSITIONS)
        )

    qualified = (
        (pl.col(games_column) >= SHRINKAGE_ANCHOR_MIN_GAMES) & ~excluded
    )
    anchor = (
        pl.col(value_column)
        .filter(qualified)
        .quantile(SHRINKAGE_ANCHOR_QUANTILE)
        .over(group_by)
        .alias("baseline_anchor_ppg")
    )

    table = table.with_columns(anchor).with_columns(
        # A group with nobody qualified would produce a null anchor and
        # silently null out every shrunk baseline in it. Fall back to the
        # group's own median rather than losing the rows.
        pl.col("baseline_anchor_ppg").fill_null(
            pl.col(value_column).median().over(group_by)
        )
    )

    games = pl.col(games_column).cast(pl.Float64).fill_null(0.0)
    confidence = (
        pl.when(excluded)
        .then(pl.lit(1.0))
        .otherwise(games / (games + float(k)))
    )

    return table.with_columns(
        confidence.alias("baseline_confidence")
    ).with_columns(
        (pl.col("baseline_confidence") * pl.col(value_column)
         + (1 - pl.col("baseline_confidence")) * pl.col("baseline_anchor_ppg"))
        .alias(f"{value_column}_shrunk")
    )


def build_veteran_feature_table(seasons=[2023, 2024, 2025], scheme=DEFAULT_SCHEME):
    raw = load_veteran_stats(seasons)
    season_stats = aggregate_season_stats(raw)
    return apply_season_weighting(season_stats, scheme)

def attach_current_team(veteran_features):
    """
    Joins each player's most current team onto the veteran feature table
    using nflreadpy's load_players() `latest_team` column. This resolves
    the mid-season-trade issue from Phase 2 (89 players had multi-team
    stats in 2025) for free, since latest_team already reflects each
    player's most recent team rather than their first-seen team.
    """
    players = nfl.load_players().select(["gsis_id", "latest_team"]).rename(
        {"gsis_id": "player_id", "latest_team": "team"}
    )
    players = normalize_team_column(players)
    return veteran_features.join(players, on="player_id", how="left")