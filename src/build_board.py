"""
Builds the Excel draft board from data/player_features.csv.

WHY THIS FILE EXISTS
--------------------
Through v7 the board was assembled ad hoc, which made it the only
artifact in the project with no source. That's backwards -- the board is
the actual deliverable, and everything upstream exists to produce it.
It also meant the ranking logic that matters most on draft day (VOR,
replacement levels, draft targets) lived nowhere it could be reviewed or
re-run.

Every rule below was reverse-engineered from 2026_Draft_Board_v7.xlsx and
verified against it: replacement levels, VOR, sort order, value delta,
and draft-target text all reproduce v7 given v7's inputs.

USAGE
-----
    # Lebron James (12 team) -- writes 2026_LebronJames_Board_v9.xlsx
    python -m src.build_board

    # Dunlap Family (6 team) -- writes 2026_DunlapFamily_Board_v9.xlsx
    python -m src.build_board --config league_config_dunlap.json

    python -m src.build_board --note "refreshed ADP"
    python -m src.build_board --output some/path.xlsx

ONE FILE PER LEAGUE
-------------------
The filename carries the league and the MODEL version, and nothing else.
Rebuilding overwrites in place and appends a row to the in-workbook "Build
History" sheet, so the count of rebuilds is visible without accumulating a
folder of near-identical spreadsheets. Bump MODEL_VERSION when weights or
features change; a data refresh (new ADP, edited injury overrides) keeps the
same version and is told apart by the git hash and timestamp in that sheet.

Requires openpyxl (not currently in .venv):
    pip install openpyxl
"""

import argparse
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path

import polars as pl
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from src.ranking import load_situational_weights

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "player_features.csv"
CONFIG_PATH = PROJECT_ROOT / "league_config.json"

# MODEL_VERSION bumps when the model changes -- new weights, new features,
# a refit. It does NOT bump for a data refresh (new ADP pull, updated injury
# overrides), because nothing about the ranking logic moved. Rebuilds of the
# same model version are told apart by the Build History sheet, not by the
# filename, which is what keeps one file per league instead of a pile of
# near-identical spreadsheets.
# Phase 10 (Aug 4): 9 -> 10. This one is an honest bump, unlike the 8 -> 9
# documented in PHASE_8_PLAN.md. Features changed (age replaced experience;
# usage_trend_share added at RB and TE) and all three positions were refit,
# so ranks move for real.
# Phase 11 (Aug 4): 10 -> 11. No refit -- the weights are byte-identical to
# v10. Replacement level changed from starter slots to expected players
# drafted (CP6), which moves every VOR and reorders the board, so it is a
# ranking-logic change and bumps the version even though no coefficient did.
MODEL_VERSION = 11

HISTORY_SHEET = "Build History"
INJURY_OVERRIDES_PATH = PROJECT_ROOT / "injury_overrides.csv"

# Status in injury_overrides.csv that removes a player from contention.
# Any OTHER status string is recorded in the Notes column but changes no
# ranking -- so you can jot "QUESTIONABLE - hamstring" as a reminder
# without it silently moving anyone.
OUT_STATUS = "OUT_SEASON"

# Phase 11 CP8. Statuses that cost a player a KNOWN, PARTIAL slice of the
# season rather than all of it. PUP and NFI are not opinions about health --
# they are roster designations with a rule attached: a player who opens the
# regular season on either list is ineligible for the first four games. That
# is a floor, not an estimate, which is what separates these from
# QUESTIONABLE and lets them move a number at all.
#
# Anything longer is player-specific and belongs in the optional
# `games_missed` column of injury_overrides.csv, which overrides this
# default. Kittle's Achilles is the obvious candidate -- four games is
# almost certainly generous to him.
PARTIAL_STATUSES = {"PUP", "NFI"}
PUP_DEFAULT_GAMES_MISSED = 4

# How the single FLEX slot is expected to be filled across the league.
# This is a modeling ASSUMPTION, not a league rule -- it's the standard
# full-PPR split and it only moves replacement levels by a fraction of a
# roster spot. Documented here because it silently shifts every VOR.
#
# Phase 11 CP6 demoted this to a fallback: it is now used only by
# compute_starter_ranks(), which no longer drives VOR. See
# compute_replacement_ranks() for why.
FLEX_SPLIT = {"RB": 0.40, "WR": 0.40, "TE": 0.20}

# Roster slots every team fills but this model does not rank. They still
# consume real picks, so they must come out of the pick pool before skill
# positions are allocated -- otherwise a 16-round draft looks like it has
# 2 x num_teams more skill picks than it does, and replacement level lands
# too deep at every position.
UNMODELED_SLOTS_PER_TEAM = 2  # 1 K + 1 DST

MODELED_POSITIONS = ["QB", "RB", "WR", "TE"]

FONT_NAME = "Arial"

# Row fill by position, for players inside the draftable pool.
POSITION_FILLS = {
    "QB": "CFE2F3",  # light blue
    "RB": "D9EAD3",  # light green
    "WR": "FCE5CD",  # light orange
    "TE": "EAD1DC",  # light mauve
}
UNDRAFTABLE_FILL = "D9D9D9"  # gray: has ADP but ranks past the last pick
NO_ADP_FILL = "F4CCCC"       # pink: no real ADP, hard-capped to the bottom
OUT_FILL = "E06666"          # strong red: out for the season, do not draft
HEADER_FILL = "1F4E78"
INJURY_FILL = "FF0000"

COLUMNS = [
    ("Rank", 7), ("Pos", 6), ("Player", 24), ("Adj PPG", 9), ("Sit Adj", 8),
    ("VOR", 8),
    # Phase 11 CP8. Availability, kept OUT of the ranking on purpose.
    # Exp Gm is this league's regular season minus known PUP/NFI absence;
    # Exp Pts is Adj PPG x Exp Gm. Adj PPG stays an honest per-game rate,
    # so Kittle does not move in the Rank column -- the cost of missing
    # September shows up here instead, and it is genuinely a different
    # number in each league (12-game regular season in Dunlap, 14 in
    # Lebron James), which no single rank column could express.
    ("Exp Gm", 8), ("Exp Pts", 9),
    # Phase 11. Plain-language decomposition of Sit Adj: the largest
    # signed contributions, relative to an average player at that
    # position. Generated from the same weights that produce the number
    # in Sit Adj, so the two cannot disagree.
    ("Why (value drivers)", 46),
    ("Draft Target", 24), ("Team", 7), ("ADP (Ovr)", 10),
    ("ADP (Rd.Pk)", 12), ("Value Δ (picks)", 12), ("Has ADP", 4), ("Bye", 6),
    ("Rook", 6), ("Recent Injury", 12), ("GP (sample)", 11),
    # Phase 10. Age is a model input at RB/WR/TE (it replaced
    # `experience`, which it beat at every position). Usage Trend is
    # also a model input at all three -- WR only after the training set
    # widened from three seasons to five, which took it from p=0.23 to
    # p=0.034.
    #
    # "Trend n" is how many seasons the slope was fitted on. 3 is a full
    # slope, 2 is a two-point slope, 0 means none could be fitted and the
    # player was mean-imputed. Worth showing rather than hiding: the
    # trend signal turned out to be CARRIED by the n=2 players, not
    # weakened by them.
    ("Age", 6), ("Usage Trend", 11), ("Trend n", 8),
    ("Notes (manual)", 45),
]

# Column label -> 1-based sheet index. Phase 10 left a comment warning that
# "if you insert another column, these all move again" above a block of
# hardcoded indices; Phase 11 inserts three columns, so that warning gets
# retired rather than obeyed. Look columns up by name.
COLUMN_INDEX = {label: i for i, (label, _) in enumerate(COLUMNS, start=1)}


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return json.load(f)


def league_slug(config):
    """
    'Dunlap Family Fantasy Football' -> 'DunlapFamily'.

    Prefers an explicit `config_key` if the config sets one, since the league
    name is a display string that may change wording without meaning a
    different league.
    """
    key = config.get("config_key")
    if key:
        return "".join(part.capitalize() for part in str(key).split("_"))
    words = [w for w in str(config["league_name"]).split() if w.lower() != "fantasy"]
    return "".join(w.capitalize() for w in words[:2])


def git_short_hash():
    """
    Short commit hash, or 'nogit' if unavailable.

    This is what actually distinguishes two rebuilds of the same model
    version, so it's worth having even when it fails softly.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return result.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def read_build_history(output_path):
    """
    Pulls the existing Build History rows out of a board we're about to
    overwrite.

    This is the mechanism that lets one file per league still answer "how many
    times has this been rebuilt." The alternative -- a new file per build --
    answers the same question by accumulating near-identical spreadsheets, and
    then you have to remember which one you printed.

    A build history that silently resets to 1 would be worse than none at all,
    so a corrupt or unreadable existing file raises rather than starting over.
    """
    output_path = Path(output_path)
    if not output_path.exists():
        return []
    workbook = load_workbook(output_path, read_only=True)
    if HISTORY_SHEET not in workbook.sheetnames:
        workbook.close()
        return []
    sheet = workbook[HISTORY_SHEET]
    rows = [
        [cell for cell in row]
        for row in sheet.iter_rows(min_row=2, values_only=True)
        if row and row[0] is not None
    ]
    workbook.close()
    return rows


def apply_injury_overrides(players, path=INJURY_OVERRIDES_PATH):
    """
    Joins the manually-maintained injury_overrides.csv onto the player
    table, adding `out_for_season` (bool) and `injury_note` (str).

    WHY THIS IS MANUAL
    ------------------
    The model's own `recent_major_injury` feature only flags players who
    ended the 2025 REGULAR SEASON on IR. It is structurally blind to
    anything that happens in the 2026 offseason or preseason, and no
    nflverse feed reliably covers August injuries -- official injury
    reports don't begin until Week 1.

    That blindness is dangerous specifically BECAUSE ADP gets refreshed.
    The market drops an injured player within hours; the model's
    projection doesn't move at all, since it's built from trailing
    2023-25 per-game rates. `value_delta` is (adp_rank - model_rank), so
    the collapse in ADP shows up as a huge positive value delta and the
    board recommends him as a bargain. Ricky Pearsall went from -11 to
    roughly +87 in exactly this way. Without this file, every injury
    between now and draft day makes the board worse, not staler.

    File format -- player_name,status,note[,games_missed]
    Only status == OUT_STATUS changes the RANKING. Anything else is
    carried into the Notes column and leaves rank alone.

    PHASE 11 CP8 -- PUP AND NFI
    ---------------------------
    A third case sits between "fine" and "gone." PUP and NFI are not
    health opinions, they are roster designations with a rule attached: a
    player who opens the regular season on either list cannot play the
    first four games. Treating that as a note (the old behavior) had
    Kittle and Charbonnet showing at full value; treating it as
    OUT_SEASON would be a lie in the other direction.

    So those statuses set `expected_games_missed` -- four by default, or
    whatever the optional `games_missed` column says for a specific
    player. That number feeds Exp Gm / Exp Pts and NOTHING else. It does
    not touch `adjusted_fantasy_points_per_game`, so it cannot move rank
    or VOR.

    That restraint is deliberate and it is the phase's open question,
    answered: PPG is a RATE, and a torn Achilles doesn't make Kittle worse
    per game he plays -- it makes him play fewer of them. Folding
    availability into the rate would corrupt the one number the whole
    model is fitted to predict, and would silently double-count the moment
    a future phase models availability directly. Season-long value lives
    in its own column instead, where it can be read against the rate
    rather than baked into it.

    UNMATCHED NAMES RAISE. A typo'd name silently matching nothing would
    leave an injured player sitting in the draftable pool looking like a
    steal -- the exact failure this file exists to prevent -- so it fails
    loudly instead.
    """
    empty = players.with_columns([
        pl.lit(False).alias("out_for_season"),
        pl.lit(None, dtype=pl.String).alias("injury_note"),
        pl.lit(0.0).alias("expected_games_missed"),
        pl.lit(None, dtype=pl.String).alias("injury_status"),
    ])

    if not Path(path).exists():
        print(f"No {Path(path).name} found -- no injury overrides applied.")
        return empty

    overrides = pl.read_csv(path)
    if overrides.height == 0:
        print(f"{Path(path).name} is empty -- no injury overrides applied.")
        return empty

    known_names = set(players.select("player_name").to_series().to_list())
    unmatched = [
        name for name in overrides.select("player_name").to_series().to_list()
        if name not in known_names
    ]
    if unmatched:
        raise ValueError(
            f"{Path(path).name}: these names match no player in "
            f"player_features.csv: {unmatched}. Fix the spelling to match "
            f"the nflverse name exactly -- an unmatched override does "
            f"nothing, which would leave an injured player draftable."
        )

    status = (pl.col("status").cast(pl.String).str.to_uppercase()
              .str.strip_chars())

    # `games_missed` is optional -- the file is hand-maintained and most
    # rows have no reason to carry one. Absent column, absent value, and
    # blank string all fall back to the PUP default.
    if "games_missed" in overrides.columns:
        stated_games = pl.col("games_missed").cast(pl.Float64, strict=False)
    else:
        stated_games = pl.lit(None, dtype=pl.Float64)

    overrides = overrides.select([
        "player_name",
        (status == OUT_STATUS).alias("out_for_season"),
        pl.col("note").cast(pl.String).alias("injury_note"),
        status.alias("injury_status"),
        pl.when(status.is_in(list(PARTIAL_STATUSES)))
          .then(stated_games.fill_null(float(PUP_DEFAULT_GAMES_MISSED)))
          .otherwise(0.0)
          .alias("expected_games_missed"),
    ])

    joined = players.join(overrides, on="player_name", how="left").with_columns([
        pl.col("out_for_season").fill_null(False),
        pl.col("expected_games_missed").fill_null(0.0),
    ])

    out_count = joined.select(pl.col("out_for_season").sum()).item()
    partial = joined.filter(pl.col("expected_games_missed") > 0)
    print(f"Injury overrides: {overrides.height} entries, {out_count} marked {OUT_STATUS}, "
          f"{partial.height} on PUP/NFI")
    for row in partial.select(["player_name", "injury_status",
                               "expected_games_missed"]).iter_rows():
        print(f"   {row[0]}: {row[1]}, -{row[2]:.0f} games")
    return joined


def compute_expected_points(players, config):
    """
    Phase 11 CP8. Adds `expected_games` and `expected_total_points`.

    `expected_games` is this LEAGUE'S regular season minus known absence,
    and the denominator is the regular season on purpose -- not the
    17-week NFL calendar. Weeks 15-17 are worth nothing in a league whose
    final is week 14, and nothing at all in Dunlap, whose season ends in
    week 12. Those are the games that decide whether you reach the
    playoffs at all.

    That is why the same four-game PUP absence costs 4/12 = 33% of a
    Dunlap season against 4/14 = 29% of a Lebron James one, and why the
    same player is honestly worth different amounts in the two leagues --
    something the board has never had to express before.

    Out-for-season players get zero expected games, which is the one place
    Exp Pts and the ranking agree.
    """
    season = float(config.get("fantasy_season_length", 17))

    expected_games = (
        pl.when(pl.col("out_for_season"))
        .then(pl.lit(0.0))
        .otherwise(
            (pl.lit(season) - pl.col("expected_games_missed")).clip(0.0, season)
        )
    )

    return players.with_columns([
        expected_games.alias("expected_games"),
        (pl.col("adjusted_fantasy_points_per_game") * expected_games)
        .alias("expected_total_points"),
    ])


# Phase 11. How each model feature reads in English on the board.
#
# `label` receives the player's raw feature value so a driver can carry the
# number that caused it ("age 31" beats "age"), and the SIGN of the printed
# contribution always comes from the arithmetic, never from the wording --
# so a feature whose coefficient flips in a later refit relabels itself
# correctly with no edit here.
#
# Note the two mean-reversion terms. `workload_share` and
# `position_competition_ppg` both carry negative weights, which reads
# backwards until you remember what the regression predicts: not PPG, but
# the DELTA from a player's own trailing baseline. A back already holding
# 60% of his backfield has his usage priced in and little room left to
# grow; a receiver whose position group is otherwise weak has already
# banked that. So the wording describes the situation, not a judgment.
DRIVER_LABELS = {
    "age": lambda v: f"age {v:.0f}",
    "workload_share": lambda v: f"{v * 100:.0f}% team share",
    "usage_trend_share": lambda v: f"role trend {v * 100:+.1f}pp/yr",
    "trend_missing": lambda v: "thin usage history",
    "qb_changed": lambda v: "new QB" if v else "same QB",
    "team_changed": lambda v: "new team" if v else "same team",
    "recent_major_injury": lambda v: "2025 IR" if v else "no 2025 IR",
    "position_competition_ppg": lambda v: f"teammates {v:.1f} PPG",
    "experience": lambda v: f"{v:.0f} yrs exp",
    "continuity_score": lambda v: f"continuity {v:.2f}",
}

# A driver has to move the projection by at least this much to be worth a
# reader's attention mid-draft. Below it the term is real but not
# actionable, and printing four of them buries the one that matters.
DRIVER_MIN_ABS = 0.15
DRIVER_MAX_TERMS = 4


def build_value_drivers(players, weights_by_position=None):
    """
    Phase 11. Adds `value_drivers`: a short signed list of what is pushing
    a player's projection up or down, largest effect first.

        "-1.4 62% team share · +0.9 role trend +3.1pp/yr · -0.5 age 30"

    Each term is a real number out of the fitted model, not a description
    of one:

        contribution_f = (player_value_f - position_mean_f) x weight_f

    so the terms are deviations from an AVERAGE player at that position,
    and they sum -- with the position's base offset -- exactly to the Sit
    Adj column beside them. That identity is the whole point. A "why"
    column written by hand, or generated from a separate set of rules,
    would eventually contradict the number it is explaining, and on draft
    day you would have no way to tell which one was lying. This one cannot
    drift: it is computed from the same JSON that computes the adjustment,
    and verify_adjustments.py asserts the two reconcile.

    Two housekeeping notes. Nulls are filled with the position mean, which
    is what apply_situational_weights() does, so an imputed feature
    contributes exactly 0 and correctly never appears as a driver -- "no
    opinion" should not read as a reason. And the position base offset
    (intercept plus the mean-value terms) is around -0.1 PPG at every
    position, small enough to omit from the string unless it clears the
    same threshold as everything else.

    Rookies take no situational adjustment at all, so they say so rather
    than showing an empty cell that looks like missing data.
    """
    if weights_by_position is None:
        weights_by_position = load_situational_weights()

    rows = players.to_dicts()
    drivers = []

    for row in rows:
        if row.get("is_rookie"):
            drivers.append("rookie — cohort baseline, no situational adj")
            continue

        spec = weights_by_position.get(row["position"])
        if not spec:
            # K/DST never reach here, and QB did until Phase 10.
            drivers.append(None)
            continue

        means = spec.get("feature_means", {})
        centers = spec.get("centers", {})

        terms = []
        base = float(spec["intercept"])
        for feature, weight in spec["weights"].items():
            mean = float(means.get(feature, 0.0))
            center = float(centers.get(feature, 0.0))

            # Same fill rule as apply_situational_weights(): a missing
            # value means "average player," not zero.
            raw = row.get(feature)
            value = mean if raw is None else float(raw)

            base += (mean - center) * weight
            contribution = (value - mean) * weight
            if abs(contribution) >= DRIVER_MIN_ABS:
                label = DRIVER_LABELS.get(feature, lambda v, f=feature: f)(value)
                terms.append((abs(contribution), f"{contribution:+.1f} {label}"))

        if abs(base) >= DRIVER_MIN_ABS:
            terms.append((abs(base), f"{base:+.1f} position base"))

        terms.sort(key=lambda t: -t[0])
        text = " · ".join(t[1] for t in terms[:DRIVER_MAX_TERMS])
        if not text:
            text = "average situation — no driver above ±0.1"

        # Availability is not part of Sit Adj and must not look like it
        # is, so it goes after a pipe rather than into the sum.
        missed = row.get("expected_games_missed") or 0.0
        if row.get("out_for_season"):
            text += " | OUT for season"
        elif missed > 0:
            text += f" | {row.get('injury_status') or 'INJ'} −{missed:.0f} gm"

        drivers.append(text)

    return players.with_columns(pl.Series("value_drivers", drivers, dtype=pl.String))


def compute_starter_ranks(config):
    """
    LEGACY (Phase 8 - Phase 10). How many players at each position get
    drafted as STARTERS league-wide.

    Superseded by compute_replacement_ranks() below, and retained only so
    build_notes() can show both numbers -- the gap between them is the
    single biggest change in v11 and hiding it would make the board's
    movement inexplicable.
    """
    teams = config["num_teams"]
    slots = config["roster_slots"]
    flex_slots = slots.get("FLEX", 0)

    ranks = {}
    for position in MODELED_POSITIONS:
        starters = teams * slots.get(position, 0)
        flex_share = teams * flex_slots * FLEX_SPLIT.get(position, 0.0)
        ranks[position] = round(starters + flex_share)
    return ranks


def compute_replacement_ranks(config, players):
    """
    Phase 11 CP6. How many players at each position come off the board
    across the WHOLE draft. The next one down is replacement level: the
    best player still sitting on waivers when the draft ends.

    WHY THE OLD RULE WAS WRONG
    --------------------------
    Replacement used to be the last STARTER -- QB12 in a 12-team league,
    which is defensible, but QB6 in a 6-team one, which is not. In a
    6-team league QB7 through QB32 are all unowned, so the quarterback you
    can have for nothing is a fine starting NFL quarterback, not the worst
    rostered one. Setting replacement at QB6 measured every quarterback
    against a bar almost nobody has to clear, and the board responded by
    putting Josh Allen 5th overall in a league where you can stream the
    position. The bug was invisible at 12 teams, where starter count and
    waiver depth roughly agree; it only surfaced when a second league
    forced the comparison.

    The fix is to count picks, not starters. A 6-team 16-round draft is 96
    picks, of which 12 go to kickers and defenses, leaving 84 for skill
    positions -- against 168 in the 12-team league. Fewer picks means a
    shallower cut into every position, which is the actual mechanism by
    which a shallow league makes waivers rich.

    HOW THE 84 GET SPLIT
    --------------------
    By observed draft behavior: the position mix of the first N players in
    ADP order, out-for-season players removed since nobody spends a pick
    on them. This is the plan's "expected players drafted per position"
    and it is the one place ADP earns its keep -- it is the only evidence
    the project has about what drafters actually do, and it enters through
    replacement level rather than through any player's projection, so the
    statistics-only rule on `adjusted_fantasy_points_per_game` holds.

    KNOWN LIMITATION, stated plainly: FFC's ADP comes from 12-team mocks,
    so the mix of its first 84 picks is a 12-team drafter's mix, not a
    6-team drafter's. A 6-team room, facing no scarcity at all, would
    almost certainly take FEWER than the 8 quarterbacks this yields. The
    error therefore runs in the conservative direction -- it understates
    how far QB should fall -- and a league can override the split outright
    with an `expected_drafted` block in its config if you ever have real
    draft results to fit.

    Sanity condition from the plan: the 6-team board must push QB and TE
    DOWN relative to the 12-team board. It does. Josh Allen goes from 7th
    to 15th on Dunlap while rising to 7th on Lebron James, and the
    quarterbacks inside the top 30 go 4 -> 1 on the shallow board and
    1 -> 4 on the deep one.
    """
    override = config.get("expected_drafted")
    if override:
        return {p: int(override[p]) for p in MODELED_POSITIONS if p in override}

    teams = config["num_teams"]
    rounds = config["total_rounds"]
    skill_picks = teams * rounds - teams * UNMODELED_SLOTS_PER_TEAM

    # Draft order = ADP order. Out-for-season players are dropped: they
    # consume no pick, so leaving them in would push replacement one slot
    # shallower at their position for no reason.
    drafted = (
        players
        .filter(pl.col("has_adp") & ~pl.col("out_for_season"))
        .sort("adp", nulls_last=True)
        .head(skill_picks)
    )

    counts = dict(
        drafted.group_by("position").len().iter_rows()
    )
    ranks = {p: int(counts.get(p, 0)) for p in MODELED_POSITIONS}

    # If the ADP feed is shorter than the draft (it currently runs ~201
    # players deep against 168 skill picks in the 12-team league, so this
    # is headroom, not a live problem), scale the observed shares up to
    # the full pick count rather than silently setting replacement too
    # shallow -- which would inflate every VOR on the board.
    observed = sum(ranks.values())
    if observed and observed < skill_picks:
        scale = skill_picks / observed
        print(f"NOTE: ADP covers {observed} of {skill_picks} skill picks -- "
              f"scaling observed position shares by {scale:.2f}x.")
        ranks = {p: max(1, round(n * scale)) for p, n in ranks.items()}

    # A position must have at least as many drafted as there are starting
    # slots; otherwise replacement lands above a player somebody has to
    # start. Only bites on a feed with almost no ADP data.
    floors = compute_starter_ranks(config)
    ranks = {p: max(ranks[p], floors.get(p, 1)) for p in ranks}

    return ranks


def compute_vor(players, replacement_ranks, value_column="adjusted_fantasy_points_per_game"):
    """
    Value Over Replacement: a player's PPG minus his position's
    replacement-level PPG.

    This is what makes the board draftable. Ranking by raw PPG would put
    22 quarterbacks at the top -- Josh Allen scores more than any running
    back alive -- but in a 1-QB league the relevant question isn't "who
    scores most," it's "how much do I gain over the guy I could get for
    free at this position." Allen is worth ~5 points over QB12; Puka
    Nacua is worth ~9 over WR29. That's the real gap.
    """
    frames = []
    for position, rank in replacement_ranks.items():
        pool = players.filter(pl.col("position") == position).sort(
            value_column, descending=True, nulls_last=True
        )
        if pool.height == 0:
            continue

        # Players who are out for the season don't count toward
        # replacement level -- nobody can start them, so they aren't the
        # freely-available alternative this whole calculation is about.
        # Removing one from above the replacement rank pulls a slightly
        # worse player into that slot, which lowers replacement PPG and
        # correctly makes everyone else at the position a bit more
        # valuable. VOR is still computed FOR them, so the board can show
        # what they'd have been worth healthy.
        available = pool.filter(~pl.col("out_for_season"))
        if available.height == 0:
            available = pool

        # If a position has fewer players than its replacement rank,
        # fall back to the worst available rather than indexing off the end.
        index = min(rank, available.height) - 1
        replacement_ppg = available.select(value_column).to_series()[index]
        frames.append(
            pool.with_columns(
                (pl.col(value_column) - replacement_ppg).alias("vor")
            )
        )
    return pl.concat(frames, how="vertical")


def format_slot(overall_pick, teams):
    """
    Turns an overall pick number into 'Beginning/Middle/End of Round N'.

    Rounding is half-up rather than Python's default banker's rounding --
    round(32.5) returns 32 in Python, which silently lands a player in the
    wrong third of a round. At a boundary this is a coin flip anyway:
    being one pick off is noise next to a typical adp_stdev of 4-6 picks.
    """
    pick = max(1, int(math.floor(overall_pick + 0.5)))
    round_number = math.ceil(pick / teams)
    pick_in_round = pick - (round_number - 1) * teams

    if pick_in_round <= teams / 3:
        third = "Beginning of"
    elif pick_in_round <= 2 * teams / 3:
        third = "Middle of"
    else:
        third = "End of"
    return f"{third} Round {round_number}"


def format_pick(overall_pick, teams):
    """
    Turns an overall pick number into compact 'Rd.Pk' form: 13th overall is
    '3.01' in a 6-team league and '2.01' in a 12-team one.

    This is why the board stores ADP as an overall pick number and formats it
    at write time rather than shipping FFC's own `adp_formatted` string. FFC
    returns Rd.Pk already rendered for a 12-team draft, which is silently
    wrong on a 6-team board -- '2.01' would print next to a player who
    actually goes in the third round of your draft.

    The ORDERING is league-agnostic and transfers fine; the round labels are
    not. See build_notes() for the caveat that survives this conversion.
    """
    pick = max(1, int(math.floor(overall_pick + 0.5)))
    round_number = math.ceil(pick / teams)
    pick_in_round = pick - (round_number - 1) * teams
    return f"{round_number}.{pick_in_round:02d}"


def compute_draft_targets(players, teams):
    """
    Adds `rank`, `value_delta`, and `draft_target`.

    Sort order is (has_adp, vor) descending. Players with no real ADP are
    hard-capped below every ADP-bearing player regardless of how good the
    stats say they are -- if the market has no opinion on someone, the
    model's opinion alone isn't worth a pick you could spend on a known
    quantity.

    Draft target answers "when do I actually take him," which is not the
    same as "how good is he":

      - Bargain (value_delta >= 0, model likes him more than the market):
        target = his real ADP minus one standard deviation of that ADP.
        You wait as long as you safely can instead of reaching to his
        pure-value rank, and the cushion is his OWN volatility -- a
        consensus player needs less room than a divisive one.

      - Market premium (value_delta < 0): show 'Fair value: Round X' from
        the model's own rank. Don't pay up for what the stats don't
        support; take him only if he actually falls that far.

      - No ADP: plain model-rank slot, since there's no market to compare.
    """
    # Ties on VOR break by ADP (earlier ADP wins), then by player_name.
    #
    # ADP as a TIEBREAKER does not violate the statistics-only rule --
    # same standing as depth charts in FEATURE_SPEC.md. It never enters
    # adjusted_fantasy_points_per_game or vor, so it cannot move a player
    # past anyone the model actually separated. It only decides the order
    # of players the model rated *identically*, where the alternative is
    # alphabetical, i.e. arbitrary.
    #
    # Exact VOR ties are real right now: rookies sharing a position/round
    # cohort get byte-identical projections (Jeremiyah Love and Jadarian
    # Price), so some third key is required for a reproducible board.
    # player_name stays as a final key because no-ADP players have a null
    # adp and would otherwise still be nondeterministic among themselves.
    # Phase 12 removes the ties themselves.
    # out_for_season sorts FIRST, so those players land below everyone --
    # below even the no-ADP block. They keep a real VOR so you can see
    # what they'd have been worth, but they can never surface as a pick.
    players = players.sort(
        ["out_for_season", "has_adp", "vor", "adp", "player_name"],
        descending=[False, True, True, False, False],
        nulls_last=True,
    ).with_row_index("rank", offset=1)

    # value_delta = where the market takes him minus where the model ranks
    # him. Positive = the model is higher on him than the room is.
    players = players.with_columns(
        pl.col("adp").rank(method="dense").cast(pl.Int32).alias("adp_rank")
    ).with_columns(
        (pl.col("adp_rank") - pl.col("rank").cast(pl.Int32)).alias("value_delta")
    )

    rows = players.to_dicts()
    targets = []
    adp_slots = []
    for row in rows:
        # ADP re-expressed in THIS league's draft shape, not FFC's 12-team one.
        if row["has_adp"] and row["adp"] is not None:
            adp_slots.append(format_pick(row["adp"], teams))
        else:
            adp_slots.append(None)

        if row["out_for_season"]:
            # No round, ever. The whole point is that this player must not
            # read as takeable at any price -- and his value_delta gets
            # blanked in write_workbook for the same reason.
            targets.append("DO NOT DRAFT — out for season")
        elif not row["has_adp"]:
            targets.append(format_slot(row["rank"], teams))
        elif row["value_delta"] is not None and row["value_delta"] >= 0:
            cushion = row["adp_stdev"] or 0.0
            targets.append(format_slot(row["adp"] - cushion, teams))
        else:
            targets.append("Fair value: " + format_slot(row["rank"], teams))

    return players.with_columns([
        pl.Series("draft_target", targets),
        pl.Series("adp_slot", adp_slots, dtype=pl.String),
    ])


def build_notes(replacement_ranks, teams, rounds, config=None, starter_ranks=None):
    """The explanatory block at the top of the sheet. It lives here so the
    board always ships with an accurate description of how it was built."""
    order = [p for p in MODELED_POSITIONS if p in replacement_ranks]
    levels = " / ".join(f"{p}{replacement_ranks[p]}" for p in order)

    season = (config or {}).get("fantasy_season_length")
    weeks = (config or {}).get("regular_season_weeks")

    replacement_note = ""
    if starter_ranks:
        old = " / ".join(f"{p}{starter_ranks[p]}" for p in order
                         if p in starter_ranks)
        replacement_note = (
            f"CHANGED IN v11: replacement level used to be the last STARTER ({old}) and is now "
            f"the last player DRAFTED ({levels}) -- {teams * rounds} picks minus "
            f"{teams * UNMODELED_SLOTS_PER_TEAM} for K/DST, split by the position mix of that "
            "many players in ADP order. The old rule was defensible at 12 teams and wrong at 6: "
            "it set replacement at QB6 in a league where QB7 through QB32 are all on waivers, so "
            "the board was telling you to spend an early pick on a quarterback you could stream. "
            "Every VOR on this sheet is larger than it was in v10 because the bar moved down; "
            "what matters is that it moved down FURTHER at RB and WR than at QB and TE.\n"
        )

    availability_note = ""
    if season:
        window = f"weeks {weeks[0]}-{weeks[1]}" if weeks else f"{season:g} weeks"
        availability_note = (
            f"Exp Gm / Exp Pts = availability, deliberately kept OUT of Rank and VOR. Exp Gm is "
            f"this league's REGULAR season ({window}, {season:g} games) minus known PUP/NFI "
            "absence; Exp Pts is Adj PPG x Exp Gm. The denominator is the regular season, not "
            "the 17-week NFL calendar, because games after your league's final are worth nothing. "
            "A four-game PUP absence therefore costs a different share here than in a league with "
            "a different schedule, which is why the same player can be worth more in one of your "
            "two leagues than the other. Adj PPG stays a pure per-game rate: an injury makes a "
            "player play fewer games, not play worse in the ones he plays.\n"
        )

    return (
        "Rank/VOR = who's best by the stats-only model. VOR = points per game above that "
        f"position's replacement level ({levels}), which is why a 26-PPG quarterback does not "
        "outrank a 20-PPG receiver -- in a 1-QB league the gap over the freely available "
        "alternative is what a pick actually buys.\n"
        + replacement_note
        + availability_note
        + "Why (value drivers) = what moved him off an AVERAGE player at his position, biggest "
        "effect first, in points per game. These are the actual terms of the fitted model, not a "
        "description of it: they sum to the Sit Adj column beside them. '62% team share' reading "
        "negative is not a typo -- the model predicts the CHANGE from a player's own trailing "
        "baseline, and a back already holding most of his backfield has that usage priced in.\n"
        "Draft Target = when to actually take him. For bargains (Value Δ >= 0) it is real ADP "
        "minus a one-standard-deviation cushion (his own adp_stdev), so you wait as late as you "
        "safely can rather than reaching to pure value. For players the market likes MORE than "
        "the model (Value Δ < 0) it shows \"Fair value: Round X\" -- the model's own rank. "
        "Don't pay a premium the stats don't support; take him only if he falls that far.\n"
        "Value Δ = ADP rank minus model rank; positive = model likes him more than the market. "
        "Sit Adj = the situational adjustment applied to his raw statistical baseline; it is now "
        "two-sided (v7 and earlier were negative for every player -- the regression intercept was "
        "missing from the code, forcing a uniform penalty of 3.59 RB / 2.37 WR / 1.58 TE).\n"
        f"Gray rows = ranked past pick {teams * rounds}, the last pick of a {teams}-team "
        f"{rounds}-round draft. Pink rows = no real ADP, hard-capped below every ADP-bearing "
        "player regardless of stats. Recent Injury (red) = ended the 2025 regular season on IR, "
        "reference only.\n"
        "DARK RED rows = flagged OUT_SEASON in injury_overrides.csv: sorted below everything, "
        "Value Δ blanked, no draft round shown. That file is maintained by hand because the "
        "model cannot see 2026 preseason injuries -- and because refreshing ADP without it is "
        "actively harmful: the market drops an injured player instantly while the model's "
        "trailing-average projection does not move, so the gap gets labelled a bargain.\n"
        "Rookies (Rook = R) take no situational adjustment and share one cohort baseline per "
        "position/round, so two rookies of the same draft round can tie exactly. K and DST are "
        "not modeled; draft those separately.\n"
        + adp_caveat(teams)
    )


def adp_caveat(teams, source_teams=12):
    """
    The honest label on the ADP columns.

    ADP (Ovr) is an overall pick number and ADP (Rd.Pk) re-expresses it in this
    league's draft shape. Both are derived from FFC mock drafts run at
    `source_teams`. Rescaling preserves the ORDER players come off the board;
    it does not reproduce the BEHAVIOR of a different-sized draft. In a 6-team
    league only half as many players are taken, so scarcity at QB and TE
    largely disappears and those positions genuinely go later than a rescaled
    12-team ADP implies. The column is a reference for order, not a prediction
    of when someone goes in your draft.
    """
    base = (
        f"ADP (Ovr) = overall pick number in {source_teams}-team FFC mock drafts. "
        f"ADP (Rd.Pk) is that same number re-expressed for a {teams}-team draft "
        f"(pick 13 reads {format_pick(13, teams)} here)."
    )
    if teams == source_teams:
        return base
    return (
        base + " CAUTION: rescaling preserves the ORDER players go, not the "
        f"BEHAVIOR of a {teams}-team draft. Only {teams} of every {source_teams} "
        "picks happen here, so positional scarcity is much weaker -- QB and TE in "
        "particular should be expected to last longer than this column suggests. "
        "Treat it as a consensus ranking, not a predicted draft slot."
    )


def write_workbook(board, replacement_ranks, config, output_path, build_note=None,
                   starter_ranks=None):
    """Writes the formatted sheet. Values only -- no formulas, since every
    number here is a model output that would be wrong if a user edited a
    cell and Excel recalculated something downstream from it."""
    teams = config["num_teams"]
    rounds = config["total_rounds"]
    last_pick = teams * rounds

    wb = Workbook()
    ws = wb.active
    ws.title = "Draft Board"

    n_cols = len(COLUMNS)
    last_col = get_column_letter(n_cols)

    # Title
    ws.merge_cells(f"A1:{last_col}1")
    title = ws["A1"]
    title.value = (
        f"{config['league_name']} League — 2026 Draft Board "
        "(Statistics-Only Model, VOR Draft Order)"
    )
    title.font = Font(name=FONT_NAME, size=14, bold=True)
    ws.row_dimensions[1].height = 22

    # Notes block
    ws.merge_cells(f"A2:{last_col}6")
    notes = ws["A2"]
    notes.value = build_notes(replacement_ranks, teams, rounds, config, starter_ranks)
    notes.font = Font(name=FONT_NAME, size=9, color="555555")
    notes.alignment = Alignment(wrap_text=True, vertical="top")

    # Header
    header_row = 8
    thin_bottom = Border(bottom=Side(style="thin"))
    for i, (label, width) in enumerate(COLUMNS, start=1):
        cell = ws.cell(header_row, i, label)
        cell.font = Font(name=FONT_NAME, size=11, bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", start_color=HEADER_FILL)
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = thin_bottom
        ws.column_dimensions[get_column_letter(i)].width = width

    # Data
    for offset, row in enumerate(board.to_dicts()):
        r = header_row + 1 + offset
        rank = row["rank"]
        has_adp = bool(row["has_adp"])

        out_for_season = bool(row["out_for_season"])

        if out_for_season:
            fill_color = OUT_FILL
        elif not has_adp:
            fill_color = NO_ADP_FILL
        elif rank > last_pick:
            fill_color = UNDRAFTABLE_FILL
        else:
            fill_color = POSITION_FILLS.get(row["position"], NO_ADP_FILL)
        fill = PatternFill("solid", start_color=fill_color)

        # Value Δ is blanked for out players. It would otherwise read
        # hugely positive -- the market drops an injured player while the
        # model's trailing-average projection doesn't move -- and that gap
        # is exactly what the column normally labels "bargain."
        values = [
            rank,
            row["position"],
            row["player_name"],
            row["adjusted_fantasy_points_per_game"],
            row.get("situational_adjustment"),
            row["vor"],
            row.get("expected_games"),
            row.get("expected_total_points"),
            row.get("value_drivers"),
            row["draft_target"],
            row["team"],
            row["adp"] if has_adp else None,
            row["adp_slot"] if has_adp else None,
            row["value_delta"] if (has_adp and not out_for_season) else None,
            has_adp,
            row["bye"] if has_adp else None,
            "R" if row["is_rookie"] else None,
            ("OUT (2026)" if out_for_season
             else ("INJURED" if row["recent_major_injury"] else None)),
            row["games_played"],
            row.get("age"),
            # Stored as a share-per-season slope (0.021 = +2.1 points of
            # team share per year); rendered in percentage points, which
            # is the unit the number is actually legible in.
            (row.get("usage_trend_share") * 100.0
             if row.get("usage_trend_share") is not None else None),
            row.get("trend_seasons_used"),
            row.get("injury_note"),  # Notes -- override note, else blank for draft day
        ]

        # Left-align the text columns; everything else centers.
        left_aligned = {
            COLUMN_INDEX["Player"],
            COLUMN_INDEX["Why (value drivers)"],
            COLUMN_INDEX["Notes (manual)"],
        }
        for i, value in enumerate(values, start=1):
            cell = ws.cell(r, i, value)
            cell.fill = fill
            cell.font = Font(name=FONT_NAME, size=11)
            cell.alignment = Alignment(
                horizontal="left" if i in left_aligned else "center"
            )

        # Looked up by NAME, not position. Phase 11 adds three columns and
        # the old hardcoded indices would all have shifted silently -- the
        # number format is cosmetic, so nothing would have failed loudly.
        for label, fmt in (
            ("Adj PPG", "0.0"),
            ("Sit Adj", "+0.0;-0.0;0.0"),
            ("VOR", "0.0"),
            ("Exp Gm", "0.0"),
            ("Exp Pts", "0"),
            ("ADP (Ovr)", "0.0"),
            ("Value Δ (picks)", "+0;-0;0"),
            ("Age", "0.0"),
            ("Usage Trend", "+0.0;-0.0;0.0"),
        ):
            ws.cell(r, COLUMN_INDEX[label]).number_format = fmt

        # "Has ADP" is kept for filtering but rendered invisible -- white
        # text on the row fill -- so it doesn't add visual noise.
        ws.cell(r, COLUMN_INDEX["Has ADP"]).font = Font(
            name=FONT_NAME, size=11, color="FFFFFF"
        )

        if out_for_season:
            for label in ("Player", "Draft Target", "Recent Injury"):
                cell = ws.cell(r, COLUMN_INDEX[label])
                cell.fill = PatternFill("solid", start_color=INJURY_FILL)
                cell.font = Font(name=FONT_NAME, size=11, color="FFFFFF", bold=True)
        elif row["recent_major_injury"]:
            injury = ws.cell(r, COLUMN_INDEX["Recent Injury"])
            injury.fill = PatternFill("solid", start_color=INJURY_FILL)
            injury.font = Font(name=FONT_NAME, size=11, color="FFFFFF", bold=True)

    last_row = header_row + board.height
    ws.freeze_panes = "D9"
    ws.auto_filter.ref = f"A{header_row}:{last_col}{last_row}"

    output_path = Path(output_path)
    write_build_history(wb, output_path, config, build_note)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    return output_path


def write_build_history(workbook, output_path, config, build_note):
    """
    Appends this build to the in-workbook log, carrying prior rows forward.

    Read the prior history BEFORE the new file is saved over it -- the read
    happens here rather than in build_board() so the two can't drift apart.
    """
    prior = read_build_history(output_path)
    sheet = workbook.create_sheet(HISTORY_SHEET)

    headers = ["Build", "Built (local time)", "Model ver", "Git", "League", "Note"]
    for i, label in enumerate(headers, start=1):
        cell = sheet.cell(1, i, label)
        cell.font = Font(name=FONT_NAME, size=11, bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", start_color=HEADER_FILL)
        sheet.column_dimensions[get_column_letter(i)].width = (
            8 if i == 1 else 20 if i in (2, 5) else 11 if i in (3, 4) else 60
        )

    for offset, row in enumerate(prior):
        for i, value in enumerate(row[: len(headers)], start=1):
            sheet.cell(2 + offset, i, value).font = Font(name=FONT_NAME, size=10)

    new_row = 2 + len(prior)
    values = [
        len(prior) + 1,
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        MODEL_VERSION,
        git_short_hash(),
        config["league_name"],
        build_note or "",
    ]
    for i, value in enumerate(values, start=1):
        cell = sheet.cell(new_row, i, value)
        cell.font = Font(name=FONT_NAME, size=10, bold=True)
    sheet.cell(new_row, 6).alignment = Alignment(wrap_text=True, vertical="top")


def build_board(features_path=FEATURES_PATH, output_path=None,
                config_path=CONFIG_PATH, version=MODEL_VERSION, build_note=None):
    config = load_config(config_path)
    teams = config["num_teams"]

    players = pl.read_csv(features_path).filter(
        pl.col("position").is_in(MODELED_POSITIONS)
    )

    # CSV round-trips booleans as "true"/"false" strings; normalize once
    # here so every downstream check is a real boolean.
    for column in ["has_adp", "is_rookie", "recent_major_injury"]:
        players = players.with_columns(
            pl.col(column).cast(pl.String).str.to_lowercase().eq("true").alias(column)
        )

    players = apply_injury_overrides(players)
    players = compute_expected_points(players, config)
    players = build_value_drivers(players)

    starter_ranks = compute_starter_ranks(config)
    replacement_ranks = compute_replacement_ranks(config, players)
    print(f"Replacement ranks: {replacement_ranks} "
          f"(was, by starter slots: {starter_ranks})")

    board = compute_vor(players, replacement_ranks)
    board = compute_draft_targets(board, teams)

    if output_path is None:
        output_path = PROJECT_ROOT / f"2026_{league_slug(config)}_Board_v{version}.xlsx"

    written = write_workbook(board, replacement_ranks, config, output_path, build_note,
                             starter_ranks=starter_ranks)
    history_count = len(read_build_history(written))
    print(f"Wrote {board.height} players to {written}")
    print(f"{config['league_name']} — {teams} teams, model v{version}, "
          f"build #{history_count} ({git_short_hash()})")

    top = board.head(10).select(
        ["rank", "position", "player_name", "adjusted_fantasy_points_per_game",
         "vor", "draft_target"]
    )
    print("\nTop 10:")
    print(top)
    return board


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the 2026 Excel draft board.")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH),
                        help="league config json (default: league_config.json, "
                             "i.e. Lebron James). Use league_config_dunlap.json "
                             "for the 6-team Dunlap Family board.")
    parser.add_argument("--version", type=int, default=MODEL_VERSION,
                        help="model version for the output filename; bump only "
                             "when the MODEL changes, not for a data refresh")
    parser.add_argument("--output", type=str, default=None,
                        help="explicit output path (overrides the derived name)")
    parser.add_argument("--features", type=str, default=str(FEATURES_PATH),
                        help="path to player_features.csv")
    parser.add_argument("--note", type=str, default=None,
                        help="what changed in this build; recorded in the "
                             "Build History sheet")
    args = parser.parse_args()

    build_board(
        features_path=args.features,
        output_path=args.output,
        config_path=args.config,
        version=args.version,
        build_note=args.note,
    )
