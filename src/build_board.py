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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "player_features.csv"
CONFIG_PATH = PROJECT_ROOT / "league_config.json"

# MODEL_VERSION bumps when the model changes -- new weights, new features,
# a refit. It does NOT bump for a data refresh (new ADP pull, updated injury
# overrides), because nothing about the ranking logic moved. Rebuilds of the
# same model version are told apart by the Build History sheet, not by the
# filename, which is what keeps one file per league instead of a pile of
# near-identical spreadsheets.
MODEL_VERSION = 9

HISTORY_SHEET = "Build History"
INJURY_OVERRIDES_PATH = PROJECT_ROOT / "injury_overrides.csv"

# Status in injury_overrides.csv that removes a player from contention.
# Any OTHER status string is recorded in the Notes column but changes no
# ranking -- so you can jot "QUESTIONABLE - hamstring" as a reminder
# without it silently moving anyone.
OUT_STATUS = "OUT_SEASON"

# How the single FLEX slot is expected to be filled across the league.
# This is a modeling ASSUMPTION, not a league rule -- it's the standard
# full-PPR split and it only moves replacement levels by a fraction of a
# roster spot. Documented here because it silently shifts every VOR.
FLEX_SPLIT = {"RB": 0.40, "WR": 0.40, "TE": 0.20}

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
    ("VOR", 8), ("Draft Target", 24), ("Team", 7), ("ADP (Ovr)", 10),
    ("ADP (Rd.Pk)", 12), ("Value Δ (picks)", 12), ("Has ADP", 4), ("Bye", 6),
    ("Rook", 6), ("Recent Injury", 12), ("GP (sample)", 11),
    ("Notes (manual)", 45),
]


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

    File format -- player_name,status,note
    Only status == OUT_STATUS changes the ranking. Anything else is
    carried into the Notes column and ignored by the model.

    UNMATCHED NAMES RAISE. A typo'd name silently matching nothing would
    leave an injured player sitting in the draftable pool looking like a
    steal -- the exact failure this file exists to prevent -- so it fails
    loudly instead.
    """
    empty = players.with_columns([
        pl.lit(False).alias("out_for_season"),
        pl.lit(None, dtype=pl.String).alias("injury_note"),
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

    overrides = overrides.select([
        "player_name",
        (pl.col("status").cast(pl.String).str.to_uppercase().str.strip_chars()
         == OUT_STATUS).alias("out_for_season"),
        pl.col("note").cast(pl.String).alias("injury_note"),
    ])

    joined = players.join(overrides, on="player_name", how="left").with_columns(
        pl.col("out_for_season").fill_null(False)
    )

    out_count = joined.select(pl.col("out_for_season").sum()).item()
    print(f"Injury overrides: {overrides.height} entries, {out_count} marked {OUT_STATUS}")
    return joined


def compute_replacement_ranks(config):
    """
    How many players at each position get drafted as starters league-wide.
    That Nth player is 'replacement level' -- the guy you could have for
    free -- so his PPG is the baseline every other player is measured
    against.

    Derived from league_config.json rather than hardcoded, so changing
    league size or roster slots flows through automatically. For the
    current 12-team setup this reproduces v7's QB12 / RB29 / WR29 / TE14.
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


def build_notes(replacement_ranks, teams, rounds):
    """The explanatory block at the top of the sheet. It lives here so the
    board always ships with an accurate description of how it was built."""
    levels = " / ".join(f"{p}{r}" for p, r in replacement_ranks.items())
    return (
        "Rank/VOR = who's best by the stats-only model. VOR = points per game above that "
        f"position's replacement level ({levels}), which is why a 26-PPG quarterback does not "
        "outrank a 20-PPG receiver -- in a 1-QB league the gap over the freely available "
        "alternative is what a pick actually buys.\n"
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


def write_workbook(board, replacement_ranks, config, output_path, build_note=None):
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
    notes.value = build_notes(replacement_ranks, teams, rounds)
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
            row.get("injury_note"),  # Notes -- override note, else blank for draft day
        ]

        for i, value in enumerate(values, start=1):
            cell = ws.cell(r, i, value)
            cell.fill = fill
            cell.font = Font(name=FONT_NAME, size=11)
            cell.alignment = Alignment(
                horizontal="left" if i in (3, n_cols) else "center"
            )

        # Column indices below are 1-based positions in COLUMNS. Adding the
        # "ADP (Ovr)" column at position 9 shifted everything after it right
        # by one -- if you insert another column, these all move again.
        ws.cell(r, 4).number_format = "0.0"
        ws.cell(r, 5).number_format = "+0.0;-0.0;0.0"
        ws.cell(r, 6).number_format = "0.0"
        ws.cell(r, 9).number_format = "0.0"       # ADP (Ovr)
        ws.cell(r, 11).number_format = "+0;-0;0"  # Value Δ

        # "Has ADP" is kept for filtering but rendered invisible -- white
        # text on the row fill -- so it doesn't add visual noise.
        ws.cell(r, 12).font = Font(name=FONT_NAME, size=11, color="FFFFFF")

        if out_for_season:
            for i in (3, 7, 15):  # Player, Draft Target, Recent Injury
                cell = ws.cell(r, i)
                cell.fill = PatternFill("solid", start_color=INJURY_FILL)
                cell.font = Font(name=FONT_NAME, size=11, color="FFFFFF", bold=True)
        elif row["recent_major_injury"]:
            injury = ws.cell(r, 15)
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

    replacement_ranks = compute_replacement_ranks(config)
    print(f"Replacement ranks: {replacement_ranks}")

    board = compute_vor(players, replacement_ranks)
    board = compute_draft_targets(board, teams)

    if output_path is None:
        output_path = PROJECT_ROOT / f"2026_{league_slug(config)}_Board_v{version}.xlsx"

    written = write_workbook(board, replacement_ranks, config, output_path, build_note)
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
