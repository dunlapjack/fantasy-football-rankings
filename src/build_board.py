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
    python -m src.build_board                 # writes 2026_Draft_Board_v8.xlsx
    python -m src.build_board --version 9
    python -m src.build_board --output some/path.xlsx

Requires openpyxl (not currently in .venv):
    pip install openpyxl
"""

import argparse
import json
import math
from pathlib import Path

import polars as pl
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "player_features.csv"
CONFIG_PATH = PROJECT_ROOT / "league_config.json"

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
HEADER_FILL = "1F4E78"
INJURY_FILL = "FF0000"

COLUMNS = [
    ("Rank", 7), ("Pos", 6), ("Player", 24), ("Adj PPG", 9), ("Sit Adj", 8),
    ("VOR", 8), ("Draft Target", 24), ("Team", 7), ("ADP (Rd.Pk)", 12),
    ("Value Δ (picks)", 12), ("Has ADP", 4), ("Bye", 6), ("Rook", 6),
    ("Recent Injury", 12), ("GP (sample)", 11), ("Notes (manual)", 45),
]


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return json.load(f)


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
        # If a position has fewer players than its replacement rank,
        # fall back to the worst available rather than indexing off the end.
        index = min(rank, pool.height) - 1
        replacement_ppg = pool.select(value_column).to_series()[index]
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
    players = players.sort(
        ["has_adp", "vor", "adp", "player_name"],
        descending=[True, True, False, False],
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
    for row in rows:
        if not row["has_adp"]:
            targets.append(format_slot(row["rank"], teams))
        elif row["value_delta"] is not None and row["value_delta"] >= 0:
            cushion = row["adp_stdev"] or 0.0
            targets.append(format_slot(row["adp"] - cushion, teams))
        else:
            targets.append("Fair value: " + format_slot(row["rank"], teams))

    return players.with_columns(pl.Series("draft_target", targets))


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
        "player regardless of stats. Recent Injury (red) = ended last regular season on IR, "
        "reference only -- it cannot see playoff-time or current-offseason injuries.\n"
        "Rookies (Rook = R) take no situational adjustment and share one cohort baseline per "
        "position/round, so two rookies of the same draft round can tie exactly. K and DST are "
        "not modeled; draft those separately."
    )


def write_workbook(board, replacement_ranks, config, output_path):
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

        if not has_adp:
            fill_color = NO_ADP_FILL
        elif rank > last_pick:
            fill_color = UNDRAFTABLE_FILL
        else:
            fill_color = POSITION_FILLS.get(row["position"], NO_ADP_FILL)
        fill = PatternFill("solid", start_color=fill_color)

        values = [
            rank,
            row["position"],
            row["player_name"],
            row["adjusted_fantasy_points_per_game"],
            row.get("situational_adjustment"),
            row["vor"],
            row["draft_target"],
            row["team"],
            row["adp_formatted"] if has_adp else None,
            row["value_delta"] if has_adp else None,
            has_adp,
            row["bye"] if has_adp else None,
            "R" if row["is_rookie"] else None,
            "INJURED" if row["recent_major_injury"] else None,
            row["games_played"],
            None,  # Notes (manual) -- left blank on purpose, for draft day
        ]

        for i, value in enumerate(values, start=1):
            cell = ws.cell(r, i, value)
            cell.fill = fill
            cell.font = Font(name=FONT_NAME, size=11)
            cell.alignment = Alignment(
                horizontal="left" if i in (3, n_cols) else "center"
            )

        ws.cell(r, 4).number_format = "0.0"
        ws.cell(r, 5).number_format = "+0.0;-0.0;0.0"
        ws.cell(r, 6).number_format = "0.0"
        ws.cell(r, 10).number_format = "+0;-0;0"

        # "Has ADP" is kept for filtering but rendered invisible -- white
        # text on the row fill -- so it doesn't add visual noise.
        ws.cell(r, 11).font = Font(name=FONT_NAME, size=11, color="FFFFFF")

        if row["recent_major_injury"]:
            injury = ws.cell(r, 14)
            injury.fill = PatternFill("solid", start_color=INJURY_FILL)
            injury.font = Font(name=FONT_NAME, size=11, color="FFFFFF", bold=True)

    last_row = header_row + board.height
    ws.freeze_panes = "D9"
    ws.auto_filter.ref = f"A{header_row}:{last_col}{last_row}"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    return output_path


def build_board(features_path=FEATURES_PATH, output_path=None, version=8):
    config = load_config()
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

    replacement_ranks = compute_replacement_ranks(config)
    print(f"Replacement ranks: {replacement_ranks}")

    board = compute_vor(players, replacement_ranks)
    board = compute_draft_targets(board, teams)

    if output_path is None:
        output_path = PROJECT_ROOT / f"2026_Draft_Board_v{version}.xlsx"

    written = write_workbook(board, replacement_ranks, config, output_path)
    print(f"Wrote {board.height} players to {written}")

    top = board.head(10).select(
        ["rank", "position", "player_name", "adjusted_fantasy_points_per_game",
         "vor", "draft_target"]
    )
    print("\nTop 10:")
    print(top)
    return board


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the 2026 Excel draft board.")
    parser.add_argument("--version", type=int, default=8,
                        help="version number for the default output filename")
    parser.add_argument("--output", type=str, default=None,
                        help="explicit output path (overrides --version)")
    parser.add_argument("--features", type=str, default=str(FEATURES_PATH),
                        help="path to player_features.csv")
    args = parser.parse_args()

    build_board(
        features_path=args.features,
        output_path=args.output,
        version=args.version,
    )
