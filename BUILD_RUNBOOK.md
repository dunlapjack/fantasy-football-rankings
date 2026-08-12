# Build runbook — Phase 13.5 → v14 boards

Two passes. The first proves the wiring is right against a clean control; the second
produces the boards you draft from. Doing it in this order means that if a rank looks
wrong later, you already know it wasn't Phase 13.5.

Run everything from the repo root with the venv active.

---

## Pass 0 — five seconds, no data

```bash
python -m src.playing_time --selftest
```

Checks the board-side hook against a synthetic frame built to be nasty in the ways the
real one is: picks as strings, an undrafted rookie with an empty pick, a veteran who must
come out at zero, a rookie already on PUP, and a position with no fitted model. Exits
non-zero on any mismatch.

This exists because the Aug 12 build died on a dtype, not on maths. Everything expensive
had passed — the gate had validated the arithmetic on 719 players — and `pick` arrives
from a CSV as a string, so the first rookie hit `'str' - 'float'`. Run this before any
build; it costs nothing.

---

## Pass 1 — verify the wiring (no data refresh)

Phase 13.5 makes one precise claim: it changes `expected_games` for rookies, therefore
Exp Gm and Exp Pts, and it moves **no player's rank**. Pass 1 executes that claim instead
of trusting it.

Builds go to `verify/` so they don't collide with the real v14 filenames.

```bash
mkdir verify

python -m src.build_board --config league_config_32team.json      --version 14 --output verify/32team_prerefresh.xlsx  --note "Phase 13.5 wiring check"
python -m src.build_board --config league_config_lebronjames.json --version 14 --output verify/12team_prerefresh.xlsx  --note "Phase 13.5 wiring check"
python -m src.build_board --config league_config_dunlap.json      --version 14 --output verify/6team_prerefresh.xlsx   --note "Phase 13.5 wiring check"
```

Each build should print two gate lines before it writes anything:

```
Holdout gate: PASSED (folds [2025, 2024, 2023])
Playing-time gate: PASSED (ships predictor B, n=719)
Rookie availability: N rookies marked down on Exp Pts (rank unaffected by construction)
```

Then check all three against the frozen v13:

```bash
python -m src.compare_boards boards_v13_frozen/2026_32Team_Board_v13.xlsx verify/32team_prerefresh.xlsx --expect rank-identical
python -m src.compare_boards boards_v13_frozen/2026_12Team_Board_v13.xlsx verify/12team_prerefresh.xlsx --expect rank-identical
python -m src.compare_boards boards_v13_frozen/2026_6Team_Board_v13.xlsx  verify/6team_prerefresh.xlsx  --expect rank-identical
```

**What passing looks like.** Zero ranks moved, VOR and Adj PPG unchanged for all 1088
players, and Exp Gm / Exp Pts changed for rookies only. The comparator exits 0 and says so.

**What failing means.** If a rank moved, the hook is wired wrong — most likely
`expected_games_for_rookies` is running somewhere that feeds VOR rather than only Exp Pts.
Do not continue to Pass 2. The comparator exits 1 and names what moved.

**If the playing-time gate line says `ships predictor ?`** — the gate JSON on disk predates
the `ship` field. Harmless, but re-run `python -m src.playing_time --gate` to get a complete
record with the incremental test in it.

---

## Pass 2 — refresh the data and build the real boards

Your features were built Aug 6. This pulls current ADP, injury designations and depth
charts.

```bash
python -m src.pipeline
```

Then rebuild the injury overrides by hand if anything has happened since Aug 7 —
`injury_overrides.csv` is hand-maintained and the pipeline does not touch it. An unmatched
name raises rather than passing silently, which is the behaviour you want.

```bash
python -m src.build_board --config league_config_32team.json      --version 14 --note "Phase 13.5 + data refresh"
python -m src.build_board --config league_config_lebronjames.json --version 14 --note "Phase 13.5 + data refresh"
python -m src.build_board --config league_config_dunlap.json      --version 14 --note "Phase 13.5 + data refresh"
```

Version stays at 14. The convention in `build_board.py` is explicit — bump on a MODEL
change, not a data refresh — and Pass 2 is a data refresh of the same model. That is also
why Pass 1 wrote to `verify/`: overwriting it here is intended.

Now look at what six days of data actually did:

```bash
python -m src.compare_boards boards_v13_frozen/2026_32Team_Board_v13.xlsx 2026_32Team_Board_v14.xlsx --top 60
```

Ranks **should** move here, and every move is attributable to data rather than to Phase
13.5, because Pass 1 already established the model change moves nothing. Read the top-60
position mix line: if the RB/WR/QB counts have shifted much, that is ADP moving
`expected_drafted`, and it interacts with the QB59 fragility recorded in the plan.

---

## Pass 3 — read the boards

```bash
python -m src.sanity_top_n --top 60
```

Mechanical checks only: duplicates, nulls, retirees, dead-feature drivers. Then read the
top 60 of each board yourself for football reasons the script deliberately refuses to
judge.

For the 32-team superflex board, read it alongside `QB59_stress_test.xlsx` — the
**Worst rank** column is the one to draft off, and the 46 players that stay top-60 under
every QB scenario are the ones the board is actually confident about.

---

## What is now guarded

`build_board` refuses to build if:

- the holdout gate is missing, failing, or older than `situational_weights.json` /
  `rookie_weights.json`
- `playing_time.json` exists but its gate is missing, failing, or older than the model

Absent `playing_time.json` is allowed — that is a v13 board, which is wrong about rookie
Exp Pts but was what shipped for months and affects no rank. Absent is a known state;
stale is a lie.

---

## The pre-draft refresh (repeat before each draft)

Injuries and ADP move; the model does not need refitting for either. This is a data
refresh, so `MODEL_VERSION` stays where it is.

```bash
# 1. Hand-edit injury_overrides.csv first. The pipeline never touches it.
#    OUT_SEASON and PUP/NFI change numbers; QUESTIONABLE is a note only.
#    An unmatched name RAISES, which is what you want. A missing name is
#    silent, and leaves an injured player looking like a bargain.

python -m src.pipeline

python -m src.build_board --config league_config_dunlap.json      --version 15 --note "pre-draft refresh"
python -m src.build_board --config league_config_lebronjames.json --version 15 --note "pre-draft refresh"

# 2. Read what the refresh actually did, draftable range only.
python -m src.compare_boards <previous>.xlsx 2026_6Team_Board_v15.xlsx --focus 200

# 3. Mechanical screen, then read the top 60 yourself.
python -m src.sanity_top_n --top 60
```

Two things to watch in step 2:

- **Large moves with `dPPG` near zero are sort-order effects, not revaluations.** Check the
  GAINED/LOST ADP flag. `has_adp` gates above VOR, so a player entering the FFC feed vaults
  over the entire no-ADP block without the model changing its mind about him.
- **The top-60 position mix.** If the QB count shifts on the 32-team board, ADP has moved
  `expected_drafted` and you are near the QB49/50 cliff — re-read `QB59_stress_test.xlsx`.
