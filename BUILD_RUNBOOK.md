# Build runbook

Two different jobs live here and they are not the same length.

- **[Before each draft](#before-each-draft)** — a data refresh. ~10 minutes. This is the
  one that repeats.
- **[After a model change](#after-a-model-change)** — verification. An hour, and only when
  you have changed the model rather than the data.

Everything runs from the repo root with the venv active.

---

## Before each draft

Injuries and ADP move; the model does not need refitting for either. `MODEL_VERSION` stays
where it is — the convention in `build_board.py` is explicit that a data refresh is not a
version bump.

### 1. Update the two hand-maintained files

`injury_overrides.csv` is the one input the pipeline never touches, and it is the one most
likely to be stale.

    OUT_SEASON     zeroes the player out. Ranking and Exp Pts both.
    PUP / NFI      four games missed by default, or set games_missed.
                   Touches Exp Pts only, never rank.
    QUESTIONABLE   a note on the board. Changes no number.

An unmatched name **raises**, which is the behaviour you want. A *missing* name is silent,
and leaves an injured player sitting in the pool looking like a bargain — which is the
failure this file exists to prevent.

`position_overrides.csv` is the other one, and it is checked far less often because it
changes far less often. It corrects nflverse positions — currently one row, Travis Hunter,
who nflverse carries as a CB and who was therefore **absent from the board entirely**, not
ranked low. Add a row when a player you can see being drafted isn't on the sheet at all:

    player_name,position,note

An unmatched name raises here too, and that raise is informative: it means nflverse has no
rows for that name at all, so the player has no offensive snaps in the 2023–25 window and
this file is the wrong tool for him. Jarquez Hunter, Will Howard, Zack Kuntz, Dont'e
Houston and Lawrance McCutcheon are all in that category — missing for lack of data, not
lack of a label. (Audric Estimé was on that list until an nflverse refresh brought him
back, which is a reminder that the list is a snapshot, not a fact about the players.)

    python -m src.position_overrides     # selftest + prints what's in the file, no network

### 2. Refresh and rebuild

```bash
python -m src.pipeline
python -m src.mock_adp     # 32-team only; see below

python -m src.build_board --config league_config_32team.json      --version 17 --note "pre-draft refresh"
python -m src.build_board --config league_config_12team.json --version 17 --note "pre-draft refresh"
python -m src.build_board --config league_config_6team.json      --version 17 --note "pre-draft refresh"
python -m src.build_board --config league_config_8team.json       --version 17 --note "pre-draft refresh"
```

Each build must print both gate lines before it writes anything:

```
Holdout gate: PASSED (folds [2025, 2024, 2023])
Playing-time gate: PASSED (ships predictor B, n=719)
```

The 32-team build must also print the line that says which feed it read:

```
ADP: 32-Team Superflex PPR reads MOCK DRAFT ADP from data/mock_adp_32team.csv
```

If that line is missing, the board fell back to FFC and goes pink from round 7 on.

### 2b. The 32-team ADP feed is mock drafts, not FFC

`src/mock_adp.py` rebuilds `data/mock_adp_32team.csv` from the transcribed boards in
`data/mock_boards/`. It only needs re-running when a **new mock is added** — it does not
depend on the pipeline — but it is cheap and re-running it after `src.pipeline` costs
nothing.

Adding a mock: transcribe it into `data/mock_boards/`, one `round.pick|name|pos|team` line
per pick, add it to `MOCKS` in `src/mock_adp.py`, and re-run. The module **raises** if the
pick count doesn't match teams × rounds, and `audit()` **raises if one player resolves to
two picks inside the same mock** — the failure that put A.J. Brown in round 7 with every
other check green (Phase 13.6b). Check the position counts it prints against the live
board, and read the "two rooms disagree" list it prints at the end: a cell resolved to the
wrong human usually shows up there as a several-hundred-pick gap.

### 3. Regenerate the stress test — 32-team only

```bash
python -m src.qb_stress_test
```

**Do not skip this after a rebuild.** `QB_stress_test.xlsx` is computed from the board's
Adj PPG, so a refresh silently invalidates it. A worst-case rank computed from superseded
projections, sitting next to a current board, is worse than not having the file.

`SHIPPED` in that module is a hand-kept copy of `expected_drafted` from
`league_config_32team.json`. If you change one, change both, or the sweep is stress-testing
a board nobody shipped.

### 4. Read what the refresh did

```bash
python -m src.compare_boards 2026_32Team_Board_v16.xlsx 2026_32Team_Board_v17.xlsx --focus 200
python -m src.sanity_top_n --top 60
```

Three things to watch:

- **Movers clustered on a few TEAMS are nflverse churn, not your change.** August roster
  moves drop undrafted rookies out of the player table, which recomputes position
  competition for everyone left on those teams. The v16 → v17 diff showed 63 such movers on
  PHI/ATL/PIT/NE/CLE from three UDFAs disappearing — none of it caused by the rebuild's
  actual change. Group the movers by team before concluding anything.
- **A large move with `dPPG` near zero is a sort-order effect, not a revaluation.** Check
  the GAINED/LOST ADP flag. `compute_draft_targets` sorts by `(out_for_season, has_adp,
  vor, ...)`, so `has_adp` gates *above* VOR — a player entering the FFC feed vaults over
  the entire no-ADP block without the model changing its mind about him.
- **The top-60 position mix.** If the QB count moves on the 32-team board, check
  `expected_drafted` — you are near the QB49/50 cliff. Note that on the 32-team board ADP
  can no longer move `expected_drafted` on its own: the counts are typed in from the mocks
  and the config raises if they don't sum to teams × rounds.

`sanity_top_n` checks only mechanical things — duplicates, nulls, retirees, drivers citing
dead features. Everything it prints under "board LIKES / FADES" is a judgement call it is
deliberately refusing to make for you.

---

## Draft day, 32-team superflex

**Draft off the board.** `2026_32Team_Board_v17.xlsx`, in rank order.

### Why not the stress test's `Worst rank` column

An earlier version of this runbook said to draft off it. That was wrong, and the reason is
worth keeping because it is a general trap.

`Worst rank` is the worst rank a player holds across QB45–QB70. For a quarterback that is
his rank under **QB45–49** — the scenario we examined and rejected. QB62 is a measured
count, now off two real mocks; QB45 is a point on a sweep that nothing observed. Drafting off
`Worst rank` means hedging toward the assumption you decided not to believe.

It is worse than merely inconsistent. The two errors are **8:1 asymmetric**: waiting on QB
when the room takes 59 costs ~150 points (you start a replacement-level QB in a superflex
slot); taking one early when the room takes 45 costs ~19 (a slightly worse skill player
twenty picks later). `Worst rank` is a **minimax rule, and minimax minimizes regret in rank
space, not in points space.** Under a loss function that lopsided it does not reduce
exposure — it maximizes exposure to the expensive error.

The general form: **a robustness criterion is only conservative if the losses it is hedging
across are symmetric.** Check that before adopting one.

For RB, WR and TE the two orderings differ by at most 14–26 places, so nothing is lost
outside quarterback anyway.

### What the stress test is still for

- **A tiebreaker, not an ordering.** If two players sit adjacent on the board and one is in
  the robust 46 (RB 20, WR 18, QB 6, TE 2) and the other is not, prefer the robust one.
- **A live read in rounds 1–2.** If quarterbacks are *not* going early, that is evidence
  your room is not the mock's room, and the QB6–QB16 tier gets worse rather than better.
  Those fourteen swing 40–80 places on the assumption. This is a mid-draft update, which is
  the form the information should have taken from the start.

**Two places your judgement should outrank the board:**

- **Quarterbacks.** QB has no situational features and no baseline shrinkage — a QB's
  projection is his own trailing average and nothing else. The board fades young QBs by
  ~41 picks against ADP versus ~17 for veterans, because it has no development curve. See
  the QB section in `PHASE_8-14_PLAN.md`.
- **Rookies who share a cohort.** Rookie baselines are one value per position × round
  bucket, so two first-round backs can be literally the same player to this model. If you
  have any read separating them, it is information the board does not have.

---

## After a model change

Only when the *model* changed, not the data. The point is to prove the change did what it
claimed and nothing else.

```bash
python -m src.playing_time --selftest     # 5 seconds, no network
mkdir verify
python -m src.build_board --config league_config_32team.json --version <new> --output verify/32team.xlsx --note "wiring check"
python -m src.compare_boards <previous>.xlsx verify/32team.xlsx --expect rank-identical
```

`--expect rank-identical` exits non-zero if any rank, VOR or Adj PPG moved. Use it whenever
a change is supposed to leave ranking alone; omit it when ranks are meant to move and you
only want to see where.

Build to `verify/` so the check does not collide with the real filenames, and run it
*before* refreshing data — otherwise a rank change could be the model or six days of ADP
and you will not know which.

### What is guarded automatically

`build_board` refuses to build if:

- the holdout gate is missing, failing, or older than `situational_weights.json` /
  `rookie_weights.json`
- `playing_time.json` exists but its gate is missing, failing, or older than the model

Absent `playing_time.json` is allowed — that is a pre-13.5 board, wrong about rookie
Exp Pts but rank-neutral. Absent is a known state; stale is a lie.

---

## Appendix — the Phase 13.5 verification (Aug 12, done)

Kept as a worked example of the pattern above, not as something to repeat.

Passes 0–1 built all three boards to `verify/` from unrefreshed features and checked them
against frozen v13 with `--expect rank-identical`. All three passed: 0 of 1088 ranks moved,
231 rookies' Exp Gm and Exp Pts changed. That established Phase 13.5 was wired correctly
*before* a data refresh could muddy the picture — so when Pass 2 moved 878 ranks, every one
was attributable to data.

Two bugs surfaced during it that the rank check was not looking for: `pick` arriving as a
string, and eleven undrafted QBs assigned zero expected games. Both are recorded in
`PHASE_8-14_PLAN.md`. The lesson worth keeping is that every expensive check had passed —
the gate had validated the arithmetic on 719 held-out players — and what broke the build
was a type conversion.
