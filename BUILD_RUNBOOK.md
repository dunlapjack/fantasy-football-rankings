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

### 1. Update the injury file by hand

`injury_overrides.csv` is the one input the pipeline never touches, and it is the one most
likely to be stale.

    OUT_SEASON     zeroes the player out. Ranking and Exp Pts both.
    PUP / NFI      four games missed by default, or set games_missed.
                   Touches Exp Pts only, never rank.
    QUESTIONABLE   a note on the board. Changes no number.

An unmatched name **raises**, which is the behaviour you want. A *missing* name is silent,
and leaves an injured player sitting in the pool looking like a bargain — which is the
failure this file exists to prevent.

### 2. Refresh and rebuild

```bash
python -m src.pipeline

python -m src.build_board --config league_config_32team.json      --version 15 --note "pre-draft refresh"
python -m src.build_board --config league_config_lebronjames.json --version 15 --note "pre-draft refresh"
python -m src.build_board --config league_config_dunlap.json      --version 15 --note "pre-draft refresh"
```

Each build must print both gate lines before it writes anything:

```
Holdout gate: PASSED (folds [2025, 2024, 2023])
Playing-time gate: PASSED (ships predictor B, n=719)
```

### 3. Regenerate the stress test — 32-team only

```bash
python -m src.qb_stress_test
```

**Do not skip this after a rebuild.** `QB59_stress_test.xlsx` is computed from the board's
Adj PPG, so a refresh silently invalidates it. A worst-case rank computed from superseded
projections, sitting next to a current board, is worse than not having the file.

### 4. Read what the refresh did

```bash
python -m src.compare_boards <previous board>.xlsx 2026_32Team_Board_v15.xlsx --focus 200
python -m src.sanity_top_n --top 60
```

Two things to watch:

- **A large move with `dPPG` near zero is a sort-order effect, not a revaluation.** Check
  the GAINED/LOST ADP flag. `compute_draft_targets` sorts by `(out_for_season, has_adp,
  vor, ...)`, so `has_adp` gates *above* VOR — a player entering the FFC feed vaults over
  the entire no-ADP block without the model changing its mind about him.
- **The top-60 position mix.** If the QB count moves on the 32-team board, ADP has shifted
  `expected_drafted` and you are near the QB49/50 cliff.

`sanity_top_n` checks only mechanical things — duplicates, nulls, retirees, drivers citing
dead features. Everything it prints under "board LIKES / FADES" is a judgement call it is
deliberately refusing to make for you.

---

## Draft day, 32-team superflex

Have both files open: `2026_32Team_Board_v15.xlsx` and `QB59_stress_test.xlsx`.

**Draft off the `Worst rank` column.** 46 players are top-60 under every QB scenario from
45 to 70 — RB 20, WR 17, QB 7, TE 2. Those are the ones the board is genuinely confident
about.

**The QB6–QB16 tier is a live read, not a pre-commitment.** Those fourteen swing 40–80
places on the QB assumption. If quarterbacks go fast in rounds 1–2, the mock's QB59 is
right and that tier is a real value band. If they are still sitting at pick 60, replacement
is shallower than modelled and they are forty ranks worse than the board says.

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
