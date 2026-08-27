# Phase 15 scratch

Exploration scripts from Phase 15. **None of these ship anything.** They
are kept for the same reason `PHASE_8-14_PLAN.md` keeps the ideas that
were cut: what got rejected, and on what evidence, is most of what a
model is.

Run them from the repo root, not from this directory — they import `src`
and read `data/` relative to the working directory:

```bash
python notebooks/phase15/scratch_qb_battery.py
```

Most need `data/backtest_features.csv`, which `python -m src.backtest`
builds. Each script states its own question and decision rule in a module
docstring; read that before the code.

## What Phase 15 asked, and what came back

Phase 15 tested six candidate changes to a decision. **One shipped.**

| Question | Verdict |
|---|---|
| Can missed games be predicted from injury history? | No. Injury-caused absence persists at r = 0.06–0.14 |
| Do QBs need a situational model? | 21 of 23 features cut. But they revert to the mean, hard |
| Can rookie RB/WRs be told apart within a round? | No. Nothing beat a constant |
| Is games played the right confidence unit? | Yes, and K = 2 is right |
| Should the board rank on expected points instead of rate? | No. The gain was a not-in-the-NFL detector |
| Is an outside injury-risk feed worth buying? | No. There is no signal in the data to buy |

The one that shipped is `src/qb_reversion.py`, gated by
`data/qb_reversion_gate.json`.

## The scripts

### 15a — availability and injury prediction

- **`scratch_avail_explore.py`** — measures the selection hole
  (`backtest_features.csv` contains no zero-game seasons) and the raw
  correlations. Shows that every injury-report feature correlates
  *positively* with availability, because only real players get listed on
  an injury report.
- **`scratch_avail_model.py`** — separates role from health and runs the
  nested blocks. Injury history fails the 1-SE gate; predicting
  injury-caused absence directly beats nothing at all.
- **`scratch_designation_table.py`** — week-1 designation to games played,
  whole population.
- **`scratch_designation_relevant.py`** — same, restricted to
  fantasy-relevant players, which is who the override file is about.
- **`scratch_designation_stability.py`** — the check that stopped a change
  from shipping. The PUP numbers survive every stability test and still
  should not be used, because they measure Reserve/PUP while
  `injury_overrides.csv` tags Active/PUP. Different designation, same
  three letters.

### 15b — quarterbacks

- **`scratch_qb_battery.py`** — all 23 candidate features through
  `holdout.run_holdout` at QB. Twenty-one cut. `baseline_ppg` survives
  with a negative coefficient, which is mean reversion.
- **`scratch_qb_reversion.py`** — separates real reversion from the
  `delta = actual − baseline` arithmetic artifact by switching the scoring
  target to `actual_ppg`, where the artifact cannot help.
- **`scratch_qb_gate_design.py`** — the support guard. Shows unguarded
  reversion scoring better while inflating 50 thin quarterbacks in the
  live pool, which is the Phase 11 CP5 failure. This is why
  `qb_reversion.py` guards at 16 games and why the guard was not chosen by
  held-out gain.

### 15c — rookies

- **`scratch_rookie_battery.py`** — nine pre-snap features at every
  position. Exact draft `pick` fails at RB and WR, along with everything
  else. The README's second known limitation stands.

### 15d — confidence

- **`scratch_confidence.py`** — four estimators against the shipped
  games-based shrinkage. Opportunity-based confidence loses outright.

  Note this script scores raw baselines on the full pool and prefers a
  larger K. `src/shrinkage.py` scores the composite prediction on the
  low-confidence subgroup and does not, at 0.8 SE against CP5's 2-SE bar.
  When two instruments disagree, the one with the pre-committed decision
  rule wins. K stayed at 2.

### 15e — what the board ranks on

- **`scratch_rank_basis.py`** — rebuilds the frame without the selection
  filter (6,114 player-seasons, 46% zero-game) and compares ranking on
  rate against ranking on rate × E[games]. Looks decisive: +13% more
  actual points in the top 100, 7 of 7 seasons.
- **`scratch_rank_basis_stress.py`** — takes that result apart. Restricted
  to players who held a real role, the gain falls to +1.3%; on the top 24
  it wins 4 of 7 seasons. Rejected.

  Worth keeping from it: role-adjusted prior availability, which adds
  injury weeks back so the feature measures the role a player held rather
  than the games his body allowed. The 20 established producers who lost
  most of a season to injury played 10.7 games the next year; raw
  availability predicts 4.8, role-adjusted predicts 11.4. Any future
  veteran expected-games model should be built on that version.
