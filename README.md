# fantasy-football-rankings

A statistics-only fantasy football draft board. It pulls real NFL play-by-play and
player data, projects per-game fantasy points under a specific league's scoring rules,
adjusts those projections for situation and injury, and writes a ranked Excel board you
can draft from.

**One ground rule shapes the whole project: no analyst opinions.** Every number on the
board is derived from historical NFL statistics. Average draft position (ADP) appears on
the board as a reference column so you can see where the market disagrees with the model
— it is never an input to the model.

The same model produces a different board for every league, because scarcity and injury
math are league-dependent. A four-game absence costs 4/12 of a 12-week season and 4/14 of
a 14-week one; the last starting QB is QB12 in a 12-team league and QB6 in a 6-team one.
Those differences move players by dozens of ranks, so each league gets its own config and
its own board.

## Output

| Board | League | Config |
|---|---|---|
| `2026_6Team_Board_v17.xlsx` | 6-team full PPR, redraft | `league_config_6team.json` |
| `2026_8Team_Board_v17.xlsx` | 8-team full PPR, redraft | `league_config_8team.json` |
| `2026_12Team_Board_v17.xlsx` | 12-team full PPR, keepers | `league_config_12team.json` |
| `2026_32Team_Board_v17.xlsx` | 32-team superflex PPR | `league_config_32team.json` |

Each board carries per-player projected points per game, value over replacement, a rank,
the drivers behind the adjustment, and an ADP comparison column.

## How the model works

1. **Baseline projection** — per-game production over 2023–2025, weighted 50/30/20 toward
   the most recent season, scored under the league's own scoring table.
2. **Shrinkage** — players with few games get pulled toward their position's mean, so a
   three-game sample doesn't outrank a full season.
3. **Quarterback mean reversion** — a QB's baseline gets 54% of its own weight and
   the rest goes to the position mean, because quarterbacks regress far harder
   than skill positions do. It applies only to quarterbacks with 16+ games of
   history; a backup's small sample is a precise estimate of a different thing,
   not a noisy estimate of a starter's, so he is left alone.
4. **Situational adjustment** — a fitted linear model over features the player doesn't
   control: position-group competition on his own team, the team's pass/run tendency, a
   playcaller change flag, and workload share.
5. **Rookies** — projected from a draft-slot cohort baseline (average PPG by position and
   draft round across the 2021–2025 rookie classes), then given the same situational
   treatment as veterans.
6. **Replacement level and VOR** — replacement is set at the last player likely to be
   *drafted* at each position, not the last starter, then value over replacement drives
   the final cross-position ranking.
7. **Injury and availability** — hand-maintained overrides zero out season-enders and
   discount PUP/NFI players by expected games missed, denominated on that league's
   regular season.

## Validation

Model changes have to survive three gates before a board will build at all.
`build_board.py` refuses to run if any is missing, failing, or older than the weights it
validates:

- **Holdout gate** — the situational and rookie weights are checked against held-out
  seasons (2023, 2024, 2025 folds), not the data they were fit on.
- **Playing-time gate** — the expected-games predictor is validated on 719 held-out
  players.
- **QB reversion gate** — the quarterback reversion weight is validated on held-out
  seasons, scored on actual PPG rather than delta so the `delta = actual − baseline`
  artifact cannot manufacture the result. It is the only one of the three that moves
  rank, so a board built on it should be diffed against the previous one.

Beyond the gates, `src/compare_boards.py` diffs two boards and can assert
`--expect rank-identical`, so a change that was supposed to leave ranking alone can be
proven to have done so. `src/verify_adjustments.py` and `src/sanity_top_n.py` check the
mechanical properties — duplicates, nulls, retirees, dead features cited as drivers.

Several ideas were tested and **rejected** rather than shipped, and the reasoning is kept
in `PHASE_8-14_PLAN.md`: combine metrics as rookie predictors, a QB draft-round signal
(n=3 per bucket — noise), and drafting off the stress test's worst-case rank column
(minimax minimizes regret in rank space, but the underlying losses are 8:1 asymmetric).

## Data sources

- **[nflverse](https://github.com/nflverse)** via `nflreadpy` — player stats, rosters,
  team assignments, and the player ID mapping.
- **[Fantasy Football Calculator](https://fantasyfootballcalculator.com)** — ADP, used as
  a reference column only.
- **Hand-maintained CSVs** — `playcaller_history.csv` (offensive playcaller by team and
  season), `injury_overrides.csv`, `position_overrides.csv`, `keeper_history.csv`.
- **Transcribed mock drafts** in `data/mock_boards/` — the 32-team superflex board reads
  ADP from these rather than from FFC, because no public feed covers that format.

## Quickstart

```bash
git clone https://github.com/dunlapjack/fantasy-football-rankings.git
cd fantasy-football-rankings

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Build the feature table (hits the network; a few minutes)
python -m src.pipeline

# Build a board
python -m src.build_board --config league_config_12team.json --version 17 --note "first build"
```

To use your own league, copy one of the `league_config_*.json` files and edit
`num_teams`, `roster_slots`, `scoring`, `regular_season_weeks`, and `playoff_weeks`.
`board_label` sets the output filename.

`BUILD_RUNBOOK.md` covers the full refresh procedure and what to check in the output.

## Repo layout

```
src/
  pipeline.py            orchestrates the feature build
  features.py            veteran baselines, shrinkage, team assignment
  rookies.py             draft-slot cohort baselines
  situational.py         competition, tendency, playcaller-change features
  ranking.py             applies fitted situational weights
  scoring.py             league scoring tables
  fit_weights.py         fits the situational model
  qb_reversion.py        QB mean reversion and its gate
  fit_rookie_weights.py  fits the rookie model
  holdout.py             out-of-sample gate
  playing_time.py        expected-games model and its gate
  backtest.py            historical replay
  adp.py                 Fantasy Football Calculator feed
  mock_adp.py            32-team ADP from transcribed mock drafts
  build_board.py         replacement level, VOR, Excel output
  compare_boards.py      board-to-board diff with assertions
  verify_adjustments.py  cross-league checks
  sanity_top_n.py        mechanical sanity checks
  draft_sim.py           draft-strategy simulator

data/                    feature tables, fitted weights, gate results, mock boards
charts/                  diagnostic plots
notebooks/               exploratory analysis
```

## Documentation

- `BUILD_RUNBOOK.md` — how to refresh and rebuild, and what to verify afterward.
- `FEATURE_SPEC.md` — the feature definitions.
- `PHASE_8-14_PLAN.md` — the full development log: every phase, every bug, and the
  ideas that were tested and cut.
- `PHASE_14_DRAFT_STRATEGY.md` — findings from the draft simulator.

## Known limitations

- **Quarterbacks have no situational features.** Phase 15 ran all 23 plausible
  candidates through the holdout at QB on the full 2017–2025 window and 21 were
  cut inside the training fold — age, experience, team tendency, playcaller
  change, O-line continuity, position competition and QB rushing volume all
  included. The model still has no development curve for quarterbacks. It does
  now pull their baselines toward the position mean (see step 3 above), which is
  the one thing that survived.
- **Rookies who share a position/round bucket are the same player to this model.** Two
  first-round running backs get the same cohort baseline before situational adjustment.
- **Kickers and defenses are scored, not modelled.** They get projections but none of the
  situational treatment.

## License

[MIT](LICENSE)
