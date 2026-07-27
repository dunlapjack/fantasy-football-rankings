# Feature Specification

## Season window
- Using 2023, 2024, 2025 for veteran stats
- Weighting: 2025 = 50%, 2024 = 30%, 2023 = 20%

## Per-position features (all players)
- QB: pass yards/game, pass TD/game, INT/game, rush yards/game, rush TD game, games played
- RB: rush yards/game, rush TD/game, receptions/game, rec yards/game, targets/game, games played
- WR/TE: targets/game, receptions/game, rec yards/game, rec TD/game, games played
- K: FG made/attempted by distance band, XP made
- DST: output of calculate_dst_points(), scoring handled separately

## Situational context features (all players, not just rookies)
- Incumbent position-group competition: points-per-game-while-active for other players at the position on that team, last 1-2 seasons (injury-adjusted via rate, not season totals)
- Team pass/run tendency: last season's team-level pass attempts/game and rush attempts/game
- Coaching/playcaller change: derived from playcaller_history.csv — flag if playcaller differs from prior season for that team
- Depth chart position: secondary tie-breaker only, not a primary weighted input

## Rookie handling
- Draft-slot cohort baseline: average points-per-game by position + round, from 2021-2025 rookie classes (offense only: QB, RB, WR, TE)
- Finding: RB/WR/TE show a clear declining trend as round increases — usable signal
- QB showed spikes in rounds 4 and 7, but n=3 players each — noise, not signal, disregarded
- Rookies combine: draft-slot cohort baseline + situational context features (same as veterans)
- This value is frozen pre-draft, not updated live during the season

## Data quality notes
- No duplicate gsis_id values found in players table
- 89 players had multi-team stats within 2025 (trades/waivers) — handle by using most recent team, not first-seen team, when assigning current roster
- Rows with null player_id exist in player_stats — investigate source before Phase 3, likely need filtering out of player-level joins