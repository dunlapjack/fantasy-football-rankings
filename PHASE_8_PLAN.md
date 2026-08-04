# Phase 8+ Plan — 2026 Drafts (board complete August 22)

Chapter 2 of the project. Phases 1–7 built a working statistics-only pipeline and
shipped `2026_Draft_Board_v7.xlsx`. This chapter is about **calibration**: making the
adjustments trustworthy rather than just present.

Ground rules carried over unchanged: statistics only, no analyst opinions, ADP is a
reference column and never a model input. League rules unchanged from Phase 7
(12-team full PPR, 6-pt pass TD, 16 rounds, keeper rules per `league_config.json`).

---

## Problems this chapter fixes

Three complaints came out of Phase 7. Two of them are the same bug.

**1. Every situational adjustment is negative.**
The Phase 6 weights in `ranking.py` were fit as `delta ~ features` *with a constant
term*, but `apply_situational_weights()` only sums `feature × weight` — the constant
was never transcribed into the code. Refitting `data/backtest_features.csv` reproduces
the exact coefficients now in the file and recovers the dropped intercepts:

| Position | Dropped intercept | Mean adjustment as shipped | Mean adjustment with intercept | Actual mean delta in data |
|---|---|---|---|---|
| RB | +3.592 | −3.43 | −0.06 | −0.06 |
| WR | +2.365 | −3.02 | −0.74 | −0.74 |
| TE | +1.576 | −1.40 | +0.01 | +0.01 |

`workload_share` and `experience` are always-positive features with negative slopes,
so dropping a positive intercept *guarantees* a negative adjustment for every player.
The model never claimed universal decline — the code did.

**2. Players took too heavy a hit (only 6 skill players over 15 PPG).**
Same bug. The raw baseline has 29 skill players over 15 PPG; the phantom penalty
erases 23 of them.

**3. Rookies valued too high.**
Partly the same bug — rookies are hardcoded to zero adjustment, so while veterans ate
a −3.4 phantom penalty, rookies floated up untouched. Partly its own problem: the
cohort baseline assigns one number per position/round bucket with no team context, so
Jeremiyah Love (ARI) and Jadarian Price (SEA) both project 15.12 PPG and tie at model
ranks 4–5. Phase 12 addresses the second half.

---

## Timeline

**Revised Aug 3.** Two changes landed the same day.

**A second draft was added.** The Dunlap Family Fantasy Football league drafts first,
so the finished board is needed by **August 22**, not August 29. That league is 6 teams,
no keepers, regular season weeks 1–12 with playoffs in weeks 13–14 — scoring and roster
slots otherwise identical to Lebron James. It gets its own `league_config_dunlap.json`,
and both boards ship from the same model.

**Phase 9 was cut.** Validation killed it before it shipped (see below). That returns
four days, which go to Phase 11 — which grew, because the 12-week season makes the
games-available problem sharper — and to Phase 13, restoring its full original window.

| Phase | Original | Revised |
|---|---|---|
| 8 — Intercept fix | Aug 2–4 | **complete** (Aug 2, `048b7cd`) |
| 9 — Playcaller position-PPG | Aug 5–9 | **cut** (Aug 3) |
| 10 — Usage trend and age curves | Aug 10–14 | **complete** (Aug 4) |
| 11 — Confidence, injuries, PUP | Aug 15–18 | Aug 8–12 |
| 12 — Rookie-specific model | Aug 19–23 | Aug 13–16 |
| 13 — Full refit and validation | Aug 24–27 | Aug 17–20 |
| 14 — Draft day prep (both boards) | Aug 28–29 | Aug 21–22 |

Phase 13 is back to its original 4 days and Phase 11 gained one. Phase 12 is the only
phase still short of its original window (4 days vs 5) — deliberate, since it's the
phase most likely to fail on its own terms (n = 5 rookie classes) and already has a
documented fallback. Phase 13 CP2 (holdout validation) does not get cut for time under
any circumstances.

### Phase 8 — Intercept fix and reproducible weights (Aug 2–4) — COMPLETE

The bug is small. The lesson is not: weights were hand-copied from a notebook that no
longer exists in the repo. Fixing the number without fixing the process invites the
same class of error again.

- **CP1** — `src/fit_weights.py`: fits per-position OLS with intercept, prints
  coefficients / std errors / p-values / R², writes `data/situational_weights.json`.
  Weights stop being hand-transcribed.
- **CP2** — `ranking.py` loads weights + intercept from that JSON.
- **CP3** — Verify: adjustments two-sided, >15 PPG count recovers, rookies fall
  relative to veterans. Spot-check Nacua / Gibbs / Love against v7.
- **Deliverable:** `v8` board. Everything after this builds on a correct baseline.

### Phase 9 — Playcaller position-PPG — CUT (Aug 3)

The premise was that `coach_changed` throws away *which* playcaller, and that some
coordinators reliably feed a position. CP1 built the table; CP2 tested whether the
signal was real before shrinking it. It isn't. Code retained in
`src/playcaller_ppg.py` and `src/playcaller_validate.py` as documented negative
evidence — the tests are worth keeping, and the question is worth being able to
re-ask cheaply in a future offseason.

**What CP1 found (and why it was misleading).** Playcaller position-PPG has an ICC of
0.28 QB / 0.31 RB / 0.40 WR / 0.15 TE, and split-half correlations of +0.18 to +0.47.
Encouraging on its face. (CP1's *printed* summary compared raw between-playcaller sd
against within-playcaller sd, which is not a valid comparison — a mean over n seasons
already has its noise divided by n. The corrected ANOVA figures above look better than
what printed, not worse. The error was in the diagnostic, not the data.)

**What killed it.** "Playcaller" and "team" are very nearly the same variable — a
coach who keeps his job is measured on one roster for five years. The movers test
isolates them: for playcallers who worked at more than one team, does the effect at
team A predict the effect at team B?

| | QB | RB | WR | TE |
|---|---|---|---|---|
| r across teams | −0.05 | −0.20 | −0.12 | −0.41 |

Zero or negative at all four positions. The persistence was the roster, not the coach:
Sean McVay's 42.8 WR PPG is Kupp and Nacua, Ben Johnson's 31.5 RB PPG is Gibbs and
Montgomery. n is thin (11 pairs per position), but four positions agreeing is not
nothing. The predictive test confirms it — playcaller history added **+0.005 to +0.026
R²** over the team's own prior season, with the two predictors correlated 0.61–0.75.

**The better design also failed.** Measuring the *player* against his own baseline
rather than the position room differences out roster talent by construction, and
matches the model's actual target (`delta = actual − baseline`). Split-half of mean
delta by playcaller, 2023–25: RB −0.01, WR −0.05, TE +0.32, ALL +0.32. The two
positions with the most data are flat zero; between-coach spread is 1.1–1.7 PPG before
noise correction. This is the more damaging null of the two, because this design has no
confound to blame.

**Caveat, recorded honestly.** The player-relative test only had 2023–25, since
`backtest_features.csv` doesn't extend further back. A 2021–25 version was costed at
roughly a day and declined on the compressed schedule. This is a decision made on
suggestive evidence, not conclusive evidence, and it is the first candidate to revisit
if a future offseason has time.

**Finding carried to Phase 13.** The `coach_changed` flag that already ships in
`continuity_score` moves mean delta by 0.1–0.4 PPG against standard deviations of
2.5–7.1. It may not earn its slot either. Retested as part of the Phase 13 refit.

### Phase 10 — Usage trend and age curves — COMPLETE (Aug 4)

Both features tested; both shipped, though not everywhere. Model v10.

**CP1 — Usage trend.** Three bases were tested rather than assuming one:
share-of-team-volume, raw per-game volume, and share-slope-over-mean-share. Share
won cleanly. (Note that "share of team volume" and "same basis as `workload_share`"
turned out to be the same thing — `compute_workload_share` already divides by team
pace — so the third variant was substituted to keep three genuinely distinct tests.)

| (on the original 3-season training set) | RB | WR | TE |
|---|---|---|---|
| `usage_trend_share` | **+6.98** (p=0.002) | +6.54 (p=0.23) | **+16.79** (p=0.008) |
| `usage_trend_volume` | +0.26 (p=0.003) | +0.23 (p=0.16) | +0.37 (p=0.061) |
| `usage_trend_relative` | +1.45 (p=0.024) | +0.55 (p=0.31) | +1.22 (p=0.060) |

**Revised twice as the training window widened (see below).** Final, on 2017–2025:
RB **+4.68** (p=0.0004), WR **+6.20** (p=0.075), TE **+14.16** (p=0.0001).

WR is the cautionary one. Its p-value went **0.23 → 0.034 → 0.075** across the three
training windows — cut on three seasons, reinstated on five, and back near the line on
nine. Non-monotonic, and second-weakest at the largest sample. Alpha was fixed at 0.10
beforehand and the test never changed, so nothing here is a search for a passing
p-value; but a feature that oscillates around the bar is not behaving like RB and TE,
which sit at p<0.001 and barely move. It ships because moving the bar after seeing a
result is worse than shipping a marginal feature. **It is the first coefficient that
should fall if Phase 13 CP2's holdout disagrees.**

Trend does not need the `team_changed` null-out that `workload_share` requires: each
season's share uses that season's own team, so a mid-window mover still gets an
apples-to-apples slope. No David Montgomery problem.

**The low-confidence assumption was backwards.** The plan assumed 2-season slopes
would need discounting. The signal is *carried* by them: RB trend is p=0.0013 with
2-season players included and p=0.259 on 3-season-only. That is a power artifact, not
a contradiction — the 3-season 95% CI is [−3.43, +12.86], which contains the 2-season
estimate of +7.82. Explicit discount interactions tested p=0.57 (RB) and p=0.76 (TE),
so no discount ships. Within the 2-season subgroup trend holds at p=0.0019 while age
goes flat (p=0.55), so it is not merely a young-player proxy. `trend_low_confidence`
is now a display flag only; the model term is `trend_missing`.

### QB's first weight, and its first suppressed intercept (Aug 4)

QB carried no weight from Phase 5 until now. On 2017–2025 (n=278 against 96 on three
seasons) age lands at **−0.196 PPG per year, p=0.0014**, stable in all nine folds. The
long-standing "nothing works for QB" result was partly a statement about sample size.

The fitted intercept, however, is a **selection artifact and does not ship**. Mean delta
by games threshold:

| min games | QB | RB | WR | TE |
|---|---|---|---|---|
| 0+ | **−0.93** | −0.69 | −0.47 | −0.19 |
| 8+ (`MIN_GAMES`) | **+0.70** | −0.08 | −0.07 | +0.12 |
| 12+ | **+1.16** | +0.23 | +0.18 | +0.44 |

QB swings from worst position to best as the filter tightens, roughly triple RB's
movement, because quarterback is one-per-team: a mediocre receiver still plays eight
games, a mediocre quarterback is benched. So +0.70 means "quarterbacks who keep their
job beat their baseline" — unknowable on draft day. Applied uniformly it lifted every QB
against every other position and put Josh Allen in the Lebron James top 10.

`fit_weights.SUPPRESS_LEVEL_SHIFT` removes the level and keeps the slope: a 24-year-old
QB still gets +1.04 and a 36-year-old −1.31, spread identical (sd 0.983 either way),
ordering within the position untouched. The fitted value is preserved as
`intercept_fitted` alongside `level_shift_removed`, and `verify_adjustments` checks the
adjusted identity — otherwise it would fail QB for obeying instructions, and the fix
would look like loosening the one tolerance that must stay exact.

**Revisit in Phase 11.** Modelling games-available directly is the real fix; if that
lands, this suppression must come back out or the two will double-count.

**CP2 — Age curves.** `age` (from `birth_date`, as of Sept 1) replaces `experience`
at all three positions, beating it on both adjusted R² and AIC everywhere. With both
terms in the model experience goes insignificant at every position (p = 0.38 to 0.85)
while age holds. Quadratic age is dead (p = 0.71 to 0.94). Binned age was competitive
only at TE and non-monotonic there — the middle bin came in *positive* — which is
noise, not a curve.

The plan predicted RB decline would be steeper than WR. In the controls-only
comparison it is (−0.535 vs −0.398), but once trend enters the RB slope falls to
−0.359. **Part of what the model has been calling RB age decline is declining usage.**

QB was retested with age — the one genuinely new input it had never seen — at p=0.68
linear and p=0.52 quadratic. QB remains unadjusted. That is now a null confirmed
across three phases.

**CP3 — Refit and rebuild.** RB and WR keep every term; TE drops `trend_missing`
(p=0.85). Both boards rebuilt at v10. Top-30 movement is modest and legible: largest
top-30 moves are Ashton Jeanty +18, Tucker Kraft +15, Bucky Irving +8 (trend +19.5 pp
per season). The largest moves anywhere are veterans whose usage collapsed — Nick
Chubb −222 (trend −26.3), Austin Ekeler −322 on the Dunlap board (trend −29.7).

**Three bugs found while verifying, not while building.** All three are the Phase 6
error in different costumes — a coefficient separated from the constants it was
fitted with. None would have been visible reading the spreadsheet.

1. **`drop_nulls` would have deleted a non-random 209 rows.** `usage_trend_share` is
   null for players under two usable seasons, and `fit_position` dropped those rows.
   Those rows have mean delta **+0.97 against −0.79** for the rest, so dropping them
   would have pushed every intercept down for a non-statistical reason. Now
   mean-imputed with a `trend_missing` indicator.
2. **Centering had to travel with the coefficient.** `age` enters centered at the
   position mean. Applying a centered coefficient to raw age is wrong by
   `coef × center` — **−9.48 PPG on every RB**, a uniform shift, precisely the Phase 6
   signature. Centers now ship inside the same JSON object as the weights.
3. **The intercept came from a different model than the slopes.** `fit_position` fit
   the full spec, kept the significant subset, and shipped the *full* model's
   intercept alongside them. It never bit because Phase 8 dropped nothing; TE's
   `trend_missing` is the first real drop. Survivors are now refit and that refit's
   intercept ships. Measured bias was small here (+0.019 PPG) only because the dropped
   coefficient was small — it scales with whatever gets cut next.

`src/verify_adjustments.py` now gates all three. It runs the real apply path over the
fit sample and asserts `mean(adjustment) == mean(delta)`, an OLS identity that holds
for any correct fit and breaks under all three variants. Phase 6's gap was −3.43 at
RB; current gaps are ~1e-16. It also checks two-sidedness, that every weight has both
a feature mean and a center, and that the weights file is not older than the CSV it
claims to describe.

**Cross-checked independently.** The whole bake-off was run twice — statsmodels, and
a separate numpy/hand-rolled-t-distribution implementation — agreeing to four decimal
places on every coefficient, including reproducing the shipped Phase 8 intercepts
(3.5925 / 2.3653 / 1.5764) exactly.

**The aging shape changed with every data expansion, and that pattern is the lesson.**
Three windows, three different answers:

| training window | what the age curve looked like |
|---|---|
| 3 seasons (947 rows) | steep cliff after 29 — RB −0.687 under 29 vs **−0.963** at 29+ |
| 5 seasons (1,575) | no cliff; one straight line fits, quadratic dead (p=0.79–0.92) |
| 9 seasons (2,775) | curved, in the **opposite** direction — RB −0.640 under 29, **+0.023** after |

The cliff was 32 RBs aged 29+ drawing a shape out of noise, and it was flagged at the
time as having wide, overlapping CIs. That caveat turned out to be the whole story. **A
wide confidence interval is not a weaker version of a finding; it is the finding saying
it might be nothing.**

At nine seasons RB age-squared is significant inside the shipped spec (p=0.004, AIC
−6.3): decline is front-loaded through the twenties and flat afterwards. It was **not
adopted** — see Phase 13 CP1, which is already scoped for exactly this. Two reasons for
holding. The flat tail rests on 57 backs over 30, and those are the ones who lasted,
which is the same survivorship that forced the QB level-shift suppression below. And
the steep young end likely overlaps `trend_missing`, which Phase 11 CP5 is re-testing
anyway. WR and TE remain linear on all three windows.

Every one of these answers was in-sample. Phase 13 CP2's holdout is the only test that
does not move when data is added to it.

### Training window widened to 2017–2025 (Aug 4)

Prompted by asking whether more history was available. Two things called "3 years"
were being conflated:

- **The baseline window** — each player's trailing 3 seasons, weighted 50/30/20. This
  projects a *player*, and the original instinct that recent football predicts best
  holds. Unchanged.
- **The training set** — which target seasons the regression learns from. This
  estimates what a coaching change or a year of age is *worth*, and those relationships
  do not go stale the way a player's form does. Small samples were the live constraint.

Done in two steps. 2021–2022 were already usable for free (`playcaller_history.csv`
started at 2021 and `compute_coach_continuity` reads the target season's own row),
taking 947 → 1,575 rows. Then the file was extended by hand back to 2016, unlocking
2017 onward: **2,775 rows**, a 2.9× increase over where the phase started.

`build_backtest_dataset` derives the earliest legal target season from the file itself
rather than a constant, and raises rather than silently returning nulls for
`coach_changed`.

**Re-testing every previously-cut feature became mandatory, not optional.** A cut means
"no evidence found," and evidence is what sample size buys — so cuts are what a wider
window threatens, while kept features merely get refit. The sweep reinstated
`position_competition_ppg` (cut in Phase 6 at p=0.77; now p=0.0023 RB / p=0.0009 TE,
stable in all nine folds) and gave QB its first weight in the project's history.
`returning_oline_starters` also cleared 0.10, at TE only, and was **not** adopted — no
reason it should help tight ends and not backs, which is what a false positive looks
like. Roughly thirty tests were run; at alpha=0.10 about three should pass on luck
alone, so both adoptions were required to have had a stated football rationale in
advance.

What moved across the whole expansion: WR usage trend passed then half-failed (above);
the age curve changed shape twice (above); `trend_missing` at RB decayed from +1.546
(p=0.014) to +0.651 (p=0.076) as more data arrived; `continuity_score` collapsed and
was replaced by `qb_changed` (below). R² fell at RB (0.236 → 0.185) — what in-sample R²
does when it stops having room to overfit. No coefficient flips sign in any of the nine
folds. The 2-season-slope discount was retested twice and is still unwarranted.

**A hand-maintained file's first season cannot be trusted.** Extending the file exposed
that `changed_from_prior_year` had been defaulted to `false` for 31 of 32 teams in
2021 — there was no 2020 row to derive it from. Harmless while training started in
2023, and instantly a fifth of the training set once 2021 became a target. The column
is now derived from the playcaller names rather than typed, which makes the error
structurally impossible. Deriving it immediately surfaced a second bug: the
year-over-year lookup has to follow the *franchise*, not the abbreviation, or the
Chargers (SD→LAC, 2017) and Raiders (OAK→LV, 2020) silently produce null flags. See
`team_codes.FRANCHISE_PREDECESSORS`.

Going earlier than 2016 requires more hand research. Worth costing before Phase 12,
whose rookie model still has the thinnest sample in the project — though it now has 9
draft classes rather than 5.

### Phase 9 re-tested on the wider window — CUT STANDS (Aug 4)

The Phase 9 cut was recorded as resting on "suggestive evidence, not conclusive," because
`backtest_features.csv` only reached back to 2023. Re-run with no code changes:

- **Movers test** (does the effect travel with a coach who changes teams?) — negative at
  all four positions again: QB −0.17, RB −0.12, WR −0.13, TE −0.41.
- **Player-relative split-half**, the better-specified design and the one the plan called
  the more damaging null — RB +0.016, WR −0.026, TE −0.026 on 174 pairs rather than a
  handful. Not a small effect; nothing.
- **Predictive test** — playcaller history adds 0.004–0.026 R² over the team's own prior
  season, with the two predictors correlated 0.61–0.75.

The decision rule (positive movers test AND real gain in test 3 or 4) fails on the first
clause. Worth stating why more data cannot rescue this, unlike WR usage trend: a coach
who keeps his job is measured on one roster, so "playcaller" and "team" are nearly the
same variable. That is a confound, not a power problem, and no sample size separates
them.

### Phase 11 — Baseline confidence, injuries, and PUP (Aug 8–12)

**Expanded, and the reason matters.** The Dunlap league's regular season is weeks 1–12
with playoffs in 13–14, so the fantasy-relevant season is 12 games, not 17. Every
games-available calculation in this phase is now league-dependent: a four-game PUP
absence costs 4/12 = 33% of that league's season against 4/14 = 29% of Lebron James's
(weeks 1–14, playoffs 15–17). Kittle and Charbonnet should be ranked lower on the
Dunlap board than on the other one — the same player is worth different amounts in the
two leagues, and the board has never had to express that before.

Every haircut below therefore reads `fantasy_season_length` from the league config
rather than assuming 17. Note the denominator is the REGULAR season, not the full
17-week NFL calendar: those are the games that decide whether you reach the playoffs
at all, and weeks 15–17 are worth nothing in a league whose final is week 14.

Three related problems, all versions of "how much should we trust this number."

**A. Injury-blended baselines.** The 3-year weighted baseline silently folds games
missed and diminished-role injury seasons into the projection, dragging down exactly
the high-ceiling players you most want ranked correctly.

- **CP1** — Quantify: how many top-100 players have a season in the window with
  materially reduced games played?
- **CP2** — Test candidate fixes — weight seasons by games played, or down-weight
  seasons below a games threshold — against actual next-season PPG.
- **CP3** — Adopt whichever backtests better; note this touches `features.py`, which
  everything downstream depends on. Re-verify the Phase 3 spot-checks after changing it.

**B. Small-sample veterans.** A baseline built on 8 games is treated with exactly the
same confidence as one built on 37. Cam Skattebo ranks 11th on 8 games, Omarion
Hampton 18th on 9, Phil Mafah projects 13.4 PPG on a single game.

`baseline_low_confidence` does NOT cover this and never will — it flags a rookie
*cohort bucket* with too few historical players (`rookies.py:99`), and `pipeline.py:19`
hardcodes it `False` for every veteran. It also isn't displayed on the board or read by
anything downstream. This needs a separate signal.

- **CP4** — Add a games-played confidence measure for veterans; surface it on the board.
- **CP5** — Test shrinking low-sample baselines toward the position mean, with the
  shrinkage strength backtested rather than picked by feel.

  **MANDATORY: re-test `trend_missing` jointly with shrinkage. Do not fit one without
  the other.** Phase 10 shipped `trend_missing` at RB (+1.546, p=0.014) knowing it
  overlaps this checkpoint, and the overlap is not hypothetical — it lands on exactly
  the players this phase was written about:

  | Player | Baseline PPG | `trend_missing` bonus | Already flagged in CP4/B as |
  |---|---|---|---|
  | Phil Mafah | 10.90 | +1.55 | "projects 13.4 PPG on a single game" |
  | Omarion Hampton | 15.08 | +1.55 | "ranks 18th on 9 games" |
  | Cam Skattebo | 15.96 | +1.55 | "ranks 11th on 8 games" |

  A player with too little history to fit a usage slope is usually a player with too
  little history to trust the baseline of. Phase 10 pays him for the first; Phase 11
  proposes to charge him for the second. Fit them together or the board double-counts.

  **Second, narrower problem: the coefficient is being applied off its support.**
  `trend_missing` was estimated on RBs averaging 24.5 years old, only 2% of them 29+.
  The live pool it is applied to averages 27.2 with **24% aged 29+** — because a
  washed-up 34-year-old also fails to clear `MIN_TREND_GAMES` in any recent season, and
  collects the same +1.55 meant for ascending youngsters. Currently harmless: of the 12
  `trend_missing` RBs with a baseline at or above 8 PPG, exactly one (Darrell Henderson,
  29.0) is 29+, so the mismatch sits in the undraftable tail. It stops being harmless the
  moment shrinkage moves those players. Either interact `trend_missing` with age or
  restrict it by age at fit time.

**C. Replacement level in shallow leagues.** Found while verifying the first v9 boards.
`compute_replacement_ranks()` sets replacement at the last *starter* — QB12 in a
12-team league, so QB6 in a 6-team one. That's wrong the shallower the league gets. In
the Dunlap league QB7 through QB32 are all sitting on waivers, so the real fallback is
a perfectly good starting quarterback, not the worst rostered one.

The visible symptom: Josh Allen ranks 10th on the Lebron James board and **5th** on
Dunlap; Brock Bowers 16th and **10th**; Trevor Lawrence 97th and **71st**. The math is
internally consistent — RB/WR replacement climbs from rank 29 to 14, a 15-spot jump up
a steep part of the curve, while QB moves only 12 → 6, so skill positions shed more VOR
and QBs float up by comparison. But the conclusion is backwards from how a 6-team
league actually drafts, and the board is currently telling you to spend the 5th pick on
a quarterback you could stream.

- **CP6** — Derive replacement level from expected players drafted per position
  (`total_rounds × num_teams` allocated across positions by observed draft behavior)
  rather than from starter slots. Sanity condition: the 6-team board should push QB and
  TE *down* relative to the 12-team board, not up.
- **CP7** — Re-read the top 30 of both boards side by side afterward. This bug was
  invisible on the 12-team board, where starter count and waiver depth roughly agree;
  it only surfaced because a second league forced the comparison. Any replacement-level
  change needs checking at both league sizes for the same reason.

**D. PUP / NFI treatment.** `injury_overrides.csv` currently has one binary lever:
`OUT_SEASON` removes a player, everything else is a note. But PUP means missing *at
least* the first four games — not a maybe. George Kittle (torn Achilles, 13.09 adj PPG,
ADP 108) and Zach Charbonnet (torn ACL, 10.44, ADP 143) both show at full value today.

- **CP8** — Add a partial games-available haircut for `PUP` and `NFI` rather than the
  current all-or-nothing treatment. Scale the projection by expected share of the
  season available, using each league's `fantasy_season_length` — so a 4-game absence
  costs roughly 4/14 in Lebron James and 4/12 in Dunlap Family.
- **Open question:** whether the haircut should hit `adjusted_fantasy_points_per_game`
  (changes VOR and rank) or only a separate "expected total points" column (leaves PPG
  honest). PPG and season-long value diverge here for the first time in the project.

### Phase 12 — Rookie-specific model (Aug 13–16)

The biggest remaining structural gap. Rookies get a flat cohort number and no
situational adjustment at all.

- **CP1** — Build a rookie backtest set from the 2021–25 classes: cohort baseline as
  the starting point, actual rookie-season PPG as the target.
- **CP2** — Fit rookie-specific weights on situational features that are legitimately
  knowable pre-draft: team pass/rush tendency, position competition, O-line
  continuity, depth chart position. (Playcaller position-PPG was on this list until
  Phase 9 cut it.) Deliberately excludes
  `team_changed` (meaningless for rookies — Phase 6 documented why) and `experience`
  (always 0).
- **CP3** — Verify two same-round rookies on different teams now separate. Confirm
  rookie ranks are defensible against the veteran pool.
- **Risk:** n is small (5 classes). If coefficients are unstable, fall back to the
  shrinkage haircut and say so explicitly rather than shipping a fragile model.

### Phase 13 — Full refit, validation, final boards (Aug 17–20)

- **CP1** — Refit all positions with the surviving feature set. Check for
  multicollinearity — usage trend, workload share, and age will correlate.
  **Already measured in Phase 10 and it is a non-issue:** corr(workload_share, trend)
  is +0.109 / +0.064 / +0.111 at RB / WR / TE, and corr(workload_share, age) tops out
  at +0.255. Leave-one-season-out flips no shipped coefficient's sign across nine folds.

  **Carried here deliberately: RB age-squared.** Significant on 2017–2025 inside the
  shipped spec (p=0.004, AIC −6.3), describing front-loaded decline that flattens after
  29. Not adopted in Phase 10 because its flat tail rests on 57 backs over 30 — the same
  survivorship that forced the QB level-shift suppression — and its steep young end
  likely overlaps `trend_missing`. Test it here *after* Phase 11 has settled
  availability and shrinkage, and let CP2's holdout arbitrate. Note the aging shape has
  now changed with every data expansion (cliff → straight → front-loaded), so treat any
  in-sample verdict on it with suspicion.
- **CP2** — Holdout validation: fit on 2023–24, test on 2025. Does the model beat the
  raw baseline out of sample? If not, the added features are overfitting and should be
  cut. This is the honest test the project hasn't run yet. **Not cut for time.**
- **CP3** — **ANSWERED EARLY (Aug 4). `coach_changed` did not earn its slot.** Once the
  window reached nine seasons and 2021's flags were derived rather than defaulted,
  `continuity_score` fell to p=0.059, and splitting it showed why: `qb_changed` carries
  it (−0.498, p=0.080 at RB) while `coach_changed` contributes p=0.36 noise at RB,
  nothing at TE, and at WR comes out *positive* at p=0.082 — a coaching change helping
  receivers, which is not a finding. RB now uses `qb_changed` alone, renamed rather than
  silently redefined, per this checkpoint's own instruction. Nothing left to do here
  beyond confirming it survives the CP2 holdout.
- **CP4** — Build both boards from one model run. Naming convention changes here:
  `2026_DunlapFamily_Board_v9.xlsx` and `2026_LebronJames_Board_v9.xlsx`. Version
  tracks the *model*, bumping when weights or features change — so a pure data refresh
  stays v9. Build date and git short hash go in a metadata sheet, which is what
  actually distinguishes two rebuilds of the same version. (v8 staying static across
  three rebuilds is the problem this fixes.)
- **CP5** — Sanity-read the top 60 of each board by hand. Read them side by side: the
  6-team/12-week board should visibly diverge from the 12-team/17-week one, and if it
  doesn't, the league-awareness work in Phase 11 didn't land.

### Phase 14 — Draft day prep (Aug 21–22)

- Refresh ADP (it moves through August; the current pull is from late July).
- Re-run the fast keeper filter as opposing keepers are announced — Lebron James only.
  Dunlap Family is a straight redraft, so keeper columns are suppressed on that board.
- **ADP caveat for the 6-team league.** The FFC pull is `teams=12`. Six-team ADP is a
  different draft entirely — 96 players go instead of 192, and positional runs behave
  nothing alike. Either pull `teams=8` as the closest available proxy and label the
  column honestly, or drop the ADP column from the Dunlap board rather than show a
  reference number that quietly means something else. Decide before build, not during.
- Final boards printed and ready before noon on the 22nd.

**Phase 14b — second draft refresh (Aug 28).** The model is frozen after Phase 13; this
is a data refresh only, not a modeling pass. Re-pull ADP, re-check injury overrides
against the preseason, rebuild the board. No weights get refit — if the Aug 22 board
was wrong, that is a post-season finding, not a five-day fix.

**Slack:** Aug 23–27, plus the 20% shaved off each phase. That slack is the compression
buffer now. Previously the two days before the draft were held open deliberately —
nflverse updates, ADP shifts, and preseason injuries land in that window — and the
Aug 21–22 Phase 14 window preserves that for the first draft.

**Second league — resolved Aug 3.** `league_config_dunlap.json` created: 6 teams, no
keepers, weeks 1–12 regular season, playoffs 13–14, scoring and roster slots identical
to Lebron James. `build_board.py` already derives replacement level from
`num_teams`, so VOR recomputes correctly once it can take a `--config` flag (currently
`CONFIG_PATH` is a module constant). Two consequences beyond the config file:

- **Replacement level moves a long way.** 6 teams × 1 QB means the 7th-best QB is
  freely available; positional scarcity nearly vanishes and best-player-available
  logic dominates. The tier breaks and `draft_target` cushions were tuned on 12 teams
  and should be re-read, not assumed to transfer.
- **12-week season changes injury math.** See Phase 11.

**Resolved.** Both configs now carry `regular_season_weeks`, `playoff_weeks`, and
`fantasy_season_length`: Lebron James 1–14 with playoffs 15–17 (length 14), Dunlap
Family 1–12 with playoffs 13–14 (length 12).

**Versioning honesty note.** `MODEL_VERSION` went 8 → 9 for a change that touched the
filename convention and the ADP columns, not the model. The Lebron James v9 board is
rank-for-rank and VOR-for-VOR identical to v8 across all 1088 players. By the rule
stated below — bump only when the model changes — it should have stayed v8. Kept at 9
because two differently-named files both calling themselves v8 would be worse, but the
rule is only worth having if exceptions are visible, so this one is written down.

**Board versioning — settled Aug 3.** One file per league, overwritten in place:
`2026_LebronJames_Board_v9.xlsx` and `2026_DunlapFamily_Board_v9.xlsx`. The version
number tracks the MODEL, not the build — it bumps when weights or features change and
stays put for a data refresh. Every rebuild appends a row to a "Build History" sheet
inside the workbook recording build number, timestamp, model version, git hash, and a
`--note`. That answers "how many times has this been altered" without accumulating a
folder of near-identical spreadsheets, which was the actual complaint: v8 sat on three
different rebuilds with no way to tell them apart.

---

## Standing decisions

- **Kill features that don't earn their place.** `position_competition_ppg` and
  `contract_year` were tested and dropped in Phase 6. That was correct. Same bar here:
  a feature ships only if it's significant *and* improves holdout performance. Phase 9
  was cut in full under this rule — an entire phase, after the work was built.
- **A feature that persists is not a feature that predicts.** Phase 9's lesson.
  Anything measured at team level will inherit roster quality; before trusting a
  team-level signal, find the case where the two come apart (a coach who moved teams,
  a player who changed schemes) and test there.
- **Shipped features are not exempt from the bar.** `coach_changed` has been in the
  model since Phase 3 without a significance test. Phase 13 CP3 fixes that. Age of a
  feature is not evidence for it.
- **Intercepts always ship with coefficients.** Never transcribe one without the other.
- **Generalized (Phase 10): every constant a coefficient was fitted with travels with
  it.** The intercept was the first instance, not the only one. Centering constants are
  the second — a centered age coefficient applied to raw age is off by 9.5 PPG. And a
  coefficient must ship with the intercept *from its own model*: filtering a spec by
  p-value and keeping the unfiltered intercept silently mixes two fits. Anything the fit
  computed and the application needs belongs in the same serialized object, never in a
  constant in the apply-side file.
- **Verify by identity, not by eyeball.** Phase 6's bug survived because its symptoms
  were impressions ("everything looks negative"). `mean(adjustment) == mean(delta)` is
  an algebraic identity under OLS-with-intercept, so it either holds to floating point
  or the applied numbers didn't come from the fitted model. Prefer checks that can only
  pass for one reason.
- **A coefficient is only valid over the population it was fitted on.** Before shipping,
  compare the fit sample's feature distribution against the live pool's. `trend_missing`
  was fitted on RBs averaging 24.5 years old and is applied to a pool averaging 27.2.
- **`MIN_GAMES` is a selection rule, and selection rules bite unevenly by position.**
  Check any intercept by re-computing mean delta at several game thresholds. If it moves
  a lot more for one position than the others, that intercept is describing who survived
  the filter, not the position. QB moved +1.63 across the thresholds against RB's +0.60.
- **When a cut feature is re-tested on more data, the whole set of cuts must be
  re-tested.** Cheap to do, and re-testing only the one you happen to remember is
  selection by a different name. Re-testing everything then requires a multiple-comparison
  discount: prefer reinstatements that had a stated rationale before the sweep ran.
- **A hand-maintained file's earliest season is unverifiable by construction.** Anything
  derived from a prior year cannot exist for the first row, and defaulting it rather than
  nulling it hides that. Prefer deriving such fields; where that's impossible, make the
  earliest season raise rather than quietly pass.
- **Dead things get deleted at the end of each phase** (revised Aug 4, replacing
  "nothing gets deleted without approval"). The original rule existed to guard against
  losing something irrecoverably. Now that the repo is pushed, that risk is mostly
  gone — but *mostly* is doing real work in that sentence, so the rule splits:

  - **Tracked by git → delete freely.** `git log --diff-filter=D --name-only` finds it,
    `git checkout <commit>^ -- <path>` brings it back. A deletion is a reversible edit.
  - **Untracked or gitignored → confirm first.** Everything under `data/`, plus any
    board not yet committed. Git has never seen these, so deletion is permanent. Some
    are regenerable by re-running a module; some are not, and the difference is worth
    checking rather than assuming.
  - **Delete from a clean tree.** Cleanup happens after the phase's commit, never
    alongside uncommitted work — otherwise the recovery command has nothing to recover
    from.
  - **Verify redundancy, don't assume it.** Before deleting a superseded artifact,
    show it's superseded. `2026_Draft_Board_v8.xlsx` was removed only after confirming
    it matched the v9 Lebron James board on all 1088 players, rank and VOR.
- **Every phase ends with a Git commit** at the checkpoint boundary.
- **Exploratory "does this pattern hold" analysis is Jack's hands-on task** — Claude
  builds the scaffolding, Jack runs the exploration and brings back findings.

## Open items carried forward

- Null `player_id` rows in `player_stats` — flagged in Phase 2, still uninvestigated.
- **Retired players carry a live `latest_team`.** Phase 10's age column made this
  visible: the pool contains a 41-year-old WR and a 42-year-old TE, 4–6 players per
  position past the fit sample's age range. They rank nowhere near draftable so nothing
  is at stake today, but they inflate every pool-level mean the model is checked
  against, and they are the reason the live age distribution sits above the fitted one.
- **`experience` is now dead as a model input** but still computed in `situational.py`,
  shipped in `player_features.csv`, and referenced by `ranking.py`'s
  `LEGACY_SITUATIONAL_WEIGHTS` fallback. Under the delete-dead-things rule it is a
  removal candidate; it survives Phase 10 only because the legacy fallback would break.
  Decide during Phase 13, when the fallback itself is worth reconsidering.
- K and DST are not modeled at all. Acceptable (draft them last), but it should be a
  decision rather than an omission.
- No holdout validation has ever been run. Phase 13 CP2 is the first. Every R² quoted
  so far is in-sample.
