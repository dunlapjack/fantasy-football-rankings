# Phase 8+ Plan — 2026 Drafts (board complete August 22)

Chapter 2 of the project. Phases 1–7 built a working statistics-only pipeline and
shipped `2026_Draft_Board_v7.xlsx`. This chapter is about **calibration**: making the
adjustments trustworthy rather than just present.

Ground rules carried over unchanged: statistics only, no analyst opinions, ADP is a
reference column and never a model input. League rules unchanged from Phase 7
(12-team full PPR, 6-pt pass TD, 16 rounds, keeper rules per `league_config_lebronjames.json`).

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

### Phase 11 A — CP1–CP3 closed, `discount_thin` ADOPTED (Aug 4)

**This section was first written as "NOTHING ADOPTED" and is now the
opposite.** The first CP2 run scored the silently-corrupted five-season
training set described in the incident above. Re-run on nine seasons, the
verdict flips. Both versions are left visible in git history rather than
quietly overwritten, because the interesting thing here is not which answer
won — it is that a data bug and a modelling result are indistinguishable from
the output of a decision rule.

**CP2 — `discount_thin` clears both clauses.** Paired ΔMAE on the AFFECTED
subgroup, nine seasons (n=1617):

| scheme | paired ΔMAE | SE | ratio | all positions positive? |
|---|---|---|---|---|
| **`discount_thin`** | **+0.0619** | **0.0289** | **2.14** | **yes** |
| `recency_x_games` | +0.0329 | 0.0324 | 1.02 | no — RB −0.025 |
| `games` | −0.0577 | 0.0328 | — | no |
| `drop_thin` | −0.1635 | 0.0729 | — | no |

Per position, `discount_thin`: QB **+0.4260 ± 0.1902**, WR +0.0420 ± 0.0396,
TE +0.0275 ± 0.0332, RB **+0.0005 ± 0.0549**.

Read honestly: the effect is real but small and **carried almost entirely by
quarterbacks**. RB is +0.0005, which is zero. That is consistent with CP1's
finding that the thin-season population is disproportionately backup QBs, and
with the earlier observation that affected-QB MAE (4.55) is far worse than any
other position's.

**Why believe this and not the earlier null.** The point estimate barely moved
between the two runs — 0.0702 to 0.0619 — while the standard error fell from
0.0378 to 0.0289. More data shrank the error bar around a stable estimate,
which is the signature of a real effect being measured better, not of a
different effect appearing. The five-season run also had RB at −0.013; nine
seasons puts it at +0.0005. Both are zero; the narrow run just had a noisier
zero that happened to land negative and trip clause (b).

Still +0.06 PPG against situational adjustments that run ±3, and still
in-sample. Phase 13 CP2's holdout remains the arbiter.

**CP1 is the finding, and it redirects the phase.** Exposure in the pool that
matters is small: **8%** of the market top 100 (by ADP) has a season under 8
games in the window. The plan's premise — that injury-blended baselines drag
down "exactly the high-ceiling players you most want ranked correctly" — is
mostly not happening to the players actually being drafted.

The model top 100 is 19% exposed, and *who* is on that list is the real result:
Carson Wentz, Jeff Driskel, Marcus Mariota, Jacoby Brissett, Jameis Winston,
Easton Stick, Joe Flacco, Russell Wilson. **Backup quarterbacks, nearly all
without an ADP.** Jeff Driskel sits inside the model's top 100 at 17.15 adj PPG
on a *one-game* sample; Phil Mafah at 13.51 on one game.

Their rates are not depressed by injury — they are **inflated by spot duty**,
and the model cannot tell the sample is worthless. That is Section B's problem,
not A's, and CP1 shows B's population is both larger and more distorted than
A's. The two halves of this phase were sized backwards when it was written.

The QB numbers agree from the other direction: affected-QB MAE is 4.15 against
~2.7 at skill positions, and `discount_thin` at QB is +0.6252 ± 0.2531 — the
only result past 2 SE anywhere in the run. **Not adopted.** It rests on 25
moved rows and was found by subgroup search after the pooled test failed, which
is the same shape of evidence as the aging curve that changed three times.
Carried to Phase 13 CP2's holdout to be tested rather than believed.

`features.DEFAULT_SCHEME` is now `discount_thin`. This changes
`fantasy_points_per_game` everywhere, so both the training set and the live
board move: rerun order is `backtest → shrinkage sweep → fit_weights →
pipeline → build_board`.

**CP5 must be re-measured on top of it.** The shrinkage sweep below was fitted
against `recency` baselines. Both changes target thin-history players, so
stacking two independently-measured effects is the same double-count the plan
forbade for `trend_missing`. The sweep re-runs after the baseline changes and
before any K is adopted.

> **CP4 display decision (Aug 4):** the games-played confidence signal gets
> **colour-coded onto the existing `GP (sample)` column** — red/amber shading by
> how thin the baseline is, with the reason appended to the Why column — rather
> than a new Confidence column. The board was just reordered for density and a
> 24th column works against that.

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

### Phase 11 C and D shipped — CP6, CP7, CP8 closed (Aug 4)

C and D were taken first because neither needs a refit, and both fix things
visible on the v10 boards today. A and B (injury-blended baselines,
small-sample shrinkage fit jointly with `trend_missing`) are untouched and
still carry every note written above them.

**CP6 — replacement level now counts picks, not starters.** `compute_replacement_ranks()`
takes `total_rounds × num_teams`, subtracts `UNMODELED_SLOTS_PER_TEAM × num_teams`
for the kickers and defenses that consume real picks but are not modeled, and
splits the remainder by the position mix of that many players in ADP order,
out-for-season players removed. The starter-slot rule survives as
`compute_starter_ranks()` and now feeds only the notes block, where the old and
new levels print side by side.

| League | Old (starters) | New (drafted) |
|---|---|---|
| Lebron James (12) | QB12 / RB29 / WR29 / TE14 | QB22 / RB51 / WR74 / TE21 |
| Dunlap Family (6) | QB6 / RB14 / WR14 / TE7 | QB8 / RB32 / WR38 / TE6 |

**CP7 — the sanity condition holds, and it is now a test rather than a
reading.** Quarterbacks inside the top 30 go 1 → 4 on the 12-team board and
4 → 1 on the 6-team one; Josh Allen goes 11th → 7th on Lebron James and 7th →
15th on Dunlap. Trey McBride is the only tight end left in either top 30.
`verify_adjustments.py` gained `check_replacement_levels()`, which rebuilds
both boards and hard-fails if the best QB or TE ever ranks *higher* in the
shallow league than the deep one. That assertion is the actual deliverable —
the bug hid for three phases because nobody compared the two boards, and now
nothing can be shipped without comparing them.

**Known limitation, deliberately accepted.** The position split comes from
FFC's 12-team mocks, so the first 84 picks are a 12-team drafter's mix, not a
6-team drafter's. A 6-team room facing no scarcity would take fewer than 8
quarterbacks. The error runs conservative — it understates how far QB should
fall — and a config can override the split outright with an `expected_drafted`
block once there are real draft results to fit.

**CP8 — PUP/NFI get a partial haircut, and it lands off the ranking.** The
open question is answered: the haircut does **not** touch
`adjusted_fantasy_points_per_game`. PPG is a rate, and a torn Achilles does not
make Kittle worse in the games he plays — it makes him play fewer. Two new
columns carry it instead: `Exp Gm` (league regular season minus known absence)
and `Exp Pts` (Adj PPG × Exp Gm). Rank and VOR are unmoved, so nothing
double-counts if a later phase models availability directly.

`PARTIAL_STATUSES = {PUP, NFI}` defaults to 4 games missed — the NFL rule, not
an estimate — overridable per player by the new optional `games_missed` column
in `injury_overrides.csv`. **Kittle is the row to revisit:** four games is
almost certainly generous for a torn Achilles, and the column is empty for him.

The league-dependence the phase intro demanded now exists: the same four-game
absence costs 4/12 in Dunlap and 4/14 in Lebron James, read from
`fantasy_season_length`, denominated on the REGULAR season.

**Also shipped: the "Why (value drivers)" column.** Each player carries a
signed decomposition of his own situational adjustment — `+0.9 role trend
+3.1pp/yr · −1.4 62% team share · −0.5 age 30` — computed as
`(value − position_mean) × weight` from the same `situational_weights.json`
that produces the number beside it, so the two cannot disagree.
`check_value_drivers()` asserts the full decomposition reconciles to
`situational_adjustment` to 1e-6. Terms below ±0.15 PPG are hidden and at most
four print, so the visible string is a summary, not the sum — the test checks
the identity underneath, which is the thing that could actually rot.

One side effect worth noting: `write_workbook()` no longer addresses columns by
hardcoded index. Phase 10 left a comment warning that inserting a column would
shift them all silently; Phase 11 inserts three, so the indices became a
`COLUMN_INDEX` name lookup instead of obeying the warning.

**MODEL_VERSION 10 → 11 with no refit.** Weights are byte-identical. The bump
is for the replacement-level change, which reorders the board on ranking logic
alone.

**Board columns reordered by draft-day importance** (Draft Target / VOR / Adj
PPG / Value Δ / ADP now sit immediately right of the frozen name block; model
internals moved to the far right). The row writer is keyed by column label
instead of list position, so future reordering is a one-line edit and a missing
value raises instead of silently shifting every cell right.

### INCIDENT — stale `DEFAULT_TARGET_SEASONS` silently narrowed the training set (Aug 4)

Phase 10 widened the training window to 2017–2025 by passing `--seasons` on the
command line, and `playcaller_history.csv` was extended back to 2016 to support
it. **The result was never written back to `DEFAULT_TARGET_SEASONS`**, which
still read `[2021..2025]`.

A later bare `python -m src.backtest` therefore rebuilt the training set at five
seasons and overwrote the nine-season file. Nothing failed:

| | rows after MIN_GAMES | RB | WR | TE | QB |
|---|---|---|---|---|---|
| shipped weights (9 seasons) | 2,750 | 711 | 1145 | 616 | 278 |
| silently rebuilt (5 seasons) | 1,575 | 402 | 662 | 354 | 157 |

43% of the training data gone, every coefficient moved, no error anywhere. The
only visible symptom was the Phase 11 B sweep reporting RB `trend_missing` at
+1.051 against the +0.6506 in the shipped JSON — a disagreement that reads like
a modelling question and was actually a data question.

**Nothing shipped was affected.** `situational_weights.json` and
`player_features.csv` both predate the bad regeneration and were never refit
from it, so the live boards are correct. The damage was confined to
`data/backtest_features.csv`, which is gitignored and regenerates.

Fixed three ways:

- `DEFAULT_TARGET_SEASONS` now holds 2017–2025, so the default matches what the
  model was actually fitted on.
- The comment block above it, which explained why 2021 was the earliest
  possible season, was false after `playcaller_history.csv` was extended and has
  been rewritten.
- **`warn_if_narrower_than_available()`** closes the direction the existing
  guard never covered. `build_backtest_dataset()` already raised when asked for
  seasons *earlier* than the playcaller file supports — asking for too much
  fails loudly. Asking for too *little* succeeded silently. It now prints a
  warning naming the unused seasons before it overwrites anything.

The general lesson worth keeping: a one-sided guard is a guard against the
direction you already thought of. This one protected against the failure that
announces itself and not the one that doesn't.

### Phase 11 B — CP4/CP5 shipped, and verification caught two things (Aug 4)

**Adopted:** shrinkage at K=2 toward each position's 30th percentile,
`james_stein` form. All three pre-committed clauses pass on `discount_thin`
baselines: +0.0919 ± 0.0334 (2.75 SE) on the low-confidence subgroup, full-pool
MAE improves at every K, and K=2 is the interior argmax of a 0–8 sweep. The
effect barely moved when the baseline changed underneath it (+0.0950 →
+0.0919), which is the evidence that CP3 and CP5 capture different things
rather than double-counting.

**`trend_missing` resolved, as the plan required.** RB's shipped +0.651 exists
only at K=0 — it fails alpha at every K from 1 to 8. Independently, the
`trend_missing × age` interaction is **−0.3844 (p=0.029)** on `discount_thin`
baselines with the main effect collapsing to +0.0458. It is now dropped from
the RB spec by the fit, and **WR picks up a negative one (−0.4474)** that the
unshrunk baseline had been masking. Both open questions the plan carried into
this phase are closed by the same joint fit.

**CP4 shipped as shading, not a column** — `GP (sample)` amber under 17 games,
orange under 8, with a `| −1.8 thin sample (8 gm)` suffix on the Why column.

#### Verification failure 1 — QB shrinkage inflated backups. FIXED.

`check_shrunk_baseline()` warned that 437 of 856 players moved UP. The cause is
positional: the anchor is the 30th percentile of players with 16+ games, which
lands at 3.44 / 3.86 / 2.88 at RB/WR/TE and at **12.52 at QB**, because
quarterback is one-per-team and "played 16 games" is "was the starter."

Nathan Peterman went −0.40 → **8.21**; 59% of quarterbacks moved up.

This is the same survivorship `fit_weights.SUPPRESS_LEVEL_SHIFT` already
documents in its own comment. The root cause is a population mismatch: the
shrinkage sweep scored only players who went on to play 8+ games in the target
season, so Peterman was never in the population the anchor was fitted on, but
he is in the population it gets applied to. Shrinkage assumes a small sample is
a noisy estimate of the same quantity; for a backup QB it is a precise estimate
of a different one.

`SHRINKAGE_EXCLUDED_POSITIONS = {"QB"}`. 51 of 111 quarterbacks were being
inflated by 0.5+ PPG and no longer are. `discount_thin` still applies at QB and
its QB-driven benefit stands — re-weighting a player's own seasons cannot
inflate him toward anyone else's number.

#### Verification failure 2 — the TE sanity check. TEST CHANGED, with reasons.

CP7's check asserted the BEST player at a position ranks lower in the shallow
league. It failed on TE: 21 deep, 19 shallow.

Investigated before concluding. The starter floor was the obvious suspect —
the one piece of starter-based logic left inside the pick-based calculation —
and removing it moved TE only 19 → 20, so that was not the cause. The actual
reason is that **the TE production curve is flat**: replacement moves TE21 →
TE7 between the leagues, which sounds enormous, but costs about as much as
RB52 → RB32 costs running backs. The two roughly cancel. TE is genuinely
neutral across the two leagues and that is a legitimate model output.

The test was measuring a claim about positional value with a statistic that
turns on one player shuffling past a neighbour. It now counts how many of a
position appear in the top 30 — QB 2 → 1, TE 1 → 1, both passing — and the old
rank comparison is retained as a SOFT check, since it is noisier but strictly
more sensitive.

Recorded at length because changing a test that failed is the move that
deserves the most scrutiny, and the reasoning should be auditable later rather
than taken on trust.

#### Verification failure 3 — the check that cried wolf

`check_shrunk_baseline()` warned "shrinkage mostly lowers projections — 437 of
856 moved up" on every run. The premise was arithmetically false: shrinking
toward the **30th percentile** raises everyone below the 30th percentile. That
is what the anchor is. Roughly half the pool moving up is the mechanism
working.

Worse, the noise hid the signal. The QB inflation above was a genuine defect,
and this check's only comment on it was a warning that was already firing for a
harmless reason. A check that always warns is a check nobody reads.

Rewritten to assert the thing that actually matters — **shrinkage must never
inflate a player into draftability** — scoped to ADP-bearing players and
promoted from soft to hard. Of 156 draftable non-QB veterans exactly one moves
up: Jonathon Brooks, +0.38 on a 3-game sample. The 371 fringe players who move
up have a median raw projection of 1.40 PPG and are now printed as context
rather than asserted against.

#### Sanity read on the shipped v12 board

Josh Allen ranks 9th on the 12-team board and 18th on the 6-team one. Checked
rather than assumed, since QB is now the only unshrunk position: replacement is
**QB22 = Tua Tagovailoa, 17.19 PPG on 42 games** — a real veteran starter, so
Allen's VOR of 9.03 rests on a defensible bar.

**Latent fragility worth recording.** Jeff Driskel (17.10 PPG on a ONE-game
sample) is QB23, one spot below the line. `compute_vor()` draws replacement
from the model's top-N by projection, which can include players with no ADP and
no real sample. Had Driskel landed a tenth of a point higher he would have set
the quarterback replacement level for the entire board. Excluding QB from
shrinkage removed the correction that was holding him down, so this is more
exposed than it was. Candidate fix for Phase 13: draw the replacement player
from the ADP-bearing pool rather than the raw projection ranking.

### OPEN BUG found while explaining the board — `qb_changed` false positives

Surfaced by a question about Christian McCaffrey, not by a test, which is
itself worth noting.

`compute_qb_continuity()` defines last season's quarterback as **the player
with the most pass attempts in 2025**, and compares him to the current 2026
depth-chart QB1. That is the right definition only when the starter played a
normal season. When a starter misses significant time, his backup can lead the
team in attempts — so the 2025 "primary QB" becomes the backup, the 2026 QB1 is
the returning starter, and the flag fires `qb_changed = True` for a team whose
quarterback situation is **stabilizing**, not changing. The feature reads the
sign backwards in exactly the case it most wants to get right.

McCaffrey is flagged and pays −0.30 PPG for it. San Francisco's 2025 attempts
leader needs checking against Purdy's missed time; if Mac Jones led, the flag
is wrong and CMC is being charged for Purdy's return.

Eight teams carry the flag: ATL, CLE, LV, MIA, MIN, NYJ, SF, WAS. At least two
more are suspicious on the same pattern — Miami and Washington both had starters
miss time in 2025.

Not fixed here, because it is a feature-definition change and every RB
coefficient was fitted with the current definition; changing it means a refit.
Carrying it to **Phase 13 CP1** alongside RB age-squared.

- Fix candidate: define last season's QB by **games started**, not attempts,
  or by attempts among players who started at least half the team's games.
- The same fault exists in the fitting data, so it is not purely a live-board
  problem — it has been adding noise to the coefficient all along, which if
  anything means the true `qb_changed` effect is larger than −0.4977.
- Second, separate check: `.fill_null(True)` on line 98 means a team with no
  matched depth-chart QB1 is silently flagged as changed. Confirm all 32 teams
  matched before trusting any of the eight.

### OPEN BUG — `position_competition_ppg` is diluted by roster length

Found the same way, one question later.

`compute_position_competition()` averages the trailing baseline PPG of **every
other player at the position on the roster**, unweighted. Rosters at this stage
of August are camp rosters, so that average is dominated by how many bodies a
team happens to be carrying rather than by who the player actually competes
with.

Detroit lists 6 running backs; Pittsburgh lists 13. Gibbs' competition score is
2.32, and the arithmetic behind it is Pacheco 8.85 plus **four players at 2.30,
0.37, 0.10 and 0.00**. The one back who threatens his touches contributes a
fifth of the number. Against a league RB average of 6.33 this is worth **+0.81
PPG** to Gibbs — most of which is a statement about Detroit's roster *count*,
not its depth chart.

Sensitivity, to size the problem:

| Definition | Gibbs' competition | Contribution |
|---|---|---|
| Current (mean of all 5 others) | 2.32 | **+0.81** |
| Best backup only (Pacheco) | 8.85 | **−0.51** |

A 1.3 PPG swing on a definition choice nobody has tested. For scale, the actual
football event Jack asked about — Montgomery (12.72) leaving for Houston and
Pacheco (8.85) arriving — moves Gibbs only **+0.16 PPG**. The definition is
worth eight times the roster move.

- **Direction agreed (Aug 4):** restrict the average to the teammates who
  actually compete for the touches, i.e. the **top k other players by trailing
  baseline, excluding self** — NOT "the 2nd and 3rd string." The exclusion is
  what makes it symmetric: for Gibbs the pool is Pacheco and Ozigbo, and for
  Pacheco it is Gibbs. A backup's competition *is* the starter, and a
  depth-chart-worded rule would zero that out for every non-starter on the
  board.
- Candidate set to backtest against actual next-season PPG: k=1 (max), k=2,
  k=3, snap- or depth-weighted mean, and **dropping the feature entirely** —
  which stays on the list because it was dead from Phase 6 to Phase 10 and may
  only look alive now because roster-length noise correlates with something
  real. Pick by backtest, not by feel.
- Same caveat as `qb_changed`: the fitting data carries the identical
  definition, so this needs a refit, not a live-board patch. Both go to
  **Phase 13 CP1**.
- Note this feature was *dead* from Phase 6 until Phase 10, when the wider
  training window revived it. It is plausible it only looks significant now
  because roster-length noise happens to correlate with something real
  (good teams carrying fewer camp bodies at a position they've solved). Worth
  testing the max-based definition before trusting the coefficient at all.

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

### Phase 12 — CODE WRITTEN, NOT YET RUN (Aug 6)

All three checkpoints are implemented and unexecuted. The fit has not happened, so
**every claim below is about the code, not about a result.** Nothing here says the
rookie model works.

New files: `src/rookie_backtest.py` (CP1), `src/fit_rookie_weights.py` (CP2),
`src/verify_rookies.py` (CP3). Run in that order, then `python -m src.pipeline`.

Two leakage guards are the substance of CP1, and both are places this phase would
otherwise have produced a good-looking wrong answer:

- **Leave-one-class-out cohort baselines.** `rookies.py` averages PPG across all of
  2021–25. Using that as the baseline for a 2023 rookie measures his delta against a
  number that already contains 2023. At n=5 each class is a fifth of its own
  baseline, so this is not a rounding concern — it shrinks delta toward zero for
  reasons unrelated to any feature, and the features then get credit for variance the
  baseline already ate.
- **Week-1 depth chart, not the latest.** The live path reads the newest snapshot,
  which is right for a season that hasn't started. Reading the newest snapshot of a
  *finished* season reads `pos_rank 1` for a rookie **because** he broke out. The
  earliest snapshot is the closest analog nflverse offers — still imperfect, since a
  team's first scrape may land later than the August snapshot the live model sees, so
  treat any `pos_rank` coefficient as an optimistic ceiling.

Also settled in CP1: **the cohort baseline is computed after the `MIN_GAMES` filter,
over exactly the rows that get fitted.** Both sides of the delta then move together
and the level nets out, which is why `SUPPRESS_LEVEL_SHIFT` is empty here where the
veteran fit needs it for QB. A large rookie intercept is therefore a *symptom* — it
means the filter and the baseline came apart — and `fit_rookie_weights` prints a
banner past ±1.5 PPG.

**The fallback is wired, not just documented.** `fit_is_trustworthy` is per position
(something cleared alpha *and* nothing flipped sign across the five folds), and
`ranking.load_rookie_weights()` silently drops the rest. A position that fails keeps
the flat cohort baseline — exactly what it had before Phase 12 — so failure costs
nothing and there is no path by which a fragile coefficient reaches a board.

**Open question for whoever runs it.** `COHORT_SEASONS` is five classes because
`load_depth_charts()` is thin before 2021. If `pos_rank` fails to earn its slot, that
constraint is gone and the window can widen to 2017 for one re-run. Check that
**before** concluding that n is the binding problem.

### 32-team superflex league added (Aug 6)

`league_config_32team.json`. 32 teams, 10 rounds, weeks 1–14 / playoffs 15–17, no
keepers, 4-point passing TDs, −1 interceptions. Roster is 2 RB / 2 WR / 1 TE / 1 FLEX
/ 1 SUPERFLEX / 3 bench — **no dedicated QB slot and no K or DST**, both confirmed
deliberate.

Four things broke or would have, none of them loudly:

1. **`UNMODELED_SLOTS_PER_TEAM` was a hardcoded 2.** Correct for both leagues that
   existed when it was written, which is exactly why it looked like a constant. This
   league starts no kicker and no defense, so it would have removed 64 picks from a
   320-pick draft that does not spend them — replacement level ~20% too shallow at
   every position, every VOR inflated, no error anywhere. Now derived from
   `roster_slots`. `make_charts_phase11.py` held a duplicate copy and now imports.
2. **`SUPERFLEX_SPLIT` added.** Without it `compute_starter_ranks()` gives QB zero
   starters, since this league has no QB slot for the count to read, and the floor
   that currently holds QB replacement honest (QB27) would not exist.
3. **`verify_adjustments.py` would have raised `ValueError`.** `deep, shallow =
   list(configs.keys())` unpacks exactly two. Named explicitly now, with a soft check
   for the superflex direction — nothing in that file could previously catch a
   `SUPERFLEX` slot being ignored.
4. **Scoring was computed once, for one league.** See below. This is the serious one.

### Every projection was in Lebron James scoring (Aug 6)

`features.py` scores every player under `league_config_lebronjames.json` and nothing
downstream re-expresses it. Invisible until now because the 12-team and 6-team configs
differ in teams, weeks and keepers — and in nothing that touches a point value.

The 32-team league is the first with different **scoring**. Uncorrected, its board
would rank quarterbacks on numbers **2.5–3.5 PPG too high apiece**, in a superflex
league where QB is the scarcest position and 27 of them come off the board. Nothing
would have raised; the columns all populate and every number looks plausible. The
board would simply have told you to draft quarterbacks.

`build_board.rescore_for_league()` fixes it exactly rather than approximately —
`passing_tds_per_game` and `passing_interceptions_per_game` are already in
`player_features.csv`, and `pass_td`/`interception` are the only scalar keys that
differ. **Any other differing key raises** rather than being silently ignored, because
a league that changes `reception` makes this arithmetic incomplete and a wrong answer
would be worse than a crash.

Two known gaps, neither papered over:

- ~~**Rookie QBs cannot be rescored.**~~ **FIXED (Aug 6).** The fix came from noticing
  what a cohort baseline *is*: the mean PPG of a set of real rookie seasons. The mean
  passing TD and interception rate of that **same set** is therefore the honest rate to
  correct that baseline with — the same average over the same players, one column
  across. Nothing modelled, nothing invented.

  `rookies.aggregate_rookie_season()` now carries those rates through to
  `player_features.csv`, so rookie QBs take the identical rescore path as veterans with
  no special case anywhere. Carried at skill positions too, where they are ~0 — not
  because a rookie receiver's passing matters, but so neither the CSV schema nor the
  lookup needs a "QB is special" branch, and a branch is what would rot.

  Note the magnitude is **smaller than the veteran correction**, and for a real reason:
  rookie QBs throw fewer touchdowns, so a −2/TD change costs them less. Roughly −1.1 to
  −1.6 PPG against −2.3 to −3.3 for established starters. The earlier "~2–3 PPG"
  estimate was veteran rates applied to rookies.

  `apply_rookie_baselines()` **raises** on baseline CSVs predating the new columns
  rather than joining nulls — the failure mode being prevented is precisely a silent
  return to the bug. `rescore_for_league()`'s warning is now keyed on the rate actually
  being null rather than on `is_rookie`, so it stays true as the data changes.
- **The fitted weights were fitted in base scoring.** QB carries one weight (`age`,
  −0.19 PPG/yr) with its level shift already suppressed, so the residual is a fraction
  of a point and reorders nothing. Recorded because it is real, not because it is
  urgent. A league that changed `reception` would make it matter at every position at
  once, and that league is the one that should refit.

### Superflex ADP — a second feed, not a fudge (Aug 6)

`adp.py` pulled `ppr` only, i.e. one-QB mocks. For a superflex board that feed is not
noisy, it is **biased in a known direction**: quarterbacks come off the board earlier
and roughly twice as deep than it will ever show, so `compute_replacement_ranks()`
sets QB replacement far too shallow and undervalues every QB. This is the plan's
existing `teams=12` caveat one step worse — there the mix is the wrong *size*, here it
is the wrong *shape*.

FFC publishes no `superflex` endpoint. It publishes `2qb`, live for 2026, and that is
now pulled alongside `ppr` on every run and attached as `adp_2qb` etc. Leagues select
via `"adp_format"` in the config; `build_board.select_adp_variant()` remaps the
suffixed columns onto the canonical names once, so nothing downstream learns there was
a choice and all boards still ship from one model run (CP4 holds).

**The caveat that has to stay attached to the number:** 2QB *requires* a second
starting quarterback where superflex only *permits* one, so `2qb` overstates QB demand
somewhat. That bias runs opposite to `ppr`'s and is much smaller. Bracketing the truth
between two feeds beats picking one and forgetting which way it leans. Printed at
build time, not left in a comment.

### Phase 12 — FIRST FIT (Aug 6)

n=241 rookie-seasons over five classes: QB 20, RB 68, TE 44, WR 109. Mean delta
overall +0.045, against ~0.000 for an in-sample baseline — the leave-one-class-out
guard is engaged. Per-position deltas run +0.19 / +0.08 / −0.14 / −0.21 against
**standard deviations of 3.0–4.7**, which is the number to hold onto: rookie outcomes
inside a (position, round) cell vary by 3–5 PPG and the features are trying to explain
a slice of that.

**What survived.**

| Pos | n | R² | shipped |
|---|---|---|---|
| QB | 20 | — | not fitted, below `MIN_ROWS_TO_FIT` |
| RB | 68 | 0.101 | `rush_att_pg` −0.444 (p=0.043) |
| WR | 109 | 0.024 | `position_competition_ppg` −0.530 (p=0.065) |
| TE | 43 | — | nothing cleared alpha |

**WR `position_competition_ppg` is the phase's actual result**, and it is the one that
was supposed to be. Negative, as predicted — better incumbents means fewer targets —
and stable across all five folds (−0.39 to −0.68). It is the only feature in the set
that distinguishes two same-round rookies on different teams by something other than
team pace, and it is the reason Love and Price no longer tie. R²=0.024 is small and
should be quoted whenever the coefficient is.

**RB `rush_att_pg` came out NEGATIVE, and it is not measuring what its name says.**
A rookie back landing on a high-volume rushing team does *worse* against his cohort.
The coherent reading is that heavy rushing volume is evidence a team already has an
established back — so this is a competition proxy, arriving at a position where
`position_competition_ppg` itself failed (p=0.32). Worth testing directly in Phase 13
rather than left as an inference: if the two are measuring one thing, the better-named
one should be able to do the job.

**RB `age` was dropped after the folds, and this changed the fitter.** It cleared alpha
at p=0.052 with no sign flip anywhere, which reads as stable until the folds are
actually read: 0.918, 1.126, 1.314, 0.739, **0.286**. Withholding one class moves it by
4.6× end to end. Sign stability was the only LOSO bar the veteran fit ever needed, and
it passed something it should not have.

`fit_rookie_weights` now also flags **magnitude** instability (`STABILITY_RATIO = 3.0`,
deliberately loose — every fold shares 80% of its rows with the shipped model) and adds
a **stage 3**: drop the unstable, refit once, re-run LOSO. Once, not in a loop —
iterating until everything passes is fitting the fold structure, which at five folds is
fitting five numbers. Stage 3 is what saves RB: `rush_att_pg` is solid across every
fold and would otherwise have been discarded along with `age`, since trustworthiness is
judged per position.

**Two checks were wrong and both cried wolf.**

- **The intercept banner.** RB's intercept is +12.04 PPG and the check fired. It is
  meaningless: `rush_att_pg` enters *uncentered* with a mean near 26.5 and a
  coefficient of −0.444, so the intercept must carry +11.8 just to cancel it. An
  intercept is only a level when every feature is centered, and only `age` and `pick`
  are. The check now reads the fitted value at the feature means — which OLS forces to
  equal mean(delta) — giving +0.19 at RB and +0.08 at WR. Same lesson this project
  keeps relearning, applied one step earlier than usual: a coefficient means nothing
  apart from the constants it was fitted with, and that governs *reading* them too.
- **The separation check counted players, not teams.** It failed on WR round 3 (9
  players, 8 distinct) and round 6 (7, 6) — one duplicate pair each, both same-team.
  Two rookies on the same team at the same position in the same round are identical in
  every feature the model has and *should* tie; separating them would mean inventing
  something. CP3 asked about different teams, and the check now asks that.

**Shipped after stage 3.** RB `rush_att_pg` −0.364 (folds −0.257 to −0.487), WR
`position_competition_ppg` −0.530 (folds −0.391 to −0.681). RB's R² falls 0.101 → 0.056
once `age` is removed, which is the honest number: most of what `age` was explaining was
one class. All four `verify_rookies` sections pass — reconciliation to 1e−8 at both
positions, 12 of 12 cells separating by team, rookies at 2.8–11.1% of each position's
top 36, no cross-contamination.

QB and TE tie across teams in every cell, correctly and visibly — the harness marks them
`<-- TIE ACROSS TEAMS` without judging them, which is what the flat cohort baseline
looks like when it is still in force.

**Standing gaps.** QB has no rookie model (n=20 against 4 rows per feature) and rookie
QBs are also stranded in base scoring — both landing on the same position, on the board
most sensitive to it. TE fits nothing. `COHORT_SEASONS` can widen to 2017 now that
`pos_rank` failed everywhere, since depth-chart coverage was the only reason for the
narrow window; that is the cheapest available shot at QB and TE.

**MODEL_VERSION 12 → 13, and it nearly didn't happen.** Phase 12 was built alongside a
new league and the attention was on that league — but `ranking.apply_situational_weights`
is shared, so the moment `rookie_weights.json` appeared, rookie ranks moved on **all
three boards**, not just the 32-team one. That is a ranking-logic change by this
constant's own definition. Shipping it under v12 would have put two different models
under one version number, which is the Build History problem one level up: there it was
three rebuilds of one version, here it would have been two models. **All three boards
need rebuilding off the bump** — the 12-team and 6-team files on disk predate Phase 12
and are now stale in a way their filenames do not admit.

**One reporting bug worth recording**, because it is the same shape as several real
ones: the run printed "DROPPED AFTER LEAVE-ONE-CLASS-OUT" followed by nothing. Stage 3
overwrote the flag list with the *refitted* model's flags, which are empty whenever the
refit succeeded. "What got dropped" and "does what remains hold up" are different
questions and now have different keys (`instability_flags`, `residual_instability`).
Nothing numeric was affected — but a diagnostic that silently reports the wrong model is
how Phase 6 went unnoticed for two phases.

### Phase 12 — window widened to 2017–2025 (Aug 6)

`COHORT_SEASONS` now matches `backtest.py`. 241 rookie-seasons becomes roughly 430.

**Why this is not fishing, which is the first thing to ask.** The window is being changed
*after* seeing QB and TE fail, which is exactly the shape of move `fit_weights` warns
about under ON MULTIPLE TESTING. It is defensible for one specific reason, and if that
reason does not hold this should be reverted: **the 2021 floor was never a judgement
about the right amount of history.** It was a data constraint, stated as such at the
time — `load_depth_charts()` is thin before 2021 and `pos_rank` was the only feature
needing it. That constraint is void, because `pos_rank` failed at every position
(p = 0.16 to 0.54) and is being removed. Dropping a constraint whose stated
justification no longer applies is not the same as widening until something passes.

Alpha stays at 0.10, the candidate list does not grow, and this is recorded as a
**second look at the same hypotheses** — a feature clearing alpha here that did not at
five classes deserves more suspicion, not less.

**The trap it opens, and the guard.** Had `pos_rank` been kept, `depth_chart_missing`
would stop meaning "this rookie was buried" and start meaning "this season predates good
scraping" — a season label wearing a feature's name. It would very likely test
significant, because early and late classes differ for a hundred reasons, and the
coefficient would be uninterpretable. Worse than no feature, because it looks like a
finding.

`season_confounded_features()` computes each candidate's missing rate per class and
**removes** any whose spread exceeds `SEASON_CONFOUND_SPREAD = 0.35`, along with its
companion indicator. Removal rather than warning is deliberate: a warning arrives after
the number exists, and numbers that exist get used. `rookie_backtest` prints the
per-class coverage table so the decision is legible rather than magic.

One consequence worth noting because it would have been a silent failure:
`verify_rookies` rebuilds the fit sample to check reconciliation, and reading
`FEATURE_SPECS` after the filter had removed something would drop nulls on a different
column set, changing which rows survive and breaking the OLS identity for reasons
unrelated to the weights. The effective spec now ships in the JSON as
`features_considered` rather than being recomputed in two places.

**Expected outcome, stated before the run.** RB ~122, WR ~196, TE ~79 — all comfortably
fitted. **QB ~36, still below `MIN_ROWS_TO_FIT = 40`.** Roughly four rookie
quarterbacks per class clear `MIN_GAMES`, and nine classes does not fix that. TE is the
position this stands to help; QB needs a different idea, not more of the same seasons.

### The rookie haircut — asked twice, answered neither time (Aug 6)

Phase 12's stated fallback was "fall back to the shrinkage haircut." Two attempts to
size it, and **both instruments were wrong.** Recorded in full because a plausible
number from a broken test is more dangerous than no number.

**Attempt 1 — sweep λ in `projection = λ × cohort + (1−λ) × anchor`.** Result: λ=1.0
wins, monotonically, RMSE rising from 3.79 to 5.46 as λ falls. Clean-looking and close
to a mathematical necessity: the cohort baseline **is** the conditional mean of the very
population being scored, so shrinking it toward anything else must raise its own RMSE
unless it buys more variance than it costs in bias — and with 40–100 players per cohort
cell there is little variance left to buy. A test that could only return one answer.

**Attempt 2 — compare mean residuals, rookies vs veterans.** Gap of −0.057 PPG, which
the script declared meaningful and it is not, for two independent reasons:

- **The threshold was borrowed.** `MEANINGFUL_GAIN = 0.02` was chosen for RMSE churn in
  the competition bake-off — different statistic, different units. Against residual SDs
  of 3–5 PPG and n = 2775/415, the standard error on that gap is ≈0.21, so −0.057 is a
  quarter of one standard error. Now compared against its own SE (|z| < 2) instead of a
  constant carried in from elsewhere.
- **Both sides are pinned near zero by construction.** OLS with an intercept forces the
  veteran residual to equal mean(delta), which the situational adjustment then absorbs;
  the rookie cohort baseline is the mean of its own cell. The comparison mostly measures
  what two fitting procedures already flattened.

**What the real question is.** Both baselines are estimated on players who cleared
`MIN_GAMES = 8`, so each is *"expected PPG **given** you earn a role."* The board then
applies them to everyone — which is far more forgiving to rookies, because a much larger
share of them never earn the role. `rookie_backtest` kept **415 of 600** drafted rookies
who took a snap, and rookies with no snap never entered that 600 at all.

So the question is not "is the cohort mean biased" but **"what is P(earns a role) for a
rookie versus a veteran, and does the board price the difference."** It does not. That is
a playing-time model, adjacent to the `expected_games` machinery Phase 11 CP8 built for
PUP/NFI, and it is real work rather than a constant.

**Not to be attempted before Aug 22.** Handle rookies by judgement on draft day: the
board's rookie ranks are its least-evidenced output and its largest disagreements with
the market, and now the reason is written down.

### Phase 13 CP1 — the Tuten test, and what it cost to run (Aug 6)

**Question:** the board ranks Bhayshul Tuten ~150th against an ADP of 51, with every
driver on him *positive*. He ranks low because his baseline is thin production. The
market prices an expected 2026 role; the model prices demonstrated snaps and has no
channel through which "he is the starter now" can reach a projection. Depth chart rank
is the only pre-season, statistics-only signal that carries role information, and
`FEATURE_SPEC.md` had ruled it "secondary tie-breaker only" — a decision, never tested.

**Answer: it works at running back and nowhere else.**

| pos | `pos_rank` ablation, mean of 3 folds | |
|---|---|---|
| **RB** | **+0.080** | ships, coefficient −1.06 PPG per place |
| TE | −0.025 | cut |
| WR | −0.224 | cut |

Being RB1 rather than RB3 is worth ~2 PPG. That is the missing channel, and it exists at
exactly the position Tuten plays. Why only there is a football answer: a running back
depth chart is close to a declaration of touch allocation, while receivers rotate by
package and a WR3 label says little about targets.

**The WR failure is the more useful half.** `pos_rank` was sign-stable across all nine
LOSO folds — −0.286, −0.319, −0.299, −0.306, −0.280, −0.301, −0.350, −0.305, and then
**−0.886** on 2025. Same sign every time, so the sign-flip check waved it through, and
the tripled coefficient took the entire WR model below a constant out of sample
(−0.2516 RMSE).

A magnitude-stability bar had existed in `fit_rookie_weights` since RB `age` failed it
there, and had never been ported to the veteran fitter — for no better reason than that
the veteran fit had not yet met a feature that needed it. Now ported, with one
deliberate difference: **the rookie fitter drops unstable features, the veteran fitter
reports them and lets the gate decide.** Two automatic droppers in sequence would be
fitting the fold structure twice.

**Ablation is conditional, and that nearly cost four good features.** The first gate run
failed six items. Four of them — WR `workload_share` at −0.158, TE `workload_share`,
WR `depth_chart_missing`, RB `trend_missing` — were measured *with the broken WR
`pos_rank` still in the model*. Removing it returned WR `workload_share` to +0.028 /
+0.045 / +0.019. **A feature's ablation number is only meaningful against a model that
is otherwise sound**, so a gate failure list should be worked one item at a time,
worst first, not cut wholesale.

**`trend_missing` is machinery, not a candidate.** It failed the gate at RB: cleared
alpha in one fold of three, worth −0.0003 RMSE when it did. Cutting it would have been
wrong — it exists so `usage_trend_share` (186 of 711 RB rows mean-imputed) can be told
apart from "we do not know," and judging it by predictive ablation is the same category
error as ablating an intercept.

Subjecting it to alpha caused a real problem too: it flickered in and out across folds,
so `usage_trend_share` did not mean the same thing in each one and the fold comparison
the gate rests on was not comparing like with like. Companions are now forced in
whenever their imputed partner ships, and exempted from the gate's feature rule.

**That surfaced a live bug of its own.** TE has been shipping `usage_trend_share` with
152 rows imputed and **no indicator** since Phase 10, because alpha cut `trend_missing`
at p=0.68 — directly against this file's own stated rule. Nothing noticed, because the
rule lived in a comment instead of in the code. It does not any more.

**Gate passed, and the board moved.** Tuten goes #148 → #109 on the 12-team board with
`+0.9 pos_rank` in his drivers. The gap that remains against his ADP of 51 is his
15-game sample — the model saying "unproven," which is the part it should say.

**The side effect that needs a human read: running backs went 25 → 31 of the
12-team top 60.** The mechanism is sound. `pos_rank` pushes committee and handcuff backs
down hard, so RB54 — the 12-team replacement — is now a materially worse player, and
every real starter gains VOR against him. The 6-team board is **unchanged** at 26,
because RB32 there is still a genuine starter and the effect has nowhere to bite. That
divergence is the league-awareness machinery working exactly as designed.

Sound mechanism, but 31 of 60 is a strategic claim, and CP5 exists for precisely this
kind of call. Worth checking that the backs filling ranks 25-31 are ones you would
actually take there.

Worth noting `pos_rank` and `workload_share` are correlated by construction and pull in
opposite directions at RB — "you have the job" against "your usage is already priced
in." Both clear alpha with tight standard errors and both earn their slot on the
holdout, so it is not degenerate, but it is the multicollinearity CP1 was told to look
for and it is now on the record.

### Competition definition bake-off — no change, and that is the finding (Aug 6)

Five definitions per position, fitted as five separate models and scored on the same
three folds. **Not** one model containing all five: they measure the same quantity, so
in a single regression they mask each other, split one effect five ways, and hand the
win to whichever way the noise fell. That failure would have looked exactly like a
result.

`ppg` — the incumbent mean-of-all-teammates — wins or ties everywhere, and top1/top2/top3
land within 0.01 RMSE of it.

**The more interesting number is `none`.** The whole feature is worth +0.043 at RB,
+0.016 at TE, +0.009 at WR. The Aug 4 scoping note found the definition choice was worth
1.3 PPG on Gibbs against 0.16 for the actual roster move it was asked about — and that
was true. But 1.3 PPG of *individual movement* buys essentially nothing in *predictive
accuracy*. Those two things are far less related than a large coefficient suggests, and
that is worth remembering the next time one looks impressive.

### Phase 13 CP2 — HOLDOUT VALIDATION, RUN AT LAST (Aug 6)

Three folds: hold out 2025, 2024, 2023, refit on the rest, predict the held-out season.
`src/holdout.py`. Every R² this project has ever quoted was in-sample; this is the first
out-of-sample evidence it has.

**The plan's test had a hole, found while writing it.** CP2 asked "does the model beat
the raw baseline out of sample?" — a bar a model of pure noise can clear, because the
intercept alone clears it and the intercept is not a feature. So there are three
predictors: **RAW** (delta = 0), **LEVEL** (delta = the training mean, no features), and
**MODEL**. The number that matters is **MODEL vs LEVEL**. As it happens LEVEL beats RAW
by at most ±0.04 anywhere, so the intercepts are contributing nothing and the veteran
gains are genuinely features — but that had to be measured, not assumed.

| model / pos | 2025 | 2024 | 2023 | mean | folds passed |
|---|---|---|---|---|---|
| veteran RB | +0.155 | +0.291 | +0.491 | **+0.313** | 3/3 |
| veteran WR | +0.420 | +0.425 | +0.245 | **+0.363** | 3/3 |
| veteran TE | +0.160 | +0.085 | +0.112 | **+0.119** | 3/3 |
| veteran QB | +0.104 | −0.128 | −0.044 | −0.023 | 1/3 |
| rookie RB | −0.153 | 0.000 | −0.143 | −0.099 | 0/3 |
| rookie WR | 0.000 | 0.000 | 0.000 | 0.000 | 0/3 |
| rookie TE | +0.038 | +0.327 | +0.234 | +0.200 | 2/3 |

**The veteran model is real.** RB, WR and TE beat a constant in every fold, and `age` is
the workhorse throughout (+0.189 RB, +0.161 WR by ablation). Nine phases of work on
those three positions survives its first honest test. That is the headline and it should
be said before the rest.

**QB `age` does not survive, and it was Phase 10's headline finding.** "QB carries a
weight for the first time in the project's history" — 1/3 folds, mean −0.023. The
in-sample story was coherent (p=0.020 on 157 quarterback-seasons, stable across all nine
LOSO folds) and it is still wrong out of sample. LOSO stability was never evidence of
prediction, which is exactly why this file exists.

**`usage_trend_share` at WR falls, as it was pre-committed to.** `fit_weights` recorded
it as "the first coefficient that should fall if Phase 13 CP2's holdout disagrees." It
disagrees: selected in only 1 of 3 training folds, and negative (−0.016) when selected.
`trend_missing` at WR is the same story (1 fold, −0.008). The pre-registration worked —
the call was made before the evidence arrived and the evidence settled it.

**The rookie model largely does not survive, one day after being built.**

- **rookie RB: 0/3, and `position_competition_ppg` actively hurts (−0.148 mean).**
- **rookie WR: 0/3, and it is the strongest negative in the table.** The feature failed
  to clear alpha in *every* training fold. It only becomes significant when the test
  season is included in the fit — which is the definition of the thing a holdout is for.
- **rookie TE: 2/3, mean +0.200 — but read which feature.** `age` carries it (+0.235),
  `pos_rank` barely registers (+0.055). That is the **reverse** of the in-sample fit,
  where `pos_rank` had the larger coefficient (−1.03 vs −0.76) and `age` looked
  secondary. The suspicion recorded when `pos_rank` doubled its coefficient on the
  second look was correct.

  **Caveat that has to travel with this:** rookie TE test folds are n = 10, 7, 9. A
  +0.327 RMSE gain on seven players is not a finding. It is the least-evidenced cell in
  the table and it survives on the strength of the feature, not the sample.

**What this cost.** Phases 10 and 12 lose most of what they claimed. That is the system
working — the alternative was carrying four fictional features into a draft. But it also
says something about the standard of evidence used up to now: every one of these cleared
alpha, several cleared it twice, and LOSO stability passed them all.

### Two verification checks had drifted from what ships (Aug 6)

Both found by reading output that said everything was fine. Same failure in two places:
**a check that reimplements part of the shipping path instead of calling it.**

**1. "rookies take no situational adjustment" went stale the day Phase 12 shipped.**
True from Phase 5 to Phase 11, false the moment `rookie_weights.json` existed. It
hard-FAILED and printed *"Do not build a board from this"* about boards that were
correct — and `build_board` does not read that file, so they built anyway. A gate that
says stop while the thing it guards proceeds trains you to ignore it, which is worse
than having no gate. Now asserts the real invariant: rookies at positions with **no**
rookie model must be exactly zero.

**2. `check_replacement_levels` was measuring a board that does not exist.**
`build_board` applies `select_adp_variant()` and `rescore_for_league()` before it touches
any PPG or ADP column. This check skipped both. Harmless for the two original leagues —
default feed, base scoring, nothing to transform — and wrong for the 32-team one:

| | QB | RB | WR | TE | feed |
|---|---|---|---|---|---|
| checker | 53 | 109 | 125 | 39 | ppr, 202 picks |
| board | 43 | 89 | 140 | 48 | 2qb, 184 picks |

QB replacement off by ten places, WR by fifteen, and quarterbacks compared at 6-point
passing TDs — which is most of why its top-30 showed 14 of them. **It passed**, on
numbers belonging to no board anyone would draft from. That is the more dangerous kind
of wrong.

**3. And then the replacement fix reintroduced it.** The rewritten superflex check
recomputed replacement ranks itself, *after* the per-league loop, reading the leaked
`players` variable — which by then held the 32-team frame, already switched to the `2qb`
feed and 4-point scoring. It reported the 12-team league at **QB36** while the loop three
lines above printed **QB22** for the same league. It passed, on a number belonging to no
league.

Three instances of one bug in one day, the third written by the hand fixing the second.
The loop's own results are now captured per label and the check reads those;
`compute_replacement_ranks` is called in exactly one place in the file. Removing the
opportunity beats restating the rule.

The generalizable rule, and the one all three fixes encode: **a check that reimplements
part of the shipping path must call the shipping path.** It is the same principle
`verify_adjustments` was founded on — run `ranking.apply_situational_weights`, not a
restatement of it — applied one level out to the league-specific transforms.

### Mock draft run — and it reverses the QB call (Aug 6)

Roster corrected first: bench is **4**, not 3. 11 rounds, **352 skill picks**.

A 32-team superflex mock was run with one human team and 31 autodrafters. Observed
counts entered as `expected_drafted`, which retires the 42%-extrapolation warning
entirely — replacement level on that board is now measured rather than guessed.

| pos | model @352 | observed | diff | model share | observed share |
|---|---|---|---|---|---|
| QB | 47 | **59** | **+12** | 13.4% | **16.8%** |
| RB | 98 | 92 | −6 | 27.8% | 26.1% |
| WR | 154 | 145 | −9 | 43.8% | 41.2% |
| TE | 53 | 56 | +3 | 15.1% | 15.9% |

**Twelve of thirty-two first-round picks were quarterbacks.** The ADP-tail
extrapolation implied ~13% of the draft would be QBs; the room took 16.8% overall and
38% of round one.

**This reverses the "don't pay superflex prices for QB" conclusion recorded above, and
the reversal is instructive.** That call rested on replacement sitting at QB43, and
QB43 came from extrapolating 42% of a draft off the last 40 picks of a non-superflex-
adjacent feed. Replacement is now QB59 — a much worse quarterback, so every real one
gains VOR — while RB and WR come out *shallower* than modelled and lose a little. Both
effects push the same direction.

The 4-point-passing-TD finding still stands on its own terms: an elite QB really does
lose ~3 PPG against the 6-point scoring all superflex advice assumes. That was correct
and it was half the equation. The replacement half was wrong, and it was the half
resting on a guess the code was already warning about in capital letters.

**Caveat that travels with these numbers:** 31 of 32 teams autodrafted, so this is the
platform's superflex ADP behaviour, not a human room's. If the real league drafts live
and engaged, the mix could move. It is still incomparably better evidence than the tail
of a 12-team feed, and it is the first draft-behaviour data this project has ever had.

`compute_replacement_ranks` now **raises** if `expected_drafted` does not sum to the
draft's pick count. Hand-entered counts replace the ADP machinery entirely, so nothing
downstream can notice if they are short or long — ten missing picks would move
replacement at every position with no error anywhere. A draft has a known length, which
makes this checkable, so it is checked.

### What the fix revealed: superflex barely lifts QB at 4 points (Aug 6)

Once the check read the right feed and the right scoring, quarterbacks in the superflex
top 30 went **14 → 1** — the same as the 12-team board. That is not a bug, and it is
probably the most useful thing on that board.

| pos | player | PPG | replacement | repl PPG | VOR |
|---|---|---|---|---|---|
| QB | Josh Allen | 23.76 | QB43 | 12.20 | 11.56 |
| WR | Puka Nacua | 20.21 | WR140 | 4.57 | **15.64** |
| RB | Jahmyr Gibbs | 20.59 | RB89 | 5.28 | **15.31** |

Allen outscores Nacua by 3.6 PPG and is worth 4.1 *less*. Two things stack: at 4-point
passing TDs an elite QB loses ~3 PPG against the 6-point assumption everyone's superflex
advice is built on, and quarterbacks have high floors — the 43rd-best QB still scores 12
where the 140th-best WR scores 5. VOR is a *distance* above replacement, and at QB that
distance is compressed however many come off the board.

**Practical read for that draft: do not pay superflex prices for quarterbacks.** The
room will, because superflex convention assumes 6-point passing. That gap is an edge, and
it is the one place this board disagrees loudly with consensus.

The top-30 count was the wrong instrument for the check, though — it was built to catch a
`SUPERFLEX` slot being silently ignored, and that shows up in replacement **depth**,
upstream of scoring and VOR compression. Now asserted directly: superflex must draft
deeper at QB than the 1-QB league (QB43 vs QB22) and must respect its own starter floor.
Both hard checks, because both are structural facts rather than judgement calls.

### Cuts applied, and the gate (Aug 6)

Removed from the **candidate lists**, not merely unshipped — leaving them in means alpha
can readmit them at the next refit, and alpha is the bar that passed them to begin with.

| Cut | Evidence |
|---|---|
| veteran QB `age` (position removed) | 1/3 folds, mean −0.023 |
| veteran WR `usage_trend_share` | 1/3 folds selected, −0.016 when selected |
| veteran WR `trend_missing` | 1/3 folds selected, −0.008 |
| veteran RB `qb_changed` | 1/3 folds selected, −0.031 |
| rookie RB (position removed) | 0/3, feature −0.148 |
| rookie WR (position removed) | 0/3, never cleared alpha in any fold |
| rookie QB (position removed) | never fittable — 35 rows against a 40 threshold |
| rookie TE `pos_rank` | +0.055 against `age`'s +0.235 |

What ships: **veteran RB / WR / TE**, and **rookie TE on `age` alone**. QBs take a zero
adjustment, as they did from Phase 5 through Phase 9. Cutting `pos_rank` also retires the
week-1-versus-August depth chart mismatch — the model no longer leans on a feature
measured at a different moment than it is applied.

**The gate.** `python -m src.holdout --gate` runs all three folds, writes
`data/holdout_gate.json`, and exits non-zero on failure. `build_board` refuses to build
without a passing gate.

The rule, fixed so it cannot drift: a shipped **position** must beat a constant on
average across folds; a shipped **feature** must have non-negative mean ablation value.
Averages, not unanimity — one bad fold in three is noise, and demanding 3/3 would cut
features that are real.

Three ways to fail, and the third is the one that matters: **MISSING** (never validated),
**FAILED** (something shipped has no out-of-sample value), and **STALE** — the weights
are newer than the gate. Mtimes are compared rather than just reading `passed`, because
refitting after a green gate leaves a passing file describing a model that no longer
exists. A green light for the wrong model is worse than no light. Same failure the Build
History sheet was added to solve, and the same shape as the stale-weights check already
in `verify_adjustments`.

`--skip-gate` exists and says so loudly in the output. It is for emergencies, not for
when the gate is inconvenient.

### Phase 12 — refit on nine classes (Aug 6)

415 rookie-seasons, up from 241. All four sections of `verify_rookies` pass, 16 of 16
cells separate by team, and **TE now fits.**

**The 2021 floor was based on an assumption nobody checked, and it was wrong.**
Depth-chart missingness across 2017–2025 runs 0% to 13.6%, spread 0.14 — comfortably
inside the 0.35 confound bar. Coverage was never thin. The constraint that shaped the
entire phase, that forced n=5, and that was cited as the reason QB and TE could not be
fitted, did not exist. It cost this phase its statistical power for four classes'
worth of data that was sitting there the whole time.

| Feature | coef @5 | p @5 | coef @9 | p @9 | |
|---|---|---|---|---|---|
| RB `rush_att_pg` | −0.410 | 0.043 | −0.220 | 0.125 | dropped out |
| RB `position_competition_ppg` | −0.334 | 0.321 | −0.414 | 0.063 | **swapped in** |
| WR `position_competition_ppg` | −0.629 | 0.065 | −0.434 | 0.095 | held |
| TE `pos_rank` | −0.574 | 0.541 | −1.181 | 0.044 | **swapped in, coef ×2.06** |
| TE `age` | −0.791 | 0.117 | −0.681 | 0.045 | **swapped in, coef ×0.86** |

**The RB swap answers the question this plan carried to Phase 13, early and cleanly.**
The five-class note read: *"if `rush_att_pg` and `position_competition_ppg` are
measuring one thing, the better-named one should be able to do the job."* On nine
classes it does. `rush_att_pg` falls to p=0.125 and `position_competition_ppg` takes
over. RB and WR now ship the same feature, which is the coherent result — competition
for touches is the rookie landing-spot story at both positions, and `rush_att_pg` was
a worse-named proxy for it that only won on a sample too small to separate them.

**R² FELL at RB (0.101 → 0.037) and WR (0.024 → 0.011) on more data, and that is an
improvement.** The five-class fits were partly describing their own sample. A lower R²
carried by a feature that holds across nine folds is a better model than a higher one
carried by a feature that does not — and RB's old headline feature is exactly the one
that did not survive.

**TE `pos_rank` is the result to be suspicious of, per this plan's own instruction.**
It was pre-committed that anything clearing alpha on the second look deserves *more*
scrutiny, and this cleared from p=0.541. The tell is not the p-value, it is the
coefficient: it **doubled**. Pure power gain tightens the standard error and leaves the
estimate roughly where it was — which is what TE `age` did (SE 0.333, coef ×0.86, a
textbook power story). `pos_rank` tightened its SE (0.930 → 0.575) *and* doubled, which
says the 2017–2020 classes carry a stronger relationship than 2021–2025 rather than
merely a better-measured one.

It is stable across all nine folds (−0.683 to −1.228) and it ships. But **the
"optimistic ceiling" caveat in `rookie_backtest`'s docstring is now load-bearing** in a
way it was not when written: the historical feature is a week-1 depth chart, taken
after final cuts, while the live model reads an August one taken before them. TE's
primary feature is the one most exposed to that gap, so live TE rookie projections
should be expected to be more confident than they deserve. **First candidate to cut if
Phase 13 CP2's holdout disagrees.**

**Visible cost of the refit:** Jeremiyah Love was 8th on the 12-team board under the
five-class model and is out of the top 10 under the nine-class one. Same player, same
data, different training window. That is the honest size of the uncertainty here and it
should temper how hard any rookie is drafted off this board.

**QB: 35 rows against `MIN_ROWS_TO_FIT = 40`.** Predicted before the run and unchanged.
Roughly four rookie quarterbacks per class clear `MIN_GAMES`; nine classes does not fix
it and neither will more seasons. QB needs a different idea or the flat baseline.

**One message bug**, no numbers affected: the coverage line printed "above 0.35" as
static text regardless of the actual spread, so it claimed `pos_rank` would be dropped
while the guard was correctly keeping it. The guard was right and its narration was
wrong — now conditional.

### First run of the 32-team board — three bugs (Aug 6)

The board built. Two crashes and one wrong number, and the wrong number is the one
worth reading.

**1. `load_depth_charts()` returns two different schemas.** Historical seasons come
back as `club_code` / `week` / `depth_team`; the live 2026 pull comes back as `team` /
`dt` / `pos_rank` / `pos_abb`. nflverse rebuilt the historical charts from a different
source than the live scrape, so this is permanent, not version skew.
`rookies.get_latest_depth_chart()` only ever sees the live shape, which is why the
project never hit it until Phase 12 asked for history. `week_one_depth_chart()` now
handles both; `rookies.py` was deliberately left alone rather than speculatively
rewritten against a shape it does not see.

Silver lining: the historical shape carries `week`, which is a *better* answer to leak
#2 than `dt`. `dt` is a scrape timestamp that could land anywhere in the preseason;
week 1 is a dated, pre-outcome moment.

**2. `round` reads back from CSV as a string** whenever the column contains nulls
(undrafted rookies). It still grouped correctly, so this surfaced three lines later at
a format specifier rather than anywhere near the cause.

**3. QB replacement level came out at QB63.** There are 32 starting quarterbacks in the
NFL. QB63 is a third-stringer projecting near zero, which handed every real
quarterback a VOR roughly equal to his whole projection and put Josh Allen 3rd overall
— a superflex board should raise QBs, but not by pricing them against nobody.

The cause was the ADP shortfall rule, and it is a real modelling error rather than a
typo. The FFC feed covers 184 of this league's 320 skill picks, and the old rule scaled
every position's count by `skill_picks/observed` — 1.74× — which asserts that the
position mix of the covered draft continues unchanged through the 42% that isn't
covered. **QB and TE draw from a finite startable pool; RB and WR do not.** Rounds 8–10
of any draft are running back and receiver dart throws. Nobody takes a 40th
quarterback. Scaling QB at the same rate as WR says otherwise.

Now extrapolated using the position mix of the **deepest covered picks**
(`TAIL_WINDOW = 40`) instead of the mix of the whole feed. If quarterbacks have stopped
going by the time the feed runs out, the tail says so and the extrapolation stops adding
them — no invented parameter, no hand-set cap, still drafter behaviour, just read at
the point that describes the picks being estimated. Past
`EXTRAPOLATION_WARN_SHARE = 0.25` the run states plainly what fraction of the draft is
estimated rather than observed.

**This changes nothing on the existing boards.** Both have feed coverage well past their
pick count (201 against 168 and 84), so neither ever enters the branch. The rule was
only ever exercised by a league big enough to outrun the feed, and the first such league
is the one that exposed it.

**Still outstanding on this board:** replacement level rests on 40 observed picks and an
assumption. Set an `expected_drafted` block from real results as soon as there are any.

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
