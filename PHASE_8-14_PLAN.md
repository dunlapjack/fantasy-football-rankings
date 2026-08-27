# Phase 8+ Plan — 2026 Drafts (board complete August 22)

Chapter 2 of the project. Phases 1–7 built a working statistics-only pipeline and
shipped `2026_Draft_Board_v7.xlsx`. This chapter is about **calibration**: making the
adjustments trustworthy rather than just present.

Ground rules carried over unchanged: statistics only, no analyst opinions, ADP is a
reference column and never a model input. League rules unchanged from Phase 7
(12-team full PPR, 6-pt pass TD, 16 rounds, keeper rules per `league_config_12team.json`).

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

**A second draft was added.** The 6-team league drafts first,
so the finished board is needed by **August 22**, not August 29. That league is 6 teams,
no keepers, regular season weeks 1–12 with playoffs in weeks 13–14 — scoring and roster
slots otherwise identical to the 12-team league. It gets its own `league_config_6team.json`,
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
against every other position and put Josh Allen in the 12-team top 10.

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
Chubb −222 (trend −26.3), Austin Ekeler −322 on the 6-team board (trend −29.7).

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

**Expanded, and the reason matters.** The 6-team league's regular season is weeks 1–12
with playoffs in 13–14, so the fantasy-relevant season is 12 games, not 17. Every
games-available calculation in this phase is now league-dependent: a four-game PUP
absence costs 4/12 = 33% of that league's season against 4/14 = 29% of the 12-team's
(weeks 1–14, playoffs 15–17). Kittle and Charbonnet should be ranked lower on the
6-team board than on the other one — the same player is worth different amounts in the
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
the 6-team league QB7 through QB32 are all sitting on waivers, so the real fallback is
a perfectly good starting quarterback, not the worst rostered one.

The visible symptom: Josh Allen ranks 10th on the 12-team board and **5th** on the
6-team; Brock Bowers 16th and **10th**; Trevor Lawrence 97th and **71st**. The math is
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
  costs roughly 4/14 in the 12-team league and 4/12 in the 6-team.
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
| 12-team | QB12 / RB29 / WR29 / TE14 | QB22 / RB51 / WR74 / TE21 |
| 6-team | QB6 / RB14 / WR14 / TE7 | QB8 / RB32 / WR38 / TE6 |

**CP7 — the sanity condition holds, and it is now a test rather than a
reading.** Quarterbacks inside the top 30 go 1 → 4 on the 12-team board and
4 → 1 on the 6-team one; Josh Allen goes 11th → 7th on the 12-team board and 7th →
15th on the 6-team. Trey McBride is the only tight end left in either top 30.
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
absence costs 4/12 in the 6-team league and 4/14 in the 12-team, read from
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

### Every projection was in 12-team scoring (Aug 6)

`features.py` scores every player under `league_config_12team.json` and nothing
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

### Phase 13 extended — posrank bake-off and the cut-feature audit (Aug 7)

Two questions, both answered NO CHANGE, and the second one turned up something about
the gate itself.

**Posrank: level, promotion, or both?** Jack's objection was sharp — why should a back
who was RB1 last year and is RB1 again be paid for it, when the model predicts *change*
from his own baseline?

| variant | RB | WR | TE |
|---|---|---|---|
| neither | +0.326 | **+0.371** | **+0.116** |
| level (shipped) | **+0.425** | +0.149 | +0.076 |
| promotion | +0.398 | +0.381 | +0.074 |
| both | +0.406 | +0.149 | +0.074 |

**The level wins at RB and strictly dominates**: better than promotion alone, and adding
promotion *to* it makes it worse. So they are not complements. The explanation is the
one the level always had — the baseline is a **three-year weighted average**, so a
current RB1 who spent 2023–24 as RB2 has a baseline that understates his role and should
beat it. A year-over-year promotion flag cannot see that multi-year drift. The objection
was reasonable and the answer is that the level already contains the promotion signal
plus something else. Shipped spec unchanged at every position.

**The audit: was anything cut wrongly?** Six features removed on in-sample p-values
alone, re-tested out of sample. **Nothing was.** Two long-open questions close:

- **`age_squared` is dead** (−0.034 / −0.001 / −0.000 forced in). Carried from Phase 10
  as an open question about a flat tail resting on 57 backs over 30. Answered.
- **`experience` is dead** (−0.032 at TE). Confirms `age` was the right replacement on
  evidence rather than on AIC.

`returning_oline_starters`, `coach_changed` and the `qb_changed` control were all inside
noise — and the control behaving correctly is what made the one hit worth chasing.

**The audit half-failed on its first run, in the now-familiar way.** It added each
candidate and let alpha decide. When alpha rejected a feature in every fold the model
never changed, the score came back `+0.0000`, and four of six candidates were never
actually tested while appearing to have "no effect." That made the audit circular — it
used alpha to decide whether to test a feature, when alpha is the mechanism under
suspicion. Fixed with a second column that forces every term in at `alpha=1.0`.

### The gate's three folds are a window, and it nearly cost us (Aug 7)

The forced audit produced exactly one hit: `continuity_score` at RB, **+0.0332**. Against
the 0.0121 spread of the other seventeen tests that is 2.7 standard deviations — but
across eighteen tests the family-wise chance of one noise draw that big is **≈0.05**, and
the churn band it cleared had no multiple-testing correction in it. So: probe it on all
nine folds instead of three.

    2017 -0.085   2018 +0.008   2019 +0.000   2020 -0.037   2021 +0.017
    2022 -0.031   2023 +0.032   2024 +0.041   2025 +0.027
    mean -0.0031   sd 0.0405

**The gate's three seasons are the three best folds in the set.** On all nine the feature
is dead. Not reinstated.

**The uncomfortable part is not about `continuity_score`.** `GATE_SEASONS` is always
2023/2024/2025, and the fold-to-fold spread here is **0.041 — larger than most of the
effect sizes currently shipping.** Every feature in the model passed a three-fold test
that this example shows can be window-dependent.

That does not mean the model is wrong; it means the gate is weaker evidence than it has
been treated as. Running it at nine folds is free information and is the obvious next
check. Whether the gate should PERMANENTLY use nine is a real trade — recent seasons
resemble 2026 most, and a feature that only works post-2022 might be a regime change
rather than noise — but that argument should be had after seeing the numbers, not
instead of seeing them.

### Nine-fold gate diagnostic — the veteran model holds (Aug 7)

Ran the gate across all nine seasons rather than the usual three. **The veteran model
passes everywhere**, and by a wider margin than the three-fold number suggested:

| position | 9-fold mean | positive folds |
|---|---|---|
| veteran RB | **+0.306** | 8/9 |
| veteran WR | **+0.259** | 9/9 |
| veteran TE | **+0.134** | 8/9 |

`pos_rank` at RB — the newest and least-tested feature, and the one that moved Tuten and
pushed RBs to 31 of the 12-team top 60 — is positive in 8 of 9 folds including 2017 and
2019. It is not a post-2022 artifact. That was the specific worry and it is answered.

**Rookie TE failed, and then the failure turned out to be the gate's arithmetic.**

    season  2025  2024  2023  2022  2021  2020  2019  2018  2017
    n_test    10     7     9    11     7     3     5    11     6
    gain    +.36  +.34  +.17  +.23  +.23  -.53  -.68  +.45  -.79

**All three negative folds are the three smallest.** The gate averaged per-fold RMSE
gains, which weights a 3-player fold exactly as heavily as an 11-player one — harmless
for veterans, whose folds run 71–137 rows, and decisive here:

    unweighted mean of folds   -0.0260   fails
    pooled by test-set size    +0.0981   passes

An RMSE computed on three tight ends is one unlucky player. The gate now **pools squared
errors** across folds instead of averaging RMSEs — `rmse² × n` is a fold's sum of squared
errors, so summing those and dividing by total n gives the RMSE over all held-out players
at once, which is what anyone reading the number assumes it already was.

**Stated plainly because it flipped a verdict:** the fix was not chosen because rookie TE
failed. Equal-weighting folds of wildly unequal size is wrong whichever direction it
lands, and it would have been just as wrong if it had let something through. It only
became visible because the nine-fold run created folds small enough for it to matter.

Also fixed: the diagnostic printed `Wrote data/holdout_gate.json` while correctly not
writing it. The file was untouched, but a message that says it wrote something it did not
is the kind of thing that gets believed later.

**Gate stays at three folds by default.** Recent seasons resemble 2026 most, and
`--gate-seasons all` now exists for anything marginal. The three-fold gate is no longer
being treated as stronger evidence than it is.

**Both gates pass after the fix.** Nine folds, pooled over every held-out player:

| model | pooled gain | held-out players | unweighted mean |
|---|---|---|---|
| veteran RB | **+0.318** | 711 | +0.306 |
| veteran WR | **+0.260** | 1145 | +0.259 |
| veteran TE | **+0.142** | 616 | +0.134 |
| rookie TE | **+0.130** | 69 | **−0.026** |

The last column is the point. For the three veteran positions, pooling and averaging
agree to within 0.01 — fold sizes are near-equal so the choice never mattered. For rookie
TE, with folds of 3 to 11, it is the difference between shipping and cutting.

**The model is now validated on 2,541 held-out player-seasons across nine years**, which
is a materially stronger claim than the three-fold gate supported this morning.

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
  `2026_6Team_Board_v9.xlsx` and `2026_12Team_Board_v9.xlsx`. Version
  tracks the *model*, bumping when weights or features change — so a pure data refresh
  stays v9. Build date and git short hash go in a metadata sheet, which is what
  actually distinguishes two rebuilds of the same version. (v8 staying static across
  three rebuilds is the problem this fixes.)
- **CP5** — Sanity-read the top 60 of each board by hand. Read them side by side: the
  6-team/12-week board should visibly diverge from the 12-team/17-week one, and if it
  doesn't, the league-awareness work in Phase 11 didn't land.

### Phase 14 — Draft day prep (Aug 21–22)

- Refresh ADP (it moves through August; the current pull is from late July).
- Re-run the fast keeper filter as opposing keepers are announced — 12-team league only.
  The 6-team league is a straight redraft, so keeper columns are suppressed on that board.
- **ADP caveat for the 6-team league.** The FFC pull is `teams=12`. Six-team ADP is a
  different draft entirely — 96 players go instead of 192, and positional runs behave
  nothing alike. Either pull `teams=8` as the closest available proxy and label the
  column honestly, or drop the ADP column from the 6-team board rather than show a
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

**Second league — resolved Aug 3.** `league_config_6team.json` created: 6 teams, no
keepers, weeks 1–12 regular season, playoffs 13–14, scoring and roster slots identical
to the 12-team league. `build_board.py` already derives replacement level from
`num_teams`, so VOR recomputes correctly once it can take a `--config` flag (currently
`CONFIG_PATH` is a module constant). Two consequences beyond the config file:

- **Replacement level moves a long way.** 6 teams × 1 QB means the 7th-best QB is
  freely available; positional scarcity nearly vanishes and best-player-available
  logic dominates. The tier breaks and `draft_target` cushions were tuned on 12 teams
  and should be re-read, not assumed to transfer.
- **12-week season changes injury math.** See Phase 11.

**Resolved.** Both configs now carry `regular_season_weeks`, `playoff_weeks`, and
`fantasy_season_length`: the 12-team league 1–14 with playoffs 15–17 (length 14), the
6-team 1–12 with playoffs 13–14 (length 12).

**Versioning honesty note.** `MODEL_VERSION` went 8 → 9 for a change that touched the
filename convention and the ADP columns, not the model. The 12-team v9 board is
rank-for-rank and VOR-for-VOR identical to v8 across all 1088 players. By the rule
stated below — bump only when the model changes — it should have stayed v8. Kept at 9
because two differently-named files both calling themselves v8 would be worse, but the
rule is only worth having if exceptions are visible, so this one is written down.

**Board versioning — settled Aug 3.** One file per league, overwritten in place:
`2026_12Team_Board_v9.xlsx` and `2026_6Team_Board_v9.xlsx`. The version
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
    it matched the v9 12-team board on all 1088 players, rank and VOR.
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

---

## Phase 13.5 — playing time, and the QB59 stress test (Aug 12)

Two pieces of work, one of them a detour that turned out to matter more than the thing
it was detouring from. Both were prompted by the same question — *is the number of
running backs near the top of the 32-team board statistically sound?* — and the answer
to that question is **yes, and it is the least fragile thing on the board.**

### The RB question, answered three ways

**1. Does the historical RB curve actually fall off?** Yes, and consistently. Ranking
every player by *actual* season PPG within his own position and season, over
2017–2025 (`backtest_features.csv`, players with ≥8 games):

| decline in actual PPG | RB | WR | TE |
|---|---|---|---|
| rank 12 → 36 | **6.30** | 4.42 | 5.08 |
| rank 18 → 48 | **6.83** | 5.18 | 4.60 |
| rank 24 → 60 | **7.58** | 5.69 | 4.66 |

RB declines faster than WR **in nine seasons out of nine** on the 12→36 measure, and the
gap is widening rather than shrinking — +1.30 in 2017, +3.63 in 2024, +3.23 in 2025.
This is not a one-cycle artifact and it is not a modelling choice; it is what the
position did.

**2. Is WR depth real?** Yes. Mean actual PPG at WR48 is 9.38 against 6.37 at RB48. The
receiver curve is genuinely flatter, which is the other half of the same finding and the
reason the board is comfortable letting receivers wait.

**3. Is the count an artifact of replacement level?** No, and this is the part worth
recording. The 32-team board puts **20 RB, 20 WR, 18 QB, 2 TE** in its top 60. Moving RB
replacement ten ranks in either direction changes that mix by **zero players**; moving it
twenty ranks changes it by **one**. RB replacement sits on a dead-flat plateau — 34
players tied at 4.63 Adj PPG, seven of them inside the top 92 — so the RB VOR level is
insensitive to exactly where the bar is put.

That plateau is worth naming as a separate caveat even though it does not affect the
count: **RB VOR magnitudes between roughly RB40 and RB90 are not meaningfully ordered.**
They are differences against a floor made of identically-valued unknowns. The top-60
conclusion survives it; a "who is the better RB63" conclusion would not.

### The weights check

Independent verification, not a re-run of the project's own scripts:

- **Shipped weights are applied exactly.** Recomputed `Sit Adj` from
  `data/situational_weights.json` for all 316 modelled veterans on the 32-team board,
  applying intercepts and centres by hand. **Max |difference| = 0** (float noise, ~2e-15).
  The fit and the board agree.
- `expected_drafted` sums to 352 = 32 × 11, and `compute_replacement_ranks` raises if it
  does not.
- No sign flips and no magnitude instability in leave-one-season-out for any
  position/feature.
- Every coefficient p < 0.05 except `RB/trend_missing` (0.051) and `TE/trend_missing`
  (0.68). Both are missingness indicators paired to an imputed feature; keeping them is
  correct — dropping the indicator while keeping the imputed variable is what would bias.
- Holdout gate passed, all feature gains positive except `veteran/TE/trend_missing` at
  −0.0026, which is noise.

**Nothing in the weights needs to change before a draft.**

### The QB59 stress test — the real fragility

The board's most load-bearing assumption is not RB. It is `expected_drafted.QB = 59`,
recorded from **one** 32-team mock in which **31 of 32 teams autodrafted**. Sweeping QB
from 45 to 70, redistributing the difference across RB/WR/TE by observed share so the
total stays at 352:

| QB drafted | QB in top 60 | RB in top 60 | QB in round 1 | QB1 overall |
|---|---|---|---|---|
| 45 | 7 | 25 | 1 | 8 |
| **49** | **8** | 24 | 1 | 8 |
| **50** | **13** | 24 | 3 | 8 |
| 55 | 18 | 20 | 6 | 6 |
| **59 (shipped)** | **18** | **20** | **7** | **5** |
| 65 | 19 | 20 | 12 | 3 |
| 70 | 21 | 20 | 15 | 1 |

**The board is bimodal and the cliff is between QB49 and QB50.** Below it, eight
quarterbacks make the top 60; above QB55, eighteen do. One mock draft's count is being
asked to locate the board on one side of a step change.

Two things follow, and they point in opposite directions:

- **Reassuring:** the shipped value sits on the *stable* side. QB55, 59 and 62 produce
  effectively the same board. The estimate would have to be wrong by ten in one specific
  direction to matter.
- **Not reassuring:** the sampling error on a count of 59 out of 352 picks is roughly
  ±7 at one standard error, and positional runs make draft picks positively correlated
  rather than independent, so the true band is wider than that. **It straddles the
  cliff.** And the model's own ADP-tail extrapolation — the thing the mock replaced —
  said 47, which is on the far side.

**Where the volatility lives.** Non-QB players move at most 14–26 ranks across the whole
sweep. Elite quarterbacks barely move either: Josh Allen is top-8 in every scenario. The
entire instability is concentrated in the **QB6–QB16 tier**, which swings 46–83 ranks:
Drake Maye 19↔65, Joe Burrow 20↔70, Justin Herbert 25↔81, Kyler Murray 36↔101.

**Draft-day rule, and it is a live read rather than a pre-commitment.** Draft off the
*worst-case* rank in `QB59_stress_test.xlsx` and you cannot be wrong by more than the
swing. **46 players are top-60 under every scenario from QB45 to QB70 — 20 RB, 17 WR,
7 QB, 2 TE.** Note that RB is 20 in the robust list and 20 on the shipped board: the
running back count is the one number the QB assumption cannot touch. Then watch rounds
1–2. If quarterbacks go fast, QB59 is right and the middle-QB tier is a genuine value
band. If quarterbacks are still sitting at pick 60, replacement is shallower than
modelled and that tier is forty ranks worse than the board says.

Artifacts: `QB59_stress_test.xlsx` (robust board sorted by worst-case rank, plus the full
scenario grid), `boards_v13_frozen/` (v13 boards and weights preserved so the 32-team
draft can run off them regardless of what follows).

### Phase 13.5a — the playing-time model

Phase 13's closing note asked the right question and deferred it: *"what is
P(earns a role) for a rookie versus a veteran, and does the board price the
difference."* It does not. Here is the size of the hole, measured against files already
in `data/`:

| | rows | P(≥8 games) |
|---|---|---|
| `rookie_backtest_features.csv` | 415 | **100.0%** |
| `backtest_features.csv` | 4177 | **66.4%** |

**Zero percent of the rookie training sample failed the availability test. Thirty-four
percent of the veteran sample did.** The two baselines the board sets against each other
are not the same kind of number, and the whole difference runs in the rookies' favour.

The chain that produces this is three filters deep, and the last one is documented while
the first two are not:

    drafted                                    load_rookie_class()
      → appears on a week-1 roster snapshot    inner join to team
      → recorded a stat line that season       inner join to outcomes  (600 survive)
      → played ≥ 8 games                       MIN_GAMES filter        (415 survive)

`rookie_backtest.py`'s own comment says the surviving sample "is a sample of rookies who
were given a chance — and the intercept is a statement about THEM, not about all
rookies." Correct, and left unquantified. `src/playing_time.py` quantifies it.

**What the module does.** Rebuilds the universe with both inner joins loosened, so a
rookie who never dressed is a row with `actual_games_played = 0` rather than an absence.
Then three outputs:

1. **Availability rates** — P(snap), P(≥8 games), E[games] per position × round bucket,
   leave-one-class-out, with cells under n=15 falling back to the position rate and
   saying so.
2. **A realisation ratio** — E[season points | drafted] ÷ (season length × E[PPG | ≥8
   games]). 1.00 would mean no hole; every point below is the board's rookie premium,
   priced.
3. **A fitted `expected_games`** on draft capital, feeding
   `expected_games_missed` — the column PUP/NFI already populates — so availability turns
   into Exp Pts in exactly one place.

**Two design commitments, both deliberate.**

*Availability does not touch the rate.* `compute_expected_points` states the rule — PPG
is a rate, a player who misses games is not worse per game, and folding availability into
the rate corrupts the quantity everything is fitted to predict. Not relaxed here. The
consequence is honest and worth stating: **this alone will not move a rookie's rank.**
It moves Exp Pts.

*The feature set is short on purpose.* `pick` and `position` are the only things every
drafted rookie has. Every richer feature in `rookie_backtest_features.csv` — landing
spot, position competition, depth chart, O-line — exists **only for players who made a
roster**. Conditioning availability on them would re-introduce, one level down, exactly
the survivorship this module removes.

**The rank-moving question is separate, and it is a diagnostic first.**
`--sweep` rebuilds the leave-one-class-out cohort baseline at MIN_GAMES ∈ {0,1,2,4,6,8,
10,12}. A baseline that slides smoothly across that grid means 8 is as defensible as 6
and the population is simply changing. A baseline that **steps** means the threshold is
choosing the projection. Whichever it is, **nothing ships on the strength of that table** —
it is evidence for the holdout, per Phase 13 CP2.

**Status: written, not yet run.** The module needs `nflreadpy` and network access, so it
runs on Jack's machine, not in the analysis sandbox. Every number in this section that is
*not* about `playing_time.py`'s own output was computed directly and is checkable.

### Phase 13.5b — combine data

Not started. It hooks into `fit_expected_games`, and that is not an arbitrary choice:
combine measurables exist for every drafted player **regardless of whether he ever
played**, which makes them one of the few feature families admissible against an
availability target. That property is why the two halves of Phase 13.5 were scoped
together in the first place.

### Carried forward

- **`expected_drafted.QB = 59` rests on one autodrafted mock and the board is bimodal
  around it.** A second mock — ideally a human room — is the single highest-value data
  collection available to this project. One more draft's counts would halve the
  uncertainty on the number the whole QB tier depends on.
- **RB replacement sits on a 34-player plateau.** Harmless for the top-60 conclusion,
  fatal for any conclusion about a specific back between RB40 and RB90.

### Phase 13.5a — FIRST RUN (Aug 12). The sweep steps, and the module had a bug.

**The universe is bigger than the plan assumed.** 719 drafted offensive rookies
2017–2025: 600 took a snap (83.4%), **415 reached 8 games (57.7%)**, 119 never played
(16.6%). So the shipped rookie projection is estimated on 58% of the population and
applied to 100% of it.

*Lower bound, not a measurement.* The universe is drafted players with a non-null
nflverse `gsis_id`. A drafted player who never signed may have no id at all. 719 over
nine classes is ~80/class against ~90 expected, so roughly ten players a year are still
missing — and every one of them is a true zero. The 16.6% is the floor.

#### The sweep steps, and it steps exactly where the theory predicts

Correlation between a cell's P(≥8 games) and the size of its baseline step from
threshold 0 to 8: **−0.81** across all twelve cells.

*(Recomputed after the MIN_CELL_N fix: **−0.88**. The −0.81 above was measured on the first
run, whose RB and TE Round 1 cells were contaminated by the position-mean fallback. The
chart script recomputes from disk and is the authority; this line is left in place because
the correction is the point.)*

| cell | P(≥8gm) | baseline 0 → 8 | step | realisation |
|---|---|---|---|---|
| QB Day 3 | 7.4% | 7.97 → 15.64 | **+96%** | **0.07** |
| QB Day 2 | 33.3% | 9.31 → 12.34 | +33% | 0.30 |
| TE Day 3 | 40.5% | 3.97 → 5.00 | +26% | 0.34 |
| RB Day 3 | 55.6% | 4.28 → 5.30 | +24% | 0.47 |
| WR Day 3 | 46.6% | 3.48 → 4.17 | +20% | 0.41 |
| WR Round 1 | 89.7% | 10.15 → 10.83 | +7% | 0.82 |
| RB Round 1 | ~high | 14.92 → 14.92 | **0%** | — |

Round 1 does not move at all — nearly every first-rounder plays, so the filter has no one
to remove. **Day-3 quarterback nearly doubles**, because 92.6% of them are removed and
the survivors are Brock Purdy. The board currently projects every Day-3 rookie QB at
15.64 PPG. Its realisation ratio is **0.07** — it is paying roughly fourteen times what
that player is worth.

**The argument this settles.** `MIN_GAMES = 8` on the veteran side is applied to the
*baseline* — prior seasons, to stabilise a rate before predicting anything. On the rookie
side the identical constant is applied to the *outcome*. Those are not the same operation
with the same justification. The second is conditioning on the dependent variable, and it
has been hiding behind the fact that both are spelled `8`.

#### The bug: the first run got Round 1 backwards

`MIN_CELL_N` was 15 with a fallback to the position-wide rate. Round-1 RB (~1.5/class)
and Round-1 TE (~1.2/class) both fell under the bar after leave-one-class-out, and both
silently borrowed their whole position:

    RB Round 1   P(>=8gm) 62.6%   E[games]  9.41   source=position
    RB Day 2     P(>=8gm) 74.4%   E[games] 11.36   source=cell

**A first-round running back rated less available than a second-rounder**, and the same
inversion at TE (53.1% against Day 2's 65.0%). The realisation table printed the tell in
plain sight — `RB Round 1  n=168`, larger than the entire Day 3 cell — and the sweep had
been saying it all along: RB Round 1's baseline is 14.92 at *every* threshold from 0 to
12, which can only happen if essentially every first-round back plays twelve games.

This is the same failure class as the three `compute_replacement_ranks` bugs of Aug 6:
**a number computed on a different population than its label claims.** Three fixes:

1. `MIN_CELL_N` 15 → 8. These are means; a mean over twelve first-round backs is thin, but
   a mean over 168 backs of all rounds is a mean of the wrong thing.
2. Availability falls back to the **fitted pick model**, which is monotone in draft
   position by construction and therefore *cannot* produce "round 1 is worse than round
   2." Realisation ratios substitute nothing at all — a thin cell reports its own number,
   its own n, and `thin: true`.
3. **A monotonicity assertion**, with a 10-point tolerance so sampling noise in a
   twelve-player cell warns rather than halts. Checked against the first run's numbers it
   raises on RB (+11.5%) and TE (+10.4%) and passes QB (−20.6%) and WR (−9.2%) — it
   catches exactly the two contaminated cells and neither clean one.

Also dropped threshold 1 from the sweep: it printed identical to 0 in all twelve cells,
necessarily, because the sweep operates on snap-takers.

#### Does the decomposition close?

Keeping PPG a rate and putting availability in `expected_games` is only legitimate if the
product lands near the truth. It cannot land exactly — E[games × ppg] ≠ E[games] ×
E[ppg | played] whenever the two correlate, and they obviously do. Measured over all
twelve cells on a 14-game season, the product recovers a **median 0.89** of actual rookie
production, range 0.82–1.03. Biased low by ~11%, one-directional, roughly constant across
cells. Against a status quo that overstates Day-3 QB by 14×, that is an acceptable trade
in the safe direction, and it is now recomputed every run rather than remembered.

#### What this actually does to the 2026 boards — very little, and that is the finding

| | rookies in draftable range | Exp Pts multiplier (median) | **rank-moving rate multiplier** |
|---|---|---|---|
| 12-team (top 192) | 15 | 0.78 | **0.94** (min 0.81) |
| 32-team (top 352) | 30 | 0.69 | **0.94** (min 0.75) |

**No rookie inside either board's top 80 moves more than 6% on rate.** Every rookie in
the 32-team top 120 is a first-rounder, and first-round rates barely move (RB 14.92 →
14.92, TE 9.10 → 9.10, WR 10.83 → 10.15). The corrections concentrate at rank 164+ and
184+ — Carson Beck 0.75, Jonah Coleman 0.81, Cyrus Allen 0.83.

**So the availability hole is real, large, and almost entirely outside the range where
this year's decisions get made.** The 32-team board happens to contain no Day-3 rookie
quarterbacks, which is luck rather than design — the 0.07 cell is a live landmine in any
year where one is draftable.

The Exp Pts column is a different story: first-round rookies should be marked down
15–22% there, and that column is currently telling Jeremiyah Love and Jadarian Price they
will play all fourteen games.

**Open: six rookies on the 32-team board have no round or pick** (undrafted, floored to
round 7 by `rookies.py`). They inherit Day-3 rates, which carry the largest corrections,
and they were not in this analysis because they have no draft capital to fit on.

#### Status

Module fixed but **not re-run**, and nothing is wired into a board. Next: re-run
`python -m src.playing_time` and confirm the monotonicity assertion passes, then
`python -m src.holdout --gate`. The rate change (MIN_GAMES 8 → 0) is the only piece that
can move a rank and it is the only piece that must clear the gate; the `expected_games`
change touches Exp Pts alone and is safe by construction.

### Phase 13.5a — SECOND RUN (Aug 12). Clean, and the fix moved the right numbers.

The monotonicity assertion passed silently and every cell now reads `source=cell` — with
`MIN_CELL_N` at 8, Round-1 RB (n=12) and Round-1 TE (n=10) clear the bar on their own and
the fitted-model fallback was never needed. What it corrected:

| cell | first run (contaminated) | second run (own cell) |
|---|---|---|
| RB Round 1 | P(≥8gm) 62.6%, E[gm] 9.41, **realisation 0.54** | P(≥8gm) **92.3%**, E[gm] **13.23**, **realisation 0.79** |
| TE Round 1 | P(≥8gm) 53.1%, E[gm] 8.07, **realisation 0.46** | P(≥8gm) **100.0%**, E[gm] **14.91**, **realisation 0.91** |

Availability is now monotone in draft capital at all four positions — QB 68.9/32.7/12.1,
RB 77.8/66.8/49.2, TE 87.7/57.8/36.6, WR 81.3/72.1/40.3. **First-round tight ends played
in 100% of cases and cleared 8 games in 100% of cases**, n=10 across nine classes.
Realisation 0.91 says the board is barely overpaying for them at all.

The decomposition check independently reproduced the hand-computed **0.89**, this time
over 108 cell-seasons rather than 12 hand-worked cells.

### The Aug 12 `holdout --gate` run passed, and it did not test any of this

`holdout.MODELS` has two entries, `veteran` and `rookie`. The gate ran the veteran model
(RB/WR/TE) and the rookie TE model, and passed both. **Neither knows `playing_time.py`
exists.** That run is evidence v13 still holds up. It is not evidence about Phase 13.5,
and reading it as a green light is the exact failure mode `run_gate`'s docstring now warns
about.

Phase 13.5 gets **its own gate**, in `playing_time.py`, writing to
`data/playing_time_gate.json` — deliberately not `holdout_gate.json`, because two gates
sharing a file means whichever ran last silently claims to be the state of both.

It cannot reuse `holdout.py` and the reason is structural: that gate asks "does this
FEATURE earn its slot," by ablation against a fitted model. Phase 13.5 asks "which
POPULATION should the baseline be estimated on, and should it be multiplied by an
availability term." There is no feature to ablate. Forcing it through the ablation
machinery would mean inventing a feature that is really a population choice, which is how
you get a passing gate that tested nothing.

**The test.** Leave one rookie class out; predict every drafted rookie in it — the full
population, zeros included, because that is the population the board applies these numbers
to. Score on **season total points**, not PPG: total points is what a roster spot returns,
and a player with no games has no PPG to score against but unambiguously scored zero.

Four predictors, a 2×2, so the rate change and the availability change each justify
themselves rather than shipping as a bundle:

    A  rate@8 x season length        <- what ships today
    B  rate@8 x fitted availability
    C  rate@0 x season length
    D  rate@0 x fitted availability  <- the proposal

**Rule, fixed before the numbers are seen:** D must beat A or the phase ships nothing.
B and C are each compared to A to say which half did the work; a change that does not beat
A on its own does not ship on its own.

    python -m src.playing_time --gate

### The eight "HURTS out of sample" flags are all noise, and the display should say so

The gate output flagged eight feature-folds as hurting. Every one is inside a quarter of
one standard error of zero. SE on a single fold's RMSE is `RMSE / sqrt(2n)`, which for
these fold sizes is **0.18 to 0.31**:

| flag | delta | in SE |
|---|---|---|
| TE workload_share 2025 | −0.0517 | **−0.25** |
| RB workload_share 2025 | −0.0438 | −0.17 |
| TE age 2024 | −0.0263 | −0.13 |
| WR recent_major_injury 2025 | −0.0191 | −0.11 |
| TE trend_missing 2023 | −0.0105 | −0.05 |
| TE position_competition_ppg 2023 | −0.0099 | −0.05 |
| RB pos_rank 2025 | −0.0034 | −0.01 |
| RB trend_missing 2025 | −0.0003 | −0.00 |

The flag fires on the **sign** of a quantity whose noise floor is ~0.2, and sign is the
one thing that cannot be read at that resolution. This is the same error the plan already
recorded for `MEANINGFUL_GAIN = 0.02` on Aug 7 — "the threshold was borrowed... −0.057 is
a quarter of one standard error" — reappearing in the ablation display rather than in a
threshold. The gate's own pooled rule is right and the per-fold arrow is what should
change: print the SE beside the delta, or suppress the arrow below 1 SE.

**One thing in that table is not noise.** `RB workload_share` ranges −0.044 (2025) to
+0.355 (2023), a spread of **1.46 SE** — the largest in the set, and it is the RB model's
largest coefficient (−6.69). `WR age` spreads 1.08 SE, `RB pos_rank` 0.73. Set against
the plan's own Aug 7 finding that the gate's three seasons were *the three best folds* for
`continuity_score`, `workload_share` at RB is the feature most worth running
`--gate-seasons all` against before the next refit. It may be a real post-2022 change in
how backs are used; it may be 2023 doing all the work. Both are consistent with three
folds and only nine can tell them apart.

### Carried forward, updated

- Run `python -m src.playing_time --gate`. Nothing ships until D beats A.
- Run `python -m src.holdout --gate --gate-seasons all` on RB `workload_share` before the
  next refit.
- The per-fold `HURTS` arrow needs an SE beside it or a 1-SE suppression threshold.

### Phase 13.5 GATE (Aug 12) — passed, and it refuted half the proposal

719 held-out drafted rookies, leave-one-class-out, scored on season total points against
the full drafted population:

| predictor | RMSE | vs A | mean bias |
|---|---|---|---|
| **A** rate@8 × full season *(shipped today)* | 102.01 | — | **+62.21** |
| **B** rate@8 × availability | 59.41 | +42.60 | **+1.30** |
| **C** rate@0 × full season | 76.43 | +25.58 | +39.72 |
| **D** rate@0 × availability *(the proposal)* | 59.33 | +42.68 | −6.29 |

D beats A, so the gate passes. **But D beats B by 0.08 RMSE, which is 0.05 of one
standard error** (1 SE = 1.57 at n=719), while moving the mean bias from +1.30 — 0.6 SE
from zero, i.e. unbiased — to −6.29, which is 2.8 SE from zero and biased low.

    availability alone captures        99.8% of the total available improvement
    the rate change on top of it       0.2%

**The stated rule was incomplete and the result showed where.** It asked whether each half
beat A. It did not ask whether the second half added anything *on top of* the first, and
that is the comparison that decided the phase. The incremental test is now part of
`run_gate` so nobody has to rediscover it.

**Why, and the project predicted it in writing.** From `compute_expected_points`, written
months before this module existed: folding availability into the rate "would silently
double-count the moment a future phase models availability directly." That is precisely
what rate@0 does. Lowering the games threshold pulls the rate down *because the players it
admits played fewer games* — it is an availability correction wearing a rate's clothing.
Multiply it by an explicit availability term and the same correction is applied twice. The
−6.29 is that double-count, measured.

**The shipping decision is B, not D.**

- `rookie_backtest.MIN_GAMES` **stays at 8** and is not touched.
- No cohort baseline changes. `adjusted_fantasy_points_per_game` is untouched.
- **No player's rank moves anywhere on any board.** Exp Pts only.
- The MIN_GAMES sweep remains a correct diagnostic of a real selection effect. It just
  turns out that effect is fully absorbed by the availability term and does not want a
  second correction.

The design commitment made at the top of `playing_time.py` — availability goes in
`expected_games`, never in the rate — held. The one place it was doubted mid-phase is the
one place the gate said no.

**In plain units: the board has been overpaying the average drafted rookie by 62 points
per season. B removes essentially all of it and leaves a residual bias of +1.3.**

### Wired in

`build_board.prepare_board_frame` now calls `expected_games_for_rookies` between
`apply_injury_overrides` and `compute_expected_points` — after the column exists, before
its only consumer reads it. A rookie who is also on PUP takes the larger of the two
absences, never their sum, because the fitted number already averages over rookies who got
hurt.

Two implementation notes worth keeping:

- **Undrafted rookies get a pick, not a position mean.** Six rookies on the 32-team board
  have no round or pick. The first draft fell back to a position-wide availability, which
  is the same substitution that caused the round-1 inversion — a UDFA is not an average
  rookie, he is a worse-than-day-3 rookie. `rookies.py` already floors UDFAs to round 7,
  so this floors them to pick 245 and runs them through the same fitted curve. One rule for
  everybody. It extrapolates slightly past the fit's support, which is acknowledged: it
  touches Exp Pts for six undraftable players and cannot move a rank.
- **`src.rookie_backtest` is imported lazily.** It pulls `nflreadpy`, a network-backed
  data client, and `build_board` needs only two functions from this module that use `json`,
  `polars` and arithmetic. A module-level import would put a data client into the import
  graph of every board build and every `sanity_top_n` run — so a broken nflreadpy could
  stop a draft board rendering over a dependency it never uses. The board-side surface of
  `playing_time.py` is dependency-free by construction; the analysis side pays where it uses.

### Where Phase 13.5 stands

- 13.5a **done and shipped.** Rebuild boards to pick it up; ranks will be identical to v13
  and Exp Pts will drop for rookies. That identity is the check — if a rank moves, something
  is wired wrong.
- 13.5b (combine data) not started. It hooks into `fit_expected_games`, and the gate result
  raises its value rather than lowering it: availability is now the *only* live rookie
  lever, so anything that predicts availability better is the whole game.
- Still open: `--gate-seasons all` on RB `workload_share`; the per-fold `HURTS` arrow needs
  an SE beside it; a second 32-team mock for `expected_drafted.QB`.

### v14 build (Aug 12) — passed the rank check, and the rank check was not the interesting output

All three boards: **0 of 1088 ranks moved, VOR and Adj PPG unchanged**, 231 players'
Exp Gm and Exp Pts changed. Phase 13.5 is wired correctly and does exactly what it claimed.

Two bugs surfaced anyway, and neither was the thing being tested.

**1. `pick` is a string, and the build died on the first rookie.** `player_features.csv`
is a CSV, so every column round-trips as text — which is why `prepare_board_frame` already
has a loop casting `has_adp` / `is_rookie` / `recent_major_injury` back to booleans a few
lines above the new hook. `pick` needed the same and did not get it:
`unsupported operand type(s) for -: 'str' and 'float'`.

The fix removed the mechanism as well as the symptom. The original used
`pl.struct(...).map_elements()` — a Python UDF, which accepts whatever the column holds and
finds out at runtime. It is now a native polars expression, so a bad dtype fails at cast
time with the column named. `--selftest` was added at the same time: a synthetic frame,
five seconds, no nflverse, built to be nasty in the ways the real frame is.

Worth stating plainly: **every expensive check had passed.** The gate validated the
availability arithmetic across 719 held-out rookies and nine classes. What broke the build
was a type conversion, and nothing cheap stood between the two.

**2. The linear probability model goes negative inside its own training support.**

The board printed Exp Gm 0 and Exp Pts 0 for eleven undrafted rookie quarterbacks — the
value it reserves for OUT_SEASON. Cause:

| position | slope /pick | share = 0 at pick | share @245 |
|---|---|---|---|
| **QB** | −0.002785 | **230** | **0.000** |
| TE | −0.003111 | 278 | 0.104 |
| WR | −0.002479 | 349 | 0.257 |
| RB | −0.002105 | 402 | 0.331 |

**The QB curve crosses zero at pick 230, which is inside the draft.** `UDFA_EFFECTIVE_PICK
= 245` therefore produced a negative fitted share, clipped to zero. "This man will not play
a single snap" is an assertion the data does not support — the worst QB cell actually
*measured* is Day 3 at 12.1%.

And the scale was wrong in my head: **152 of 232 rookies have no pick**, not the six I
counted when I only looked at the top of the 32-team board. A tail case that is two thirds
of the population is not a tail case.

**Fix: stop extrapolating, start inheriting.** A UDFA is floored to round 7 for his cohort
baseline in `rookies.py`, so he now inherits the round-7 (Day 3) *observed availability*
too — one convention in both places, and a measurement rather than a line extended past
where it means anything. Separately, every fitted share is floored at its position's worst
observed cell, so a genuinely drafted pick-250 QB cannot fit below zero either.

**This is a patch over a misspecification, and it should be named as one.** The right fix
is to fit availability on a logit scale, where the functional form cannot leave [0, 1].
That belongs in **13.5b** alongside the combine features — not in a hotfix hours before a
draft. Carried forward.

**3. The comparator lied about rookies.** It checked `Rook` for a value starting with "Y";
the board writes `"R"`. So it reported "0 of them rookies" against a build that had printed
"232 rookies marked down on Exp Pts" two lines earlier. Any non-empty marker counts now.
Small, but a verification tool that misreports is worse than no verification tool, and this
one was two lines from a contradiction it could have caught itself.

**New invariant in `--selftest`:** no rookie may ever be assigned zero expected games. It
is checked separately from the expected-value table on purpose — someone "fixing" a
failing test by editing the expectations would not catch it, and Exp Gm 0 is a claim no
availability model is entitled to make about a healthy player.

**v14 verified clean (Aug 12, build #2).** Selftest passes all eight rows. All three boards:
0 of 1088 ranks moved, VOR and Adj PPG unchanged, 231 rookies' Exp Gm and Exp Pts changed.
Undrafted quarterbacks now read Exp Gm 1.69 and Exp Pts 12.5 rather than 0 and 0.

**The 232-vs-231 gap resolved, and it is not an error.** The build reported 232 rookies
marked down; the comparator found 231 rows changed. The difference is **Chris Brazzell II**,
a rookie WR (round 3, pick 83) marked OUT_SEASON. His `expected_games_missed` is non-zero,
so he counted in the build's tally — but `compute_expected_points` zeroes an out-for-season
player regardless of availability, so his Exp Gm was already 0 in v13 and stayed 0.

The build's count now excludes out-for-season rookies. These two numbers exist to
cross-check each other, and a count that cannot be reconciled against the comparator is
worse than no count. They should agree exactly from here.

### v14 shipped (Aug 12) — data refresh read, and the QB cliff was not touched

**The headline for the imminent draft: the top-60 position mix is IDENTICAL.**
QB 18, RB 20, WR 20, TE 2, before and after six days of ADP movement. The board did not
move off the stable side of the QB49/QB50 cliff, and `expected_drafted` still reads the
observed mock counts. The RB conclusion holds unchanged.

**How much really moved.** The raw count says 878 of 1082, which is nearly meaningless:

| range | moved at all | moved 10+ | moved 25+ |
|---|---|---|---|
| top 120 | — | **1** | — |
| top 352 (draftable) | 218 | 10 | 10 |

208 of the 218 draftable-range moves are ±3 place shuffles caused by 5 players entering
and 6 leaving the pool. **One player inside the top 120 moved by ten or more.**

**And five of the ten big movers are not revaluations at all.**
`compute_draft_targets` sorts by `(out_for_season, has_adp, vor, adp, player_name)`, so
`has_adp` is a hard gate **above** VOR — gaining ADP coverage vaults a player over the
entire no-ADP block regardless of what the model thinks of him.

| player | move | dPPG | cause |
|---|---|---|---|
| Stefon Diggs | 188 → 91 | −0.24 | **gained ADP** |
| Brian Robinson | 245 → 173 | 0.00 | **gained ADP** |
| De'Zhaun Stribling | 226 → 163 | 0.00 | **gained ADP** |
| Chimere Dike | 214 → 160 | 0.00 | **gained ADP** |
| Oronde Gadsden II | 146 → 207 | 0.00 | **lost ADP** |

Four of these have a projection change of *exactly zero*. The model did not revalue anyone;
the FFC feed added three names and dropped one. Coverage went 185 → 188.

**The actionable one is Gadsden.** The model still rates him at 8.64 PPG — which is where
rank 146 came from — and he fell 61 places purely because the market stopped listing him.
If he is going to be drafted in a real room, the board is now under-ranking him by design.
That design is defensible ("if the market has no opinion, the model's opinion alone isn't
worth a pick") but it is a rule about markets, not about football, and it should be
overridden by hand when you know better.

**Comparator improvements, both forced by this run.** `--focus` (default 200) confines the
rank report to the draftable range; the first version led with Owen Wright +481, a rank
589 → 1070 move nobody will ever act on, while burying the fact that only one top-120
player moved. And rank moves now carry `dPPG` and a GAINED/LOST ADP marker, so a
sort-order effect can never again be read as a revaluation.

**Also verified:** rookie Adj PPG is unchanged for all 232 (cohort baselines don't move on
a veteran data refresh), 225 rookies took the availability markdown, and undrafted QBs read
Exp Gm 1.69 rather than 0.

---

## Phase 13.5b — the logit refit (Aug 12)

Sequenced first, ahead of the combine features, because adding predictors to a
misspecified functional form means the same failure persists and no feature test is
trustworthy against a broken baseline.

### The evidence

Leave-one-class-out predictive **binomial deviance**, 2017–2025, with a 500-resample
bootstrap over players:

| pos | linear | logit | gain | 95% CI | P(logit better) |
|---|---|---|---|---|---|
| **QB** | 1195.6 | 707.6 | **+488.0** | [+63, +823] | **99.0%** |
| RB | 1657.5 | 1653.9 | +3.5 | [−6, +20] | 74.0% |
| WR | 2123.9 | 2122.9 | +1.0 | [−19, +27] | 51.8% |
| **TE** | 864.7 | 887.3 | **−22.7** | [−43, +3] | **3.2%** |

Read honestly: **the logit repairs quarterback, is a coin flip at RB and WR, and is
genuinely worse at tight end.** TE's −22.7 is not noise — the bootstrap puts it at 96.8%
confidence that the linear form fits TE better.

The linear predictions were clipped to [1e-6, 1−1e-6] before scoring. Unclipped, its
negative fitted values give infinite deviance. The clip is a courtesy to the incumbent and
the QB gap above is therefore a **lower bound**.

### It ships for all four positions anyway, and not because of fit

- **Admissibility beats deviance.** A model that can output a negative probability is
  wrong whatever it scores. TE's linear form crosses zero at pick 278 — outside a 262-pick
  draft, but by sixteen picks, and that margin is the only thing between it and the QB bug.
- **Keeping linear for TE means keeping the floor patch for TE.** One form removes a class
  of failure; two forms retain it in one corner and add a per-position exception to defend.
- **The portfolio is overwhelmingly positive.** TE gives up 0.17 deviance per observation;
  QB gains 4.69.

The cost is real and is recorded rather than rounded away. If TE ever earns features of its
own, this is the first thing to re-test.

### What the refit removed

- `CLIP_TO_OBSERVED_FLOOR` — retired. A logistic curve cannot reach 0 or 1 for any finite
  input, so there is nothing left to clamp. The guarantee now lives in the functional form
  instead of in a constant somebody has to remember to keep correct.
- `_ridge_fit` and `_clip01` — deleted under the delete-dead-things rule. Leaving a fitter
  on disk that nothing ships is how it gets re-adopted by autocomplete in six weeks.

The UDFA inheritance stays. It was never about the functional form — a pickless rookie has
no `pick` to feed any curve, and inheriting the measured Day 3 share is a missing-data rule.

### A comment that was confidently wrong, corrected

`ALPHA = 0.10` was documented in `playing_time.py` as a "ridge penalty… the project's one
regularisation constant." It is neither. `fit_weights.fit_position` calls plain
`sm.OLS(...).fit()` and uses ALPHA as a **p-value cutoff** in its two-stage keep/drop —
there is no ridge anywhere in this project. The number was right and the sentence
explaining it was invented.

Recorded rather than quietly deleted. A confident wrong comment is worse than no comment,
and the only defence against the next one is noticing this one.

### Selftest strengthened

The synthetic fixture is now on the logit scale, and the PUP row was inverted: the fitted
absence (7.17 games) now **exceeds** the PUP absence (4.0), so `max()` is exercised from
the opposite side. Under the old fixture a `sum()` bug would have passed; now it returns
11.17 and fails. The pick-250 QB — the exact input that returned a negative probability
under the linear model — returns 3.2%.

### Still to do in 13.5b

- Combine measurables. Join is **not** on `gsis_id`: the nflverse combine table carries
  `pfr_id` and `cfb_id` only, so it has to bridge through `load_draft_picks`. And combine
  data covers **invitees only**, which is non-random missingness that needs its own
  indicator alongside — the same treatment `depth_chart_missing` and `trend_missing`
  already get.
- Rookie-usage tendency at **draft team** and at **playcaller**, tested separately. Must key
  off draft team, never the week-1 roster snapshot, or it reintroduces the survivorship
  this entire phase exists to remove. Playcaller cells will be thin (~6 drafted skill
  rookies each) — thinner than the round-1 cells that already broke once.

### The logit re-gated (Aug 12) — it improved the shipping predictor

|  | linear | logit |
|---|---|---|
| **B** rate@8 × availability | RMSE 59.41, bias +1.30 | **RMSE 58.92, bias +1.09** |
| **D** rate@0 × availability | RMSE 59.33, bias −6.29 | RMSE 59.17, bias −6.29 |
| **D over B** | +0.08 (+0.05 SE) | **−0.25 (−0.16 SE)** |

The refit improved the predictor that actually ships, on both RMSE and bias. And the
SHIP B conclusion **strengthened**: under the linear form the rate change was merely
worthless (+0.05 SE); under the logit it is actively negative (−0.16 SE). Two functional
forms now agree that availability is the whole effect and the rate change double-counts.

Fitted tails are sane at last — QB pick 245 reads 4.1% rather than 0.0%, TE 13.5%. Every
slope is significant past any threshold worth quoting (QB p=2e-81, WR p=6e-143).

### Usage tendency — screened out BEFORE building it

Permutation test, 2000 shuffles of the team label, on the standard deviation of team mean
availability:

| | observed sd | p |
|---|---|---|
| raw `available_share` | 9.70% | **0.018** |
| after removing draft capital | 6.84% | **0.153** |
| playcaller, same residual | 8.53% | 0.136 |

**The raw team effect is real and it is draft capital wearing a franchise's name.** Teams
whose rookies play more are teams that draft rookies higher — and `pick` is already the
model's only feature. Residualise availability on pick within position and no team effect
is detectable. The playcaller cut is no better despite four times the resolution, and it
was always going to be thin: 74 of 95 playcallers have under 12 rookies, only 45% of
players get a usable cell.

Sizing, for the record: 32 of 33 teams clear n=12 at team level (median 22 rookies each),
so this is not a coverage failure. The cells are fine. There is just nothing in them once
draft capital is removed.

**Stated with the right strength.** p = 0.153 is not proof of absence — 32 teams × 9
classes × ~22 players cannot see an effect smaller than roughly 3 points of share. The
claim is *"no team effect detectable beyond draft capital at this sample size,"* which is
a reason not to spend a gate run, not a proof that coaching staffs are interchangeable.

`usage_tendency()` is kept rather than deleted — a screen is not a gate — but its docstring
now carries this result so nothing adopts it without re-reading, and names collinearity
with `pick` as the specific thing to check if it ever is.

**This is what testing before building is for.** The feature had a plausible mechanism, a
clean data source, sufficient cell sizes, and a real-looking raw signal. It took one
permutation test to find that the signal was already in the model.

### Combine — the remaining live candidate

`src/rookie_traits.py` written, not yet run (needs nflreadpy). Three things it handles that
are not obvious:

- **The join is not on `gsis_id`.** The combine table has none; it carries `pfr_id`, so it
  bridges combine.pfr_id → draft_picks.pfr_player_id → gsis_id. Lossy, so the match rate is
  reported rather than assumed.
- **Combine data is invitees only**, and invitation correlates with prospect status, which
  correlates with the target. Each measurable is mean-imputed *within position* with its own
  indicator, plus a `combine_missing` flag — same treatment `depth_chart_missing` already
  gets. Pooling the `forty` mean across QB/RB/WR/TE would fabricate an implausible value
  rather than a neutral one.
- **`load_draft_picks` is a leak hazard.** It ships `games`, `seasons_started`, `car_av`,
  `w_av`, `receptions` and more in the same frame as draft capital — and `games` is career
  games played, which is very nearly the target. `DRAFT_PICK_COLUMNS` is therefore an
  **allowlist, not a drop-list**, so a new nflverse column arrives excluded by default
  rather than included until somebody notices.

### Combine measurables — tested out of sample, rejected (Aug 12)

`ht` and `wt` correlate at **0.71**, so testing them separately doubles the multiplicity
for one underlying fact. Collapsed into `size` = mean of within-position z-scores, and run
at **all four positions as one pre-registered feature** rather than picking the winner
after looking. Leave-one-class-out binomial deviance, imputation means computed inside each
training fold, 300 bootstrap resamples:

| pos | n | pick only | +size | gain | 95% CI | P(gain > 0) |
|---|---|---|---|---|---|---|
| QB | 104 | 707.6 | 673.2 | +34.4 | [−43, +116] | 73.3% |
| RB | 195 | 1653.9 | 1667.5 | −13.6 | [−71, +18] | 11.0% |
| WR | 290 | 2122.9 | 2130.5 | −7.6 | [−65, +52] | 23.3% |
| TE | 130 | 887.3 | 856.7 | +30.7 | [−54, +120] | 70.7% |

**Not one interval excludes zero.** QB and TE look encouraging and are indistinguishable
from noise at these sample sizes.

The in-sample screen had said the same thing first: 5 of 31 feature × position partial
correlations reached p < 0.05 against 1.6 expected by chance, and Benjamini-Hochberg at
FDR 10% kept nothing. Four of the five were `ht` and `wt` at QB and TE — two correlated
columns reporting one fact twice, which is what motivated the composite.

**Two bugs found in the process, both quiet ones.**

- `ht` is a **string** — `"6-2"` — despite the nflverse dictionary typing it numeric.
  `fill_null(mean())` over a string column returns null in polars, so height was silently
  left unimputed and would have entered the fitter as a constant: a feature that could
  never earn its slot, looking like evidence that height does not matter. Now parsed to
  inches, and it raises if the format changes.
- My screening script returned deviance `0.0` when every fold was skipped, which printed
  as `QB bench +707.6` — the largest "gain" in the table and pure artifact. QBs do not
  bench press at the combine, so `nanmean` was NaN and every fold `continue`d. Fixed to
  return None and fail loudly. **A missing-data path that returns a number instead of
  nothing produces the most exciting result on the page.**

### Phase 13.5b closed

| | outcome |
|---|---|
| Logit refit | **SHIPPED.** Gate predictor B improved 59.41 → 58.92 RMSE, bias +1.30 → +1.09 |
| Usage tendency (team + playcaller) | **REJECTED** — collinear with draft capital, p = 0.153 |
| Combine measurables | **REJECTED** — no position's bootstrap CI excludes zero |

**What the phase established: availability is draft capital.** That is now a finding
rather than an assumption. The two feature families with a plausible mechanism and a clean
data source were both tested and both failed; the model is short because the data is, not
because nobody looked.

`MODEL_VERSION` 14 → 15. Every rookie's Exp Gm changes under the logit — undrafted QBs
most of all, 0.0% → 4.1%. Rank still does not move, and `--expect rank-identical` against
v14 is the check.

---

## Carried forward: QB is the least-modelled position on the board (Aug 12)

Found while reading the v15 sanity output. Four of the seven names on the 32-team board's
fade list are quarterbacks, and **all four have empty driver strings.** Nothing fires
because there is nothing to fire.

**QB has no situational weights.** `fit_weights.FEATURE_SPECS` covers RB, WR and TE only.
Phase 13 CP2's holdout cut age, which was Phase 10's headline finding and QB's only
feature, and no replacement was ever found.

**QB also has no baseline shrinkage.** `SHRINKAGE_EXCLUDED_POSITIONS = {"QB"}`, and that
exclusion is correct for the reason already recorded beside it: shrinkage moved 59% of
quarterbacks UP, because for a backup QB a small sample is not a noisy estimate of the same
quantity, it is a precise estimate of a different one. Verified on the live table —
`fantasy_points_per_game` and `fantasy_points_per_game_shrunk` are identical for every QB
and differ by up to 4.97 for RBs.

So a quarterback's `adjusted_fantasy_points_per_game` is his own recency-weighted trailing
average, with thin seasons discounted, and **nothing else.** No population anchor, no
situational adjustment. He is the only position modelled purely from his own history.

### What that does, measured

Board rank minus market ADP, across the 36 quarterbacks on the 32-team board that carry an
ADP:

| experience | mean board-vs-market gap | n |
|---|---|---|
| ≤ 1 yr | **−41.1** | 6 |
| 2–3 yr | −21.8 | 7 |
| 4–6 yr | −21.4 | 8 |
| 7+ yr | −17.3 | 15 |

`corr(games in sample, gap) = +0.34`. The board fades every quarterback relative to the
market, and it fades the young ones roughly two and a half times harder.

**The mechanism is not over-shrinkage — it is the absence of a development curve.** A
second-year quarterback's trailing average is largely his rookie season, which is
systematically depressed. The market prices in improvement. The model has no term that
can, because the term that would have — age — was cut by the holdout for failing out of
sample at QB specifically.

### Why it matters most exactly where it is worst

In the 32-team superflex, **18 of the top 60 are quarterbacks.** The position that
determines the board's shape is simultaneously:

- the least-modelled position on it,
- the one whose replacement level rests on a single autodrafted mock (QB59), and
- the one whose ranks sit on the cliff that assumption straddles.

On the 12-team and 6-team boards QB is 7 and 4 of the top 60, so the exposure is small.
This is a 32-team superflex problem.

### Candidates, if this is picked up

Ordered by how likely they are to survive a gate, most likely first:

1. **Re-run the QB feature bake-off with the current candidate set.** QB was dropped after
   `age` failed; the RB/WR/TE features were never tested at QB against the nine-fold gate.
   `pass_att_pg` and `team_changed` in particular have plausible QB mechanisms and already
   ship in `player_features.csv`. Cheapest possible test, no new data.
2. **A starter/backup split before anything else is fitted.** The shrinkage comment already
   establishes that QBs are two populations, not one, and that treating them as one broke a
   different mechanism. Every QB model since has still been fitted on the pooled set.
3. **An experience or games-played term** aimed directly at the measured pattern above.
   Fits the evidence, but it is a curve fitted to a gap against ADP, which is a market
   quantity — close to fitting the model to the market, and it should be tested against
   actual outcomes rather than against the gap that motivated it.

### Also carried: the rookie cohort tie

Jeremiyah Love and Jadarian Price hold identical `Adj PPG` of 15.12 and sit at ranks 23 and
24, separated only by the ADP tiebreak. The market separates them by 26 picks. Rookie
baselines are one value per position × round bucket, so two first-round backs are the same
player to this model. `pick` is already a candidate in `fit_rookie_weights` and did not
clear alpha for RB or WR; only TE/age ships. Worth re-testing now that Phase 13.5 has shown
`pick` carries real signal on the availability side.

### Correction: `Worst rank` is the wrong drafting order (Aug 12)

The QB59 stress test shipped with the instruction "draft off the Worst rank column."
Jack caught that this contradicts the QB59 decision, and he is right.

`Worst rank` is the worst rank a player holds across QB45–QB70. For a quarterback that is
his rank under **QB45–49** — the scenario the phase examined and rejected. QB59 is a
measured count off a real mock; QB45 is a point on a sweep that nothing observed. So the
instruction amounted to: keep the measured assumption, then draft as though the rejected
one might be true.

**And the asymmetry makes it actively wrong, not merely inconsistent.** The two errors are
8:1 — waiting on QB when the room takes 59 costs ~150 points; taking one early when the
room takes 45 costs ~19. `Worst rank` is a minimax criterion, and **minimax minimizes
regret in rank space, not in points space.** Under a loss function that lopsided it does
not reduce exposure; it maximizes exposure to the expensive error.

The generalizable rule, and the one this project should carry: **a robustness criterion is
only conservative if the losses it hedges across are symmetric.** Check that before
adopting one. A minimax rule applied to an asymmetric loss is a confident bet wearing the
costume of caution.

Note also that the difference was never league-wide: for RB, WR and TE the two orderings
differ by at most 14–26 places. The entire disagreement lived in the one position where
the loss function is most lopsided, which is precisely where a symmetric rule does the
most damage.

The stress test keeps two legitimate uses, both narrower than an ordering: the robust 46
as a **tiebreaker** between adjacent players, and the rounds 1–2 QB run as a **live read**.
`BUILD_RUNBOOK.md` is corrected.

---

## Phase 13.6 — twelve rounds, and ADP that reaches them (Aug 12)

The league added a fifth bench spot: 32 teams × 12 rounds = **384 picks**, up from 352.
Two things had to change with it, and only one of them was the arithmetic.

### The board died at round 6, and had all along

FFC's deepest 2026 2QB data runs out around **pick 188**. That is 5.9 rounds of a 32-team
draft. Every player past it had `has_adp = false`, which on this board means hard-capped
below every ADP-bearing player and shaded pink regardless of projection — so from round 7
on, the sheet ordered players by a rule that had nothing to do with the model. Six of
twelve rounds. The board was unusable in exactly the half of the draft where a 32-team
room stops being obvious.

This was true before the roster change; twelve rounds only made it louder. It never
surfaced because the earlier phases' verification all lived in the top 60, where FFC is
dense and the failure is invisible.

### Fix: the mocks become the feed

`src/mock_adp.py` builds ADP out of two real 32-team superflex mocks — 736 picks
transcribed by hand from the Sleeper boards, resolved to player_ids, averaged. Coverage
goes **188 → 312 players, 5.9 → 9.8 rounds**. Of the model's top 200, six lack ADP, and
they are Joe Flacco, Derek Carr, Russell Wilson, Jeff Driskel, Easton Stick and Emari
Demercado. Those pink rows are correct.

**It replaces FFC on this board rather than filling its gaps**, and that is the load-bearing
decision. The tempting version keeps FFC through pick 188 and uses mocks past it. But
`compute_replacement_ranks` reads the ADP *order* of the first `skill_picks` names, and
pick 150 of a 12-team draft is not the same draft capital as pick 150 of a 32-team draft.
Mixing two scales inside one ordering corrupts replacement level at every position with
nothing downstream able to notice — the same class of silent error the `expected_drafted`
sum check was added to catch. One scale, or the other.

**The honest cost:** two drafts against FFC's hundreds, and mock A had 31 of 32 teams
autodrafting. That thinness ships onto the sheet as `times_drafted` and `adp_stdev` rather
than living in a comment.

### Two mocks of different lengths

Mock A ran 11 rounds, mock B 12. Averaging raw pick numbers would treat A's last pick as
32 picks earlier than B's when both mean *the end of the draft*, so A is rescaled to the
384-pick shape (×384/352) and averaged there.

A player taken in **only one** mock is not a player with one observation — he is a player
one room passed on entirely, which in a 32-team draft is evidence. The missing observation
is censored at pick 385, the first pick that did not happen. He sorts behind a player taken
at the same depth twice, and his stdev blows up to say how little is known, which widens
his Draft Target cushion rather than faking a precision two drafts cannot support. 17 of
312 players are in this state.

### Replacement level: same treatment, same arithmetic

`expected_drafted` goes from the Phase 13 mock's raw counts to the average of both mocks,
with A put on the 12-round scale first:

| | QB | RB | WR | TE | total |
|---|---|---|---|---|---|
| mock A, as drafted (11 rd) | 59 | 92 | 145 | 56 | 352 |
| mock A, scaled to 12 rd | 64 | 100 | 158 | 61 | 384 |
| mock B, as drafted (12 rd) | 60 | 97 | 165 | 62 | 384 |
| **shipped average** | **62** | **99** | **162** | **61** | **384** |

The bar moves down 0.0–0.35 PPG depending on position — QB +0.17, RB +0.00, WR +0.30,
TE +0.35 of VOR — so this is not a re-ranking, it is a recalibration. **QB62 is on the same
side of the QB49/50 cliff Phase 13.5 found**, which is the only thing the QB count has ever
had to get right.

### What verified this

The transcription is 736 hand-read cells, so it gets a checksum rather than trust: the
position counts implied by each transcription reproduce the counts read off the live
boards exactly — QB 59 / RB 92 / WR 145 / TE 56 for A, QB 60 / RB 97 / WR 165 / TE 62 for
B. A single misread cell would break one of those totals. Both landed.

Name resolution is first-initial + surname prefix (the boards truncate: "T. Henders…"),
narrowed by the team the cell prints, walked in pick order so a player already taken can't
be taken twice — which is what separates the two `B. Robinson / RB / ATL` cells in each
mock into Bijan in round 1 and Brian in round 4.

**Known gap, unchanged by this phase:** Travis Hunter, Jarquez Hunter, Will Howard and
Zack Kuntz are drafted in both mocks and appear on neither board, because nflreadpy's
player table doesn't carry them as modeled skill players (Hunter is a CB there). They were
undraftable here before and still are. Worth a Phase 14 look, since Travis Hunter went in
round 6 of both mocks.

---

## Phase 13.7 — Travis Hunter was never on the board (Aug 12)

Phase 13.6's transcription surfaced seven names drafted in both 32-team mocks that the
model has never heard of. One of them is not like the others.

**Travis Hunter went in round 6 of both mocks and had never appeared on any board here.**
Not ranked low — absent. `features.load_veteran_stats` filters nflverse stat rows to
positions QB/RB/WR/TE before reading a single number, and nflverse carries Hunter as a
**CB**. Every receiving row he has was dropped at the source, three modules deep, silently.

That is worse than a bad projection. A player rated 180th is an opinion you can argue with
on the clock; a player who isn't on the sheet produces no opinion at all, and the gap is
invisible precisely because nothing is there to notice.

### The filter is the model's membership test, so the fix goes upstream of it

`position_overrides.csv`, hand-maintained, same convention as `injury_overrides.csv`:

    player_name,position,note
    Travis Hunter,WR,"nflverse carries him as CB..."

Applied by `src/position_overrides.py` at every point where an nflverse position is read
before a position filter — `features.load_veteran_stats` (the baseline) and
`situational._load_season_usage` (the usage trend). It could not be a single patch in
`features.py` because four modules pull their own frames from nflverse and each applies
its own filter; a fix in one would have given Hunter a baseline and no trend.

**A file rather than a constant**, because the next miscoded player should be a one-line
edit by whoever notices him missing, not a code change by whoever still remembers this.

### An unmatched name raises, and the raise is the diagnostic

`strict=True` on the veteran stats frame: a name matching nothing there means the override
did nothing and the player stays off the board, which is the exact failure the file exists
to prevent. It is off for depth charts and roster snapshots, where a listed player
legitimately may be absent (Hunter has no offensive depth-chart row, so he takes
`depth_chart_missing` like any other player without one).

That raise also **sorts the other six missing names into the right bucket**. Jarquez
Hunter, Will Howard, Zack Kuntz, Audric Estime, Dont'e Houston, Lawrance McCutcheon: adding
them to this file raises, because nflverse has no offensive rows for them at all in
2023–25. They are missing for lack of DATA, not lack of a label, and no relabelling
rescues a player with no baseline. They stay off the board and this file is honest about
why.

### He is scored as an ordinary WR

Deliberately, over two tempting alternatives. Flagging `baseline_low_confidence` would
shrink him harder toward the WR mean on the argument that his offensive snap share is
unlike his comparables; blanking his situational adjustment would sidestep features
computed off a CB listing. Both are thumbs on the scale that **no backtest has tested**,
applied to exactly one player — which is the shape of special case the holdout gate exists
to catch. The model has no two-way concept and is not being given one for a sample of one.
His receiving line goes through the same shrinkage, the same situational terms and the
same replacement level as every other receiver, and if that is wrong it will be wrong in a
way that is visible and arguable on the sheet.

### Why this bumps the model version, and the prediction it falsified

15 → 16 → **17**. The bump was argued for on the grounds that adding a player to a position
pool moves other players: Hunter joins the WR group `position_competition_top_k` averages
over at Jacksonville, and the pool baseline shrinkage anchors against.

**Measured against v16, that was wrong.** Zero Jacksonville players moved. Two of 450
receivers moved, both for unrelated reasons. Every rank below Hunter's 166 shifted by
exactly +1 — an insertion, not a revaluation. His PPG sits below the top-K competition set
at his own team, and one player added to a 450-man pool does not move a percentile anchor.
The mechanism was plausible, cheap to check, and did not happen.

It still bumps, on the honest reason instead of the guessed one: **the board contains a
player it did not contain before.** A version that tracked only whether existing numbers
moved would call two boards with different rosters the same version.

### Verification, and a trap in the v16 → v17 diff

Hunter lands at **rank 166, Adj PPG 8.12, mock ADP 185.0, Value Δ +4** — the model
independently puts him within four picks of where two rooms actually drafted him, which is
about as much external confirmation as a single player gets here. His drivers read
`+1.2 age 23 · -0.9 2025 IR · -0.3 19% team share | -1.2 thin sample (7 gm)`, and the thin
sample is real: seven games of receiving is what there is.

**`compare_boards v16 v17` reports 63 players whose Adj PPG moved, and none of them are
Hunter's doing.** Between the two pipeline runs nflverse dropped three undrafted rookies
(Anthony Hankerson, Liam Clifford, Mante Morrow) from its player table, which recomputes
position competition for their teammates — the movers cluster on PHI, ATL, PIT, NE and CLE,
and the largest is Elijah Mitchell at rank ~900. The universe went 1087 → 1085: minus three
rookies, plus Hunter.

This is worth writing down because the diff is exactly the shape that invites a wrong
conclusion. Sixty-three moves appearing in the same rebuild as a universe change look
caused by it, and the check that separates them is one line: **which teams moved.** None of
them was Jacksonville.

---

## Phase 13.6b — A.J. Brown went in round 7, and every checksum said fine (Aug 12)

Jack read the board, saw A.J. Brown at ADP 196.8, checked the mocks, and found him at 1.27
in **both**. Three players were corrupted, not one, and the way this got past verification
is more useful than the bug.

### What happened

Mock A's pick 11.16 is a cell reading `A. Brown / QB` with no team — some camp quarterback
the model has never heard of. The matcher had two independent weaknesses that only
combined into damage here:

1. **Position was negotiable when the team was blank.** The fallback returned candidates of
   any position rather than none, so a QB cell matched A.J. Brown, the receiver.
2. **An already-drafted player could be drafted again.** `free = [...] or pool` was written
   to avoid dropping a pick, and instead assigned Brown a *second* pick at 336.

`average()` stores one pick per player per mock in a dict. The second write silently
replaced the first. A.J. Brown's mock A pick went from 27 to 336, and his ADP from 28 to
197 — six rounds late, on a first-round receiver.

Two more were hit the same way. `J. Lovett / RB` at 11.9 matched **Jeremiyah Love** (the
symmetric prefix rule let a LONGER board name match a shorter universe one — "Lovett"
reaching "Love"), dragging Love from 29 to 178. A second `C. Sutton / WR` cell at 12.19 —
a different Sutton with no team printed — re-took Courtland Sutton and moved him from 96 to
237.

### Why every check passed

This is the part worth keeping. The transcription was **correct**; the position counts
still reproduced the live boards exactly; both mocks still had precisely 352 and 384 picks.
None of it could have caught this, and for one reason: **no pick was lost.** One player was
credited with two of them. Every check in place counted picks, and counting picks is blind
to a bug that conserves them.

The invariant that catches it is the one the draft itself enforces and nothing in the code
had asserted: **within a single mock, a player appears at most once.** `audit()` now raises
on it before anything is written.

**The general form, and it is not specific to draft boards:** a checksum over TOTALS cannot
see an error that redistributes within the total. Every check here summed or counted;
the failure was a permutation. When adding a checksum, ask what class of error it is blind
to by construction, because that is where the next bug will live.

### The other three fixes, all from the same audit

- **Prefix matching is now asymmetric.** Truncation only makes the board's name shorter, so
  the board's surname may be a prefix of the model's ("Henders" → Henderson) and never the
  reverse. This is what stopped "Lovett" reaching Love.
- **`LAR` vs `LA`.** The boards print Sleeper's abbreviations, the model stores nflverse's,
  and they agree on everything except the Rams. Nine players — Nacua, Kyren Williams,
  Stafford, Corum, Adams and four more — had their team check silently disabled and
  resolved correctly only because no same-surname rival happened to exist. Now normalized
  through the project's own `TEAM_ABBR_FIXES`.
- **Team-confirmed cells resolve first.** Both mocks contain an anonymous `T. Scott / WR` a
  few picks *before* `T. Scott / WR / LAR`, and pick-order-first handed the only Tyler Scott
  in the universe (a Ram) to the anonymous one. Resolution now runs in two passes.

### What the re-run changed, and what it did not

**Three ADPs moved: A.J. Brown 196.8 → 28.2, Jeremiyah Love 178.1 → 29.1, Courtland Sutton
236.8 → 95.8.** Every other player on the board is byte-identical. Coverage went 312 → 308:
Audric Estimé joined, and five deep fliers (Damien Harris, Dante Miller, Dae'Quan Wright,
Robert Henry Jr., Tyreik McAllister) dropped out because each was a position mismatch with
no team to confirm it — picks 340-380 cells that were matching the wrong human all along.

### An independent check, added because the internal ones were not enough

Every internal check was consistent with a corrupted board, so the audit now includes one
that comes from outside the transcription: **compare each player's mock ADP rank against
FFC's 2QB rank.** 188 players appear in both feeds; the median disagreement is 12 places.
The largest are Mendoza (+95, a rookie QB a 32-team superflex room reaches for), Stafford
(-73) and McBride (+47) — all genuine room-vs-market differences on unambiguous names. A
cell resolved to the wrong human would surface here as a player the market ranks nowhere
near where "he" went, and none does.

## Phase 13.8 — an 8-team board, and the ADP feed that does not exist (Aug 13)

### The board

`league_config_8team.json`, cloned from the 12-team config with `num_teams` 8 and `keeper_rule`
null (the 8-team league is redraft). Verified programmatically that no other key differs
from the 12-team config beyond the three naming fields. Nothing in `src/` changed — the
board is entirely a function of the config, which is what `board_label` was built for.

Build line added to the runbook's refresh block:

    python -m src.build_board --config league_config_8team.json --version 17

### The question it raised, which was the more interesting one

`FFC_TEAMS = 12` in `src/adp.py` is hardcoded and every board reads that one feed. The 6-
and 8-team boards rescale it for DISPLAY via `format_pick`, and `adp_caveat` attaches a
CAUTION saying rescaling preserves order but not behavior. That much was known and
documented.

What was not front-of-mind is that ADP is **not** display-only. `compute_replacement_ranks`
walks the same list to `skill_picks` and takes the position mix of those picks as
replacement level, which sets VOR, which sets rank. Its own docstring calls this "the one
place ADP earns its keep." So the feed does not just mislabel a column.

The structural worry: **pick 112 in a 12-team feed is round 9 — mid-draft. Pick 112 in a
real 8-team room is round 14 — late draft.** The method reads a mid-draft position mix and
applies it to a late-draft moment. The deeper a board cuts into the feed relative to the
feed's own room size, the more that bites.

### The sensitivity, which is higher than it looks

Walking the current feed and pricing the result against the projection curve:

| League | Skill picks | QB | RB | WR | TE |
|---|---|---|---|---|---|
| 6-team | 84 | 8 | 32 | 39 | **7 (floor)** |
| 8-team | 112 | 15 | 37 | 51 | **10 (floor)** |
| 12-team | 168 | 22 | 55 | 70 | 21 |

Moving the QB cut by ±4 shifts every quarterback's VOR by ~1.0 PPG. RB at −6 is +1.63, TE
at −6 is +2.74. **The median gap between adjacent players in the top 112 is 0.06 PPG.** The
board is densely packed, so a whole-position VOR shift of 1.0 PPG is worth roughly fifteen
places of wholesale movement, not one. Replacement-level error is amplified here, not
absorbed. That is the number to remember the next time a change touches replacement level.

Two things already insulate the shallow boards: the TE starter floor binds on both (7 > 5
derived at 6-team, 10 > 9 at 8-team), so TE ignores ADP there regardless; and the feed
covers 210 players against 112 picks needed, so the tail-extrapolation machinery that
produced QB63 on the first 32-team build never fires.

### The answer: there is no other feed

`check_8team_adp.py` (throwaway, deleted) pulled `teams=8` and `teams=12` and cut both at
112. Zero delta at all four positions. Identical payload depth, 211 offensive players each.

That agreement was too perfect to be agreement. The follow-up compared raw ADP values:
**256 of 256 players byte-identical across the two feeds.** FFC ignores `teams` for
ppr/2026 and serves one payload. `teams=6` returns a 400.

So the decision — do not spend the hour wiring a per-league feed — is right, but **not for
the reason the zero-delta output appeared to give.** Nothing was measured and found equal.
There is one feed, and the round-9-vs-round-14 conflation remains **untested, not
refuted.** Recording it the other way would have been exactly the failure the runbook names
elsewhere: absent is a known state, stale is a lie.

Noted in `src/adp.py` at the constant, because a future attempt to pass `teams=8` will
appear to succeed while changing nothing.

### The general form, which is the part worth keeping

**When two independent sources agree perfectly, check that they are independent before
crediting the agreement.** Four matching position counts read as confirmation; a matching
byte count read as one source wearing two hats. The check that distinguished them cost
thirty seconds and inverted the finding.

This is the same shape as the 13.6b lesson one section up — a checksum over totals cannot
see a permutation within the total — with the roles reversed. There the aggregate hid a
real difference; here the aggregate invented a real sameness. Both are answered by asking
what the check is blind to by construction.

### If this ever needs a real answer

It needs mock drafts run in an 8-team room, transcribed into `data/mock_boards/` the way
the 32-team board's were in 13.6, and an `expected_drafted` block typed off them. That is
the path already built for exactly this problem, and it is the only one that does not
depend on a vendor exposing a parameter it does not expose. Not worth it for a league where
waivers are deep enough to fix a QB mistake in week 2 — which is the honest reason to leave
this alone, and a better one than "measured, no difference."
