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
| 10 — Usage trend and age curves | Aug 10–14 | Aug 3–7 |
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

### Phase 10 — Usage trend and age curves (Aug 3–7)

Two features that are currently too blunt.

- **CP1** — *Usage trend*: slope of target share / carry share across the last 3
  seasons. `workload_share` sees a 22% share; it can't tell rising from falling.
- **CP2** — *Age curves*: replace linear `experience` with position-specific age
  terms. One linear slope across RB and WR is almost certainly wrong — RB decline is
  steeper and starts earlier. Test quadratic or binned age against the current term.
- **CP3** — Backtest both, keep what's significant.

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
- **CP2** — Holdout validation: fit on 2023–24, test on 2025. Does the model beat the
  raw baseline out of sample? If not, the added features are overfitting and should be
  cut. This is the honest test the project hasn't run yet. **Not cut for time.**
- **CP3** — Retest `coach_changed`. Phase 9 found it moves mean delta by 0.1–0.4 PPG
  against sds of 2.5–7.1. It ships today inside `continuity_score` on the strength of
  Phase 3 reasoning, never a significance test. Same bar as everything else: keep it
  only if it earns its slot. If it goes, `continuity_score` becomes `qb_changed` alone
  and should be renamed rather than silently redefined.
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
- **Nothing gets deleted without Jack's approval.** Superseded code gets commented and
  labeled, not removed.
- **Every phase ends with a Git commit** at the checkpoint boundary.
- **Exploratory "does this pattern hold" analysis is Jack's hands-on task** — Claude
  builds the scaffolding, Jack runs the exploration and brings back findings.

## Open items carried forward

- Null `player_id` rows in `player_stats` — flagged in Phase 2, still uninvestigated.
- K and DST are not modeled at all. Acceptable (draft them last), but it should be a
  decision rather than an omission.
- No holdout validation has ever been run. Phase 13 CP2 is the first. Every R² quoted
  so far is in-sample.
