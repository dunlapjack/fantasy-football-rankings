# Phase 14 — how to actually draft off the board

*In plain English, the way Phases 12 and 13 were written.*

---

## The question

> Using the 12-team board I find myself drafting 5 RBs and 2 QBs before even
> getting my 2nd WR. Should I always go best available, or fill my starting
> lineup first, or somewhere in between? Is there a statistically best way?

Short answer: **best available is the second-worst thing you can do**, the only
thing worse being drafting straight off consensus ADP. It is not close, it is
not a matter of taste, and the reason is a specific and fixable flaw in how the
board is built rather than anything wrong with the projections.

---

## Why the board does this to you

`build_board.py` sorts by VOR — a player's points per game minus the points per
game of the last player drafted at his position. Look at when that number is
computed: **once, before the draft starts, against an empty roster.** It never
moves.

It cannot move. It is a property of the player and the league. Nothing in it
knows what you already have.

So in round 8, holding four running backs, the board still prices RB5 at
exactly what he was worth when you had none — even though he will start zero
games for you all season. The same arithmetic recommends a second quarterback
in a one-QB league: QB2's VOR is a positive number, positive numbers sort above
smaller positive numbers, and the receiver who would actually enter your lineup
has a smaller one.

Your instinct was right, and it was right for the reason you'd guess. Best
available on a static-VOR board is not best available **to you**.

---

## How it was tested

`src/draft_sim.py`. Simulate the draft, simulate the season, count the points.

- Opponents draft by ADP plus each player's own `adp_stdev` as noise, with
  roster caps, from your real draft slot in each league.
- Your team drafts by one of six **policies** (below).
- The resulting roster plays out that league's regular season **week by week**:
  byes are certain, injuries are a Markov chain, the lineup is filled optimally
  from whoever is healthy, and any slot you can't fill falls back to the third-
  best undrafted player at that position — because in a real league you'd
  stream him.
- 120 drafts per policy per league, with **common random numbers**: every policy
  faces the same opponents and the same injuries in simulation *n*, so the
  difference between two policies is the policies and not the luck.

The week-by-week loop is the part that makes the experiment honest. Score a
roster by its starters alone and bench players are worth exactly zero, so the
simulator would always say "fill your lineup, then draft nothing" — an answer
produced by the scoring function rather than by football. Byes and injuries are
what make a 4th running back worth anything, so they have to be in there.

### The six policies

| | what it does |
|---|---|
| **adp** | draft the consensus board. The control — what a normal manager does. |
| **board** | pure best-available off your v17 board. **This is your current behaviour.** |
| **caps** | best-available, but never more than 2 QB / 6 RB / 7 WR / 2 TE. |
| **starters_first** | fill every starting slot, then revert to best-available. |
| **vona** | rank by how much better a player is than the best man at *his position* who'll survive to your next pick. Prices scarcity; ignores your roster. |
| **marginal** | one-ply lookahead: put the candidate on your roster, finish the rest of the draft greedily with who'll realistically be there, score the completed team. Take whoever leads to the best finished roster. |

---

## The results

Points scored by your starting lineup across the regular season, versus pure
best-available. Every number below is many standard errors from zero — the
smallest one that matters is 7σ.

### 12-team (pick 3, keeping Olave at a round-8 cost)

| policy | season points | vs best-available |
|---|---|---|
| **marginal** | 1535.6 | **+54.7** |
| vona | 1530.4 | +49.4 |
| starters_first | 1505.8 | +24.8 |
| caps | 1489.9 | +9.0 |
| *board (best available)* | *1481.0* | *—* |
| adp | 1404.6 | −76.4 |

**+54.7 points is about 3.9 points a week, every week, for free.** That is
larger than the entire measured edge of the projection model over a flat guess,
which Phase 13 put at 0.43 PPG at running back. You have been giving back more
value at the draft table than the model earns you.

Look at what each policy actually drafts, averaged over 120 drafts:

| policy | QB | RB | WR | TE |
|---|---|---|---|---|
| board | 2.5 | **7.9** | **2.5** | 1.1 |
| marginal | 3.2 | 4.2 | 4.5 | 2.1 |

There it is — your complaint, reproduced exactly by the simulator without being
told to. Pure best-available on the 12-team board hands you **eight running
backs and two and a half receivers.** You have two WR slots and a flex. You are
starting a waiver-wire receiver most weeks.

### 8-team (averaged over all eight draft slots)

| policy | season points | vs best-available |
|---|---|---|
| **vona** | 1599.6 | **+37.6** |
| starters_first | 1594.4 | +32.5 |
| marginal | 1593.8 | +31.8 |
| caps | 1566.4 | +4.4 |
| *board* | *1561.9* | *—* |
| adp | 1505.0 | −57.0 |

Board best-available drafts RB7.4 / WR2.5 here too. The top three policies are
inside each other's error bars — in a shallow league it does not matter *which*
lineup-aware rule you use, only that you use one.

### 6-team (pick 5)

| policy | season points | vs best-available |
|---|---|---|
| **starters_first** | 1432.5 | **+23.0** |
| vona | 1429.7 | +20.2 |
| marginal | 1427.6 | +18.1 |
| caps | 1410.2 | +0.7 |
| *board* | *1409.5* | *—* |
| adp | 1333.3 | −76.1 |

Smallest gap of the four, and the crude rule wins. That makes sense: with six
teams the waiver wire is so rich that a bad pick is cheap to fix. Note this is a
12-week season, so +23 is still about +1.9 a week.

### 32-team superflex (pick 4, Puka already taken)

| policy | season points | vs best-available |
|---|---|---|
| **marginal** | 1280.8 | **+40.7** |
| vona | 1261.7 | +21.6 |
| caps | 1241.6 | +1.5 |
| starters_first | 1241.4 | +1.3 |
| *board* | *1240.1* | *—* |
| adp | 1066.0 | −174.1 |

The one league where `starters_first` does nothing. With 32 teams and 12 rounds
you barely get past your starters anyway, so "fill your lineup first" is not a
rule — it's a description of what happens regardless. Here the whole edge is
scarcity, which is why the lookahead is the only thing that helps and why
following ADP is catastrophic (−174, and your board's mock-derived replacement
levels are the reason it knows better).

---

## Is this an artifact of the injury assumption?

That was the load-bearing guess, so the whole bakeoff was re-run at three injury
levels: none at all, the base rates, and 1.5× the base rates.

| league | no injuries | base | high |
|---|---|---|---|
| 12-team | marginal > vona > starters > caps > board | *same* | *same* |
| 8-team | vona > starters > marginal > caps > board | *same* | *same* |
| 6-team | starters > vona > marginal > caps > board | *same* | *same* |
| 32-team | marginal > vona > starters > caps > board | marginal > vona > caps > starters > board | *same* |

The ordering is stable everywhere except a swap between the two policies that
are tied anyway in the 32-team league. **Best-available finishes last among the
model-based policies in all twelve runs.** The conclusion does not depend on the
injury model.

---

## Three things worth knowing beyond the headline

**1. Adding positional caps to best-available barely helps.** `caps` gained
+9.0, +4.4, +0.7 and +1.5. Capping is a rule about *counting*; the problem is
about *value*. Don't reach for "max 5 RBs" and think you've fixed it — you
haven't, and the sim says so in four leagues independently.

**2. The exact position counts hardly matter once you're lineup-aware.** Forcing
the good policy to a hard cap at each position, 12-team league:

| cap | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| QB | 1534 | **1539** | 1537 | | | |
| RB | | | 1530 | 1534 | **1538** | 1537 |
| WR | | | 1535 | 1537 | 1537 | **1537** |
| TE | 1525 | **1537** | 1537 | | | |

Almost flat. Two real signals: **taking only one tight end costs about 12
points** — you want a second one — and **stopping at three running backs costs
about 8.** Everything else is inside the noise. This is the useful news: you do
not have to hit a precise roster shape, you just have to stop doing the one bad
thing.

The 8-team and 6-team probes say the same thing in the same shape, which is the
part that makes it believable rather than a quirk of one pool: 8-team TE1 costs
9 (1591 vs 1600) and RB3 costs 8 (1593 vs 1601); 6-team RB3 costs 7 (1420 vs
1427). Quarterback is flat in every league — 1 / 2 / 3 differ by at most 4
points anywhere, so the "2 QBs" half of your complaint turns out to be
harmless. It is the running backs that were costing you.

**3. Your board is far more RB-heavy than the market, and that is what makes
best-available dangerous *here specifically*.** The ADP policy drafts RB3.4 /
WR7.7 in the 12-team league; your board's best-available drafts RB7.9 / WR2.5.
Those are nearly mirror images. Phase 13 already flagged this — running backs
went from 25 to 31 of the top 60 when depth-chart position was added, and the
note said to trust the model's *ordering* of backs but be skeptical of *how
many* it wants. This experiment is the cost of not being skeptical: taking them
in board order is what produces the eight-back roster.

---

## Addendum — can you just run VONA by hand?

`marginal` is not runnable by a human. It mock-drafts the rest of your roster
for every candidate, every pick. That is a computer's job.

So the follow-up question is the practical one: **is plain VONA good enough?**
Tested directly, 120 drafts each, against the same baseline:

| policy | 12-team | 32-team |
|---|---|---|
| **vona + QB/TE caps** | **+60.0** | +29.9 |
| vona + caps + starters-first | +59.7 | +30.4 |
| marginal (the expensive one) | +54.7 | **+40.7** |
| vona + starters-first, no caps | +54.4 | +21.9 |
| vona, raw | +49.4 | +21.6 |
| *best available* | *—* | *—* |

Two clean results.

**In the 12-team league, VONA with a cap actually BEATS the lookahead** — +60.0
against +54.7. The cheap rule you can run in your head is the best thing tested.
That is a good outcome and slightly embarrassing for the sophisticated policy.

**In the 32-team league it does not**, capturing +29.9 of the lookahead's +40.7,
so about three quarters. Superflex with 12 rounds is a harder problem and a
one-pick horizon leaves value on the table. Still far better than best-available.

**The cap is the part that matters, not the starters-first rule.** Compare rows
2 and 3 against rows 4 and 5: adding "fill your starters first" to VONA is worth
nothing on its own (+54.4 vs +49.4), while adding the caps is worth ten points.
Raw VONA's only real failure mode is quarterback hoarding — it drafts 5.5 of
them in a one-QB league, because it prices scarcity and is blind to the fact
that only one can start. Cap QB at 2 and TE at 2 and that failure disappears.

---

## What to actually do on draft day

**The rule: VONA, capped. Take the player with the biggest gap between his
points and the best man at his position you expect to still be there at your
next pick — never taking a 3rd QB or a 3rd TE.**

### How to run it on the clock

1. **Count the gap.** How many picks until your next turn? At 12-team pick 3
   your picks are #3, #22, #27, #46, #51, #70, #75, ~~#94~~, #99, #118, #123,
   #142, #147, #166. From #3 the gap is 19. From #22 it is only 5.
2. **For each position, find the survivor.** Run down the board in ADP order,
   skip the next *gap* players, and note the best guy left at QB, RB, WR, TE.
3. **Subtract.** VONA = your candidate's Adj PPG − that survivor's Adj PPG.
4. **Take the biggest number.** If several are within about a point of each
   other, nothing is scarce — take the board's top rank among them.
5. **Never break the cap.** Two QBs maximum, two TEs maximum.

### Worked example — 12-team, your first two picks

**Round 1, pick #3.** 19 players go before you pick again. Survivors expected at
#22: QB Allen 26.4, RB Skattebo 15.2, TE McBride 15.3, WR Nabers 14.8.

| VONA | pos | player | Adj PPG | board rank |
|---|---|---|---|---|
| **+5.4** | WR | Puka Nacua | 20.2 | 5 |
| +5.3 | RB | Jahmyr Gibbs | 20.5 | 1 |
| +4.3 | RB | Bijan Robinson | 19.5 | 2 |
| +3.9 | RB | Christian McCaffrey | 19.2 | 3 |
| +3.6 | RB | De'Von Achane | 18.8 | 4 |
| +3.4 | WR | Amon-Ra St. Brown | 18.2 | 7 |

The board says Gibbs, rank 1. VONA says Nacua, rank 5 — by a whisker, and the
whisker is the whole idea. Gibbs scores 0.3 more per game than Nacua, but if you
pass on running backs you still get Skattebo at 15.2, while if you pass on
receivers the best left is Nabers at 14.8. The receiver cliff is steeper, so the
receiver is worth marginally more. These two are close enough that either is
fine — what VONA is telling you is that Nacua is *not* the reach the board's
rank-5 makes him look like.

**Round 2, pick #22.** Only 5 picks until #27, so almost nothing changes hands.

| VONA | pos | player | Adj PPG | board rank |
|---|---|---|---|---|
| **+2.8** | QB | Josh Allen | 26.4 | 10 |
| +0.0 | RB | Cam Skattebo | 15.2 | 11 |
| +0.0 | TE | Trey McBride | 15.3 | 27 |
| +0.0 | WR | Malik Nabers | 14.8 | 31 |
| −0.1 | RB | Kyren Williams | 15.2 | 12 |

Allen is the only player on the board falling off a cliff — Lamar (23.7) is the
best quarterback surviving to #27, so Allen is worth +2.8 and everyone else is
worth zero. **Everyone else is worth zero because the gap is only 5 picks**:
Skattebo, McBride and Nabers will all still be sitting there in five picks, so
taking one now buys you nothing you can't have later.

That is the lesson to carry into the draft. **A short gap means take the cliff;
a long gap means take the best player.** Back-to-back picks at the turn are when
you can afford to wait; a 19-pick wait is when scarcity actually costs you.

And note what VONA does *not* say here: it does not say take a running back
because you have none. Roster need never enters it. That is fine — and it is
also exactly why the QB cap is not optional, because on this same logic VONA
would happily take Lamar at #27 too, and Mahomes after that.

### One rule for all four leagues

Extending the VONA test to the 8- and 6-team leagues collapses the per-league
advice into a single rule. **Capped VONA with starters-first wins or ties in
every league**, and it is the only policy that does:

| policy | 12-team | 8-team | 6-team | 32-team |
|---|---|---|---|---|
| **vona + caps + starters-first** | **+59.7** | **+44.3** | **+24.9** | **+30.4** |
| vona + caps only | +60.0 | +39.8 | +20.4 | +29.9 |
| vona + starters-first only | +54.4 | +44.7 | +24.7 | +21.9 |
| vona, raw | +49.4 | +37.6 | +20.2 | +21.6 |
| marginal (the lookahead) | +54.7 | +31.8 | +18.1 | +40.7 |
| *best available* | *—* | *—* | *—* | *—* |

Neither guard is sufficient alone and which one carries the weight flips by
league — caps do the work at 12 and 32 teams, starters-first does it at 8 and 6.
Using both is never worse than using the better one, so there is no reason to
tune it per league. Note also that the expensive lookahead is now beaten in
three leagues out of four; it survives only in the 32-team superflex.

### So: which positions do I fill first?

Fill your **starting lineup**, in whatever order VONA points at. That is
1 QB, 2 RB, 2 WR, 1 TE, plus one more RB/WR/TE for the flex — seven players.
(32-team: no dedicated QB slot, so it is 2 RB, 2 WR, 1 TE, a flex, and a
superflex — also seven.)

The test on the clock is one question: **if I take this guy, does he walk
straight into an empty starting slot, or does he sit on my bench?** If he'd sit
on the bench, he waits. That is the entire restriction, and it lifts the moment
all seven slots are covered.

A position therefore stays open longer than "one and done" — the flex is a real
slot and RB, WR and TE all compete for it:

| position | stays open until you have |
|---|---|
| QB | 1 |
| TE | 1 (2 if a TE takes your flex) |
| RB | 3 — two starters plus the flex; only 2 if a WR or TE took the flex first |
| WR | 3 — same reason |

So **WR, RB, RB, RB, TE, WR, QB is a completely legal opening**, and it is worth
walking through because it looks like it should not be:

| pick | take | holding before | rule allows | ok? |
|---|---|---|---|---|
| 1 | WR | — | QB RB TE WR | yes |
| 2 | RB | WR1 | QB RB TE WR | yes |
| 3 | RB | RB1 WR1 | QB RB TE WR | yes |
| 4 | **RB** | RB2 WR1 | QB RB TE WR | **yes** — he fills the FLEX |
| 5 | TE | RB3 WR1 | QB TE WR | yes |
| 6 | WR | RB3 WR1 TE1 | QB WR | yes |
| 7 | QB | RB3 WR2 TE1 | QB | yes |

After seven picks you have QB1 / RB3 / WR2 / TE1, which starts
QB–RB–RB–WR–WR–TE–FLEX with nothing wasted. The third back was legal because the
flex was empty; a **fourth** back at pick 5 would not have been, because by then
the flex was spoken for and he would have been the first man on your bench.

That is the difference between this and what the board was doing to you. It was
not the running backs. It was taking the *fourth, fifth and sixth* of them while
your second receiver slot and your tight end slot were still empty.

After those seven, drop the restriction and run plain VONA to the end of the
draft — still respecting the two caps, which never lift:

- **QB ≤ 2** (≤ 3 in the 32-team superflex)
- **TE ≤ 2**
- **No cap on RB or WR.**

That is the entire rule. There is no RB or WR limit because the probes found
none worth having: past 4 RBs and 4 WRs the curve is flat to within noise.

**A correction to your original instinct, which the numbers earned.** The
resulting rosters are still running-back heavy — capped VONA ends the 12-team
draft at RB6.5 / WR3.6 on average, *more* backs than the lookahead's 4.2, and it
scores five points higher. So five running backs was never the problem. Taking
them **before you had two receivers and a flex** was. Once your lineup is
covered, a seventh back is a perfectly good use of a bench pick.

---

## What this does not answer

- **It assumes the projections are right.** Everything here is measured in the
  board's own points-per-game. If a projection is wrong, this simulator will
  confidently draft the wrong player in a well-constructed way. Phase 13's
  holdout says the ordering is worth about 0.4 PPG over a flat guess at RB — real,
  and modest.
- **Opponents are more disciplined than a real room.** They follow ADP with
  noise and respect positional caps. Real drafters are worse than that, in ways
  that should make every policy here look slightly better, not worse.
- **The 12-team keeper set is partial.** Olave at a round-8 cost is exact; the
  four opponent keepers are removed from the pool but their forfeited rounds are
  drawn at random. That affects all policies identically, so it cannot move the
  comparison — it just makes the absolute point totals approximate.
- **It optimises points, not wins.** Points are the right primary metric and
  the head-to-head machinery exists in the file, but converting points into
  playoff odds needs a weekly-variance assumption this project has never
  measured.
