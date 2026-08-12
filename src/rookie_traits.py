"""
Phase 13.5b. The two feature families that are admissible against an
availability target, and the one table that will leak if you let it.

WHY THESE TWO AND NOTHING ELSE
------------------------------
`playing_time.fit_expected_games` predicts what fraction of a season a
drafted rookie actually plays, over the FULL drafted population --
including the 16.6% who never take a snap. That target makes almost every
feature in this project inadmissible, and the test is blunt:

    Does this feature exist for a player who never made a roster?

Landing spot, position competition, depth-chart rank, O-line continuity:
all no. Every one of them is measured from a week-1 roster snapshot, so
conditioning availability on them re-introduces, one level down, exactly
the survivorship Phase 13.5a was written to remove.

Two families pass:

  COMBINE MEASURABLES   Recorded in February, before anyone has a roster.
                        A player who runs a 4.38 and then never dresses
                        still has the 4.38.

  USAGE TENDENCY        A property of the team that DRAFTED him, which is
                        known the moment his name is called. Some staffs
                        play rookies immediately; some redshirt them.

That is why the combine work and the tendency work were scoped into one
phase: they are not two ideas, they are the two things left that are
allowed to be in the model.

THE TABLE THAT WILL LEAK
------------------------
`nfl.load_draft_picks()` is the bridge for both families, and it also
carries CAREER OUTCOME columns:

    games   seasons_started   car_av   w_av   dr_av   to
    receptions   rush_yards   pass_attempts   probowls   allpro   hof

**`games` is career games played.** It is very nearly the target with the
serial numbers filed off, and every column beside it is a summary of what
happened AFTER the draft. Nothing in this file may read them, and
`DRAFT_PICK_COLUMNS` below is an allowlist rather than a drop-list for
exactly that reason -- a new nflverse column should arrive excluded by
default, not included until somebody notices.

THE JOIN IS NOT ON gsis_id
--------------------------
The combine table has no `gsis_id`. It carries `pfr_id` and `cfb_id`, so
it has to bridge: combine.pfr_id -> draft_picks.pfr_player_id ->
draft_picks.gsis_id, which is what the rest of this project keys on. That
bridge is lossy -- a player missing from either side drops out -- so
`combine_features` reports its own match rate rather than assuming one.

COMBINE DATA IS INVITEES ONLY
-----------------------------
Not every drafted player is invited, and the ones who are not are not a
random sample -- invitation correlates with prospect status, which
correlates with the target. So `combine_missing` ships as a feature
alongside the measurables, and each measurable gets mean-imputation with
its own indicator. This is the same treatment `pos_rank` /
`depth_chart_missing` and `usage_trend_share` / `trend_missing` already
get, and for the same reason: dropping the rows would select on the thing
being predicted, and imputing without an indicator would let "did not run
the 3-cone" read as "ran an average 3-cone."

THE VERDICT (Aug 12): NEITHER FAMILY EARNS A SLOT
-------------------------------------------------
Both were tested and both were rejected, so this module ships DATA and a
record, not a feature. `fit_expected_games` still fits `pick` and nothing
else.

USAGE TENDENCY -- screened out before building. Permutation test, 2000
shuffles of the team label, on the sd of team mean availability:

    raw available_share            9.70%   p = 0.018
    after removing draft capital   6.84%   p = 0.153
    playcaller, same residual      8.53%   p = 0.136

The raw effect is real and it is draft capital wearing a franchise's
name. `pick` is already in the model.

COMBINE MEASURABLES -- tested out of sample and failed. `size` (mean of
within-position z-scores of ht and wt, chosen as ONE feature because
corr(ht, wt) = 0.71 and testing them separately doubles the multiplicity
for one underlying fact), leave-one-class-out binomial deviance, 300
bootstrap resamples:

    pos   pick only    +size      gain   95% CI            P(gain > 0)
    QB        707.6    673.2     +34.4   [ -43, +116]      73.3%
    RB       1653.9   1667.5     -13.6   [ -71,  +18]      11.0%
    WR       2122.9   2130.5      -7.6   [ -65,  +52]      23.3%
    TE        887.3    856.7     +30.7   [ -54, +120]      70.7%

Not one confidence interval excludes zero. QB and TE look encouraging and
are not distinguishable from noise at n = 104 and n = 130.

An in-sample screen said the same thing first: 5 of 31 feature x position
partial correlations reached p < 0.05 against 1.6 expected by chance, and
Benjamini-Hochberg at FDR 10% kept nothing. Four of those five were `ht`
and `wt` at QB and TE -- two correlated columns reporting one fact twice,
which is what motivated the composite.

WHAT THIS PHASE ACTUALLY ESTABLISHED. Availability is draft capital.
That is now a finding rather than an assumption: the two feature families
with a plausible mechanism and a clean source were tested, and the model
is short because the data is, not because nobody looked.

USAGE
-----
    python -m src.rookie_traits              # build + report coverage
    python -m src.rookie_traits --no-write
"""

import argparse
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "rookie_traits.csv"
PLAYCALLER_PATH = PROJECT_ROOT / "playcaller_history.csv"

# ALLOWLIST, NOT A DROP-LIST. See the leak warning in the module
# docstring: `load_draft_picks` ships career outcomes in the same frame as
# draft capital, and `games` is nearly the target. Anything not named here
# does not enter this pipeline, so a future nflverse column arrives
# excluded rather than included-until-noticed.
DRAFT_PICK_COLUMNS = [
    "season",
    "round",
    "pick",
    "team",
    "gsis_id",
    "pfr_player_id",
    "position",
]

# Combine measurables kept, in the order a scout would read them.
#
# `ht` and `wt` are size; `forty` is long speed; `bench` is upper-body
# strength; `vertical` and `broad_jump` are lower-body explosiveness;
# `cone` and `shuttle` are change of direction. They are NOT standardised
# by position here, because `fit_expected_games` fits one model per
# position -- a 4.55 forty is slow for a receiver and fast for a tight
# end, and fitting within position handles that without a z-score step
# that would need its own defence.
COMBINE_MEASURABLES = [
    "ht", "wt", "forty", "bench", "vertical", "broad_jump", "cone", "shuttle",
]

# A team's usage tendency is unusable below this many prior drafted
# rookies -- it becomes one player's injury luck wearing a franchise's
# name. Cells under the bar fall back to the league mean of the same fold.
#
# Deliberately higher than playing_time.MIN_CELL_N (8). That constant
# guards a MEAN OF THE TARGET in a position/round cell; this one guards a
# FEATURE that will be handed to a fitter, where a noisy value does not
# just read badly, it gets a coefficient.
MIN_TENDENCY_N = 12


def _draft_picks(seasons):
    """Draft picks, allowlisted columns only."""
    import nflreadpy as nfl

    picks = nfl.load_draft_picks()
    available = [c for c in DRAFT_PICK_COLUMNS if c in picks.columns]
    missing = set(DRAFT_PICK_COLUMNS) - set(available)
    if missing:
        print(f"   NOTE: draft_picks is missing {sorted(missing)} -- "
              f"nflverse may have renamed them.")
    return (
        picks.select(available)
        .filter(pl.col("season").is_in(list(seasons)))
        .rename({"team": "draft_team", "gsis_id": "player_id"})
    )


def combine_features(seasons):
    """
    Combine measurables per drafted player, keyed on `player_id`.

    Returns one row per player with the measurables, a per-measurable
    `<name>_missing` flag, and a single `combine_missing` flag for players
    with no combine row at all.
    """
    import nflreadpy as nfl

    picks = _draft_picks(seasons)
    combine = nfl.load_combine()

    if "pfr_id" not in combine.columns:
        raise ValueError(
            "load_combine() has no `pfr_id`, which is the only bridge to "
            "gsis_id -- the combine table carries no nflverse id of its own. "
            f"Columns present: {combine.columns}"
        )

    combine = (
        combine.select(["pfr_id", *[c for c in COMBINE_MEASURABLES
                                    if c in combine.columns]])
        .filter(pl.col("pfr_id").is_not_null())
        .unique(subset=["pfr_id"], keep="first")
    )

    # `ht` IS A STRING, WHATEVER THE DICTIONARY SAYS (Aug 12). The nflverse
    # combine dictionary types it "numeric" and describes it as "feet and
    # inches"; the actual column holds "6-2".
    #
    # This failed quietly rather than loudly, which is the part worth
    # recording. `fill_null(mean())` over a string column returns null in
    # polars, so height was silently left unimputed and would have entered
    # the fitter as a constant -- a feature that could never earn its slot
    # and would have looked like evidence that height does not matter.
    # Parsed to inches here, and it raises if the format changes.
    if "ht" in combine.columns and combine.schema["ht"] == pl.String:
        feet = pl.col("ht").str.split("-").list.get(0).cast(pl.Float64, strict=False)
        inches = pl.col("ht").str.split("-").list.get(1).cast(pl.Float64, strict=False)
        combine = combine.with_columns((feet * 12.0 + inches).alias("ht"))
        parsed = int(combine.select(pl.col("ht").is_not_null().sum()).item())
        if parsed == 0:
            raise ValueError(
                "`ht` is a string but no value parsed as feet-inches. The "
                "combine feed changed format; fix the parse rather than "
                "letting height ship as a constant."
            )

    joined = picks.join(
        combine, left_on="pfr_player_id", right_on="pfr_id", how="left"
    )

    present = [c for c in COMBINE_MEASURABLES if c in joined.columns]
    joined = joined.with_columns(
        pl.all_horizontal([pl.col(c).is_null() for c in present])
        .alias("combine_missing")
    )

    # Mean-impute each measurable WITHIN POSITION, with its own indicator.
    # Within position because the pooled mean of `forty` across QB, RB, WR
    # and TE is not a plausible value for any of them, and an implausible
    # imputation is a fabricated observation rather than a neutral one.
    for column in present:
        joined = joined.with_columns([
            pl.col(column).is_null().alias(f"{column}_missing"),
            pl.col(column)
            .fill_null(pl.col(column).mean().over("position"))
            .alias(column),
        ])

    matched = int(joined.select((~pl.col("combine_missing")).sum()).item())
    print(f"Combine: {matched} of {joined.height} drafted players matched "
          f"({matched / joined.height:.1%}). "
          f"Unmatched keep combine_missing=True and position-mean values.")
    for column in present:
        missing = int(joined.select(pl.col(f"{column}_missing").sum()).item())
        print(f"   {column:11s} missing {missing:4d} "
              f"({missing / joined.height:5.1%})")
    return joined


def usage_tendency(universe, fold_season=None, level="draft_team"):
    """
    How often this team's (or playcaller's) OTHER drafted rookies played.

    LEAKAGE IS THE WHOLE DIFFICULTY HERE, and it has two doors.

      1. The player himself. A tendency computed over a group that
         includes him is partly a copy of his own outcome. Excluded by
         leave-one-out within the group.

      2. His draft class. `fit_expected_games` is validated
         leave-one-class-out, so a tendency built from the FULL history
         lets the held-out season inform its own prediction through the
         team average. This is the subtler door and it is the one that
         makes a feature look brilliant in validation and do nothing on a
         board.

    Hence `fold_season`: when set, only seasons OTHER than that one
    contribute. This function must therefore be called INSIDE the fold
    loop, never once up front -- a precomputed tendency column is a leaked
    tendency column, however carefully the rest of the fit is done.

    `level="playcaller"` joins `playcaller_history.csv` on
    (season, draft_team). It is thin: 74 of 95 playcallers have fewer than
    MIN_TENDENCY_N rookies, and only 45% of players get a usable cell.

    SCREENED BEFORE BUILDING, AND IT DID NOT SURVIVE (Aug 12)
    --------------------------------------------------------
    Permutation test, 2000 shuffles of the team label, measuring the
    standard deviation of team mean availability:

        RAW available_share             observed 9.70%   p = 0.018
        AFTER removing draft capital    observed 6.84%   p = 0.153
        playcaller, same residual       observed 8.53%   p = 0.136

    The raw team effect is real and it is **draft capital wearing a
    franchise's name.** Teams whose rookies play more are teams that draft
    rookies higher, and `pick` is already the model's only feature. Once
    availability is residualised on pick within position, no team effect
    is detectable, and the playcaller cut is no better despite four times
    the resolution.

    Stated with the right strength: p = 0.153 is not proof of absence. It
    is 32 teams x 9 classes x ~22 players, and an effect smaller than
    about 3 points of share would not be visible here. The honest claim is
    **"no team effect detectable beyond draft capital at this sample
    size,"** which is a reason not to spend a gate run on it, not a proof
    that coaching staffs are interchangeable.

    The function is kept rather than deleted because the screen is a
    screen, not the gate -- but nothing should adopt it without re-reading
    this, and the collinearity with `pick` is the specific thing to check
    if it ever is adopted.
    """
    frame = universe
    if fold_season is not None:
        frame = frame.filter(pl.col("season") != fold_season)

    key = "draft_team"
    if level == "playcaller":
        frame = _attach_playcaller(frame)
        universe = _attach_playcaller(universe)
        key = "playcaller"

    league_mean = float(frame.select(pl.col("available_share").mean()).item())

    totals = frame.group_by(key).agg([
        pl.col("available_share").sum().alias("share_sum"),
        pl.len().alias("group_n"),
    ])

    out = universe.join(totals, on=key, how="left").with_columns([
        pl.col("share_sum").fill_null(0.0),
        pl.col("group_n").fill_null(0),
    ])

    # Leave-one-out inside the group. A player's own share comes out of
    # both numerator and denominator, so his tendency is genuinely about
    # everyone else.
    #
    # BRANCH IN PYTHON, NOT IN THE EXPRESSION. The first version wrote
    # `(fold_season is None) | (pl.col("season") != fold_season)`, which
    # builds `pl.col("season") != None` even when the left side is already
    # True -- polars evaluates both branches of `|` and a comparison
    # against None yields null, not False. It warned rather than failing,
    # which is worse: the null would have propagated into `loo_n` and
    # silently made every leave-one-out denominator wrong.
    if fold_season is None:
        own_included = pl.lit(True)
    else:
        own_included = pl.col("season") != fold_season
    loo_sum = pl.col("share_sum") - pl.when(own_included).then(
        pl.col("available_share")).otherwise(0.0)
    loo_n = pl.col("group_n") - pl.when(own_included).then(1).otherwise(0)

    return out.with_columns([
        pl.when(loo_n >= MIN_TENDENCY_N)
        .then(loo_sum / loo_n)
        .otherwise(pl.lit(league_mean))
        .alias(f"tendency_{level}"),
        (loo_n < MIN_TENDENCY_N).alias(f"tendency_{level}_thin"),
        loo_n.alias(f"tendency_{level}_n"),
    ]).drop(["share_sum", "group_n"])


def _attach_playcaller(frame):
    """Joins playcaller_history.csv on (season, draft_team)."""
    if "playcaller" in frame.columns:
        return frame
    history = pl.read_csv(PLAYCALLER_PATH).select([
        pl.col("season").cast(pl.Int64),
        pl.col("team").alias("draft_team"),
        "playcaller",
    ])
    joined = frame.join(history, on=["season", "draft_team"], how="left")
    return joined.with_columns(
        pl.col("playcaller").fill_null("UNKNOWN")
    )


def build(seasons=None, write=True):
    from src import rookie_backtest

    seasons = list(seasons or rookie_backtest.COHORT_SEASONS)

    print("PHASE 13.5b -- rookie traits")
    print("Reminder: nothing here may read draft_picks' career columns "
          "(games, car_av, seasons_started, ...). `games` is the target.\n")

    traits = combine_features(seasons)

    universe_path = PROJECT_ROOT / "data" / "playing_time_universe.csv"
    if universe_path.exists():
        universe = pl.read_csv(universe_path)
        merged = universe.join(
            traits.drop(["season", "round", "pick", "position"]),
            on="player_id", how="left",
        )
        for level in ("draft_team", "playcaller"):
            scored = usage_tendency(merged, fold_season=None, level=level)
            thin = int(scored.select(pl.col(f"tendency_{level}_thin").sum()).item())
            print(f"\nTendency ({level}): {scored.height - thin} of "
                  f"{scored.height} players get a real group estimate, "
                  f"{thin} fall back to the league mean "
                  f"(under n={MIN_TENDENCY_N}).")
            merged = scored
        traits = merged
        print("\nNOTE: the tendency columns above are the FULL-HISTORY version "
              "and exist for coverage reporting only.\n"
              "      fit_expected_games must call usage_tendency(fold_season=...) "
              "inside each fold.")

    if write:
        traits.write_csv(OUTPUT_PATH)
        print(f"\nWrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)} "
              f"({traits.height} rows)")
    return traits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    build(write=not args.no_write)


if __name__ == "__main__":
    main()
