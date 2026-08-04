"""
Phase 9 CP2 -- does the playcaller signal survive validation?

CP1 produced a table that looks encouraging: playcaller position-PPG has a
between-playcaller ICC of 0.15-0.40, and a split-half correlation of +0.18 to
+0.47. Under the original plan the next step was to shrink those means and fit
them. This module exists because that would have been the wrong next step.

The problem is that "playcaller" and "team" are nearly the same variable. A
playcaller who keeps his job for five years is measured on five seasons of one
roster. Any reliability we see could be the coach OR could be the roster, and
the CP1 table cannot tell them apart.

Three tests, in increasing order of how much they matter:

1. VARIANCE DECOMPOSITION -- corrects CP1's naive comparison. CP1 printed the
   raw sd of playcaller means against mean within-playcaller sd, which is not
   apples-to-apples: a playcaller mean over n seasons already has its noise
   divided by n, so the raw between figure understates true spread. Proper
   one-way ANOVA components fix that.

2. MOVERS TEST -- the confound check. Restrict to playcallers who worked for
   more than one team, and ask whether their effect at team A predicts their
   effect at team B. If the effect travels with the coach, this is positive.
   If it was the roster all along, it collapses.

3. PREDICTIVE TEST -- the only test the model cares about. For each team-season,
   compare predicting this year's position PPG from the team's own prior year
   against adding the playcaller's history from strictly prior seasons. If the
   playcaller term adds no R2 over what the team's own past already told us,
   it earns nothing.

Also tests the player-relative alternative: instead of measuring the position
ROOM, measure whether players beat THEIR OWN baseline under a given playcaller.
That design differences out roster talent, which is the whole confound, and it
matches the model's actual target (delta = actual - baseline). Better design;
tested here rather than assumed.

Run:  python -m src.playcaller_validate
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl

from src.playcaller_ppg import (
    HISTORY_SEASONS,
    POSITIONS,
    TEAM_SEASON_OUTPUT,
    load_playcaller_history,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKTEST_PATH = PROJECT_ROOT / "data" / "backtest_features.csv"


def load_team_season():
    if not TEAM_SEASON_OUTPUT.exists():
        raise FileNotFoundError(
            f"{TEAM_SEASON_OUTPUT} not found -- run `python -m src.playcaller_ppg` first."
        )
    return pl.read_csv(TEAM_SEASON_OUTPUT).filter(pl.col("playcaller").is_not_null())


def variance_decomposition(team_season):
    """
    One-way ANOVA components per position.

    var_between is the variance genuinely attributable to the playcaller;
    var_within is season-to-season noise inside one playcaller's tenure.
    ICC = var_between / (var_between + var_within) -- the share of total
    variation the playcaller identity accounts for.

    Note this measures "playcaller OR the roster he is attached to." It cannot
    separate them. That is test 2's job.
    """
    rows = []
    for position in POSITIONS:
        subset = team_season.filter(pl.col("position") == position)
        groups = [
            g["position_ppg"].to_numpy()
            for _, g in subset.group_by(["playcaller", "playcaller_role"])
        ]
        counts = np.array([len(g) for g in groups], dtype=float)
        means = np.array([g.mean() for g in groups])
        values = subset["position_ppg"].to_numpy()

        k, total_n = len(groups), len(values)
        grand = values.mean()

        ss_between = float((counts * (means - grand) ** 2).sum())
        ss_within = float(sum(((g - g.mean()) ** 2).sum() for g in groups))
        ms_between = ss_between / (k - 1)
        ms_within = ss_within / (total_n - k)

        # n0 = effective group size for unbalanced designs
        n0 = (total_n - (counts ** 2).sum() / total_n) / (k - 1)
        var_between = max((ms_between - ms_within) / n0, 0.0)
        icc = var_between / (var_between + ms_within)

        rows.append({
            "position": position,
            "n_playcallers": k,
            "naive_between_sd": float(means.std(ddof=1)),
            "true_between_sd": float(np.sqrt(var_between)),
            "within_sd": float(np.sqrt(ms_within)),
            "icc": icc,
        })
    return pl.DataFrame(rows)


def movers_test(team_season):
    """
    Same playcaller, different teams -- does the effect travel?

    Each playcaller's seasons are centred against that season's league mean
    (so a leaguewide scoring shift doesn't masquerade as a coach effect), then
    averaged per team. Correlating team-A effect against team-B effect asks
    directly: is this the coach, or the building?

    n is small -- only a handful of playcallers changed teams inside a 5-year
    window -- so a single position's result would be weak evidence. Four
    positions pointing the same way is considerably stronger.
    """
    centred = team_season.with_columns(
        (pl.col("position_ppg") - pl.col("position_ppg").mean().over(["season", "position"]))
        .alias("vs_league")
    )

    rows = []
    for position in POSITIONS:
        subset = centred.filter(pl.col("position") == position)
        pairs = []
        for (playcaller, _role), group in subset.group_by(["playcaller", "playcaller_role"]):
            by_team = group.group_by("team").agg(pl.col("vs_league").mean())
            if by_team.height >= 2:
                pairs += list(combinations(by_team["vs_league"].to_list(), 2))
        if len(pairs) >= 4:
            a = np.array([p[0] for p in pairs])
            b = np.array([p[1] for p in pairs])
            r = float(np.corrcoef(a, b)[0, 1])
        else:
            r = float("nan")
        rows.append({"position": position, "n_pairs": len(pairs), "r_across_teams": r})
    return pl.DataFrame(rows)


def predictive_test(team_season):
    """
    Out-of-sample-ish: for each team-season t, does the playcaller's mean over
    his seasons BEFORE t add explanatory power over that team's own t-1 value?

    Both predictors are built only from prior seasons, so this is an honest
    forward test rather than an in-sample fit. `r_team_vs_pc` is the tell: when
    it is high, the two predictors are largely the same information, because a
    playcaller who stayed put IS his team's own history.
    """
    rows = []
    for position in POSITIONS:
        subset = team_season.filter(pl.col("position") == position)
        records = subset.to_dicts()
        by_team_season = {(r["team"], r["season"]): r["position_ppg"] for r in records}

        actual, prior_team, prior_pc = [], [], []
        for record in records:
            season, team, caller = record["season"], record["team"], record["playcaller"]
            previous = by_team_season.get((team, season - 1))
            if previous is None:
                continue
            history = [
                r["position_ppg"] for r in records
                if r["playcaller"] == caller and r["season"] < season
            ]
            if not history:
                continue
            actual.append(record["position_ppg"])
            prior_team.append(previous)
            prior_pc.append(float(np.mean(history)))

        if len(actual) < 20:
            rows.append({"position": position, "n": len(actual)})
            continue

        y = np.array(actual)
        team_x = np.array(prior_team)
        pc_x = np.array(prior_pc)

        def r_squared(*predictors):
            design = np.column_stack([np.ones(len(y)), *predictors])
            beta, *_ = np.linalg.lstsq(design, y, rcond=None)
            residual = y - design @ beta
            return 1 - residual.var() / y.var(), beta

        r2_team, _ = r_squared(team_x)
        r2_both, beta = r_squared(team_x, pc_x)

        rows.append({
            "position": position,
            "n": len(y),
            "r2_team_only": r2_team,
            "r2_team_plus_playcaller": r2_both,
            "r2_gain": r2_both - r2_team,
            "playcaller_coef": float(beta[2]),
            "r_team_vs_pc": float(np.corrcoef(team_x, pc_x)[0, 1]),
        })
    return pl.DataFrame(rows)


def player_relative_test(min_players_per_cell=3):
    """
    The alternative design: measure the PLAYER, not the room.

    `delta` in backtest_features.csv is actual_ppg - baseline_ppg -- how a
    player did against his own recent norm. Averaging delta across every player
    on a team gives "did players beat their own baselines under this
    playcaller," which differences out roster talent by construction. It is
    also the model's literal target, so a signal here would drop straight into
    the existing fit.

    Split-half across seasons: if a playcaller reliably lifts players above
    their baselines, his mean delta in one season should predict another.

    Caveat REMOVED Aug 4. This test originally ran on 2023-25 only, because
    that was all backtest_features.csv covered, and Phase 9's cut was recorded
    as "a decision made on suggestive evidence, not conclusive evidence" for
    exactly that reason. The training window now covers 2021-25, so this reads
    five seasons with no code change -- it simply consumes whatever
    backtest_features.csv holds.

    That matters because the widened window has already overturned two
    small-sample conclusions (WR usage trend, cut at p=0.23, came back at
    p=0.034; position_competition_ppg, cut at p=0.77, came back at p=0.04).
    If this test still comes up empty on five seasons, the Phase 9 cut is
    materially better evidenced than when it was made.
    """
    if not BACKTEST_PATH.exists():
        print(f"{BACKTEST_PATH.name} not found -- skipping player-relative test.")
        return None

    backtest = pl.read_csv(BACKTEST_PATH)
    history = load_playcaller_history().select(
        ["season", "team", "playcaller", "playcaller_role"]
    )
    merged = backtest.join(history, on=["season", "team"], how="left")

    unmatched = merged.filter(pl.col("playcaller").is_null()).height
    if unmatched:
        print(f"NOTE: {unmatched} player-seasons had no playcaller match.")
    merged = merged.filter(pl.col("playcaller").is_not_null())

    rows = []
    for position in ["ALL"] + POSITIONS:
        subset = merged if position == "ALL" else merged.filter(pl.col("position") == position)
        cells = (
            subset.group_by(["playcaller", "season"])
            .agg([
                pl.col("delta").mean().alias("mean_delta"),
                pl.len().alias("n_players"),
            ])
            .filter(pl.col("n_players") >= min_players_per_cell)
        )
        pairs = []
        for _, group in cells.group_by("playcaller"):
            if group.height >= 2:
                pairs += list(combinations(group["mean_delta"].to_list(), 2))
        if len(pairs) >= 8:
            a = np.array([p[0] for p in pairs])
            b = np.array([p[1] for p in pairs])
            r = float(np.corrcoef(a, b)[0, 1])
        else:
            r = float("nan")
        rows.append({
            "position": position,
            "n_pairs": len(pairs),
            "split_half_r": r,
            "between_coach_sd": float(cells["mean_delta"].std() or float("nan")),
        })
    return pl.DataFrame(rows)


def coach_change_test():
    """
    Sanity check on the feature that already ships.

    `coach_changed` feeds continuity_score today. If a playcaller change moved
    outcomes, players on teams with a change should show a different mean delta
    than players on teams without one. Reported against the standard deviation,
    because a 0.3 PPG difference on a 4.0 sd is not a finding.
    """
    if not BACKTEST_PATH.exists():
        return None

    backtest = pl.read_csv(BACKTEST_PATH)
    history = load_playcaller_history().select(
        ["season", "team", "changed_from_prior_year"]
    )
    merged = backtest.join(history, on=["season", "team"], how="left")

    return (
        merged.filter(pl.col("changed_from_prior_year").is_not_null())
        .group_by(["position", "changed_from_prior_year"])
        .agg([
            pl.col("delta").mean().alias("mean_delta"),
            pl.col("delta").std().alias("sd_delta"),
            pl.len().alias("n"),
        ])
        .sort(["position", "changed_from_prior_year"])
    )


def main():
    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_width_chars(160)

    team_season = load_team_season().filter(pl.col("season").is_in(HISTORY_SEASONS))

    print("\n" + "=" * 72)
    print("PHASE 9 CP2 -- VALIDATION BEFORE SHRINKAGE")
    print("=" * 72)

    print("\n--- 1. Variance decomposition (corrects CP1's naive comparison) ---")
    print(variance_decomposition(team_season))
    print("Nonzero ICC means SOMETHING persists across a playcaller's seasons.")
    print("It does not yet mean that something is the playcaller.")

    print("\n--- 2. Movers test: does the effect travel with the coach? ---")
    print(movers_test(team_season))
    print("Positive r => the coach. Near zero or negative => the roster.")

    print("\n--- 3. Predictive test: does it beat the team's own prior year? ---")
    print(predictive_test(team_season))
    print("r2_gain is the feature's entire case. r_team_vs_pc near 1 means the")
    print("playcaller mean is mostly re-reading the team's own history.")

    relative = player_relative_test()
    if relative is not None:
        print("\n--- 4. Player-relative alternative (room talent differenced out) ---")
        print(relative)
        print("Positive split_half_r => playcallers reliably lift players above")
        print("their own baselines. This is the better-specified design, so a")
        print("null here is the more damaging of the two nulls.")

    changed = coach_change_test()
    if changed is not None:
        print("\n--- 5. Does the EXISTING coach_changed flag move anything? ---")
        print(changed)
        print("Compare mean_delta across changed True/False against sd_delta.")

    print("\n" + "=" * 72)
    print("Decision rule: ship the feature only if the movers test is positive")
    print("AND test 3 or test 4 shows real gain. Anything less is a team-quality")
    print("proxy wearing a coach's name.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
