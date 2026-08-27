"""
Charts for the Phase 11 findings -- replacement level, baseline
weighting, and shrinkage.

Companion to src/make_charts.py, which covers Phase 10. Split by phase
rather than merged because each reads different inputs and answers
different questions; merging them would produce one module that reloads
four files to draw eight unrelated pictures.

Same deliberate pandas/matplotlib choice as make_charts.py, for the same
reason: nothing imports from here, so the inconsistency with the project's
Polars convention is contained to a script that emits images.

USAGE
-----
    python -m src.make_charts_phase11

Inputs, and what happens if one is missing (each chart is skipped
individually, with a line saying so, rather than the run failing):

    data/player_features.csv            -> replacement_levels, shrinkage_effect
    data/shrinkage_sweep.csv            -> shrinkage_sweep
    data/baseline_scheme_comparison.csv -> baseline_schemes

Requires matplotlib (`pip install matplotlib`).
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHARTS_DIR = PROJECT_ROOT / "charts"
FEATURES_PATH = PROJECT_ROOT / "data" / "player_features.csv"
SWEEP_PATH = PROJECT_ROOT / "data" / "shrinkage_sweep.csv"
SCHEMES_PATH = PROJECT_ROOT / "data" / "baseline_scheme_comparison.csv"
CONFIGS = [
    ("12-team", PROJECT_ROOT / "league_config_12team.json", "#1565C0"),
    ("6-team", PROJECT_ROOT / "league_config_6team.json", "#C62828"),
    ("32-team SF", PROJECT_ROOT / "league_config_32team.json", "#2E7D32"),
]

POSITIONS = ["QB", "RB", "WR", "TE"]
POSITION_COLORS = {"QB": "#6A1B9A", "RB": "#2E7D32", "WR": "#1565C0", "TE": "#EF6C00"}

# Imported rather than re-declared (Aug 6). These were local copies of
# build_board's constants, which is fine right up until one of them
# changes -- and UNMODELED_SLOTS_PER_TEAM just did, from a hardcoded 2 to
# a function of roster_slots, because the 32-team league starts no kicker
# and no defense. A stale duplicate here would not error; it would draw a
# chart that quietly disagreed with the board it claims to mirror.
from src.build_board import FLEX_SPLIT, SUPERFLEX_SPLIT, unmodeled_slots_per_team

SHRINKAGE_K = 2

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def load_features():
    df = pd.read_csv(FEATURES_PATH)
    df["is_rookie"] = df["is_rookie"].astype(str).str.lower().eq("true")
    return df


def replacement_ranks(df, config):
    """Mirrors build_board.compute_replacement_ranks / compute_starter_ranks."""
    teams, rounds = config["num_teams"], config["total_rounds"]
    skill_picks = teams * rounds - teams * unmodeled_slots_per_team(config)

    drafted = df[df["adp"].notna()].sort_values("adp").head(skill_picks)
    counts = Counter(drafted["position"])

    slots = config["roster_slots"]
    flex = slots.get("FLEX", 0)
    superflex = slots.get("SUPERFLEX", 0)
    starters = {
        p: max(1, round(
            teams * slots.get(p, 0)
            + teams * flex * FLEX_SPLIT.get(p, 0.0)
            + teams * superflex * SUPERFLEX_SPLIT.get(p, 0.0)
        ))
        for p in POSITIONS
    }
    drafted_ranks = {p: max(counts.get(p, 0), starters[p]) for p in POSITIONS}
    return starters, drafted_ranks


def chart_replacement_levels(df):
    """
    The Phase 11 CP6 fix, drawn.

    Each panel is a position's production curve -- every player sorted by
    projection. The two markers are where replacement level sits under the
    old starter-slot rule and the new picks-drafted rule, for both
    leagues. The gap between them IS the bug: in the 6-team league the old
    rule put the quarterback bar at QB6, as if the 7th-best quarterback in
    football were unobtainable, when in fact QB7 through QB32 are all
    sitting on waivers.

    VOR is the vertical distance from a player down to his position's
    marker, so a marker sitting higher makes every player at that position
    worth less.
    """
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))

    configs = [(label, json.load(open(path)), color) for label, path, color in CONFIGS]

    for ax, position in zip(axes, POSITIONS):
        pool = (df[df["position"] == position]
                .sort_values("adjusted_fantasy_points_per_game", ascending=False)
                .reset_index(drop=True))
        depth = min(len(pool), 80)
        ax.plot(range(1, depth + 1),
                pool["adjusted_fantasy_points_per_game"].head(depth),
                color=POSITION_COLORS[position], lw=2, zorder=3)

        for offset, (label, config, color) in enumerate(configs):
            starters, drafted = replacement_ranks(df, config)
            for rank, style, name in ((starters[position], ":", "old: starters"),
                                      (drafted[position], "-", "new: drafted")):
                if rank > depth:
                    continue
                ppg = pool["adjusted_fantasy_points_per_game"].iloc[rank - 1]
                ax.axvline(rank, color=color, ls=style, lw=1.3, alpha=0.85, zorder=2)
                ax.annotate(f"{position}{rank}",
                            xy=(rank, ppg), xytext=(3, 8 + offset * 11),
                            textcoords="offset points", fontsize=7,
                            color=color, fontweight="bold")

        ax.set_title(position)
        ax.set_xlabel("player rank at position")
        if position == POSITIONS[0]:
            ax.set_ylabel("adjusted PPG")

    handles = [plt.Line2D([], [], color=c, ls="-", lw=1.5, label=f"{l}: drafted (new)")
               for l, _, c in CONFIGS]
    handles += [plt.Line2D([], [], color=c, ls=":", lw=1.5, label=f"{l}: starters (old)")
                for l, _, c in CONFIGS]
    fig.legend(handles=handles, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.10), frameon=False, fontsize=8)
    fig.suptitle(
        "Phase 11 CP6 — replacement level: last starter vs last player drafted",
        y=1.19, fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "replacement_levels.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote replacement_levels.png")


def chart_shrinkage_effect(df):
    """
    Two panels: the shrinkage weight curve, and what it did to the pool.

    Left is the mechanism -- confidence = games / (games + 2) -- with the
    live distribution of baseline sample sizes underneath it, so the curve
    can be read against how many players actually sit in its steep part.

    Right is every veteran's raw projection against his shrunk one. The
    diagonal is "unchanged"; distance below it is how much the model
    declined to believe. Points are coloured by sample size, which is the
    whole story: the far-below-diagonal points are all small samples.
    """
    veterans = df[~df["is_rookie"] & df["position"].isin(POSITIONS)].copy()
    veterans["move"] = (veterans["fantasy_points_per_game_shrunk"]
                        - veterans["fantasy_points_per_game"])

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.4))

    games = np.arange(0, 56)
    left.plot(games, games / (games + SHRINKAGE_K), color="#37474F", lw=2.2, zorder=3)
    left.set_ylim(0, 1.05)
    left.set_xlabel("games in the 3-year baseline window")
    left.set_ylabel("weight on the player's own record")
    left.set_title(f"Shrinkage weight at K={SHRINKAGE_K}")

    twin = left.twinx()
    twin.hist(veterans["games_played"].clip(upper=55), bins=28,
              color="#90A4AE", alpha=0.35, zorder=1)
    twin.set_ylabel("players", color="#607D8B")
    twin.grid(False)

    for label, gp in (("Mafah", 1), ("Skattebo", 8), ("Gibbs", 49)):
        weight = gp / (gp + SHRINKAGE_K)
        left.plot([gp], [weight], "o", color="#C62828", ms=5, zorder=4)
        left.annotate(f"{label} ({gp} gm)", xy=(gp, weight), xytext=(6, -10),
                      textcoords="offset points", fontsize=7.5, color="#C62828")

    scatter = right.scatter(
        veterans["fantasy_points_per_game"],
        veterans["fantasy_points_per_game_shrunk"],
        c=veterans["games_played"].clip(upper=51),
        cmap="viridis", s=13, alpha=0.75, zorder=3,
    )
    limit = float(veterans["fantasy_points_per_game"].max()) * 1.05
    right.plot([0, limit], [0, limit], color="#455A64", ls="--", lw=1, zorder=2)
    right.set_xlabel("raw baseline PPG")
    right.set_ylabel("shrunk baseline PPG")
    right.set_title("Every veteran, before vs after")
    fig.colorbar(scatter, ax=right, label="games in window")

    moved = veterans[veterans["move"] <= -0.5]
    fig.suptitle(
        f"Phase 11 CP5 — baseline shrinkage toward each position's 30th percentile "
        f"({len(moved)} veterans cut by 0.5+ PPG; QB excluded)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "shrinkage_effect.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote shrinkage_effect.png")


def chart_shrinkage_sweep():
    """
    Why K=2, with the decision rule drawn on rather than described.

    The shaded band is +/- 2 standard errors. A point whose error bar
    clears zero is a K that beat the incumbent by the margin the rule
    demanded BEFORE any of these numbers existed. The peak sitting in the
    interior rather than at an end is the third clause -- an optimum at
    the boundary would mean the range was wrong, not that the answer was
    the boundary.
    """
    sweep = pd.read_csv(SWEEP_PATH)
    if "paired_dmae_low" not in sweep.columns:
        print("  skipped shrinkage_sweep.png -- re-run `python -m src.shrinkage` "
              "to record the paired statistics")
        return

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2))

    for (anchor, form), group in sweep.groupby(["anchor", "form"]):
        group = group.sort_values("k")
        challengers = group[group["k"] > 0]
        label = f"{anchor} / {form}"

        left.errorbar(challengers["k"], challengers["paired_dmae_low"],
                      yerr=2 * challengers["paired_se_low"],
                      marker="o", ms=4, capsize=3, lw=1.6, label=label)
        right.plot(group["k"], group["mae_low"], marker="o", ms=4, lw=1.6, label=label)

    left.axhline(0, color="#455A64", lw=1)
    left.set_xlabel("K (prior strength, in games)")
    left.set_ylabel("paired ΔMAE vs K=0")
    left.set_title("Improvement on low-confidence players (±2 SE)")
    left.legend(fontsize=7.5, frameon=False)

    right.set_xlabel("K (prior strength, in games)")
    right.set_ylabel("MAE, low-confidence players")
    right.set_title("Absolute error on the same group")
    right.legend(fontsize=7.5, frameon=False)

    best = sweep[sweep["k"] > 0].nlargest(1, "paired_dmae_low")
    if len(best):
        k = int(best["k"].iloc[0])
        for ax in (left, right):
            ax.axvline(k, color="#C62828", ls="--", lw=1.2, alpha=0.8)
        left.annotate(f"adopted K={k}", xy=(k, float(best["paired_dmae_low"].iloc[0])),
                      xytext=(8, 10), textcoords="offset points",
                      fontsize=8, color="#C62828", fontweight="bold")

    fig.suptitle("Phase 11 CP5 — choosing the shrinkage strength",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "shrinkage_sweep.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote shrinkage_sweep.png")


def chart_baseline_schemes():
    """
    Phase 11 CP2/CP3: the five candidate season-weighting schemes, scored
    against the incumbent on the players any of them actually move.

    Bars are paired improvements with 2-SE whiskers. The dashed line is
    zero. A bar whose whisker clears the line beat the pre-committed bar;
    everything else is a scheme that looked promising and wasn't.

    Split by position on the right because clause (b) of the rule was
    about exactly that -- a scheme winning overall while losing at one
    position is picking up something positional, not something about
    injuries, and belongs in the per-position weights instead.
    """
    detail = pd.read_csv(SCHEMES_PATH)
    incumbent = "recency"

    playable = detail[detail["actual_games_played"] >= 8]
    affected = playable[(playable["thin_seasons"] > 0)
                        | (playable["seasons_present"] < 3)]

    base = (affected[affected["scheme"] == incumbent]
            .set_index(["player_id", "season"])["abs_error"])

    def paired(frame):
        joined = frame.set_index(["player_id", "season"])
        improvement = base.reindex(joined.index) - joined["abs_error"]
        improvement = improvement[improvement.abs() > 1e-9].dropna()
        if len(improvement) < 2:
            return 0.0, 0.0
        return float(improvement.mean()), float(improvement.std() / np.sqrt(len(improvement)))

    schemes = [s for s in affected["scheme"].unique() if s != incumbent]
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.2),
                                      gridspec_kw={"width_ratios": [1, 1.5]})

    stats = [paired(affected[affected["scheme"] == s]) for s in schemes]
    deltas = [d for d, _ in stats]
    errors = [2 * se for _, se in stats]
    colors = ["#2E7D32" if d - 2 * se > 0 else "#B0BEC5"
              for (d, se) in stats]

    left.barh(schemes, deltas, xerr=errors, color=colors, capsize=4)
    left.axvline(0, color="#455A64", lw=1, ls="--")
    left.set_xlabel("paired ΔMAE vs recency (±2 SE)")
    left.set_title("All affected players")

    width = 0.8 / len(schemes)
    for i, scheme in enumerate(schemes):
        values, bars = [], []
        for j, position in enumerate(POSITIONS):
            subset = affected[(affected["scheme"] == scheme)
                              & (affected["position"] == position)]
            delta, se = paired(subset)
            values.append(delta)
            bars.append(2 * se)
        offsets = np.arange(len(POSITIONS)) + i * width - 0.4 + width / 2
        right.bar(offsets, values, width, yerr=bars, capsize=2, label=scheme)

    right.axhline(0, color="#455A64", lw=1, ls="--")
    right.set_xticks(range(len(POSITIONS)))
    right.set_xticklabels(POSITIONS)
    right.set_ylabel("paired ΔMAE vs recency")
    right.set_title("By position — clause (b): no scheme may go the wrong way")
    right.legend(fontsize=7.5, frameon=False, ncol=2)

    fig.suptitle("Phase 11 CP2 — which season-weighting scheme predicts best "
                 "(green = cleared the pre-committed 2-SE bar)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "baseline_schemes.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote baseline_schemes.png")


def main():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing Phase 11 charts to {CHARTS_DIR}")

    if FEATURES_PATH.exists():
        features = load_features()
        chart_replacement_levels(features)
        if "fantasy_points_per_game_shrunk" in features.columns:
            chart_shrinkage_effect(features)
        else:
            print("  skipped shrinkage_effect.png -- no shrunk baseline; "
                  "re-run `python -m src.pipeline`")
    else:
        print(f"  skipped 2 charts -- {FEATURES_PATH.name} not found")

    if SWEEP_PATH.exists():
        chart_shrinkage_sweep()
    else:
        print(f"  skipped shrinkage_sweep.png -- {SWEEP_PATH.name} not found; "
              f"run `python -m src.shrinkage`")

    if SCHEMES_PATH.exists():
        chart_baseline_schemes()
    else:
        print(f"  skipped baseline_schemes.png -- {SCHEMES_PATH.name} not found; "
              f"run `python -m src.baseline_weighting --cp2`")


if __name__ == "__main__":
    main()
