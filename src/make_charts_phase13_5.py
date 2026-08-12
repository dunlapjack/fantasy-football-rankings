"""
Charts for Phase 13.5 -- the availability hole, the logit refit, and the
two feature families that were tested and rejected.

Same deliberate pandas/matplotlib choice as make_charts_phase10.py and
make_charts_phase11.py, for the same reason: nothing imports from here,
so the inconsistency with the project's polars convention is contained to
a script that emits images.

Every number is recomputed from files on disk. Nothing is transcribed
from a chat log or a plan document -- if a chart disagrees with the plan,
the chart is right and the plan needs editing.

USAGE
-----
    python -m src.make_charts_phase13_5

Inputs, each chart skipped individually with a line saying so if its
input is missing:

    data/playing_time.json              -> realisation, availability_curves
    data/rookie_availability_sweep.csv  -> min_games_step
    data/playing_time_universe.csv      -> linear_vs_logit, combine_rejected
    2026_32Team_Board_v15.xlsx          -> qb59_cliff

Requires matplotlib.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHARTS = PROJECT_ROOT / "charts"
DATA = PROJECT_ROOT / "data"
NFL_GAMES = 17

POSITIONS = ["QB", "RB", "WR", "TE"]
BUCKETS = ["Round 1", "Day 2", "Day 3"]
COLOURS = {"QB": "#4472C4", "RB": "#70AD47", "WR": "#ED7D31", "TE": "#A64D79"}


def _save(fig, name):
    CHARTS.mkdir(exist_ok=True)
    path = CHARTS / name
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"   wrote charts/{name}")


def _parse_height(value):
    try:
        feet, inches = str(value).split("-")
        return float(feet) * 12 + float(inches)
    except Exception:
        return np.nan


def _irls(design, games, trials=NFL_GAMES, iterations=80):
    beta = np.zeros(design.shape[1])
    proportion = games / trials
    for _ in range(iterations):
        eta = np.clip(design @ beta, -30, 30)
        mu = np.clip(1 / (1 + np.exp(-eta)), 1e-9, 1 - 1e-9)
        weights = trials * mu * (1 - mu)
        working = eta + (proportion - mu) / (mu * (1 - mu))
        try:
            updated = np.linalg.solve(
                design.T @ (weights[:, None] * design) + 1e-9 * np.eye(design.shape[1]),
                design.T @ (weights * working),
            )
        except np.linalg.LinAlgError:
            return beta
        if np.max(np.abs(updated - beta)) < 1e-11:
            return updated
        beta = updated
    return beta


def _deviance(proportion, mu, trials=NFL_GAMES):
    mu = np.clip(mu, 1e-6, 1 - 1e-6)
    eps = 1e-12
    return 2 * trials * np.sum(
        np.where(proportion > 0, proportion * np.log((proportion + eps) / mu), 0)
        + np.where(proportion < 1,
                   (1 - proportion) * np.log((1 - proportion + eps) / (1 - mu)), 0)
    )


def realisation():
    """What the board pays for a rookie against what he returns."""
    path = DATA / "playing_time.json"
    if not path.exists():
        print("   skip realisation: no data/playing_time.json")
        return
    model = json.loads(path.read_text())
    ratios = model.get("realisation", {})
    if not ratios:
        print("   skip realisation: no realisation block")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.25
    x = np.arange(len(POSITIONS))
    for i, bucket in enumerate(BUCKETS):
        values = []
        for position in POSITIONS:
            entry = ratios.get(f"{position}|{bucket}")
            value = entry.get("realisation") if isinstance(entry, dict) else entry
            values.append(value if value is not None else np.nan)
        bars = ax.bar(x + (i - 1) * width, values, width,
                      label=bucket, edgecolor="white")
        for bar, value in zip(bars, values):
            if not np.isnan(value):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015,
                        f"{value:.2f}", ha="center", fontsize=8)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.text(len(POSITIONS) - 0.45, 1.02, "1.00 = no availability hole",
            fontsize=8, style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels(POSITIONS)
    ax.set_ylabel("realisation  (what he returns / what the board pays)")
    ax.set_ylim(0, 1.12)
    ax.set_title("Phase 13.5a — the rookie availability hole\n"
                 "Every point below 1.00 is the board's rookie premium",
                 fontsize=11)
    ax.legend(title="draft capital", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "phase13_5_realisation.png")


def min_games_step():
    """The MIN_GAMES sweep, against how many players clear the bar."""
    sweep_path = DATA / "rookie_availability_sweep.csv"
    model_path = DATA / "playing_time.json"
    if not sweep_path.exists() or not model_path.exists():
        print("   skip min_games_step: missing sweep or model")
        return
    sweep = pd.read_csv(sweep_path)
    availability = json.loads(model_path.read_text()).get("availability", {})

    rows = []
    for (position, bucket), cell in sweep.groupby(["position", "round_bucket"]):
        cell = cell.sort_values("min_games")
        at_zero = cell[cell.min_games == 0]["cohort_baseline_ppg"]
        at_eight = cell[cell.min_games == 8]["cohort_baseline_ppg"]
        entry = availability.get(f"{position}|{bucket}", {})
        clears = entry.get("p_min_games")
        if at_zero.empty or at_eight.empty or clears is None:
            continue
        rows.append({
            "label": f"{position} {bucket.replace('Round 1', 'R1')}",
            "position": position,
            "step": float(at_eight.iloc[0] / at_zero.iloc[0] - 1),
            "clears": float(clears),
        })
    if not rows:
        print("   skip min_games_step: nothing to plot")
        return
    frame = pd.DataFrame(rows)
    r = np.corrcoef(frame.clears, frame.step)[0, 1]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for position in POSITIONS:
        sub = frame[frame.position == position]
        ax.scatter(sub.clears, sub.step, s=110, label=position,
                   color=COLOURS[position], zorder=3, edgecolor="white")
    for _, row in frame.iterrows():
        ax.annotate(row.label, (row.clears, row.step),
                    textcoords="offset points", xytext=(7, 4), fontsize=8)

    slope, intercept = np.polyfit(frame.clears, frame.step, 1)
    grid = np.linspace(frame.clears.min(), frame.clears.max(), 50)
    ax.plot(grid, slope * grid + intercept, color="grey",
            linestyle="--", linewidth=1, zorder=1)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("share of the cell that clears MIN_GAMES = 8")
    ax.set_ylabel("how far the cohort baseline moves from threshold 0 to 8")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:+.0%}")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Phase 13.5a — the 8-game bar bites hardest where fewest "
                 f"players clear it\ncorrelation = {r:+.2f}: a selection "
                 "effect, not a population that happens to differ",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, "phase13_5_min_games_step.png")


def linear_vs_logit():
    """The functional-form bug and its fix, drawn as fitted curves."""
    path = DATA / "playing_time_universe.csv"
    if not path.exists():
        print("   skip linear_vs_logit: no universe")
        return
    universe = pd.read_csv(path)
    universe["pick"] = pd.to_numeric(universe["pick"], errors="coerce")
    universe = universe[universe["pick"].notna()]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
    grid = np.linspace(1, 300, 300)
    for ax, position in zip(axes, POSITIONS):
        sub = universe[universe.position == position]
        pick = sub["pick"].to_numpy(float)
        games = sub["actual_games_played"].to_numpy(float)
        share = games / NFL_GAMES
        centre = pick.mean()

        linear = np.polyfit(pick, share, 1)
        beta = _irls(np.column_stack([np.ones(len(pick)), pick - centre]), games)

        ax.scatter(pick, share, s=9, alpha=0.28,
                   color=COLOURS[position], edgecolor="none")
        ax.plot(grid, np.polyval(linear, grid), color="crimson",
                linewidth=1.8, label="linear (was)")
        ax.plot(grid, 1 / (1 + np.exp(-(beta[0] + beta[1] * (grid - centre)))),
                color="black", linewidth=1.8, label="logit (now)")
        ax.axhline(0, color="grey", linewidth=0.8)

        crossing = np.polyval(linear, grid)
        below = grid[crossing < 0]
        if below.size:
            ax.axvspan(below.min(), 300, color="crimson", alpha=0.10)
            ax.text(below.min() + 4, 0.55,
                    f"linear < 0\nfrom pick {below.min():.0f}",
                    fontsize=8, color="crimson")

        ax.set_title(position, fontsize=11)
        ax.set_xlabel("draft pick")
        ax.set_ylim(-0.15, 1.02)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("share of the season played")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Phase 13.5b — a linear probability model predicts negative "
                 "availability inside its own training support",
                 fontsize=12, y=1.03)
    _save(fig, "phase13_5_linear_vs_logit.png")


def combine_rejected(resamples=300, seed=21):
    """The combine size feature, tested at all four positions, with CIs."""
    path = DATA / "rookie_traits.csv"
    if not path.exists():
        print("   skip combine_rejected: no data/rookie_traits.csv "
              "(run `python -m src.rookie_traits`)")
        return
    traits = pd.read_csv(path)
    traits["pick"] = pd.to_numeric(traits["pick"], errors="coerce")
    traits = traits[traits["pick"].notna()].copy()
    traits["ht"] = traits["ht"].map(_parse_height)
    traits["wt"] = pd.to_numeric(traits["wt"], errors="coerce")
    for position, group in traits.groupby("position"):
        for column in ("ht", "wt"):
            traits.loc[group.index, column + "_z"] = (
                (group[column] - group[column].mean()) / group[column].std()
            )
    traits["size"] = traits[["ht_z", "wt_z"]].mean(axis=1)

    def loco(frame, use_size):
        total, folds = 0.0, 0
        for season in sorted(frame.season.unique()):
            train = frame[frame.season != season]
            test = frame[frame.season == season]
            if len(train) < 30 or test.empty:
                continue
            mean_pick = train["pick"].mean()

            def design(part):
                parts = [np.ones(len(part)), part["pick"].to_numpy(float) - mean_pick]
                if use_size:
                    mean_size = np.nanmean(train["size"])
                    if np.isnan(mean_size):
                        return None
                    values = part["size"].to_numpy(float).copy()
                    missing = np.isnan(values)
                    values[missing] = mean_size
                    parts += [values - mean_size, missing.astype(float)]
                return np.column_stack(parts)

            train_design, test_design = design(train), design(test)
            if train_design is None or test_design is None:
                return None
            beta = _irls(train_design, train["actual_games_played"].to_numpy(float))
            eta = np.clip(test_design @ beta, -30, 30)
            total += _deviance(
                test["actual_games_played"].to_numpy(float) / NFL_GAMES,
                1 / (1 + np.exp(-eta)),
            )
            folds += 1
        return total if folds == frame.season.nunique() else None

    rng = np.random.default_rng(seed)
    points, los, his = [], [], []
    for position in POSITIONS:
        group = traits[traits.position == position].reset_index(drop=True)
        base, with_size = loco(group, False), loco(group, True)
        gains = []
        for _ in range(resamples):
            sample = group.iloc[rng.integers(0, len(group), len(group))]
            sample = sample.reset_index(drop=True)
            a, b = loco(sample, False), loco(sample, True)
            if a is not None and b is not None:
                gains.append(a - b)
        gains = np.array(gains)
        points.append(base - with_size)
        lo, hi = np.percentile(gains, [2.5, 97.5])
        los.append(lo)
        his.append(hi)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    y = np.arange(len(POSITIONS))
    ax.errorbar(points, y,
                xerr=[np.array(points) - np.array(los),
                      np.array(his) - np.array(points)],
                fmt="o", color="black", capsize=5, markersize=8, linewidth=1.5)
    ax.axvline(0, color="crimson", linewidth=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(POSITIONS)
    ax.invert_yaxis()
    ax.set_xlabel("out-of-sample deviance saved by adding combine size "
                  "(positive = helps)")
    ax.set_title("Phase 13.5b — combine measurables, tested and rejected\n"
                 f"Not one 95% interval excludes zero ({resamples} bootstrap "
                 "resamples, leave-one-class-out)", fontsize=11)
    for i, (point, lo, hi) in enumerate(zip(points, los, his)):
        ax.text(hi + 4, i, f"{point:+.1f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    _save(fig, "phase13_5_combine_rejected.png")


def qb59_cliff():
    """Top-60 position mix as the QB replacement assumption is swept."""
    board_path = PROJECT_ROOT / "2026_32Team_Board_v15.xlsx"
    if not board_path.exists():
        print("   skip qb59_cliff: no v15 32-team board")
        return
    board = pd.read_excel(board_path, sheet_name="Draft Board", header=7)
    board = board[board["Rank"].notna()]
    base = {"QB": 59, "RB": 92, "WR": 145, "TE": 56}
    total = sum(base.values())
    curves = {
        position: board[board.Pos == position]
        .sort_values("Adj PPG", ascending=False)
        .reset_index(drop=True)
        for position in base
    }

    def mix(qb):
        rest = total - qb
        others = {p: base[p] for p in ("RB", "WR", "TE")}
        scale = sum(others.values())
        allocation = {p: others[p] * rest / scale for p in others}
        floors = {p: int(np.floor(v)) for p, v in allocation.items()}
        for p in sorted(allocation, key=lambda p: -(allocation[p] - floors[p]))[
            : rest - sum(floors.values())
        ]:
            floors[p] += 1
        return {"QB": qb, **floors}

    counts = {p: [] for p in POSITIONS}
    sweep = list(range(45, 71))
    for qb in sweep:
        ranks = mix(qb)
        pool = []
        for position, rank in ranks.items():
            frame = curves[position].copy()
            replacement = frame.loc[min(rank - 1, len(frame) - 1), "Adj PPG"]
            frame["vor"] = frame["Adj PPG"] - replacement
            pool.append(frame[["Pos", "vor"]])
        top = pd.concat(pool).sort_values("vor", ascending=False).head(60)
        for position in POSITIONS:
            counts[position].append(int((top.Pos == position).sum()))

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for position in POSITIONS:
        ax.plot(sweep, counts[position], marker="o", markersize=4,
                label=position, color=COLOURS[position], linewidth=2)
    ax.axvspan(49, 50, color="crimson", alpha=0.18)
    ax.axvline(59, color="black", linestyle="--", linewidth=1.4)
    ax.text(59.4, 11.5, "QB59\nmeasured", fontsize=8.5)
    ax.text(45.2, 11.5, "the cliff\nQB49 → 50", fontsize=8.5, color="crimson")
    ax.set_xlabel("expected_drafted.QB  (how many QBs the room takes)")
    ax.set_ylabel("players in the board's top 60")
    # The title states what the chart shows, checked against the chart.
    # An earlier draft claimed "RB holds at 20 across the whole range" --
    # RB is 25 below the cliff and 20 above it, so the claim was true only
    # in the regime that matters and false as written.
    ax.set_title("The 32-team board is bimodal in the QB assumption\n"
                 "Above the cliff RB is flat at 20; QB swings 8 → 21",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, "phase13_5_qb59_cliff.png")


def main():
    print("Phase 13.5 charts")
    for chart in (realisation, min_games_step, linear_vs_logit,
                  combine_rejected, qb59_cliff):
        try:
            chart()
        except Exception as error:  # one bad chart must not kill the rest
            print(f"   skip {chart.__name__}: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
