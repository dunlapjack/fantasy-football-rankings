"""
Charts for the Phase 10 findings -- what shipped, what didn't, and why.

WHY PANDAS AND NOT POLARS
-------------------------
Every other module here uses Polars. This one doesn't, deliberately: it
is a side utility that has to run anywhere (including environments
without the full project stack), and matplotlib speaks pandas natively.
Nothing downstream imports from this file, so the inconsistency is
contained to one script that produces images.

USAGE
-----
    python -m src.make_charts

Reads data/backtest_features.csv, writes PNGs to charts/. Regenerates
from whatever data is currently there, so re-running after the backtest
window is extended will refresh every chart.

Requires matplotlib (`pip install matplotlib`).
"""

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKTEST_PATH = PROJECT_ROOT / "data" / "backtest_features.csv"
CHARTS_DIR = PROJECT_ROOT / "charts"

MIN_GAMES = 8
POSITIONS = ["RB", "WR", "TE"]
POSITION_COLORS = {"RB": "#2E7D32", "WR": "#1565C0", "TE": "#EF6C00"}

# Controls held constant when isolating one feature, matching the
# specs in fit_weights.FEATURE_SPECS.
#
# These deliberately EXCLUDE age, because age is the variable of
# interest in the aging charts.
CONTROLS = {
    "RB": ["continuity_score", "workload_share"],
    "WR": ["team_changed", "workload_share", "recent_major_injury"],
    "TE": ["workload_share"],
}

# For the usage-trend chart, age MUST be held constant too -- it is in
# every shipped spec. Leaving it out is not a cosmetic simplification:
# older receivers tend to be losing snaps, so without age in the
# controls the WR panel shows a trend effect at p=0.03 that the actual
# model puts at p=0.23 and cuts. A chart that disagrees with the model
# it is illustrating is worse than no chart.
TREND_CONTROLS = {pos: controls + ["age"] for pos, controls in CONTROLS.items()}

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# --------------------------------------------------------------------
# minimal OLS -- avoids a statsmodels dependency in a plotting script
# --------------------------------------------------------------------

def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = FPMIN if abs(d) < FPMIN else d
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = FPMIN if abs(d) < FPMIN else d
        c = 1.0 + aa / c
        c = FPMIN if abs(c) < FPMIN else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = FPMIN if abs(d) < FPMIN else d
        c = 1.0 + aa / c
        c = FPMIN if abs(c) < FPMIN else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def _t_pvalue(t, df):
    """Two-sided p-value for Student's t."""
    t, a, b = abs(float(t)), 0.5 * df, 0.5
    x = df / (df + t * t)
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    bt = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def ols(X, y):
    """Returns (betas, standard errors, p-values) with intercept first."""
    n = len(y)
    X1 = np.column_stack([np.ones(n), X])
    k = X1.shape[1]
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    dof = n - k
    se = np.sqrt(np.diag(np.linalg.pinv(X1.T @ X1)) * (resid @ resid / dof))
    return beta, se, [_t_pvalue(t, dof) for t in beta / se]


def load():
    df = pd.read_csv(BACKTEST_PATH)
    for c in ["qb_changed", "coach_changed", "team_changed",
              "recent_major_injury", "trend_missing"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower().eq("true").astype(float)
    df["continuity_score"] = df.qb_changed + df.coach_changed
    return df[df.actual_games_played >= MIN_GAMES].copy()


def _partial(frame, feature, controls):
    """
    Residualizes both the outcome and `feature` against the controls, so
    the scatter shows the feature's own contribution rather than a
    picture confounded by everything else.
    """
    sub = frame.dropna(subset=controls + ["delta", feature]).copy()
    if len(sub) < 20:
        return None, None, None
    C = sub[controls].values.astype(float)
    C1 = np.column_stack([np.ones(len(sub)), C])
    resid = lambda v: v - C1 @ np.linalg.lstsq(C1, v, rcond=None)[0]
    return resid(sub[feature].values.astype(float)), resid(sub.delta.values), sub


# --------------------------------------------------------------------
# charts
# --------------------------------------------------------------------

def chart_age_curves(df):
    """How much a player falls off per year of age, by position."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, pos in zip(axes, POSITIONS):
        p = df[(df.position == pos)].dropna(subset=["age", "delta"] + CONTROLS[pos])
        color = POSITION_COLORS[pos]

        bins = [21, 23, 25, 27, 29, 31, 40]
        labels, means, errs, counts = [], [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            b = p[(p.age >= lo) & (p.age < hi)]
            if len(b) < 5:
                continue
            labels.append(lo + (min(hi, 34) - lo) / 2)
            means.append(b.delta.mean())
            errs.append(b.delta.std() / np.sqrt(len(b)))
            counts.append(len(b))
        ax.errorbar(labels, means, yerr=errs, fmt="o", color=color,
                    capsize=3, markersize=6, label="actual, grouped by age")
        for x, y, n in zip(labels, means, counts):
            ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                        xytext=(0, 11), ha="center", fontsize=6.5, color="#555")

        xs = np.linspace(p.age.min(), p.age.max(), 50)
        for label, mask, style in [
            ("what the model uses (one straight line)", p.age > 0, "-"),
            ("under 29 only", p.age < 29, ":"),
            ("29 and older only", p.age >= 29, "--"),
        ]:
            q = p[mask]
            if len(q) < 25:
                continue
            X = np.column_stack([q[CONTROLS[pos]].values.astype(float),
                                 q.age.values - q.age.mean()])
            beta, _, _ = ols(X, q.delta.values)
            slope = beta[-1]
            centre = q.age.mean()
            base = q.delta.mean()
            lo, hi = (xs.min(), xs.max()) if style == "-" else (
                (xs.min(), 29) if "under" in label else (29, xs.max()))
            seg = np.linspace(lo, hi, 20)
            ax.plot(seg, base + slope * (seg - centre), style, color=color,
                    linewidth=1.8 if style == "-" else 1.3,
                    label=f"{label}: {slope:+.2f} PPG/yr")

        ax.axhline(0, color="#999", linewidth=0.8)
        ax.set_title(f"{pos}")
        ax.set_xlabel("age at start of season")
        ax.legend(fontsize=6.5, loc="lower left")
    axes[0].set_ylabel("points per game vs. his own\nrecent form  (0 = held serve)")
    fig.suptitle("Aging: how far a player lands from his own recent form, by age\n"
                 "One straight line fits well at every position. A steeper drop after 29 "
                 "appeared on three seasons of data and vanished on five --\nit was a small "
                 "sample, not a cliff.",
                 fontsize=10.5, y=1.08)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "age_curves.png", bbox_inches="tight")
    plt.close(fig)


def chart_usage_trend(df):
    """Rising vs falling usage -- real for RB/TE, nothing for WR."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, pos in zip(axes, POSITIONS):
        p = df[df.position == pos]
        x, y, sub = _partial(p, "usage_trend_share", TREND_CONTROLS[pos])
        if x is None:
            continue
        color = POSITION_COLORS[pos]
        ax.scatter(x * 100, y, s=13, alpha=0.45, color=color, edgecolors="none")

        beta, se, pv = ols(x.reshape(-1, 1), y)
        xs = np.linspace((x * 100).min(), (x * 100).max(), 20)
        ax.plot(xs, beta[0] + beta[1] * xs / 100, color="black", linewidth=1.8)

        verdict = "REAL SIGNAL -- used" if pv[1] < 0.10 else "NO SIGNAL -- not used"
        box = "#2E7D32" if pv[1] < 0.10 else "#B71C1C"
        ax.set_title(f"{pos}   {verdict}", color=box, fontweight="bold")
        ax.annotate(f"{beta[1] * 0.01:+.2f} PPG per 1 point of share gained per season"
                    f"\n(p = {pv[1]:.3f}, n = {len(sub)})",
                    xy=(0.03, 0.03), xycoords="axes fraction", fontsize=7,
                    bbox=dict(boxstyle="round", fc="white", ec=box, alpha=0.9))
        ax.axhline(0, color="#999", linewidth=0.8)
        ax.axvline(0, color="#999", linewidth=0.8)
        ax.set_xlabel("usage trend\n(points of team share gained per season)")
    axes[0].set_ylabel("points per game vs. his own\nrecent form  (0 = held serve)")
    fig.suptitle("Is his role growing or shrinking?\n"
                 "Climbing usage predicts a jump at all three spots. Receivers only cleared "
                 "the bar once the training set\nwent from three seasons to five -- same "
                 "effect, finally enough data to see it.",
                 fontsize=10.5, y=1.08)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "usage_trend.png", bbox_inches="tight")
    plt.close(fig)


def chart_trend_sample(df):
    """Why 2-season players were kept instead of discounted."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    groups = [("players with 3 seasons of history", lambda p: p[p.trend_seasons_used == 3]),
              ("all players with 2 or more", lambda p: p[p.trend_seasons_used >= 2])]
    width, offsets = 0.35, [-0.19, 0.19]
    for gi, (label, selector) in enumerate(groups):
        centres, lows, highs, xs = [], [], [], []
        for pi, pos in enumerate(POSITIONS):
            p = selector(df[df.position == pos]).dropna(
                subset=CONTROLS[pos] + ["delta", "usage_trend_share", "age"])
            if len(p) < 25:
                continue
            X = np.column_stack([p[CONTROLS[pos]].values.astype(float),
                                 p.age.values - p.age.mean(),
                                 p.usage_trend_share.values])
            beta, se, _ = ols(X, p.delta.values)
            centres.append(beta[-1] * 0.01)
            lows.append((beta[-1] - 1.96 * se[-1]) * 0.01)
            highs.append((beta[-1] + 1.96 * se[-1]) * 0.01)
            xs.append(pi + offsets[gi])
        ax.errorbar(xs, centres,
                    yerr=[np.array(centres) - np.array(lows),
                          np.array(highs) - np.array(centres)],
                    fmt="o", capsize=5, markersize=7, linewidth=1.8,
                    label=label, color=["#B0BEC5", "#1565C0"][gi])
    ax.axhline(0, color="#B71C1C", linewidth=1.2, linestyle="--",
               label="zero = usage direction tells you nothing")
    ax.set_xticks(range(len(POSITIONS)))
    ax.set_xticklabels(POSITIONS)
    ax.set_ylabel("effect of usage trend\n(PPG per point of share/season)")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title("Why we kept players with only two seasons of history\n"
                 "Restricting to three-season players (gray) doesn't kill the effect -- it just "
                 "widens the error bars until\nthey cross zero. Same answer, less certainty. "
                 "Dropping those players would have thrown away the signal.",
                 fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "trend_sample_size.png", bbox_inches="tight")
    plt.close(fig)


def chart_age_vs_experience(df):
    """The bake-off: what we tried for aging and what won."""
    forms = ["years in league\n(old model)", "age", "age + age squared", "age brackets"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, pos in zip(axes, POSITIONS):
        p = df[df.position == pos].dropna(subset=CONTROLS[pos] + ["delta", "age", "experience"]).copy()
        p["age_c"] = p.age - p.age.mean()
        C = p[CONTROLS[pos]].values.astype(float)
        breaks = {"RB": [24, 26, 28], "WR": [25, 27, 29], "TE": [25, 27, 29]}[pos]
        specs = {
            forms[0]: p[["experience"]].values.astype(float),
            forms[1]: p[["age_c"]].values,
            forms[2]: np.column_stack([p.age_c, p.age_c ** 2]),
            forms[3]: np.column_stack([(p.age >= b).astype(float) for b in breaks]),
        }
        scores = []
        for name, extra in specs.items():
            X = np.column_stack([C, extra])
            beta, _, _ = ols(X, p.delta.values)
            resid = p.delta.values - np.column_stack([np.ones(len(p)), X]) @ beta
            ss_res, ss_tot = resid @ resid, ((p.delta - p.delta.mean()) ** 2).sum()
            n, k = len(p), X.shape[1] + 1
            scores.append(100 * (1 - (ss_res / (n - k)) / (ss_tot / (n - 1))))
        # Highlight what SHIPPED, not what scored highest. At TE the
        # bracketed version scores a shade better, and we rejected it
        # anyway: its middle bracket (27-28) comes out POSITIVE, i.e. it
        # claims tight ends briefly get better at 27 and then resume
        # declining. That is not an aging curve, it is a small sample
        # drawing shapes in noise. Letting the bar chart crown a winner
        # by score alone would quietly overrule that judgment.
        shipped = forms[1]
        colors = ["#B0BEC5"] * len(forms)
        colors[forms.index(shipped)] = POSITION_COLORS[pos]
        ax.barh(forms, scores, color=colors)
        ax.set_title(f"{pos}  --  shipped: age")
        ax.set_xlabel("how much of the variation it explains (%)")
        ax.invert_yaxis()
        if pos == "TE" and scores[3] > scores[1]:
            ax.annotate("brackets edge it on score, but the\n27-28 bracket comes out POSITIVE --\n"
                        "noise, not a curve. Rejected.",
                        xy=(0.97, 0.06), xycoords="axes fraction", ha="right", fontsize=6.5,
                        bbox=dict(boxstyle="round", fc="#FFF3E0", ec="#EF6C00", alpha=0.95))
    fig.suptitle("What we tried for aging, and what actually won\n"
                 "Plain age beat years-in-league at all three spots. Curved and bracketed "
                 "versions added nothing worth the complexity.",
                 fontsize=10.5, y=1.05)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "age_vs_experience.png", bbox_inches="tight")
    plt.close(fig)


def main():
    CHARTS_DIR.mkdir(exist_ok=True)
    df = load()
    seasons = sorted(df.season.unique())
    print(f"Charting {len(df)} player-seasons from {seasons[0]}-{seasons[-1]}")
    chart_age_curves(df)
    chart_usage_trend(df)
    chart_trend_sample(df)
    chart_age_vs_experience(df)
    for f in sorted(CHARTS_DIR.glob("*.png")):
        print(f"  wrote {f.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
