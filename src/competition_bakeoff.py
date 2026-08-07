"""
Phase 13 CP1. Chooses the `position_competition` definition by held-out
error instead of by argument.

THE QUESTION
------------
The shipped feature averages the trailing PPG of EVERY teammate at the
position. A receiver on a team carrying six WRs -- three of them camp
bodies projecting near zero -- therefore reads as facing LESS competition
than one on a team carrying three real players. That is roster length,
not football.

The plan sized this on Aug 4 and the number is why it is being measured
rather than debated: the DEFINITION choice moves Gibbs by 1.3 PPG, while
the roster event it was asked about -- Montgomery out, Pacheco in --
moves him 0.16. The definition is worth eight times the football.

FIVE CANDIDATES, ONE MODEL EACH
-------------------------------
    none    drop the feature entirely
    ppg     incumbent: mean of all other teammates
    top1    the single best other player
    top2    mean of the two best others
    top3    mean of the three best others

Fitted as FIVE SEPARATE MODELS per position, not as one model containing
all five. They measure the same quantity, so in a single regression they
would be collinear, mask each other, split one effect five ways, and
hand the win to whichever way the noise fell. That failure would look
exactly like a result. Separate models, compared out of sample, is the
only version of this that answers the question asked.

EXCLUDING SELF IS WHAT KEEPS IT SYMMETRIC, and it is why "top k" is not
"the backups." For Gibbs the pool is Pacheco and Ozigbo; for Pacheco the
pool is Gibbs. A backup's competition IS the starter.

WHAT WINS
---------
Mean MODEL-minus-LEVEL RMSE across the same three folds the gate uses.
Not R^2, not a p-value -- both of those chose the incumbent already, and
this exists because neither is evidence about prediction.

`none` winning is a real outcome and stays on the ballot. The feature was
dead from Phase 6 to Phase 10 and may only look alive because roster
noise correlates with something real.

USAGE
-----
    python -m src.competition_bakeoff

Reports a winner per position. Nothing is applied automatically: put the
winning name into fit_weights.COMPETITION_DEFINITION, refit, re-run the
gate.
"""

import argparse

import polars as pl

from src import fit_weights as veteran
from src.holdout import GATE_SEASONS, run_holdout

CANDIDATES = {
    "none": None,
    "ppg": "position_competition_ppg",
    "top1": "position_competition_top1",
    "top2": "position_competition_top2",
    "top3": "position_competition_top3",
}

# A definition has to beat the incumbent by more than this to displace
# it. Swapping a shipped definition for a 0.01 RMSE gain is churn, and
# churn on three folds is noise.
DISPLACEMENT_MARGIN = 0.02


def base_spec(position):
    """The position's shipped features with every competition definition
    stripped out, so each candidate can be added back alone."""
    return [
        f for f in veteran.FEATURE_SPECS[position]
        if not f.startswith("position_competition")
    ]


def main():
    parser = argparse.ArgumentParser(description="Phase 13 CP1 competition bake-off.")
    parser.add_argument("--alpha", type=float, default=veteran.ALPHA)
    args = parser.parse_args()

    df = veteran.load_backtest()

    missing = [
        column for column in CANDIDATES.values()
        if column and column not in df.columns
    ]
    if missing:
        raise SystemExit(
            f"\n{missing} are not in backtest_features.csv.\n"
            f"Rebuild it first:  python -m src.backtest\n"
        )

    results = {}
    for position in veteran.FEATURE_SPECS:
        base = base_spec(position)
        print(f"\n{'=' * 74}")
        print(f"{position}   base spec: {base}")
        print(f"{'=' * 74}")
        print(f"   {'definition':<10}" + "".join(f"{s:>10}" for s in GATE_SEASONS)
              + f"{'mean':>10}")

        scores = {}
        for name, column in CANDIDATES.items():
            spec = base + ([column] if column else [])
            gains = []
            for season in GATE_SEASONS:
                result = run_holdout(df, position, spec, veteran, season, args.alpha)
                if result is None:
                    continue
                gains.append(
                    result["scores"]["LEVEL"]["rmse"] - result["scores"]["MODEL"]["rmse"]
                )
            if not gains:
                continue
            scores[name] = sum(gains) / len(gains)
            print(f"   {name:<10}" + "".join(f"{g:>+10.4f}" for g in gains)
                  + f"{scores[name]:>+10.4f}")

        if not scores:
            continue

        best = max(scores, key=scores.get)
        incumbent = "ppg" if "ppg" in scores else "none"
        margin = scores[best] - scores.get(incumbent, float("-inf"))

        if best == incumbent:
            verdict = f"KEEP {incumbent}"
        elif margin > DISPLACEMENT_MARGIN:
            verdict = (f"SWITCH to {best} (+{margin:.4f} RMSE over {incumbent})")
        else:
            verdict = (f"KEEP {incumbent} -- {best} leads by only {margin:+.4f}, "
                       f"inside the {DISPLACEMENT_MARGIN} churn band")
        print(f"\n   -> {verdict}")
        results[position] = (best, verdict)

    print(f"\n\n{'=' * 74}")
    print("SUMMARY")
    print(f"{'=' * 74}")
    for position, (_, verdict) in results.items():
        print(f"   {position:<5}{verdict}")
    print("\nNothing was applied. Set fit_weights.COMPETITION_DEFINITION to the")
    print("winner, then:  python -m src.fit_weights && python -m src.holdout --gate")
    print("\nIf `none` wins anywhere, that is a real answer -- the feature was dead")
    print("from Phase 6 to Phase 10 and has never had an out-of-sample defence.")


if __name__ == "__main__":
    main()
