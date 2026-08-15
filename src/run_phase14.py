"""
Phase 14 runner: the bakeoff, the sensitivity checks, and the roster-shape
probes, dumped to JSON for the write-up.

Two passes on purpose. The bakeoff and the sensitivity sweep answer the
question that was asked and run in a couple of minutes per league; the
positional probes are a refinement and cost several times as much. Results
are written to disk after every league, so a run that is killed halfway
still leaves usable output rather than nothing.
"""
import json
from collections import defaultdict

import src.draft_sim as D
from src.draft_sim import (LEAGUES, MODELED, load_config, run_league,
                           PROJECT_ROOT)

SIMS = 120
SWEEP_SIMS = 30
PROBE_SIMS = 30
RESULTS = PROJECT_ROOT / "data" / "phase14_results.json"


def capped(base, maxpos):
    """Wrap a policy with a hard positional cap, for the shape probes."""
    def f(available, roster, ctx):
        have = defaultdict(int)
        for p in roster:
            have[p.pos] += 1
        pool = [p for p in available if have[p.pos] < maxpos.get(p.pos, 99)]
        return D.POLICIES[base](pool or available, roster, ctx)
    return f


def save(out):
    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2)


def main():
    out = {}

    # ---- pass 1: bakeoff + sensitivity ---------------------------------
    for key in LEAGUES:
        cfg = load_config(PROJECT_ROOT / LEAGUES[key]["config"])
        print(f"\n=== {cfg['league_name']} ({key}) ===", flush=True)
        res, _, _ = run_league(key, sims=SIMS, seed=17, quiet=True)
        entry = {"league_name": cfg["league_name"], "teams": cfg["num_teams"],
                 "rounds": cfg["total_rounds"], "slot": LEAGUES[key]["slot"],
                 "sims": SIMS, "policies": {}}
        for name, r in res.items():
            entry["policies"][name] = {
                "mean": r["mean"], "se": r["se"],
                "delta": r.get("delta", 0.0), "delta_se": r.get("delta_se", 0.0),
                "pos": r["pos"],
                "roster": [(p.pos, p.name, round(p.ppg, 1), p.rank)
                           for p in r["sample_roster"]]}
        for name, r in sorted(res.items(), key=lambda kv: -kv[1]["mean"]):
            sig = r["delta"] / r["delta_se"] if r.get("delta_se") else 0.0
            print(f"  {name:<16} {r['mean']:8.1f}  vs board {r['delta']:+7.1f} "
                  f"({sig:+5.1f} sd)  "
                  + " ".join(f"{p}{r['pos'].get(p, 0):.1f}" for p in MODELED),
                  flush=True)

        sweep = {}
        for scale, label in ((0.0, "no_injuries"), (1.0, "base"), (1.5, "high")):
            r3, _, _ = run_league(key, sims=SWEEP_SIMS, injury_scale=scale,
                                  seed=29, quiet=True,
                                  policies=["board", "starters_first", "caps",
                                            "vona", "marginal"])
            sweep[label] = {n: r3[n]["mean"] for n in r3}
            order = sorted(r3, key=lambda n: -r3[n]["mean"])
            print(f"  sweep {label:<12} " + " > ".join(order), flush=True)
        entry["sweep"] = sweep
        out[key] = entry
        save(out)

    # ---- pass 2: positional-count probes -------------------------------
    for key in LEAGUES:
        print(f"\n=== probes: {out[key]['league_name']} ===", flush=True)
        probes = {}
        for pos, values in (("QB", (1, 2, 3)), ("RB", (3, 4, 5, 6)),
                            ("WR", (3, 4, 5, 6)), ("TE", (1, 2, 3))):
            for v in values:
                tag = f"{pos}<={v}"
                D.POLICIES[tag] = capped("marginal", {pos: v})
                r2, _, _ = run_league(key, sims=PROBE_SIMS, policies=[tag],
                                      seed=17, quiet=True)
                probes[tag] = {"mean": r2[tag]["mean"], "se": r2[tag]["se"],
                               "pos": r2[tag]["pos"]}
                del D.POLICIES[tag]
            group = [t for t in probes if t.startswith(pos + "<")]
            best = max(group, key=lambda t: probes[t]["mean"])
            print(f"  {pos}: " + "  ".join(
                f"{t.split('<=')[1]}:{probes[t]['mean']:.0f}" for t in group)
                + f"   best={best}", flush=True)
        out[key]["probes"] = probes
        save(out)

    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
