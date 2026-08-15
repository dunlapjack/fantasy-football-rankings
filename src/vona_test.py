"""Can VONA be run by hand? Test the variants a human would actually execute."""
import random
from collections import defaultdict
import src.draft_sim as D
from src.draft_sim import *

def vona_need(available, roster, ctx):
    """VONA, but only among positions where a STARTING slot is still open.
    Once the lineup is full, plain VONA. This is what a person does naturally."""
    need = D._slots_open(roster, ctx)
    pool = [p for p in available if p.pos in need] if need else available
    return D.policy_vona(pool or available, roster, ctx)

def vona_cap(available, roster, ctx):
    """VONA with the roster limits a person would keep in their head."""
    have = defaultdict(int)
    for p in roster: have[p.pos]+=1
    caps = {"QB":2,"TE":2} if not ctx["superflex"] else {"QB":3,"TE":2}
    pool=[p for p in available if have[p.pos] < caps.get(p.pos,99)]
    return D.policy_vona(pool or available, roster, ctx)

def vona_both(available, roster, ctx):
    """Both guards: fill starters first, and respect QB/TE caps."""
    have = defaultdict(int)
    for p in roster: have[p.pos]+=1
    caps = {"QB":2,"TE":2} if not ctx["superflex"] else {"QB":3,"TE":2}
    pool=[p for p in available if have[p.pos] < caps.get(p.pos,99)]
    need = D._slots_open(roster, ctx)
    sub=[p for p in pool if p.pos in need]
    return D.policy_vona(sub or pool or available, roster, ctx)

D.POLICIES.update(vona_need=vona_need, vona_cap=vona_cap, vona_both=vona_both)
for key in ("12team","8team","6team","32team"):
    print(f"\n=== {key} ===", flush=True)
    res,_,_ = run_league(key, sims=120, seed=17, quiet=True,
        policies=["board","vona","vona_need","vona_cap","vona_both","marginal"])
    for n,r in sorted(res.items(), key=lambda kv:-kv[1]["mean"]):
        sig = r["delta"]/r["delta_se"] if r.get("delta_se") else 0
        print(f"  {n:<12} {r['mean']:8.1f}  vs board {r['delta']:+7.1f} ({sig:+5.1f} sd)  "
              + " ".join(f"{p}{r['pos'].get(p,0):.1f}" for p in D.MODELED), flush=True)
