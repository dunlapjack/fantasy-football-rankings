"""
Does the frozen-ADP assumption in the Wait columns actually cost anything?

Two policies, identical except for where the survivor comes from:
  vona_live   -- recomputes the survivor from who is ACTUALLY still on the
                 board at this moment in this draft.
  vona_static -- reads the precomputed board column, anchored to each
                 player's own ADP and frozen before the draft started.
Both carry the same guards (starters-first + QB/TE caps), so the ONLY
difference measured is the staleness of the survivor estimate.
"""
import random
from collections import defaultdict
import src.draft_sim as D
from src.draft_sim import *
from vona_columns import compute_wait_cost, snake_gaps

STATIC = {}   # (league, player idx, gap) -> precomputed wait

def build_static(key, players, gaps):
    rows=[{"position":p.pos,"adjusted_fantasy_points_per_game":p.ppg,
           "adp":p.adp,"has_adp":True,"out_for_season":False,"_i":p.idx}
          for p in players]
    w=compute_wait_cost(rows,gaps)
    for g in gaps:
        for r,v in zip(rows,w[g]):
            STATIC[(key,r["_i"],g)] = v

def guards(available, roster, ctx):
    have=defaultdict(int)
    for p in roster: have[p.pos]+=1
    caps={"QB":3,"TE":2} if ctx["superflex"] else {"QB":2,"TE":2}
    pool=[p for p in available if have[p.pos]<caps.get(p.pos,99)]
    need=D._slots_open(roster,ctx)
    sub=[p for p in pool if p.pos in need]
    return sub or pool or available

def make_static(key,gaps):
    def f(available, roster, ctx):
        pool=guards(available,roster,ctx)
        g=min(gaps,key=lambda x:abs(x-ctx["gap"])) if ctx["gap"] else gaps[0]
        # Only compare Wait among players realistically in play right now.
        # The column is anchored to each man's own ADP, so a tail player's
        # number is inflated by the fact that everyone behind HIM is
        # worthless -- it is not comparable to a first-rounder's.
        window=set(x.idx for x in sorted(pool,key=lambda x:x.adp)[:30])
        pool=[p for p in pool if p.idx in window]
        best,bk=None,None
        for p in pool:
            v=STATIC.get((key,p.idx,g))
            k=(-(v if v is not None else -99), p.rank)
            if bk is None or k<bk: best,bk=p,k
        return best
    return f

def vona_live(available, roster, ctx):
    return D.policy_vona(guards(available,roster,ctx), roster, ctx)

for key,slot in (("12team",3),("32team",4)):
    lg=D.LEAGUES[key]; cfg=load_config(D.PROJECT_ROOT/lg["config"])
    pl=load_board(D.PROJECT_ROOT/lg["board"], D.PROJECT_ROOT/"data"/"player_features.csv")
    gaps=snake_gaps(int(cfg["num_teams"]),slot)
    build_static(key,pl,gaps)
    D.POLICIES["vona_live"]=vona_live
    D.POLICIES["vona_static"]=make_static(key,gaps)
    print(f"\n=== {key} (gaps {gaps}) ===",flush=True)
    res,_,_=run_league(key,sims=120,seed=17,quiet=True,
                       policies=["board","vona_static","vona_live"])
    for n,r in sorted(res.items(),key=lambda kv:-kv[1]["mean"]):
        print(f"  {n:<12} {r['mean']:8.1f}  vs board {r['delta']:+7.1f}  "
              + " ".join(f"{p}{r['pos'].get(p,0):.1f}" for p in D.MODELED),flush=True)
    a=res["vona_static"]["totals"]; b=res["vona_live"]["totals"]
    d=[x-y for x,y in zip(a,b)]
    m=sum(d)/len(d); import math
    se=math.sqrt(sum((x-m)**2 for x in d)/(len(d)-1)/len(d))
    print(f"  static minus live: {m:+.1f} pts  (se {se:.1f})",flush=True)
