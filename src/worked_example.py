"""A worked VONA example at Jack's real 12-team picks."""
import src.draft_sim as D
from src.draft_sim import *

key="12team"; lg=D.LEAGUES[key]; cfg=load_config(D.PROJECT_ROOT/lg["config"])
pl=load_board(D.PROJECT_ROOT/lg["board"], D.PROJECT_ROOT/"data"/"player_features.csv")
teams=12; slot=3
rounds=D.skill_rounds(cfg)
picks=[]
for r in range(1,rounds+1):
    picks.append((r-1)*teams + (slot if r%2==1 else teams-slot+1))
print("Your 12-team picks (round 8 forfeited for Olave):")
print("  " + "  ".join(f"R{i+1}:#{p}" for i,p in enumerate(picks)))

byname={p.name:p for p in pl}
taken=set()
# Olave is yours already; the four known opponent keepers are gone.
for n in ["Chris Olave","Ja'Marr Chase","Jonathan Taylor","Javonte Williams","George Pickens"]:
    taken.add(byname[n].idx)

def table(pick_no, next_pick, taken, title):
    avail=[p for p in pl if p.idx not in taken]
    # everyone with ADP better than this pick is assumed gone
    avail=[p for p in avail if p.adp >= pick_no-6]
    gap=next_pick-pick_no
    by_adp=sorted(avail,key=lambda p:p.adp)
    fb=D._next_available_at(by_adp,gap)
    print(f"\n{title}")
    print(f"  pick #{pick_no}; next pick #{next_pick}; {gap} players go in between")
    print(f"  best expected to survive to #{next_pick}: " +
          "  ".join(f"{k} {v.name.split()[-1]} {v.ppg:.1f}" for k,v in
                    sorted(fb.items())))
    rows=[]
    for p in avail:
        alt=fb.get(p.pos)
        rows.append((p.ppg-(alt.ppg if alt else 0), p))
    rows.sort(key=lambda t:(-t[0], t[1].rank))
    print(f"  {'VONA':>6}  {'pos':<3} {'player':<24} {'PPG':>5} {'board rk':>8}")
    for v,p in rows[:8]:
        print(f"  {v:+6.1f}  {p.pos:<3} {p.name:<24} {p.ppg:5.1f} {p.rank:8d}")
    print(f"  -- for contrast, top of the board right now: " +
          ", ".join(f"{p.pos} {p.name.split()[-1]}" for p in
                    sorted(avail,key=lambda p:p.rank)[:5]))
    return rows

r1=table(3, 22, taken, "ROUND 1 — pick #3")
best=r1[0][1]; taken.add(best.idx)
print(f"\n  => you take {best.name}")
# simulate picks 4..21 going by ADP
gone=[p for p in sorted((q for q in pl if q.idx not in taken), key=lambda q:q.adp)][:18]
for g in gone: taken.add(g.idx)
r2=table(22, 27, taken, "ROUND 2 — pick #22")
