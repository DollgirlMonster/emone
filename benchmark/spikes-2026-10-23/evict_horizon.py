"""Belady limited to an H-token horizon: evict a resident expert not used in the
next H tokens (ties and fallback by the shipped LFU order). Shows how much of the
oracle's gain a perfect short forecast would capture."""
import collections, sys
sys.path.insert(0, sys.argv[0].rsplit('/', 1)[0])
from evict_sim import load

def simulate(steps, slots, horizon):
    resident=set(); count=collections.Counter(); last={}; hits=misses=0
    for t, experts in enumerate(steps):
        need=[e for e in experts if e not in resident]
        hits+=len(experts)-len(need); misses+=len(need)
        soon=set()
        for f in steps[t+1:t+1+horizon]: soon.update(f)
        protected=set(experts)
        for e in need:
            if len(resident)>=slots:
                cands=[r for r in resident if r not in protected]
                victim=min(cands,key=lambda r:(r in soon, count[r], last[r]))
                resident.discard(victim)
            resident.add(e)
        for e in experts: count[e]+=1; last[e]=t
    return hits, misses

path=sys.argv[1]; slots=int(sys.argv[2]); layers=load(path)
pos=max(len(s) for s in layers.values())
for h in (0,1,2,4,8,16,64):
    H=M=0
    for steps in layers.values():
        a,b=simulate(steps,slots,h); H+=a; M+=b
    print(f"  horizon={h:3d} hit={H/(H+M):.4f} misses/token={M/pos:.1f}")
