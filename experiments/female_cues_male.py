"""Experiment: present *female cues* to the male brain (no P1 forcing) and ask whether the
courtship circuit (P1 -> pIP10 song) turns on by itself through the real wiring.
Cues: LC10a (fly-sized visual target), ORN_VA1v (Or47b, fly pheromones), ppk23 (7,11-HD contact).
"""
import sys, time, numpy as np
sys.path.insert(0, ".")
from twobrains.connectome import Connectome
from twobrains.lif import LIFBrain

m = Connectome.load("malecns")
pop = {"LC10a": m.select(r"^LC10a$"), "ORN_VA1v": m.select(r"^ORN_VA1v$"), "ppk23": m.select(receptor="putative_ppk23"),
       "ORN_DA1": m.select(r"^ORN_DA1$"), "JO-AB": m.select(r"^JO-(A|B)")}
P1 = m.select(r"^pC1_"); read = {k: m.select(rf"^{k}$") for k in ["pIP10", "pMP2", "DNa02", "DNp09", "DNa01", "aSP22", "vPR6", "ps1 MN", "hg1 MN"]}
print({k: v.size for k, v in pop.items()}, "ppk23 sides:", m.meta.loc[pop['ppk23'], 'side'].value_counts().to_dict())
brain = LIFBrain(m.W)
conds = {"vision only (LC10a)": ["LC10a"], "smell only (ORN_VA1v)": ["ORN_VA1v"], "touch only (ppk23)": ["ppk23"],
         "vision+smell+touch": ["LC10a", "ORN_VA1v", "ppk23"], "male pheromone cVA (ORN_DA1)": ["ORN_DA1"], "song (JO-A/B)": ["JO-AB"]}
for label, keys in conds.items():
    idx = np.concatenate([pop[k] for k in keys]); brain.reset(); t0 = time.time()
    rec = brain.run(500, poisson_idx=idx, poisson_rate_hz=100); r = rec.rate()
    print(f"\n== {label}: {rec.i.size:,} spikes, {int((r>0).sum()):,} active, {time.time()-t0:.1f}s")
    print(f"   P1 mean {r[P1].mean():5.1f} Hz, active P1 {(r[P1]>0).sum()}/{P1.size}, top P1 types {m.meta.loc[P1[np.argsort(-r[P1])[:5]],'type'].tolist()}")
    print("   " + "  ".join(f"{k}={np.round(r[v],0).astype(int).tolist()}" for k, v in read.items()))
