"""Validation 1: activate male P1 (pC1) neurons -> does the pulse-song descending neuron pIP10 fire?
Known biology: optogenetic P1 activation drives courtship song via pIP10 (von Philipsborn 2011; Ding 2019).
"""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, ".")
from twobrains.connectome import Connectome
from twobrains.lif import LIFBrain

m = Connectome.load("malecns"); print(m.summary())
P1 = m.select(r"^pC1_")            # male-specific P1 cluster
readout = {k: m.select(rf"^{k}$") for k in ["pIP10", "vPR6", "pMP2", "DNa02", "DNp13", "aSP22", "MN9", "DNp01"]}
print("P1 neurons:", P1.size, "| readouts:", {k: v.size for k, v in readout.items()})

brain = LIFBrain(m.W)
for label, drive in [("baseline (no input)", {}), ("P1 @150Hz Poisson", dict(poisson_idx=P1, poisson_rate_hz=150))]:
    brain.reset(); t0 = time.time()
    rec = brain.run(500, **drive)
    r = rec.rate()
    print(f"\n== {label}: {rec.i.size:,} spikes in 500ms, {time.time()-t0:.1f}s wall, {int((r>0).sum()):,} active neurons")
    print("   P1 mean rate %.1f Hz" % r[P1].mean())
    for k, idx in readout.items():
        print(f"   {k:6s} rates: {np.round(r[idx],1).tolist()}")
    top = np.argsort(-r)[:15]
    print("   top firing types:", m.meta.loc[top, 'type'].tolist())
