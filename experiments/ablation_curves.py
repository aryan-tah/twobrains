"""Random and targeted deletion curves for both brains -> results/ablation_<brain>.json"""
import sys, json, time, numpy as np; sys.path.insert(0, ".")
from twobrains.connectome import Connectome
from twobrains.ablation import assays, score_all

name = sys.argv[1] if len(sys.argv) > 1 else "flywire"
FRACS = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SEEDS = [0, 1, 2]
c = Connectome.load(name); A = assays(c); N = c.N; m = c.meta
print(c.summary()); t0 = time.time()
intact = score_all(c, np.ones(N, bool), A)
print("intact:", {k: round(v[0], 2) for k, v in intact.items()}, "| assay pops:", {k: (int(a['inp'].size), int(a['out'].size) if a['out'] is not None else 0) for k, a in A.items()})
res = {"brain": name, "N": int(N), "assays": {k: a["name"] for k, a in A.items()}, "intact": {k: v[0] for k, v in intact.items()}, "random": [], "targeted": {}}
protected = np.zeros(N, bool)                       # never delete the neurons we drive or read out (else the assay is trivially dead)
for a in A.values(): protected[a["inp"]] = True; protected[a["out"]] = True
print("protected (assay I/O) neurons:", int(protected.sum()))
for f in FRACS:
    for seed in SEEDS:
        rng = np.random.default_rng(100 + seed); keep = np.ones(N, bool)
        cand = np.flatnonzero(~protected); kill = rng.choice(cand, int(f * N), replace=False); keep[kill] = False
        sc = score_all(c, keep, A, intact, seed=seed)
        res["random"].append({"frac": f, "seed": seed, "removed": int((~keep).sum()), **{k: v[1] for k, v in sc.items()}})
        print(f"random {f*100:4.0f}% seed {seed}: " + " ".join(f"{k}={v[1]:.2f}" for k, v in sc.items()) + f"  [{time.time()-t0:.0f}s]", flush=True)
        if f == 0: break
deg = np.asarray(abs(c.W).sum(axis=0)).ravel() + np.asarray(abs(c.W).sum(axis=1)).ravel()
sc_map = {"optic lobes": m.superclass.isin(["ol_intrinsic", "optic"]).values,
          "central brain intrinsic": m.superclass.isin(["cb_intrinsic", "central"]).values,
          "all inhibitory (GABA)": (m.nt == "gaba").values, "all glutamatergic": (m.nt == "glutamate").values,
          "all dopaminergic": (m.nt == "dopamine").values, "right hemisphere": (m.side == "R").values, "left hemisphere": (m.side == "L").values,
          "top 1% hubs by synapse count": deg >= np.percentile(deg, 99), "top 10% hubs": deg >= np.percentile(deg, 90),
          "Kenyon cells (mushroom body)": (m.cls == "Kenyon_Cell").values, "central complex": (m.cls == "CX").values,
          "all descending neurons except read-outs": m.superclass.isin(["descending_neuron", "descending"]).values}
if name == "malecns": sc_map["ventral nerve cord"] = m.superclass.str.startswith("vnc").fillna(False).values
for label, mask in sc_map.items():
    keep = ~(mask & ~protected); sc = score_all(c, keep, A, intact)
    res["targeted"][label] = {"removed": int((~keep).sum()), **{k: v[1] for k, v in sc.items()}}
    print(f"targeted [{label}] removed {int((~keep).sum()):,}: " + " ".join(f"{k}={v[1]:.2f}" for k, v in sc.items()), flush=True)
json.dump(res, open(f"results/ablation_{name}.json", "w"), indent=1); print("saved")
