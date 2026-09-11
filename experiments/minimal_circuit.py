"""Smallest sub-network that still performs each behaviour (delta-debugging over the neurons that fire when intact).
-> results/minimal_<brain>.json"""
import sys, json, time, numpy as np; sys.path.insert(0, ".")
from twobrains.connectome import Connectome
from twobrains.lif import LIFBrain
from twobrains.ablation import assays, params, masked, T_MS

name = sys.argv[1] if len(sys.argv) > 1 else "flywire"; THRESH = 0.5
c = Connectome.load(name); A = assays(c); p = params(c); N = c.N
out = {"brain": name, "N": int(N), "circuits": {}}
for k, a in A.items():
    t0 = time.time()
    b = LIFBrain(c.W, p, seed=0); cnt = b.run(T_MS, poisson_idx=a["inp"], poisson_rate_hz=a["hz"]).count()
    if a["out"] is None: a["out"] = np.flatnonzero((cnt > 0) & c.meta.fru_dsx.notna().values)
    def score(keep):
        W = masked(c.W, keep); bb = LIFBrain(W, p, seed=0); cc = bb.run(T_MS, poisson_idx=a["inp"][keep[a["inp"]]], poisson_rate_hz=a["hz"]).count()
        return float((cc[a["out"]] > 0).sum()) if k == "hear" else float(cc[a["out"]].mean() / (T_MS / 1000))
    base = score(np.ones(N, bool))
    if base <= 0: print(k, "intact score 0, skipping"); continue
    fixed = np.zeros(N, bool); fixed[a["inp"]] = True; fixed[a["out"]] = True
    cand = np.flatnonzero((cnt > 0) & ~fixed)                  # only neurons that fired can matter
    keep = np.zeros(N, bool); keep[fixed] = True; keep[cand] = True
    s0 = score(keep); print(f"[{k}] intact {base:.1f}; keeping only the {cand.size} neurons that fired (+{int(fixed.sum())} I/O): {s0:.1f}", flush=True)
    if s0 < THRESH * base: print("   active-set alone fails; keeping full network as circuit"); out["circuits"][k] = {"neurons": int(N), "score": base, "intact": base}; continue
    # ddmin: try removing chunks; shrink chunk size when nothing can be removed
    rng = np.random.default_rng(0); n_chunks = 2; tests = 0
    while True:
        cur = np.flatnonzero(keep & ~fixed); rng.shuffle(cur); removed_any = False
        for chunk in np.array_split(cur, n_chunks):
            if chunk.size == 0: continue
            trial = keep.copy(); trial[chunk] = False; s = score(trial); tests += 1
            if s >= THRESH * base: keep = trial; removed_any = True
        cur = np.flatnonzero(keep & ~fixed)
        print(f"   chunks={n_chunks:4d} remaining={cur.size:6d} tests={tests} [{time.time()-t0:.0f}s]", flush=True)
        if not removed_any:
            if n_chunks >= cur.size: break
            n_chunks = min(cur.size, n_chunks * 2)
        if cur.size == 0: break
    final = np.flatnonzero(keep); s = score(keep)
    out["circuits"][k] = {"name": a["name"], "intact": base, "score": s, "neurons": int(final.size), "ids": c.ids[final].tolist(),
                          "types": c.meta.loc[final, "type"].fillna("?").value_counts().to_dict()}
    print(f"[{k}] MINIMAL CIRCUIT: {final.size} neurons ({final.size/N*100:.3f}% of brain) keeps score {s:.1f}/{base:.1f}; types: {list(out['circuits'][k]['types'].items())[:12]}", flush=True)
    json.dump(out, open(f"results/minimal_{name}.json", "w"), indent=1)
print("done")
