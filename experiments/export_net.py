"""Export a connectome as a compact binary for the in-browser LIF (web/data/<name>_net.bin) + metadata JSON."""
import sys, json, struct, numpy as np; sys.path.insert(0, ".")
from twobrains.connectome import Connectome
from twobrains.ablation import assays, params

name = sys.argv[1] if len(sys.argv) > 1 else "flywire"
assert name in ("flywire", "malecns"), "usage: export_net.py [flywire|malecns]"
c = Connectome.load(name); W = c.W.tocsc(); W.sort_indices(); N = c.N; p = params(c); A = assays(c)
pos = np.load(f"cache/{name}.soma.npy").astype(np.float32); ok = np.isfinite(pos[:, 0])
pos[~ok] = np.nan
with open(f"web/data/{name}_net.bin", "wb") as f:
    f.write(struct.pack("<4sIII", b"TBN1", N, W.nnz, 0))
    f.write(W.indptr.astype(np.int32).tobytes()); f.write(W.indices.astype(np.int32).tobytes())
    f.write(np.clip(W.data, -32767, 32767).astype(np.int16).tobytes()); f.write(pos.tobytes())
m = c.meta
# hearing assay read-out: Fru+/Dsx+ neurons that respond when intact
if "hear" in A:
    from twobrains.lif import LIFBrain
    cnt = LIFBrain(c.W, p, seed=0).run(300, poisson_idx=A["hear"]["inp"], poisson_rate_hz=150).count()
    A["hear"]["out"] = np.flatnonzero((cnt > 0) & m.fru_dsx.notna().values)
groups = {"hemisphere_L": (m.side == "L").values, "hemisphere_R": (m.side == "R").values, "optic": m.superclass.isin(["ol_intrinsic", "optic"]).values,
          "central": m.superclass.isin(["cb_intrinsic", "central"]).values, "gaba": (m.nt == "gaba").values, "sensory": m.superclass.str.contains("sensory").fillna(False).values,
          "descending": m.superclass.isin(["descending_neuron", "descending"]).values, "kenyon": (m.cls == "Kenyon_Cell").values, "cx": (m.cls == "CX").values}
deg = np.asarray(abs(c.W).sum(axis=0)).ravel() + np.asarray(abs(c.W).sum(axis=1)).ravel()
meta = {"name": name, "N": int(N), "nnz": int(W.nnz), "params": {k: getattr(p, k) for k in ["dt", "tau_m", "tau_syn", "v_rest", "v_reset", "v_th", "t_rfc", "t_dly", "w_syn", "g_cap", "b_adapt", "tau_adapt"]},
        "assays": {k: {"name": a["name"], "inp": a["inp"].tolist(), "hz": a["hz"], "out": a["out"].tolist(), "mode": "responders" if k == "hear" else "rate"} for k, a in A.items()},
        "groups": {k: np.flatnonzero(v).tolist() for k, v in groups.items()}, "degree_rank": np.argsort(-deg).astype(int).tolist(),
        "protected": sorted(set(int(i) for a in A.values() for i in np.concatenate([a["inp"], a["out"]])))}
json.dump(meta, open(f"web/data/{name}_meta.json", "w"), separators=(",", ":"))
import os; print("bin", round(os.path.getsize(f"web/data/{name}_net.bin") / 1e6, 1), "MB; meta", round(os.path.getsize(f"web/data/{name}_meta.json") / 1e6, 1), "MB; assays", {k: (len(v["inp"]), len(v["out"])) for k, v in meta["assays"].items()})
