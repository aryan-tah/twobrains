"""Export an arena run into a compact JSON bundle for web/index.html.
usage: uv run python experiments/export_web.py runs/aroused.json web/data/aroused.json [n_display_per_brain]"""
import sys, json, numpy as np, pandas as pd; sys.path.insert(0, ".")
from twobrains.connectome import Connectome

src, dst = sys.argv[1], sys.argv[2]; ND = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
MAX_EDGES = int(sys.argv[4]) if len(sys.argv) > 4 else 60000
run = json.load(open(src)); spk = np.load(src.replace(".json", "") + ".spikes.npz")
out = {"tick_ms": run["tick_ms"], "arena_r": run["arena_r"], "ticks": len(run["log"]), "bodies": [], "series": {}, "brains": {}}
for rec in run["log"]:
    m, f = rec["male"]["body"], rec["female"]["body"]
    out["bodies"].append([round(m["x"], 2), round(m["y"], 2), round(m["heading"]), round(f["x"], 2), round(f["y"], 2), round(f["heading"]),
                          round(m["singing"], 2), round(f["accept"], 2), round(f["reject"], 2), round(rec["dist"], 2)])
keys = ["P1", "song", "turn_L", "turn_R", "forward", "vision_L", "vision_R", "smell", "touch_L", "touch_R", "accept", "reject", "_active"]
for who in ["male", "female"]:
    out["series"][who] = {k: [round(rec[who]["out"].get(k, 0) * (100 if not k.startswith("_") else 1), 1) for rec in run["log"]] for k in keys if k in run["log"][0][who]["out"]}

GROUPS = {  # display colour groups (name -> type regex / selector)
    "male":   {"P1": r"^pC1_", "song DN (pIP10)": r"^pIP10$", "turning DN (DNa02)": r"^DNa02$", "vision (LC10a)": r"^LC10a$",
               "pheromone (ORN_VA1v)": r"^ORN_VA1v$", "hearing (JO)": r"^JO-(A|B)"},
    "female": {"pC1": r"^pC1[a-e]$", "accept DN (vpoDN)": r"^DNp37$", "reject DN (DNp13)": r"^DNp13$", "turning DN (DNa02)": r"^DNa02$",
               "vision (LC10a)": r"^LC10a$", "hearing (JO)": r"^JO-(A|B)", "cVA (ORN_DA1)": r"^ORN_DA1$"},
}
for who, name in [("male", "malecns"), ("female", "flywire")]:
    c = Connectome.load(name); pos = np.load(f"cache/{name}.soma.npy"); rng = np.random.default_rng(0)
    idx, ptr = spk[f"{who}_idx"], spk[f"{who}_ptr"]
    fired = np.zeros(c.N, np.int64); np.add.at(fired, idx, 1)
    ok = np.isfinite(pos[:, 0])
    special = np.unique(np.concatenate([c.select(r) for r in GROUPS[who].values()]))
    special = special[ok[special]]
    # display set: all special + everything that fired (with soma) + random background
    firedset = np.flatnonzero((fired > 0) & ok); firedset = np.setdiff1d(firedset, special)
    if firedset.size > ND: firedset = rng.choice(firedset, ND, replace=False)
    bg = np.setdiff1d(np.flatnonzero(ok), np.concatenate([special, firedset]))
    bg = rng.choice(bg, min(bg.size, max(0, ND * 2 - special.size - firedset.size)), replace=False)
    disp = np.concatenate([special, firedset, bg]); order = np.argsort(disp); disp = disp[order]
    group = np.zeros(disp.size, np.int8)
    for gi, (gname, r) in enumerate(GROUPS[who].items(), start=1):
        group[np.isin(disp, c.select(r))] = gi
    remap = -np.ones(c.N, np.int64); remap[disp] = np.arange(disp.size)
    per_tick = []
    for t in range(len(ptr) - 1):
        s = remap[idx[ptr[t]:ptr[t + 1]]]; per_tick.append(s[s >= 0].tolist())
    p = pos[disp]
    # strongest synaptic connections among displayed neurons (for the wiring layer)
    sub = c.W[disp][:, disp].tocoo()
    keep = np.argsort(-np.abs(sub.data))[:MAX_EDGES]
    edges = np.stack([sub.col[keep], sub.row[keep]], 1)          # [pre, post] display indices
    esign = (sub.data[keep] > 0).astype(np.int8)
    out["brains"][who] = {"edges": edges.astype(int).tolist(), "esign": esign.tolist(),"n_total": int(c.N), "n_display": int(disp.size), "groups": ["other"] + list(GROUPS[who].keys()),
                          "pos": np.round(p, 1).tolist(), "group": group.tolist(), "type": c.meta.loc[disp, "type"].fillna("").tolist(),
                          "spikes": per_tick, "fired_total": int((fired > 0).sum())}
    print(who, "display", disp.size, "of", c.N, "| neurons that fired at least once:", int((fired > 0).sum()))
json.dump(out, open(dst, "w"), separators=(",", ":")); print("wrote", dst, round(__import__("os").path.getsize(dst) / 1e6, 1), "MB")
