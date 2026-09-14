"""Print the ablation and minimal-circuit results as markdown tables."""
import json, sys, numpy as np
name = sys.argv[1] if len(sys.argv) > 1 else "flywire"
r = json.load(open(f"results/ablation_{name}.json")); keys = list(r["assays"])
print(f"### {name}: random deletion (mean of 3 seeds)\n\n| deleted | " + " | ".join(keys) + " |\n|---:|" + "---:|" * len(keys))
for f in sorted(set(x["frac"] for x in r["random"])):
    rows = [x for x in r["random"] if x["frac"] == f]
    print(f"| {f*100:.0f} % | " + " | ".join(f"{np.mean([x[k] for x in rows]):.2f}" for k in keys) + " |")
print(f"\n### {name}: targeted deletion\n\n| delete | removed | " + " | ".join(keys) + " |\n|---|---:|" + "---:|" * len(keys))
for lab, v in r["targeted"].items():
    print(f"| {lab} | {v['removed']:,} | " + " | ".join(f"{v[k]:.2f}" for k in keys) + " |")
m = json.load(open(f"results/minimal_{name}.json"))
print(f"\n### {name}: minimal circuits\n\n| capability | neurons | score / intact |\n|---|---:|---:|")
for k, v in m["circuits"].items():
    print(f"| {k} | {v['neurons']} | {v['score']:.1f} / {v['intact']:.1f} |")
