"""Parameter sweep: find a regime where both brains are stable (activity stops when stimulus stops)
and stimulus-specific, and the courtship read-outs respond."""
import sys, numpy as np; sys.path.insert(0, ".")
from twobrains.connectome import Connectome
from twobrains.lif import LIFBrain, LIFParams

M = Connectome.load("malecns"); F = Connectome.load("flywire")
male = {"cues": {"LC10a": M.select(r"^LC10a$"), "ORN_VA1v": M.select(r"^ORN_VA1v$"), "ppk23": M.select(receptor="putative_ppk23")},
        "read": {"P1": M.select(r"^pC1_"), "pIP10": M.select(r"^pIP10$"), "DNa02": M.select(r"^DNa02$"), "DNp09": M.select(r"^DNp09$")}}
male["cues"]["ALL"] = np.concatenate(list(male["cues"].values()))
fem = {"cues": {"JO-AB": F.select(r"^JO-(A|B)"), "LC10a": F.select(r"^LC10a$"), "ORN_DA1": F.select(r"^ORN_DA1$")},
       "read": {"pC1": F.select(r"^pC1[a-e]$"), "vpoDN": F.select(r"^DNp37$"), "DNp13": F.select(r"^DNp13$"), "DNa02": F.select(r"^DNa02$"), "DNp09": F.select(r"^DNp09$")}}
fem["cues"]["ALL"] = np.concatenate(list(fem["cues"].values()))

def probe(conn, spec, p, hz=150):
    b = LIFBrain(conn.W, p); out = []
    for k, idx in spec["cues"].items():
        b.reset(); on = b.run(300, poisson_idx=idx, poisson_rate_hz=hz); off = b.run(300); con, coff = on.count(), off.count()
        reads = " ".join(f"{r}={con[i].mean()/.3:4.0f}" for r, i in spec["read"].items())
        out.append(f"    {k:8s} on {on.i.size:>7,}sp/{(con>0).sum():>6,}act | off {off.i.size:>7,}sp/{(coff>0).sum():>5,}act | {reads}")
    return "\n".join(out)

grid = [dict(w_syn=0.275, b_adapt=2, in_norm=300), dict(w_syn=0.275, b_adapt=2, in_norm=600), dict(w_syn=0.275, b_adapt=0, in_norm=300),
        dict(w_syn=0.275, b_adapt=2, in_norm=150), dict(w_syn=0.2, b_adapt=2, in_norm=300)]
for g in (grid if len(sys.argv) < 2 else [eval(sys.argv[1])]):
    p = LIFParams(**g); print(f"\n### {g}"); print("  MALE"); print(probe(M, male, p)); print("  FEMALE"); print(probe(F, fem, p))
