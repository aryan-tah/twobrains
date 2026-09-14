"""Ablation toolkit: delete neurons from a connectome and re-test a battery of capabilities.

Each assay drives a real sensory/command population with Poisson spikes for 300 ms and reads the
firing rate of a real output population. Score = rate / intact rate. Everything else is the wiring.
"""
from __future__ import annotations
import numpy as np, scipy.sparse as sp
from .connectome import Connectome
from .lif import LIFBrain, LIFParams
from .arena import MALE_PARAMS, FEMALE_PARAMS

T_MS = 300.0   # ms of simulated time per assay

def assays(c: Connectome) -> dict:
    s = c.select
    A = {
      "steer":   dict(name="steer toward target (left eye → left DNa02)", inp=s(r"^LC10a$", side="L"), hz=100, out=s(r"^DNa02$", side="L")),
      "escape":  dict(name="escape (looming LPLC2/LC4 → giant fiber DNp01)", inp=np.concatenate([s(r"^LPLC2$"), s(r"^LC4$")]), hz=100, out=s(r"^DNp01$")),
      "walk":    dict(name="walk forward (touch/leg input → DNp09/DNa01)", inp=s(receptor="putative_ppk23") if c.name == "malecns" else s(r"^LC10a$"), hz=100, out=np.concatenate([s(r"^DNp09$"), s(r"^DNa01$"), s(r"^DNa02$")])),
    }
    if c.name == "malecns":
        A["song"] = dict(name="song (P1 → pIP10)", inp=s(r"^pC1_"), hz=150, out=s(r"^pIP10$"))
    else:
        A["hear"] = dict(name="hear song (JO-A/B → Fru+/Dsx+ auditory neurons)", inp=s(r"^JO-(A|B)"), hz=150, out=None)   # out = fru/dsx neurons that respond when intact
        A["accept"] = dict(name="accept (pC1a → vpoDN)", inp=s(r"^pC1a$"), hz=150, out=s(r"^DNp37$"))
    return A

def params(c: Connectome) -> LIFParams:
    return MALE_PARAMS if c.name == "malecns" else FEMALE_PARAMS

def masked(W: sp.csc_matrix, keep: np.ndarray) -> sp.csc_matrix:
    d = sp.diags(keep.astype(np.float32))
    return (d @ W @ d).tocsc()

def run_assay(W, p: LIFParams, a: dict, keep: np.ndarray, seed=0):
    inp = a["inp"][keep[a["inp"]]]           # deleted input neurons cannot be driven
    b = LIFBrain(W, p, seed=seed)
    rec = b.run(T_MS, poisson_idx=inp, poisson_rate_hz=a["hz"])
    return rec.count()

def score_all(c: Connectome, keep: np.ndarray, A: dict, intact: dict | None = None, seed=0) -> dict:
    """Return {assay: (rate, score)} for a keep-mask; `intact` = baseline counts to normalise against."""
    W = masked(c.W, keep) if not keep.all() else c.W
    p = params(c); out = {}
    for k, a in A.items():
        cnt = run_assay(W, p, a, keep, seed)
        if a["out"] is None:                    # hearing: responders defined from the intact run
            if intact is None:
                fd = c.meta.fru_dsx.notna().values
                a["out"] = np.flatnonzero((cnt > 0) & fd)
            rate = float((cnt[a["out"]] > 0).sum()) if a["out"] is not None and a["out"].size else 0.0   # number of responders
        else:
            rate = float(cnt[a["out"]].mean() / (T_MS / 1000))
        base = intact[k][0] if intact else rate
        out[k] = (rate, rate / base if base > 0 else 0.0)
    return out
