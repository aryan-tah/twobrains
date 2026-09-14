"""Load a connectome (MaleCNS v1.0 or FlyWire v783) into a unified sparse form.

Unified representation (cached as .npz):
  ids        : int64 [N]    original body/root ids
  W          : CSC  [N,N]   signed synaptic weight matrix, W[post, pre] = sign * count
  meta       : DataFrame    per-neuron annotations (type, superclass, side, nt, dimorphism, fru_dsx)
"""
from __future__ import annotations
import os, numpy as np, pandas as pd, scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(ROOT, "cache")

INHIBITORY = {"gaba", "glutamate"}   # Shiu et al. 2024 convention: GABA & Glu inhibitory, others excitatory
MIN_SYN = 5                          # drop connections with fewer than 5 synapses (the FlyWire paper's "strong connection" threshold)


def _sign(nt: pd.Series) -> np.ndarray:
    return np.where(nt.fillna("acetylcholine").str.lower().isin(INHIBITORY), -1, 1).astype(np.int8)


class Connectome:
    def __init__(self, name: str, ids, W: sp.csc_matrix, meta: pd.DataFrame):
        self.name, self.ids, self.W, self.meta = name, ids, W, meta
        self.N = len(ids)
        self.index = pd.Series(np.arange(self.N), index=ids)
        self.meta = self.meta.reset_index(drop=True)

    # ---- neuron selection helpers -------------------------------------------------
    def select(self, type_regex: str | None = None, side: str | None = None, **eq) -> np.ndarray:
        m = np.ones(self.N, bool)
        if type_regex:
            m &= self.meta["type"].fillna("").str.contains(type_regex, regex=True) if not any(c in type_regex for c in "(|") else self.meta["type"].fillna("").str.match(type_regex).values
        if side:
            m &= (self.meta["side"].fillna("") == side).values
        for k, v in eq.items():
            m &= (self.meta[k].fillna("") == v).values
        return np.flatnonzero(m)

    def types(self, idx) -> pd.Series:
        return self.meta.loc[idx, "type"].value_counts()

    def summary(self):
        return (f"{self.name}: {self.N:,} neurons, {self.W.nnz:,} connections "
                f"({int((self.W.data<0).sum()):,} inhibitory), "
                f"{int(abs(self.W).sum()):,} synapses")

    @classmethod
    def load(cls, name: str) -> "Connectome":
        path = os.path.join(CACHE, f"{name}.npz")
        metap = os.path.join(CACHE, f"{name}.meta.parquet")
        if not os.path.exists(path):
            {"malecns": build_malecns, "flywire": build_flywire}[name]()
        z = np.load(path)
        W = sp.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=(z["N"], z["N"]))
        return cls(name, z["ids"], W, pd.read_parquet(metap))


def _save(name, ids, W: sp.csc_matrix, meta: pd.DataFrame):
    os.makedirs(CACHE, exist_ok=True)
    W.sort_indices()
    np.savez(os.path.join(CACHE, f"{name}.npz"), ids=ids, data=W.data, indices=W.indices,
             indptr=W.indptr, N=W.shape[0])
    meta.to_parquet(os.path.join(CACHE, f"{name}.meta.parquet"))


# ---- MaleCNS v1.0 ---------------------------------------------------------------------
def build_malecns():
    d = os.path.join(DATA, "malecns")
    ann = pd.read_feather(os.path.join(d, "body-annotations.feather"))
    nt = pd.read_feather(os.path.join(d, "body-neurotransmitters.feather"))[["body", "consensus_nt"]]
    ann = ann.merge(nt, left_on="bodyId", right_on="body", how="left")
    # keep real neurons only: has a superclass and is not glia/orphan/unimportant
    bad = {"Glia", "Orphan", "Orphan-artifact", "Unimportant", "Out of scope", "Orphan hotknife", "PRT Orphan", "RT Orphan"}
    keep = ann.superclass.notna() & ~ann.statusLabel.isin(bad)
    ann = ann[keep].copy()
    meta = pd.DataFrame({
        "id": ann.bodyId.values, "type": ann.type.values, "flywire_type": ann.flywireType.values,
        "superclass": ann.superclass.values, "cls": ann["class"].values, "side": ann.somaSide.fillna(ann.rootSide).values,
        "nt": ann.consensus_nt.values, "dimorphism": ann.dimorphism.values, "fru_dsx": ann.fruDsx.values,
        "receptor": ann.receptorType.values, "instance": ann.instance.values,
    })
    ids = meta.id.values.astype(np.int64)
    index = pd.Series(np.arange(len(ids)), index=ids)
    w = pd.read_feather(os.path.join(d, "connectome-weights.feather"))
    w = w[(w.weight >= MIN_SYN) & w.body_pre.isin(index.index) & w.body_post.isin(index.index)]
    pre = index[w.body_pre.values].values; post = index[w.body_post.values].values
    sign = _sign(meta.nt)[pre]
    W = sp.csc_matrix((w.weight.values.astype(np.float32) * sign, (post, pre)), shape=(len(ids),) * 2)
    _save("malecns", ids, W, meta)
    print(Connectome("malecns", ids, W, meta).summary())


# ---- FlyWire v783 (female) ------------------------------------------------------------
def build_flywire():
    d = os.path.join(DATA, "flywire")
    ann = pd.read_csv(os.path.join(d, "flywire_annotations/supplemental_files/Supplemental_file1_neuron_annotations.tsv"),
                      sep="\t", low_memory=False)
    meta = pd.DataFrame({
        "id": ann.root_id.values, "type": ann.cell_type.values, "flywire_type": ann.cell_type.values,
        "superclass": ann.super_class.values, "cls": ann.cell_class.values, "side": ann.side.map({"left": "L", "right": "R", "center": "M"}).values,
        "nt": ann.top_nt.values, "dimorphism": ann.dimorphism.values, "fru_dsx": ann.fru_dsx.values,
        "receptor": np.full(len(ann), None), "instance": ann.hemibrain_type.values,
    })
    ids = meta.id.values.astype(np.int64)
    index = pd.Series(np.arange(len(ids)), index=ids)
    c = pd.read_feather(os.path.join(d, "proofread_connections_783.feather"),
                        columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"])
    c = c.groupby(["pre_pt_root_id", "post_pt_root_id"], as_index=False).syn_count.sum()  # sum over neuropils
    c = c[(c.syn_count >= MIN_SYN) & c.pre_pt_root_id.isin(index.index) & c.post_pt_root_id.isin(index.index)]
    pre = index[c.pre_pt_root_id.values].values; post = index[c.post_pt_root_id.values].values
    sign = _sign(meta.nt)[pre]
    W = sp.csc_matrix((c.syn_count.values.astype(np.float32) * sign, (post, pre)), shape=(len(ids),) * 2)
    _save("flywire", ids, W, meta)
    print(Connectome("flywire", ids, W, meta).summary())


if __name__ == "__main__":
    import sys
    for n in sys.argv[1:] or ["malecns"]:
        {"malecns": build_malecns, "flywire": build_flywire}[n]()
