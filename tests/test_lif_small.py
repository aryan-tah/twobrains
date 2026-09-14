"""Unit tests for the LIF simulator on a tiny hand-built network (no connectome data needed)."""
import numpy as np, scipy.sparse as sp
from twobrains.lif import LIFBrain, LIFParams


def chain(n=3, w=60.0):
    """0 -> 1 -> 2 ... with strong excitatory synapses."""
    rows, cols, vals = [], [], []
    for i in range(n - 1):
        rows.append(i + 1); cols.append(i); vals.append(w)
    return sp.csc_matrix((np.array(vals, np.float32), (rows, cols)), shape=(n, n))


def test_silent_without_input():
    b = LIFBrain(chain())
    rec = b.run(100)
    assert rec.i.size == 0


def test_spike_propagates_down_chain():
    b = LIFBrain(chain(), LIFParams(w_syn=0.275))
    rec = b.run(100, poisson_idx=np.array([0]), poisson_rate_hz=200)
    counts = rec.count()
    assert counts[0] > 0 and counts[1] > 0 and counts[2] > 0


def test_inhibitory_synapse_never_excites():
    W = chain(2, w=-60.0)
    b = LIFBrain(W)
    rec = b.run(100, poisson_idx=np.array([0]), poisson_rate_hz=200)
    assert rec.count()[1] == 0
