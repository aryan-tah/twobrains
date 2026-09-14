import numpy as np, scipy.sparse as sp
from twobrains.lif import LIFBrain, LIFParams


def test_delay_ring_size():
    b = LIFBrain(sp.csc_matrix((2, 2), dtype=np.float32), LIFParams(dt=0.1, t_dly=1.8))
    assert b.delay_steps == 18


def test_g_cap_limits_drive():
    W = sp.csc_matrix((np.array([5000.0], np.float32), ([1], [0])), shape=(2, 2))
    b = LIFBrain(W, LIFParams(g_cap=30.0))
    b.run(20, poisson_idx=np.array([0]), poisson_rate_hz=500)
    assert b.g.max() <= 30.0 + 1e-3


def test_adaptation_raises_threshold():
    W = sp.csc_matrix((2, 2), dtype=np.float32)
    b = LIFBrain(W, LIFParams(b_adapt=3.0))
    b.run(50, poisson_idx=np.array([0]), poisson_rate_hz=400)
    assert b.a[0] > 0
