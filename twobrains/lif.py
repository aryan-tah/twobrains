"""Event-driven leaky integrate-and-fire simulation of a whole connectome.

Parameters follow Shiu et al. 2024 (Nature), "A Drosophila computational brain model
reveals sensorimotor processing":
    dv/dt = (v_rest - v + g) / tau_m          (unless refractory)
    dg/dt = -g / tau_syn
    spike when v >= v_th -> v = v_reset, refractory t_rfc
    each presynaptic spike adds  w_syn * (signed synapse count)  to g after delay t_dly
"""
from __future__ import annotations
import numpy as np, scipy.sparse as sp
from dataclasses import dataclass, field


@dataclass
class LIFParams:
    dt: float = 0.1        # ms
    tau_m: float = 20.0    # ms
    tau_syn: float = 5.0   # ms
    v_rest: float = -52.0  # mV
    v_reset: float = -52.0
    v_th: float = -45.0
    t_rfc: float = 2.2     # ms
    t_dly: float = 1.8     # ms
    w_syn: float = 0.275   # mV per synapse
    w_cap: float = 0.0     # cap on |synapse count| per connection (0 = no cap)
    b_adapt: float = 0.0   # mV threshold increase per spike (spike-frequency adaptation); 0 = off
    tau_adapt: float = 100.0  # ms decay of adaptation
    g_cap: float = 0.0     # if >0: clip synaptic drive g to [-g_cap, +g_cap] mV (reversal-potential-like saturation)
    norm_all: bool = False # with in_norm: rescale EVERY neuron's inputs to exactly in_norm (up and down), i.e. a
                           # "fraction of inputs active" model instead of absolute synapse counts
    in_norm: float = 0.0   # if >0: scale a neuron's incoming weights so its excitatory in-synapses sum to <= in_norm


class LIFBrain:
    def __init__(self, W: sp.csc_matrix, p: LIFParams = LIFParams(), seed: int = 0):
        self.W = W.tocsc().astype(np.float32)
        if p.w_cap:
            self.W.data = np.clip(self.W.data, -p.w_cap, p.w_cap)
        if p.in_norm:
            exc_in = np.asarray(self.W.maximum(0).sum(axis=1)).ravel()
            scale = (p.in_norm / np.maximum(exc_in, 1e-9)).astype(np.float32)
            if not p.norm_all:
                scale = np.minimum(1.0, scale)
            scale[exc_in <= 0] = 1.0
            self.W = sp.diags(scale) @ self.W          # row (postsynaptic) scaling
            self.W = self.W.tocsc().astype(np.float32)
        self.N = W.shape[0]
        self.p = p
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        p = self.p
        self.v = np.full(self.N, p.v_rest, np.float32)
        self.g = np.zeros(self.N, np.float32)
        self.rfc_until = np.zeros(self.N, np.float32)
        self.a = np.zeros(self.N, np.float32)          # adaptive threshold offset
        self.t = 0.0
        self.delay_steps = max(1, int(round(p.t_dly / p.dt)))
        self.queue: list[np.ndarray] = [np.empty(0, np.int64) for _ in range(self.delay_steps)]
        self.qi = 0
        self._dm = np.exp(-p.dt / p.tau_m)
        self._ds = np.exp(-p.dt / p.tau_syn)
        self._da = np.exp(-p.dt / p.tau_adapt)

    def _deliver(self, spiked: np.ndarray):
        """Add synaptic input from presynaptic spikes (columns of W)."""
        if spiked.size == 0:
            return
        sub = self.W[:, spiked]                      # CSC column gather, cheap
        inp = np.asarray(sub.sum(axis=1)).ravel()    # dense N (float32)
        self.g += (self.p.w_syn * inp).astype(np.float32)

    def step(self, poisson_idx: np.ndarray | None = None, poisson_rate_hz: float = 0.0,
             current_idx: np.ndarray | None = None, current_mv: float = 0.0) -> np.ndarray:
        """Advance one dt. Returns indices of neurons that spiked this step.

        poisson_idx : neurons receiving external Poisson spikes at poisson_rate_hz; each external
                      spike forces one output spike (equivalent to Shiu et al.'s w_syn*f_poi drive)
        current_idx : neurons receiving a constant depolarising drive (mV added to g each ms)
        """
        p = self.p
        # 1. deliver delayed spikes
        self._deliver(self.queue[self.qi])
        # 2. external drive
        if poisson_idx is not None and poisson_idx.size and poisson_rate_hz > 0:
            k = self.rng.random(poisson_idx.size) < poisson_rate_hz * p.dt / 1000.0
            self.v[poisson_idx[k]] = 1e3          # external spike: force the neuron to fire (bypasses g_cap and adaptive threshold)
        if current_idx is not None and current_idx.size and current_mv:
            self.g[current_idx] += current_mv * p.dt
        # 3. integrate (exact exponential Euler for the linear parts)
        active = self.t >= self.rfc_until
        self.v = np.where(active, p.v_rest + (self.v - p.v_rest) * self._dm + self.g * (1 - self._dm), self.v)
        self.g *= self._ds
        if p.g_cap:
            np.clip(self.g, -p.g_cap, p.g_cap, out=self.g)
        # 4. spikes
        if p.b_adapt:
            self.a *= self._da
            spiked = np.flatnonzero((self.v >= p.v_th + self.a) & active)
            self.a[spiked] += p.b_adapt
        else:
            spiked = np.flatnonzero((self.v >= p.v_th) & active)
        if spiked.size:
            self.v[spiked] = p.v_reset
            self.rfc_until[spiked] = self.t + p.t_rfc
        self.queue[self.qi] = spiked
        self.qi = (self.qi + 1) % self.delay_steps
        self.t += p.dt
        return spiked

    def run(self, t_ms: float, record: bool = True, **drive) -> "SpikeRecord":
        n = int(round(t_ms / self.p.dt))
        times, ids = [], []
        for i in range(n):
            s = self.step(**drive)
            if record and s.size:
                times.append(np.full(s.size, self.t, np.float32)); ids.append(s)
        return SpikeRecord(np.concatenate(times) if times else np.empty(0, np.float32),
                           np.concatenate(ids) if ids else np.empty(0, np.int64), t_ms, self.N)


@dataclass
class SpikeRecord:
    t: np.ndarray
    i: np.ndarray
    duration_ms: float
    N: int

    def rate(self, idx=None) -> np.ndarray:
        """Mean firing rate (Hz) per neuron over the record."""
        c = np.bincount(self.i, minlength=self.N).astype(float) / (self.duration_ms / 1000.0)
        return c if idx is None else c[idx]

    def count(self):
        return np.bincount(self.i, minlength=self.N)
