"""Two-fly courtship arena: couples two independent connectome brains through the world.

World tick = 10 ms of simulated time. Each tick:
  1. world state -> sensory Poisson rates for each brain (vision, pheromone, song)
  2. each brain advances 10 ms of LIF dynamics
  3. motor read-outs (descending neuron spike counts) -> body velocity / song / receptivity
  4. bodies move; everything logged

Every sensory and motor mapping is an explicit, documented *engineering* assignment based on
the known function of that cell type. Nothing here claims the simulated fly "feels" anything.
"""
from __future__ import annotations
import json, math, time, numpy as np
from dataclasses import dataclass, field, asdict
from .connectome import Connectome
from .lif import LIFBrain, LIFParams

ARENA_R = 10.0      # mm  (standard courtship chamber ~ 20 mm diameter)

# Calibrated LIF regimes (see experiments/sweep.py). The male dataset reports ~2.3x more input
# synapses per neuron than FlyWire, so its per-synapse gain is scaled down accordingly.
MALE_PARAMS = LIFParams(w_syn=0.2, g_cap=30, b_adapt=3, tau_adapt=200)
FEMALE_PARAMS = LIFParams(w_syn=0.275, g_cap=30, b_adapt=3, tau_adapt=200)
TICK_MS = 10.0
BODY_LEN = 2.5      # mm
MAX_SPEED = 15.0    # mm/s  walking
MAX_TURN = 500.0    # deg/s
VISION_RANGE = 12.0 # mm   LC10a: small-object (fly-sized) detector, frontal visual field
SMELL_RANGE = 5.0   # mm   volatile pheromone (Or47b/ORN_VA1v) plume
TOUCH_RANGE = 1.2   # mm   contact pheromone (ppk23 leg gustatory, 7,11-HD)
SONG_RANGE = 8.0    # mm   pulse song audible range (JO-A/B)
MAX_RATE = 150.0    # Hz   Poisson rate at full stimulus


def _wrap(a):  # degrees -> [-180, 180)
    return (a + 180.0) % 360.0 - 180.0


@dataclass
class Body:
    x: float; y: float; heading: float   # mm, mm, degrees (0 = +x)
    speed: float = 0.0; turn: float = 0.0
    singing: float = 0.0                 # 0..1 pulse-song intensity this tick
    accept: float = 0.0; reject: float = 0.0

    def rel(self, other: "Body"):
        """distance (mm) and bearing (deg, signed, 0 = dead ahead, +left) to the other fly."""
        dx, dy = other.x - self.x, other.y - self.y
        d = math.hypot(dx, dy)
        bearing = _wrap(math.degrees(math.atan2(dy, dx)) - self.heading)
        return d, bearing

    def move(self, dt_s: float):
        self.heading = (self.heading + self.turn * dt_s) % 360.0
        self.x += self.speed * dt_s * math.cos(math.radians(self.heading))
        self.y += self.speed * dt_s * math.sin(math.radians(self.heading))
        r = math.hypot(self.x, self.y)
        if r > ARENA_R - BODY_LEN / 2:      # wall: slide along it and bounce heading inward
            k = (ARENA_R - BODY_LEN / 2) / r
            self.x *= k; self.y *= k
            self.heading = (math.degrees(math.atan2(-self.y, -self.x)) + np.random.uniform(-60, 60)) % 360


class Fly:
    """A connectome brain + body + the sensory/motor mapping for its sex."""

    def __init__(self, name: str, sex: str, conn: Connectome, body: Body, params: LIFParams | None = None, seed=0,
                 arousal_hz: float = 0.0, knockout: list[str] | None = None):
        """arousal_hz : tonic Poisson drive (Hz) to the sex-specific command neurons (male P1 / female pC1a),
                       standing in for the neuromodulatory internal state the connectome does not contain.
           knockout   : list of type regexes whose neurons are silenced (all their synapses removed)."""
        self.name, self.sex, self.c, self.body = name, sex, conn, body
        params = params or (MALE_PARAMS if sex == "male" else FEMALE_PARAMS)
        W = conn.W
        self.knocked = np.empty(0, int)
        if knockout:
            self.knocked = np.unique(np.concatenate([conn.select(k) for k in knockout]))
            keep = np.ones(conn.N, np.float32); keep[self.knocked] = 0
            import scipy.sparse as sp
            W = (sp.diags(keep) @ W @ sp.diags(keep)).tocsc()
        self.brain = LIFBrain(W, params, seed=seed)
        self.arousal_hz = arousal_hz
        self.rng = np.random.default_rng(seed + 1)
        s = conn.select
        # ---- sensory populations (side-resolved) --------------------------------------
        self.sense = {
            "vision_L": s(r"^LC10a$", side="L"), "vision_R": s(r"^LC10a$", side="R"),
            "smell":    s(r"^ORN_VA1v$"),                       # Or47b: fly-derived pheromones, drives courtship
            "cVA":      s(r"^ORN_DA1$"),                        # Or67d: male pheromone cVA
            "song":     s(r"^JO-(A|B)"),                        # Johnston's organ, pulse-song tuned
            "touch_L":  s(receptor="putative_ppk23", side="L") if sex == "male" else np.empty(0, int),
            "touch_R":  s(receptor="putative_ppk23", side="R") if sex == "male" else np.empty(0, int),
        }
        # ---- motor read-outs ----------------------------------------------------------
        self.motor = {
            "turn_L": s(r"^DNa02$", side="L"), "turn_R": s(r"^DNa02$", side="R"),   # DNa02: ipsilateral turning
            "forward": s(r"^DNp09$"),                                                # DNp09: forward walking / pursuit
            "song":   s(r"^pIP10$") if sex == "male" else np.empty(0, int),         # pulse song command
            "accept": s(r"^DNp37$") if sex == "female" else np.empty(0, int),       # vpoDN: vaginal plate opening
            "reject": s(r"^DNp13$") if sex == "female" else np.empty(0, int),       # ovipositor extrusion
        }
        self.state_pop = s(r"^pC1_") if sex == "male" else s(r"^pC1a$")   # arousal target
        self.watch = {k: v for k, v in {**self.sense, **self.motor,
                      "P1": s(r"^pC1_") if sex == "male" else s(r"^pC1[a-e]$")}.items() if v.size}
        self._counts = np.zeros(conn.N, np.int64)
        self.spike_log: list[np.ndarray] = []      # per tick: indices of neurons that spiked (full resolution)

    # ------------------------------------------------------------------------------------
    def sensory_rates(self, other: "Fly") -> dict[str, float]:
        d, bearing = self.body.rel(other.body)
        r = {}
        vis = max(0.0, 1 - d / VISION_RANGE) if abs(bearing) < 90 else 0.0
        r["vision_L"] = MAX_RATE * vis * (0.5 + 0.5 * max(0, math.sin(math.radians(bearing))))
        r["vision_R"] = MAX_RATE * vis * (0.5 + 0.5 * max(0, -math.sin(math.radians(bearing))))
        r["smell"] = MAX_RATE * max(0.0, 1 - d / SMELL_RANGE)
        r["cVA"] = MAX_RATE * max(0.0, 1 - d / SMELL_RANGE) if other.sex == "male" else 0.0
        touch = 1.0 if d < TOUCH_RANGE else 0.0
        r["touch_L"] = MAX_RATE * touch * (1.0 if bearing > 0 else 0.3)
        r["touch_R"] = MAX_RATE * touch * (1.0 if bearing < 0 else 0.3)
        r["song"] = MAX_RATE * other.body.singing * max(0.0, 1 - d / SONG_RANGE)
        return r

    def tick(self, rates: dict[str, float], background_hz: float = 0.0):
        """Advance the brain TICK_MS with the given sensory Poisson rates; return spike counts per watched pop."""
        n = int(TICK_MS / self.brain.p.dt)
        self._counts[:] = 0
        pops = [(self.sense[k], v) for k, v in rates.items() if v > 0 and self.sense[k].size]
        if self.arousal_hz > 0:
            pops.append((self.state_pop, self.arousal_hz))
        p = self.brain.p
        for _ in range(n):
            v = self.brain.v
            for idx, hz in pops:
                k = self.rng.random(idx.size) < hz * p.dt / 1000.0
                v[idx[k]] = 1e3
            if background_hz:
                k = self.rng.random(self.brain.N) < background_hz * p.dt / 1000.0
                v[k] = 1e3
            sp = self.brain.step()
            self._counts[sp] += 1
        self.spike_log.append(np.flatnonzero(self._counts).astype(np.int32))
        out = {k: float(self._counts[v].mean()) for k, v in self.watch.items()}
        out["_total"] = int(self._counts.sum()); out["_active"] = int((self._counts > 0).sum())
        return out

    def act(self, out: dict[str, float]):
        """Map motor neuron spike counts (per 10 ms, mean over population) to body commands."""
        b = self.body
        rate = lambda k: out.get(k, 0.0) * (1000.0 / TICK_MS)          # -> Hz
        b.turn = float(np.clip((rate("turn_L") - rate("turn_R")) / 100.0, -1, 1)) * MAX_TURN
        # forward drive: DNp09 (forward walking) plus bilateral DNa02 activity (turning-while-walking)
        b.speed = float(np.clip((rate("forward") + 0.5 * (rate("turn_L") + rate("turn_R"))) / 100.0, 0, 1)) * MAX_SPEED
        b.singing = float(np.clip(rate("song") / 100.0, 0, 1))
        b.accept = float(np.clip(rate("accept") / 100.0, 0, 1))
        b.reject = float(np.clip(rate("reject") / 100.0, 0, 1))


class Arena:
    def __init__(self, male: Fly, female: Fly, background_hz: float = 0.0):
        self.male, self.female, self.bg = male, female, background_hz
        self.t = 0.0
        self.log: list[dict] = []

    def step(self):
        rm = self.male.sensory_rates(self.female)
        rf = self.female.sensory_rates(self.male)
        om = self.male.tick(rm, self.bg)
        of = self.female.tick(rf, self.bg)
        self.male.act(om); self.female.act(of)
        self.male.body.move(TICK_MS / 1000); self.female.body.move(TICK_MS / 1000)
        d, _ = self.male.body.rel(self.female.body)
        self.t += TICK_MS
        self.log.append({"t": self.t, "dist": d,
                         "male": {"body": asdict(self.male.body), "in": rm, "out": om},
                         "female": {"body": asdict(self.female.body), "in": rf, "out": of}})
        return self.log[-1]

    def run(self, seconds: float, verbose_every: float = 1.0):
        n = int(seconds * 1000 / TICK_MS); t0 = time.time()
        for i in range(n):
            rec = self.step()
            if verbose_every and (i + 1) % int(verbose_every * 1000 / TICK_MS) == 0:
                m, f = rec["male"], rec["female"]
                print(f"t={rec['t']/1000:5.1f}s d={rec['dist']:4.1f}mm | M P1={m['out'].get('P1',0)*100:5.1f}Hz "
                      f"song={m['body']['singing']:.2f} v={m['body']['speed']:4.1f} | "
                      f"F pC1={f['out'].get('P1',0)*100:5.1f}Hz acc={f['body']['accept']:.2f} rej={f['body']['reject']:.2f} "
                      f"v={f['body']['speed']:4.1f} | {time.time()-t0:5.0f}s wall")
        return self.log

    def save(self, path: str):
        """Writes <path> (JSON: bodies, population rates per tick) and <path>.spikes.npz (which neurons fired each tick)."""
        with open(path, "w") as fh:
            json.dump({"tick_ms": TICK_MS, "arena_r": ARENA_R, "male": {"n": self.male.c.N, "arousal_hz": self.male.arousal_hz, "knockout": self.male.knocked.tolist()},
                       "female": {"n": self.female.c.N, "arousal_hz": self.female.arousal_hz, "knockout": self.female.knocked.tolist()}, "log": self.log}, fh)
        def pack(fly):
            ptr = np.zeros(len(fly.spike_log) + 1, np.int64); ptr[1:] = np.cumsum([s.size for s in fly.spike_log])
            return np.concatenate(fly.spike_log) if fly.spike_log else np.empty(0, np.int32), ptr
        mi, mp = pack(self.male); fi, fp = pack(self.female)
        np.savez_compressed(path.replace(".json", "") + ".spikes.npz", male_idx=mi, male_ptr=mp, female_idx=fi, female_ptr=fp)
