"""Run a male + female two-brain courtship episode and save the log.
usage: uv run python experiments/run_arena.py [seconds] [out.json] [male_arousal_hz] [female_arousal_hz] [male_knockout_regex,...] [female_knockout_regex,...]"""
import sys, json, math; sys.path.insert(0, ".")
from twobrains.connectome import Connectome
from twobrains.lif import LIFParams
from twobrains.arena import Arena, Fly, Body

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 10
out = sys.argv[2] if len(sys.argv) > 2 else "runs/episode.json"
ma = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
fa = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
mk = sys.argv[5].split(",") if len(sys.argv) > 5 and sys.argv[5] else None
fk = sys.argv[6].split(",") if len(sys.argv) > 6 and sys.argv[6] else None
M = Connectome.load("malecns"); F = Connectome.load("flywire")
male = Fly("male", "male", M, Body(-4.0, 0.0, 0.0), seed=1, arousal_hz=ma, knockout=mk)
female = Fly("female", "female", F, Body(4.0, 0.5, 180.0), seed=2, arousal_hz=fa, knockout=fk)
print(f"male arousal {ma} Hz knockout {mk} ({male.knocked.size} neurons) | female arousal {fa} Hz knockout {fk} ({female.knocked.size} neurons)")
print("male senses", {k: v.size for k, v in male.sense.items()}, "motor", {k: v.size for k, v in male.motor.items()})
print("female senses", {k: v.size for k, v in female.sense.items()}, "motor", {k: v.size for k, v in female.motor.items()})
arena = Arena(male, female)
arena.run(secs, verbose_every=0.5)
arena.save(out); print("saved", out)
