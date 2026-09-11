# TwoBrains

**Two real brains. One arena. A male fruit fly brain and a female fruit fly brain, simulated together.**

Every viral fly-brain demo so far runs one connectome inside one game (Doom, Mario, Minecraft, Beat Saber).
This is different: the September 2026 male connectome (MaleCNS v1.0, Janelia / Google) and the female
FlyWire brain (FAFB v783) are simulated as two independent spiking networks, and they are wired to each
other only through the channels real flies use to court:

```
            ♂ MALE BRAIN (164,611 neurons)                ♀ FEMALE BRAIN (139,248 neurons)
            ────────────────────────────                  ──────────────────────────────
  sees her  LC10a  ──► ... ──► DNa02  turns toward her    hears him  JO-A/B ──► ... ──► pC1 ──► vpoDN  accept
  smells her ORN_VA1v ──► ...                             sees him   LC10a  ──► ... ──► DNa02  turns
  touches her ppk23 ──► ... ──► DNp09 walks               smells him ORN_DA1 (cVA)           DNp13  reject
  P1 ──► pIP10  sings ─────────── pulse song ────────────────────────► JO-A/B
```

Nothing in this repository is trained. The only thing that decides what each fly does is the
synaptic wiring measured by electron microscopy, plus a documented sensory/motor mapping.

## What it does

`twobrains/` is a small, fast, event-driven leaky integrate-and-fire simulator for whole connectomes
(≈2 s of wall time per 0.5 s of simulated brain time on an M4 Pro, for 165k neurons / 6.2M connections).
`twobrains/arena.py` couples two brains through a 20 mm courtship chamber: every 10 ms of simulated time
the world is converted into Poisson drive on the real sensory neuron types, both brains step forward,
descending-neuron spike counts are read out as body commands, and the bodies move.

`web/index.html` replays a run: both brains as soma-position point clouds flashing in real time, the arena,
and the courtship read-outs.

## Validation before behaviour

The model is only interesting if it reproduces known optogenetics. It does:

| Known result | In the model |
|---|---|
| Activating male P1 neurons drives pulse song via the descending neuron pIP10 (von Philipsborn 2011, Ding 2019) | P1 drive → pIP10 fires |
| Female pC1a neurons drive vaginal plate opening via vpoDN (Wang 2020) | pC1a drive → vpoDN fires |
| DNa02 drives ipsilateral turning (Rayshubskiy 2020) | left-eye LC10a input → left DNa02 only; right → right only, in both sexes |
| Leg contact-pheromone neurons (ppk23) promote courtship pursuit | ppk23 drive → DNp09 (forward walking) |

## Results (20 s episodes, both flies start 8 mm apart)

| condition | ♂ P1 arousal | ♀ pC1a arousal | knockout | time within 2 mm | ♂ song DN (pIP10) | ♀ song input (JO) | ♀ active neurons / 10 ms | ♀ accept DN (vpoDN) |
|---|---|---|---|---|---|---|---|---|
| pure wiring | 0 | 0 | – | 26 % | 0.1 Hz | 0 Hz | 171 | 0 |
| **courtship** | 10 Hz | 0 | – | **97 %** | **11 Hz** | **44 Hz** | **308** | 0 |
| courtship, receptive ♀ | 10 Hz | 20 Hz | – | 27 % | 11 Hz | 39 Hz | 280 | **4.3 Hz** |
| mute ♂ | 10 Hz | 20 Hz | ♂ pIP10 silenced | 62 % | 0 | 0 | 173 | 4.7 Hz |
| blind ♂ | 10 Hz | 20 Hz | ♂ LC10a silenced | **0 %** | 11 Hz | 26 Hz | 168 | 4.4 Hz |

What this says, in order:

1. **The male finds her by sight.** Left-eye LC10a input drives only the left DNa02, right drives right, so he turns toward
   her and closes the distance. Silence his 275 LC10a neurons and he never gets within 2 mm.
2. **A little internal state turns him into a singer.** 10 Hz of tonic P1 drive on its own gives ~11 Hz pIP10 output; the
   female cues add to it (state gating, as in the real fly).
3. **His song is the channel into her brain.** With song, her whole-brain activity nearly doubles (171 → 308 active neurons
   per 10 ms) and ~30 Fru+/Dsx+ auditory neurons light up. Mute him and her activity is back at baseline.
4. **Her wiring alone does not say yes.** No sensory input we tried (song, continuous or pulsed at 35 ms; cVA; vision;
   all combined) drives her pC1 or vpoDN. Song input in fact *suppresses* vpoDN slightly, consistent with the Fru+ global
   inhibition (aSP8-like) that sharpens song tuning. Female receptivity in this model, as in the real animal, needs internal
   state: 20 Hz of pC1a drive is enough for vpoDN to fire.

So the honest one-liner is: *two real brains, wired to each other through the real channels, produce pursuit, song and
hearing on wiring alone; consent needs a hormone the connectome does not contain.*

## Model

Leaky integrate-and-fire neurons with the parameters of Shiu et al. 2024 (Nature): τm 20 ms, rest/reset −52 mV,
threshold −45 mV, τsyn 5 ms, 1.8 ms delay, 2.2 ms refractory, 0.275 mV per synapse, GABA and glutamate inhibitory.
Connections with fewer than 5 synapses are dropped.

Two additions were necessary and are documented in `twobrains/lif.py`:

* **Cross-dataset gain calibration.** The MaleCNS synapse detector reports ≈2.3× more input synapses per neuron
  than FlyWire (mean 341 vs 146 excitatory in-synapses). With the published gain, stimulating just 5 neurons drives
  the entire male network into a permanent seizure. The male's per-synapse gain is scaled to 0.2 mV.
* **Saturation and adaptation.** Synaptic drive is capped at 30 mV (a reversal-potential-like ceiling) and neurons
  have spike-frequency adaptation (3 mV per spike, τ 200 ms). Without these, strong olfactory input locks the
  mushroom body and central complex into self-sustaining activity that never switches off.

What the connectome does **not** contain is internal state: neuromodulators, hormones, mating status. In real flies
P1 is gated by these, and so is female receptivity. The arena therefore exposes one knob per fly, `arousal_hz`,
a tonic drive to P1 (male) or pC1a (female), which is exactly what the optogenetic experiments above do.
With it at zero you get the pure-wiring behaviour; turning it up is the "optogenetic" condition.

## Run it

```bash
uv sync
# data (CC-BY 4.0): MaleCNS flat connectome from gs://flyem-male-cns, FlyWire edges from Zenodo 10676866,
# FlyWire annotations from github.com/flyconnectome/flywire_annotations  — see data/README.md
uv run python twobrains/connectome.py malecns flywire        # build sparse caches
uv run python experiments/validate_male.py                   # P1 -> pIP10
uv run python experiments/run_arena.py 20 runs/aroused.json 40 0    # 20 s episode, male P1 arousal 40 Hz
uv run python experiments/export_web.py runs/aroused.json web/data/aroused.json
cd web && python3 -m http.server 8000                        # open http://localhost:8000
```

Knockouts: `run_arena.py 20 runs/no_song.json 40 0 '^pIP10$'` silences the male's song neurons.

## Data

* MaleCNS v1.0 — Janelia FlyEM, Cambridge Connectomics Group, Google Research (2026). CC-BY 4.0.
* FlyWire FAFB v783 — Dorkenwald et al. 2024, Schlegel et al. 2024. CC-BY 4.0.
* Neuron annotations — flyconnectome/flywire_annotations; MaleCNS body annotations.

## Honest caveats

This is a connectome-derived computational model, not a fly. Every sensory and motor mapping is an engineering
assignment based on the published function of that cell type. The simulated brains do not learn, feel, or want
anything. The claim being tested is narrow and falsifiable: *does the measured wiring, on its own, route the
sensory signals of one sex's brain to the courtship outputs that the other sex's brain can perceive?*
