# TwoBrains · Delete the Brain

[![license](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE) [![data](https://img.shields.io/badge/data-CC--BY_4.0-green.svg)](data/README.md)

**How much of a real brain can you delete before it stops working?**

A complete fruit fly brain, 139,248 neurons and 2.7 million connections measured by electron microscopy
(FlyWire v783), runs as a spiking neural network live in a browser tab. Neurons are deleted, at random or
by strategy, and five real circuits are re-tested each time: driven at their real sensory inputs, read at
their real descending outputs. Nothing is trained. The only thing deciding whether a test passes is the wiring.

`web/delete.html` is a self-running 20-second story: intact brain → 25% → 50% → 75% → 90% deleted → the whole
left hemisphere deleted → the brain collapsed to its 316-neuron escape circuit → to 4 neurons. Every step is a
live simulation, not a recording.

## Contents

1. [Results](#results)
2. [How it works](#how-it-works)
3. [Run it](#run-it)
4. [Caveats](#caveats-stated-plainly)
5. [Data](#data)

## Results

Random deletion, mean of 3 seeds, score relative to the intact brain (female brain, FlyWire):

| deleted | steer toward target | escape looming threat | walk / turn | hear song | accept mate |
|---:|---:|---:|---:|---:|---:|
| 10 % | 0.88 | 1.01 | 1.07 | 0.85 | 1.00 |
| 30 % | 0.88 | 1.04 | 1.00 | 0.85 | 1.00 |
| 50 % | 0.54 | 1.05 | 0.73 | 0.87 | 1.00 |
| 70 % | 0.29 | 1.05 | 0.60 | 0.87 | 1.00 |
| 90 % | 0.12 | 1.08 | 0.20 | 0.92 | 1.00 |

Targeted deletion (same brain):

| delete… | neurons removed | steer | escape | walk | hear | accept |
|---|---:|---:|---:|---:|---:|---:|
| both optic lobes | 77,541 | 1.00 | 1.00 | 1.00 | 0.93 | 1.00 |
| the entire left hemisphere | 69,463 | **0.00** | 1.04 | 0.90 | 0.93 | 1.00 |
| the entire right hemisphere | 68,651 | 1.00 | 1.04 | 0.90 | 0.87 | 1.00 |
| all central-brain interneurons | 32,371 | **0.00** | 1.08 | **0.00** | 1.03 | 1.00 |
| all inhibitory (GABA) neurons | 19,144 | 1.12 | 1.04 | **2.00** | 1.05 | 1.00 |
| the top 10 % most-connected neurons | 13,881 | **0.00** | 1.08 | 0.20 | 0.95 | 1.00 |
| the top 1 % most-connected neurons | 1,386 | 0.88 | 1.08 | 1.60 | 0.93 | 1.06 |
| the mushroom body (all Kenyon cells) | 5,177 | 1.00 | 1.00 | 1.00 | 0.93 | 1.00 |
| the central complex | 2,875 | 1.00 | 1.00 | 1.00 | 0.93 | 1.00 |

The smallest sub-network that still performs each behaviour (delta-debugging over the neurons that fire when intact,
keeping ≥ 50 % of the intact response):

| capability | neurons | share of brain | what is left |
|---|---:|---:|---|
| accept a mate | **4** | 0.003 % | 2 × pC1a, 2 × vpoDN |
| steer toward a target | 120 | 0.09 % | 115 × LC10a, 1 × DNa02, and 4 interneurons (LAL027, AOTU027, AOTUv3B, CB0359) |
| walk / turn | 248 | 0.18 % | LC10a, DNa02, DNp09, DNa01, LAL027, AOTU015a and 2 others |
| escape a looming threat | 316 | 0.23 % | 210 × LPLC2, 104 × LC4, 2 × DNp01. No interneurons at all |
| hear a song | 372 | 0.27 % | Johnston's organ neurons and 13 Fru⁺/Dsx⁺ auditory neurons |

The same experiments on the male central nervous system (MaleCNS v1.0, 164,611 neurons incl. nerve cord) give the same
picture: escape survives 90 % random deletion (1.02), song survives it (1.08), steering falls to 0.25, walking is gone
by 70 %. Deleting all 22,061 GABAergic neurons *increases* walking output 6.5× (disinhibition). The minimal escape
circuit is 313 neurons: LPLC2 + LC4 + the giant fiber. Full numbers in `results/`.

**What it means.** Reflexes that matter for survival are wired short and wide: hundreds of looming detectors converge
directly on the giant fiber, so no single neuron matters and the reflex is immune to random damage. Behaviours that
need integration, like steering toward a small target, run through a handful of hub interneurons in the lateral
accessory lobe and die when those are hit. Deleting whole brain regions that are famous for learning and navigation
(mushroom body, central complex) changes none of these five reflex tests, which is exactly what you would expect for
tests that do not involve learning or navigation.

## How it works

* `twobrains/connectome.py` loads MaleCNS v1.0 or FlyWire v783 into a signed sparse matrix (GABA/glutamate inhibitory, ≥ 5 synapses).
* `twobrains/lif.py` is an event-driven leaky integrate-and-fire simulator with the parameters of Shiu et al. 2024
  (τm 20 ms, threshold −45 mV, 0.275 mV per synapse, 1.8 ms delay), plus a 30 mV synaptic-drive ceiling and
  spike-frequency adaptation (3 mV / spike, τ 200 ms) which are required to keep the network out of a seizure state.
  The male gain is calibrated to 0.2 mV because MaleCNS reports ≈ 2.3× more input synapses per neuron than FlyWire.
* `twobrains/ablation.py` defines the capability assays and masks neurons out of the matrix.
* `experiments/ablation_curves.py`, `experiments/minimal_circuit.py` produce `results/`.
* `web/lif-worker.js` is the same model in JavaScript, integrating only non-resting neurons, so a whole-brain test takes
  ~0.3–1 s in the browser. `experiments/export_net.py` writes the binary it loads.
* `web/index.html` is the earlier two-brain scene (male + female brains coupled through courtship channels), kept as a second chapter.

## Run it

```bash
uv sync                                                   # see data/README.md for the ~2 GB of CC-BY data
uv run python twobrains/connectome.py malecns flywire     # build caches
uv run python experiments/ablation_curves.py flywire      # ~5 min
uv run python experiments/minimal_circuit.py flywire      # ~3 min
uv run python experiments/export_net.py flywire           # binary for the browser
cd web && python3 -m http.server 8000                     # open http://localhost:8000/delete.html
```

## Caveats, stated plainly

This is a connectome-derived computational model, not a fly. Every assay is an engineering choice grounded in the
published function of a cell type. Input and read-out neurons of an assay are never deleted (otherwise every test is
trivially destroyed). Scores are firing rates over 300 ms relative to the intact network; "hear" counts responding
Fru⁺/Dsx⁺ auditory neurons. The model has no neuromodulation, no plasticity, and no internal state.

## Data

MaleCNS v1.0 (Janelia FlyEM, Cambridge Connectomics, Google Research, 2026) · FlyWire FAFB v783 (Dorkenwald et al.,
Schlegel et al., 2024) · annotations from flyconnectome/flywire_annotations. All CC-BY 4.0.
