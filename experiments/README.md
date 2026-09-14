# experiments

| script | what it does | output |
|---|---|---|
| `validate_male.py` | P1 → pIP10 song validation on the male brain | stdout |
| `female_cues_male.py` | present female cues to the male brain | stdout |
| `sweep.py` | parameter sweep for a stable, responsive regime | stdout |
| `run_arena.py` | two-brain courtship episode | `runs/*.json`, `runs/*.spikes.npz` |
| `export_web.py` | arena run → web bundle | `web/data/*.json` |
| `ablation_curves.py` | random + targeted deletion curves | `results/ablation_*.json` |
| `minimal_circuit.py` | smallest sub-network per capability | `results/minimal_*.json` |
| `export_net.py` | connectome → browser binary | `web/data/*_net.bin`, `*_meta.json` |
| `build_artifact.py` | bundle the two-brain page into one HTML | `dist/` |
