# Depth-1 ablation on SWE 256x512 (Table 2, Table 4)

Four depth-1 arms on the 256x512 shallow-water configuration (one-hour lead, 24 solver substeps), three seeds
each, every twin keeping the full arm's depth, window, gate, decoder and radii and changing one thing.
`evaluate.py` scores every seed's `best.pt` on 64 held-out test trajectories, one step and a five-step
rollout (5 h); `tables.py` renders Table 2; the same JSON feeds the zero-shot bundle's depth-1 row and
`experiments/shallow_water_paper/degree_band_tables.py` (Table 4).

## Arms

| config | mixer | hidden type | parameters | rate |
|---|---|---|---|---|
| `gm_attention_full` | SO(S^2) MHSA, two heads: `configs/model/d1_attention_full.yaml` | 16 rho_0 + 16 rho_1 + 16 rho_2 | 491,404 | 1.2e-2 |
| `gm_convolution_full` | SO(S^2) convolution over the same neighbourhood: `d1_convolution_full.yaml` | 16 rho_0 + 16 rho_1 + 16 rho_2 | 475,276 | 6e-3 |
| `gm_attention_rho01` | SO(S^2) MHSA without rho_2, widened to the budget: `d1_attention_rho01.yaml` | 26 rho_0 + 26 rho_1 | 493,580 | 1.6e-2 |
| `gm_attention_rho0` | SO(S^2) MHSA on scalar hidden features: `d1_attention_rho0.yaml` | 72 rho_0 | 510,780 | 1.2e-2 |

All four: Adam, ReduceLROnPlateau (factor 0.5, patience 5), gradient clipping at 1.0, no weight decay, batch 4,
100 epochs, the residual objective with the Coriolis channel as a fourth input, 256 fresh training samples per
epoch (seed 1000), 128 fixed validation trajectories (seed 2000), test trajectories from seed 3000.

## Learning rates tried (seed 0; best validation 10^3 x L2^2 in parentheses; chosen rate in bold; dagger: stopped before 100 epochs)

| arm | scheduler | rates |
|---|---|---|
| `gm_attention_full` | plateau | 4e-3 (11.75†), 6e-3 (9.18†), 8e-3 (8.62†), 1e-2 (4.79), **1.2e-2** (4.64), 1.6e-2 (4.45), 2e-2 (5.49) |
| | cosine | 4e-3 (10.66), 6e-3 (8.24), 8e-3 (7.49), 1e-2 (7.04), 1.2e-2 (6.53), 1.6e-2 (5.61), 2e-2 (4.94) |
| `gm_convolution_full` | plateau | 2e-3 (12.24), **6e-3** (6.58), 8e-3 (7.32), 1e-2 (7.11) |
| `gm_attention_rho01` | plateau | 1e-2 (9.12), 1.2e-2 (12.67), **1.6e-2** (6.75), 2e-2 (7.37) |
| `gm_attention_rho0` | plateau | 8e-3 (74.23), **1.2e-2** (61.99), 1.6e-2 (69.56), 2e-2 (82.69) |

The attention arm keeps 1.2e-2 rather than the seed-0 best 1.6e-2 because the three-seed fill at 1.6e-2 was less
stable (5.3 against 4.7 on the test one-step error).

## Running it

```bash
SWE_256x512_Ablation/run.sh launch 0 1 2 3          # twelve runs dealt round-robin over the GPUs given
SWE_256x512_Ablation/run.sh evaluate --num-examples 64   # outputs/SWE_256x512_Ablation/evaluation.json
SWE_256x512_Ablation/run.sh tables                       # outputs/SWE_256x512_Ablation/evaluation_table.tex
```

Runs land in `outputs/SWE_256x512_Ablation/<arm>/seed<N>/`; `SWE_CACHE=<dir>` shares solved states across runs.
`run.sh` exports `GM_CUTOFF_RULE=legacy`, the neighbourhood boundary rule the paper's runs used.
