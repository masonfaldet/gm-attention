# SWE 128x256 benchmark (Table 5)

The released shallow-water configuration of "Attention on the Sphere": 128x256 equiangular solver states, a
30-minute lead (12 solver substeps), three arms, three seeds each. `evaluate.py` scores every seed's `best.pt`
on 64 held-out test trajectories: one step and a five-step rollout (2.5 h), on each model's own grid and
after a least-squares fit through degree 42; `tables.py` renders Table 5 and
`experiments/shallow_water_paper/degree_band_tables.py` the error-by-degree appendix table.

## Arms

| config | model | parameters | optimizer |
|---|---|---|---|
| `gm_attention_depth1` | SO(S^2) Transformer D1: `configs/model/d1_attention_full.yaml` (L5 icosphere in/out, four attention blocks on the L4 latent, hidden type 16(rho_0 + rho_1 + rho_2)) | 491,404 | Adam 1.2e-2, ReduceLROnPlateau 0.5 / 5, grad clip 1.0 |
| `sfno` | SFNO, torch-harmonics' `sfno_sc2_layers4_e32`: `configs/model/sfno.yaml` at embed dim 32 | 541,408 | Adam 3.2e-2, ReduceLROnPlateau 0.5 / 5, grad clip 1.0 |
| `s2_transformer` | S^2 Transformer (local), the registry's `s2ntransformer_sc2_layers4_e128`: `configs/model/s2_transformer.yaml` | 535,424 | the released script's own: Adam 1e-4, ReduceLROnPlateau 0.1 / 10, no clipping |

All arms: batch 4, 100 epochs, the residual objective with the Coriolis channel `<p, z>` as a fourth input,
256 fresh training samples per epoch from a fixed seed schedule (`train_seed` 1000), 128 fixed validation
trajectories (seed 2000), test trajectories from seed 3000, normalization statistics from 128 samples (seed
4000); the quadrature-weighted squared L2 loss on the training grid. Parameter counts are real scalars
(SFNO's complex weights count twice).

## Learning rates tried (seed 0; best validation 10^3 x L2^2 in parentheses; chosen rate in bold)

| arm | rates |
|---|---|
| `gm_attention_depth1` | 1e-2 (0.51), **1.2e-2** (0.45), 1.6e-2 (0.40); 1.2e-2 kept because its three-seed fill on 256x512 was more stable than 1.6e-2's (Table 2) |
| `sfno` | 8e-3 (0.39), 1.6e-2 (0.22), **3.2e-2** (0.24), 4.8e-2 (0.32), 6.4e-2 (0.42), 1.28e-1 (1.28) |
| `s2_transformer` | **1e-4** (2.93), the released script's rate, not searched |

## Running it

From the repository root, with the environment installed (`uv sync`):

```bash
SWE_128x256_Benchmark/run.sh launch 0 1 2 3 4 5 6 7 8   # nine GPUs, one run each (or: sequential 0)
SWE_128x256_Benchmark/run.sh evaluate --num-examples 64  # outputs/SWE_128x256_Benchmark/evaluation.json
SWE_128x256_Benchmark/run.sh tables                      # the LaTeX tables beside it
```

Runs land in `outputs/SWE_128x256_Benchmark/<arm>/seed<N>/` (resolved config, `paper_protocol.json`,
`logs/metrics.jsonl`, `checkpoints/{last,best}.pt`); progress: `scripts/progress.sh outputs/SWE_128x256_Benchmark`.
`SWE_CACHE=<dir>` caches solved states across runs (results are identical without it). The S^2 Transformer
runs several times faster with torch-harmonics' compiled CUDA extensions (`UV_PROJECT_ENVIRONMENT=<env>`
selects an environment holding them); the other arms do not use them.

The paper's runs used the library's *legacy* neighbourhood boundary rule (`GM_CUTOFF_RULE=legacy`, which
`run.sh` exports); it decides membership of nodes lying exactly on a cutoff radius and changes no other edge.
