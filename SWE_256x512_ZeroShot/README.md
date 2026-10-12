# Zero-shot grid transfer on SWE 256x512 (Table 1, Table 4)

The single-level SO(S^2) Transformer (SL) trained on the L5 icosphere and evaluated, unchanged, on the L6
icosphere and on Legendre-Gauss grids, beside SFNO and the single-level S^2 Transformer on their own grids
and the depth-1 arm of the ablation bundle. Three seeds per arm; 64 held-out test trajectories; one step (1 h)
and a five-step rollout (5 h); errors on each grid's own quadrature and after a least-squares fit through
degree 85.

## Arms

| config | model | reads / writes | parameters | optimizer |
|---|---|---|---|---|
| `gm_l5_100` | SL: `configs/model/sl_attention.yaml`, an SO(S^2) convolution encoder, four SO(S^2) Transformer blocks and an SO(S^2) convolution decoder, all on the L5 icosphere, hidden type 16(rho_0 + rho_1 + rho_2), attention chunked at 2048 target nodes | 10,242 icosphere nodes | 491,404 | Adam 2e-2, cosine to zero over 100 epochs, grad clip 1.0 |
| `sfno` | SFNO at embed dim 22: `configs/model/sfno.yaml` | 256x512 equiangular | 503,866 | Adam 1.6e-2, ReduceLROnPlateau 0.5 / 5, grad clip 1.0 |
| `s2_transformer_sl` | S^2 Transformer (local) without its internal downsampling: `configs/model/s2_transformer_single_level.yaml`, attention on the 128x256 Legendre-Gauss view of the solver states it reads | 128x256 Legendre-Gauss | 535,424 | Adam 6e-3, ReduceLROnPlateau 0.5 / 5, grad clip 1.0, batch 4 as two accumulated micro-batches |
| (depth-1 row) | `gm_attention_full` of `SWE_256x512_Ablation`, borrowed from that bundle's evaluation | L5 icosphere | 491,404 | see that bundle |

Batch 4, 100 epochs, the residual objective with the Coriolis channel as a fourth input, the data protocol of
the ablation bundle. The SL arm trains with the fused pair-term kernels and chunks keeping their saved tensors
(`GM_FUSED_PAIRS=1 GM_CHUNK_REMATERIALIZE=0`, exported by `run.sh` for this arm; the same function up to
floating-point summation order, several times faster and more memory), which needs `nvcc` and a C++17 `g++` on `PATH`.

## Learning rates tried (seed 0; best validation 10^3 x L2^2 in parentheses; chosen rate in bold; dagger: stopped early)

| arm | scheduler | rates |
|---|---|---|
| `gm_l5_100` | cosine, 50 epochs | **2e-2** (1.31); the arm then trained for the references' 100 epochs at that rate |
| | plateau, 50 epochs | 6e-3 (5.86†), 1e-2 (5.32†), 1.4e-2 (1.75†), 1.8e-2 (1.41†), 2e-2 (1.32†), 2.4e-2 (2.21†) |
| `sfno` | plateau | 1e-3 (1.97), 2e-3 (1.56), 4e-3 (1.28), 8e-3 (0.95), **1.6e-2** (0.84), 3.2e-2 (0.88) |
| `s2_transformer_sl` | plateau | 3e-4 (27.21), 1e-3 (10.46), 3e-3 (6.45), **6e-3** (5.73), 1.2e-2 (13.43) |

## Running it

```bash
SWE_256x512_ZeroShot/run.sh launch 0 1 2                 # nine runs dealt round-robin over the GPUs given
SWE_256x512_ZeroShot/run.sh evaluate-parallel 0 1 2 3    # one (grid, seed) job per GPU -> outputs/SWE_256x512_ZeroShot/parts/
SWE_256x512_ZeroShot/run.sh merge                        # parts + the ablation's depth-1 row -> evaluation.json
SWE_256x512_ZeroShot/run.sh tables                       # the LaTeX table
```

`merge` reads the depth-1 row from `outputs/SWE_256x512_Ablation/evaluation.json` (`ABLATION_EVALUATION` points it
elsewhere) after checking the two studies share the protocol and the test trajectories. The 256x512
Legendre-Gauss grid's attention geometry (5e8 edges) exceeds a 32 GB GPU; that row runs on the host:

```bash
SWE_256x512_ZeroShot/run.sh evaluate --num-examples 64 --projector-device cpu --require-seeds 0,1,2 \
    --grids legendre-gauss:256x512 --gm-device cpu --chunk-size 64 --append
```

The evaluation jobs pin the chunk where a grid needs it (L6 at 1024 target nodes, Legendre-Gauss 128x256 at
256). Table 4's rows come from `experiments/shallow_water_paper/degree_band_tables.py` on the three
evaluation JSONs. `run.sh` exports `GM_CUTOFF_RULE=legacy`, the neighbourhood boundary rule the paper's runs used.
