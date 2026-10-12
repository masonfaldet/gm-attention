# The released script

The references of Table 3 are trained by torch-harmonics v0.8.0's own `examples/segmentation/train.py` on
this bundle's data file. `upstream/` matches the tag `v0.8.0`
(commit `c7afb5461e6c4c9298ce5afd3ada1f8436cdc15d` of https://github.com/NVIDIA/torch-harmonics,
BSD-3-Clause, `upstream/LICENSE`): the training script, the model registry it imports
(`s2nsegformer_sc2_layers4_e128` is the paper's S^2 SegFormer, `nsegformer_sc2_layers4_e128` the planar
SegFormer, `s2unet_sc2_layers4_e128` the spherical UNet), the planar baselines and the released metrics module.
`SHA256SUMS` records their digests and `run_released.py` refuses to start if any file differs.

The script is pointed at the bundle's file (`--data_path data/2D3DS`) and splits and normalizes with its own
lines (`random_split(..., [0.95, 0.025, 0.025], generator=torch.Generator().manual_seed(333))`,
`compute_stats_s2`), so the split is the same permutation of the same file as every other arm's.

`run_released.py` executes the script with `runpy` after four adjustments, none of which changes a number:
`--seed N` replaces the script's two `manual_seed(333)` calls (the split generator stays 333, so seeds share the
split and differ in initialization and shuffle order); `metrics_fixed.py` is the released metrics module with
one mis-indented docstring line fixed, served under the released name by `shims/sitecustomize.py`; `shims/`
stands in for `wandb` (nothing is logged) and, for the registry import, for `natten`; the figure calls that need
cartopy are no-ops without it. The runner refuses to train on a GPU without the released CUDA extensions
(`--allow-fallback` overrides).

## Environment

The script needs torchvision, pandas, matplotlib and tqdm (pyproject's `reference` group); it goes into its
own environment with torch-harmonics' extensions built there:

```bash
UV_PROJECT_ENVIRONMENT=.venv-ref uv sync --inexact --group reference
uv pip install --python .venv-ref/bin/python setuptools wheel ninja
FORCE_CUDA_EXTENSION=1 TORCH_CUDA_ARCH_LIST="7.0" \
  uv pip install --python .venv-ref/bin/python --no-build-isolation --no-deps --force-reinstall --no-cache-dir torch-harmonics==0.8.0
S2D3DS_128x256_Segmentation/reference/run.sh check
```

The planar SegFormer needs the real `natten` (0.17.x) in that environment, a source build against the
installed torch (`uv pip install --python .venv-ref/bin/python --no-build-isolation natten==0.17.5`, with
CMake >= 3.20 on `PATH`).

## Running, adopting, scoring

```bash
MODEL=s2nsegformer_sc2_layers4_e128 S2D3DS_128x256_Segmentation/reference/run.sh launch 0 1 2   # seeds 0 1 2
MODEL=s2nsegformer_sc2_layers4_e128 S2D3DS_128x256_Segmentation/reference/run.sh adopt
MODEL=s2nsegformer_sc2_layers4_e128 S2D3DS_128x256_Segmentation/reference/run.sh evaluate --device cuda:0
```

Each seed runs the script with its defaults stated explicitly (`--num_epochs 200 --batch_size 8
--learning_rate 5e-4 --max_grad_norm 4.0 --amp_mode none`) into
`outputs/S2D3DS_128x256_Segmentation/<arm>/seed<N>/`. `adopt.py` then writes, beside the released files, the
`segmentation_protocol.json` and `checkpoints/last.pt` the bundle's evaluator reads: the protocol comes from the
bundle's transcription of the registry entry (`configs/model/{s2_segformer,r2_segformer,s2_unet}.yaml`) with the
optimizer settings read from the run log, and the released state dict is loaded strictly into that model.
`evaluate` is the bundle's `evaluate.py` on the adopted runs (`--arms <arm>=<arm>`, report split, last epoch),
writing `outputs/S2D3DS_128x256_Segmentation/evaluation_released.json` (`--output` names another file for the
other two references).
