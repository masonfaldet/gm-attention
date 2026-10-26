# Testing

```bash
uv run pytest                 # from the repository root
uv run pytest -m slow         # the full-resolution checks, minutes on a CPU
uv run pytest tests/nn        # one directory
```

The suite runs on the CPU and downloads nothing: the shallow-water tests synthesize their trajectories
with torch-harmonics' solver at a small resolution, and the segmentation tests write a synthetic Stanford
2D-3D-S file in the released converter's layout (`tests/support/synthetic_2d3ds.py`). Run it from the
repository root: the bundle configurations resolve their shared groups through a Hydra search path
relative to the working directory.

## What is checked

| directory | subject |
|---|---|
| `tests/representations` | the irreps of $SO(2)$ and their complex form, field types and feature fields, the gauge-transformation law; the dense reference implementations themselves |
| `tests/discretization` | the grids and their quadrature, tangent frames and regauging, edge geometry (log map, transport angles, angular harmonics, the cutoff rule), the icosphere and equiangular hierarchies, the coordinate frames and the per-row stencil of a tensor-product grid against its edge set |
| `tests/nn` | every layer and block: gauge equivariance, the steerable kernel and the intertwiner against the dense references in `tests/support`, the factored kernel against the per-edge readout, the fused kernels against the pure-PyTorch path, chunked evaluation against unchunked, the fixed-grid reference and the stencil path against the graph path (below) |
| `tests/training` | the training loop, checkpointing and bit-for-bit resumption, the losses and metrics of the paper, logging, seeding |
| `tests/data` | the solver-synthesized shallow-water data: determinism per index and seed, agreement with the solver's own synthesis |
| `tests/experiments` | the shallow-water models and protocol (training, transfer across grids, spectral comparison, rollouts) and the segmentation model, data and training, each through a tiny end-to-end run of the bundle's configuration |
| `tests/test_*.py` | the four bundles' evaluators, table renderers and the released-baseline harness, on small synthetic runs |

## The fixed grid

The tensor-product grid in the coordinate gauge has its own files, each with the graph path as the oracle
on the same `SphericalDiscretization` (`build_tensor_grid_discretization`), in float64 (`atol` 1e-10) and
float32, with and without leading batch axes against the tiled graph:

| file | subject |
|---|---|
| `tests/discretization/test_stencil.py` | the coordinate frames and the fold of the edge geometry into per-row slots, checked against every edge |
| `tests/nn/test_fixed_grid_attention.py`, `tests/nn/test_fixed_grid_oracle.py` | the per-row reference: the slot matrices against the kernel on basis vectors, the attention and MHSA against the graph path and against an oracle written from the definition |
| `tests/nn/test_stencil_plan.py` | the bucketed plan: every row in one bucket, padding to the bucket's width, the phantom slot, the gathers against the stencil's own, the partition against brute force |
| `tests/nn/test_stencil_kernel.py` | the gather assembly of the slot matrices against the reference's `steerable_kernel_matrices` on every shape of type pair, both layouts, the phantom row, the per-slot factor, and its gradients |
| `tests/nn/test_stencil_attention.py` | the vectorized attention and MHSA against the graph path and the reference for every bucket count, gradients with respect to the input and every parameter, bitwise equivariance to longitude shifts, `torch.compile(fullgraph=True)` of the dense stage |
| `tests/nn/test_stencil_forward.py` | the layers' `forward(x, plan=...)`: identical to the stencil functions, against the graph path in both dtypes, leading batch axes, gradients, the plan in place of the edges through a transformer block, a `StencilPlans` container resolved at each layer's cutoff (the segmentation block's two radii from one container), and the errors for both arguments, neither, a missing cutoff, or a plan passed to chunked evaluation |
| `tests/experiments/shallow_water/test_single_level_stencil.py`, `tests/experiments/stanford_2d3ds/test_model_stencil.py` | the models on Gauss-Legendre grids: `GmAttentionSingleLevel` on the plans of `build_tensor_grid_geometry` against the (chunked) graph path, and the GM SegFormer on a hierarchy with `stencil_plans=True` against the same hierarchy without them, values and gradients in both dtypes; tiling drops the plans, equiangular or icosphere stages are refused |
| `tests/nn/test_stencil_convolution.py` | the convolution with the general and the depthwise kernel, both windows, `self_interaction` and `normalize`, against the graph path; the depthwise tables against the kernel on basis vectors; the two consumers of the depthwise mix (`SO2TransformerBlock` with its convolution mixer and the segmentation block) with the stencil path swapped in |

The Triton kernel in `gm_attention.nn.stencil_attention_triton` has no test: it needs a CUDA device.

## Markers and switches

`slow` marks checks that build a full-resolution model and run a real forward and backward pass; they are
deselected by default (`addopts` in `pyproject.toml`). The fused kernels are switched on per test through
the `GM_FUSED_PAIRS`, `GM_FUSED_FACTORED`, `GM_FUSED_ATTENTION` and `GM_FACTORED_KERNEL` environment
variables, so the environment's own values do not change what a test checks; a test of a fused path builds
its C++ CPU version on first use, which needs a C++17 compiler and `ninja`. `GM_CUTOFF_RULE` is likewise
set by the tests that compare the two cutoff rules.

## The support package

`tests/support` holds code the tests need that the library does not ship: `bases.py` and
`intertwiners.py`, dense real-arithmetic implementations of the steerable kernel bases and the $SO(2)$
intertwiners, written for transparency rather than speed, against which the complex-arithmetic fast paths
in `gm_attention.nn.kernels` and `gm_attention.nn.linear` are checked; and the synthetic dataset writer.
Nothing under `src/` imports from it.

## Profiling the paths

Two scripts time a model's forward and forward-plus-backward through every implementation of the same
function: the native graph path, the fused kernels (`GM_FUSED_PAIRS`, or `GM_FACTORED_KERNEL` with
`GM_FUSED_FACTORED`, each with `GM_FUSED_ATTENTION`), the stencil path, and, where a model has bipartite
transfers, the stencil path with fused merges. The switches are set on the modules for each run, so one process
compares all paths on one geometry; every path's output is checked against the native one and the table counts
the calls that actually entered a fused kernel (a kernel declines widths or dtypes it does not support and the
layer falls back to its native code).

```bash
uv run python scripts/profile_single_level_paths.py --shape 64 128            # GmAttentionSingleLevel, released widths
uv run python scripts/profile_segformer_paths.py --img-size 128 256 --released  # GM SegFormer, released widths on Gauss-Legendre stages
```

Both run the model on Gauss-Legendre grids in the coordinate gauge, which the stencil path requires (the
released configurations run on icospheres); `--profile-ops N` adds the top operators of one step per path,
`--paths` picks a subset. The helpers are in `experiments/profiling.py`, tested in `tests/experiments/test_profiling.py`.
On a CPU the fused extensions are the CUDA kernels' reference drivers and run slower than the native path.

## Continuous integration

`.github/workflows/ci.yml` lints with `ruff check .` (the rules in `ruff.toml`), byte-compiles every
Python file and runs the suite on a CPU runner. `.github/workflows/docs.yml` builds this site with
`mkdocs build --strict` and publishes it to GitHub Pages.
