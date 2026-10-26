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
| `tests/nn` | every layer and block: gauge equivariance, the steerable kernel and the intertwiner against the dense references in `tests/support`, the factored kernel against the per-edge readout, the fused kernels against the pure-PyTorch path, chunked evaluation against unchunked, the dense fixed-grid attention against the graph path |
| `tests/training` | the training loop, checkpointing and bit-for-bit resumption, the losses and metrics of the paper, logging, seeding |
| `tests/data` | the solver-synthesized shallow-water data: determinism per index and seed, agreement with the solver's own synthesis |
| `tests/experiments` | the shallow-water models and protocol (training, transfer across grids, spectral comparison, rollouts) and the segmentation model, data and training, each through a tiny end-to-end run of the bundle's configuration |
| `tests/test_*.py` | the four bundles' evaluators, table renderers and the released-baseline harness, on small synthetic runs |

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

## Continuous integration

`.github/workflows/ci.yml` lints with `ruff check .` (the rules in `ruff.toml`), byte-compiles every
Python file and runs the suite on a CPU runner. `.github/workflows/docs.yml` builds this site with
`mkdocs build --strict` and publishes it to GitHub Pages.
