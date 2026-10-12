# Gauge-Equivariant Aggregation for Spherical Attention: code

[![CI](https://github.com/masonfaldet/gm-attention/actions/workflows/ci.yml/badge.svg)](https://github.com/masonfaldet/gm-attention/actions/workflows/ci.yml)
[![Docs](https://github.com/masonfaldet/gm-attention/actions/workflows/docs.yml/badge.svg)](https://masonfaldet.github.io/gm-attention/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Supplementary code for the paper. `src/gm_attention` is the library (SO(2)-steerable field types, spherical
discretizations and neighbourhood geometry, the gauge-equivariant attention, convolution and transformer
blocks with their fused CUDA kernels, and a training loop); `experiments/` holds the shallow-water and
segmentation experiments; the four bundle directories reproduce the paper's tables, each with its configs, a
`run.sh` driver, an evaluator, a table renderer and a README stating the protocol and the learning rates tried.

| bundle | paper |
|---|---|
| `SWE_128x256_Benchmark/` | Table 5, the released shallow-water configuration |
| `SWE_256x512_Ablation/` | Table 2, the depth-1 ablation; Table 4 (with the zero-shot bundle) |
| `SWE_256x512_ZeroShot/` | Table 1, zero-shot grid transfer of the single-level model |
| `S2D3DS_128x256_Segmentation/` | Table 3, Stanford 2D-3D-S segmentation |

## Environment

Python 3.12 and [uv](https://docs.astral.sh/uv/). From this directory:

```bash
uv sync            # torch, torch-harmonics 0.8.0, hydra, h5py, ...
```

On Linux the lock pins torch 2.6.0 from PyTorch's CUDA 11.8 index (Volta GPUs), and numpy, scipy and h5py to
releases with glibc-2.17 wheels; elsewhere the packages resolve normally. Every experiment ran on Tesla V100
32 GB GPUs. The fused CUDA kernels (`src/gm_attention/nn/fused/`, switched on by the `GM_FUSED_*`
environment variables the bundle READMEs name) compile on first use with `nvcc` and a C++17 `g++`; without the
switches every model runs through the pure-PyTorch path, the same function up to floating-point summation order. The
released segmentation baselines run in a second environment (`S2D3DS_128x256_Segmentation/reference/README.md`).

## Layout

```
configs/            model / training / objective / grid groups the bundles compose
src/gm_attention/   the library
experiments/        shallow_water (models, training loop), shallow_water_paper (the paper's SWE protocol,
                    evaluation, tables), stanford_2d3ds (the SO(S^2)-SegFormer, data, evaluation)
tests/              the test suite (below); tests/support holds the dense reference implementations the
                    fast paths are checked against and a synthetic 2D-3D-S writer
docs/, mkdocs.yml   the documentation site (below)
scripts/            toolchain check for the fused kernels; a progress reader for running studies
outputs/, logs/     written by the runs
```

Data: the shallow-water trajectories are synthesized by torch-harmonics' solver from fixed seeds at run time
(nothing to download); the segmentation data are downloaded and converted by
`S2D3DS_128x256_Segmentation/prepare_data.py`.

## Tests

```bash
uv run pytest            # from the repository root; `-m slow` adds the full-resolution checks
```

The suite runs on the CPU and downloads nothing: gauge equivariance of every layer and block, the
steerable kernels and intertwiners against dense reference implementations (`tests/support/`), the fused
kernels against the pure-PyTorch path, the discretizations and their quadrature, the solver-synthesized
shallow-water data, tiny end-to-end trainings through each bundle's configuration, and the bundles'
evaluators and table renderers. Tests of the fused kernels build their C++ CPU versions on first use (a
C++17 compiler and `ninja`, which `uv sync` installs). `uv run ruff check .` lints with the rules in
`ruff.toml`; `.github/workflows/ci.yml` runs both.

## Documentation

```bash
uv sync --group docs && uv run mkdocs serve   # http://127.0.0.1:8000
```

The site is published at <https://masonfaldet.github.io/gm-attention/>. It (`docs/`, built by MkDocs) has the mathematics behind the library (conventions, the geometry of the
sphere, the discretization, the representations of SO(2), gauge-equivariant aggregation), the API
reference rendered from the docstrings, and the bundle READMEs. `.github/workflows/docs.yml` publishes it
to GitHub Pages on every push to `main`.

## Citation

`CITATION.cff` holds the metadata GitHub renders under "Cite this repository". The paper is under review;
until it is published, cite it as

```bibtex
@misc{faldet2026gauge,
  title  = {Gauge-Equivariant Aggregation for Spherical Attention},
  author = {Faldet, Mason},
  year   = {2026},
  note   = {Under review as a conference paper at ICLR 2027}
}
```

## License

MIT, see [LICENSE](LICENSE). The released torch-harmonics files vendored under
`S2D3DS_128x256_Segmentation/reference/upstream/` keep their own licence, in that directory.
