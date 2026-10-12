# Contributing

This repository is the reference implementation of a paper: its purpose is to let the paper's tables be
reproduced and its method be read. Bug reports, reproduction reports and corrections are welcome as issues
or pull requests; new features are out of scope unless they fix a reproduction.

## Setting up

Python 3.12 and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync                      # the library, the experiments' dependencies, pytest, ruff, ninja
uv sync --group docs         # additionally MkDocs, for the documentation site
```

## Checks

Run them from the repository root; the continuous-integration workflow runs the same three, plus a
byte-compilation pass over every Python file outside the vendored released code.

```bash
uv run ruff check .          # pyflakes and syntax-error rules (ruff.toml)
uv run pytest                # the suite; tests marked `slow` are deselected (`uv run pytest -m slow` runs them)
uv run mkdocs build --strict # the documentation site, into site/
```

The suite runs on the CPU in minutes and needs no data download: the shallow-water trajectories are
synthesized by the solver, and the segmentation tests write a small synthetic dataset. A C++17 compiler and
`ninja` let the tests of the fused kernels build their CPU versions; without a compiler those tests fail at
build time rather than silently passing.

## Changing the library

The library under `src/gm_attention` is exactly the code the reported experiments ran. A change to it
should come with the test that would have caught the bug, and must not change the configurations under
the four bundle directories or the numbers their READMEs report without saying so in the pull request.
