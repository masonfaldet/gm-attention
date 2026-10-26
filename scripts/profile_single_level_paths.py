#!/usr/bin/env python
"""Time the single-level shallow-water model's evaluation paths: native, fused and stencil.

``GmAttentionSingleLevel`` (encoder convolution, N transformer blocks, decoder
convolution) runs here on a Gauss-Legendre grid in the coordinate gauge
(the released configuration runs on the level-5 icosphere, which has no
stencil). ``build_tensor_grid_geometry`` builds the grid's discretization and
its plans at the model's two cutoffs side by side, so the graph-path
configurations and the stencil one see the same geometry. Every operator
lives on the one grid, so there is no fused-plus-stencil mix: the fused
switches act on the graph path alone.

Usage, from the repository root:

    uv run python scripts/profile_single_level_paths.py                       # released widths, 32x64
    uv run python scripts/profile_single_level_paths.py --shape 64 128 --repeats 2
    uv run python scripts/profile_single_level_paths.py --chunk-size 1024     # the graph path chunked as the released runs
    uv run python scripts/profile_single_level_paths.py --paths native stencil --profile-ops 15
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repository root

from experiments.profiling import (  # noqa: E402
    SINGLE_GRID_PATHS,
    PathConfig,
    format_table,
    profile_paths,
    resolve_device,
    select_paths,
)
from experiments.shallow_water.models import CONDITIONED_FIELD_TYPE, GmAttentionSingleLevel  # noqa: E402
from gm_attention.nn.stencil_plan import build_tensor_grid_geometry  # noqa: E402
from gm_attention.representations.field_type import SO2FeatureField  # noqa: E402

#: The released sl_attention configuration (configs/model/sl_attention.yaml), minus its icosphere level.
DEEP_CUTOFF = 0.27571504347419135
CONV_CUTOFF = 0.13962634015954636


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shape", type=int, nargs=2, default=[32, 64], metavar=("NLAT", "NLON"))
    parser.add_argument("--multiplicity", type=int, default=16, help="copies of rho_0 + rho_1 + rho_2 (released: 16)")
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--deep-cutoff", type=float, default=DEEP_CUTOFF)
    parser.add_argument("--conv-cutoff", type=float, default=CONV_CUTOFF)
    parser.add_argument("--chunk-size", type=int, default=None, help="chunk the graph path by target range (released: 1024); plans run whole")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-buckets", type=int, default=8, help="buckets per stencil plan")
    parser.add_argument("--paths", nargs="*", default=None, help=f"subset of {[p.name for p in SINGLE_GRID_PATHS]}")
    parser.add_argument("--profile-ops", type=int, default=0, help="print this many top operators per path")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = getattr(torch, args.dtype)
    nlat, nlon = int(args.shape[0]), int(args.shape[1])
    print(f"GmAttentionSingleLevel on the {nlat}x{nlon} Gauss-Legendre grid, {dtype}, {device}")

    start = time.perf_counter()
    discretization, plans = build_tensor_grid_geometry(
        "legendre-gauss", nlat, nlon, [args.deep_cutoff, args.conv_cutoff], dtype=dtype, device=device,
        max_buckets=args.max_buckets,
    )
    print(f"geometry built in {time.perf_counter() - start:.1f}s: {discretization.n_nodes} nodes; "
          f"attention cutoff {args.deep_cutoff:.4g} with {discretization.edges_within(args.deep_cutoff).n_edges} edges, "
          f"convolution cutoff {args.conv_cutoff:.4g} with {discretization.edges_within(args.conv_cutoff).n_edges} edges")
    for cutoff in plans.cutoffs:
        print(f"  cutoff {cutoff:.4g}: {plans.for_cutoff(cutoff)}")

    torch.manual_seed(args.seed)
    model = GmAttentionSingleLevel(
        base_field_type={0: 1, 1: 1, 2: 1}, multiplicity=args.multiplicity, num_layers=args.num_layers,
        num_heads=args.num_heads, deep_cutoff=args.deep_cutoff, conv_cutoff=args.conv_cutoff, window="hard",
        nonlinearity="scalar_gate_double_sigmoid", self_interaction="linear", chunk_size=args.chunk_size,
        dtype=dtype, device=device,
    ).eval()
    parameters = list(model.parameters())
    print(f"model: {sum(p.numel() for p in parameters)} parameters, hidden type {model.hidden_field_type!r}, "
          f"{args.num_layers} blocks, {args.num_heads} heads, chunk_size {args.chunk_size}")

    generator = torch.Generator().manual_seed(args.seed)
    x = SO2FeatureField(
        CONDITIONED_FIELD_TYPE,
        {
            0: torch.randn(discretization.n_nodes, 2, dtype=dtype, generator=generator).to(device),
            1: torch.randn(discretization.n_nodes, 1, 2, dtype=dtype, generator=generator).to(device),
        },
    )

    def build_run(config: PathConfig):
        geometry = plans if config.stencil else discretization
        return lambda: model(x, geometry)

    paths = select_paths(SINGLE_GRID_PATHS, args.paths)
    results = profile_paths(
        paths, build_run, parameters, device=device, repeats=args.repeats, warmup=args.warmup,
        profile_operators=args.profile_ops,
    )
    print()
    print(format_table(results, device))


if __name__ == "__main__":
    main()
