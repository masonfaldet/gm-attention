#!/usr/bin/env python
"""Time the GM SegFormer's evaluation paths: native, fused, stencil, and stencil with fused merges.

The stages run on Gauss-Legendre grids in the coordinate gauge (the stencil
path needs them; the released model runs on icosphere stages), the input
level on the equiangular grid as the data. One hierarchy is built with
stencil plans; the graph-path configurations receive the same hierarchy with
the plans stripped, so every path sees the same discretizations and
transfers. The merges and interpolations are bipartite and stay on the graph
path in every configuration; ``stencil+fused-*`` runs them through the fused
kernels while the stages run on the stencil.

Usage, from the repository root:

    uv run python scripts/profile_segformer_paths.py                      # small model, 64x128, two stages
    uv run python scripts/profile_segformer_paths.py --released           # the released widths and depths
    uv run python scripts/profile_segformer_paths.py --img-size 128 256 --stages 4 --released --repeats 2
    uv run python scripts/profile_segformer_paths.py --paths native stencil --profile-ops 15
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repository root

from experiments.profiling import (  # noqa: E402
    MULTI_GRID_PATHS,
    PathConfig,
    format_table,
    profile_paths,
    resolve_device,
    select_paths,
)
from experiments.stanford_2d3ds.geometry import (  # noqa: E402
    build_segformer_hierarchy,
    segformer_radii,
    segformer_stage_shapes,
)
from experiments.stanford_2d3ds.model import GmSegformer  # noqa: E402
from gm_attention.representations.field_type import SO2FeatureField, SO2FieldType  # noqa: E402

#: The released sos2_segformer architecture (configs/model/sos2_segformer.yaml), minus its icosphere stages.
RELEASED = dict(
    in_channels=3, base_field_type={0: 3, 1: 1, 2: 1}, multiplicities=[2, 4, 6, 8], heads=[1, 1, 2, 4],
    depths=[3, 4, 6, 3], window="hard", nonlinearity="scalar_gate_double_sigmoid", readout="convolution",
    readout_channels=[8, 16, 32, 32], attention_batched=True, n_radial_basis=8, radial_hidden_features=[14],
    radial_basis="gaussian", bias=True, value_kind="steerable", ffn_expansion=2, vertical_conditioning=True,
    decoder="fusion", decoder_channels=32, decoder_dropout=0.0, stem_channels=8, stem_kind="pointwise",
    gravity_conditioning=True, gravity_projection=True, gravity_readout=True, gravity_readout_expansion=2,
    normalized_mix=True, normalized_readout=True, head_hidden=64,
)
#: A small configuration for quick runs: two stages, narrow types.
SMALL = dict(
    in_channels=3, base_field_type={0: 1, 1: 1, 2: 1}, multiplicities=[2, 4], heads=[1, 2], depths=[1, 2],
    window="hard", nonlinearity="scalar_gate_double_sigmoid", readout="convolution", attention_batched=True,
    ffn_expansion=2, normalized_mix=True, normalized_readout=True, decoder="fusion",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--img-size", type=int, nargs=2, default=[64, 128], metavar=("NLAT", "NLON"))
    parser.add_argument("--stages", type=int, default=None, help="number of stages (default: 4 with --released, else 2)")
    parser.add_argument("--released", action="store_true", help="the released widths, depths and options on Gauss-Legendre stages")
    parser.add_argument("--num-classes", type=int, default=13)
    parser.add_argument("--readout-rows", type=float, default=1.5)
    parser.add_argument("--interpolate-spacings", type=float, default=1.0)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-buckets", type=int, default=8, help="buckets per stencil plan")
    parser.add_argument("--paths", nargs="*", default=None, help=f"subset of {[p.name for p in MULTI_GRID_PATHS]}")
    parser.add_argument("--profile-ops", type=int, default=0, help="print this many top operators per path")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = getattr(torch, args.dtype)
    img_size = (int(args.img_size[0]), int(args.img_size[1]))
    n_stages = args.stages if args.stages is not None else (4 if args.released else 2)
    stages = segformer_stage_shapes(img_size, 2, n_stages)
    radii = segformer_radii(
        stages, grid_internal="legendre-gauss", readout_rows=args.readout_rows,
        interpolate_spacings=args.interpolate_spacings,
    )
    print(f"GM SegFormer on {img_size[0]}x{img_size[1]} with Gauss-Legendre stages {list(stages)}, {dtype}, {device}")

    start = time.perf_counter()
    hierarchy = build_segformer_hierarchy(
        img_size, stages, radii, grid_internal="legendre-gauss", lateral=True, stencil_plans=True, dtype=dtype, device=device,
    )
    graph_hierarchy = dataclasses.replace(
        hierarchy, levels=tuple(dataclasses.replace(level, plans=None) for level in hierarchy.levels)
    )
    print(f"hierarchy with stencil plans built in {time.perf_counter() - start:.1f}s")
    for level in hierarchy.levels[1:]:
        assert level.discretization is not None and level.plans is not None
        edges = {f"{c:.3g}": level.discretization.edges_within(c).n_edges for c in level.plans.cutoffs}
        print(f"  {level.describe()}: edges per cutoff {edges}; {level.plans}")
        for cutoff in level.plans.cutoffs:
            plan = level.plans.for_cutoff(cutoff)
            print(f"    cutoff {cutoff:.3g}: {plan}")

    architecture = dict(RELEASED if args.released else SMALL)
    if n_stages != len(architecture["multiplicities"]):
        for key in ("multiplicities", "heads", "depths", "readout_channels"):
            if key in architecture:
                architecture[key] = (list(architecture[key]) * n_stages)[:n_stages]
    torch.manual_seed(args.seed)
    model = GmSegformer(
        num_classes=args.num_classes, merge_cutoffs=radii.merge, attention_cutoffs=radii.attention,
        mix_cutoffs=radii.mix, interpolate_cutoffs=radii.interpolate, readout_cutoffs=radii.readout,
        drop_path_rate=0.0, dtype=dtype, device=device, **architecture,
    ).eval()
    parameters = list(model.parameters())
    print(f"model: {sum(p.numel() for p in parameters)} parameters, stages {architecture['multiplicities']} x base type, "
          f"heads {architecture['heads']}, depths {architecture['depths']}")

    generator = torch.Generator().manual_seed(args.seed)
    x = SO2FeatureField(
        SO2FieldType({0: 3}), {0: torch.randn(hierarchy.finest.n_nodes, 3, dtype=dtype, generator=generator).to(device)}
    )

    def build_run(config: PathConfig):
        geometry = hierarchy if config.stencil else graph_hierarchy
        return lambda: model(x, geometry)

    paths = select_paths(MULTI_GRID_PATHS, args.paths)
    results = profile_paths(
        paths, build_run, parameters, device=device, repeats=args.repeats, warmup=args.warmup,
        profile_operators=args.profile_ops,
    )
    print()
    print(format_table(results, device))
    print("\nfused switches act on the graph-path operators only: with stencil plans the stages run on the dense layout and "
          "the bipartite merges carry the fused kernels; `fused attention` has no operator left to act on there.")


if __name__ == "__main__":
    main()
