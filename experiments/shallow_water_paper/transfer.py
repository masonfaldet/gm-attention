"""Zero-shot transfer of a fitted paper-state model to grids it was not trained on.

A run is read back from its run directory (config, per-channel
normalization from ``paper_protocol.json``, weights); :func:`geometry_for_config`
rebuilds the geometry on the target grid from the run's own cutoffs, and
:func:`predict` forms the reconstructed prediction with the same Coriolis
conditioning as the training step. Weights and physical radii are the
run's; only the grid changes. The single-level arm runs on any grid; the
depth-1 arms transfer within the icosphere family only, since
``IcoEncode``/``IcoDecode`` use the icosphere's parent-child structure,
with the encode radius pinned to the trained level's own edge length.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from experiments.shallow_water.grid_transfer import GridSpec
from experiments.shallow_water_paper.data import (
    CHANNELS,
    ScalarNormalization,
    ScalarState,
)
from experiments.shallow_water_paper.train import (
    CONDITIONED_FIELD_TYPE,
    PROTOCOL_FILENAME,
    SINGLE_LEVEL_KIND,
    Geometry,
    resolve_level,
)
from gm_attention.discretization.discretization import SphericalDiscretization
from gm_attention.discretization.frames import build_tangent_frames
from gm_attention.discretization.grid import SphericalGrid
from gm_attention.discretization.grids.icosphere import (
    build_icosphere_multiresolution,
    build_icosphere_refinement,
)
from gm_attention.representations.field_type import SO2FeatureField
from gm_attention.training.checkpointing import load_checkpoint

logger = logging.getLogger(__name__)


def _config_of_run(checkpoint: Path) -> DictConfig:
    """The resolved config a checkpoint's run trained with: the run directory's
    ``.hydra/config.yaml``, the composed config with every override applied."""
    hydra_config = checkpoint.parent.parent / ".hydra" / "config.yaml"
    if not hydra_config.is_file():
        raise FileNotFoundError(f"No {hydra_config} beside the checkpoint, so its architecture cannot be read.")
    loaded = OmegaConf.load(hydra_config)
    assert isinstance(loaded, DictConfig)
    return loaded

FIELD_TYPES = {
    "input_field_type": {0: len(CHANNELS) + 1},
    "output_field_type": {0: len(CHANNELS)},
}


@dataclass
class FittedRun:
    """A paper-state run's config, kind, training level, normalization, and
    model -- everything a transfer needs besides the target grid."""

    cfg: DictConfig
    kind: str
    native_level: int
    normalization: ScalarNormalization | None
    model: nn.Module
    #: The checkpoint's epoch, when loaded from one.
    epoch: int | None = None

    @property
    def single_level(self) -> bool:
        return self.kind == SINGLE_LEVEL_KIND

    @property
    def native(self) -> str:
        return f"icosphere:{self.native_level}"


def fitted_encode_radius(native_level: int, device: torch.device) -> float:
    """The depth-1 encode radius the weights were fitted with: the trained
    level's own coarse edge length, held fixed at every transfer level."""
    _, _, refinement = build_icosphere_refinement(native_level - 1, dtype=torch.float32, device=device)
    return float(refinement.coarse_edge_length)


def build_model(cfg: DictConfig, chunk_size: int | None, device: torch.device) -> nn.Module:
    """The run's architecture with the paper's field types, chunked for
    evaluation on grids denser than the training one."""
    architecture = cfg.model.architecture
    kind = str(cfg.model.get("kind", "?"))
    if not kind.startswith("gm_attention"):
        raise TypeError(
            f"{kind!r} is not a GM arm: the transfer and rollout probes rebuild the "
            "icosphere geometry, which the reference's tensor grid has none of."
        )
    # A chunk size given here overrides the run's; None keeps the run's own
    # rather than being passed through, which would un-chunk the model.
    if kind == SINGLE_LEVEL_KIND:
        chunking = {} if chunk_size is None else {"chunk_size": chunk_size}
        model: nn.Module = hydra.utils.instantiate(architecture, **chunking, **FIELD_TYPES)
    else:
        chunking = {} if chunk_size is None else {"evaluation_chunk_size": chunk_size}
        model = hydra.utils.instantiate(
            architecture,
            encode_cutoff=fitted_encode_radius(resolve_level(cfg), device),
            **chunking,
            **FIELD_TYPES,
        )
    return model.to(device).eval()


def load_run(checkpoint: Path, chunk_size: int | None, device: torch.device) -> FittedRun:
    cfg = _config_of_run(checkpoint)
    run_dir = checkpoint.parent.parent
    protocol = json.loads((run_dir / PROTOCOL_FILENAME).read_text())
    normalization = (
        ScalarNormalization.from_state_dict(protocol["normalization"])
        if protocol.get("normalization") is not None
        else None
    )
    model = build_model(cfg, chunk_size, device)
    metadata = load_checkpoint(checkpoint, model=model, map_location=device, restore_rng_state=False)
    logger.info("loaded %s (epoch %s, step %s)", checkpoint, metadata.epoch, metadata.global_step)
    return FittedRun(
        cfg=cfg, kind=str(cfg.model.get("kind", "?")), native_level=resolve_level(cfg),
        normalization=normalization, model=model, epoch=int(metadata.epoch),
    )


def geometry_for_config(
    cfg: DictConfig, kind: str, native_level: int, spec: GridSpec, device: torch.device,
) -> tuple[SphericalGrid, Tensor, Geometry, Tensor]:
    """``(grid, frames, geometry, attention degrees)`` for ``spec``: one
    prepared discretization for the single-level arm on any grid, the
    depth-1 chain with the fitted encode radius on an icosphere. Built from
    the run's config alone: the geometry depends on the architecture's
    cutoffs, not on the weights, so a single geometry serves every seed of
    a run."""
    architecture = cfg.model.architecture
    deep_cutoff = float(architecture.deep_cutoff)
    grid = spec.build(device)

    if kind == SINGLE_LEVEL_KIND:
        conv_cutoff = float(architecture.conv_cutoff)
        frames = build_tangent_frames(grid.nodes)
        discretization = SphericalDiscretization.from_grid(
            grid, coverage_radius=max(deep_cutoff, conv_cutoff), frames=frames
        )
        discretization.prepare_windows([deep_cutoff, conv_cutoff])
        degrees = discretization.edges_within(deep_cutoff).row_ptr.diff()
        return grid, frames, discretization, degrees

    if spec.family != "icosphere":
        raise TypeError(
            f"a {kind!r} run transfers within the icosphere family only: its "
            "IcoEncode/IcoDecode are built on the icosphere's parent-child "
            f"structure, and {spec.label} has none."
        )
    assert spec.level is not None
    chain = build_icosphere_multiresolution(
        spec.level, depth=1,
        coarsest_cutoff=deep_cutoff,
        local_cutoffs={0: float(architecture.decode_refine_cutoff)},
        down_cutoffs={0: fitted_encode_radius(native_level, device)},
        dtype=torch.float32, device=device,
    )
    degrees = chain.levels[1].edges_within(deep_cutoff).row_ptr.diff()
    return chain.finest.grid, chain.finest.frames, chain, degrees


def predict(run: FittedRun, geometry: Geometry, coriolis: Tensor, state: ScalarState) -> Tensor:
    """The reconstructed ``(n_nodes, 3)`` prediction, exactly as the
    training step builds it."""
    conditioned = SO2FeatureField(
        CONDITIONED_FIELD_TYPE, {0: torch.cat([state.input, coriolis.unsqueeze(-1)], dim=-1)}
    )
    with torch.no_grad():
        update = run.model(conditioned, geometry)[0]
    return state.input + update if bool(run.cfg.objective.residual) else update
