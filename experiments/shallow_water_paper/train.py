"""Hydra entry point: training on the (geopotential, vorticity, divergence)
state.

"Attention on the Sphere" (Bonev et al., arXiv:2505.11157, Appendix C.2)
trains on the three scalars ``torch_harmonics``' ``PdeDataset`` returns and
scores with the componentwise sphere-integrated losses of its Eq. 13-14.
Every arm is trained on that representation: the data is
:class:`~experiments.shallow_water_paper.data.ScalarShallowWaterDataset`
(the three scalars at the icosphere nodes, or on the reference's tensor
view, z-scored per channel); the loss is the squared ``L2`` averaged over
the three channels (``paper_squared_l2``), which checkpoints and early
stopping monitor; the GM field types are ``{0: 4}`` in (the state plus the
Coriolis scalar) and ``{0: 3}`` out. Under the residual objective the model
predicts an update that is added to the three-channel input outside the
model.

Usage::

    uv run python -m experiments.shallow_water_paper.train -cp <config dir> -cn <arm> seed=<n>
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch_harmonics.examples.losses import get_quadrature_weights

from experiments.shallow_water_paper.data import (
    CHANNELS,
    ScalarNormalization,
    ScalarShallowWaterDataset,
    ScalarState,
    calibrate_scalar_normalization,
)
from experiments.shallow_water_paper.metrics import MONITOR_METRIC, ScalarStateMetrics
from gm_attention.data.shallow_water import coriolis_channel
from gm_attention.discretization.discretization import SphericalDiscretization
from gm_attention.discretization.frames import build_tangent_frames
from gm_attention.discretization.grid import SphericalGrid
from gm_attention.discretization.grids.icosphere import (
    IcosphereMultiresolution,
    build_icosphere_grid,
    build_icosphere_multiresolution,
    build_icosphere_refinement,
)
from gm_attention.discretization.grids.torch_harmonics import build_torch_harmonics_grid
from gm_attention.representations.field_type import SO2FeatureField, SO2FieldType
from gm_attention.training.checkpointing import CheckpointManager
from gm_attention.training.engine import StepFn, StepOutput, Trainer
from gm_attention.training.logging import MetricLogger
from gm_attention.training.provenance import code_revision, runtime_switches
from gm_attention.training.reproducibility import set_seed
from gm_attention.training.schedules import (
    ScheduleSpec,
    build_scheduler,
    check_resumed_scheduler,
)

log = logging.getLogger(__name__)

#: The model's output: the three prognostic scalars.
STATE_FIELD_TYPE = SO2FieldType({0: len(CHANNELS)})
#: The model's input: the three scalars plus the Coriolis scalar ``c`` (invariant, so at frequency 0).
CONDITIONED_FIELD_TYPE = SO2FieldType({0: len(CHANNELS) + 1})

PROTOCOL_FILENAME = "paper_protocol.json"

#: The single-level arm: one discretization with two prepared radii instead of the depth-1 chain.
SINGLE_LEVEL_KIND = "gm_attention_single"

#: The forward's second argument: the depth-1 chain for GmAttentionDepth1, one level for GmAttentionSingleLevel.
Geometry = IcosphereMultiresolution | SphericalDiscretization

#: The reference models: a tensor grid, batched, (batch, 4, nlat, nlon) in and (batch, 3, nlat, nlon) out.
TENSOR_GRID_KINDS = ("attention_on_sphere", "sfno")


@dataclass
class ScalarExperiment:
    """What :func:`build_scalar_experiment` produces: the model on the run's
    device, the step both phases use, and where its data lives -- an
    icosphere ``grid`` with the ``geometry`` the forward takes for the GM
    arms, or an equiangular ``view`` ``(nlat, nlon)`` for the reference."""

    model: nn.Module
    step_fn: StepFn
    grid: SphericalGrid | None = None
    geometry: Geometry | None = None
    view: tuple[int, int] | None = None
    #: The view's grid family (tensor kinds): ``equiangular`` or ``legendre-gauss``.
    view_grid: str = "equiangular"

    @property
    def nodes(self) -> Tensor | None:
        return self.grid.nodes if self.grid is not None else None

    @property
    def n_nodes(self) -> int:
        if self.grid is not None:
            return int(self.grid.n_nodes)
        assert self.view is not None
        return self.view[0] * self.view[1]

    @property
    def batched(self) -> bool:
        """Whether the model takes a batch tensor (the reference) rather
        than one graph per call (every GM kind)."""
        return self.view is not None


def resolve_view(cfg: DictConfig) -> tuple[int, int]:
    """The ``(nlat, nlon)`` the reference model sees: the config's
    ``model_view``, else the solver grid."""
    view = cfg.model.get("model_view") or cfg.data.get("model_view")
    if view is None:
        return int(cfg.data.nlat), int(cfg.data.nlon)
    return int(view.nlat), int(view.nlon)


def resolve_view_grid(cfg: DictConfig) -> str:
    """The grid family of the reference's view, ``model_view.grid``
    (``equiangular`` unless the config says ``legendre-gauss``)."""
    view = cfg.model.get("model_view") or cfg.data.get("model_view")
    return str(view.get("grid", "equiangular")) if view is not None else "equiangular"


def resolve_level(cfg: DictConfig) -> int:
    """The icosphere level the run builds at: a model config's own
    ``subdivisions`` over the shared ``grid``."""
    if "subdivisions" in cfg.model:
        return int(cfg.model.subdivisions)
    return int(cfg.grid.subdivisions)


def resolve_device(spec: str | None) -> torch.device:
    if spec:
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_scalar_experiment(cfg: DictConfig, device: torch.device) -> ScalarExperiment:
    """Build the arm ``cfg.model`` names, with a GM model's two field types
    set to the three-scalar state.

    Depth-1 arms get the depth-1 chain with the encode radius set to the
    hop's coarse edge length; the single-level arm gets one discretization
    with both configured radii prepared; the reference arms get their
    tensor view with its quadrature and Coriolis map.
    """
    architecture = cfg.model.architecture
    kind = str(cfg.model.get("kind", ""))
    residual = bool(cfg.objective.residual)

    if kind in TENSOR_GRID_KINDS:
        # The reference config declares in_chans=4 / out_chans=3 itself;
        # quadrature and the Coriolis map live on the view grid.
        nlat, nlon = resolve_view(cfg)
        family = resolve_view_grid(cfg)
        reference: nn.Module = hydra.utils.instantiate(architecture).to(device)
        weights = (
            get_quadrature_weights(nlat, nlon, grid=family, tile=True, normalized=True)
            .reshape(-1)
            .to(dtype=torch.float32, device=device)
        )
        view_grid = build_torch_harmonics_grid(family, nlat, nlon, dtype=torch.float32, device=device)
        coriolis_map = coriolis_channel(view_grid.nodes).reshape(1, nlat, nlon).to(device)
        return ScalarExperiment(
            model=reference,
            step_fn=reference_step(ScalarStateMetrics(weights).to(device), coriolis_map, residual=residual),
            view=(nlat, nlon), view_grid=family,
        )

    level = resolve_level(cfg)
    field_types = {
        "input_field_type": {0: len(CHANNELS) + 1},
        "output_field_type": {0: len(CHANNELS)},
    }
    geometry: Geometry
    grid: SphericalGrid

    if kind == SINGLE_LEVEL_KIND:
        grid = build_icosphere_grid(level, dtype=torch.float32, device=device)
        deep_cutoff = float(architecture.deep_cutoff)
        conv_cutoff = float(architecture.conv_cutoff)
        discretization = SphericalDiscretization.from_grid(
            grid, coverage_radius=max(deep_cutoff, conv_cutoff), frames=build_tangent_frames(grid.nodes)
        )
        discretization.prepare_windows([deep_cutoff, conv_cutoff])
        geometry = discretization
        model: nn.Module = hydra.utils.instantiate(architecture, **field_types).to(device)
    else:
        _, _, refinement = build_icosphere_refinement(level - 1, dtype=torch.float32, device=device)
        encode_cutoff = float(refinement.coarse_edge_length)
        multiresolution = build_icosphere_multiresolution(
            level,
            depth=1,
            coarsest_cutoff=float(architecture.deep_cutoff),
            local_cutoffs={0: float(architecture.decode_refine_cutoff)},
            down_cutoffs={0: encode_cutoff},
            dtype=torch.float32,
            device=device,
        )
        geometry = multiresolution
        grid = multiresolution.finest.grid
        model = hydra.utils.instantiate(
            architecture, encode_cutoff=encode_cutoff, **field_types
        ).to(device)

    metrics = ScalarStateMetrics(grid.quadrature_weights).to(device)
    coriolis = coriolis_channel(grid.nodes).to(device)
    step_fn = scalar_step(geometry, metrics, coriolis, residual=residual)
    return ScalarExperiment(model=model, step_fn=step_fn, grid=grid, geometry=geometry)


def reference_step(
    metrics: ScalarStateMetrics, coriolis: Tensor, *, residual: bool
) -> StepFn:
    """The reference model's batched step: ``(batch, 3, nlat, nlon)`` in,
    the static ``(1, nlat, nlon)`` Coriolis map concatenated as a fourth
    channel, three prognostic channels out. Scored on the flattened view
    with the same metrics as the GM arms, under the view's quadrature.
    """

    def step(model: nn.Module, batch: ScalarState, device: torch.device) -> StepOutput:
        prognostic = batch.input  # (batch, 3, nlat, nlon)
        conditioned = torch.cat(
            [prognostic, coriolis.expand(prognostic.shape[0], -1, -1, -1)], dim=-3
        )
        update = model(conditioned)  # (batch, 3, nlat, nlon)
        prediction = prognostic + update if residual else update

        size, channels = prediction.shape[0], prediction.shape[1]
        flat_prediction = prediction.permute(0, 2, 3, 1).reshape(size, -1, channels)
        flat_target = batch.target.permute(0, 2, 3, 1).reshape(size, -1, channels)
        result = metrics(flat_prediction, flat_target)
        return StepOutput(
            loss=result[MONITOR_METRIC],
            metrics={name: float(value.detach()) for name, value in result.items()},
        )

    return step


def scalar_step(
    geometry: Geometry,
    metrics: ScalarStateMetrics,
    coriolis: Tensor,
    *,
    residual: bool,
) -> StepFn:
    """One sample per call; ``Trainer``'s accumulation averages several
    into one optimizer step.

    The Coriolis channel is appended on the way in only: under ``residual``
    the three-channel input is added to the model's three-channel update,
    so ``c`` is neither predicted nor added into the prediction.
    """

    def step(model: nn.Module, sample: ScalarState, device: torch.device) -> StepOutput:
        conditioned = SO2FeatureField(
            CONDITIONED_FIELD_TYPE,
            {0: torch.cat([sample.input, coriolis.unsqueeze(-1)], dim=-1)},
        )
        update = model(conditioned, geometry)[0]  # (n_nodes, 3)
        prediction = sample.input + update if residual else update

        result = metrics(prediction, sample.target)
        return StepOutput(
            loss=result[MONITOR_METRIC],
            metrics={name: float(value.detach()) for name, value in result.items()},
        )

    return step


# ---------------------------------------------------------------- data


def build_normalization(cfg: DictConfig, device: torch.device) -> ScalarNormalization | None:
    if not cfg.data.normalize:
        return None
    return calibrate_scalar_normalization(
        seed=int(cfg.data.normalization_seed),
        num_examples=int(cfg.data.normalization_examples),
        nlat=int(cfg.data.nlat),
        nlon=int(cfg.data.nlon),
        lead_time_seconds=float(cfg.data.lead_time_seconds),
        solver_substeps=int(cfg.data.solver_substeps),
        mach=float(cfg.data.mach),
        omega=float(cfg.data.omega),
        spectral_cache_dir=cfg.data.get("spectral_cache_dir", None),
        dtype=torch.float32,
        device=device,
    )


def build_datasets(
    cfg: DictConfig,
    experiment: ScalarExperiment,
    device: torch.device,
    normalization: ScalarNormalization | None,
) -> tuple[ScalarShallowWaterDataset, ScalarShallowWaterDataset]:
    """Train and validation sets where the experiment's model reads them:
    at the icosphere's nodes, or on the reference's equiangular view."""
    nodes = experiment.nodes
    shared: dict[str, Any] = {
        "view": experiment.view,
        "view_grid": experiment.view_grid,
        "normalization": normalization,
        "nlat": int(cfg.data.nlat),
        "nlon": int(cfg.data.nlon),
        "lead_time_seconds": float(cfg.data.lead_time_seconds),
        "solver_substeps": int(cfg.data.solver_substeps),
        "mach": float(cfg.data.mach),
        "omega": float(cfg.data.omega),
        "spectral_cache_dir": cfg.data.get("spectral_cache_dir", None),
        "device": device,
    }
    train = ScalarShallowWaterDataset(
        nodes,
        base_seed=int(cfg.data.train_seed),
        split="train",
        sample_policy=cfg.data.train_sample_policy,
        num_examples=int(cfg.data.num_train_examples),
        **shared,
    )
    val = ScalarShallowWaterDataset(
        nodes,
        base_seed=int(cfg.data.val_seed),
        split="val",
        sample_policy=cfg.data.val_sample_policy,
        num_examples=int(cfg.data.num_val_examples),
        **shared,
    )
    return train, val


class _IndexStandIn(torch.utils.data.Dataset[int]):
    """Ints only, to replay the shuffle stream on resume without solving."""

    def __init__(self, length: int) -> None:
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> int:
        return index


def build_loaders(
    cfg: DictConfig,
    train: ScalarShallowWaterDataset,
    val: ScalarShallowWaterDataset,
    *,
    batched: bool = False,
    skip_epochs: int = 0,
) -> tuple[DataLoader, DataLoader, int]:
    """Loaders and the Trainer's accumulation steps.

    ``training.batch_size`` is the number of samples averaged into one
    optimizer step for every arm, realized two ways: a GM kind takes one
    graph per call, so it gets single-sample loaders and
    ``accumulation_steps=batch_size``; the reference batches normally and
    accumulates nothing, unless ``training.micro_batch_size`` splits its
    batch into micro-batches whose gradients accumulate to the same step.
    The shuffle order has its own seed; a resume fast-forwards it so epoch
    ``N`` of a resumed run sees the order epoch ``N`` would have had.
    """
    order = torch.Generator()
    order.manual_seed(int(cfg.data.loader_order_seed))
    batch_size = int(cfg.training.batch_size)
    micro = cfg.training.get("micro_batch_size")
    loader_batch = batch_size
    if batched and micro is not None:
        micro = int(micro)
        if micro < 1 or batch_size % micro:
            raise ValueError(f"training.micro_batch_size must divide batch_size ({batch_size}), got {micro}.")
        loader_batch = micro
    if skip_epochs > 0:
        stand_in = DataLoader(
            _IndexStandIn(len(train)),
            batch_size=loader_batch if batched else None,
            shuffle=True,
            generator=order,
        )
        for _ in range(skip_epochs):
            for _ in stand_in:
                pass
    if batched:
        train_loader = DataLoader(train, batch_size=loader_batch, shuffle=True, generator=order)
        val_loader = DataLoader(val, batch_size=loader_batch, shuffle=False)
        return train_loader, val_loader, batch_size // loader_batch
    train_loader = DataLoader(train, batch_size=None, shuffle=True, generator=order)
    val_loader = DataLoader(val, batch_size=None, shuffle=False)
    return train_loader, val_loader, batch_size


# ------------------------------------------------------------ protocol

#: Protocol keys that define the data; a run directory whose saved protocol differs on any of them is not trained into.
DATA_KEYS = (
    "task", "channels", "normalization",
    "train_seed", "val_seed", "test_seed", "normalization_seed", "loader_order_seed",
    "normalization_examples", "train_sample_policy", "val_sample_policy",
    "nlat", "nlon", "lead_time_seconds", "solver_substeps", "mach", "omega",
    "grid_subdivisions", "grid_n_nodes", "model_view", "num_train_examples", "num_val_examples",
)


def real_parameter_count(model: nn.Module) -> int:
    """Parameters as real degrees of freedom: a complex weight counts two.

    ``numel`` counts a complex number once, which halves SFNO's spectral
    weights against the real weights of the transformer and the GM arms;
    every parameter match in this benchmark is made on this count.
    """
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())


def build_protocol(
    cfg: DictConfig, normalization: ScalarNormalization | None, experiment: ScalarExperiment
) -> dict[str, Any]:
    """What this run's data and model are, as plain JSON values."""
    model = experiment.model
    return {
        "task": "(geopotential, vorticity, divergence)",
        "channels": list(CHANNELS),
        "normalization": normalization.state_dict() if normalization is not None else None,
        "train_seed": int(cfg.data.train_seed),
        "val_seed": int(cfg.data.val_seed),
        "test_seed": int(cfg.data.test_seed),
        "normalization_seed": int(cfg.data.normalization_seed),
        "loader_order_seed": int(cfg.data.loader_order_seed),
        "normalization_examples": int(cfg.data.normalization_examples),
        "train_sample_policy": str(cfg.data.train_sample_policy),
        "val_sample_policy": str(cfg.data.val_sample_policy),
        "nlat": int(cfg.data.nlat),
        "nlon": int(cfg.data.nlon),
        "lead_time_seconds": float(cfg.data.lead_time_seconds),
        "solver_substeps": int(cfg.data.solver_substeps),
        "mach": float(cfg.data.mach),
        "omega": float(cfg.data.omega),
        "grid_subdivisions": None if experiment.batched else resolve_level(cfg),
        "grid_n_nodes": experiment.n_nodes,
        "model_view": list(experiment.view) if experiment.view is not None else None,
        "model_view_grid": experiment.view_grid if experiment.view is not None else None,
        "num_train_examples": int(cfg.data.num_train_examples),
        "num_val_examples": int(cfg.data.num_val_examples),
        "model": str(cfg.model.get("kind", "?")),
        "model_seed": int(cfg.seed),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "real_parameter_count": real_parameter_count(model),
        # Provenance, not data (a resume verifies DATA_KEYS only): the code
        # and the implementation switches the run was launched under.
        "code_revision": code_revision(),
        "switches": runtime_switches(),
    }


def _same_value(a: Any, b: Any) -> bool:
    """Equality with a relative tolerance on floats, so a normalization
    re-estimated on another device still verifies."""
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same_value(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same_value(x, y) for x, y in zip(a, b, strict=True))
    return bool(a == b)


def verify_or_save_protocol(protocol: dict[str, Any], directory: str | Path = ".") -> Path:
    """Write the protocol into a fresh run directory, or verify it against
    the one already there: the checkpoints in that directory were trained
    against the saved data, and a resume must not quietly change it."""
    path = Path(directory) / PROTOCOL_FILENAME
    if path.exists():
        previous = json.loads(path.read_text())
        differing = [key for key in DATA_KEYS if not _same_value(previous.get(key), protocol[key])]
        if differing:
            raise ValueError(
                f"{path} describes different data than this config would produce "
                f"(differing: {', '.join(differing)}). Refusing to overwrite it: the "
                "checkpoints here were trained against the saved protocol. Point "
                "hydra.run.dir at a fresh directory, or restore the config that "
                "produced this one."
            )
        log.info("data protocol verified against %s", path)
        return path
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    log.info("data protocol -> %s", path)
    return path


# ---------------------------------------------------------------- main


def _resume_point(cfg: DictConfig) -> tuple[Path | None, int]:
    """``(checkpoint, completed_epochs)`` to continue from, or ``(None, 0)``."""
    if not bool(cfg.training.get("resume", False)):
        return None, 0
    candidate = Path("checkpoints") / "last.pt"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"training.resume=true but {candidate.resolve()} does not exist; "
            "nothing to continue. Start fresh without the flag."
        )
    payload = torch.load(candidate, map_location="cpu", weights_only=False)
    completed = int(payload["epoch"]) + 1
    check_resumed_scheduler(
        payload.get("scheduler"),
        ScheduleSpec.from_mapping(cfg.training.scheduler),
        max_epochs=int(cfg.training.max_epochs),
        completed_epochs=completed,
    )
    log.info("resuming after %d completed epochs from %s", completed, candidate)
    return candidate, completed


@hydra.main(version_base=None, config_path="../../configs", config_name="paper")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.seed))
    device = resolve_device(cfg.device)
    log.info("paper-state experiment: model=%s device=%s", cfg.model.get("kind", "?"), device)

    experiment = build_scalar_experiment(cfg, device)
    log.info("model parameters: %d", sum(p.numel() for p in experiment.model.parameters()))

    normalization = build_normalization(cfg, device)
    if normalization is not None:
        log.info("normalization: %s", normalization.state_dict())
    verify_or_save_protocol(build_protocol(cfg, normalization, experiment))

    resume_from, completed_epochs = _resume_point(cfg)

    train_dataset, val_dataset = build_datasets(cfg, experiment, device, normalization)
    train_loader, val_loader, accumulation_steps = build_loaders(
        cfg, train_dataset, val_dataset, batched=experiment.batched, skip_epochs=completed_epochs
    )

    optimizer = torch.optim.Adam(
        experiment.model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    spec = ScheduleSpec.from_mapping(cfg.training.scheduler)
    if spec.restart and resume_from is None:
        raise ValueError(
            "training.scheduler.restart=true requires training.resume=true: a "
            "fresh run's cosine already starts at training.lr."
        )
    scheduler, scheduler_metric = build_scheduler(
        optimizer,
        spec,
        max_epochs=int(cfg.training.max_epochs),
        monitor=MONITOR_METRIC,
        completed_epochs=completed_epochs,
    )
    if scheduler_metric is None and cfg.training.early_stopping_patience is not None:
        log.warning(
            "early_stopping_patience=%s with a cosine schedule: pass "
            "training.early_stopping_patience=null to let the schedule run out.",
            cfg.training.early_stopping_patience,
        )

    trainer = Trainer(
        experiment.model,
        optimizer,
        experiment.step_fn,
        eval_step=experiment.step_fn,
        scheduler=scheduler,
        scheduler_step_on="epoch",
        scheduler_metric=scheduler_metric,
        accumulation_steps=accumulation_steps,
        device=device,
        grad_clip_norm=cfg.training.grad_clip_norm,
        logger=MetricLogger("logs"),
        checkpoint_manager=CheckpointManager(
            "checkpoints",
            monitor=MONITOR_METRIC,
            mode="min",
            keep_periodic_every=cfg.training.checkpoint_every,
        ),
        early_stopping_patience=cfg.training.early_stopping_patience,
        validate_every=int(cfg.training.validate_every),
        on_epoch_start=train_dataset.set_epoch,
    )
    trainer.fit(
        train_loader,
        max_epochs=int(cfg.training.max_epochs),
        val_loader=val_loader,
        resume_from=resume_from,
        restore_scheduler=not spec.restart,
    )


if __name__ == "__main__":
    main()
