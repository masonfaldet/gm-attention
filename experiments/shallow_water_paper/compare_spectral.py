"""Evaluate fitted paper-state models on one spectral yardstick.

Each arm is loaded from its checkpoint and evaluated on its own grid, on the
same held-out trajectories (fixed by the data seeds), at the same lead time,
with its run's own normalization (checked to agree across arms). Every
prediction is projected onto the spherical harmonics of degree ``0 .. L``
inclusive, so the band-limited error is one continuum quantity for all arms
and splits by degree; the native quadrature error, pooled over the
evaluation set, and the targets' band coverage are reported beside it. With
``steps`` > 0 each arm is rolled out autoregressively and scored every hour
against the solver's own continuation of the same trajectory.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from omegaconf import DictConfig
from torch import Tensor

from experiments.shallow_water.grid_transfer import GridSpec
from experiments.shallow_water_paper.transfer import _config_of_run
from experiments.shallow_water_paper.data import (
    ScalarNormalization,
    ScalarShallowWaterDataset,
    ScalarState,
)
from experiments.shallow_water_paper.metrics import ScalarStateMetrics
from experiments.shallow_water_paper.rollout import true_trajectory
from experiments.shallow_water_paper.spectral import (
    GridProjection,
    SpectralErrorAccumulator,
    SpectralProjector,
    target_band_coverage,
)
from experiments.shallow_water_paper.train import (
    PROTOCOL_FILENAME,
    SINGLE_LEVEL_KIND,
    TENSOR_GRID_KINDS,
    ScalarExperiment,
    build_scalar_experiment,
    real_parameter_count,
    resolve_view,
    resolve_view_grid,
    scalar_step,
)
from experiments.shallow_water_paper.transfer import (
    FittedRun,
    geometry_for_config,
    load_run,
    predict,
    resolve_level,
)
from gm_attention.data.shallow_water import coriolis_channel
from gm_attention.discretization.grid import SphericalGrid
from gm_attention.discretization.grids.torch_harmonics import build_torch_harmonics_grid
from gm_attention.training.checkpointing import load_checkpoint

logger = logging.getLogger(__name__)

#: Data fields every arm must agree on for "the same trajectories" to hold.
SHARED_DATA_KEYS = (
    "nlat", "nlon", "lead_time_seconds", "solver_substeps", "mach", "omega",
    "val_seed", "test_seed", "normalization_seed", "normalization_examples",
)


#: Progress-log period, in trajectories.
PROGRESS_EVERY = 8


@dataclass
class LoadedArm:
    """A fitted arm on its native grid, ready to predict."""

    label: str
    cfg: DictConfig
    experiment: ScalarExperiment
    normalization: ScalarNormalization | None
    grid: SphericalGrid
    coriolis: Tensor  # (n_nodes,) for GM kinds, (1, nlat, nlon) for tensor kinds
    #: Keep the input's global mean per channel in every prediction (:func:`constrain_zero_mean`).
    zero_mean_update: bool = False
    #: The evaluation grid of a transferred arm; None means the run's training grid.
    grid_label: str | None = None
    #: Mean number of attention neighbours per node on this grid; None for models without a neighbourhood.
    mean_degree: float | None = None
    #: Provenance: the checkpoint file and its epoch, when loaded from one.
    checkpoint: str | None = None
    epoch: int | None = None

    @property
    def device(self) -> torch.device:
        return self.coriolis.device

    # A LoadedArm is its own source (see ArmSource): already built, never released.
    def build(self, device: torch.device) -> LoadedArm:
        return self

    def release(self) -> int | None:
        return None

    @property
    def kind(self) -> str:
        return str(self.cfg.model.get("kind", "?"))


def arm_from_experiment(
    label: str, cfg: DictConfig, experiment: ScalarExperiment,
    normalization: ScalarNormalization | None, device: torch.device, *, zero_mean_update: bool = False,
) -> LoadedArm:
    """Bind an experiment (a model on its grid or view) into an arm."""
    if experiment.batched:
        assert experiment.view is not None
        nlat, nlon = experiment.view
        grid = build_torch_harmonics_grid(experiment.view_grid, nlat, nlon, dtype=torch.float32, device=device)
        coriolis = coriolis_channel(grid.nodes).reshape(1, nlat, nlon).to(device)
    else:
        assert experiment.grid is not None
        grid = experiment.grid
        coriolis = coriolis_channel(grid.nodes).to(device)
    experiment.model.eval()
    return LoadedArm(label=label, cfg=cfg, experiment=experiment, normalization=normalization,
                     grid=grid, coriolis=coriolis, zero_mean_update=zero_mean_update,
                     mean_degree=attention_mean_degree(cfg, experiment))


def attention_mean_degree(cfg: DictConfig, experiment: ScalarExperiment) -> float | None:
    """Mean number of neighbours each node attends to (or convolves over)
    on the experiment's grid: the deep cutoff's neighbourhood for GM kinds,
    the local S2 transformer's own neighbourhood; None for models without
    one (SFNO is global)."""
    try:
        geometry: Any = experiment.geometry
        if geometry is not None:
            cutoff = float(cfg.model.architecture.deep_cutoff)
            levels = getattr(geometry, "levels", None)
            discretization = levels[1] if levels is not None else geometry
            return float(discretization.edges_within(cutoff).row_ptr.diff().float().mean())
        # torch-harmonics keeps the sparse neighbourhood as CSR offsets; DISCO
        # convolutions register the same buffer name, so only attention modules count.
        degrees = [
            cast(Tensor, getattr(module, "psi_roff_idx")).diff().float().mean()  # noqa: B009
            for module in experiment.model.modules()
            if "Attention" in type(module).__name__ and hasattr(module, "psi_roff_idx")
        ]
        return float(torch.stack(degrees).mean()) if degrees else None
    except Exception:
        logger.warning("could not measure the attention neighbourhood of %s", cfg.model.get("kind"), exc_info=True)
        return None


class ArmSource(Protocol):
    """What :func:`evaluate_arms` needs from an arm before building it
    (its label, config and normalization, for the shared-data checks) and
    the two calls that bracket its evaluation."""

    label: str
    cfg: DictConfig
    normalization: ScalarNormalization | None

    def build(self, device: torch.device) -> LoadedArm: ...

    def release(self) -> int | None:
        """Drop what the arm held; returns the ``id`` of a grid whose last
        user this was (so cached datasets on it can go), else None."""
        ...


@dataclass
class GridContext:
    """One evaluation grid's geometry for a GM run, built on first use,
    shared by every seed transferred onto it and released after the last;
    ``device`` lets a grid too large for the GPU be built on the host."""

    spec: GridSpec
    cfg: DictConfig
    device: torch.device
    users: int = 0
    _built: tuple[SphericalGrid, Any, Tensor, float] | None = field(default=None, repr=False)

    def acquire(self) -> tuple[SphericalGrid, Any, Tensor, float]:
        if self._built is None:
            kind = str(self.cfg.model.get("kind", "?"))
            grid, _, geometry, degrees = geometry_for_config(self.cfg, kind, resolve_level(self.cfg), self.spec, self.device)
            coriolis = coriolis_channel(grid.nodes).to(self.device)
            edges = int(degrees.sum())
            logger.info("%s: %d nodes, %.3g attention edges (about %.1f GiB of geometry), mean degree %.0f, on %s",
                        self.spec.label, grid.n_nodes, edges, edges * 140 / 2**30, float(degrees.float().mean()), self.device)
            self._built = (grid, geometry, coriolis, float(degrees.float().mean()))
        return self._built

    def release(self) -> int | None:
        self.users -= 1
        if self.users <= 0 and self._built is not None:
            grid_id = id(self._built[0])
            self._built = None
            return grid_id
        return None


@dataclass
class LazyArm:
    """An arm built only when :func:`evaluate_arms` reaches it and dropped
    right after: the checkpoint's own grid (``context`` None, via
    :func:`load_arm`) or a transfer onto ``context``'s grid with the
    geometry shared across the seeds that name the same context."""

    label: str
    checkpoint: Path
    chunk_size: int | None = None
    context: GridContext | None = None
    zero_mean_update: bool = False
    cfg: DictConfig = field(init=False)
    normalization: ScalarNormalization | None = field(init=False)
    _arm: LoadedArm | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.cfg = _config_of_run(self.checkpoint)
        protocol = json.loads((self.checkpoint.parent.parent / PROTOCOL_FILENAME).read_text())
        self.normalization = (
            ScalarNormalization.from_state_dict(protocol["normalization"])
            if protocol.get("normalization") is not None else None
        )
        if self.context is not None:
            self.context.users += 1

    def build(self, device: torch.device) -> LoadedArm:
        if self.context is None:
            self._arm = load_arm(self.label, self.checkpoint, device, self.chunk_size, zero_mean_update=self.zero_mean_update)
            return self._arm
        run = load_run(self.checkpoint, self.chunk_size, self.context.device)
        grid, geometry, coriolis, mean_degree = self.context.acquire()
        metrics = ScalarStateMetrics(grid.quadrature_weights).to(self.context.device)
        step = scalar_step(geometry, metrics, coriolis, residual=bool(run.cfg.objective.residual))
        experiment = ScalarExperiment(model=run.model, step_fn=step, grid=grid, geometry=geometry)
        experiment.model.eval()
        logger.info("%s: %s (epoch %s) transferred to %s", self.label, run.kind, run.epoch, self.context.spec.label)
        self._arm = LoadedArm(
            label=self.label, cfg=run.cfg, experiment=experiment, normalization=run.normalization,
            grid=grid, coriolis=coriolis, zero_mean_update=self.zero_mean_update,
            grid_label=self.context.spec.label, mean_degree=mean_degree,
            checkpoint=str(self.checkpoint), epoch=run.epoch,
        )
        return self._arm

    def release(self) -> int | None:
        self._arm = None
        return self.context.release() if self.context is not None else None


def load_arm(
    label: str, checkpoint: Path, device: torch.device, chunk_size: int | None, *, zero_mean_update: bool = False,
) -> LoadedArm:
    cfg = _config_of_run(checkpoint)
    experiment = build_scalar_experiment(cfg, device)
    if str(cfg.model.get("kind", "")) == SINGLE_LEVEL_KIND and chunk_size is not None:
        setattr(experiment.model, "chunk_size", chunk_size)  # noqa: B010 -- nn.Module attribute typing
    metadata = load_checkpoint(checkpoint, model=experiment.model, map_location=device, restore_rng_state=False)
    logger.info("%s: loaded %s (epoch %s)", label, checkpoint, metadata.epoch)
    protocol = json.loads((checkpoint.parent.parent / PROTOCOL_FILENAME).read_text())
    normalization = (
        ScalarNormalization.from_state_dict(protocol["normalization"])
        if protocol.get("normalization") is not None else None
    )
    arm = arm_from_experiment(label, cfg, experiment, normalization, device, zero_mean_update=zero_mean_update)
    arm.checkpoint, arm.epoch = str(checkpoint), int(metadata.epoch)
    return arm


def arm_dataset(arm: LoadedArm, *, num_examples: int, split: str, device: torch.device) -> ScalarShallowWaterDataset:
    cfg = arm.cfg
    seed = int(cfg.data.val_seed if split == "val" else cfg.data.test_seed)
    common = dict(
        base_seed=seed, split=split, normalization=arm.normalization,
        nlat=int(cfg.data.nlat), nlon=int(cfg.data.nlon),
        lead_time_seconds=float(cfg.data.lead_time_seconds), solver_substeps=int(cfg.data.solver_substeps),
        mach=float(cfg.data.mach), omega=float(cfg.data.omega), num_examples=num_examples,
        spectral_cache_dir=cfg.data.get("spectral_cache_dir", None), device=device,
    )
    if arm.experiment.batched:
        return ScalarShallowWaterDataset(view=arm.experiment.view, view_grid=arm.experiment.view_grid, **common)  # type: ignore[arg-type]
    return ScalarShallowWaterDataset(arm.grid.nodes, **common)  # type: ignore[arg-type]


def step_native(arm: LoadedArm, state: Tensor) -> Tensor:
    """One application of the arm to a state in its own layout, ``(n, 3)``
    for GM kinds or ``(3, nlat, nlon)`` for tensor kinds, returning the
    reconstructed next state in the same layout."""
    prediction = _step_native(arm, state)
    return constrain_zero_mean(arm, state, prediction) if arm.zero_mean_update else prediction


def _step_native(arm: LoadedArm, state: Tensor) -> Tensor:
    residual = bool(arm.cfg.objective.residual)
    if arm.experiment.batched:
        conditioned = torch.cat([state.unsqueeze(0), arm.coriolis.unsqueeze(0)], dim=1)  # (1, 4, nlat, nlon)
        with torch.no_grad():
            update = arm.experiment.model(conditioned)[0]  # (3, nlat, nlon)
        return state + update if residual else update
    assert arm.experiment.geometry is not None
    run = FittedRun(cfg=arm.cfg, kind=arm.kind, native_level=resolve_level(arm.cfg),
                    normalization=arm.normalization, model=arm.experiment.model)
    return predict(run, arm.experiment.geometry, arm.coriolis, ScalarState(state, state))


def constrain_zero_mean(arm: LoadedArm, state: Tensor, prediction: Tensor) -> Tensor:
    """The prediction with the input's global mean copied into each
    channel: the update ``prediction - state`` loses its quadrature-weighted
    mean over the sphere.

    The shallow-water tendencies carry a Laplacian factor that vanishes at
    degree 0, so the solver conserves the global mean of geopotential,
    vorticity and divergence exactly, and any predicted change of it is
    error. Subtracting the update's mean changes the degree-0 coefficient
    alone, so at one step the band-limited error loses its ``D(0)`` term
    and nothing else (up to out-of-band content the band fit aliases into
    degree 0 on an icosphere; none on a quadrature grid); along a rollout
    the constraint also stops that error from being fed back. Applied post
    hoc to a fitted model, on its native grid's quadrature.
    """
    weights = arm.grid.quadrature_weights.reshape(-1).to(prediction)
    update = point_major(arm, prediction - state)  # (n_nodes, 3)
    mean = (update * weights.unsqueeze(-1)).sum(dim=0) / weights.sum()  # (3,)
    return prediction - mean.reshape((3, 1, 1) if arm.experiment.batched else (1, 3))


def point_major(arm: LoadedArm, state: Tensor) -> Tensor:
    """``(n_nodes, 3)`` on the arm's grid, whatever its native layout."""
    return state.movedim(0, -1).reshape(-1, 3) if arm.experiment.batched else state


def predict_point_major(arm: LoadedArm, state: ScalarState) -> tuple[Tensor, Tensor]:
    """``(prediction, target)`` as ``(n_nodes, 3)`` on the arm's grid."""
    return point_major(arm, step_native(arm, state.input)), point_major(arm, state.target)


@dataclass
class RolloutStep:
    """Pooled errors after ``step`` applications of an arm to its own output,
    against the solver's state at that hour of the same trajectory."""

    step: int
    hours: float
    rollout: dict[str, float]
    one_step: dict[str, float]
    persistence: dict[str, float]
    D: list[float]
    coverage: float


@dataclass
class ArmResult:
    label: str
    kind: str
    grid: str
    n_nodes: int
    parameters: int
    condition: float
    metrics: dict[str, float]
    D: list[float]
    paper_l1: float
    paper_l2: float
    rollout: list[RolloutStep] = field(default_factory=list)
    zero_mean_update: bool = False
    mean_degree: float | None = None
    #: Provenance: the grid the run trained on, the checkpoint and its epoch.
    training_grid: str = ""
    checkpoint: str | None = None
    epoch: int | None = None
    #: GM kinds: the deep mixer ("attention" or "convolution") and the hidden SO(2) field type as {frequency: copies}.
    mixer: str | None = None
    hidden_type: dict[int, int] | None = None


def gm_architecture_of(cfg: DictConfig) -> tuple[str | None, dict[int, int] | None]:
    """``(mixer, hidden type)`` of a GM run, None for tensor kinds."""
    if not str(cfg.model.get("kind", "")).startswith("gm_attention"):
        return None, None
    a = cfg.model.architecture
    base = {int(k): int(v) for k, v in dict(a.base_field_type).items()}
    multiplicity = int(a.get("multiplicity", 1))
    return str(a.get("deep_mixer", "attention")), {k: v * multiplicity for k, v in base.items()}


def training_grid_of(cfg: DictConfig) -> str:
    """The grid a run trained on: its view for tensor kinds, its icosphere level otherwise."""
    if str(cfg.model.get("kind", "")) in TENSOR_GRID_KINDS:
        nlat, nlon = resolve_view(cfg)
        return f"{resolve_view_grid(cfg)}:{nlat}x{nlon}"
    return f"icosphere:{resolve_level(cfg)}"


def check_arms_share_data(arms: Sequence[ArmSource]) -> None:
    if not arms:
        raise ValueError("at least one arm is required")
    first = arms[0]
    for arm in arms[1:]:
        for key in SHARED_DATA_KEYS:
            a, b = first.cfg.data.get(key), arm.cfg.data.get(key)
            if a != b:
                raise ValueError(f"{arm.label} trained on different data than {first.label}: data.{key} {b!r} vs {a!r}.")
        if (first.normalization is None) != (arm.normalization is None):
            raise ValueError(f"{arm.label} and {first.label} disagree on normalization.")
        if first.normalization is not None and arm.normalization is not None:
            # Compared in the units the loss sees: a relative tolerance on the
            # raw geopotential mean would hide material z-score shifts.
            mean_difference = (first.normalization.mean - arm.normalization.mean) / first.normalization.std
            close = bool((mean_difference.abs() <= 1e-6).all()) and torch.allclose(
                first.normalization.std, arm.normalization.std, rtol=1e-6, atol=0.0)
            if not close:
                raise ValueError(f"{arm.label}'s normalization differs from {first.label}'s; the errors would not share a scale.")


def _pooled(accumulator: SpectralErrorAccumulator) -> tuple[dict[str, float], list[float]]:
    report = accumulator.report()
    D = report.pop("D")
    assert isinstance(D, list)
    return {k: float(v) for k, v in report.items() if not isinstance(v, list)}, D


def rollout_arm(
    arm: LoadedArm, dataset: ScalarShallowWaterDataset, projection: GridProjection | None, *, steps: int, num_examples: int,
    bandwidth: int, hours_per_step: float, teacher_forcing: bool = True,
) -> tuple[list[RolloutStep], list[list[Tensor]]]:
    """``steps`` applications of the arm to its own output on every example,
    each hour scored against the solver's own state of that trajectory:
    the rollout, the teacher-forced one-step error at that hour, and
    persistence, all pooled. Returns the spectral trajectories too, so the
    caller can check the arms rolled out against the same truth.
    ``teacher_forcing`` False skips the forced pass (its ``one_step`` dicts
    come back empty), halving the forwards."""
    kinds = ("rollout", "one_step", "persistence")
    accumulators = {
        k: [SpectralErrorAccumulator(bandwidth, quadrature_weights=arm.grid.quadrature_weights) for _ in range(steps)]
        for k in kinds
    }
    spectral_trajectories: list[list[Tensor]] = []
    per_hour_targets: list[list[Tensor]] = [[] for _ in range(steps)]
    for index in range(num_examples):
        truth, spectral = cast(tuple[list[Tensor], list[Tensor]], true_trajectory(dataset, index, steps, return_spectral=True))
        spectral_trajectories.append(spectral)
        state = truth[0]
        for k in range(1, steps + 1):
            state = step_native(arm, state)
            target = point_major(arm, truth[k])
            accumulators["rollout"][k - 1].add(projection, point_major(arm, state), target)
            if teacher_forcing:
                forced = step_native(arm, truth[k - 1])
                accumulators["one_step"][k - 1].add(projection, point_major(arm, forced), target)
            accumulators["persistence"][k - 1].add(projection, point_major(arm, truth[0]), target)
            per_hour_targets[k - 1].append(spectral[k])
        if (index + 1) % PROGRESS_EVERY == 0 or index + 1 == num_examples:
            logger.info("%s: rollout %d/%d trajectories", arm.label, index + 1, num_examples)
    rows = []
    for k in range(1, steps + 1):
        rollout, D = _pooled(accumulators["rollout"][k - 1])
        one_step = _pooled(accumulators["one_step"][k - 1])[0] if teacher_forcing else {}
        persistence, _ = _pooled(accumulators["persistence"][k - 1])
        rows.append(RolloutStep(
            step=k, hours=k * hours_per_step, rollout=rollout, one_step=one_step, persistence=persistence, D=D,
            coverage=target_band_coverage(per_hour_targets[k - 1], arm.normalization, bandwidth),
        ))
    return rows, spectral_trajectories


def evaluate_arms(
    arms: Sequence[ArmSource], *, bandwidth: int, num_examples: int, split: str, device: torch.device,
    steps: int = 0, projector_device: torch.device | None = None, teacher_forcing: bool = True,
    on_result: Callable[[list[ArmResult], float], None] | None = None,
) -> tuple[list[ArmResult], float]:
    """Every arm on its own grid, the same trajectories, one band; with
    ``steps`` > 0, also rolled out that many hours on the same trajectories.

    Arms are built one at a time and released as soon as they are scored
    (see :class:`LazyArm`); datasets are cached per grid, so seeds sharing
    a :class:`GridContext` share the synthesis too, and dropped when the
    grid's last user releases it. ``projector_device`` (default ``device``)
    is where the band fits live; the host, for grids whose float64 basis
    would not fit beside the model. ``on_result`` is called with the
    results so far after every arm, so a caller can save them as it goes.
    """
    if num_examples < 1:
        raise ValueError("num_examples must be at least 1")
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    if split not in ("val", "test"):
        raise ValueError("split must be 'val' or 'test'")
    check_arms_share_data(arms)
    if len({arm.label for arm in arms}) != len(arms):
        raise ValueError("arm labels must be unique")
    projector = SpectralProjector(bandwidth, projector_device or device)
    hours_per_step = float(arms[0].cfg.data.lead_time_seconds) / 3600.0

    datasets: dict[int, ScalarShallowWaterDataset] = {}
    targets: list[Tensor] | None = None
    coverage = 0.0
    results = []
    reference_trajectories: list[list[Tensor]] | None = None
    for source in arms:
        arm = source.build(device)
        key = id(arm.grid)
        dataset = datasets.get(key)
        if dataset is None:
            dataset = arm_dataset(arm, num_examples=num_examples, split=split, device=arm.device)
            # The same spectral trajectories for everyone: read once, and
            # every further grid checked against them rather than assumed.
            first = dataset.source.spectral_pair(0)[1]
            if targets is None:
                targets = [dataset.source.spectral_pair(i)[1] for i in range(num_examples)]
                coverage = target_band_coverage(targets, arm.normalization, bandwidth)
            elif not torch.allclose(first.to(targets[0].device), targets[0]):
                raise RuntimeError(f"{arm.label} does not share the held-out trajectories")
            datasets[key] = dataset

        projection: GridProjection | None
        if arm.grid.n_nodes < projector.n_basis:
            # Too few points to determine the band's coefficients: this row keeps
            # its native-grid errors and carries no band-limited ones.
            projection = None
            logger.info("%s: %s, %d nodes, fewer than the band's %d coefficients: native errors only",
                        arm.label, arm.kind, arm.grid.n_nodes, projector.n_basis)
        else:
            projection = projector.fit(arm.grid)
            logger.info("%s: %s, %d nodes, fit condition %.2f", arm.label, arm.kind, arm.grid.n_nodes, projection.condition)
        accumulator = SpectralErrorAccumulator(bandwidth, quadrature_weights=arm.grid.quadrature_weights)
        paper = ScalarStateMetrics(arm.grid.quadrature_weights).to(arm.device)
        l1_sum = l2_sum = 0.0
        for index in range(num_examples):
            prediction, target = predict_point_major(arm, dataset[index])
            accumulator.add(projection, prediction, target)
            paper_metrics = paper(prediction, target)
            l1_sum += float(paper_metrics["paper_l1"]); l2_sum += float(paper_metrics["paper_l2"])
            if (index + 1) % PROGRESS_EVERY == 0 or index + 1 == num_examples:
                logger.info("%s: one step %d/%d", arm.label, index + 1, num_examples)
        report = accumulator.report()
        D = report.pop("D")
        assert isinstance(D, list)
        view = arm.experiment.view
        grid_label = arm.grid_label or (
            f"{arm.experiment.view_grid}:{view[0]}x{view[1]}" if view is not None else f"icosphere:{resolve_level(arm.cfg)}"
        )
        rollout_rows: list[RolloutStep] = []
        if steps > 0:
            rollout_rows, trajectories = rollout_arm(
                arm, dataset, projection, steps=steps, num_examples=num_examples,
                bandwidth=bandwidth, hours_per_step=hours_per_step, teacher_forcing=teacher_forcing,
            )
            # The same solver states at every hour for every arm -- checked
            # at the last hour, where any divergence would have compounded.
            if reference_trajectories is None:
                reference_trajectories = trajectories
            else:
                for mine, theirs in zip(trajectories, reference_trajectories, strict=True):
                    if not torch.allclose(mine[-1].to(theirs[-1].device), theirs[-1]):
                        raise RuntimeError(f"{arm.label} rolled out against a different trajectory than {arms[0].label}")
        mixer, hidden_type = gm_architecture_of(arm.cfg)
        results.append(ArmResult(
            label=arm.label, kind=arm.kind, grid=grid_label,
            n_nodes=int(arm.grid.n_nodes), parameters=real_parameter_count(arm.experiment.model),
            condition=projection.condition if projection is not None else float("nan"),
            metrics={k: float(v) for k, v in report.items() if not isinstance(v, list)},
            D=D, paper_l1=l1_sum / num_examples, paper_l2=l2_sum / num_examples, rollout=rollout_rows,
            zero_mean_update=arm.zero_mean_update, mean_degree=arm.mean_degree,
            training_grid=training_grid_of(arm.cfg), checkpoint=arm.checkpoint, epoch=arm.epoch,
            mixer=mixer, hidden_type=hidden_type,
        ))
        del projection, accumulator, paper, arm
        released = source.release()
        if released is not None:
            datasets.pop(released, None)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if on_result is not None:
            on_result(results, coverage)
    return results, coverage
