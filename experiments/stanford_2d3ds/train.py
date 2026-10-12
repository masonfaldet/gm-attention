"""Hydra entry point: Stanford 2D3DS segmentation under the released protocol.

"Attention on the Sphere" (NeurIPS 2025, Section 4.1) trains its SegFormers
on 128x256 panoramas with a quadrature-weighted cross-entropy loss, AdamW at
weight decay 0.1, a cosine schedule over 200 epochs, drop path 0.1, batches
of 8, and reports cross-entropy, IoU and accuracy on a held-out 2.5 % split.
This script runs that loop (``configs/training/segformer.yaml``) for two
kinds of model:

- ``gm_segformer``: :class:`~experiments.stanford_2d3ds.model.GmSegformer`,
  one panorama per forward call as a scalar field on the equiangular grid,
  the released batch size realized as gradient accumulation, or, with
  ``training.panoramas_per_forward=B``, ``B`` panoramas per call as one
  graph (the hierarchy tiled into a disjoint union), the same numbers to
  summation order;
- ``s2_segformer``, ``s2_unet``, ``r2_segformer``: the released models,
  batched.

Both kinds are scored by the released loss and metric classes on the same
``(batch, classes, nlat, nlon)`` tensors. Run from the repository root::

    uv run python -m experiments.stanford_2d3ds.train -cp S2D3DS_128x256_Segmentation/configs -cn sos2_segformer seed=0
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch_harmonics.examples.losses import CrossEntropyLossS2, get_quadrature_weights

from experiments.stanford_2d3ds.data import (
    IGNORE_INDEX,
    SegmentationSplits,
    dataset_digest,
    image_to_field,
    images_to_field,
    load_splits,
    logits_to_image,
    logits_to_images,
)
from experiments.stanford_2d3ds.geometry import (
    SegformerRadii,
    build_segformer_hierarchy,
    segformer_radii,
    segformer_stage_shapes,
)
from experiments.stanford_2d3ds.metrics import AccuracyS2, IntersectionOverUnionS2
from experiments.stanford_2d3ds.model import real_parameter_count
from gm_attention.discretization.edge_geometry import cutoff_rule
from gm_attention.discretization.grids.equiangular import TensorGridHierarchy
from gm_attention.nn import stochastic_depth
from gm_attention.training.checkpointing import CheckpointManager
from gm_attention.training.engine import StepFn, StepOutput, Trainer
from gm_attention.training.logging import MetricLogger
from gm_attention.training.provenance import code_revision
from gm_attention.training.reproducibility import set_seed
from gm_attention.training.schedules import (
    ScheduleSpec,
    build_scheduler,
    check_resumed_scheduler,
)

log = logging.getLogger(__name__)

#: The training objective and the monitored validation metric.
MONITOR_METRIC = "cross_entropy"
PROTOCOL_FILENAME = "segmentation_protocol.json"
GM_KIND = "gm_segformer"
#: The released models, batched, reading (batch, 3, nlat, nlon) tensors.
TENSOR_KINDS = ("s2_segformer", "s2_unet", "r2_segformer")


@dataclass
class SegmentationScorer:
    """The released loss and metrics on one ``(batch, C, nlat, nlon)`` tensor.

    ``accuracy`` is Eq. (23) of "Attention on the Sphere": per-class true
    positives plus true negatives, averaged over classes, which equals
    ``1 - 2 U / C`` for a mismatched area fraction ``U`` and ``C`` classes
    and so depends on the class count. ``pixel_accuracy`` is the
    quadrature-weighted fraction of valid pixels labelled correctly, which
    does not.
    """

    loss: nn.Module
    iou: nn.Module
    accuracy: nn.Module
    quadrature: Tensor  # (nlat, nlon), normalized
    ignore_index: int = IGNORE_INDEX

    @classmethod
    def build(
        cls, shape: tuple[int, int], *, grid: str, label_smoothing: float, device: torch.device
    ) -> SegmentationScorer:
        nlat, nlon = shape
        quadrature = get_quadrature_weights(nlat=nlat, nlon=nlon, grid=grid, tile=True).to(device)
        return cls(
            loss=CrossEntropyLossS2(nlat=nlat, nlon=nlon, grid=grid, weight=None, smooth=label_smoothing).to(device),
            iou=IntersectionOverUnionS2(nlat=nlat, nlon=nlon, grid=grid, weight=None).to(device),
            accuracy=AccuracyS2(nlat=nlat, nlon=nlon, grid=grid, weight=None).to(device),
            quadrature=quadrature / quadrature.sum(),
        )

    def confusion(self, logits: Tensor, target: Tensor) -> Tensor:  # (batch, C, C)
        """Quadrature-weighted confusion matrix per panorama: row = target
        class, column = predicted class, entries the area of the valid pixels
        with that pair (normalized grid weights, so a panorama's entries sum
        to its valid area fraction). The released IoU and accuracy are
        computed by the released classes, not from it."""
        num_classes = logits.shape[1]
        prediction = logits.argmax(dim=1)
        valid = target != self.ignore_index
        weights = (self.quadrature.expand_as(target) * valid).to(logits.dtype)
        index = (target.clamp_min(0) * num_classes + prediction).flatten(1)
        flat = torch.zeros(target.shape[0], num_classes * num_classes, dtype=logits.dtype, device=logits.device)
        flat.scatter_add_(1, index, weights.flatten(1))
        return flat.reshape(target.shape[0], num_classes, num_classes)

    @staticmethod
    def class_iou_from_confusion(confusion: Tensor) -> Tensor:  # (batch, C)
        """Per-class IoU ``tp / (tp + fp + fn)``; NaN for a class absent from
        both the target and the prediction."""
        tp = confusion.diagonal(dim1=-2, dim2=-1)
        union = confusion.sum(dim=-1) + confusion.sum(dim=-2) - tp
        return torch.where(union > 0, tp / union.clamp_min(torch.finfo(confusion.dtype).tiny), torch.nan)

    def class_iou(self, logits: Tensor, target: Tensor) -> Tensor:  # (batch, C)
        return self.class_iou_from_confusion(self.confusion(logits, target))

    def mean_class_iou(self, logits: Tensor, target: Tensor) -> Tensor:
        """mIoU: the per-class IoU averaged over the classes present in the
        target or the prediction, then over the batch."""
        return self.class_iou(logits, target).nanmean(dim=-1).nanmean()

    def pixel_accuracy(self, logits: Tensor, target: Tensor) -> Tensor:
        """Area fraction of the valid pixels whose argmax matches the target;
        NaN for a panorama with no valid pixel, as the released IoU is."""
        prediction = logits.argmax(dim=1)
        valid = target != self.ignore_index
        weights = self.quadrature.expand_as(target)
        correct = (weights * (valid & (prediction == target))).sum(dim=(-2, -1))
        covered = (weights * valid).sum(dim=(-2, -1))
        return torch.where(covered > 0, correct / covered.clamp_min(torch.finfo(weights.dtype).tiny), torch.nan).nanmean()

    def __call__(self, logits: Tensor, target: Tensor) -> StepOutput:
        """``logits`` ``(batch, C, nlat, nlon)``, ``target`` ``(batch, nlat, nlon)``."""
        loss = self.loss(logits, target)
        with torch.no_grad():
            iou = self.iou(logits, target)
            accuracy = self.accuracy(logits, target)
            pixel_accuracy = self.pixel_accuracy(logits, target)
            mean_class_iou = self.mean_class_iou(logits, target)
        return StepOutput(
            loss=loss,
            metrics={
                MONITOR_METRIC: float(loss.detach()),
                "iou": float(iou),
                "accuracy": float(accuracy),
                "pixel_accuracy": float(pixel_accuracy),
                "mean_class_iou": float(mean_class_iou),
            },
        )


@dataclass
class SegmentationExperiment:
    """The model on the run's device, the step both phases use, and its geometry."""

    model: nn.Module
    step_fn: StepFn
    shape: tuple[int, int]
    num_classes: int
    hierarchy: TensorGridHierarchy | None = None
    radii: SegformerRadii | None = None
    stage_shapes: tuple[tuple[int, int], ...] = ()

    @property
    def batched(self) -> bool:
        """Whether the model takes a batch tensor rather than one graph per call."""
        return self.hierarchy is None


def resolve_device(spec: str | None) -> torch.device:
    """The device ``spec`` names (default: CUDA when available), also made
    the process's current CUDA device: torch-harmonics 0.8.0's attention
    extension launches on the current CUDA stream without a device guard,
    so a tensor on ``cuda:N`` under another current device is an illegal
    memory access."""
    device = torch.device(spec) if spec else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and device.index is not None and torch.cuda.is_available():
        torch.cuda.set_device(device)  # a bare "cuda" already means the current device
    return device


def gm_step(hierarchy: TensorGridHierarchy, shape: tuple[int, int], scorer: SegmentationScorer) -> StepFn:
    """One panorama per call (``(C, nlat, nlon)``: image -> scalar field ->
    logits field -> the released loss on its ``(1, C, nlat, nlon)`` view),
    or ``B`` panoramas per call (``(B, C, nlat, nlon)``) as one graph: the
    fields concatenated along the nodes and the hierarchy tiled ``B`` times,
    a disjoint union in which every layer aggregates each panorama within
    its own copy, so the logits are those of ``B`` single calls to summation
    order and the loss is their mean. Drop path draws once per panorama
    under the union (:func:`stochastic_depth.copies`). The tiled hierarchies
    are built once per ``B`` and kept."""
    unions: dict[int, TensorGridHierarchy] = {1: hierarchy}

    def union(count: int) -> TensorGridHierarchy:
        if count not in unions:
            unions[count] = hierarchy.tile(count)
        return unions[count]

    def step(model: nn.Module, batch: tuple[Tensor, Tensor], device: torch.device) -> StepOutput:
        image, target = batch
        if image.ndim == 3:
            logits = logits_to_image(model(image_to_field(image), hierarchy), shape)
            return scorer(logits, target.unsqueeze(0))
        count = int(image.shape[0])
        with stochastic_depth.copies(count):
            field = model(images_to_field(image), union(count))
        return scorer(logits_to_images(field, shape), target)

    return step


def tensor_step(scorer: SegmentationScorer) -> StepFn:
    """The released model's batched step."""

    def step(model: nn.Module, batch: tuple[Tensor, Tensor], device: torch.device) -> StepOutput:
        images, targets = batch
        return scorer(model(images), targets)

    return step


def build_experiment(
    cfg: DictConfig,
    device: torch.device,
    *,
    shape: tuple[int, int],
    num_classes: int,
    in_channels: int = 3,
) -> SegmentationExperiment:
    """The arm ``cfg.model`` names, sized to the dataset.

    The GM arm's per-stage radii come from the grid chain
    (:func:`~experiments.stanford_2d3ds.geometry.segformer_radii`) and are
    injected here; a released model gets ``img_size``/``in_chans``/``out_chans``.
    """
    kind = str(cfg.model.get("kind", ""))
    grid = str(cfg.data.get("grid", "equiangular"))
    scorer = SegmentationScorer.build(
        shape, grid=grid, label_smoothing=float(cfg.training.get("label_smoothing", 0.0)), device=device
    )

    if kind in TENSOR_KINDS:
        container = OmegaConf.to_container(cfg.model.architecture, resolve=True)
        assert isinstance(container, dict)
        settings: dict[str, Any] = {str(key): value for key, value in container.items()}
        target = hydra.utils.get_class(str(settings.pop("_target_")))
        # Called directly, not through hydra.utils.instantiate: the released
        # filter-basis builder caches on its arguments and needs a hashable tuple.
        for key in ("kernel_shape", "attn_kernel_shape"):
            if key in settings:
                settings[key] = tuple(int(v) for v in settings[key])
        reference: nn.Module = target(
            img_size=tuple(shape), in_chans=in_channels, out_chans=num_classes, **settings
        ).to(device)
        return SegmentationExperiment(
            model=reference, step_fn=tensor_step(scorer), shape=shape, num_classes=num_classes
        )

    if kind != GM_KIND:
        raise ValueError(f"unknown model kind {kind!r}; expected {GM_KIND!r} or one of {TENSOR_KINDS}.")

    architecture = cfg.model.architecture
    num_stages = len(architecture.multiplicities)
    stage_shapes = segformer_stage_shapes(shape, int(cfg.model.scale_factor), num_stages)
    grid_internal = str(cfg.model.get("grid_internal", "equiangular"))
    icosphere_levels = cfg.model.get("icosphere_levels", None)
    levels = None if icosphere_levels is None else [int(v) for v in icosphere_levels]
    readout_rows = cfg.model.get("readout_rows", None)
    stem_rows = cfg.model.get("stem_rows", None)
    radii = segformer_radii(
        stage_shapes, grid_internal=grid_internal, icosphere_levels=levels,
        interpolate_spacings=float(cfg.model.get("interpolate_spacings", 1.0)),
        readout_rows=None if readout_rows is None else float(readout_rows),
        img_size=shape, grid=grid,
        stem_rows=None if stem_rows is None else float(stem_rows),
    )
    hierarchy = build_segformer_hierarchy(
        shape, stage_shapes, radii, grid=grid, grid_internal=grid_internal,
        icosphere_levels=levels, lateral=str(architecture.get("decoder", "fusion")) == "fusion",
        dtype=torch.float32, device=device,
    )
    for index, level in enumerate(hierarchy.levels):
        edges = "-" if level.discretization is None else str(level.discretization.n_edges)
        log.info("level %d: %s, %d nodes, coverage edges %s", index, level.describe(), level.n_nodes, edges)
    model: nn.Module = hydra.utils.instantiate(
        architecture,
        in_channels=in_channels,
        num_classes=num_classes,
        merge_cutoffs=list(radii.merge),
        attention_cutoffs=list(radii.attention),
        mix_cutoffs=list(radii.mix),
        interpolate_cutoffs=list(radii.interpolate),
        readout_cutoffs=None if radii.readout is None else list(radii.readout),
        stem_cutoff=radii.stem,
        _convert_="all",
    ).to(device)
    return SegmentationExperiment(
        model=model, step_fn=gm_step(hierarchy, shape, scorer), shape=shape,
        num_classes=num_classes, hierarchy=hierarchy, radii=radii, stage_shapes=stage_shapes,
    )


# ---------------------------------------------------------------- data


def open_splits(cfg: DictConfig) -> SegmentationSplits:
    """The dataset the config names, split and normalized the released way.

    ``data.dataset_file`` is relative to the directory the command was
    launched from, not to the run directory Hydra changes into;
    ``to_absolute_path`` resolves it the same way under ``hydra.main`` and
    from a plain ``compose``.
    """
    splits = load_splits(
        hydra.utils.to_absolute_path(str(cfg.data.dataset_file)),
        split_seed=int(cfg.data.split_seed),
        split_ratios=[float(r) for r in cfg.data.split_ratios],
        exclude_polar_fraction=float(cfg.data.get("exclude_polar_fraction", 0.0)),
    )
    expected = tuple(int(v) for v in cfg.data.img_size)
    if splits.shape != expected:
        raise ValueError(
            f"{splits.dataset_file} holds {splits.shape[0]}x{splits.shape[1]} panoramas but "
            f"data.img_size is {expected}; the model is built for the file's grid, so state "
            "the file's shape rather than another."
        )
    log.info(
        "dataset %s: %d samples, %d classes, splits %s, normalization %s",
        splits.dataset_file, len(splits.dataset), splits.num_classes, splits.sizes(),
        splits.normalization.state_dict(),
    )
    return splits


class _IndexStandIn(torch.utils.data.Dataset[int]):
    """Indices only: replays the shuffle stream on resume without reading data."""

    def __init__(self, length: int) -> None:
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> int:
        return index


def build_loaders(
    cfg: DictConfig,
    splits: SegmentationSplits,
    *,
    batched: bool,
    skip_epochs: int = 0,
) -> tuple[DataLoader[Any], DataLoader[Any], int]:
    """Train and validation loaders and the Trainer's accumulation steps.

    ``training.batch_size`` is the number of panoramas averaged into one
    optimizer step, as in the released script; a released model batches
    normally, and the GM arm realizes it as ``batch_size /
    panoramas_per_forward`` accumulated forwards of
    ``training.panoramas_per_forward`` panoramas each (default 1:
    single-sample loaders with ``accumulation_steps=batch_size``; more runs
    that many panoramas as one graph, see :func:`gm_step`). The shuffle
    order has its own seed and a resume fast-forwards it; it is the same
    order for any ``panoramas_per_forward``, since the loader draws it per
    panorama before grouping. Validation is one panorama per forward, so the
    logged per-sample metrics are the same quantities for every arm.
    """
    order = torch.Generator()
    order.manual_seed(int(cfg.data.loader_order_seed))
    batch_size = int(cfg.training.batch_size)
    per_forward = panoramas_per_forward(cfg)
    workers = int(cfg.data.get("num_workers", 0))
    train, val = splits.subset("train"), splits.subset("val")
    if skip_epochs > 0:
        stand_in = DataLoader(
            _IndexStandIn(len(train)), batch_size=batch_size if batched else None,
            shuffle=True, generator=order,
        )
        for _ in range(skip_epochs):
            for _ in stand_in:
                pass
    if batched:
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, generator=order, num_workers=workers)
        # One panorama per validation batch: the released metrics pool areas
        # over a batch, so a batched IoU is not the per-sample mean the GM arm logs.
        val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=workers)
        return train_loader, val_loader, 1
    train_loader = DataLoader(
        train, batch_size=None if per_forward == 1 else per_forward, shuffle=True, generator=order, num_workers=workers
    )
    val_loader = DataLoader(val, batch_size=None, shuffle=False, num_workers=workers)
    return train_loader, val_loader, batch_size // per_forward


def panoramas_per_forward(cfg: DictConfig) -> int:
    """``training.panoramas_per_forward`` (default 1), validated: at least 1
    and a divisor of ``training.batch_size``, so every optimizer step
    averages exactly ``batch_size`` panoramas."""
    per_forward = int(cfg.training.get("panoramas_per_forward", 1))
    batch_size = int(cfg.training.batch_size)
    if per_forward < 1:
        raise ValueError(f"training.panoramas_per_forward must be at least 1, got {per_forward}.")
    if batch_size % per_forward != 0:
        raise ValueError(
            f"training.panoramas_per_forward ({per_forward}) must divide training.batch_size ({batch_size})."
        )
    return per_forward


# ------------------------------------------------------------ protocol

#: The protocol keys that define the data; a run directory whose saved protocol differs on any of them is refused.
DATA_KEYS = (
    "task", "dataset_sha256", "num_samples", "num_classes",
    "class_labels", "split_seed", "split_ratios", "split_sizes", "split_indices", "normalization",
    "exclude_polar_fraction", "img_size", "grid", "loader_order_seed",
)  # dataset_file and dataset_size are recorded, not compared: the digest is the identity

#: The keys that define the model and its training; a resume that changes any of them is refused unless declared with ``training.allow_protocol_drift=true``.
MODEL_KEYS = (
    "model", "model_config", "model_seed", "parameter_count", "real_parameter_count",
    "stage_shapes", "grid_internal", "stage_levels", "radii", "cutoff_rule",
)
TRAINING_KEYS = ("training_config",)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def build_protocol(cfg: DictConfig, splits: SegmentationSplits, experiment: SegmentationExperiment) -> dict[str, Any]:
    """What this run's data and model are, as plain JSON values."""
    model = experiment.model
    protocol: dict[str, Any] = {
        "task": "stanford_2d3ds_segmentation",
        "dataset_file": str(splits.dataset_file),
        "dataset_size": int(splits.dataset_file.stat().st_size),
        "dataset_sha256": dataset_digest(splits.dataset_file),
        "num_samples": len(splits.dataset),
        "num_classes": int(splits.num_classes),
        "class_labels": list(splits.class_labels),
        "class_histogram": list(splits.class_histogram),
        "split_seed": int(cfg.data.split_seed),
        "split_ratios": [float(r) for r in cfg.data.split_ratios],
        "split_sizes": splits.sizes(),
        "split_indices": {name: list(index) for name, index in splits.indices.items()},
        "normalization": splits.normalization.state_dict(),
        "exclude_polar_fraction": float(cfg.data.get("exclude_polar_fraction", 0.0)),
        "img_size": [int(v) for v in splits.shape],
        "grid": str(cfg.data.get("grid", "equiangular")),
        "loader_order_seed": int(cfg.data.loader_order_seed),
        "model": str(cfg.model.get("kind", "?")),
        "model_config": OmegaConf.to_container(cfg.model, resolve=True),
        "training_config": _training_identity(cfg),
        "model_seed": int(cfg.seed),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "real_parameter_count": real_parameter_count(model),
        "stage_shapes": [list(s) for s in experiment.stage_shapes],
        "grid_internal": str(cfg.model.get("grid_internal", "equiangular")) if experiment.hierarchy is not None else None,
        "stage_levels": (
            [list(level.shape) for level in experiment.hierarchy.levels[1:]]
            if experiment.hierarchy is not None else None
        ),
    }
    protocol["radii"] = (
        None if experiment.radii is None else {
            "merge": list(experiment.radii.merge),
            "attention": list(experiment.radii.attention),
            "mix": list(experiment.radii.mix),
            "interpolate": list(experiment.radii.interpolate),
            "readout": None if experiment.radii.readout is None else list(experiment.radii.readout),
            "stem": experiment.radii.stem,
        }
    )
    protocol["cutoff_rule"] = cutoff_rule()  # which edges a boundary shell contributes; see edge_geometry
    protocol["git_revision"] = code_revision()
    protocol["versions"] = {
        "torch": torch.__version__, "torch_harmonics": _package_version("torch-harmonics"),
    }
    protocol["history"] = []
    return protocol


def _training_identity(cfg: DictConfig) -> dict[str, Any]:
    """The resolved training config minus the resume switches (``resume``,
    ``allow_protocol_drift`` and ``scheduler.restart``), which a continuation sets."""
    training = cast(dict[str, Any], OmegaConf.to_container(cfg.training, resolve=True))
    training.pop("resume", None)
    training.pop("allow_protocol_drift", None)
    scheduler = training.get("scheduler")
    if isinstance(scheduler, dict):
        scheduler = dict(scheduler)
        scheduler.pop("restart", None)
        training["scheduler"] = scheduler
    return training


def _same_value(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same_value(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same_value(x, y) for x, y in zip(a, b, strict=True))
    return bool(a == b)


def record_protocol_event(entry: dict[str, Any], directory: str | Path = ".") -> Path:
    """Append an event to the run's protocol history (``finished``, say)."""
    path = Path(directory) / PROTOCOL_FILENAME
    protocol = json.loads(path.read_text())
    protocol["history"] = list(protocol.get("history", [])) + [dict(entry, git_revision=code_revision())]
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    return path


def verify_or_save_protocol(
    protocol: dict[str, Any], directory: str | Path = ".", *, allow_drift: bool = False,
    event: str = "resume", completed_epochs: int | None = None,
) -> Path:
    """Write the protocol into a fresh run directory, or verify it against
    the one already there.

    A resume may not change the data (refused outright) nor the model or the
    training settings: those are refused too unless the change is declared
    with ``allow_drift`` (``training.allow_protocol_drift=true``), in which
    case the saved protocol is updated and the change, with the code
    revision it was made at, is appended to its ``history``. A resume at
    another code revision is recorded there as well.
    """
    path = Path(directory) / PROTOCOL_FILENAME
    if not path.exists():
        path.write_text(json.dumps(protocol, indent=2) + "\n")
        log.info("protocol -> %s", path)
        return path
    previous = json.loads(path.read_text())
    # Compare like with like: on disk, int dict keys are strings and tuples lists.
    protocol = json.loads(json.dumps(protocol))
    differing = [key for key in DATA_KEYS if not _same_value(previous.get(key), protocol[key])]
    if differing:
        raise ValueError(
            f"{path} describes different data than this config would produce "
            f"(differing: {', '.join(differing)}). Refusing to overwrite it: the "
            "checkpoints here were trained against the saved protocol. Point "
            "hydra.run.dir at a fresh directory, or restore the config that produced this one."
        )
    drift = {
        key: (previous.get(key), protocol[key])
        for key in MODEL_KEYS + TRAINING_KEYS
        if not _same_value(previous.get(key), protocol[key])
    }
    if drift and not allow_drift:
        raise ValueError(
            f"{path} describes a different model or training protocol than this config "
            f"(differing: {', '.join(drift)}). The checkpoints here were trained as the saved "
            "protocol says; restore that config, use a fresh hydra.run.dir, or declare the "
            "change with training.allow_protocol_drift=true to have it recorded."
        )
    revision_changed = previous.get("git_revision") != protocol["git_revision"]
    if drift or revision_changed:
        entry = {
            "event": event,
            "git_revision": protocol["git_revision"],
            "previous_git_revision": previous.get("git_revision"),
            "completed_epochs": completed_epochs,
            "changed": {key: {"previous": old, "current": new} for key, (old, new) in drift.items()},
        }
        updated = dict(previous)
        updated.update({key: protocol[key] for key in drift})
        # The protocol's revision is the latest code that touched the run;
        # the history keeps every earlier one.
        updated["git_revision"] = protocol["git_revision"]
        updated["history"] = list(previous.get("history", [])) + [entry]
        path.write_text(json.dumps(updated, indent=2) + "\n")
        log.warning(
            "protocol %s: resuming with %s; recorded in its history",
            path, ("changes to " + ", ".join(drift)) if drift else "the code at another revision",
        )
        return path
    log.info("protocol verified against %s", path)
    return path


# ---------------------------------------------------------------- main


def build_optimizer(cfg: DictConfig, model: nn.Module) -> torch.optim.Optimizer:
    """AdamW (the released script's) or Adam, on every parameter."""
    kind = str(cfg.training.get("optimizer", "adamw")).lower()
    lr, weight_decay = float(cfg.training.lr), float(cfg.training.weight_decay)
    if kind == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if kind == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"training.optimizer must be 'adamw' or 'adam', got {kind!r}.")


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


@hydra.main(version_base=None, config_path="../../configs", config_name="segmentation")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.seed))
    device = resolve_device(cfg.device)
    log.info("segmentation: model=%s device=%s", cfg.model.get("kind", "?"), device)

    splits = open_splits(cfg)
    experiment = build_experiment(
        cfg, device, shape=splits.shape, num_classes=splits.num_classes, in_channels=splits.in_channels
    )
    log.info("model parameters: %d", real_parameter_count(experiment.model))
    # Every refusal that can stop this launch precedes the protocol write, so
    # the file never describes training that did not happen.
    resume_from, completed_epochs = _resume_point(cfg)
    if resume_from is None and (Path("checkpoints/last.pt").is_file() or Path("logs/metrics.jsonl").is_file()):
        raise FileExistsError(
            f"{Path.cwd()} already holds a run (checkpoints/last.pt or logs/metrics.jsonl) and "
            "training.resume is false: a fresh run here would overwrite its checkpoints and append "
            "to its logs. Pass training.resume=true to continue it, or use a fresh hydra.run.dir."
        )
    verify_or_save_protocol(
        build_protocol(cfg, splits, experiment),
        allow_drift=bool(cfg.training.get("allow_protocol_drift", False)),
        event="resume" if resume_from is not None else "relaunch",
        completed_epochs=completed_epochs,
    )

    train_loader, val_loader, accumulation_steps = build_loaders(
        cfg, splits, batched=experiment.batched, skip_epochs=completed_epochs
    )

    optimizer = build_optimizer(cfg, experiment.model)
    spec = ScheduleSpec.from_mapping(cfg.training.scheduler)
    if spec.restart and resume_from is None:
        raise ValueError(
            "training.scheduler.restart=true requires training.resume=true: a "
            "fresh run's cosine already starts at training.lr."
        )
    scheduler, scheduler_metric = build_scheduler(
        optimizer, spec, max_epochs=int(cfg.training.max_epochs),
        monitor=MONITOR_METRIC, completed_epochs=completed_epochs,
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
            "checkpoints", monitor=MONITOR_METRIC, mode="min",
            keep_periodic_every=cfg.training.checkpoint_every,
        ),
        early_stopping_patience=cfg.training.early_stopping_patience,
        validate_every=int(cfg.training.validate_every),
    )
    trainer.fit(
        train_loader,
        max_epochs=int(cfg.training.max_epochs),
        val_loader=val_loader,
        resume_from=resume_from,
        restore_scheduler=not spec.restart,
    )
    # The loop ended, by its last epoch or by early stopping: recorded for
    # the evaluation.
    last = torch.load(Path("checkpoints") / "last.pt", map_location="cpu", weights_only=False)
    record_protocol_event({"event": "finished", "epochs": int(last["epoch"]) + 1, "max_epochs": int(cfg.training.max_epochs)})
    log.info("finished after %d epochs", int(last["epoch"]) + 1)


if __name__ == "__main__":
    main()
