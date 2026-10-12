"""Learning-rate schedules shared by the experiments, and the resume guard.

``plateau`` is ``ReduceLROnPlateau`` on the monitored validation metric,
scaling the rate by ``factor`` after ``patience`` epochs without
improvement. ``cosine`` is ``CosineAnnealingLR`` from the optimizer's rate
to ``eta_min`` over exactly ``max_epochs`` epochs, stepped once per epoch.
The horizon is part of the cosine schedule, so :func:`check_resumed_scheduler`
refuses to extend a resumed cosine run; a warm restart
(:attr:`ScheduleSpec.restart`) starts a new cycle over the added epochs with
the optimizer's moments kept, accepted only where the checkpoint's cycle has
finished. The experiments read their ``training.scheduler`` block into a
:class:`ScheduleSpec`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from gm_attention.training.engine import Scheduler


@dataclass(frozen=True)
class ScheduleSpec:
    """A config's ``training.scheduler`` block as plain values.

    ``factor``/``patience`` apply to ``plateau`` only, ``eta_min`` and
    ``restart`` to ``cosine`` only; a restart on any other kind is refused
    at construction rather than ignored.
    """

    kind: str = "plateau"
    factor: float = 0.5
    patience: int = 5
    eta_min: float = 0.0
    restart: bool = False

    def __post_init__(self) -> None:
        if self.kind not in ("plateau", "cosine"):
            raise ValueError(
                f"training.scheduler.kind must be 'plateau' or 'cosine', got {self.kind!r}."
            )
        if self.restart and self.kind != "cosine":
            raise ValueError("training.scheduler.restart applies to the cosine schedule only.")

    @classmethod
    def from_mapping(cls, scheduler: Mapping[str, Any]) -> ScheduleSpec:
        """Read a ``training.scheduler`` block (a dict or an OmegaConf node)."""
        return cls(
            kind=str(scheduler.get("kind", "plateau")),
            factor=float(scheduler.get("factor", 0.5)),
            patience=int(scheduler.get("patience", 5)),
            eta_min=float(scheduler.get("eta_min", 0.0)),
            restart=bool(scheduler.get("restart", False)),
        )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    spec: ScheduleSpec,
    *,
    max_epochs: int,
    monitor: str,
    completed_epochs: int = 0,
) -> tuple[Scheduler, str | None]:
    """The schedule ``spec`` names, and the metric it steps on (``None``
    for the clock-driven cosine).

    ``monitor`` is the validation metric the plateau rule watches.
    ``completed_epochs`` matters only for a cosine warm restart, whose new
    cycle spans ``max_epochs - completed_epochs``.
    """
    if spec.kind == "plateau":
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=spec.factor, patience=spec.patience
        )
        return plateau, monitor

    if spec.restart and completed_epochs < 1:
        raise ValueError(
            "training.scheduler.restart=true needs a resumed run: a fresh run's "
            "cosine already starts at training.lr."
        )
    horizon = max_epochs - completed_epochs if spec.restart else max_epochs
    if horizon < 1:
        raise ValueError(
            f"training.max_epochs={max_epochs} leaves no epochs for a new cosine "
            f"cycle after {completed_epochs} completed."
        )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=horizon, eta_min=spec.eta_min
    )
    return cosine, None


def scheduler_kind_of(state: Mapping[str, Any] | None) -> str | None:
    """Which schedule a checkpointed scheduler ``state_dict`` came from,
    read off its keys; ``None`` when there is no state or it is neither."""
    if state is None:
        return None
    if "num_bad_epochs" in state:
        return "plateau"
    if "T_max" in state:
        return "cosine"
    return None


def check_resumed_scheduler(
    state: Mapping[str, Any] | None,
    spec: ScheduleSpec,
    *,
    max_epochs: int,
    completed_epochs: int = 0,
) -> None:
    """Refuse a resume whose schedule would diverge from the checkpoint's.

    ``scheduler.load_state_dict`` only updates ``__dict__``, so a plateau
    state loaded into a cosine scheduler (or the reverse) runs a hybrid no
    config describes; and a cosine ``T_max`` is restored from the
    checkpoint, so raising ``max_epochs`` would not lengthen the decay
    (past ``T_max`` the cosine climbs back up). A cosine state records
    ``last_epoch`` steps into ``T_max``, so its cycle ends at epoch
    ``completed_epochs - last_epoch + T_max``: a plain resume must keep
    ``max_epochs`` there, and a warm restart is accepted only once that
    cycle has finished. One kind change is allowed, a cosine restart from
    a plateau checkpoint: the restart discards the scheduler state and
    runs a fresh cosine cycle from ``training.lr`` over the added epochs.
    """
    found = scheduler_kind_of(state)
    if found == "plateau" and spec.kind == "cosine" and spec.restart:
        if max_epochs <= completed_epochs:
            raise ValueError(
                f"training.max_epochs={max_epochs} adds no epochs after the "
                f"{completed_epochs} completed; annealing a plateau run needs a longer horizon."
            )
        return
    if found is not None and found != spec.kind:
        raise ValueError(
            f"checkpoints/last.pt was written under a {found!r} schedule but "
            f"training.scheduler.kind={spec.kind!r}: a resumed run keeps the "
            "schedule it started with. Resume with the original kind, start a "
            "fresh run, or anneal a plateau run with training.scheduler.kind=cosine "
            "training.scheduler.restart=true."
        )
    if found != "cosine" or state is None:
        return

    t_max, last_epoch = int(state["T_max"]), int(state["last_epoch"])
    if spec.restart:
        if last_epoch < t_max:
            raise ValueError(
                f"checkpoints/last.pt is {last_epoch} epochs into a {t_max}-epoch "
                "cosine cycle: a warm restart begins where a cycle ends. Drop "
                "training.scheduler.restart to continue this cycle."
            )
        if max_epochs <= completed_epochs:
            raise ValueError(
                f"training.max_epochs={max_epochs} adds no epochs after the "
                f"{completed_epochs} completed; a warm restart needs a longer horizon."
            )
        return

    cycle_end = completed_epochs - last_epoch + t_max
    if cycle_end != max_epochs:
        raise ValueError(
            f"checkpoints/last.pt holds a cosine cycle ending at epoch {cycle_end} "
            f"but training.max_epochs={max_epochs}: a cosine run cannot be extended, "
            f"its horizon is the schedule. Resume with training.max_epochs={cycle_end}, "
            "or pass training.scheduler.restart=true to start a new cycle from "
            "training.lr over the added epochs."
        )
