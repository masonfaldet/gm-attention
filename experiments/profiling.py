"""Timing a model's evaluation paths against each other: native PyTorch, fused kernels, stencil, and their mixes.

A layer of this library can run through three implementations of the same
function: the native graph path (eager PyTorch over the edge set), the fused
kernels switched on by ``GM_FUSED_PAIRS`` / ``GM_FACTORED_KERNEL`` +
``GM_FUSED_FACTORED`` / ``GM_FUSED_ATTENTION``, and the stencil path over a
:class:`~gm_attention.nn.stencil_plan.StencilPlans` on a Gauss-Legendre grid
in the coordinate gauge. The fused switches act only on graph-path operators,
so a model whose same-grid operators run on the stencil can still run its
bipartite transfers (the SegFormer's patch merges) fused: the combinations
here are the ones that have an effect.

:func:`profile_paths` runs every :class:`PathConfig` of a model under the
switches it names (set on the modules for the duration of the run, as the
tests set them, since the environment variables are read at import), times
the forward and the forward plus backward over the parameters, checks every
path's output against the native one, and prints a table.
``scripts/profile_segformer_paths.py`` and
``scripts/profile_single_level_paths.py`` drive it for the two models.
"""

from __future__ import annotations

import contextlib
import gc
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from gm_attention.nn import fused_attention, fused_factored, fused_pairs, kernels
from gm_attention.representations.field_type import SO2FeatureField

FusedKind = Literal["none", "pairs", "factored"]


@dataclass(frozen=True)
class PathConfig:
    """One way of evaluating a model.

    ``stencil`` hands the model its stencil plans (same-grid operators on the
    dense fixed-grid layout); ``fused`` picks the steerable kernels' fused
    path on the operators that stay on the graph: ``"pairs"`` is the per-edge
    pairs kernel (``GM_FUSED_PAIRS``), ``"factored"`` the factored
    aggregation (``GM_FACTORED_KERNEL`` with ``GM_FUSED_FACTORED``);
    ``fused_attention`` switches ``GM_FUSED_ATTENTION`` (the scores-and-
    softmax op and the fused depthwise aggregation) for graph-path attention
    and depthwise convolutions.
    """

    name: str
    stencil: bool = False
    fused: FusedKind = "none"
    fused_attention: bool = False

    @property
    def description(self) -> str:
        parts = ["stencil plans" if self.stencil else "graph path"]
        if self.fused != "none":
            parts.append(f"fused {self.fused}")
        if self.fused_attention:
            parts.append("fused attention")
        return ", ".join(parts)


#: The native graph path and the two fused flavours, then the stencil path: the paths of a model whose operators all share one grid.
SINGLE_GRID_PATHS: tuple[PathConfig, ...] = (
    PathConfig("native"),
    PathConfig("fused-pairs", fused="pairs", fused_attention=True),
    PathConfig("fused-factored", fused="factored", fused_attention=True),
    PathConfig("stencil", stencil=True),
)

#: The same, plus the stencil path with each fused flavour on the graph-path transfers between grids.
MULTI_GRID_PATHS: tuple[PathConfig, ...] = (
    *SINGLE_GRID_PATHS,
    PathConfig("stencil+fused-pairs", stencil=True, fused="pairs"),
    PathConfig("stencil+fused-factored", stencil=True, fused="factored"),
)


#: The functions the layers call to enter a fused kernel: ``(module, name)``; counted while a path runs.
FUSED_ENTRY_POINTS: tuple[tuple[object, str], ...] = (
    (fused_pairs, "forward_general"),
    (fused_factored, "forward"),
    (fused_factored, "forward_aggregated"),
    (fused_attention, "softmax"),
    (fused_attention, "softmax_head"),
    (fused_attention, "forward_depthwise"),
)


@contextlib.contextmanager
def switched(config: PathConfig, device: torch.device) -> Iterator[dict[str, int]]:
    """The library's fused switches set for ``config`` on ``device``, restored on exit.

    The modes are module constants read from the environment at import;
    setting them on the modules is how the tests select a path per test. A
    mode of ``"cpu"`` enables a fused kernel on either device, ``"1"`` on
    CUDA only. Yields a counter of the calls that entered each fused kernel
    (:data:`FUSED_ENTRY_POINTS`): a fused kernel also declines a call whose
    widths or dtype it does not support, in which case the layer runs its
    native code and the count stays zero.
    """
    mode = "cpu" if device.type == "cpu" else "1"
    saved = (fused_pairs.MODE, fused_factored.MODE, fused_attention.MODE, kernels.FACTORED)
    originals = [(module, name, getattr(module, name)) for module, name in FUSED_ENTRY_POINTS]
    counts: dict[str, int] = {f"{module.__name__.rsplit('.', 1)[-1]}.{name}": 0 for module, name in FUSED_ENTRY_POINTS}  # type: ignore[attr-defined]

    def counting(module: object, name: str, function: Callable[..., object]) -> Callable[..., object]:
        key = f"{module.__name__.rsplit('.', 1)[-1]}.{name}"  # type: ignore[attr-defined]

        def wrapped(*args: object, **kwargs: object) -> object:
            counts[key] += 1
            return function(*args, **kwargs)

        return wrapped

    try:
        fused_pairs.MODE = mode if config.fused == "pairs" else "0"
        fused_factored.MODE = mode if config.fused == "factored" else "0"
        kernels.FACTORED = config.fused == "factored"
        fused_attention.MODE = mode if config.fused_attention else "0"
        for module, name, function in originals:
            setattr(module, name, counting(module, name, function))
        yield counts
    finally:
        fused_pairs.MODE, fused_factored.MODE, fused_attention.MODE, kernels.FACTORED = saved
        for module, name, function in originals:
            setattr(module, name, function)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(fn: Callable[[], object], repeats: int, warmup: int, device: torch.device) -> tuple[float, float]:
    """``(min, mean)`` wall time in seconds of ``fn`` over ``repeats`` calls after ``warmup`` calls."""
    for _ in range(warmup):
        fn()
    _synchronize(device)
    times: list[float] = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        fn()
        _synchronize(device)
        times.append(time.perf_counter() - start)
    return min(times), sum(times) / len(times)


def _flat(output: object) -> Tensor:
    return output.to_flat() if isinstance(output, SO2FeatureField) else output  # type: ignore[return-value]


@dataclass
class PathTiming:
    """The measurements of one path; ``error`` set when the path could not run."""

    config: PathConfig
    forward_min: float = float("nan")
    forward_mean: float = float("nan")
    train_min: float = float("nan")  # forward plus backward over the parameters
    train_mean: float = float("nan")
    max_difference: float = float("nan")  # of the forward output against the first path's
    peak_memory: int | None = None  # bytes, CUDA only
    fused_calls: dict[str, int] | None = None  # calls that entered each fused kernel during one forward + backward
    error: str | None = None


def profile_paths(
    paths: Sequence[PathConfig],
    build_run: Callable[[PathConfig], Callable[[], object]],
    parameters: Sequence[Tensor],
    *,
    device: torch.device,
    repeats: int = 3,
    warmup: int = 1,
    profile_operators: int = 0,
) -> list[PathTiming]:
    """Time every path's forward and forward-plus-backward.

    Parameters
    ----------
    paths
        The configurations, the first being the reference every output is
        compared against (normally ``"native"``).
    build_run
        Returns, for a configuration, a zero-argument callable evaluating
        the model with the geometry that configuration needs (plans or
        discretization); the fused switches are set around every call.
    parameters
        The parameters the backward differentiates with respect to.
    device, repeats, warmup
        Where and how often; times are wall-clock minima and means.
    profile_operators
        If positive, also print this many top operators by self time of one
        forward-plus-backward per path (``torch.profiler``).
    """
    reference: Tensor | None = None
    results: list[PathTiming] = []
    for config in paths:
        timing = PathTiming(config)
        try:
            with switched(config, device) as counts:
                run = build_run(config)

                def train_step() -> None:
                    loss = _flat(run()).square().sum()
                    torch.autograd.grad(loss, parameters, allow_unused=True)

                with torch.no_grad():
                    output = _flat(run()).detach().clone()
                # The fused calls of one forward + backward, before the timed repetitions.
                for key in counts:
                    counts[key] = 0
                train_step()
                timing.fused_calls = {key: value for key, value in counts.items() if value}
                with torch.no_grad():
                    timing.forward_min, timing.forward_mean = _timed(run, repeats, warmup, device)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                timing.train_min, timing.train_mean = _timed(train_step, repeats, warmup, device)
                if device.type == "cuda":
                    timing.peak_memory = torch.cuda.max_memory_allocated(device)
                if profile_operators > 0:
                    activities = [torch.profiler.ProfilerActivity.CPU]
                    if device.type == "cuda":
                        activities.append(torch.profiler.ProfilerActivity.CUDA)
                    with torch.profiler.profile(activities=activities) as profile:
                        train_step()
                    sort_by = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
                    print(f"\n--- {config.name}: top operators of one forward + backward ({config.description})")
                    print(profile.key_averages().table(sort_by=sort_by, row_limit=profile_operators))
            if reference is None:
                reference = output
                timing.max_difference = 0.0
            else:
                timing.max_difference = float((output - reference).abs().max())
        except Exception as error:  # a path unavailable here (no compiler, no CUDA, a build failure)
            timing.error = f"{type(error).__name__}: {str(error).splitlines()[0][:120]}"
        results.append(timing)
    return results


def format_table(results: Sequence[PathTiming], device: torch.device) -> str:
    """The results as a fixed-width table, milliseconds; a path that failed shows its error."""
    header = f"{'path':<24s} {'forward min':>12s} {'mean':>9s} {'fwd+bwd min':>12s} {'mean':>9s} {'max|diff| vs native':>20s}"
    if device.type == "cuda":
        header += f" {'peak memory':>12s}"
    header += "  fused kernel calls (one fwd+bwd)"
    lines = [header, "-" * len(header)]
    for timing in results:
        if timing.error is not None:
            lines.append(f"{timing.config.name:<24s} unavailable: {timing.error}")
            continue
        line = (
            f"{timing.config.name:<24s} {timing.forward_min * 1e3:>9.1f} ms {timing.forward_mean * 1e3:>6.1f} ms "
            f"{timing.train_min * 1e3:>9.1f} ms {timing.train_mean * 1e3:>6.1f} ms {timing.max_difference:>20.3e}"
        )
        if device.type == "cuda" and timing.peak_memory is not None:
            line += f" {timing.peak_memory / 2**20:>9.0f} MiB"
        calls = timing.fused_calls or {}
        line += "  " + (", ".join(f"{key} x{value}" for key, value in calls.items()) if calls else "none")
        lines.append(line)
    return "\n".join(lines)


def select_paths(available: Sequence[PathConfig], names: Sequence[str] | None) -> list[PathConfig]:
    """The configurations named in ``names`` in that order, or all of them."""
    if not names:
        return list(available)
    by_name = {config.name: config for config in available}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise SystemExit(f"unknown path(s) {unknown}; choose from {list(by_name)}.")
    return [by_name[name] for name in names]


def resolve_device(name: str) -> torch.device:
    """``"auto"`` is CUDA when available, else the CPU."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
