"""Seeding and RNG-state capture for reproducible, resumable training runs.

``set_seed`` seeds ``random``, ``numpy`` and ``torch`` (CPU and CUDA);
``get_rng_state``/``set_rng_state`` capture and restore the same RNGs as one
picklable object, so a checkpoint resumes a run bit-for-bit. Data seeding is
separate: :func:`derive_data_seed` maps ``(base_seed, split, index, epoch)``
to one stable seed and :func:`forked_torch_seed` runs a block under it
without disturbing the surrounding RNG state, so ``dataset[i]`` names one
fixed sample regardless of how many random numbers model initialization or
dropout consumed. The model seed and the data seeds (``train_seed``,
``val_seed``, ``test_seed``, ``normalization_seed``, ``loader_order_seed``)
are independent configuration values.
"""

from __future__ import annotations

import contextlib
import hashlib
import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

# Inside int64's positive half, so a derived seed round-trips through
# JSON/YAML run metadata without a signedness surprise.
_SEED_MODULUS = 2**63


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed ``random``, ``numpy``, and ``torch`` (CPU and every CUDA device).

    Parameters
    ----------
    seed
        Seed value, shared across every RNG.
    deterministic
        If ``True``, additionally request bitwise-reproducible CUDA kernels
        via ``torch.use_deterministic_algorithms`` and disable cuDNN's
        autotuner. If ``False`` (the default), the autotuner is enabled and
        picks algorithms by timing, which is not bit-reproducible across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # a no-op when CUDA is unavailable

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def derive_data_seed(base_seed: int, split: str, index: int, epoch: int = 0) -> int:
    """One stable seed identifying a single dataset sample.

    ``(base_seed, split, index, epoch)`` determines the returned value
    completely and identically in every process, so ``dataset[i]`` resolves
    to the same sample across runs, machines and architectures. BLAKE2b over
    a UTF-8 serialization rather than Python's ``hash``, which is
    randomized per process. ``split`` is a literal name (``"train"``,
    ``"val"``, ``"test"``, ``"normalization"``) rather than a numeric
    offset, so two splits cannot collide by arithmetic.

    Parameters
    ----------
    base_seed
        The split's own configured data seed.
    split
        Literal split name, mixed into the hash as itself.
    index
        Sample index within the split.
    epoch
        Epoch number, for an epoch-indexed sampling policy. Left at ``0``
        by a fixed-index policy, where a given index must resolve to the
        same sample in every epoch.

    Returns
    -------
    int
        A seed in ``[0, 2**63)``.
    """
    payload = f"{int(base_seed)}|{split}|{int(index)}|{int(epoch)}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % _SEED_MODULUS


@contextlib.contextmanager
def forked_torch_seed(seed: int, device: torch.device | str | None = None) -> Iterator[None]:
    """Run a block under ``seed`` without disturbing the surrounding RNG state.

    Wraps ``torch.random.fork_rng``: the CPU (and, for a CUDA ``device``,
    that device's) RNG state is saved on entry, reseeded to ``seed``, and
    restored on exit, so data generation that draws from the global RNG
    (``ShallowWaterSolver.random_initial_condition`` accepts no
    ``torch.Generator``) is a pure function of its sample seed.

    Parameters
    ----------
    seed
        Seed applied inside the block, typically from
        :func:`derive_data_seed`.
    device
        Device whose RNG to additionally fork and reseed. Only CUDA devices
        carry a separate RNG worth forking; anything else forks CPU alone.
    """
    resolved = torch.device(device) if device is not None else None
    devices = (
        [resolved.index if resolved.index is not None else torch.cuda.current_device()]
        if resolved is not None and resolved.type == "cuda"
        else []
    )

    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed(seed)
        yield


@dataclass(frozen=True, slots=True)
class RngState:
    """A snapshot of every RNG :func:`set_seed` touches."""

    python_state: Any
    numpy_state: dict[str, Any]
    torch_state: torch.Tensor
    cuda_state: list[torch.Tensor] | None


def get_rng_state() -> RngState:
    """Capture the current state of every RNG :func:`set_seed` touches."""
    return RngState(
        python_state=random.getstate(),
        numpy_state=np.random.get_state(legacy=False),
        torch_state=torch.get_rng_state(),
        cuda_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def set_rng_state(state: RngState) -> None:
    """Restore an :class:`RngState` captured by :func:`get_rng_state`.

    The generator states are coerced to CPU byte tensors:
    ``torch.load(..., map_location="cuda")`` moves every tensor of a
    checkpoint, and ``torch.set_rng_state`` accepts only a CPU ``ByteTensor``.
    """
    random.setstate(state.python_state)
    np.random.set_state(state.numpy_state)
    torch.set_rng_state(state.torch_state.to(device="cpu", dtype=torch.uint8))

    if state.cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [tensor.to(device="cpu", dtype=torch.uint8) for tensor in state.cuda_state]
        )
