"""Tests for gm_attention.training.reproducibility."""

from __future__ import annotations

import random
import subprocess
import sys

import numpy as np
import torch

from gm_attention.training.reproducibility import (
    derive_data_seed,
    forked_torch_seed,
    get_rng_state,
    set_rng_state,
    set_seed,
)


def test_set_seed_makes_torch_draws_reproducible() -> None:
    set_seed(0)
    first = torch.randn(5)
    set_seed(0)
    second = torch.randn(5)

    assert torch.equal(first, second)


def test_set_seed_makes_numpy_and_random_draws_reproducible() -> None:
    set_seed(1)
    first = (np.random.rand(3), random.random())
    set_seed(1)
    second = (np.random.rand(3), random.random())

    assert np.array_equal(first[0], second[0])
    assert first[1] == second[1]


def test_set_seed_deterministic_true_sets_cudnn_flags() -> None:
    set_seed(0, deterministic=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_set_seed_deterministic_false_enables_cudnn_benchmark() -> None:
    set_seed(0, deterministic=False)
    assert torch.backends.cudnn.benchmark is True


# ==========================================================================
# derive_data_seed
# ==========================================================================


def test_derive_data_seed_is_deterministic_within_a_process() -> None:
    assert derive_data_seed(1000, "train", 7) == derive_data_seed(1000, "train", 7)


def test_derive_data_seed_is_stable_across_processes_with_different_hash_seeds() -> None:
    """The reason this uses BLAKE2b and not Python's built-in ``hash``.

    ``hash`` on a str is randomized per process by PYTHONHASHSEED, so a
    ``hash``-derived sample seed would differ between two runs of the same
    config. Two subprocesses with deliberately different PYTHONHASHSEED
    values must still agree.
    """
    program = (
        "from gm_attention.training.reproducibility import derive_data_seed;"
        "print(derive_data_seed(1000, 'val', 3, 2))"
    )
    outputs = [
        subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": hash_seed, "PATH": ""},
        ).stdout.strip()
        for hash_seed in ("0", "12345")
    ]

    assert outputs[0] == outputs[1]


def test_derive_data_seed_changes_when_any_single_component_changes() -> None:
    reference = derive_data_seed(1000, "train", 7, 2)

    assert derive_data_seed(1001, "train", 7, 2) != reference  # base_seed
    assert derive_data_seed(1000, "val", 7, 2) != reference  # split
    assert derive_data_seed(1000, "train", 8, 2) != reference  # index
    assert derive_data_seed(1000, "train", 7, 3) != reference  # epoch


def test_derive_data_seed_defaults_epoch_to_zero() -> None:
    assert derive_data_seed(1000, "train", 7) == derive_data_seed(1000, "train", 7, 0)


def test_derive_data_seed_lands_in_the_documented_range() -> None:
    seeds = [derive_data_seed(0, split, index) for split in ("train", "val") for index in range(64)]

    assert all(0 <= seed < 2**63 for seed in seeds)


def test_derive_data_seed_does_not_collide_across_a_realistic_index_range() -> None:
    seeds = {
        derive_data_seed(base, split, index)
        for base in (1000, 2000, 3000)
        for split in ("train", "val", "test", "normalization")
        for index in range(256)
    }

    assert len(seeds) == 3 * 4 * 256


# ==========================================================================
# forked_torch_seed
# ==========================================================================


def test_forked_torch_seed_makes_the_block_reproducible() -> None:
    with forked_torch_seed(1234):
        first = torch.randn(5)
    with forked_torch_seed(1234):
        second = torch.randn(5)

    assert torch.equal(first, second)


def test_forked_torch_seed_leaves_the_surrounding_cpu_rng_untouched() -> None:
    torch.manual_seed(7)
    expected = torch.randn(4)

    torch.manual_seed(7)
    with forked_torch_seed(999):
        torch.randn(100)  # consume a lot of the *forked* stream
    actual = torch.randn(4)

    assert torch.equal(actual, expected)


def test_forked_torch_seed_block_is_independent_of_the_surrounding_state() -> None:
    torch.manual_seed(1)
    with forked_torch_seed(555):
        from_one = torch.randn(4)

    torch.manual_seed(2)
    torch.randn(37)  # arbitrary unrelated draws, as model init/dropout would do
    with forked_torch_seed(555):
        from_two = torch.randn(4)

    assert torch.equal(from_one, from_two)


def test_forked_torch_seed_accepts_a_cpu_device_without_forking_cuda() -> None:
    torch.manual_seed(3)
    expected = torch.randn(4)

    torch.manual_seed(3)
    with forked_torch_seed(42, device=torch.device("cpu")):
        torch.randn(10)
    actual = torch.randn(4)

    assert torch.equal(actual, expected)


def test_rng_state_round_trip_reproduces_subsequent_draws() -> None:
    set_seed(2)
    torch.randn(10)  # advance every RNG away from its freshly-seeded state
    np.random.rand(10)
    random.random()

    state = get_rng_state()
    expected = (torch.randn(5), np.random.rand(5), random.random())

    # Advance further, so the RNGs are no longer where `state` was captured.
    torch.randn(3)
    np.random.rand(3)
    random.random()

    set_rng_state(state)
    actual = (torch.randn(5), np.random.rand(5), random.random())

    assert torch.equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])
    assert actual[2] == expected[2]

def test_rng_state_restores_from_a_relocated_tensor() -> None:
    """torch.load(..., map_location="cuda") moves every tensor in a
    checkpoint, the generator states among them, and torch.set_rng_state
    accepts only a CPU ByteTensor. A checkpoint resumed onto a GPU
    therefore failed with "RNG state must be a torch.ByteTensor", which
    names neither the cause nor the fix.

    The relocation is simulated here by changing dtype rather than device,
    since the check must run without CUDA -- it exercises the same coercion.
    """
    import dataclasses

    original = get_rng_state()
    relocated = dataclasses.replace(
        original, torch_state=original.torch_state.to(torch.int64)
    )

    set_rng_state(relocated)  # must not raise

    assert torch.equal(torch.get_rng_state(), original.torch_state)


def test_restoring_reproduces_the_same_draws() -> None:
    """The property the coercion must not quietly break."""
    state = get_rng_state()
    first = torch.randn(4)

    set_rng_state(state)

    assert torch.equal(torch.randn(4), first)
