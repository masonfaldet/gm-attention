"""Tests for gm_attention.data.shallow_water.

The centerpiece is the cross-check the module's own docstring promises:
evaluate ArbitraryPointSHT *at an equiangular grid's own points* and confirm
it agrees with torch_harmonics' actual InverseRealSHT (the ground truth this
module is a generalization of) to numerical precision. Passing this is what
justifies trusting the same code away from that grid, e.g. at icosphere
nodes, where there is no reference implementation to compare against
directly.

The dataset half covers what ``ShallowWaterDataset`` ships here: the solver,
the seeding that makes an index name one fixed trajectory, and
``spectral_pair``. Synthesis into a model's state representation (and the
normalization of it) lives in ``experiments/shallow_water_paper/data.py``
and is tested there.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch_harmonics import InverseRealSHT
from torch_harmonics.examples.shallow_water_equations import ShallowWaterSolver
from torch_harmonics.quadrature import _precompute_longitudes, clenshaw_curtiss_weights

from gm_attention.data.shallow_water import (
    ArbitraryPointSHT,
    ShallowWaterDataset,
    coriolis_channel,
    spherical_coordinates,
)
from gm_attention.discretization.grids.icosphere import build_icosphere_grid

NLAT, NLON = 12, 24


def _equiangular_nodes() -> torch.Tensor:
    """(nlat * nlon, 3) unit-sphere points, built with torch_harmonics' own
    grid convention exactly (the same clenshaw_curtiss_weights + flip that
    RealSHT/InverseRealSHT use internally), so this test's grid is
    bit-for-bit the grid the reference transforms already assume -- not a
    parallel reimplementation that could itself introduce a discrepancy."""
    cost, _ = clenshaw_curtiss_weights(NLAT, -1, 1)
    colatitude = torch.flip(torch.arccos(cost), dims=(0,))
    longitude = _precompute_longitudes(NLON)

    theta = colatitude.repeat_interleave(NLON)
    phi = longitude.repeat(NLAT)

    sin_theta = torch.sin(theta)
    nodes = torch.stack(
        [sin_theta * torch.cos(phi), sin_theta * torch.sin(phi), torch.cos(theta)], dim=-1
    )
    return nodes.double()


# ==========================================================================
# spherical_coordinates
# ==========================================================================


def test_spherical_coordinates_round_trips_the_grid_construction() -> None:
    nodes = _equiangular_nodes()
    theta, phi = spherical_coordinates(nodes)

    reconstructed = torch.stack(
        [torch.sin(theta) * torch.cos(phi), torch.sin(theta) * torch.sin(phi), torch.cos(theta)],
        dim=-1,
    )
    assert torch.allclose(reconstructed, nodes, atol=1e-10)


def test_spherical_coordinates_ranges() -> None:
    torch.manual_seed(0)
    nodes = torch.randn(50, 3, dtype=torch.float64)
    nodes = nodes / torch.linalg.vector_norm(nodes, dim=-1, keepdim=True)

    theta, phi = spherical_coordinates(nodes)

    assert bool((theta >= 0.0).all() and (theta <= math.pi).all())
    assert bool((phi >= 0.0).all() and (phi < 2 * math.pi).all())


# ==========================================================================
# ArbitraryPointSHT vs. InverseRealSHT, evaluated at the same grid
# ==========================================================================


@pytest.mark.parametrize("lmax,mmax", [(NLAT, NLON // 2 + 1), (7, 5)])
def test_scalar_synthesis_matches_inverse_real_sht_on_its_own_grid(lmax: int, mmax: int) -> None:
    torch.manual_seed(1)
    nodes = _equiangular_nodes()

    reference = InverseRealSHT(NLAT, NLON, lmax=lmax, mmax=mmax, grid="equiangular", csphase=False).double()
    ours = ArbitraryPointSHT(nodes, lmax=lmax, mmax=mmax, nlon=NLON)

    coefficients = torch.randn(2, lmax, mmax, dtype=torch.complex128)

    expected = reference(coefficients).reshape(2, NLAT * NLON)
    actual = ours(coefficients)

    assert torch.allclose(actual, expected, atol=1e-8, rtol=1e-6)


def test_scalar_synthesis_gradients_are_finite() -> None:
    nodes = _equiangular_nodes()
    ours = ArbitraryPointSHT(nodes, lmax=7, mmax=5, nlon=NLON)

    coefficients = torch.randn(7, 5, dtype=torch.complex128, requires_grad=True)
    ours(coefficients).sum().backward()

    assert coefficients.grad is not None
    assert torch.isfinite(torch.view_as_real(coefficients.grad)).all()


# ==========================================================================
# End to end: a real ShallowWaterSolver initial condition, synthesized at
# the solver's own grid, matching its own spec2grid exactly
# ==========================================================================


def test_matches_shallow_water_solver_spec2grid_on_its_own_grid() -> None:
    """The use ``experiments/shallow_water_paper/data.py`` makes of
    ``ArbitraryPointSHT``: the solver's own spectral state, synthesized
    through the scalar transform at given points. (The companion check of a
    vector synthesis against ``solver.getuv`` went with the (h, u, v)
    velocity representation, which this release does not ship.)"""
    torch.manual_seed(3)
    solver = ShallowWaterSolver(NLAT, NLON, dt=150.0, grid="equiangular").double()
    uspec = solver.random_initial_condition(mach=0.2)

    nodes = _equiangular_nodes()
    scalar_synth = ArbitraryPointSHT(nodes, lmax=solver.lmax, mmax=solver.mmax, nlon=NLON)

    expected_h = solver.spec2grid(uspec[:1]).reshape(1, NLAT * NLON)
    actual_h = scalar_synth(uspec[:1])

    assert torch.allclose(actual_h, expected_h, atol=1e-6, rtol=1e-5)


# ==========================================================================
# ShallowWaterDataset: the solver, the seeding, and spectral_pair
# ==========================================================================


def _tiny_dataset(
    *,
    num_examples: int = 4,
    base_seed: int = 1000,
    split: str = "train",
    sample_policy: str = "fixed_index",
    device: torch.device | str | None = None,
) -> ShallowWaterDataset:
    return ShallowWaterDataset(
        base_seed=base_seed,
        split=split,  # type: ignore[arg-type]
        sample_policy=sample_policy,  # type: ignore[arg-type]
        nlat=NLAT,
        nlon=NLON,
        lead_time_seconds=300.0,
        solver_substeps=2,
        num_examples=num_examples,
        device=device,
    )


def test_rejects_unknown_split_and_policy() -> None:
    for kwargs in (
        {"split": "holdout"},
        {"sample_policy": "random"},
    ):
        with pytest.raises(ValueError):
            _tiny_dataset(**kwargs)  # type: ignore[arg-type]


def test_length_matches_num_examples() -> None:
    assert len(_tiny_dataset(num_examples=7)) == 7


def test_spectral_pair_shapes_dtypes_and_finiteness() -> None:
    """A sample here is the solver's spectral state pair: ``(3, lmax, mmax)``
    complex, in the complex counterpart of the solver's dtype."""
    dataset = _tiny_dataset()
    pair = dataset.spectral_pair(0)

    assert len(pair) == 2
    for tensor in pair:
        assert tensor.shape == (3, dataset.lmax, dataset.mmax)
        assert tensor.dtype == torch.complex64
        assert torch.isfinite(torch.view_as_real(tensor)).all()


# --------------------------------------------------------------------------
# Indexed determinism -- the property the whole benchmark's validity rests on.
# A sample is observed through ``spectral_pair``: the two spectral states an
# index denotes, before any synthesis.
# --------------------------------------------------------------------------


def test_repeated_access_to_one_index_returns_an_identical_sample() -> None:
    dataset = _tiny_dataset()
    first, second = dataset.spectral_pair(0), dataset.spectral_pair(0)

    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)


def test_indexed_samples_survive_unrelated_global_rng_consumption() -> None:
    """Model initialization and dropout must not move the data underneath a run.

    They advance the same process-wide torch RNG the solver draws its
    initial condition from, so without the dataset's own RNG fork this is
    exactly the coupling that made two architectures started from one root
    seed train on different samples.
    """
    dataset = _tiny_dataset()
    reference = dataset.spectral_pair(0)

    torch.randn(10_000)  # stand-in for model init
    dataset.spectral_pair(1)  # accessing a *different* index must not disturb index 0
    torch.nn.functional.dropout(torch.randn(1000), p=0.5, training=True)
    torch.randn(37)

    for a, b in zip(dataset.spectral_pair(0), reference, strict=True):
        assert torch.equal(a, b)


def test_a_rebuilt_dataset_reproduces_every_indexed_sample() -> None:
    original = _tiny_dataset(num_examples=3)
    expected = [original.spectral_pair(index) for index in range(3)]

    torch.manual_seed(999)  # a differently-seeded process
    rebuilt = _tiny_dataset(num_examples=3)

    for index in range(3):
        for a, b in zip(rebuilt.spectral_pair(index), expected[index], strict=True):
            assert torch.equal(a, b)


def test_distinct_indices_are_distinct_trajectories() -> None:
    dataset = _tiny_dataset()
    assert not torch.allclose(dataset.spectral_pair(0)[0], dataset.spectral_pair(1)[0])


def test_split_seeds_produce_distinct_trajectories() -> None:
    """Both halves matter: a different base seed *and* a different split
    name must each be enough on their own to separate two splits."""
    train = _tiny_dataset(base_seed=1000, split="train")
    val = _tiny_dataset(base_seed=2000, split="val")
    val_same_seed = _tiny_dataset(base_seed=1000, split="val")

    assert not torch.allclose(train.spectral_pair(0)[0], val.spectral_pair(0)[0])
    assert not torch.allclose(train.spectral_pair(0)[0], val_same_seed.spectral_pair(0)[0])


def test_fixed_index_policy_ignores_the_epoch() -> None:
    dataset = _tiny_dataset(sample_policy="fixed_index")
    reference = dataset.spectral_pair(0)
    dataset.set_epoch(5)

    for a, b in zip(dataset.spectral_pair(0), reference, strict=True):
        assert torch.equal(a, b)


def test_epoch_index_policy_varies_the_sample_with_the_epoch() -> None:
    dataset = _tiny_dataset(sample_policy="epoch_index")
    first_epoch = dataset.spectral_pair(0)
    dataset.set_epoch(1)

    assert not torch.allclose(dataset.spectral_pair(0)[0], first_epoch[0])

    dataset.set_epoch(0)
    for a, b in zip(dataset.spectral_pair(0), first_epoch, strict=True):
        assert torch.equal(a, b)  # and returning to epoch 0 restores it exactly


# --------------------------------------------------------------------------
# Devices
# --------------------------------------------------------------------------


def test_explicit_device_override_is_honored() -> None:
    assert _tiny_dataset(device="cpu").solver.lap.device == torch.device("cpu")


# ==========================================================================
# Coriolis conditioning (spec 3, 9)
# ==========================================================================


def test_coriolis_is_plus_one_at_the_north_pole_zero_at_the_equator() -> None:
    poles = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
        dtype=torch.float64,
    )

    assert torch.allclose(
        coriolis_channel(poles),
        torch.tensor([1.0, 0.0, 0.0, -1.0], dtype=torch.float64),
    )


def test_coriolis_is_bounded_on_a_real_grid() -> None:
    """Bounded in [-1, 1] is the whole reason for the dimensionless form:
    raw f is ~1e-4 and would vanish next to z-scored channels."""
    nodes = build_icosphere_grid(subdivisions=3, dtype=torch.float64).nodes
    values = coriolis_channel(nodes)

    assert float(values.min()) >= -1.0 - 1e-12
    assert float(values.max()) <= 1.0 + 1e-12
    assert float(values.max()) > 0.9  # actually reaches near the poles


def test_coriolis_equals_the_z_coordinate_for_the_earth_aligned_axis() -> None:
    nodes = build_icosphere_grid(subdivisions=2, dtype=torch.float64).nodes

    assert torch.allclose(coriolis_channel(nodes), nodes[:, 2])


def test_coriolis_follows_a_reoriented_axis() -> None:
    nodes = build_icosphere_grid(subdivisions=2, dtype=torch.float64).nodes

    assert torch.allclose(coriolis_channel(nodes, (1.0, 0.0, 0.0)), nodes[:, 0])


def test_coriolis_normalizes_a_non_unit_axis() -> None:
    nodes = build_icosphere_grid(subdivisions=2, dtype=torch.float64).nodes

    assert torch.allclose(
        coriolis_channel(nodes, (0.0, 0.0, 5.0)), coriolis_channel(nodes)
    )


def test_coriolis_rejects_a_zero_axis() -> None:
    nodes = build_icosphere_grid(subdivisions=1, dtype=torch.float64).nodes

    with pytest.raises(ValueError, match="nonzero"):
        coriolis_channel(nodes, (0.0, 0.0, 0.0))
