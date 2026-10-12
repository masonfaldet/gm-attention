"""Tests for gm_attention.nn.nonlinearities (SO2ScalarGatedNonlinearity).

The supplement ships a single pointwise nonlinearity, the scalar-gated map
used by every reported model. It is a genuinely nonlinear self-map, so the
primary correctness claim -- exact gauge equivariance -- is checked directly
(apply_rotation / gauge_transform before vs. after) rather than only through
the "the gate is invariant" sub-argument. The shared constructor / forward
contract (field-type validation, positive eps, plain SiLU on scalar channels)
is kept in its parametrized form so the file reads the same as the
development suite it was taken from; the parametrization now ranges over the
one class the supplement ships.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from gm_attention.discretization.edge_geometry import compute_edge_geometry
from gm_attention.discretization.frames import (
    build_tangent_frames,
    rotate_tangent_frames,
)
from gm_attention.discretization.grids.icosphere import build_icosphere_grid
from gm_attention.nn.convolution import SO2Convolution
from gm_attention.nn.nonlinearities import SO2ScalarGatedNonlinearity
from gm_attention.representations.field_type import (
    SO2FeatureField,
    SO2FieldType,
)

ATOL = 1e-9

# The supplement ships one pointwise nonlinearity; the shared-contract tests
# below stay parametrized over it so they keep the shape they had in the
# development suite, where they ranged over the norm-gated classes too.
NONLINEARITY_CLASSES = (SO2ScalarGatedNonlinearity,)


def _random_field(
    field_type: SO2FieldType,
    batch_shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float64,
) -> SO2FeatureField:
    return SO2FeatureField.from_flat(
        field_type,
        torch.randn((*batch_shape, field_type.dimension), dtype=dtype, generator=generator),
    )


# ==========================================================================
# Construction / forward validation: the shared nonlinearity contract
# ==========================================================================


@pytest.mark.parametrize("cls", NONLINEARITY_CLASSES)
def test_rejects_a_non_field_type(cls: type) -> None:
    with pytest.raises(TypeError):
        cls({0: 1})


@pytest.mark.parametrize("cls", NONLINEARITY_CLASSES)
def test_rejects_a_nonpositive_eps(cls: type) -> None:
    with pytest.raises(ValueError):
        cls(SO2FieldType({0: 1}), eps=0.0)


@pytest.mark.parametrize("cls", NONLINEARITY_CLASSES)
def test_forward_rejects_a_mismatched_field_type(cls: type) -> None:
    nonlin = cls(SO2FieldType({0: 1}), dtype=torch.float64)
    wrong_field = SO2FeatureField.from_flat(
        SO2FieldType({0: 2}), torch.zeros(2, dtype=torch.float64)
    )

    with pytest.raises(ValueError):
        nonlin(wrong_field)


@pytest.mark.parametrize("cls", NONLINEARITY_CLASSES)
def test_scalar_channels_get_plain_silu(cls: type) -> None:
    field_type = SO2FieldType({0: 4})
    nonlin = cls(field_type, dtype=torch.float64)

    x = SO2FeatureField.from_flat(
        field_type, torch.tensor([[-2.0, -0.5, 0.5, 3.0]], dtype=torch.float64)
    )
    output = nonlin(x)

    assert torch.allclose(output[0], F.silu(x[0]), atol=ATOL)


@pytest.mark.parametrize("cls", NONLINEARITY_CLASSES)
def test_forward_is_exactly_gauge_equivariant(cls: type) -> None:
    field_type = SO2FieldType({0: 2, 1: 2, 2: 1})
    torch.manual_seed(1)
    nonlin = cls(field_type, dtype=torch.float64)

    generator = torch.Generator().manual_seed(2)
    x = _random_field(field_type, (6,), generator=generator)

    for angle_value in (0.0, 0.7391, math.pi / 2, 4.0):
        angles = torch.full((6,), angle_value, dtype=torch.float64)

        rotate_then_apply = nonlin(x.apply_rotation(angles)).to_flat()
        apply_then_rotate = nonlin(x).apply_rotation(angles).to_flat()

        assert torch.allclose(rotate_then_apply, apply_then_rotate, atol=1e-9)


def test_norm_is_exactly_preserved_under_rotation() -> None:
    """The invariance any norm-based gauge argument leans on, checked directly."""
    field_type = SO2FieldType({1: 3})
    generator = torch.Generator().manual_seed(3)
    x = _random_field(field_type, (5,), generator=generator)

    angles = torch.randn(5, dtype=torch.float64)
    rotated = x.apply_rotation(angles)

    original_norm = torch.linalg.vector_norm(x[1], dim=-1)
    rotated_norm = torch.linalg.vector_norm(rotated[1], dim=-1)

    assert torch.allclose(original_norm, rotated_norm, atol=1e-10)


# ==========================================================================
# End-to-end: inserted into a stack, the stack stays equivariant
# ==========================================================================


@pytest.mark.parametrize("cls", NONLINEARITY_CLASSES)
def test_stacked_with_convolution_the_composition_stays_gauge_equivariant(
    cls: type,
) -> None:
    """A pipeline property independent of the nonlinearity; the development
    suite ran it over the norm-gated classes as well, the supplement runs it
    over the scalar-gated map it ships."""
    grid = build_icosphere_grid(subdivisions=1, dtype=torch.float64)
    frames = build_tangent_frames(grid.nodes)
    coverage_radius = 0.6 * math.pi

    field_type = SO2FieldType({0: 1, 1: 1, 2: 1})

    torch.manual_seed(4)
    conv = SO2Convolution(field_type, field_type, cutoff=coverage_radius, dtype=torch.float64)
    nonlin = cls(field_type, dtype=torch.float64)

    def stack(x: SO2FeatureField, edges) -> SO2FeatureField:
        return nonlin(conv(x, edges))

    edges = compute_edge_geometry(
        grid, frames, coverage_radius, max_frequency=field_type.max_frequency * 2
    )

    generator = torch.Generator().manual_seed(5)
    node_features = _random_field(field_type, (grid.n_nodes,), generator=generator)

    original_output = stack(node_features, edges).to_flat()

    gauge_angles = torch.rand(grid.n_nodes, dtype=torch.float64, generator=generator) * (
        2 * math.pi
    )
    regauged_edges = compute_edge_geometry(
        grid,
        rotate_tangent_frames(frames, gauge_angles),
        coverage_radius,
        max_frequency=field_type.max_frequency * 2,
    )
    regauged_node_features = node_features.gauge_transform(gauge_angles)

    regauged_output = stack(regauged_node_features, regauged_edges).to_flat()
    expected = SO2FeatureField.from_flat(field_type, original_output).gauge_transform(
        gauge_angles
    ).to_flat()

    assert torch.allclose(regauged_output, expected, atol=1e-7)


# ==========================================================================
# SO2ScalarGatedNonlinearity
# ==========================================================================


def _gated_field_type() -> SO2FieldType:
    return SO2FieldType({0: 3, 1: 2, 2: 2})


def test_scalar_gate_is_nodewise_gauge_equivariant() -> None:
    field_type = _gated_field_type()
    torch.manual_seed(60)
    module = SO2ScalarGatedNonlinearity(field_type, dtype=torch.float64)

    x = SO2FeatureField.from_flat(
        field_type, torch.randn(9, field_type.dimension, dtype=torch.float64)
    )
    angles = torch.rand(9, dtype=torch.float64) * 2 * math.pi

    assert torch.allclose(
        module(x.gauge_transform(angles)).to_flat(),
        module(x).gauge_transform(angles).to_flat(),
        atol=1e-12,
    )


def test_one_gate_is_shared_across_both_coordinates_of_a_copy() -> None:
    """Gating the two coordinates of one irrep copy differently would be a
    non-uniform scaling of R^2, which does not commute with rotation. The
    ratio output/input must therefore be identical across the pair."""
    field_type = SO2FieldType({0: 2, 1: 3})
    torch.manual_seed(61)
    module = SO2ScalarGatedNonlinearity(field_type, dtype=torch.float64)

    x = SO2FeatureField.from_flat(
        field_type, torch.randn(5, field_type.dimension, dtype=torch.float64)
    )
    output = module(x)

    ratio = output[1] / x[1]  # (5, 3, 2)
    assert torch.allclose(ratio[..., 0], ratio[..., 1], atol=1e-12)


def test_gates_respond_to_scalar_channels_with_the_nonscalar_input_fixed() -> None:
    """The capability a norm-gated map lacks entirely: information flowing
    from scalar content into a non-scalar frequency."""
    field_type = SO2FieldType({0: 2, 1: 1})
    torch.manual_seed(62)
    module = SO2ScalarGatedNonlinearity(field_type, dtype=torch.float64)

    nonscalar = torch.randn(4, 1, 2, dtype=torch.float64)
    first = SO2FeatureField(
        field_type, {0: torch.zeros(4, 2, dtype=torch.float64), 1: nonscalar}
    )
    second = SO2FeatureField(
        field_type, {0: torch.full((4, 2), 3.0, dtype=torch.float64), 1: nonscalar}
    )

    assert not torch.allclose(module(first)[1], module(second)[1])


def test_the_gate_can_reach_near_zero_and_exceed_one() -> None:
    """The range property a sigmoid(r) in [0.5, 1) norm gate cannot have:
    a scalar gate must be able to suppress a feature and to amplify it."""
    field_type = SO2FieldType({0: 1, 1: 1})
    module = SO2ScalarGatedNonlinearity(field_type, dtype=torch.float64)

    with torch.no_grad():  # drive the gate directly
        module.gates["1"].weight.fill_(1.0)
        module.gates["1"].bias.zero_()

    ones = torch.ones(1, 1, 2, dtype=torch.float64)

    suppressed = module(
        SO2FeatureField(field_type, {0: torch.full((1, 1), -6.0, dtype=torch.float64), 1: ones})
    )
    amplified = module(
        SO2FeatureField(field_type, {0: torch.full((1, 1), 4.0, dtype=torch.float64), 1: ones})
    )

    assert float(suppressed[1].abs().max()) < 0.05
    assert float(amplified[1].abs().max()) > 1.0


def test_double_sigmoid_gate_is_strictly_positive_and_bounded_by_two() -> None:
    field_type = SO2FieldType({0: 1, 1: 1})
    module = SO2ScalarGatedNonlinearity(
        field_type, gate="double_sigmoid", dtype=torch.float64
    )
    with torch.no_grad():
        module.gates["1"].weight.fill_(1.0)
        module.gates["1"].bias.zero_()

    ones = torch.ones(5, 1, 2, dtype=torch.float64)
    scalars = torch.linspace(-20.0, 20.0, 5, dtype=torch.float64).unsqueeze(-1)

    gated = module(SO2FeatureField(field_type, {0: scalars, 1: ones}))[1]

    assert bool((gated > 0.0).all())
    assert bool((gated < 2.0).all())


def test_scalar_channels_pass_through_a_plain_silu() -> None:
    field_type = SO2FieldType({0: 2, 1: 1})
    torch.manual_seed(63)
    module = SO2ScalarGatedNonlinearity(field_type, dtype=torch.float64)

    x = SO2FeatureField.from_flat(
        field_type, torch.randn(6, field_type.dimension, dtype=torch.float64)
    )

    assert torch.allclose(module(x)[0], torch.nn.functional.silu(x[0]))


def test_output_preserves_the_field_type_and_batch_shape() -> None:
    field_type = _gated_field_type()
    module = SO2ScalarGatedNonlinearity(field_type, dtype=torch.float64)
    x = SO2FeatureField.from_flat(
        field_type, torch.randn(2, 7, field_type.dimension, dtype=torch.float64)
    )

    output = module(x)

    assert output.field_type == field_type
    assert output.batch_shape == (2, 7)


def test_gradients_are_finite_at_zero_input() -> None:
    field_type = _gated_field_type()
    torch.manual_seed(64)
    module = SO2ScalarGatedNonlinearity(field_type, dtype=torch.float64)

    flat = torch.zeros(4, field_type.dimension, dtype=torch.float64, requires_grad=True)
    module(SO2FeatureField.from_flat(field_type, flat)).to_flat().square().sum().backward()

    assert flat.grad is not None
    assert torch.isfinite(flat.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


def test_requires_scalar_channels_to_condition_on() -> None:
    with pytest.raises(ValueError, match="frequency-0 channels"):
        SO2ScalarGatedNonlinearity(SO2FieldType({1: 2}))


def test_rejects_an_unknown_gate() -> None:
    with pytest.raises(ValueError, match="gate must be"):
        SO2ScalarGatedNonlinearity(
            SO2FieldType({0: 1, 1: 1}), gate="tanh"  # type: ignore[arg-type]
        )


def test_rejects_a_mismatched_input_field_type() -> None:
    module = SO2ScalarGatedNonlinearity(SO2FieldType({0: 1, 1: 1}), dtype=torch.float64)
    wrong = SO2FeatureField.from_flat(
        SO2FieldType({0: 2}), torch.randn(3, 2, dtype=torch.float64)
    )

    with pytest.raises(ValueError):
        module(wrong)
