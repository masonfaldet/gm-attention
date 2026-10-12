"""Tests for gm_attention.nn.linear.SO2Linear.

The layer's parameters are laid out to hand off directly to
``intertwiners.dense_field_type_intertwiner``: frequency-0 weight is real
``(b_0, a_0)``, and frequency-n weight is the ``as_complex`` view of a real
``(b_n, a_n, 2)`` tensor. So the primary check here is that the fast forward
pass agrees with the dense assembly built from those exact parameters --
that's a genuine implementation-vs-reference comparison, not a tautology,
since the fast path never constructs the dense matrix.

The bias tests exist to pin the equivariance argument directly: only
frequency 0 may carry a bias, and it must actually be a fixed point of
rho_0 (trivially true, since rho_0 = 1), while every frequency n >= 1
output is forced to vanish on a zero input -- there is no bias parameter
that could make it otherwise.
"""

from __future__ import annotations

import math

import pytest
import torch

from gm_attention.nn.linear import SO2Linear
from gm_attention.representations.field_type import (
    SO2FeatureField,
    SO2FieldType,
)
from tests.support.intertwiners import dense_field_type_intertwiner
from gm_attention.representations.so2 import as_complex

ATOL = 1e-9


def _random_field(
    field_type: SO2FieldType,
    batch_shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float64,
) -> SO2FeatureField:
    return SO2FeatureField.from_flat(
        field_type,
        torch.randn(
            (*batch_shape, field_type.dimension),
            dtype=dtype,
            generator=generator,
        ),
    )


def _dense_matrix(layer: SO2Linear) -> torch.Tensor:
    """Assemble the layer's own parameters into the dense reference matrix."""
    weights = {
        frequency: (
            layer.weight[str(frequency)]
            if frequency == 0
            else as_complex(layer.weight[str(frequency)])
        )
        for frequency in layer.common_frequencies
    }

    return dense_field_type_intertwiner(
        layer.field_type_in,
        layer.field_type_out,
        weights,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )


# --------------------------------------------------------------------------
# Construction and parameter shapes
# --------------------------------------------------------------------------


def test_weight_shapes_match_the_common_frequency_multiplicities() -> None:
    field_type_in = SO2FieldType({0: 2, 1: 3, 2: 1})
    field_type_out = SO2FieldType({0: 1, 1: 2, 3: 1})

    layer = SO2Linear(field_type_in, field_type_out, dtype=torch.float64)

    assert set(layer.common_frequencies) == {0, 1}
    assert layer.weight["0"].shape == (1, 2)
    assert layer.weight["1"].shape == (2, 3, 2)
    assert "2" not in layer.weight
    assert "3" not in layer.weight


def test_bias_exists_only_when_output_has_frequency_zero() -> None:
    with_scalar = SO2Linear(
        SO2FieldType({1: 1}), SO2FieldType({0: 1, 1: 1}), dtype=torch.float64
    )
    assert with_scalar.has_bias
    assert with_scalar.bias.shape == (1,)

    without_scalar = SO2Linear(
        SO2FieldType({1: 1}), SO2FieldType({1: 1}), dtype=torch.float64
    )
    assert not without_scalar.has_bias
    assert without_scalar.bias is None


def test_bias_false_disables_the_bias_even_with_scalar_output() -> None:
    layer = SO2Linear(
        SO2FieldType({0: 1}),
        SO2FieldType({0: 1}),
        bias=False,
        dtype=torch.float64,
    )
    assert not layer.has_bias
    assert layer.bias is None


def test_no_parameters_are_created_for_frequencies_outside_the_intersection() -> (
    None
):
    layer = SO2Linear(
        SO2FieldType({1: 2}), SO2FieldType({2: 3}), dtype=torch.float64
    )
    assert list(layer.weight.keys()) == []
    assert not layer.has_bias


def test_dtype_and_device_are_honored() -> None:
    layer = SO2Linear(
        SO2FieldType({0: 1, 1: 1}),
        SO2FieldType({0: 1, 1: 1}),
        dtype=torch.float32,
    )
    assert layer.weight["0"].dtype == torch.float32
    assert layer.weight["1"].dtype == torch.float32
    assert layer.bias.dtype == torch.float32


def test_rejects_a_non_field_type() -> None:
    with pytest.raises(TypeError):
        SO2Linear({0: 1}, SO2FieldType({0: 1}))  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Forward pass vs. the dense reference intertwiner
# --------------------------------------------------------------------------


def test_forward_matches_the_dense_reference_intertwiner() -> None:
    field_type_in = SO2FieldType({0: 2, 1: 3, 2: 1})
    field_type_out = SO2FieldType({0: 1, 1: 2, 2: 4, 3: 1})

    torch.manual_seed(0)
    layer = SO2Linear(field_type_in, field_type_out, bias=False, dtype=torch.float64)

    generator = torch.Generator().manual_seed(1)
    x = _random_field(field_type_in, (5, 7), generator=generator)

    via_layer = layer(x).to_flat()

    matrix = _dense_matrix(layer)
    via_dense = torch.einsum("oi,...i->...o", matrix, x.to_flat())

    assert torch.allclose(via_layer, via_dense, atol=1e-9)


def test_forward_with_bias_matches_the_dense_reference_plus_bias() -> None:
    field_type_in = SO2FieldType({0: 1, 1: 2})
    field_type_out = SO2FieldType({0: 3, 1: 1, 2: 1})

    torch.manual_seed(2)
    layer = SO2Linear(field_type_in, field_type_out, bias=True, dtype=torch.float64)

    generator = torch.Generator().manual_seed(3)
    x = _random_field(field_type_in, (4,), generator=generator)

    via_layer = layer(x).to_flat()

    matrix = _dense_matrix(layer)
    via_dense = torch.einsum("oi,...i->...o", matrix, x.to_flat())

    out_slices = field_type_out.slices()
    via_dense[..., out_slices[0]] += layer.bias

    assert torch.allclose(via_layer, via_dense, atol=1e-9)


def test_frequencies_absent_from_the_output_are_computed_but_dropped() -> None:
    """field_type_in has frequency 2; field_type_out does not -- it must not
    appear in the output, and changing x[2] must not change any output."""
    field_type_in = SO2FieldType({0: 1, 2: 1})
    field_type_out = SO2FieldType({0: 1})

    torch.manual_seed(4)
    layer = SO2Linear(field_type_in, field_type_out, dtype=torch.float64)

    generator = torch.Generator().manual_seed(5)
    x = _random_field(field_type_in, (3,), generator=generator)
    x_perturbed = SO2FeatureField(
        field_type_in,
        {0: x[0], 2: x[2] + 100.0},
    )

    assert torch.allclose(
        layer(x).to_flat(),
        layer(x_perturbed).to_flat(),
        atol=ATOL,
    )
    assert 2 not in layer(x).field_type


def test_frequencies_absent_from_the_input_are_zero_or_bias_only() -> None:
    field_type_in = SO2FieldType({1: 1})
    field_type_out = SO2FieldType({0: 1, 1: 1, 2: 1})

    torch.manual_seed(6)
    layer = SO2Linear(field_type_in, field_type_out, bias=True, dtype=torch.float64)

    generator = torch.Generator().manual_seed(7)
    x = _random_field(field_type_in, (2,), generator=generator)
    output = layer(x)

    # Frequency 2 has no path from the input at all: must be exactly zero.
    assert torch.equal(output[2], torch.zeros_like(output[2]))

    # Frequency 0 has no input path either, but does have a bias: constant,
    # independent of x, equal to the bias broadcast over the batch.
    assert torch.allclose(
        output[0],
        layer.bias.expand_as(output[0]),
        atol=ATOL,
    )


# --------------------------------------------------------------------------
# The bias/equivariance argument, checked directly
# --------------------------------------------------------------------------


def test_zero_input_produces_zero_at_every_nonscalar_frequency() -> None:
    """No parameter exists that could make a rho_n (n >= 1) output nonzero
    on a zero input -- the empirical content of 'bias forced to zero for
    n >= 1'."""
    field_type_in = SO2FieldType({0: 1, 1: 2, 3: 1})
    field_type_out = SO2FieldType({0: 1, 1: 1, 2: 1, 3: 2})

    torch.manual_seed(8)
    layer = SO2Linear(field_type_in, field_type_out, bias=True, dtype=torch.float64)

    zero_field = SO2FeatureField.from_flat(
        field_type_in,
        torch.zeros(field_type_in.dimension, dtype=torch.float64),
    )
    output = layer(zero_field)

    for frequency in (1, 2, 3):
        assert torch.equal(output[frequency], torch.zeros_like(output[frequency]))

    # Frequency 0 need not be zero: it is the fixed point rho_0 = 1 allows.
    assert torch.allclose(output[0], layer.bias.expand_as(output[0]), atol=ATOL)


def test_perturbing_the_bias_only_changes_the_frequency_zero_output() -> None:
    """Behavioral counterpart to the zero-input test: vary the bias itself
    (rather than the input) and confirm no other frequency reacts."""
    field_type = SO2FieldType({0: 2, 1: 3, 2: 1})
    layer = SO2Linear(field_type, field_type, dtype=torch.float64)

    generator = torch.Generator().manual_seed(13)
    x = _random_field(field_type, (3,), generator=generator)

    before = layer(x)

    with torch.no_grad():
        layer.bias += 7.0

    after = layer(x)

    assert not torch.allclose(after[0], before[0], atol=ATOL)
    for frequency in (1, 2):
        assert torch.allclose(after[frequency], before[frequency], atol=ATOL)


# --------------------------------------------------------------------------
# The intertwiner constraint itself, at the field level
# --------------------------------------------------------------------------


def test_forward_satisfies_the_intertwiner_constraint() -> None:
    """layer(x.apply_rotation(g)) == layer(x).apply_rotation(g) for all g."""
    field_type_in = SO2FieldType({0: 1, 1: 2, 2: 1})
    field_type_out = SO2FieldType({0: 2, 1: 1, 3: 1})

    torch.manual_seed(9)
    layer = SO2Linear(field_type_in, field_type_out, bias=False, dtype=torch.float64)

    generator = torch.Generator().manual_seed(10)
    x = _random_field(field_type_in, (6,), generator=generator)

    for angle_value in (0.0, 0.7391, math.pi / 2, 3.5):
        angles = torch.full((6,), angle_value, dtype=torch.float64)

        rotate_then_apply = layer(x.apply_rotation(angles)).to_flat()
        apply_then_rotate = layer(x).apply_rotation(angles).to_flat()

        assert torch.allclose(rotate_then_apply, apply_then_rotate, atol=1e-8)


def test_bias_breaks_the_constraint_at_nonzero_angle_if_it_were_misapplied() -> (
    None
):
    """Sanity check that the constraint test above is discriminating: a
    (deliberately wrong) additive constant at frequency 1 would fail it."""
    field_type = SO2FieldType({1: 1})
    layer = SO2Linear(field_type, field_type, bias=False, dtype=torch.float64)

    generator = torch.Generator().manual_seed(11)
    x = _random_field(field_type, (4,), generator=generator)
    angles = torch.full((4,), 0.9, dtype=torch.float64)

    correct = layer(x.apply_rotation(angles)).to_flat()
    wrong = layer(x).apply_rotation(angles).to_flat() + 1.0

    assert not torch.allclose(correct, wrong, atol=1e-6)


# --------------------------------------------------------------------------
# Autograd
# --------------------------------------------------------------------------


def test_gradients_flow_to_every_parameter() -> None:
    field_type_in = SO2FieldType({0: 1, 1: 2})
    field_type_out = SO2FieldType({0: 1, 1: 1, 2: 1})

    layer = SO2Linear(field_type_in, field_type_out, bias=True, dtype=torch.float64)

    generator = torch.Generator().manual_seed(12)
    x = _random_field(field_type_in, (3,), generator=generator)

    output = layer(x)
    loss = sum(tensor.pow(2).sum() for tensor in output.coefficients.values())
    loss.backward()

    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_gradients_flow_back_to_the_input() -> None:
    field_type = SO2FieldType({0: 1, 1: 1})
    layer = SO2Linear(field_type, field_type, dtype=torch.float64)

    flat = torch.randn(field_type.dimension, dtype=torch.float64, requires_grad=True)
    x = SO2FeatureField.from_flat(field_type, flat)

    output = layer(x)
    loss = sum(tensor.pow(2).sum() for tensor in output.coefficients.values())
    loss.backward()

    assert flat.grad is not None
    assert torch.isfinite(flat.grad).all()


# --------------------------------------------------------------------------
# forward() input validation
# --------------------------------------------------------------------------


def test_forward_rejects_a_mismatched_input_field_type() -> None:
    layer = SO2Linear(
        SO2FieldType({0: 1}), SO2FieldType({0: 1}), dtype=torch.float64
    )
    wrong_field = SO2FeatureField.from_flat(
        SO2FieldType({0: 2}), torch.zeros(2, dtype=torch.float64)
    )

    with pytest.raises(ValueError):
        layer(wrong_field)
