"""Tests for gm_attention.nn.normalization (SO2LayerNorm, SO2Dropout).

SO2LayerNorm's defining, distinguishing feature versus a naive per-frequency
norm is that every frequency shares one pooled denominator -- so a dedicated
test perturbs only one frequency's magnitude and checks every frequency's
*output* moves. SO2Dropout's defining feature is that a non-scalar copy's
two coordinates share one Bernoulli draw -- tested directly by checking they
are always both zero or both scaled identically, never independently. Both
layers' primary correctness claim is exact gauge equivariance of what is,
in SO2Dropout's case, a genuinely stochastic map -- checked by reseeding the
RNG identically on both sides of apply_rotation, which is valid because the
mask depends only on shape/dtype/device and RNG state, never on the feature
values themselves.
"""

from __future__ import annotations

import math

import pytest
import torch

from gm_attention.discretization.edge_geometry import compute_edge_geometry
from gm_attention.discretization.frames import (
    build_tangent_frames,
    rotate_tangent_frames,
)
from gm_attention.discretization.grids.icosphere import build_icosphere_grid
from gm_attention.nn.attention import SO2Attention
from gm_attention.nn.normalization import SO2Dropout, SO2LayerNorm
from gm_attention.representations.field_type import (
    SO2FeatureField,
    SO2FieldType,
)


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
# SO2LayerNorm
# ==========================================================================


def test_layer_norm_rejects_a_non_field_type() -> None:
    with pytest.raises(TypeError):
        SO2LayerNorm({0: 1})  # type: ignore[arg-type]


def test_layer_norm_accepts_a_heterogeneous_field_type() -> None:
    """Each frequency's statistic is a mean over its own multiplicity axis,
    so the merge needs no shared multiplicity."""
    norm = SO2LayerNorm(SO2FieldType({0: 4, 1: 8, 2: 12}), dtype=torch.float64)

    assert tuple(norm.gamma["0"].shape) == (4,)
    assert tuple(norm.gamma["1"].shape) == (8,)
    assert tuple(norm.gamma["2"].shape) == (12,)
    assert tuple(norm.beta.shape) == (4,)


def test_layer_norm_rejects_a_nonpositive_eps() -> None:
    with pytest.raises(ValueError):
        SO2LayerNorm(SO2FieldType({0: 4}), eps=0.0, dtype=torch.float64)


def test_layer_norm_gamma_shapes_and_init() -> None:
    field_type = SO2FieldType({0: 3, 1: 3, 2: 3})
    norm = SO2LayerNorm(field_type, dtype=torch.float64)

    for frequency in (0, 1, 2):
        assert norm.gamma[str(frequency)].shape == (3,)
        assert torch.equal(norm.gamma[str(frequency)], torch.ones(3, dtype=torch.float64))


def test_layer_norm_beta_only_exists_for_frequency_zero() -> None:
    field_type = SO2FieldType({0: 2, 1: 2})
    norm = SO2LayerNorm(field_type, dtype=torch.float64)

    assert norm.beta.shape == (2,)
    assert torch.equal(norm.beta, torch.zeros(2, dtype=torch.float64))


def test_layer_norm_has_no_beta_without_a_scalar_channel() -> None:
    norm = SO2LayerNorm(SO2FieldType({1: 4}), dtype=torch.float64)

    assert norm.beta is None
    assert not norm.has_bias


def test_layer_norm_bias_false_suppresses_beta_even_with_a_scalar_channel() -> None:
    norm = SO2LayerNorm(SO2FieldType({0: 4}), bias=False, dtype=torch.float64)

    assert norm.beta is None
    assert not norm.has_bias


def test_layer_norm_forward_rejects_a_mismatched_field_type() -> None:
    norm = SO2LayerNorm(SO2FieldType({0: 2}), dtype=torch.float64)
    wrong = SO2FeatureField.from_flat(
        SO2FieldType({0: 3}), torch.zeros(3, dtype=torch.float64)
    )

    with pytest.raises(ValueError):
        norm(wrong)


def test_layer_norm_matches_the_boxed_formula_by_hand() -> None:
    """The worked example from the spec: rho = 2(rho_0 + rho_1), by hand."""
    field_type = SO2FieldType({0: 2, 1: 2})
    norm = SO2LayerNorm(field_type, eps=1e-5, dtype=torch.float64)

    with torch.no_grad():
        norm.gamma["0"].copy_(torch.tensor([1.5, -2.0], dtype=torch.float64))
        norm.gamma["1"].copy_(torch.tensor([0.5, 3.0], dtype=torch.float64))
        norm.beta.copy_(torch.tensor([0.1, -0.1], dtype=torch.float64))

    x0 = torch.tensor([[1.0, 3.0]], dtype=torch.float64)  # (1, 2)
    x1 = torch.tensor([[[3.0, 4.0], [1.0, 0.0]]], dtype=torch.float64)  # (1, 2, 2)
    x = SO2FeatureField(field_type, {0: x0, 1: x1})

    output = norm(x)

    mu0 = x0.mean(dim=-1, keepdim=True)  # 2.0
    centered0 = x0 - mu0  # [-1, 1]
    s0_sq = centered0.pow(2).mean(dim=-1)  # mean(1, 1) = 1.0

    squared_norms = x1.pow(2).sum(dim=-1)  # [25, 1]
    s1_sq = (0.5 * squared_norms).mean(dim=-1)  # mean(12.5, 0.5) = 6.5

    sigma_sq = (s0_sq + s1_sq) / 2  # merge over the 2 present frequencies
    d_rho = torch.sqrt(sigma_sq + 1e-5)

    expected0 = norm.gamma["0"] * (centered0 / d_rho.unsqueeze(-1)) + norm.beta
    expected1 = norm.gamma["1"].unsqueeze(-1) * (x1 / d_rho.unsqueeze(-1).unsqueeze(-1))

    assert torch.allclose(output[0], expected0, atol=1e-10)
    assert torch.allclose(output[1], expected1, atol=1e-10)


def test_layer_norm_denominator_is_shared_across_frequencies() -> None:
    """The merge: growing ONE frequency's magnitude must move every
    frequency's normalized output, unlike a naive per-type LayerNorm."""
    field_type = SO2FieldType({0: 4, 1: 4, 2: 4})
    norm = SO2LayerNorm(field_type, dtype=torch.float64)

    generator = torch.Generator().manual_seed(0)
    x = _random_field(field_type, (3,), generator=generator)

    baseline = norm(x)

    scaled_x2 = SO2FeatureField(field_type, {**x.coefficients, 2: x[2] * 50.0})
    perturbed = norm(scaled_x2)

    # Frequency 0 and 1's inputs are untouched, yet their *outputs* must
    # change because the shared denominator grew.
    assert not torch.allclose(baseline[0], perturbed[0], atol=1e-6)
    assert not torch.allclose(baseline[1], perturbed[1], atol=1e-6)


def test_layer_norm_output_does_not_depend_on_training_mode() -> None:
    field_type = SO2FieldType({0: 3, 1: 3})
    norm = SO2LayerNorm(field_type, dtype=torch.float64)

    generator = torch.Generator().manual_seed(1)
    x = _random_field(field_type, (4,), generator=generator)

    norm.train()
    train_output = norm(x).to_flat()
    norm.eval()
    eval_output = norm(x).to_flat()

    assert torch.equal(train_output, eval_output)


def test_layer_norm_forward_is_exactly_gauge_equivariant() -> None:
    field_type = SO2FieldType({0: 2, 1: 2, 2: 2})
    norm = SO2LayerNorm(field_type, dtype=torch.float64)

    generator = torch.Generator().manual_seed(2)
    with torch.no_grad():
        for parameter in norm.parameters():
            parameter.copy_(torch.randn(parameter.shape, dtype=torch.float64, generator=generator))

    x = _random_field(field_type, (6,), generator=generator)

    for angle_value in (0.0, 0.7391, math.pi / 2, 4.0):
        angles = torch.full((6,), angle_value, dtype=torch.float64)

        rotate_then_norm = norm(x.apply_rotation(angles)).to_flat()
        norm_then_rotate = norm(x).apply_rotation(angles).to_flat()

        assert torch.allclose(rotate_then_norm, norm_then_rotate, atol=1e-9)


def test_layer_norm_gradients_flow_to_gamma_beta_and_input() -> None:
    field_type = SO2FieldType({0: 2, 1: 2})
    norm = SO2LayerNorm(field_type, dtype=torch.float64)

    flat = torch.randn(field_type.dimension, dtype=torch.float64, requires_grad=True)
    x = SO2FeatureField.from_flat(field_type, flat)

    output = norm(x)
    loss = sum(t.pow(2).sum() for t in output.coefficients.values())
    loss.backward()

    assert flat.grad is not None
    assert torch.isfinite(flat.grad).all()

    for name, parameter in norm.named_parameters():
        assert parameter.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(parameter.grad).all()


# ==========================================================================
# SO2Dropout
# ==========================================================================


def test_dropout_rejects_a_non_field_type() -> None:
    with pytest.raises(TypeError):
        SO2Dropout({0: 1})  # type: ignore[arg-type]


@pytest.mark.parametrize("p", [-0.1, 1.0, 1.5])
def test_dropout_rejects_p_outside_the_valid_range(p: float) -> None:
    with pytest.raises(ValueError):
        SO2Dropout(SO2FieldType({0: 1}), p=p)


def test_dropout_forward_rejects_a_mismatched_field_type() -> None:
    dropout = SO2Dropout(SO2FieldType({0: 2}))
    wrong = SO2FeatureField.from_flat(
        SO2FieldType({0: 3}), torch.zeros(3, dtype=torch.float64)
    )

    with pytest.raises(ValueError):
        dropout(wrong)


def test_dropout_is_the_identity_in_eval_mode() -> None:
    field_type = SO2FieldType({0: 8, 1: 8})
    dropout = SO2Dropout(field_type, p=0.9)
    dropout.eval()

    generator = torch.Generator().manual_seed(3)
    x = _random_field(field_type, (5,), generator=generator)

    output = dropout(x)

    assert torch.equal(output[0], x[0])
    assert torch.equal(output[1], x[1])


def test_dropout_is_the_identity_when_p_is_zero_even_in_training() -> None:
    field_type = SO2FieldType({0: 4, 1: 4})
    dropout = SO2Dropout(field_type, p=0.0)
    dropout.train()

    generator = torch.Generator().manual_seed(4)
    x = _random_field(field_type, (5,), generator=generator)

    output = dropout(x)

    assert torch.equal(output[0], x[0])
    assert torch.equal(output[1], x[1])


def test_dropout_does_not_require_a_uniform_field_type() -> None:
    dropout = SO2Dropout(SO2FieldType({0: 3, 1: 7}), p=0.3)
    dropout.train()

    generator = torch.Generator().manual_seed(5)
    x = _random_field(SO2FieldType({0: 3, 1: 7}), (2,), generator=generator)

    torch.manual_seed(0)
    output = dropout(x)  # must not raise

    assert output.field_type == dropout.field_type


def test_dropout_empirical_keep_rate_matches_1_minus_p() -> None:
    field_type = SO2FieldType({0: 20000})
    dropout = SO2Dropout(field_type, p=0.3)
    dropout.train()

    x = SO2FeatureField(field_type, {0: torch.ones(1, 20000, dtype=torch.float64)})

    torch.manual_seed(6)
    output = dropout(x)

    keep_rate = (output[0] != 0.0).float().mean().item()
    assert math.isclose(keep_rate, 0.7, abs_tol=0.02)


def test_dropout_scales_kept_elements_by_1_over_keep_probability() -> None:
    field_type = SO2FieldType({0: 5000})
    dropout = SO2Dropout(field_type, p=0.4)
    dropout.train()

    x = SO2FeatureField(field_type, {0: torch.ones(1, 5000, dtype=torch.float64)})

    torch.manual_seed(7)
    output = dropout(x)[0]

    kept = output[output != 0.0]
    assert torch.allclose(kept, torch.full_like(kept, 1.0 / 0.6), atol=1e-10)


def test_dropout_drops_both_coordinates_of_a_copy_together() -> None:
    """The core structural claim: a copy's pair never drops independently."""
    field_type = SO2FieldType({1: 4000})
    dropout = SO2Dropout(field_type, p=0.5)
    dropout.train()

    x = torch.stack(
        [torch.full((1, 4000), 3.0, dtype=torch.float64), torch.full((1, 4000), -7.0, dtype=torch.float64)],
        dim=-1,
    )
    field = SO2FeatureField(field_type, {1: x})

    torch.manual_seed(8)
    output = dropout(field)[1]  # (1, 4000, 2)

    is_zero = output == 0.0
    # Both coordinates of a copy are zero together, or neither is.
    assert torch.equal(is_zero[..., 0], is_zero[..., 1])

    kept_mask = ~is_zero[..., 0]
    kept = output[kept_mask.unsqueeze(-1).expand_as(output)].reshape(-1, 2)
    # Within a kept copy, both coordinates carry the identical scale factor:
    # the original 3 : -7 ratio must be preserved exactly.
    ratios = kept[:, 0] / kept[:, 1]
    assert torch.allclose(ratios, torch.full_like(ratios, 3.0 / -7.0), atol=1e-10)
    # And some copies actually were dropped and some actually were kept, or
    # this test isn't exercising anything.
    assert 0 < kept_mask.sum().item() < 4000


def test_dropout_a_fixed_mask_commutes_with_rotation() -> None:
    """Reseeding identically on both sides makes the mask itself identical
    (it depends only on shape/dtype/device + RNG state, never on x's
    values), so this checks the exact algebraic claim: a scalar-times-
    identity mask commutes with rho_n(r_theta)."""
    field_type = SO2FieldType({0: 3, 1: 4, 2: 4})
    dropout = SO2Dropout(field_type, p=0.5)
    dropout.train()

    generator = torch.Generator().manual_seed(9)
    x = _random_field(field_type, (10,), generator=generator)
    angles = torch.rand(10, dtype=torch.float64, generator=generator) * (2 * math.pi)

    torch.manual_seed(42)
    rotate_then_drop = dropout(x.apply_rotation(angles)).to_flat()

    torch.manual_seed(42)
    drop_then_rotate = dropout(x).apply_rotation(angles).to_flat()

    assert torch.allclose(rotate_then_drop, drop_then_rotate, atol=1e-12)


def test_dropout_gradient_is_the_scale_at_kept_copies_and_zero_at_dropped_ones() -> None:
    field_type = SO2FieldType({1: 20})
    dropout = SO2Dropout(field_type, p=0.5)
    dropout.train()

    coefficients = torch.randn(1, 20, 2, dtype=torch.float64, requires_grad=True)
    x = SO2FeatureField(field_type, {1: coefficients})

    torch.manual_seed(10)
    output = dropout(x)[1]
    output.sum().backward()

    assert coefficients.grad is not None
    kept = (output[0, :, 0] != 0.0) | (output[0, :, 1] != 0.0)

    assert torch.allclose(
        coefficients.grad[0, kept], torch.full_like(coefficients.grad[0, kept], 2.0)
    )
    assert torch.equal(
        coefficients.grad[0, ~kept], torch.zeros_like(coefficients.grad[0, ~kept])
    )


# ==========================================================================
# End-to-end: the user's own pre-norm block, x' = x + Drop(Attention(Norm(x)))
# ==========================================================================


def test_pre_norm_block_stays_gauge_equivariant_with_a_fixed_dropout_mask() -> None:
    grid = build_icosphere_grid(subdivisions=1, dtype=torch.float64)
    frames = build_tangent_frames(grid.nodes)
    coverage_radius = 0.6 * math.pi

    field_type = SO2FieldType({0: 2, 1: 2, 2: 2})
    field_type_attn = SO2FieldType({0: 3, 1: 3})

    torch.manual_seed(11)
    norm = SO2LayerNorm(field_type, dtype=torch.float64)
    attention = SO2Attention(
        field_type, field_type_attn, field_type, cutoff=coverage_radius, dtype=torch.float64
    )
    dropout = SO2Dropout(field_type, p=0.3)
    dropout.train()

    def block(x: SO2FeatureField, edges, seed: int) -> SO2FeatureField:
        residual = attention(norm(x), edges)
        torch.manual_seed(seed)
        dropped = dropout(residual)
        return SO2FeatureField(
            field_type,
            {
                frequency: x[frequency] + dropped[frequency]
                for frequency in field_type.frequencies
            },
        )

    edges = compute_edge_geometry(
        grid, frames, coverage_radius, max_frequency=field_type.max_frequency * 2
    )

    generator = torch.Generator().manual_seed(12)
    node_features = _random_field(field_type, (grid.n_nodes,), generator=generator)

    seed = 999
    original_output = block(node_features, edges, seed).to_flat()

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

    regauged_output = block(regauged_node_features, regauged_edges, seed).to_flat()
    expected = (
        SO2FeatureField.from_flat(field_type, original_output)
        .gauge_transform(gauge_angles)
        .to_flat()
    )

    assert torch.allclose(regauged_output, expected, atol=1e-7)
