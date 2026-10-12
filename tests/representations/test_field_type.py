"""Tests for gm_attention.representations.field_type.

Two contracts matter beyond bookkeeping:

* the packed layout produced by ``to_flat`` must transform under the
  block-diagonal matrix ``sum_n I_{a_n} (x) rho_n``, which is what makes it
  safe to hand a flat vector to non-equivariant code and back; and
* ``gauge_transform`` must reproduce the gauge law that the discretization
  layer already satisfies, since every equivariance test in the project is
  written against that law.
"""

from __future__ import annotations

import math

import pytest
import torch
from omegaconf import OmegaConf

from gm_attention.discretization.edge_geometry import compute_edge_geometry
from gm_attention.discretization.frames import (
    build_tangent_frames,
    rotate_tangent_frames,
)
from gm_attention.discretization.grids.icosphere import build_icosphere_grid
from gm_attention.representations.field_type import (
    SO2FeatureField,
    SO2FieldType,
    scale_field_type,
)
from gm_attention.representations.so2 import irrep_matrix

ATOL = 1e-10


def _random_field(
    field_type: SO2FieldType,
    batch_shape: tuple[int, ...],
    dtype: torch.dtype = torch.float64,
) -> SO2FeatureField:
    generator = torch.Generator().manual_seed(0)

    return SO2FeatureField.from_flat(
        field_type,
        torch.randn(
            (*batch_shape, field_type.dimension),
            dtype=dtype,
            generator=generator,
        ),
    )


def _block_diagonal_representation(
    field_type: SO2FieldType,
    angle: torch.Tensor,
) -> torch.Tensor:
    """Dense ``sum_n I_{a_n} (x) rho_n(r_theta)`` in the packed layout."""
    blocks = [
        irrep_matrix(frequency, angle)
        for frequency, multiplicity in field_type.items()
        for _ in range(multiplicity)
    ]

    return torch.block_diag(*blocks)


# --------------------------------------------------------------------------
# SO2FieldType construction
# --------------------------------------------------------------------------


def test_field_type_sorts_frequencies_ascending() -> None:
    field_type = SO2FieldType({2: 1, 0: 3, 1: 2})

    assert field_type.frequencies == (0, 1, 2)
    assert field_type.multiplicities == (3, 2, 1)


def test_field_type_accepts_an_iterable_of_pairs() -> None:
    assert SO2FieldType([(1, 4), (0, 2)]) == SO2FieldType({0: 2, 1: 4})


def test_field_type_drops_zero_multiplicities() -> None:
    field_type = SO2FieldType({0: 4, 1: 0, 2: 3})

    assert field_type.frequencies == (0, 2)
    assert 1 not in field_type


def test_field_type_dimension_counts_irrep_dimensions() -> None:
    # 8 scalars + 4 vectors (2 each) + 2 frequency-two channels (2 each).
    assert SO2FieldType({0: 8, 1: 4, 2: 2}).dimension == 8 + 8 + 4


def test_field_type_multiplicity_returns_zero_for_absent_frequencies() -> None:
    field_type = SO2FieldType({0: 3, 2: 1})

    assert field_type.multiplicity(2) == 1
    assert field_type.multiplicity(1) == 0


def test_field_type_max_frequency() -> None:
    assert SO2FieldType({0: 1, 5: 2}).max_frequency == 5


def test_field_type_rejects_duplicate_frequencies() -> None:
    with pytest.raises(ValueError):
        SO2FieldType([(1, 2), (1, 3)])


def test_field_type_rejects_negative_multiplicity() -> None:
    with pytest.raises(ValueError):
        SO2FieldType({0: -1})


def test_field_type_rejects_negative_frequency() -> None:
    with pytest.raises(ValueError):
        SO2FieldType({-1: 2})


def test_field_type_rejects_non_integer_multiplicity() -> None:
    with pytest.raises(TypeError):
        SO2FieldType({0: 2.0})  # type: ignore[dict-item]


def test_field_type_rejects_an_empty_specification() -> None:
    with pytest.raises(ValueError):
        SO2FieldType({})


def test_field_type_rejects_an_all_zero_specification() -> None:
    with pytest.raises(ValueError):
        SO2FieldType({0: 0, 1: 0})


def test_field_type_equality_and_hashing() -> None:
    first = SO2FieldType({0: 2, 1: 1})
    second = SO2FieldType([(1, 1), (0, 2)])
    third = SO2FieldType({0: 2, 1: 2})

    assert first == second
    assert hash(first) == hash(second)
    assert first != third
    assert len({first, second, third}) == 2


def test_field_type_constructors() -> None:
    assert SO2FieldType.trivial(5) == SO2FieldType({0: 5})
    assert SO2FieldType.uniform(2, 3) == SO2FieldType({0: 3, 1: 3, 2: 3})


def test_field_type_repr_mentions_dimension() -> None:
    assert "dimension=4" in repr(SO2FieldType({0: 2, 1: 1}))


# --------------------------------------------------------------------------
# Packed layout
# --------------------------------------------------------------------------


def test_slices_partition_the_channel_axis_without_gaps() -> None:
    field_type = SO2FieldType({0: 3, 1: 2, 3: 1})
    channel_slices = field_type.slices()

    start = 0
    for frequency in field_type.frequencies:
        assert channel_slices[frequency].start == start
        start = channel_slices[frequency].stop

    assert start == field_type.dimension


def test_slices_widths_match_multiplicity_times_irrep_dimension() -> None:
    field_type = SO2FieldType({0: 3, 1: 2})
    channel_slices = field_type.slices()

    assert channel_slices[0].stop - channel_slices[0].start == 3
    assert channel_slices[1].stop - channel_slices[1].start == 4


def test_common_frequencies_is_the_sorted_intersection() -> None:
    first = SO2FieldType({0: 1, 1: 2, 3: 1})
    second = SO2FieldType({1: 5, 2: 1, 3: 4})

    assert first.common_frequencies(second) == (1, 3)


def test_common_frequencies_rejects_a_non_field_type() -> None:
    with pytest.raises(TypeError):
        SO2FieldType({0: 1}).common_frequencies({0: 1})  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# SO2FeatureField
# --------------------------------------------------------------------------


def test_from_flat_to_flat_round_trips() -> None:
    field_type = SO2FieldType({0: 3, 1: 2, 2: 1})
    flat = torch.randn(7, field_type.dimension, dtype=torch.float64)

    assert torch.equal(SO2FeatureField.from_flat(field_type, flat).to_flat(), flat)


def test_to_flat_from_flat_round_trips() -> None:
    field_type = SO2FieldType({0: 2, 1: 3})
    field = _random_field(field_type, (4, 5))

    recovered = SO2FeatureField.from_flat(field_type, field.to_flat())

    for frequency in field_type.frequencies:
        assert torch.equal(recovered[frequency], field[frequency])


def test_coefficient_shapes_follow_the_documented_layout() -> None:
    field_type = SO2FieldType({0: 3, 2: 4})
    field = _random_field(field_type, (6,))

    assert field[0].shape == (6, 3)
    assert field[2].shape == (6, 4, 2)


def test_batch_shape_dtype_and_device() -> None:
    field = _random_field(SO2FieldType({0: 2, 1: 1}), (3, 4))

    assert field.batch_shape == torch.Size((3, 4))
    assert field.dtype == torch.float64
    assert field.device == torch.device("cpu")


def test_empty_batch_shape_is_supported() -> None:
    field_type = SO2FieldType({0: 2, 1: 1})
    field = _random_field(field_type, ())

    assert field.batch_shape == torch.Size(())
    assert field[1].shape == (1, 2)


def test_field_rejects_missing_frequencies() -> None:
    field_type = SO2FieldType({0: 2, 1: 1})

    with pytest.raises(ValueError):
        SO2FeatureField(field_type, {0: torch.zeros(3, 2)})


def test_field_rejects_extra_frequencies() -> None:
    field_type = SO2FieldType({0: 2})

    with pytest.raises(ValueError):
        SO2FeatureField(
            field_type,
            {0: torch.zeros(3, 2), 1: torch.zeros(3, 1, 2)},
        )


def test_field_rejects_a_wrong_multiplicity() -> None:
    field_type = SO2FieldType({1: 3})

    with pytest.raises(ValueError):
        SO2FeatureField(field_type, {1: torch.zeros(4, 2, 2)})


def test_field_rejects_a_missing_irrep_axis() -> None:
    field_type = SO2FieldType({1: 3})

    with pytest.raises(ValueError):
        SO2FeatureField(field_type, {1: torch.zeros(4, 3)})


def test_field_rejects_mismatched_batch_shapes() -> None:
    field_type = SO2FieldType({0: 2, 1: 1})

    with pytest.raises(ValueError):
        SO2FeatureField(
            field_type,
            {0: torch.zeros(3, 2), 1: torch.zeros(4, 1, 2)},
        )


def test_field_rejects_mismatched_dtypes() -> None:
    field_type = SO2FieldType({0: 2, 1: 1})

    with pytest.raises(ValueError):
        SO2FeatureField(
            field_type,
            {
                0: torch.zeros(3, 2, dtype=torch.float64),
                1: torch.zeros(3, 1, 2, dtype=torch.float32),
            },
        )


def test_field_rejects_integer_coefficients() -> None:
    with pytest.raises(TypeError):
        SO2FeatureField(
            SO2FieldType({0: 2}),
            {0: torch.zeros(3, 2, dtype=torch.long)},
        )


def test_from_flat_rejects_a_wrong_channel_count() -> None:
    field_type = SO2FieldType({0: 2, 1: 1})

    with pytest.raises(ValueError):
        SO2FeatureField.from_flat(field_type, torch.zeros(3, 5))


def test_getitem_rejects_an_absent_frequency() -> None:
    field = _random_field(SO2FieldType({0: 2}), (3,))

    with pytest.raises(KeyError):
        field[1]


def test_to_changes_dtype_across_all_frequencies() -> None:
    field = _random_field(SO2FieldType({0: 2, 1: 1}), (3,))
    converted = field.to(dtype=torch.float32)

    assert converted.dtype == torch.float32
    assert all(
        tensor.dtype == torch.float32
        for tensor in converted.coefficients.values()
    )


# --------------------------------------------------------------------------
# Group action and the gauge law
# --------------------------------------------------------------------------


def test_apply_rotation_matches_the_dense_block_diagonal_action() -> None:
    """The packed layout must be ``sum_n I_{a_n} (x) rho_n``."""
    field_type = SO2FieldType({0: 3, 1: 2, 2: 1})
    field = _random_field(field_type, (5,))
    angle = torch.tensor(0.7391, dtype=torch.float64)

    rotated = field.apply_rotation(angle).to_flat()

    dense = _block_diagonal_representation(field_type, angle)
    expected = field.to_flat() @ dense.transpose(-1, -2)

    assert torch.allclose(rotated, expected, atol=1e-9)


def test_apply_rotation_leaves_scalar_channels_untouched() -> None:
    field_type = SO2FieldType({0: 4, 1: 1})
    field = _random_field(field_type, (6,))
    angles = torch.randn(6, dtype=torch.float64)

    rotated = field.apply_rotation(angles)

    assert torch.equal(rotated[0], field[0])
    assert not torch.allclose(rotated[1], field[1], atol=1e-3)


def test_apply_rotation_composes_by_angle_addition() -> None:
    field_type = SO2FieldType({0: 1, 1: 2, 3: 1})
    field = _random_field(field_type, (4,))

    first = torch.randn(4, dtype=torch.float64)
    second = torch.randn(4, dtype=torch.float64)

    twice = field.apply_rotation(first).apply_rotation(second).to_flat()
    once = field.apply_rotation(first + second).to_flat()

    assert torch.allclose(twice, once, atol=1e-9)


def test_apply_rotation_by_zero_is_the_identity() -> None:
    field_type = SO2FieldType({0: 2, 1: 2})
    field = _random_field(field_type, (5,))

    rotated = field.apply_rotation(torch.zeros(5, dtype=torch.float64))

    assert torch.allclose(rotated.to_flat(), field.to_flat(), atol=ATOL)


def test_gauge_transform_is_the_inverse_group_action() -> None:
    field_type = SO2FieldType({0: 2, 1: 1, 2: 2})
    field = _random_field(field_type, (5,))
    angles = torch.randn(5, dtype=torch.float64)

    assert torch.allclose(
        field.gauge_transform(angles).to_flat(),
        field.apply_rotation(-angles).to_flat(),
        atol=ATOL,
    )


def test_gauge_transform_then_rotation_recovers_the_original() -> None:
    field_type = SO2FieldType({0: 1, 1: 3})
    field = _random_field(field_type, (7,))
    angles = torch.randn(7, dtype=torch.float64)

    recovered = field.gauge_transform(angles).apply_rotation(angles)

    assert torch.allclose(recovered.to_flat(), field.to_flat(), atol=1e-9)


def test_rotation_broadcasts_angles_against_the_batch_shape() -> None:
    field_type = SO2FieldType({0: 1, 1: 2})
    field = _random_field(field_type, (3, 4))

    per_node = torch.randn(4, dtype=torch.float64)
    rotated = field.apply_rotation(per_node)

    assert rotated.batch_shape == field.batch_shape
    assert torch.allclose(
        rotated[1],
        field.apply_rotation(per_node.expand(3, 4))[1],
        atol=ATOL,
    )


def test_rotation_rejects_a_non_broadcastable_angle_shape() -> None:
    field = _random_field(SO2FieldType({0: 1, 1: 1}), (3, 4))

    with pytest.raises(ValueError):
        field.apply_rotation(torch.randn(5, dtype=torch.float64))


def test_rotation_rejects_a_mismatched_angle_dtype() -> None:
    field = _random_field(SO2FieldType({0: 1, 1: 1}), (3,))

    with pytest.raises(TypeError):
        field.apply_rotation(torch.randn(3, dtype=torch.float32))


def test_gauge_transform_reproduces_the_edge_geometry_gauge_law() -> None:
    """Tangent vectors are a frequency-one field; the two laws must agree.

    ``compute_edge_geometry`` guarantees
    ``log_coordinates' = r(theta_i)^T log_coordinates`` under a regauging by
    ``theta``. Treating the log-coordinates as a multiplicity-one frequency-one
    field, ``gauge_transform`` must produce the same tensor -- this pins the
    sign convention that every equivariant layer will rely on.
    """
    grid = build_icosphere_grid(subdivisions=1, dtype=torch.float64)
    frames = build_tangent_frames(grid.nodes)
    coverage_radius = 0.6 * math.pi

    edges = compute_edge_geometry(grid, frames, coverage_radius)

    generator = torch.Generator().manual_seed(11)
    gauge_angles = torch.rand(
        grid.n_nodes,
        dtype=torch.float64,
        generator=generator,
    ) * (2 * math.pi)

    regauged_edges = compute_edge_geometry(
        grid,
        rotate_tangent_frames(frames, gauge_angles),
        coverage_radius,
    )

    vector_type = SO2FieldType({1: 1})
    field = SO2FeatureField(
        vector_type,
        {1: edges.log_coordinates.unsqueeze(-2)},  # (n_edges, 1, 2)
    )

    # Log-coordinates live in the *target* node's gauge.
    transformed = field.gauge_transform(gauge_angles[edges.target_index])

    assert torch.allclose(
        transformed[1].squeeze(-2),
        regauged_edges.log_coordinates,
        atol=1e-9,
    )


def test_scaled_multiplies_every_multiplicity_and_keeps_the_spectrum() -> None:
    scaled = SO2FieldType({0: 2, 1: 1}).scaled(4)

    assert scaled == SO2FieldType({0: 8, 1: 4})
    assert scaled.frequencies == (0, 1)


def test_scaled_by_one_is_an_equal_field_type() -> None:
    field_type = SO2FieldType({0: 3, 2: 5})

    assert field_type.scaled(1) == field_type


def test_scaled_preserves_heterogeneous_ratios() -> None:
    """The point of a base type: the ratio between frequencies is fixed by
    the base and the width is set independently, so scaling must not
    flatten the spectrum toward one shared multiplicity."""
    base = SO2FieldType({0: 2, 1: 1, 2: 3})

    for factor in (1, 2, 7):
        scaled = base.scaled(factor)
        assert [m // factor for m in scaled.multiplicities] == list(base.multiplicities)


def test_scaled_returns_a_new_object_leaving_the_original_alone() -> None:
    field_type = SO2FieldType({0: 1, 1: 1})
    scaled = field_type.scaled(8)

    assert field_type == SO2FieldType({0: 1, 1: 1})
    assert scaled is not field_type


@pytest.mark.parametrize("factor", [0, -1, -8])
def test_scaled_rejects_a_nonpositive_factor(factor: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        SO2FieldType({0: 1}).scaled(factor)


@pytest.mark.parametrize("factor", [True, 2.0, "2", None])
def test_scaled_rejects_a_non_integer_factor(factor: object) -> None:
    """bool is included because it is an int subclass and would otherwise
    act as factor=1."""
    with pytest.raises(TypeError, match="must be an int"):
        SO2FieldType({0: 1}).scaled(factor)  # type: ignore[arg-type]


def test_scale_field_type_matches_scaled_for_an_integer_keyed_mapping() -> None:
    assert scale_field_type({0: 1, 1: 1, 2: 1}, 16) == SO2FieldType(
        {0: 1, 1: 1, 2: 1}
    ).scaled(16)


def test_scale_field_type_accepts_decimal_integer_string_keys() -> None:
    """OmegaConf stores mapping keys as str, so a YAML base_field_type can
    arrive as {"0": 2, "1": 1}."""
    assert scale_field_type({"0": 2, "1": 1}, 4) == SO2FieldType({0: 8, 1: 4})


def test_scale_field_type_accepts_an_omegaconf_mapping() -> None:
    """The real config path, not a stand-in dict."""
    config = OmegaConf.create({"base_field_type": {0: 1, 1: 1, 2: 1}})

    assert scale_field_type(config.base_field_type, 16) == SO2FieldType(
        {0: 16, 1: 16, 2: 16}
    )


@pytest.mark.parametrize("key", [" 1", "1.0", "0x1", "+1", "-1", "1_0", ""])
def test_scale_field_type_rejects_a_loosely_integer_string_key(key: str) -> None:
    """int() would accept several of these; a typo in a config must fail
    loudly rather than quietly define a different architecture."""
    with pytest.raises(TypeError, match="exact decimal integer"):
        scale_field_type({key: 1}, 4)


def test_scale_field_type_rejects_a_boolean_key() -> None:
    with pytest.raises(TypeError, match="bool"):
        scale_field_type({True: 1}, 4)


def test_scale_field_type_rejects_a_boolean_multiplicity() -> None:
    with pytest.raises(TypeError, match="must be an int"):
        scale_field_type({0: True}, 4)


def test_scale_field_type_rejects_a_zero_base_multiplicity() -> None:
    """A zero entry is treated as an error rather than a dropped
    frequency, since a frequency to be excluded is simply omitted."""
    with pytest.raises(ValueError, match="at least 1"):
        scale_field_type({0: 1, 2: 0}, 4)


def test_scale_field_type_rejects_an_empty_mapping() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        scale_field_type({}, 4)


def test_scale_field_type_rejects_a_duplicate_frequency_across_key_types() -> None:
    with pytest.raises(ValueError, match="more than once"):
        scale_field_type({0: 1, "0": 1}, 4)


def test_scale_field_type_rejects_a_negative_frequency() -> None:
    with pytest.raises(ValueError):
        scale_field_type({-1: 1}, 4)


def test_scale_field_type_resolves_the_documented_heterogeneous_example() -> None:
    """Spec 11.3: base {0: 2, 1: 1} x 4 -> {0: 8, 1: 4}, FFN x2 -> {0: 16, 1: 8}."""
    resolved = scale_field_type({0: 2, 1: 1}, 4)

    assert resolved == SO2FieldType({0: 8, 1: 4})
    assert resolved.scaled(2) == SO2FieldType({0: 16, 1: 8})
