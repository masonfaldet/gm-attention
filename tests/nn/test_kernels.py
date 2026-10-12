"""Tests for gm_attention.nn.kernels.SO2SteerableKernel.

The primary check is the same shape as step 4's for SO2Linear: build a
synthetic ``EdgeGeometry`` with hand-chosen ``(r, phi, gamma)``, extract the
kernel's own learned radial coefficients at those edges, and feed them into
``bases.dense_field_type_steerable_kernel`` (the literal Table 8.3
transcription). The fast complex-arithmetic forward pass and the dense
Kronecker-assembled reference must agree exactly -- that is a genuine
implementation-vs-reference comparison, not a tautology, since the two never
share code below the coefficient-extraction boundary.

A second suite checks the module end to end on a real discretization: that
regauging the mesh and gathering source features through ``gauge_transform``
reproduces the same result as regauging the raw geometry, which is the
transport half of the story that a synthetic-edges test can't exercise
(synthetic edges are built with arbitrary, not necessarily
gauge-consistent, r/phi/gamma).
"""

from __future__ import annotations

import math

import pytest
import torch

from gm_attention.discretization.edge_geometry import (
    EdgeGeometry,
    compute_edge_geometry,
)
from gm_attention.discretization.frames import (
    build_tangent_frames,
    rotate_tangent_frames,
)
from gm_attention.discretization.grid import SphericalGrid
from gm_attention.discretization.grids.icosphere import build_icosphere_grid
from gm_attention.nn.kernels import SO2SteerableKernel
from gm_attention.representations import so2
from tests.support.bases import dense_field_type_steerable_kernel
from gm_attention.representations.field_type import (
    SO2FeatureField,
    SO2FieldType,
)

ATOL = 1e-8


def _synthetic_edges(
    distances: torch.Tensor,  # (E,)
    polar_angles: torch.Tensor,  # (E,)
    transport_angles: torch.Tensor,  # (E,)
    *,
    self_loop: torch.Tensor | None = None,  # (E,) bool
    max_frequency: int,
) -> EdgeGeometry:
    """Build an EdgeGeometry by hand, with fully controlled r/phi/gamma.

    Only the fields SO2SteerableKernel actually reads are made meaningful
    (distances, angular/transport harmonics, and the self-loop structure via
    edge_index); row_ptr and tangent_quadrature_weights are filled in with
    trivial placeholders since nothing under test consults them.
    """
    n_edges = distances.shape[0]

    if self_loop is None:
        self_loop = torch.zeros(n_edges, dtype=torch.bool)

    # Two "nodes": 0 for a self-loop edge, 0 <- 1 otherwise.
    target_index = torch.zeros(n_edges, dtype=torch.long)
    source_index = torch.where(
        self_loop, torch.zeros(n_edges, dtype=torch.long), torch.ones(n_edges, dtype=torch.long)
    )
    edge_index = torch.stack((target_index, source_index))

    log_coordinates = distances.unsqueeze(-1) * torch.stack(
        (torch.cos(polar_angles), torch.sin(polar_angles)), dim=-1
    )

    return EdgeGeometry(
        edge_index=edge_index,
        row_ptr=torch.tensor([0, n_edges, n_edges], dtype=torch.long),
        log_coordinates=log_coordinates,
        distances=distances,
        polar_angles=polar_angles,
        transport_angles=transport_angles,
        tangent_quadrature_weights=torch.ones(n_edges, dtype=distances.dtype),
        angular_phases=so2.phase_powers(polar_angles, max_frequency),
        transport_phases=so2.phase_powers(transport_angles, max_frequency),
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


def _extract_dense_radial_coefficients(
    kernel: SO2SteerableKernel,
    edges: EdgeGeometry,
) -> dict[tuple[int, int], torch.Tensor]:
    """Recompute the kernel's own C_i^{m<-n}(r) in bases.py's dense convention.

    This re-derives the (dim, b_m, a_n)-per-edge coefficient tensors from the
    kernel's public radial submodules, independently of ``_forward_general``,
    so that feeding them to ``dense_field_type_steerable_kernel`` is a real
    cross-check of the complex-arithmetic reduction rather than a copy of it.
    """
    embedding = kernel.radial_basis(edges.distances)
    hidden = kernel.radial_trunk(embedding)

    complex_coefficients = (
        kernel.complex_head(hidden) if kernel.complex_head is not None else None
    )
    if complex_coefficients is not None:
        # The same (r/cutoff)^|k| regularity factor the kernel applies (see
        # its ``_radial_powers``). Recomputed here from _column_powers rather
        # than by calling the kernel's own helper, so this stays an
        # independent re-derivation of the coefficients and not a copy of
        # the code under test.
        normalized = edges.distances / kernel.radial_basis.cutoff
        scale = torch.stack(
            [normalized**power for power in kernel._column_powers.tolist()], dim=-1
        )
        complex_coefficients = complex_coefficients * scale

    real_coefficients = (
        kernel.real_head(hidden) if kernel.real_head is not None else None
    )

    n_edges = edges.n_edges
    result: dict[tuple[int, int], torch.Tensor] = {}

    for frequency_n, frequency_m in kernel.pairs:
        multiplicity_in = kernel.field_type_in.multiplicity(frequency_n)
        multiplicity_out = kernel.field_type_out.multiplicity(frequency_m)

        if frequency_n == 0 and frequency_m == 0:
            assert real_coefficients is not None
            matrix = real_coefficients.reshape(n_edges, multiplicity_out, multiplicity_in)
            result[(0, 0)] = matrix.unsqueeze(1)  # (E, dim=1, b_0, a_0)
            continue

        assert complex_coefficients is not None
        linear_range = kernel._pair_ranges[(frequency_n, frequency_m, "linear")]
        linear = complex_coefficients[:, linear_range].reshape(
            n_edges, multiplicity_out, multiplicity_in, 2
        )
        c1, c2 = linear[..., 0], linear[..., 1]

        if frequency_n >= 1 and frequency_m >= 1:
            antilinear_range = kernel._pair_ranges[(frequency_n, frequency_m, "antilinear")]
            antilinear = complex_coefficients[:, antilinear_range].reshape(
                n_edges, multiplicity_out, multiplicity_in, 2
            )
            c3, c4 = antilinear[..., 0], antilinear[..., 1]
            stacked = torch.stack((c1, c2, c3, c4), dim=1)  # (E, 4, b_m, a_n)
        else:
            stacked = torch.stack((c1, c2), dim=1)  # (E, 2, b_m, a_n)

        result[(frequency_n, frequency_m)] = stacked

    return result


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def test_pairs_is_the_full_cross_product_of_frequencies() -> None:
    field_type_in = SO2FieldType({0: 1, 1: 2})
    field_type_out = SO2FieldType({0: 1, 2: 1})

    kernel = SO2SteerableKernel(
        field_type_in, field_type_out, cutoff=1.0, dtype=torch.float64
    )

    assert set(kernel.pairs) == {(0, 0), (0, 2), (1, 0), (1, 2)}


def test_heads_are_absent_when_no_pair_needs_them() -> None:
    only_scalars = SO2SteerableKernel(
        SO2FieldType({0: 2}), SO2FieldType({0: 3}), cutoff=1.0, dtype=torch.float64
    )
    assert only_scalars.complex_head is None
    assert only_scalars.real_head is not None

    no_scalars = SO2SteerableKernel(
        SO2FieldType({1: 2}), SO2FieldType({2: 1}), cutoff=1.0, dtype=torch.float64
    )
    assert no_scalars.complex_head is not None
    assert no_scalars.real_head is None


def test_rejects_a_non_field_type() -> None:
    with pytest.raises(TypeError):
        SO2SteerableKernel({0: 1}, SO2FieldType({0: 1}), cutoff=1.0)  # type: ignore[arg-type]


def test_radial_basis_rejects_bad_cutoff_or_basis_count() -> None:
    with pytest.raises(ValueError):
        SO2SteerableKernel(SO2FieldType({0: 1}), SO2FieldType({0: 1}), cutoff=0.0)

    with pytest.raises(ValueError):
        SO2SteerableKernel(
            SO2FieldType({0: 1}), SO2FieldType({0: 1}), cutoff=1.0, n_radial_basis=0
        )


# --------------------------------------------------------------------------
# Master correctness test: fast path vs. the dense Table 8.3 reference
# --------------------------------------------------------------------------


def test_forward_general_matches_the_dense_reference_kernel() -> None:
    field_type_in = SO2FieldType({0: 2, 1: 2})
    field_type_out = SO2FieldType({0: 1, 1: 1, 2: 1})

    torch.manual_seed(0)
    kernel = SO2SteerableKernel(
        field_type_in,
        field_type_out,
        cutoff=1.0,
        n_radial_basis=6,
        radial_hidden_features=(8,),
        bias=False,
        dtype=torch.float64,
    )

    distances = torch.tensor([0.1, 0.4, 0.7, 0.95], dtype=torch.float64)
    polar_angles = torch.tensor([0.0, 0.6, 2.1, -1.3], dtype=torch.float64)
    transport_angles = torch.tensor([0.3, -0.9, 1.7, 0.2], dtype=torch.float64)
    edges = _synthetic_edges(
        distances, polar_angles, transport_angles, max_frequency=3
    )

    generator = torch.Generator().manual_seed(1)
    x = _random_field(field_type_in, (edges.n_edges,), generator=generator)

    via_fast = kernel._forward_general(x, edges).to_flat()

    # Independently recover the coefficients and transported inputs, then
    # combine them via the dense literal-Table-8.3 assembly.
    dense_coefficients = _extract_dense_radial_coefficients(kernel, edges)
    dense_matrix = dense_field_type_steerable_kernel(
        field_type_in, field_type_out, polar_angles, dense_coefficients
    )  # (E, out_dim, in_dim)

    transported_flat = torch.cat(
        [
            x[0] if frequency == 0 else so2.as_real(
                edges.transport_harmonic(frequency).unsqueeze(-1) * so2.as_complex(x[frequency])
            ).flatten(-2)
            for frequency in field_type_in.frequencies
        ],
        dim=-1,
    )

    via_dense = torch.einsum("eoi,ei->eo", dense_matrix, transported_flat)

    assert torch.allclose(via_fast, via_dense, atol=1e-8)


def test_forward_general_matches_the_dense_reference_with_multiplicities() -> None:
    """Same check, with multiplicities > 1 to exercise the matrix-valued
    (not just scalar) radial coefficients."""
    field_type_in = SO2FieldType({0: 3, 2: 2})
    field_type_out = SO2FieldType({1: 3, 2: 1})

    torch.manual_seed(2)
    kernel = SO2SteerableKernel(
        field_type_in,
        field_type_out,
        cutoff=2.0,
        n_radial_basis=5,
        radial_hidden_features=(),
        bias=False,
        dtype=torch.float64,
    )

    distances = torch.tensor([0.2, 1.1, 1.8], dtype=torch.float64)
    polar_angles = torch.tensor([0.5, -2.0, 1.0], dtype=torch.float64)
    transport_angles = torch.tensor([-0.4, 0.9, 2.5], dtype=torch.float64)
    edges = _synthetic_edges(
        distances, polar_angles, transport_angles, max_frequency=4
    )

    generator = torch.Generator().manual_seed(3)
    x = _random_field(field_type_in, (edges.n_edges,), generator=generator)

    via_fast = kernel._forward_general(x, edges).to_flat()

    dense_coefficients = _extract_dense_radial_coefficients(kernel, edges)
    dense_matrix = dense_field_type_steerable_kernel(
        field_type_in, field_type_out, polar_angles, dense_coefficients
    )

    transported_flat = torch.cat(
        [
            x[0] if frequency == 0 else so2.as_real(
                edges.transport_harmonic(frequency).unsqueeze(-1) * so2.as_complex(x[frequency])
            ).flatten(-2)
            for frequency in field_type_in.frequencies
        ],
        dim=-1,
    )
    via_dense = torch.einsum("eoi,ei->eo", dense_matrix, transported_flat)

    assert torch.allclose(via_fast, via_dense, atol=1e-8)


# --------------------------------------------------------------------------
# Behavior at the origin
#
# The kernel no longer substitutes a separately-learned map at self-loop
# edges. Its value at r = 0 is the genuine limit of the same formula used
# everywhere else, and these tests pin the two properties that makes true:
# every angular mode vanishes there at its own order, and what survives is
# exactly an SO(2) intertwiner.
# --------------------------------------------------------------------------


def _at_distance(field_type, kernel, distance: float, n_edges: int = 4):
    """Evaluate `kernel` on `n_edges` edges all at radius `distance`, with
    assorted polar angles so any surviving angular dependence shows up."""
    edges = _synthetic_edges(
        distances=torch.full((n_edges,), distance, dtype=torch.float64),
        polar_angles=torch.linspace(0.0, 2.0, n_edges, dtype=torch.float64),
        transport_angles=torch.zeros(n_edges, dtype=torch.float64),
        # The kernel's own requirement: it reads harmonics up to
        # in.max_frequency + out.max_frequency (the antilinear term's m + n).
        max_frequency=kernel.field_type_in.max_frequency
        + kernel.field_type_out.max_frequency,
    )
    generator = torch.Generator().manual_seed(5)
    x = _random_field(field_type, (n_edges,), generator=generator)
    return x, edges, kernel(x, edges)


def test_kernel_value_at_the_origin_is_independent_of_polar_angle() -> None:
    """At r = 0 the polar angle is meaningless, so a well-defined kernel
    cannot depend on it. Before the r^|k| factors it did."""
    field_type = SO2FieldType({0: 2, 1: 2, 2: 1})
    torch.manual_seed(4)
    kernel = SO2SteerableKernel(
        field_type, field_type, cutoff=1.0, bias=True, dtype=torch.float64
    )

    # The same input at every edge, but different polar angles.
    generator = torch.Generator().manual_seed(11)
    single = _random_field(field_type, (1,), generator=generator)
    n_edges = 5
    x = SO2FeatureField(
        field_type,
        {
            frequency: single[frequency].expand(n_edges, *single[frequency].shape[1:])
            for frequency in field_type.frequencies
        },
    )
    edges = _synthetic_edges(
        distances=torch.zeros(n_edges, dtype=torch.float64),
        polar_angles=torch.linspace(-3.0, 3.0, n_edges, dtype=torch.float64),
        transport_angles=torch.zeros(n_edges, dtype=torch.float64),
        max_frequency=4,
    )

    output = kernel(x, edges).to_flat()

    assert torch.allclose(output, output[:1].expand_as(output), atol=ATOL)


def test_cross_frequency_blocks_vanish_at_the_origin() -> None:
    """Hom(rho_n, rho_m) = 0 for n != m, so every cross-frequency block must
    be exactly zero at r = 0 -- not merely small."""
    field_type_in = SO2FieldType({1: 1})
    field_type_out = SO2FieldType({0: 1, 2: 1})  # both differ from n = 1

    torch.manual_seed(12)
    kernel = SO2SteerableKernel(
        field_type_in, field_type_out, cutoff=1.0, bias=True, dtype=torch.float64
    )

    _, _, output = _at_distance(field_type_in, kernel, 0.0)

    assert torch.allclose(output.to_flat(), torch.zeros_like(output.to_flat()), atol=ATOL)


def test_the_origin_value_is_an_so2_intertwiner() -> None:
    """The actual constraint at r = 0: W(0) must commute with the group
    action, i.e. W(0) rho_in(g) x == rho_out(g) W(0) x for every g."""
    field_type = SO2FieldType({0: 2, 1: 2, 2: 1})
    torch.manual_seed(13)
    kernel = SO2SteerableKernel(
        field_type, field_type, cutoff=1.0, bias=True, dtype=torch.float64
    )

    n_edges = 6
    generator = torch.Generator().manual_seed(14)
    x = _random_field(field_type, (n_edges,), generator=generator)
    edges = _synthetic_edges(
        distances=torch.zeros(n_edges, dtype=torch.float64),
        polar_angles=torch.zeros(n_edges, dtype=torch.float64),
        transport_angles=torch.zeros(n_edges, dtype=torch.float64),
        max_frequency=4,
    )

    angles = torch.rand(n_edges, dtype=torch.float64) * 2 * math.pi

    rotate_then_apply = kernel(x.gauge_transform(angles), edges)
    apply_then_rotate = kernel(x, edges).gauge_transform(angles)

    assert torch.allclose(
        rotate_then_apply.to_flat(), apply_then_rotate.to_flat(), atol=ATOL
    )


def test_each_angular_mode_vanishes_at_its_own_leading_order() -> None:
    """Finite-difference the r-dependence near zero and confirm the leading
    power: a k-th angular mode must decay like r^|k|, so halving r must
    scale that block by about 2^-|k|."""
    field_type_in = SO2FieldType({1: 1})
    field_type_out = SO2FieldType({0: 1, 1: 1, 2: 1})

    torch.manual_seed(15)
    kernel = SO2SteerableKernel(
        field_type_in, field_type_out, cutoff=1.0, bias=True, dtype=torch.float64
    )

    def block_norm(distance: float, frequency_out: int) -> float:
        _, _, output = _at_distance(field_type_in, kernel, distance)
        return float(output[frequency_out].detach().abs().max())

    small, half = 1e-3, 5e-4

    # n = 1 -> m = 0: k = 1, so the ratio is ~1/2.
    ratio_to_scalar = block_norm(half, 0) / block_norm(small, 0)
    assert ratio_to_scalar == pytest.approx(0.5, rel=0.05)

    # n = 1 -> m = 2: linear k = 1 dominates the antilinear k = 3 near zero,
    # so this too approaches ~1/2.
    ratio_to_two = block_norm(half, 2) / block_norm(small, 2)
    assert ratio_to_two == pytest.approx(0.5, rel=0.05)


def test_the_kernel_is_continuous_through_the_origin() -> None:
    """The property that fails without this repair: the r -> 0 limit of the
    general formula must actually equal the value at r = 0."""
    field_type = SO2FieldType({0: 1, 1: 1})
    torch.manual_seed(16)
    kernel = SO2SteerableKernel(
        field_type, field_type, cutoff=1.0, bias=True, dtype=torch.float64
    )

    generator = torch.Generator().manual_seed(17)
    single = _random_field(field_type, (1,), generator=generator)

    def value_at(distance: float) -> torch.Tensor:
        edges = _synthetic_edges(
            distances=torch.tensor([distance], dtype=torch.float64),
            polar_angles=torch.tensor([0.7], dtype=torch.float64),
            transport_angles=torch.tensor([0.0], dtype=torch.float64),
            max_frequency=2,
        )
        return kernel(single, edges).to_flat()

    at_origin = value_at(0.0)
    approaching = value_at(1e-7)

    assert torch.allclose(approaching, at_origin, atol=1e-6)


def test_the_result_does_not_depend_on_index_identity() -> None:
    """Coincident points are handled by distance, not by
    target_index == source_index -- so two distinct indices denoting the
    same point (which some grids produce) now behave correctly."""
    field_type = SO2FieldType({0: 1, 1: 1})
    torch.manual_seed(18)
    kernel = SO2SteerableKernel(
        field_type, field_type, cutoff=1.0, bias=True, dtype=torch.float64
    )

    generator = torch.Generator().manual_seed(19)
    x = _random_field(field_type, (2,), generator=generator)

    common = {
        "distances": torch.zeros(2, dtype=torch.float64),
        "polar_angles": torch.zeros(2, dtype=torch.float64),
        "transport_angles": torch.zeros(2, dtype=torch.float64),
        "max_frequency": 2,
    }
    flagged = kernel(x, _synthetic_edges(self_loop=torch.ones(2, dtype=torch.bool), **common))
    unflagged = kernel(x, _synthetic_edges(self_loop=torch.zeros(2, dtype=torch.bool), **common))

    assert torch.allclose(flagged.to_flat(), unflagged.to_flat(), atol=ATOL)


def test_gradients_are_finite_at_the_origin() -> None:
    """r = 0 is where a naive r^k parameterization produces nan gradients."""
    field_type = SO2FieldType({0: 1, 1: 1, 2: 1})
    torch.manual_seed(20)
    kernel = SO2SteerableKernel(
        field_type, field_type, cutoff=1.0, bias=True, dtype=torch.float64
    )

    _, _, output = _at_distance(field_type, kernel, 0.0)
    output.to_flat().square().sum().backward()

    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in kernel.parameters()
    )


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_forward_rejects_a_mismatched_input_field_type() -> None:
    kernel = SO2SteerableKernel(
        SO2FieldType({0: 1}), SO2FieldType({0: 1}), cutoff=1.0, dtype=torch.float64
    )
    edges = _synthetic_edges(
        torch.zeros(1, dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64),
        self_loop=torch.ones(1, dtype=torch.bool),
        max_frequency=0,
    )
    wrong_field = SO2FeatureField.from_flat(
        SO2FieldType({0: 2}), torch.zeros(1, 2, dtype=torch.float64)
    )

    with pytest.raises(ValueError):
        kernel(wrong_field, edges)


def test_forward_rejects_a_mismatched_batch_shape() -> None:
    field_type = SO2FieldType({0: 1})
    kernel = SO2SteerableKernel(field_type, field_type, cutoff=1.0, dtype=torch.float64)
    edges = _synthetic_edges(
        torch.zeros(3, dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64),
        self_loop=torch.ones(3, dtype=torch.bool),
        max_frequency=0,
    )
    x = SO2FeatureField.from_flat(field_type, torch.zeros(5, 1, dtype=torch.float64))

    with pytest.raises(ValueError):
        kernel(x, edges)


def test_forward_rejects_insufficiently_tabulated_edges() -> None:
    field_type = SO2FieldType({2: 1})
    kernel = SO2SteerableKernel(field_type, field_type, cutoff=1.0, dtype=torch.float64)

    # Needs order 2 + 2 = 4, but only orders up to 2 are tabulated.
    edges = _synthetic_edges(
        torch.full((2,), 0.3, dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
        self_loop=torch.tensor([True, False]),
        max_frequency=2,
    )
    x = SO2FeatureField.from_flat(field_type, torch.zeros(2, 2, dtype=torch.float64))

    with pytest.raises(ValueError, match="max_frequency"):
        kernel(x, edges)


def test_forward_rejects_a_distance_beyond_the_cutoff() -> None:
    field_type = SO2FieldType({0: 1})
    kernel = SO2SteerableKernel(field_type, field_type, cutoff=0.5, dtype=torch.float64)

    edges = _synthetic_edges(
        torch.tensor([0.1, 0.9], dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
        self_loop=torch.tensor([True, False]),
        max_frequency=0,
    )
    x = SO2FeatureField.from_flat(field_type, torch.zeros(2, 1, dtype=torch.float64))

    with pytest.raises(ValueError, match="cutoff"):
        kernel(x, edges)


def test_cutoff_check_is_skipped_on_a_previously_validated_edges_object() -> None:
    """The cutoff check requires a blocking distances.max().item() sync, so
    forward() only pays it once per distinct EdgeGeometry object (see
    SO2SteerableKernel.__init__'s comment) -- exercised here on CPU, where
    there is no sync cost to measure directly, by observing the resulting
    (deliberately accepted) behavior instead: mutating a tensor a
    *previously validated* EdgeGeometry object wraps, in place, after that
    validation already ran, is not caught. A genuinely new, never-before-seen
    EdgeGeometry with the same bad distance is still caught -- the check
    itself was not disabled, only skipped for objects already seen."""
    field_type = SO2FieldType({0: 1})
    kernel = SO2SteerableKernel(field_type, field_type, cutoff=0.5, dtype=torch.float64)
    x = SO2FeatureField.from_flat(field_type, torch.zeros(2, 1, dtype=torch.float64))

    edges = _synthetic_edges(
        torch.tensor([0.1, 0.2], dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
        self_loop=torch.tensor([True, False]),
        max_frequency=0,
    )

    kernel(x, edges)  # validates and caches id(edges)
    assert id(edges) in kernel._validated_edge_geometry_ids

    edges.distances.copy_(torch.tensor([0.1, 0.9], dtype=torch.float64))
    kernel(x, edges)  # not re-validated: same object, already cached

    fresh_edges = _synthetic_edges(
        torch.tensor([0.1, 0.9], dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
        self_loop=torch.tensor([True, False]),
        max_frequency=0,
    )
    with pytest.raises(ValueError, match="cutoff"):
        kernel(x, fresh_edges)


# --------------------------------------------------------------------------
# Autograd
# --------------------------------------------------------------------------


def test_gradients_flow_to_every_parameter() -> None:
    field_type_in = SO2FieldType({0: 1, 1: 1})
    field_type_out = SO2FieldType({0: 1, 2: 1})

    kernel = SO2SteerableKernel(
        field_type_in, field_type_out, cutoff=1.0, bias=True, dtype=torch.float64
    )

    edges = _synthetic_edges(
        torch.tensor([0.0, 0.4], dtype=torch.float64),
        torch.tensor([0.0, 1.1], dtype=torch.float64),
        torch.tensor([0.0, 0.2], dtype=torch.float64),
        self_loop=torch.tensor([True, False]),
        max_frequency=3,
    )

    generator = torch.Generator().manual_seed(10)
    x = _random_field(field_type_in, (2,), generator=generator)

    output = kernel(x, edges)
    loss = sum(tensor.pow(2).sum() for tensor in output.coefficients.values())
    loss.backward()

    for name, parameter in kernel.named_parameters():
        assert parameter.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient at {name}"


# --------------------------------------------------------------------------
# End-to-end gauge equivariance on a real discretization
# --------------------------------------------------------------------------


@pytest.mark.parametrize("radial_basis", ["gaussian"])
def test_end_to_end_gauge_equivariance_on_a_real_discretization(
    radial_basis: str,
) -> None:
    """Regauging the whole mesh must leave the aggregated-nowhere messages
    consistent: gathering source features through gauge_transform and
    recomputing the edge geometry under the same regauging must reproduce
    the *same physical* messages, just expressed in the new target gauges.

    Parameterized over the radial basis: equivariance rides entirely on
    the angular structure and transport, so it must hold identically no
    matter how C(r) is built. The supplement ships only the Gaussian basis
    (the one every reported arm uses), so that is the single case here.
    """
    grid = build_icosphere_grid(subdivisions=1, dtype=torch.float64)
    frames = build_tangent_frames(grid.nodes)
    coverage_radius = 0.6 * math.pi

    field_type_in = SO2FieldType({0: 1, 1: 1})
    field_type_out = SO2FieldType({0: 1, 1: 1, 2: 1})

    torch.manual_seed(11)
    kernel = SO2SteerableKernel(
        field_type_in,
        field_type_out,
        cutoff=coverage_radius,
        n_radial_basis=6,
        radial_hidden_features=(8,),
        radial_basis=radial_basis,  # type: ignore[arg-type]
        bias=True,
        dtype=torch.float64,
    )

    max_frequency = field_type_in.max_frequency + field_type_out.max_frequency
    edges = compute_edge_geometry(
        grid, frames, coverage_radius, max_frequency=max_frequency
    )

    generator = torch.Generator().manual_seed(12)
    node_features = _random_field(field_type_in, (grid.n_nodes,), generator=generator)

    def gather(field: SO2FeatureField, index: torch.Tensor) -> SO2FeatureField:
        return SO2FeatureField(
            field.field_type,
            {frequency: tensor[index] for frequency, tensor in field.coefficients.items()},
        )

    source_features = gather(node_features, edges.source_index)
    original_output = kernel(source_features, edges).to_flat()

    gauge_angles = torch.rand(grid.n_nodes, dtype=torch.float64, generator=generator) * (
        2 * math.pi
    )

    regauged_edges = compute_edge_geometry(
        grid,
        rotate_tangent_frames(frames, gauge_angles),
        coverage_radius,
        max_frequency=max_frequency,
    )
    regauged_node_features = node_features.gauge_transform(gauge_angles)
    regauged_source_features = gather(regauged_node_features, edges.source_index)

    regauged_output = kernel(regauged_source_features, regauged_edges).to_flat()

    # The message at each edge is a physical object expressed in the
    # target node's (now regauged) gauge, so it must transform by the same
    # gauge_transform law as any other frequency-typed field.
    expected = SO2FeatureField.from_flat(field_type_out, original_output).gauge_transform(
        gauge_angles[edges.target_index]
    ).to_flat()

    assert torch.allclose(regauged_output, expected, atol=1e-7)


def _single_edge_geometry(radius: float):
    """Edge geometry for two points separated by ``radius`` along a meridian."""
    nodes = torch.tensor(
        [[0.0, 0.0, 1.0], [math.sin(radius), 0.0, math.cos(radius)]],
        dtype=torch.float64,
    )
    grid = SphericalGrid(
        nodes=nodes,
        quadrature_weights=torch.full((2,), 2 * math.pi, dtype=torch.float64),
    )
    return compute_edge_geometry(
        grid, build_tangent_frames(nodes), radius * 1.5, max_frequency=8
    )


def _unit_source(field_type: SO2FieldType, n_edges: int) -> SO2FeatureField:
    """A fixed unit-ish source field, so norms across radii are comparable."""
    flat = torch.ones(n_edges, field_type.dimension, dtype=torch.float64)
    return SO2FeatureField.from_flat(field_type, flat)

# ==========================================================================
# Radial powers applied per role rather than per coefficient column
# ==========================================================================


def test_radial_powers_table_has_one_column_per_distinct_power() -> None:
    """The point of the change: a full-spectrum kernel has thousands of
    coefficient columns but only a handful of distinct powers, and it is
    the expanded per-column form that dominates memory."""
    field_type = SO2FieldType({0: 4, 1: 4, 2: 4})
    kernel = SO2SteerableKernel(field_type, field_type, cutoff=1.0, dtype=torch.float64)

    powers = kernel._radial_powers(torch.rand(7, dtype=torch.float64))

    assert powers.shape == (7, kernel._max_column_power + 1)
    assert kernel._max_column_power == 4  # |m-n| up to 2, m+n up to 4
    # One column per distinct power, whatever the width happens to be.
    assert powers.shape[1] == len(set(kernel._column_powers.tolist()))


def test_the_compact_power_table_is_far_smaller_at_benchmark_width() -> None:
    """The saving is width-dependent and only matters at real width: the
    benchmark's full-spectrum hidden type has thousands of coefficient
    columns and still just five distinct powers."""
    field_type = SO2FieldType({0: 16, 1: 16, 2: 16})
    kernel = SO2SteerableKernel(field_type, field_type, cutoff=1.0, dtype=torch.float64)

    distinct = kernel._max_column_power + 1
    columns = kernel._column_powers.numel()

    assert distinct == 5
    assert columns > 6_000
    assert columns / distinct > 1_000


def test_radial_powers_are_successive_powers_of_the_normalized_radius() -> None:
    kernel = SO2SteerableKernel(
        SO2FieldType({0: 1, 1: 1, 2: 1}), SO2FieldType({0: 1, 1: 1, 2: 1}),
        cutoff=0.8, dtype=torch.float64,
    )
    distances = torch.tensor([0.0, 0.2, 0.5, 0.8], dtype=torch.float64)

    powers = kernel._radial_powers(distances)

    normalized = distances / 0.8
    for exponent in range(kernel._max_column_power + 1):
        assert torch.allclose(powers[:, exponent], normalized**exponent)


def test_every_role_records_the_power_its_columns_carry() -> None:
    """Applying one scalar per role is only valid because every column in
    that role's slice carries the same power; this checks that the two
    tables agree."""
    field_type = SO2FieldType({0: 2, 1: 2, 2: 2})
    kernel = SO2SteerableKernel(field_type, field_type, cutoff=1.0, dtype=torch.float64)

    for (frequency_n, frequency_m, role), span in kernel._pair_ranges.items():
        recorded = kernel._pair_powers[(frequency_n, frequency_m, role)]
        columns = kernel._column_powers[span]

        assert bool((columns == recorded).all()), (
            f"role {(frequency_n, frequency_m, role)} mixes powers "
            f"{sorted(set(columns.tolist()))}"
        )
        expected = (
            abs(frequency_m - frequency_n) if role == "linear"
            else frequency_m + frequency_n
        )
        assert recorded == expected


@pytest.mark.parametrize(
    ("frequency_in", "frequency_out", "expected_order"),
    [
        (0, 1, 1),  # linear |m - n| = 1
        (0, 2, 2),  # linear |m - n| = 2
        (1, 0, 1),  # linear |m - n| = 1
        (2, 0, 2),  # linear |m - n| = 2
        (1, 2, 1),  # linear |m - n| = 1 dominates the m+n = 3 antilinear term
    ],
)
def test_a_cross_frequency_block_vanishes_at_its_documented_order(
    frequency_in: int, frequency_out: int, expected_order: int
) -> None:
    """The regularity condition the powers exist to impose, checked through
    the *contracted* output now that the scalar is applied there: halving
    the radius must scale the block by 2^-order."""
    torch.manual_seed(3)
    field_type_in = SO2FieldType({frequency_in: 2})
    field_type_out = SO2FieldType({frequency_out: 2})
    kernel = SO2SteerableKernel(
        field_type_in, field_type_out, cutoff=1.0, dtype=torch.float64
    )

    norms = []
    for radius in (0.02, 0.01):
        edges = _single_edge_geometry(radius)
        source = _unit_source(field_type_in, edges.n_edges)
        norms.append(float(kernel(source, edges).to_flat().norm()))

    ratio = norms[0] / norms[1]
    assert ratio == pytest.approx(2.0**expected_order, rel=0.05), (
        f"block {frequency_in}->{frequency_out} scales as {math.log2(ratio):.2f}, "
        f"expected {expected_order}"
    )


def test_the_scalar_block_does_not_vanish_at_the_origin() -> None:
    """The contrast case: |m - n| = 0 carries power 0, so the rho_0 -> rho_0
    block survives to r = 0 and leaves an intertwiner there."""
    torch.manual_seed(4)
    kernel = SO2SteerableKernel(
        SO2FieldType({0: 2}), SO2FieldType({0: 2}), cutoff=1.0, dtype=torch.float64
    )

    norms = [
        float(
            kernel(
                _unit_source(SO2FieldType({0: 2}), _single_edge_geometry(r).n_edges),
                _single_edge_geometry(r),
            ).to_flat().norm()
        )
        for r in (0.02, 0.01)
    ]

    assert norms[1] > 0.5 * norms[0]  # no vanishing


# --------------------------------------------------------------------------
# The radial-basis selector
# --------------------------------------------------------------------------


def test_rejects_an_unknown_radial_basis() -> None:
    with pytest.raises(ValueError, match="radial_basis"):
        SO2SteerableKernel(
            SO2FieldType({0: 1}),
            SO2FieldType({0: 1}),
            cutoff=1.0,
            radial_basis="chebyshev",  # type: ignore[arg-type]
        )


# --------------------------------------------------------------------------
# Coefficient layout
# --------------------------------------------------------------------------


def test_coefficient_chunks_partition_exactly_as_the_ranges() -> None:
    """The split order/widths and the recorded ranges are two descriptions
    of one layout; forward uses the first, tests and the dense reference
    read the second. Divergence would silently wire a role to the wrong
    coefficients."""
    kernel = SO2SteerableKernel(
        SO2FieldType({0: 2, 1: 2, 2: 1}),
        SO2FieldType({0: 1, 1: 2, 2: 2}),
        cutoff=1.0,
        dtype=torch.float64,
    )

    offset = 0
    for key, width in zip(
        kernel._coefficient_order, kernel._coefficient_widths, strict=True
    ):
        recorded = kernel._pair_ranges[key]
        assert recorded == slice(offset, offset + width)
        offset += width
    assert kernel.complex_head is not None
    assert offset == kernel.complex_head.out_features
