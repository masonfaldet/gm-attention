"""An independent oracle for ``gm_attention.nn.fixed_grid.attention_fixed_grid``.

``tests/nn/test_fixed_grid_attention.py`` checks the stencil path against the
graph path (``SO2Attention.forward``). The two share the steerable kernel's
complex-arithmetic forward, the intertwiners' forward and the softmax
arithmetic, so a defect in any of those cancels in that comparison. This
file recomputes the layer's output at a handful of target nodes from the
definition in the ``SO2Attention`` docstring alone,

    s_ij     = <W_Q f_i, rho_attn(gamma_ij) W_K f_j> / sqrt(dim rho_attn)
    alpha_ij = c(r_ij) w_ij exp(s_ij) / sum_k c(r_ik) w_ik exp(s_ik)
    y_i      = sum_j alpha_ij V(r_ij, phi_ij) rho_in(gamma_ij) f_j  (+ W_self f_i)

with an explicit Python loop over the target's neighbours. ``W_Q``, ``W_K``
and ``W_self`` are assembled from the layers' parameters by the dense
reference ``tests.support.intertwiners.dense_field_type_intertwiner``;
``V(r, phi)`` by ``tests.support.bases.dense_field_type_steerable_kernel``
from radial coefficients recomputed here from the kernel's parameters (the
Gaussian embedding, the trunk's ``nn.Linear`` / ``SiLU`` layers, the two
readout heads, the packed ``(n, m, role)`` layout and the regularity factors
``(r / cutoff)^|k|`` all written out); ``rho(gamma)`` is a block-diagonal of
hand-written ``2 x 2`` rotations on the packed layout; the window is written
out. Nothing here calls ``SO2Attention.forward`` / ``_attend``,
``SO2SteerableKernel.forward`` / ``_forward_general``, ``SO2Linear.forward``
or ``steerable_kernel_matrices``.

The neighbour geometry comes from two independent sources, one test each:
the stencil's own per-slot tables (``distances``, ``polar_angles``,
``transport_angles``, ``tangent_quadrature_weights``, with the slot's
``(source_row, offset)`` resolved to a source node by hand), and the
target's edges of the full edge set ``discretization.edges_within(cutoff)``,
which checks the fold and the gather as well as the arithmetic.

Cases: the task's types (``in {0: 2, 1: 2, 2: 1}``, ``attn {0: 1, 1: 1,
2: 1}``, ``out {0: 2, 1: 1, 2: 1}``) under both windows with the pointwise
self-interaction on and off; attention and output types without frequency
``0``; and types with a frequency present on one side only (``attn`` and
``out`` carry frequency ``2``, which the input lacks). Everything in float64
on a Gauss-Legendre ``8 x 16`` grid at cutoff ``0.7``, six random targets,
``atol = 1e-9``.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch
from torch import Tensor, nn

from gm_attention.discretization.edge_geometry import EdgeGeometry
from gm_attention.discretization.stencil import (
    TensorGridStencil,
    build_tensor_grid_discretization,
    build_tensor_grid_stencil,
)
from gm_attention.nn.attention import SO2Attention
from gm_attention.nn.fixed_grid import attention_fixed_grid
from gm_attention.nn.kernels import SO2SteerableKernel
from gm_attention.nn.linear import SO2Linear
from gm_attention.representations.field_type import SO2FeatureField, SO2FieldType
from tests.support.bases import dense_field_type_steerable_kernel
from tests.support.intertwiners import dense_field_type_intertwiner

N_LATITUDE, N_LONGITUDE = 8, 16
CUTOFF = 0.7
ATOL = 1e-9
N_TARGETS = 6
DTYPE = torch.float64

# (field_type_in, field_type_attn, field_type_out, window, self_interaction)
CASES: dict[str, tuple[SO2FieldType, SO2FieldType, SO2FieldType, str, str]] = {
    "task-types/hard/self-linear": (
        SO2FieldType({0: 2, 1: 2, 2: 1}),
        SO2FieldType({0: 1, 1: 1, 2: 1}),
        SO2FieldType({0: 2, 1: 1, 2: 1}),
        "hard",
        "linear",
    ),
    "task-types/quartic/no-self": (
        SO2FieldType({0: 2, 1: 2, 2: 1}),
        SO2FieldType({0: 1, 1: 1, 2: 1}),
        SO2FieldType({0: 2, 1: 1, 2: 1}),
        "quartic",
        "none",
    ),
    "no-frequency-0-on-attn-and-out/quartic": (
        SO2FieldType({0: 2, 1: 2, 2: 1}),
        SO2FieldType({1: 1, 2: 1}),
        SO2FieldType({1: 1, 2: 1}),
        "quartic",
        "none",
    ),
    "one-sided-frequencies/hard/self-linear": (
        SO2FieldType({0: 2, 1: 1}),
        SO2FieldType({0: 1, 2: 1}),
        SO2FieldType({1: 1, 2: 1}),
        "hard",
        "linear",
    ),
}


@dataclasses.dataclass(frozen=True)
class _Neighbour:
    """One edge ``target <- source`` as the oracle reads it."""

    source: int
    distance: float
    polar_angle: float
    transport_angle: float
    quadrature_weight: float


# --------------------------------------------------------------------------
# Fixtures: the grid, the stencil, a module and an input per case


def _geometry():
    discretization = build_tensor_grid_discretization(
        "legendre-gauss", N_LATITUDE, N_LONGITUDE, coverage_radius=CUTOFF, dtype=DTYPE
    )
    stencil = build_tensor_grid_stencil(discretization, CUTOFF, n_latitude=N_LATITUDE, n_longitude=N_LONGITUDE)
    return discretization, stencil


def _module(case: str, seed: int) -> SO2Attention:
    field_type_in, field_type_attn, field_type_out, window, self_interaction = CASES[case]
    torch.manual_seed(seed)
    return SO2Attention(
        field_type_in,
        field_type_attn,
        field_type_out,
        cutoff=CUTOFF,
        window=window,  # type: ignore[arg-type]
        self_interaction=self_interaction,  # type: ignore[arg-type]
        dtype=DTYPE,
    )


def _input(field_type: SO2FieldType, n_nodes: int, seed: int) -> SO2FeatureField:
    generator = torch.Generator().manual_seed(seed)
    return SO2FeatureField.from_flat(field_type, torch.randn(n_nodes, field_type.dimension, dtype=DTYPE, generator=generator))


def _targets(n_nodes: int, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(n_nodes, generator=generator)[:N_TARGETS].tolist()


# --------------------------------------------------------------------------
# Neighbour geometry from the two sources


def _neighbours_from_stencil(stencil: TensorGridStencil, target: int) -> list[_Neighbour]:
    """The target's slots, the source node resolved by hand from
    ``(source_row, (wo + offset) mod n_longitude)``."""
    n_longitude = stencil.n_longitude
    row, longitude = divmod(target, n_longitude)
    slots = stencil.slots(row)
    geometry = stencil.geometry
    neighbours = []
    for slot in range(slots.start, slots.stop):
        source = int(stencil.source_row[slot]) * n_longitude + (longitude + int(stencil.offset[slot])) % n_longitude
        neighbours.append(
            _Neighbour(
                source=source,
                distance=float(geometry.distances[slot]),
                polar_angle=float(geometry.polar_angles[slot]),
                transport_angle=float(geometry.transport_angles[slot]),
                quadrature_weight=float(geometry.tangent_quadrature_weights[slot]),
            )
        )
    return neighbours


def _neighbours_from_edges(edges: EdgeGeometry, target: int) -> list[_Neighbour]:
    """The target's edges of the full (unfolded) edge set."""
    first, last = int(edges.row_ptr[target]), int(edges.row_ptr[target + 1])
    neighbours = []
    for edge in range(first, last):
        assert int(edges.target_index[edge]) == target
        neighbours.append(
            _Neighbour(
                source=int(edges.source_index[edge]),
                distance=float(edges.distances[edge]),
                polar_angle=float(edges.polar_angles[edge]),
                transport_angle=float(edges.transport_angles[edge]),
                quadrature_weight=float(edges.tangent_quadrature_weights[edge]),
            )
        )
    return neighbours


# --------------------------------------------------------------------------
# The dense pieces: rho(gamma), the intertwiners, the steerable value kernel


def _representation(field_type: SO2FieldType, angle: float) -> Tensor:  # (dim, dim)
    """``rho(r_angle)`` on the packed layout: per frequency ``I_a (x) rho_n``,
    with ``rho_0 = 1`` and ``rho_n`` the plane rotation by ``n * angle``."""
    blocks = []
    for frequency, multiplicity in field_type.items():
        if frequency == 0:
            blocks.append(torch.eye(multiplicity, dtype=DTYPE))
        else:
            cosine, sine = math.cos(frequency * angle), math.sin(frequency * angle)
            rotation = torch.tensor([[cosine, -sine], [sine, cosine]], dtype=DTYPE)
            blocks.append(torch.kron(torch.eye(multiplicity, dtype=DTYPE), rotation))
    return torch.block_diag(*blocks)


def _intertwiner(layer: SO2Linear) -> tuple[Tensor, Tensor]:  # (d_out, d_in), (d_out,)
    """The layer's weights assembled by the dense reference, plus its bias as
    a packed vector (zero outside the frequency-0 channels)."""
    weights = {}
    for frequency in layer.common_frequencies:
        weight = layer.weight[str(frequency)].detach()
        weights[frequency] = weight if frequency == 0 else torch.complex(weight[..., 0], weight[..., 1])
    matrix = dense_field_type_intertwiner(
        layer.field_type_in, layer.field_type_out, weights, dtype=DTYPE, device=torch.device("cpu")
    )
    bias = torch.zeros(layer.field_type_out.dimension, dtype=DTYPE)
    if layer.bias is not None:
        bias[layer.field_type_out.slices()[0]] = layer.bias.detach()
    return matrix, bias


def _coefficient_layout(field_type_in: SO2FieldType, field_type_out: SO2FieldType) -> dict[tuple[int, int, str], slice]:
    """The complex head's output layout as documented in ``SO2SteerableKernel``:
    consecutive ``(n, m, role)`` ranges of ``2 b_m a_n`` reals, ``n`` outer
    over the input frequencies, ``m`` inner over the output frequencies, the
    ``(0, 0)`` pair skipped, the ``"antilinear"`` range right after the
    ``"linear"`` one for ``n, m >= 1``."""
    layout: dict[tuple[int, int, str], slice] = {}
    offset = 0
    for frequency_n in field_type_in.frequencies:
        for frequency_m in field_type_out.frequencies:
            if frequency_n == 0 and frequency_m == 0:
                continue
            width = 2 * field_type_out.multiplicity(frequency_m) * field_type_in.multiplicity(frequency_n)
            layout[(frequency_n, frequency_m, "linear")] = slice(offset, offset + width)
            offset += width
            if frequency_n >= 1 and frequency_m >= 1:
                layout[(frequency_n, frequency_m, "antilinear")] = slice(offset, offset + width)
                offset += width
    return layout


def _radial_features(kernel: SO2SteerableKernel, distance: float) -> Tensor:  # (trunk_width,)
    """The radial MLP's hidden features at one distance: the Gaussian
    embedding of ``s = (r / cutoff)^2`` and the trunk, layer by layer."""
    basis = kernel.radial_basis
    s = (distance / basis.cutoff) ** 2
    hidden = torch.exp(-0.5 * ((s - basis.centers.detach()) / basis.width) ** 2)  # (n_basis,)
    for layer in kernel.radial_trunk:
        if isinstance(layer, nn.Linear):
            hidden = layer.weight.detach() @ hidden
            if layer.bias is not None:
                hidden = hidden + layer.bias.detach()
        elif isinstance(layer, nn.SiLU):
            hidden = hidden * torch.sigmoid(hidden)
        else:
            raise TypeError(f"Unexpected trunk layer {type(layer).__name__}.")
    return hidden


def _value_matrix(kernel: SO2SteerableKernel, distance: float, polar_angle: float) -> Tensor:  # (d_out, d_in)
    """``V(r, phi)`` on the packed layout: the radial coefficients
    ``C_i^{m<-n}(r)`` read off the two readout heads, each phase-carrying
    block scaled by ``(r / cutoff)^|k|`` (``|k| = |m - n|`` linear, ``m + n``
    antilinear), assembled with the dense angular basis."""
    field_type_in, field_type_out = kernel.field_type_in, kernel.field_type_out
    hidden = _radial_features(kernel, distance)
    normalized = distance / kernel.radial_basis.cutoff

    def readout(head: nn.Linear | None) -> Tensor | None:
        if head is None:
            return None
        out = head.weight.detach() @ hidden
        return out if head.bias is None else out + head.bias.detach()

    complex_coefficients = readout(kernel.complex_head)
    real_coefficients = readout(kernel.real_head)
    layout = _coefficient_layout(field_type_in, field_type_out)
    # The layout is re-derived above; the kernel's own record must agree, or
    # the comparison below would not be about the arithmetic.
    assert layout == kernel._pair_ranges

    coefficients: dict[tuple[int, int], Tensor] = {}
    for frequency_n in field_type_in.frequencies:
        a_n = field_type_in.multiplicity(frequency_n)
        for frequency_m in field_type_out.frequencies:
            b_m = field_type_out.multiplicity(frequency_m)
            if frequency_n == 0 and frequency_m == 0:
                assert real_coefficients is not None
                coefficients[(0, 0)] = real_coefficients.reshape(1, b_m, a_n)
                continue
            assert complex_coefficients is not None
            linear = complex_coefficients[layout[(frequency_n, frequency_m, "linear")]].reshape(b_m, a_n, 2)
            linear = linear * normalized ** abs(frequency_m - frequency_n)
            stacked = [linear[..., 0], linear[..., 1]]
            if frequency_n >= 1 and frequency_m >= 1:
                antilinear = complex_coefficients[layout[(frequency_n, frequency_m, "antilinear")]].reshape(b_m, a_n, 2)
                antilinear = antilinear * normalized ** (frequency_m + frequency_n)
                stacked += [antilinear[..., 0], antilinear[..., 1]]
            coefficients[(frequency_n, frequency_m)] = torch.stack(stacked, dim=0)  # (dim, b_m, a_n)

    return dense_field_type_steerable_kernel(
        field_type_in, field_type_out, torch.tensor(polar_angle, dtype=DTYPE), coefficients
    )


def _window(distance: float, cutoff: float, kind: str) -> float:
    if kind == "hard":
        return 1.0 if distance <= cutoff + 1e-10 else 0.0
    if kind == "quartic":
        ratio = distance / cutoff
        return (1.0 - ratio * ratio) ** 2 if distance <= cutoff else 0.0
    raise ValueError(kind)


# --------------------------------------------------------------------------
# The oracle


def _oracle(module: SO2Attention, x_flat: Tensor, target: int, neighbours: list[_Neighbour]) -> Tensor:  # (d_out,)
    """``y_target`` from the definition, over the given neighbours."""
    query_matrix, query_bias = _intertwiner(module.query)
    key_matrix, key_bias = _intertwiner(module.key)
    scale = 1.0 / math.sqrt(module.field_type_attn.dimension)

    query = query_matrix @ x_flat[target] + query_bias  # (d_attn,)
    weights: list[float] = []
    messages: list[Tensor] = []
    for neighbour in neighbours:
        source = x_flat[neighbour.source]  # (d_in,)
        key = _representation(module.field_type_attn, neighbour.transport_angle) @ (key_matrix @ source + key_bias)
        score = float(query @ key) * scale
        window = _window(neighbour.distance, module.cutoff, module.window)
        weights.append(window * neighbour.quadrature_weight * math.exp(score))
        value = _value_matrix(module.value, neighbour.distance, neighbour.polar_angle)  # (d_out, d_in)
        messages.append(value @ (_representation(module.field_type_in, neighbour.transport_angle) @ source))

    normalization = sum(weights)
    assert normalization > 0.0
    output = torch.zeros(module.field_type_out.dimension, dtype=DTYPE)
    for weight, message in zip(weights, messages, strict=True):
        output = output + (weight / normalization) * message

    if module.self_linear is not None:
        self_matrix, self_bias = _intertwiner(module.self_linear)
        assert float(self_bias.abs().max()) == 0.0  # the self-interaction carries no bias
        output = output + self_matrix @ x_flat[target]
    return output


def _check_targets(case: str, neighbours_of) -> None:
    discretization, stencil = _geometry()
    field_type_in = CASES[case][0]
    module = _module(case, seed=10 + list(CASES).index(case))
    x = _input(field_type_in, stencil.n_nodes, seed=1)
    with torch.no_grad():
        actual = attention_fixed_grid(module, x, stencil).to_flat()  # (n_nodes, d_out)
        x_flat = x.to_flat()
        for target in _targets(stencil.n_nodes, seed=2):
            neighbours = neighbours_of(discretization, stencil, target)
            assert len(neighbours) >= 2, "a target with only its self-loop would not exercise the softmax"
            assert any(n.source == target for n in neighbours), "every target carries its self-loop"
            assert len({n.source for n in neighbours}) == len(neighbours), "no source is listed twice"
            expected = _oracle(module, x_flat, target, neighbours)
            difference = float((actual[target] - expected).abs().max())
            assert difference <= ATOL, f"target {target} ({divmod(target, N_LONGITUDE)}): max |diff| {difference:.3e}"


# --------------------------------------------------------------------------
# Tests


@pytest.mark.parametrize("case", list(CASES))
def test_fixed_grid_matches_the_oracle_on_the_stencil_geometry(case: str) -> None:
    """The oracle reads each slot's raw geometry and resolves the slot's
    source node by hand."""
    _check_targets(case, lambda _, stencil, target: _neighbours_from_stencil(stencil, target))


@pytest.mark.parametrize("case", list(CASES))
def test_fixed_grid_matches_the_oracle_on_the_full_edge_set(case: str) -> None:
    """The oracle reads the target's edges from the unfolded edge set, so
    the fold and the longitude gather are checked along with the arithmetic."""
    _check_targets(case, lambda discretization, _, target: _neighbours_from_edges(discretization.edges_within(CUTOFF), target))


def test_the_two_geometry_sources_name_the_same_neighbours() -> None:
    """The stencil's slots at a target and the edge set's edges at that
    target are the same multiset of (source, r, e^{i phi}, e^{i gamma}, w),
    the polar angle excepted at the self-loop where it is rounding noise.
    Angles are compared as phases: an angle of exactly pi comes out as +pi
    on one edge of a class and -pi on another (atan2's branch), the same
    rotation, and the layers read only the phases."""

    def phase(angle: float) -> complex:
        return complex(math.cos(angle), math.sin(angle))

    discretization, stencil = _geometry()
    edges = discretization.edges_within(CUTOFF)
    for target in _targets(stencil.n_nodes, seed=3):
        from_stencil = sorted(_neighbours_from_stencil(stencil, target), key=lambda n: n.source)
        from_edges = sorted(_neighbours_from_edges(edges, target), key=lambda n: n.source)
        assert [n.source for n in from_stencil] == [n.source for n in from_edges]
        for a, b in zip(from_stencil, from_edges, strict=True):
            assert abs(a.distance - b.distance) <= 1e-12
            assert abs(phase(a.transport_angle) - phase(b.transport_angle)) <= 1e-12
            assert abs(a.quadrature_weight - b.quadrature_weight) <= 1e-12
            if a.distance > 1e-12:
                assert abs(phase(a.polar_angle) - phase(b.polar_angle)) <= 1e-12
