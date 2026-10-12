"""Gauge-transformation tests for gm_attention.discretization.frames.

Covers the SO(2) rotation primitives (``planar_rotation_matrices``,
``rotate_tangent_frames``) in isolation, and the gauge-covariance law the
whole pipeline (frames -> edge geometry) must satisfy: rotating every
node's tangent frame by a per-node angle theta_i must rotate
log-coordinates and shift transport angles in a predictable way, while
leaving gauge-invariant quantities (geodesic distance, tangent-space
quadrature weight) untouched.
"""

from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from gm_attention.discretization.edge_geometry import compute_edge_geometry
from gm_attention.discretization.frames import (
    build_tangent_frames,
    planar_rotation_matrices,
    rotate_tangent_frames,
)
from gm_attention.discretization.grids.icosphere import build_icosphere_grid

ATOL = 1e-9

_angles = st.floats(
    min_value=-10 * math.pi,
    max_value=10 * math.pi,
    allow_nan=False,
    allow_infinity=False,
)


# --------------------------------------------------------------------------
# planar_rotation_matrices
# --------------------------------------------------------------------------


def test_planar_rotation_matrices_zero_angle_is_identity() -> None:
    r = planar_rotation_matrices(torch.zeros(5, dtype=torch.float64))
    expected = torch.eye(2, dtype=torch.float64).expand(5, -1, -1)
    assert torch.allclose(r, expected, atol=ATOL)


def test_planar_rotation_matrices_are_orthonormal() -> None:
    theta = torch.linspace(0, 4 * math.pi, 37, dtype=torch.float64)
    r = planar_rotation_matrices(theta)

    gram = torch.matmul(r.transpose(-1, -2), r)
    identity = torch.eye(2, dtype=torch.float64).expand(
        theta.shape[0], -1, -1
    )
    assert torch.allclose(gram, identity, atol=ATOL)

    determinant = torch.linalg.det(r)
    assert torch.allclose(determinant, torch.ones_like(determinant), atol=ATOL)


@given(a=_angles, b=_angles)
@settings(max_examples=50)
def test_planar_rotation_matrices_compose_by_angle_addition(
    a: float,
    b: float,
) -> None:
    theta = torch.tensor([a, b], dtype=torch.float64)
    ra, rb = planar_rotation_matrices(theta)
    composed = ra @ rb

    expected = planar_rotation_matrices(
        torch.tensor(a + b, dtype=torch.float64)
    )
    assert torch.allclose(composed, expected, atol=1e-8)


def test_planar_rotation_matrices_rejects_non_tensor() -> None:
    with pytest.raises(TypeError):
        planar_rotation_matrices([0.0, 1.0])  # type: ignore[arg-type]


def test_planar_rotation_matrices_rejects_non_floating_dtype() -> None:
    with pytest.raises(TypeError):
        planar_rotation_matrices(torch.zeros(3, dtype=torch.long))


# --------------------------------------------------------------------------
# rotate_tangent_frames
# --------------------------------------------------------------------------


@pytest.fixture
def nodes_and_frames() -> tuple[torch.Tensor, torch.Tensor]:
    grid = build_icosphere_grid(1)
    return grid.nodes, build_tangent_frames(grid.nodes)


def _assert_oriented_orthonormal(
    nodes: torch.Tensor,
    frames: torch.Tensor,
) -> None:
    gram = torch.matmul(frames.transpose(-1, -2), frames)
    identity = torch.eye(2, dtype=frames.dtype).expand(
        frames.shape[0], -1, -1
    )
    assert torch.allclose(gram, identity, atol=ATOL)

    normals = torch.linalg.cross(frames[:, :, 0], frames[:, :, 1], dim=-1)
    assert torch.allclose(normals, nodes, atol=ATOL)


def test_rotate_tangent_frames_zero_angle_is_identity(
    nodes_and_frames: tuple[torch.Tensor, torch.Tensor],
) -> None:
    _, e = nodes_and_frames
    rotated = rotate_tangent_frames(e, 0.0)
    assert torch.allclose(rotated, e, atol=ATOL)


def test_rotate_tangent_frames_preserves_oriented_orthonormality(
    nodes_and_frames: tuple[torch.Tensor, torch.Tensor],
) -> None:
    nodes, e = nodes_and_frames
    theta = torch.linspace(
        0, 2 * math.pi, nodes.shape[0], dtype=torch.float64
    )
    rotated = rotate_tangent_frames(e, theta)
    _assert_oriented_orthonormal(nodes, rotated)


def test_rotate_tangent_frames_scalar_matches_broadcast_tensor(
    nodes_and_frames: tuple[torch.Tensor, torch.Tensor],
) -> None:
    nodes, e = nodes_and_frames
    scalar_result = rotate_tangent_frames(e, 0.37)
    tensor_result = rotate_tangent_frames(
        e,
        torch.full((nodes.shape[0],), 0.37, dtype=torch.float64),
    )
    assert torch.allclose(scalar_result, tensor_result, atol=ATOL)


def test_rotate_tangent_frames_preserves_physical_vectors(
    nodes_and_frames: tuple[torch.Tensor, torch.Tensor],
) -> None:
    # A physical tangent vector v = E @ x must equal E' @ x' where
    # x' = r(theta)^T @ x, per the function's documented convention.
    nodes, e = nodes_and_frames
    generator = torch.Generator().manual_seed(0)
    theta = (
        torch.rand(nodes.shape[0], dtype=torch.float64, generator=generator)
        * 2
        * math.pi
    )
    x = torch.randn(nodes.shape[0], 2, dtype=torch.float64, generator=generator)

    v = torch.bmm(e, x.unsqueeze(-1)).squeeze(-1)

    rotated = rotate_tangent_frames(e, theta)
    r = planar_rotation_matrices(theta)
    x_prime = torch.bmm(r.transpose(-1, -2), x.unsqueeze(-1)).squeeze(-1)
    v_prime = torch.bmm(rotated, x_prime.unsqueeze(-1)).squeeze(-1)

    assert torch.allclose(v, v_prime, atol=ATOL)


@given(a=_angles, b=_angles)
@settings(max_examples=25)
def test_rotate_tangent_frames_composes_by_angle_addition(
    a: float,
    b: float,
) -> None:
    grid = build_icosphere_grid(0)
    e = build_tangent_frames(grid.nodes)

    once = rotate_tangent_frames(rotate_tangent_frames(e, a), b)
    combined = rotate_tangent_frames(e, a + b)
    assert torch.allclose(once, combined, atol=1e-8)


def test_rotate_tangent_frames_rejects_bad_angle_shape(
    nodes_and_frames: tuple[torch.Tensor, torch.Tensor],
) -> None:
    _, e = nodes_and_frames
    with pytest.raises(ValueError):
        rotate_tangent_frames(e, torch.zeros(3, dtype=torch.float64))


def test_rotate_tangent_frames_rejects_nonfinite_angles(
    nodes_and_frames: tuple[torch.Tensor, torch.Tensor],
) -> None:
    _, e = nodes_and_frames
    theta = torch.zeros(e.shape[0], dtype=torch.float64)
    theta[0] = float("inf")
    with pytest.raises(ValueError):
        rotate_tangent_frames(e, theta)


# --------------------------------------------------------------------------
# Gauge covariance of the full discretization pipeline
# --------------------------------------------------------------------------


def test_edge_geometry_is_gauge_covariant() -> None:
    """Rotating every node's frame by theta_i must transform edge geometry
    by the corresponding gauge-covariance law, derived directly from how
    log_coordinates and transport_angles are defined:

        log_coordinates'  = r(theta_i)^T @ log_coordinates
        transport_angle'  = transport_angle + theta_j - theta_i  (mod 2*pi)

    while distances and tangent_quadrature_weights, being gauge-invariant
    scalars, are left unchanged.
    """
    grid = build_icosphere_grid(1)
    frames = build_tangent_frames(grid.nodes)
    original = compute_edge_geometry(
        grid=grid,
        frames=frames,
        coverage_radius=1.0,
    )

    generator = torch.Generator().manual_seed(0)
    theta = (
        torch.rand(grid.n_nodes, dtype=torch.float64, generator=generator)
        * 2
        * math.pi
    )

    rotated_frames = rotate_tangent_frames(frames, theta)
    rotated = compute_edge_geometry(
        grid=grid,
        frames=rotated_frames,
        coverage_radius=1.0,
    )

    theta_i = theta[original.target_index]
    theta_j = theta[original.source_index]

    r = planar_rotation_matrices(theta_i)
    expected_log = torch.bmm(
        r.transpose(-1, -2),
        original.log_coordinates.unsqueeze(-1),
    ).squeeze(-1)
    assert torch.allclose(expected_log, rotated.log_coordinates, atol=ATOL)

    expected_angle = original.transport_angles + theta_j - theta_i
    angle_error = (
        torch.remainder(
            expected_angle - rotated.transport_angles + math.pi,
            2 * math.pi,
        )
        - math.pi
    )
    assert torch.allclose(
        angle_error,
        torch.zeros_like(angle_error),
        atol=ATOL,
    )

    assert torch.allclose(original.distances, rotated.distances, atol=ATOL)
    assert torch.allclose(
        original.tangent_quadrature_weights,
        rotated.tangent_quadrature_weights,
        atol=ATOL,
    )
