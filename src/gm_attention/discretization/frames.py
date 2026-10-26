from __future__ import annotations

import torch
from torch import Tensor


def rotate_tangent_frames(
    frames: Tensor,  # (n_nodes, 3, 2)
    angles: Tensor | float,  # scalar or (n_nodes,)
) -> Tensor:
    """Apply node-wise SO(2) gauge rotations to tangent frames.

    For a frame matrix

        E_i = [e_{i,1}  e_{i,2}],

    the rotated frame is

        E_i' = E_i r(theta_i),

    where

        r(theta) = [[cos(theta), -sin(theta)],
                    [sin(theta),  cos(theta)]].

    Consequently,

        e'_{i,1} = cos(theta_i)e_{i,1}
                   + sin(theta_i)e_{i,2},

        e'_{i,2} = -sin(theta_i)e_{i,1}
                   + cos(theta_i)e_{i,2}.

    Parameters
    ----------
    frames
        Oriented orthonormal tangent frames with shape
        ``(n_nodes, 3, 2)``.
    angles
        Gauge rotation angles in radians. This may be either:

        - a scalar applied at every node; or
        - a tensor of shape ``(n_nodes,)`` specifying an independent gauge
          rotation at every node.

    Returns
    -------
    Tensor
        Rotated frames with shape ``(n_nodes, 3, 2)``.

    Notes
    -----
    The gauge changes; the tangent planes do not. A tangent vector with
    coordinates ``x`` in the original frame has coordinates

        x' = r(theta)^T x = r(-theta)x.
    """
    _validate_frames(frames)

    angle_tensor = torch.as_tensor(  # scalar or (n_nodes,)
        angles,
        dtype=frames.dtype,
        device=frames.device,
    )

    if angle_tensor.ndim == 0:
        angle_tensor = angle_tensor.expand(  # (n_nodes,)
            frames.shape[0]
        )
    elif angle_tensor.shape != (frames.shape[0],):
        raise ValueError(
            "angles must be a scalar or have shape "
            f"({frames.shape[0]},), got "
            f"{tuple(angle_tensor.shape)}."
        )

    if not bool(torch.isfinite(angle_tensor).all()):
        raise ValueError(
            "angles contains nonfinite values."
        )

    gauge_rotations = planar_rotation_matrices(  # (n_nodes, 2, 2)
        angle_tensor
    )

    return torch.bmm(  # (n_nodes, 3, 2)
        frames,
        gauge_rotations,
    ).contiguous()


def planar_rotation_matrices(
    angles: Tensor,  # (...)
) -> Tensor:
    """Construct standard two-dimensional rotation matrices.

    Parameters
    ----------
    angles
        Tensor of rotation angles with arbitrary shape ``(...)``.

    Returns
    -------
    Tensor
        Rotation matrices with shape ``(..., 2, 2)``.
    """
    if not isinstance(angles, Tensor):
        raise TypeError(
            "angles must be a torch.Tensor."
        )

    if not angles.is_floating_point():
        raise TypeError(
            "angles must have a floating-point dtype."
        )

    cosine = torch.cos(angles)  # (...)
    sine = torch.sin(angles)  # (...)

    first_row = torch.stack(  # (..., 2)
        (cosine, -sine),
        dim=-1,
    )

    second_row = torch.stack(  # (..., 2)
        (sine, cosine),
        dim=-1,
    )

    return torch.stack(  # (..., 2, 2)
        (first_row, second_row),
        dim=-2,
    )


def build_tangent_frames(
    nodes: Tensor,  # (n_nodes, 3)
) -> Tensor:
    """Construct deterministic oriented orthonormal frames on the unit sphere.

    For each node ``p``, the Cartesian coordinate axis least aligned with
    ``p`` is projected into ``T_p S^2`` and normalized to give the first
    frame vector. The second frame vector is

        e_2 = p x e_1,

    so that

        e_1 x e_2 = p.

    Parameters
    ----------
    nodes
        Unit-sphere nodes with shape ``(n_nodes, 3)``.

    Returns
    -------
    Tensor
        Oriented orthonormal tangent frames with shape
        ``(n_nodes, 3, 2)``. The final dimension indexes the two frame
        vectors.

    Notes
    -----
    The construction is deterministic but not globally continuous (no
    continuous tangent frame field exists on the sphere); the
    discontinuities are changes of gauge, to which a gauge-equivariant
    layer is insensitive.
    """
    _validate_sphere_nodes(nodes)

    # Selecting the axis least aligned with p guarantees that its tangent
    # projection stays uniformly away from zero.
    reference_axis_indices = torch.argmin(  # (n_nodes,)
        torch.abs(nodes),
        dim=-1,
    )

    canonical_axes = torch.eye(  # (3, 3)
        3,
        dtype=nodes.dtype,
        device=nodes.device,
    )

    reference_axes = canonical_axes[reference_axis_indices]  # (n_nodes, 3)

    # Orthogonally project the selected ambient axis into T_p S^2:
    #
    #     a_T = a - <a, p>p.
    projection_coefficients = torch.sum(  # (n_nodes, 1)
        reference_axes * nodes,
        dim=-1,
        keepdim=True,
    )

    first_vectors = (  # (n_nodes, 3)
        reference_axes
        - projection_coefficients * nodes
    )

    first_vectors = first_vectors / torch.linalg.vector_norm(
        first_vectors,
        dim=-1,
        keepdim=True,
    )

    # e_2 = p x e_1 gives the positive orientation
    #
    #     e_1 x e_2 = p.
    second_vectors = torch.linalg.cross(  # (n_nodes, 3)
        nodes,
        first_vectors,
        dim=-1,
    )

    # Renormalize to suppress floating-point accumulation.
    second_vectors = second_vectors / torch.linalg.vector_norm(
        second_vectors,
        dim=-1,
        keepdim=True,
    )

    return torch.stack(  # (n_nodes, 3, 2)
        (first_vectors, second_vectors),
        dim=-1,
    ).contiguous()


def build_coordinate_frames(
    nodes: Tensor,  # (n_nodes, 3)
) -> Tensor:
    """Construct the spherical-coordinate frames ``(e_theta, e_phi)``.

    For a node ``p = (sin theta cos phi, sin theta sin phi, cos theta)`` the
    frame vectors are the unit tangents of the colatitude and longitude
    coordinate lines,

        e_theta = (cos theta cos phi, cos theta sin phi, -sin theta),
        e_phi   = (-sin phi, cos phi, 0),

    with ``e_theta x e_phi = p``. Unlike :func:`build_tangent_frames`, this
    frame field is covariant under rotations about the polar axis: the
    rotation by ``alpha`` carries the frame at ``(theta, phi)`` onto the
    frame at ``(theta, phi + alpha)``. On a tensor-product grid the edge
    geometry expressed in these frames therefore depends on the two
    colatitude rows and the longitude offset only, which
    :mod:`gm_attention.discretization.stencil` folds into per-row tables.
    Choosing these frames over :func:`build_tangent_frames` is a change of
    gauge, to which every gauge-equivariant layer is insensitive.

    Parameters
    ----------
    nodes
        Unit-sphere nodes with shape ``(n_nodes, 3)``, none at a pole.

    Returns
    -------
    Tensor
        Oriented orthonormal tangent frames with shape ``(n_nodes, 3, 2)``.

    Raises
    ------
    ValueError
        If a node lies at a pole, where the coordinate frame is undefined
        (``sin theta`` below the dtype tolerance). The ``"lobatto"`` and
        ``"equiangular"`` tensor-product grids carry pole nodes;
        ``"legendre-gauss"`` has none.
    """
    _validate_sphere_nodes(nodes)

    tolerance = 1e-10 if nodes.dtype == torch.float64 else 1e-5

    sin_theta = torch.linalg.vector_norm(nodes[:, :2], dim=-1)  # (n_nodes,)
    at_pole = sin_theta < tolerance

    if bool(at_pole.any()):
        raise ValueError(
            f"{int(at_pole.sum())} node(s) lie at a pole, where the coordinate "
            "frame (e_theta, e_phi) is undefined. Use a grid without pole "
            "nodes ('legendre-gauss') or build_tangent_frames."
        )

    cos_phi = nodes[:, 0] / sin_theta  # (n_nodes,)
    sin_phi = nodes[:, 1] / sin_theta  # (n_nodes,)
    cos_theta = nodes[:, 2]  # (n_nodes,)

    e_theta = torch.stack(  # (n_nodes, 3)
        (cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta),
        dim=-1,
    )
    e_theta = e_theta / torch.linalg.vector_norm(e_theta, dim=-1, keepdim=True)

    e_phi = torch.stack(  # (n_nodes, 3)
        (-sin_phi, cos_phi, torch.zeros_like(sin_phi)),
        dim=-1,
    )
    e_phi = e_phi / torch.linalg.vector_norm(e_phi, dim=-1, keepdim=True)

    return torch.stack(  # (n_nodes, 3, 2)
        (e_theta, e_phi),
        dim=-1,
    ).contiguous()


def _validate_sphere_nodes(
    nodes: Tensor,  # (n_nodes, 3)
) -> None:
    if not isinstance(nodes, Tensor):
        raise TypeError(
            "nodes must be a torch.Tensor, "
            f"got {type(nodes).__name__}."
        )

    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError(
            "nodes must have shape (n_nodes, 3), "
            f"got {tuple(nodes.shape)}."
        )

    if nodes.shape[0] == 0:
        raise ValueError(
            "At least one sphere node is required."
        )

    if nodes.dtype not in (torch.float32, torch.float64):
        raise TypeError(
            "Sphere geometry must use torch.float32 or torch.float64, "
            f"got {nodes.dtype}."
        )

    if not bool(torch.isfinite(nodes).all()):
        raise ValueError(
            "nodes contains nonfinite values."
        )

    tolerance = (
        1e-10
        if nodes.dtype == torch.float64
        else 1e-5
    )

    node_norms = torch.linalg.vector_norm(  # (n_nodes,)
        nodes,
        dim=-1,
    )

    if not bool(
        torch.allclose(
            node_norms,
            torch.ones_like(node_norms),
            atol=tolerance,
            rtol=0.0,
        )
    ):
        maximum_error = torch.max(
            torch.abs(node_norms - 1.0)
        ).item()

        raise ValueError(
            "Every node must lie on the unit sphere. "
            f"Maximum norm error: {maximum_error:.3e}."
        )


def _validate_frames(
    frames: Tensor,  # (n_nodes, 3, 2)
) -> None:
    if not isinstance(frames, Tensor):
        raise TypeError(
            "frames must be a torch.Tensor, "
            f"got {type(frames).__name__}."
        )

    if frames.ndim != 3 or frames.shape[1:] != (3, 2):
        raise ValueError(
            "frames must have shape (n_nodes, 3, 2), "
            f"got {tuple(frames.shape)}."
        )

    if frames.dtype not in (torch.float32, torch.float64):
        raise TypeError(
            "frames must use torch.float32 or torch.float64, "
            f"got {frames.dtype}."
        )

    if not bool(torch.isfinite(frames).all()):
        raise ValueError(
            "frames contains nonfinite values."
        )
