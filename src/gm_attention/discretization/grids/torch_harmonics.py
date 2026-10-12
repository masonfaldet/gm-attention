"""Tensor-product spherical grids backed by the torch-harmonics package.

Colatitude quadrature nodes and weights are computed by
``torch_harmonics.quadrature``, part of:

    torch-harmonics -- https://github.com/NVIDIA/torch-harmonics
    Copyright (c) 2022 The torch-harmonics Authors. BSD-3-Clause license.

    Bonev, B., Kurth, T., Hundt, C., Pathak, J., Baust, M., Kashinath, K.,
    Anandkumar, A. "Spherical Fourier Neural Operators: Learning Stable
    Dynamics on the Sphere." International Conference on Machine Learning,
    2023. https://arxiv.org/abs/2306.03838
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor
from torch_harmonics.quadrature import (
    clenshaw_curtiss_weights,
    legendre_gauss_weights,
    lobatto_weights,
)

from gm_attention.discretization.grid import SphericalGrid

_ColatitudeRule = Callable[..., tuple[Tensor, Tensor]]

# Grid name -> 1-D colatitude quadrature rule. "equiangular" is
# torch-harmonics' name for Clenshaw-Curtis quadrature at equispaced colatitudes.
_GRID_RULES: dict[str, _ColatitudeRule] = {
    "legendre-gauss": legendre_gauss_weights,
    "lobatto": lobatto_weights,
    "equiangular": clenshaw_curtiss_weights,
}


def build_torch_harmonics_grid(
    grid: str,
    n_latitude: int,
    n_longitude: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> SphericalGrid:
    """Construct a tensor-product quadrature grid on the unit sphere.

    Colatitude nodes and weights come from one of the quadrature rules in
    ``torch_harmonics.quadrature``.
    Longitude nodes are equispaced, exact for trigonometric polynomials up
    to degree ``n_longitude - 1``. The tensor-product weights sum exactly
    to ``4 * pi`` for every supported ``grid``.

    Parameters
    ----------
    grid
        Colatitude quadrature rule. One of:

        - ``"legendre-gauss"`` -- Gauss-Legendre nodes, exact for Legendre
          polynomials in ``cos(theta)`` up to degree ``2 * n_latitude - 1``.
        - ``"lobatto"`` -- Legendre-Gauss-Lobatto nodes, which include both
          poles.
        - ``"equiangular"`` -- Clenshaw-Curtis quadrature at equispaced
          colatitudes.
    n_latitude
        Number of colatitude nodes.
    n_longitude
        Number of equispaced longitude nodes.
    dtype
        Floating-point dtype of the returned grid.
    device
        Device of the returned grid.

    Returns
    -------
    SphericalGrid
        Grid with ``n_latitude * n_longitude`` nodes.
    """
    if grid not in _GRID_RULES:
        raise ValueError(
            f"grid must be one of {sorted(_GRID_RULES)}, got {grid!r}."
        )

    if n_latitude < 1:
        raise ValueError(f"n_latitude must be positive, got {n_latitude}.")

    if grid in ("lobatto", "equiangular") and n_latitude < 2:
        raise ValueError(
            f"The {grid!r} rule requires n_latitude >= 2, got {n_latitude}."
        )

    if n_longitude < 1:
        raise ValueError(f"n_longitude must be positive, got {n_longitude}.")

    # Every rule returns nodes/weights on [-1, 1] in cos(theta); integrating
    # in cos(theta) absorbs the sin(theta) area Jacobian.
    cos_theta, latitude_weights = _GRID_RULES[grid](  # (n_latitude,), (n_latitude,)
        n_latitude,
        a=-1.0,
        b=1.0,
    )

    # The rules return ascending cos(theta), i.e. descending theta; nodes and
    # weights are flipped together for ascending colatitude.
    colatitude = torch.flip(torch.arccos(cos_theta), dims=(0,))  # (n_latitude,)
    latitude_weights = torch.flip(latitude_weights, dims=(0,))  # (n_latitude,)

    longitude = torch.linspace(  # (n_longitude,)
        0.0,
        2.0 * math.pi,
        n_longitude + 1,
        dtype=torch.float64,
    )[:-1]

    longitude_weight = 2.0 * math.pi / n_longitude

    theta = colatitude.repeat_interleave(n_longitude)  # (n_latitude * n_longitude,)
    phi = longitude.repeat(n_latitude)  # (n_latitude * n_longitude,)

    sin_theta = torch.sin(theta)  # (n_latitude * n_longitude,)

    nodes = torch.stack(  # (n_latitude * n_longitude, 3)
        [
            sin_theta * torch.cos(phi),
            sin_theta * torch.sin(phi),
            torch.cos(theta),
        ],
        dim=-1,
    )

    quadrature_weights = (  # (n_latitude * n_longitude,)
        latitude_weights.repeat_interleave(n_longitude) * longitude_weight
    )

    return SphericalGrid(
        nodes=nodes.to(dtype=dtype, device=device).contiguous(),  # (n_latitude * n_longitude, 3)
        quadrature_weights=quadrature_weights.to(  # (n_latitude * n_longitude,)
            dtype=dtype,
            device=device,
        ).contiguous(),
        exact_nodes=nodes,  # the float64 lattice, for edge membership
    )
