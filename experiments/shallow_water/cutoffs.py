"""The S2 Transformer's three support radii, transcribed from
``torch_harmonics``' ``SphericalTransformer``.

The neighborhood-attention radius is computed from the latent latitude
count, the encoder DISCO radius from the input latitude count, and the
decoder DISCO radius, although that convolution runs on the fine grid, from
the latent latitude count. The three are named separately here and in
configs and logs. The formulas follow the implementation, which uses
``H - 1`` where the paper writes ``H`` for the attention radius.
"""

from __future__ import annotations

import math

#: Encoder kernel latitude extent of ``_compute_cutoff_radius``.
S2_ENCODER_KERNEL_LATITUDE = 3
#: Filter-basis factor of ``_compute_cutoff_radius``: 0.5 for the morlet and piecewise linear bases.
S2_MORLET_BASIS_FACTOR = 0.5


def s2_attention_cutoff(latent_nlat: int) -> float:
    """Neighborhood-attention radius, in radians, from the latent latitude
    count: ``7 sqrt(pi) / (latent_nlat - 1)``, the value
    ``SphericalAttentionBlock`` derives when ``theta_cutoff`` is unset.
    """
    if latent_nlat < 2:
        raise ValueError(f"latent_nlat must be at least 2, got {latent_nlat}.")

    return 7.0 * math.sqrt(math.pi) / (latent_nlat - 1)


def s2_disco_cutoff(
    reference_nlat: int,
    kernel_latitude: int = S2_ENCODER_KERNEL_LATITUDE,
    basis_factor: float = S2_MORLET_BASIS_FACTOR,
) -> float:
    """DISCO convolution radius, in radians:
    ``(kernel_latitude + 1) * basis_factor * pi / (reference_nlat - 1)``.

    ``reference_nlat`` is the input latitude count for the encoder and the
    latent latitude count for the decoder, although the decoder convolution
    runs on the fine grid after resampling.
    """
    if reference_nlat < 2:
        raise ValueError(f"reference_nlat must be at least 2, got {reference_nlat}.")

    return (kernel_latitude + 1) * basis_factor * math.pi / (reference_nlat - 1)
