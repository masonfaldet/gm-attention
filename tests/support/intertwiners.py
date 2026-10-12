"""Dense reference implementation of ``SO(2)`` intertwiners.

This module transcribes the intertwiner constraint literally:

    W in Hom_{SO(2)}(rho_n, rho_m)  iff  W rho_n(r_theta) = rho_m(r_theta) W
                                           for all theta.

Up to equivalence,

    Hom_{SO(2)}(rho_n, rho_m) = span_R{1}       if m = n = 0,
                               = span_R{I, J}    if m = n >= 1,
                               = {0}             if m != n,

and, for arbitrary (reducible) types built from multiplicities, an
intertwiner is block diagonal in frequency: the block connecting frequency
``n`` on the input to frequency ``m`` on the output is nonzero only if
``m = n``, and on that diagonal it is an arbitrary linear map between the
multiplicity spaces tensored with ``Hom_{SO(2)}(rho_n, rho_n)``.

Every function here works with dense ``2 x 2`` matrices and is written for
transparency rather than speed. It provides a reference implementation
against which the optimized complex-arithmetic layers in ``nn/linear.py``
are checked, and is not used in a forward pass. See
:func:`gm_attention.representations.so2.irrep_phase` for the fast path.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from gm_attention.representations import so2
from gm_attention.representations.field_type import SO2FieldType

# The intertwiner basis of Hom_{SO(2)}(rho_n, rho_n), n >= 1.
_I = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
_J = torch.tensor([[0.0, -1.0], [1.0, 0.0]])


def hom_dimension(frequency_n: int, frequency_m: int) -> int:
    """Dimension of ``Hom_{SO(2)}(rho_n, rho_m)``.

    Returns
    -------
    int
        ``1`` if ``m = n = 0``, ``2`` if ``m = n >= 1``, and ``0`` otherwise.
    """
    so2.validate_frequency(frequency_n)
    so2.validate_frequency(frequency_m)

    if frequency_n != frequency_m:
        return 0

    return 1 if frequency_n == 0 else 2


def hom_basis(
    frequency_n: int,
    frequency_m: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> Tensor:  # (dim, d_m, d_n)
    """Basis matrices of ``Hom_{SO(2)}(rho_n, rho_m)``.

    Parameters
    ----------
    frequency_n, frequency_m
        Nonnegative integer frequencies.
    dtype, device
        Floating-point dtype and device of the returned tensor.

    Returns
    -------
    Tensor
        Shape ``(hom_dimension(n, m), d_m, d_n)``. For ``m = n = 0`` this is
        ``[[[1]]]``. For ``m = n >= 1`` it is ``[I, J]``, the basis used in
        the intertwiner block. Otherwise it is empty.
    """
    dimension = hom_dimension(frequency_n, frequency_m)
    output_dimension = so2.irrep_dimension(frequency_m)
    input_dimension = so2.irrep_dimension(frequency_n)

    if dimension == 0:
        return torch.empty(
            (0, output_dimension, input_dimension),
            dtype=dtype,
            device=device,
        )

    if frequency_n == 0:
        return torch.ones((1, 1, 1), dtype=dtype, device=device)

    return torch.stack((_I, _J)).to(dtype=dtype, device=device)  # (2, 2, 2)


def dense_intertwiner_block(
    frequency_n: int,
    frequency_m: int,
    coefficients: Tensor,  # (..., hom_dimension(n, m))
) -> Tensor:  # (..., d_m, d_n)
    """Assemble an intertwiner block from basis coefficients.

    Parameters
    ----------
    frequency_n, frequency_m
        Nonnegative integer frequencies.
    coefficients
        Real tensor of shape ``(..., hom_dimension(n, m))``. For ``m = n = 0``
        this is a scalar (``M_0``); for ``m = n >= 1`` its two components are
        ``(M_n, N_n)``, so that the returned block is ``M_n I + N_n J``.

    Returns
    -------
    Tensor
        Shape ``(..., d_m, d_n)``.
    """
    dimension = hom_dimension(frequency_n, frequency_m)
    output_dimension = so2.irrep_dimension(frequency_m)
    input_dimension = so2.irrep_dimension(frequency_n)

    if not isinstance(coefficients, Tensor):
        raise TypeError(
            "coefficients must be a torch.Tensor, "
            f"got {type(coefficients).__name__}."
        )

    if coefficients.shape[-1:] != (dimension,):
        raise ValueError(
            f"coefficients must have shape (..., {dimension}) for the pair "
            f"(n={frequency_n}, m={frequency_m}), got "
            f"{tuple(coefficients.shape)}."
        )

    if dimension == 0:
        return torch.zeros(
            (*coefficients.shape[:-1], output_dimension, input_dimension),
            dtype=coefficients.dtype,
            device=coefficients.device,
        )

    basis = hom_basis(  # (dim, d_m, d_n)
        frequency_n,
        frequency_m,
        dtype=coefficients.dtype,
        device=coefficients.device,
    )

    return torch.einsum("...i,ipq->...pq", coefficients, basis)


def is_intertwiner(
    frequency_n: int,
    frequency_m: int,
    matrix: Tensor,  # (d_m, d_n)
    *,
    angles: Tensor | None = None,  # (n_samples,)
    atol: float = 1e-8,
) -> bool:
    """Numerically check ``W rho_n(r_theta) = rho_m(r_theta) W`` by sampling.

    This is an independent check of the exact-basis machinery above: it
    verifies the defining constraint directly against
    :func:`gm_attention.representations.so2.irrep_matrix` rather than by
    testing membership in ``hom_basis``'s span.

    Parameters
    ----------
    frequency_n, frequency_m
        Nonnegative integer frequencies.
    matrix
        Candidate intertwiner of shape ``(d_m, d_n)``.
    angles
        Sample angles. Defaults to 16 angles evenly spaced over one period;
        since both sides of the constraint are trigonometric polynomials of
        bounded degree in ``theta``, agreement at enough distinct samples
        implies agreement everywhere.
    atol
        Absolute tolerance for the comparison.
    """
    expected_shape = (
        so2.irrep_dimension(frequency_m),
        so2.irrep_dimension(frequency_n),
    )

    if not isinstance(matrix, Tensor) or tuple(matrix.shape) != expected_shape:
        raise ValueError(
            f"matrix must have shape {expected_shape}, "
            f"got {matrix.shape if isinstance(matrix, Tensor) else type(matrix)}."
        )

    if angles is None:
        angles = torch.linspace(
            0.0,
            2 * torch.pi,
            17,
            dtype=matrix.dtype,
            device=matrix.device,
        )[:-1]

    input_matrices = so2.irrep_matrix(frequency_n, angles)  # (S, d_n, d_n)
    output_matrices = so2.irrep_matrix(frequency_m, angles)  # (S, d_m, d_m)

    left = matrix @ input_matrices  # (S, d_m, d_n)
    right = output_matrices @ matrix  # (S, d_m, d_n)

    return bool(torch.allclose(left, right, atol=atol))


def dense_field_type_intertwiner(
    field_type_in: SO2FieldType,
    field_type_out: SO2FieldType,
    weights: Mapping[int, Tensor],
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:  # (field_type_out.dimension, field_type_in.dimension)
    """Assemble a full intertwiner from per-frequency weights.

    This is the dense ground truth for a gauge-equivariant linear layer: the
    block-diagonal-in-frequency matrix  used to check
    the fast complex-arithmetic layer in ``nn/linear.py``.

    Parameters
    ----------
    field_type_in, field_type_out
        Input and output field types.
    weights
        Must have exactly one entry per frequency in
        ``field_type_in.common_frequencies(field_type_out)`` (the frequencies
        an equivariant linear layer can carry parameters for; every other
        block of the intertwiner is forced to zero):

        - at frequency ``0``: a real tensor of shape ``(b_0, a_0)``;
        - at frequency ``n >= 1``: a complex tensor of shape ``(b_n, a_n)``,
          whose real and imaginary parts are ``M_n`` and ``N_n``.
          This is exactly the ``rho_n`` weight :func:`gm_attention.nn.linear`
           layers store, so this function doubles as the reference for their
          forward pass.
    dtype, device
        Only needed when ``weights`` is empty (no common frequencies), in
        which case the result is an all-zero matrix and there is no weight
        tensor to infer them from.

    Returns
    -------
    Tensor
        Real matrix of shape
        ``(field_type_out.dimension, field_type_in.dimension)``.
    """
    common_frequencies = set(field_type_in.common_frequencies(field_type_out))

    if set(weights) != common_frequencies:
        raise ValueError(
            "weights must have exactly one entry per common frequency "
            f"{sorted(common_frequencies)}, got {sorted(weights)}."
        )

    in_slices = field_type_in.slices()
    out_slices = field_type_out.slices()

    resolved_dtype, resolved_device = _resolve_dtype_and_device(
        weights,
        dtype,
        device,
    )

    matrix = torch.zeros(  # (out_dim, in_dim)
        (field_type_out.dimension, field_type_in.dimension),
        dtype=resolved_dtype,
        device=resolved_device,
    )

    for frequency in common_frequencies:
        weight = weights[frequency]
        multiplicity_in = field_type_in.multiplicity(frequency)
        multiplicity_out = field_type_out.multiplicity(frequency)

        if frequency == 0:
            _validate_real_weight(
                weight,
                frequency,
                (multiplicity_out, multiplicity_in),
            )
            block = weight  # (b_0, a_0)
        else:
            _validate_complex_weight(
                weight,
                frequency,
                (multiplicity_out, multiplicity_in),
            )
            coefficients = so2.as_real(weight)  # (b_n, a_n, 2)
            block = dense_intertwiner_block(  # (b_n, a_n, 2, 2)
                frequency,
                frequency,
                coefficients,
            )
            block = _assemble_multiplicity_block(block)  # (b_n*2, a_n*2)

        matrix[out_slices[frequency], in_slices[frequency]] = block.to(
            dtype=resolved_dtype,
            device=resolved_device,
        )

    return matrix


def _assemble_multiplicity_block(
    block: Tensor,  # (b, a, d_m, d_n)
) -> Tensor:  # (b * d_m, a * d_n)
    """Fold per-multiplicity-pair blocks into one dense matrix.

    The channel layout is multiplicity-outer, component-inner (see
    ``gm_attention.representations.field_type``), so folding permutes to
    ``(b, d_m, a, d_n)`` before merging -- this is exactly
    ``kron(M, I) + kron(N, J)`` in the ``n = m >= 1`` case.
    """
    b, a, d_m, d_n = block.shape

    return block.permute(0, 2, 1, 3).reshape(b * d_m, a * d_n)


def _resolve_dtype_and_device(
    weights: Mapping[int, Tensor],
    dtype: torch.dtype | None,
    device: torch.device | str | None,
) -> tuple[torch.dtype, torch.device | str | None]:
    if weights:
        # Frequency zero, if present, is the only real-valued entry;
        # otherwise every weight is complex and the real dtype it implies is
        # what the assembled (real) matrix must use.
        reference = next(iter(weights.values()))

        if not isinstance(reference, Tensor):
            raise TypeError(
                "Each weight must be a torch.Tensor, "
                f"got {type(reference).__name__}."
            )

        inferred_dtype = (
            reference.dtype
            if not reference.is_complex()
            else so2.real_dtype(reference.dtype)
        )
        inferred_device = reference.device

        if dtype is not None and dtype != inferred_dtype:
            raise ValueError(
                f"dtype={dtype} conflicts with the dtype implied by "
                f"weights, {inferred_dtype}."
            )

        if device is not None and torch.device(device) != inferred_device:
            raise ValueError(
                f"device={device} conflicts with the device implied by "
                f"weights, {inferred_device}."
            )

        return inferred_dtype, inferred_device

    if dtype is None or device is None:
        raise ValueError(
            "dtype and device must be provided when weights is empty, "
            "since there is no weight tensor to infer them from."
        )

    return dtype, device


def _validate_real_weight(
    weight: Tensor,
    frequency: int,
    expected_shape: tuple[int, int],
) -> None:
    if not isinstance(weight, Tensor):
        raise TypeError(
            f"weights[{frequency}] must be a torch.Tensor, "
            f"got {type(weight).__name__}."
        )

    if weight.is_complex():
        raise TypeError(
            f"weights[{frequency}] (frequency zero) must be real, got "
            f"{weight.dtype}."
        )

    if tuple(weight.shape) != expected_shape:
        raise ValueError(
            f"weights[{frequency}] must have shape {expected_shape}, "
            f"got {tuple(weight.shape)}."
        )


def _validate_complex_weight(
    weight: Tensor,
    frequency: int,
    expected_shape: tuple[int, int],
) -> None:
    if not isinstance(weight, Tensor):
        raise TypeError(
            f"weights[{frequency}] must be a torch.Tensor, "
            f"got {type(weight).__name__}."
        )

    if not weight.is_complex():
        raise TypeError(
            f"weights[{frequency}] (frequency >= 1) must be complex, got "
            f"{weight.dtype}."
        )

    if tuple(weight.shape) != expected_shape:
        raise ValueError(
            f"weights[{frequency}] must have shape {expected_shape}, "
            f"got {tuple(weight.shape)}."
        )
