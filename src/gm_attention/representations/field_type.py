"""Field types and feature containers for ``SO(2)``-steerable features.

Every finite-dimensional real representation of ``SO(2)`` is equivalent to a
direct sum of irreducibles, ``rho = sum_{n in I} I_{a_n} (x) rho_n``, so it
is described by the multiplicity ``a_n`` of each frequency ``n``.
:class:`SO2FieldType` is that description; :class:`SO2FeatureField` holds
the coefficients of a field of a given type as one tensor per frequency,
``(*batch_shape, a_0)`` at frequency 0 and ``(*batch_shape, a_n, 2)`` at
``n >= 1``. Frequencies with zero multiplicity are absent. The packed form
of :meth:`SO2FeatureField.to_flat` orders frequencies ascending and, within
a frequency, places the multiplicity index outside the irrep component
index, the channel ordering of ``I_{a_n} (x) rho_n``, so the flat vector
transforms under the block-diagonal matrix built from
:func:`gm_attention.representations.so2.irrep_matrix`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from torch import Tensor

from gm_attention.representations import so2


class SO2FieldType:
    """The multiplicities of each ``SO(2)`` irrep in a feature space.

    Parameters
    ----------
    multiplicities
        Either a mapping ``{frequency: multiplicity}`` or an iterable of
        ``(frequency, multiplicity)`` pairs. Frequencies must be distinct
        nonnegative integers and multiplicities nonnegative integers.
        Frequencies with multiplicity zero are dropped.

    Examples
    --------
    Eight scalar channels, four vector channels and two frequency-two
    channels, occupying ``8 + 2*4 + 2*2 = 20`` real dimensions::

        SO2FieldType({0: 8, 1: 4, 2: 2})
    """

    __slots__ = ("_frequencies", "_multiplicities")

    def __init__(
        self,
        multiplicities: Mapping[int, int] | Iterable[tuple[int, int]],
    ) -> None:
        items = self._normalize(multiplicities)

        self._frequencies: tuple[int, ...] = tuple(
            frequency for frequency, _ in items
        )
        self._multiplicities: tuple[int, ...] = tuple(
            multiplicity for _, multiplicity in items
        )

    @staticmethod
    def _normalize(
        multiplicities: Mapping[int, int] | Iterable[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        if isinstance(multiplicities, Mapping):
            raw_items = list(multiplicities.items())
        else:
            raw_items = [tuple(item) for item in multiplicities]  # type: ignore[misc]

        seen: set[int] = set()
        items: list[tuple[int, int]] = []

        for item in raw_items:
            if len(item) != 2:
                raise ValueError(
                    "Each entry must be a (frequency, multiplicity) pair, "
                    f"got {item!r}."
                )

            frequency, multiplicity = item

            so2.validate_frequency(frequency)

            if isinstance(multiplicity, bool) or not isinstance(
                multiplicity, int
            ):
                raise TypeError(
                    "A multiplicity must be an int, got "
                    f"{type(multiplicity).__name__} for frequency "
                    f"{frequency}."
                )

            if multiplicity < 0:
                raise ValueError(
                    "A multiplicity must be nonnegative, got "
                    f"{multiplicity} for frequency {frequency}."
                )

            if frequency in seen:
                raise ValueError(
                    f"Frequency {frequency} was specified more than once."
                )

            seen.add(frequency)

            if multiplicity > 0:
                items.append((frequency, multiplicity))

        if not items:
            raise ValueError(
                "A field type must contain at least one frequency with "
                "nonzero multiplicity."
            )

        items.sort()

        return items

    @classmethod
    def trivial(cls, multiplicity: int) -> SO2FieldType:
        """A field type of ``multiplicity`` invariant scalar channels."""
        return cls({0: multiplicity})

    @classmethod
    def uniform(
        cls,
        max_frequency: int,
        multiplicity: int,
    ) -> SO2FieldType:
        """A field type with equal multiplicity at every frequency.

        Parameters
        ----------
        max_frequency
            Largest frequency to include; all of ``0, ..., max_frequency``
            are present.
        multiplicity
            Multiplicity assigned to every frequency.
        """
        so2.validate_frequency(max_frequency)

        return cls(
            {
                frequency: multiplicity
                for frequency in range(max_frequency + 1)
            }
        )

    @property
    def frequencies(self) -> tuple[int, ...]:
        """Present frequencies, ascending."""
        return self._frequencies

    @property
    def multiplicities(self) -> tuple[int, ...]:
        """Multiplicities, aligned with :attr:`frequencies`."""
        return self._multiplicities

    @property
    def dimension(self) -> int:
        """Total number of real channels, ``sum_n a_n d_n``."""
        return sum(
            multiplicity * so2.irrep_dimension(frequency)
            for frequency, multiplicity in self.items()
        )

    @property
    def max_frequency(self) -> int:
        """Largest present frequency."""
        return self._frequencies[-1]

    def multiplicity(self, frequency: int) -> int:
        """Multiplicity of ``frequency``, or zero if it is absent."""
        so2.validate_frequency(frequency)

        if frequency not in self._frequencies:
            return 0

        return self._multiplicities[self._frequencies.index(frequency)]

    def uniform_multiplicity(self) -> int:
        """The shared multiplicity ``m`` such that ``self == m * (+_n rho_n)``.

        Raises
        ------
        ValueError
            If the multiplicities differ across the present frequencies.
        """
        distinct = set(self._multiplicities)

        if len(distinct) != 1:
            raise ValueError(
                "Field type is not uniform: multiplicities "
                f"{self._multiplicities} differ across frequencies "
                f"{self._frequencies}."
            )

        return next(iter(distinct))

    def scaled(self, factor: int) -> SO2FieldType:
        """This field type with every multiplicity multiplied by ``factor``.

        ``SO2FieldType({0: 2, 1: 1}).scaled(4)`` is ``{0: 8, 1: 4}``.
        Frequencies and their ordering are preserved and the result is a
        new object.

        Parameters
        ----------
        factor
            Strictly positive integer. ``1`` returns an equal field type.

        Raises
        ------
        TypeError
            If ``factor`` is not an ``int`` (``bool`` is rejected).
        ValueError
            If ``factor`` is less than one.
        """
        if isinstance(factor, bool) or not isinstance(factor, int):
            raise TypeError(
                f"factor must be an int, got {type(factor).__name__}."
            )

        if factor < 1:
            raise ValueError(f"factor must be at least 1, got {factor}.")

        return SO2FieldType(
            {
                frequency: factor * multiplicity
                for frequency, multiplicity in self.items()
            }
        )

    def items(self) -> tuple[tuple[int, int], ...]:
        """``(frequency, multiplicity)`` pairs, ascending by frequency."""
        return tuple(zip(self._frequencies, self._multiplicities, strict=True))

    def slices(self) -> dict[int, slice]:
        """Channel ranges of each frequency in the packed layout.

        Returns
        -------
        dict
            Maps each present frequency to the ``slice`` it occupies in the
            vector produced by :meth:`SO2FeatureField.to_flat`.
        """
        offsets: dict[int, slice] = {}
        start = 0

        for frequency, multiplicity in self.items():
            stop = start + multiplicity * so2.irrep_dimension(frequency)
            offsets[frequency] = slice(start, stop)
            start = stop

        return offsets

    def common_frequencies(self, other: SO2FieldType) -> tuple[int, ...]:
        """Frequencies present in both field types, ascending.

        An ``SO(2)`` intertwiner vanishes between inequivalent irreps, so a
        gauge-equivariant linear layer between two field types carries
        parameters only for these frequencies.
        """
        if not isinstance(other, SO2FieldType):
            raise TypeError(
                "other must be an SO2FieldType, "
                f"got {type(other).__name__}."
            )

        other_frequencies = set(other.frequencies)

        return tuple(
            frequency
            for frequency in self._frequencies
            if frequency in other_frequencies
        )

    def __contains__(self, frequency: object) -> bool:
        return frequency in self._frequencies

    def __len__(self) -> int:
        return len(self._frequencies)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SO2FieldType):
            return NotImplemented

        return (
            self._frequencies == other._frequencies
            and self._multiplicities == other._multiplicities
        )

    def __hash__(self) -> int:
        return hash((self._frequencies, self._multiplicities))

    def __repr__(self) -> str:
        body = ", ".join(
            f"{frequency}: {multiplicity}"
            for frequency, multiplicity in self.items()
        )

        return f"{type(self).__name__}({{{body}}}, dimension={self.dimension})"


class SO2FeatureField:
    """Coefficients of a feature field of a given :class:`SO2FieldType`.

    Parameters
    ----------
    field_type
        The type of the field.
    coefficients
        Mapping from frequency to coefficient tensor. Keys must be exactly
        the frequencies of ``field_type``.

        ``coefficients[0][..., a]`` is the value of the ``a``-th invariant
        scalar copy, shape ``(*batch_shape, a_0)``.

        ``coefficients[n][..., a, d]`` for ``n >= 1`` is coordinate ``d`` of
        the ``a``-th ``rho_n`` copy in the node's tangent frame, shape
        ``(*batch_shape, a_n, 2)``. The pair ``d = (0, 1)`` transforms
        together under a gauge rotation, as ``rho_n(r_theta)`` applied to
        that 2-vector.

        ``batch_shape``, dtype, and device are shared across frequencies.
    """

    __slots__ = ("_batch_shape", "coefficients", "field_type")

    def __init__(
        self,
        field_type: SO2FieldType,
        coefficients: Mapping[int, Tensor],
    ) -> None:
        if not isinstance(field_type, SO2FieldType):
            raise TypeError(
                "field_type must be an SO2FieldType, "
                f"got {type(field_type).__name__}."
            )

        self.field_type = field_type
        self._batch_shape = self._validate(field_type, coefficients)
        self.coefficients: dict[int, Tensor] = dict(coefficients)

    @staticmethod
    def _validate(
        field_type: SO2FieldType,
        coefficients: Mapping[int, Tensor],
    ) -> torch.Size:
        if set(coefficients) != set(field_type.frequencies):
            raise ValueError(
                "coefficients must supply exactly the frequencies of the "
                f"field type {sorted(field_type.frequencies)}, got "
                f"{sorted(coefficients)}."
            )

        batch_shape: torch.Size | None = None
        dtype: torch.dtype | None = None
        device: torch.device | None = None

        for frequency, multiplicity in field_type.items():
            tensor = coefficients[frequency]

            if not isinstance(tensor, Tensor):
                raise TypeError(
                    f"coefficients[{frequency}] must be a torch.Tensor, "
                    f"got {type(tensor).__name__}."
                )

            if tensor.dtype not in (torch.float32, torch.float64):
                raise TypeError(
                    f"coefficients[{frequency}] must use torch.float32 or "
                    f"torch.float64, got {tensor.dtype}."
                )

            dimension = so2.irrep_dimension(frequency)
            trailing = (
                (multiplicity,)
                if dimension == 1
                else (multiplicity, dimension)
            )

            if tensor.ndim < len(trailing) or (
                tuple(tensor.shape[-len(trailing) :]) != trailing
            ):
                raise ValueError(
                    f"coefficients[{frequency}] must have shape "
                    f"(*batch_shape, {', '.join(map(str, trailing))}), "
                    f"got {tuple(tensor.shape)}."
                )

            tensor_batch_shape = tensor.shape[: -len(trailing)]

            if batch_shape is None:
                batch_shape = tensor_batch_shape
                dtype = tensor.dtype
                device = tensor.device
            else:
                if tensor_batch_shape != batch_shape:
                    raise ValueError(
                        "All frequencies must share a batch shape, got "
                        f"{tuple(batch_shape)} and "
                        f"{tuple(tensor_batch_shape)} at frequency "
                        f"{frequency}."
                    )

                if tensor.dtype != dtype:
                    raise ValueError(
                        "All frequencies must share a dtype, got "
                        f"{dtype} and {tensor.dtype} at frequency "
                        f"{frequency}."
                    )

                if tensor.device != device:
                    raise ValueError(
                        "All frequencies must share a device, got "
                        f"{device} and {tensor.device} at frequency "
                        f"{frequency}."
                    )

        assert batch_shape is not None  # a field type is never empty

        return batch_shape

    @classmethod
    def from_flat(
        cls,
        field_type: SO2FieldType,
        flat: Tensor,  # (*batch_shape, field_type.dimension)
    ) -> SO2FeatureField:
        """Unpack a dense channel vector into per-frequency coefficients.

        Parameters
        ----------
        field_type
            Type describing the channel layout of ``flat``.
        flat
            Tensor of shape ``(*batch_shape, field_type.dimension)`` using
            the layout documented in the module docstring.
        """
        if not isinstance(field_type, SO2FieldType):
            raise TypeError(
                "field_type must be an SO2FieldType, "
                f"got {type(field_type).__name__}."
            )

        if not isinstance(flat, Tensor):
            raise TypeError(
                f"flat must be a torch.Tensor, got {type(flat).__name__}."
            )

        if flat.ndim == 0 or flat.shape[-1] != field_type.dimension:
            raise ValueError(
                "flat must have shape (*batch_shape, "
                f"{field_type.dimension}), got {tuple(flat.shape)}."
            )

        channel_slices = field_type.slices()
        coefficients: dict[int, Tensor] = {}

        for frequency, multiplicity in field_type.items():
            block = flat[..., channel_slices[frequency]]

            coefficients[frequency] = (
                block
                if frequency == 0
                else block.unflatten(-1, (multiplicity, 2))
            )

        return cls(field_type, coefficients)

    def to_flat(self) -> Tensor:  # (*batch_shape, field_type.dimension)
        """Pack the coefficients into a dense channel vector.

        Returns
        -------
        Tensor
            Shape ``(*batch_shape, field_type.dimension)``, inverse to
            :meth:`from_flat`.
        """
        blocks = [
            self.coefficients[frequency]
            if frequency == 0
            else self.coefficients[frequency].flatten(-2)
            for frequency in self.field_type.frequencies
        ]

        return torch.cat(blocks, dim=-1)

    def apply_rotation(
        self,
        angles: Tensor,  # broadcastable to batch_shape
    ) -> SO2FeatureField:
        """Act on the coefficients by ``rho(r_theta)``.

        The bare group action, as used to parallel transport a neighbor's
        features into a target node's gauge; :meth:`gauge_transform`
        expresses the same field in a rotated gauge.

        Parameters
        ----------
        angles
            Angles in radians, broadcastable against :attr:`batch_shape`.
        """
        self._validate_angles(angles)

        return type(self)(
            self.field_type,
            {
                frequency: so2.rotate_irrep_coefficients(
                    frequency,
                    tensor,
                    angles.unsqueeze(-1),
                )
                for frequency, tensor in self.coefficients.items()
            },
        )

    def gauge_transform(
        self,
        angles: Tensor,  # broadcastable to batch_shape
    ) -> SO2FeatureField:
        """Re-express the field in the gauge ``E' = E r(theta)``.

        Coefficients transform by the inverse representation,
        ``f' = rho(r_{-theta}) f``, leaving the underlying physical field
        unchanged. This is the transformation law a gauge-equivariant layer
        must respect.

        Parameters
        ----------
        angles
            Gauge rotation angles in radians, broadcastable against
            :attr:`batch_shape`.
        """
        self._validate_angles(angles)

        return self.apply_rotation(-angles)

    def _validate_angles(self, angles: Tensor) -> None:
        if not isinstance(angles, Tensor):
            raise TypeError(
                f"angles must be a torch.Tensor, got {type(angles).__name__}."
            )

        if angles.dtype != self.dtype:
            raise TypeError(
                "angles must share the field dtype "
                f"{self.dtype}, got {angles.dtype}."
            )

        try:
            torch.broadcast_shapes(angles.shape, self._batch_shape)
        except RuntimeError as error:
            raise ValueError(
                f"angles of shape {tuple(angles.shape)} are not broadcastable "
                f"against the batch shape {tuple(self._batch_shape)}."
            ) from error

    @property
    def frequencies(self) -> tuple[int, ...]:
        """Present frequencies, ascending."""
        return self.field_type.frequencies

    @property
    def batch_shape(self) -> torch.Size:
        """Shape shared by all coefficient tensors ahead of their type axes."""
        return self._batch_shape

    @property
    def dtype(self) -> torch.dtype:
        """Floating-point dtype of the coefficients."""
        return self.coefficients[self.field_type.frequencies[0]].dtype

    @property
    def device(self) -> torch.device:
        """Device holding the coefficients."""
        return self.coefficients[self.field_type.frequencies[0]].device

    def __getitem__(self, frequency: int) -> Tensor:
        if frequency not in self.coefficients:
            raise KeyError(
                f"Frequency {frequency} is not present in "
                f"{self.field_type!r}."
            )

        return self.coefficients[frequency]

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> SO2FeatureField:
        """Return a copy of the field on the requested device and dtype."""
        return type(self)(
            self.field_type,
            {
                frequency: tensor.to(
                    device=device,
                    dtype=dtype,
                    non_blocking=non_blocking,
                )
                for frequency, tensor in self.coefficients.items()
            },
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"field_type={self.field_type!r}, "
            f"batch_shape={tuple(self._batch_shape)}, "
            f"dtype={self.dtype}, "
            f"device='{self.device}')"
        )


def scale_field_type(
    base_field_type: Mapping[int, int] | Mapping[str, int],
    multiplicity: int,
) -> SO2FieldType:
    """Build ``multiplicity * base_field_type`` from a config mapping.

    A hidden type is configured as a base spectrum plus a width:
    ``base_field_type: {0: 1, 1: 1, 2: 1}`` with ``multiplicity: 16`` gives
    ``16(rho_0 + rho_1 + rho_2)``, and ``{0: 2, 1: 1}`` with
    ``multiplicity: 4`` gives ``{0: 8, 1: 4}``. Keys may be ``int`` or
    exact decimal-integer strings (OmegaConf delivers YAML integer keys as
    ``str``): ``"1"`` is accepted, ``" 1"``, ``"1.0"`` and ``"0x1"`` are not.

    Parameters
    ----------
    base_field_type
        Maps frequency to base multiplicity. Values must be strictly
        positive ints; a frequency that is not wanted is omitted rather
        than given multiplicity zero.
    multiplicity
        Strictly positive width multiplier.

    Returns
    -------
    SO2FieldType
        ``{n: multiplicity * b_n}``.

    Raises
    ------
    TypeError
        If a key is neither an int nor a decimal-integer string, or a value
        is not an int; ``bool`` is rejected on both sides.
    ValueError
        If the mapping is empty, a key is negative, a base multiplicity is
        not strictly positive, or ``multiplicity`` is less than one.
    """
    if not isinstance(base_field_type, Mapping):
        raise TypeError(
            "base_field_type must be a mapping of frequency to multiplicity, "
            f"got {type(base_field_type).__name__}."
        )

    if not base_field_type:
        raise ValueError("base_field_type must not be empty.")

    coerced: dict[int, int] = {}

    for raw_frequency, raw_multiplicity in base_field_type.items():
        frequency = _coerce_frequency_key(raw_frequency)

        if isinstance(raw_multiplicity, bool) or not isinstance(
            raw_multiplicity, int
        ):
            raise TypeError(
                "A base multiplicity must be an int, got "
                f"{type(raw_multiplicity).__name__} for frequency "
                f"{frequency}."
            )

        if raw_multiplicity < 1:
            raise ValueError(
                "A base multiplicity must be at least 1, got "
                f"{raw_multiplicity} for frequency {frequency}. Omit a "
                "frequency entirely rather than giving it multiplicity 0."
            )

        if frequency in coerced:
            raise ValueError(
                f"Frequency {frequency} was specified more than once "
                "(string and integer keys naming the same frequency count "
                "as duplicates)."
            )

        coerced[frequency] = raw_multiplicity

    return SO2FieldType(coerced).scaled(multiplicity)


def _coerce_frequency_key(key: object) -> int:
    """One frequency key from a config mapping, as an ``int``."""
    if isinstance(key, bool):
        raise TypeError(
            "A frequency must be an int or a decimal-integer string, got "
            f"bool ({key!r}) -- note bool is an int subclass in Python, so "
            "this would otherwise be read as a frequency."
        )

    if isinstance(key, int):
        frequency = key
    elif isinstance(key, str):
        # str.isdigit rejects the whitespace, sign and underscores that int()
        # accepts, so only an exact decimal integer passes.
        if not key.isdigit():
            raise TypeError(
                "A string frequency key must be an exact decimal integer, "
                f"got {key!r}."
            )
        frequency = int(key)
    else:
        raise TypeError(
            "A frequency must be an int or a decimal-integer string, got "
            f"{type(key).__name__} ({key!r})."
        )

    so2.validate_frequency(frequency)

    return frequency
