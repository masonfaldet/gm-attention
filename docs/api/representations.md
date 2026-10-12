# Representations

Irreducible real representations of `SO(2)`, the field types built from
direct sums of them, and the container holding the coefficients of a feature
field of a given type. The mathematics is on the page
[Representations of SO(2)](../math/so2_representations.md); the paper
introduces field types in Section 3 and their decomposition into irreducibles
in Section 4.

Conventions used by both modules. `rho_0` is the trivial representation and,
for an integer frequency `n >= 1`, `rho_n(r_theta)` is the rotation by
`n theta`, which acts on `R^2 = C` (`(x, y) <-> x + iy`) as multiplication by
`exp(i n theta)`. A feature field is stored as its coefficients with respect
to a node's tangent gauge. A change of gauge `E' = E r(theta)` transforms the
coefficients by the inverse representation, `f' = rho(r_{-theta}) f`, so the
underlying physical field is unchanged. The routines that apply `rho(r_theta)`
itself are the bare group action; the sign of a regauging is applied by
[`SO2FeatureField.gauge_transform`][gm_attention.representations.field_type.SO2FeatureField.gauge_transform].

## `SO(2)` irreps

Frequency-`n` irreps in two forms. The dense form
([`irrep_matrix`][gm_attention.representations.so2.irrep_matrix]) is a real
`d_n x d_n` matrix, with `d_0 = 1` and `d_n = 2` for `n >= 1`. The complex
form is one unit-modulus number per angle
([`irrep_phase`][gm_attention.representations.so2.irrep_phase]), and the
coefficients of a `rho_n` copy with `n >= 1` are read as a complex number with
[`as_complex`][gm_attention.representations.so2.as_complex] and written back
with [`as_real`][gm_attention.representations.so2.as_real]. `as_complex` is a
view, with gradients flowing to the real tensor, when its argument is
contiguous at an even storage offset, and copies otherwise. The linear,
kernel and attention layers read coefficients in the complex form;
[`rotate_irrep_coefficients`][gm_attention.representations.so2.rotate_irrep_coefficients]
applies `rho_n(r_theta)` to coefficients through it.

[`phase_powers`][gm_attention.representations.so2.phase_powers] tabulates
`exp(i k theta)` for `k = 0, ..., max_frequency` with a single
`torch.polar`; the edge geometry builds its per-edge tables of angular and
transport harmonics with it (see [Discretization](discretization.md)). Only
nonnegative orders are stored, negative ones being complex conjugates.

Angles and coefficients must be `float32` or `float64` tensors;
[`complex_dtype`][gm_attention.representations.so2.complex_dtype] and
[`real_dtype`][gm_attention.representations.so2.real_dtype] give the paired
complex and real dtypes. A frequency must be a nonnegative `int`
(`bool` is rejected), checked by
[`validate_frequency`][gm_attention.representations.so2.validate_frequency].

::: gm_attention.representations.so2

## Field types and feature fields

By the decomposition into irreducibles, a finite-dimensional real
representation of `SO(2)` is described up to equivalence by the multiplicity
`a_n` of each frequency `n`. [`SO2FieldType`][gm_attention.representations.field_type.SO2FieldType]
is that description: it is hashable, compared by value, and keeps only the
frequencies with nonzero multiplicity, in ascending order; its `dimension` is
`sum_n a_n d_n`. Besides the constructor it can be built with `trivial`
(scalar channels only) and `uniform` (equal multiplicity at every frequency up
to a maximum), and rescaled with `scaled`. `common_frequencies` returns the
frequencies two types share, the only ones at which an `SO(2)`-equivariant
linear map between them has parameters, since an intertwiner vanishes between
inequivalent irreps.
[`scale_field_type`][gm_attention.representations.field_type.scale_field_type]
builds a hidden type from a configuration: a base spectrum such as
`{0: 1, 1: 1, 2: 1}` times a width. Its keys may be integers or decimal-integer
strings, which is how OmegaConf delivers YAML integer keys, and the experiment
model builders use it to read their field types from the configs.

[`SO2FeatureField`][gm_attention.representations.field_type.SO2FeatureField]
holds the coefficients of a field of a given type as one tensor per
frequency: `(*batch_shape, a_0)` at frequency 0 and `(*batch_shape, a_n, 2)`
at `n >= 1`. The constructor checks that the keys are exactly the frequencies
of the type and that the tensors share batch shape, dtype (`float32` or
`float64`) and device. `to_flat` and `from_flat` convert to and from a single
channel vector of size `dimension`, ordered by ascending frequency and, within
a frequency, with the multiplicity index outside the irrep component index;
`SO2FieldType.slices()` gives each frequency's channel range in that layout.
`apply_rotation(angles)` is the bare action `rho(r_theta)`, as used to
transport a neighbor's features into a target node's gauge, and
`gauge_transform(angles)` is the transformation law `rho(r_{-theta})` that a
gauge-equivariant layer must respect. In both, `angles` must have the dtype of
the field and broadcast against its batch shape.

::: gm_attention.representations.field_type

## Dense reference implementations

The package does not ship dense implementations of the intertwiner spaces
`Hom_SO(2)(rho_n, rho_m)` or of the steerable kernel bases. The real-matrix
transcriptions of the two constraints, written for transparency rather than
speed, live in `tests/support/intertwiners.py` and `tests/support/bases.py`,
and only the tests import them: they are the ground truth against which
[`SO2Linear`][gm_attention.nn.linear.SO2Linear] and
[`SO2SteerableKernel`][gm_attention.nn.kernels.SO2SteerableKernel] are checked
(see [Testing](../development/testing.md)). The constraints and their
solutions are derived in
[Intertwiners between arbitrary representations](../math/so2_representations.md#intertwiners-between-arbitrary-representations)
and
[Steerable transformations between arbitrary representations](../math/so2_representations.md#steerable-transformations-between-arbitrary-representations),
and in Appendix B.1 of the paper.
