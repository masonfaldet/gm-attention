# Neural network layers

Gauge-equivariant `torch.nn.Module`s built on the [representations](representations.md) and
[discretization](discretization.md) layers.

Every layer maps `SO2FeatureField`s to `SO2FeatureField`s. A field stores one tensor per frequency `n`:
`(*batch, a_n)` at `n = 0` and `(*batch, a_n, 2)` at `n >= 1`, with `a_n` the multiplicity of the
frequency-`n` irrep. A layer that needs the phase of a non-scalar copy reads the trailing pair as a complex
scalar. These are the fast paths: they act on this layout frequency by frequency. The dense real-arithmetic
references they are checked against (the intertwiner and steerable-kernel bases) are test code and live in
`tests/support`; see [Testing](../development/testing.md#the-support-package).

The layers that aggregate over a neighborhood take their geometry as an `EdgeGeometry`; `SO2Convolution`,
`SO2Attention`, `SO2MHSA` and `SO2TransformerBlock` also accept a `SphericalDiscretization`, from which the
layer takes `edges_within(cutoff)`. Edges are stored target-major with a CSR-style `row_ptr` (see
[Edge geometry](discretization.md#edge-geometry)), and a node field has batch shape `(n_nodes,)` with
`n_nodes = row_ptr.shape[0] - 1`. Every layer that sums edge-level messages back to nodes does so with
`torch.segment_reduce` against `row_ptr` -- not a COO-style scatter; the fused kernels take the same
`row_ptr`. Neighbors are weighted by an aggregation window of the geodesic distance (`"hard"` or
`"quartic"`) and by the tangent-space quadrature weight. The package `gm_attention.nn` re-exports
`SO2Linear`, `SO2SteerableKernel`, `SO2Convolution`, `SO2Attention`, `SO2MHSA`, `SO2LayerNorm`,
`SO2Dropout`, `SO2TransformerBlock` and `WindowKind`; everything else is imported from its module.

Several modules carry opt-in fast paths selected by `GM_*` environment variables, collected under
[Fast-path switches](#fast-path-switches). Without them every layer runs through plain PyTorch.

## Segment reductions

`gm_attention.nn._scatter` is private plumbing, but it fixes the conventions every aggregating layer shares.
`gather_to_edges` copies a node field to edge shape along `edges.source_index` or `edges.target_index`, with
`index_select` so that its backward is a plain scatter-add. `scatter_add_to_nodes` and `scatter_max_to_nodes`
reduce edge-level values to nodes with `torch.segment_reduce` against `row_ptr`; the values must be in the
target-major edge order that `row_ptr` indexes. A node with no incident edges sums to `0` and has maximum
`-inf`. The maximum is what the attention softmax subtracts before exponentiating.

::: gm_attention.nn._scatter

## Linear layers

The intertwiner constraint as a module: a pointwise, position-independent equivariant linear map between
field types, `W rho_in(g) = rho_out(g) W` for all `g` in `SO(2)`. The map is block diagonal in frequency.
At frequency `0` the block is a real `(b_0, a_0)` matrix; at `n >= 1` it is a complex `(b_n, a_n)` matrix,
stored as a real pair on a trailing axis of size 2. A frequency present in the input type but not in the
output type contributes nothing and gets no parameter; a frequency present only in the output type is
output as zero. A bias exists only on the frequency-`0` channels, the one place equivariance permits it.
The constraint and its solution are in
[Intertwiners between arbitrary representations](../math/so2_representations.md#intertwiners-between-arbitrary-representations);
the paper states them in Appendix B.1.

::: gm_attention.nn.linear

## Steerable kernels

The position-dependent value transformation `W_V(v)` with
`W_V(gv) = rho_out(g) W_V(v) rho_in(g)^{-1}`, evaluated on every edge: one message per edge, from source
features gathered to edge shape and transported into the target gauge with `transport_field`. A fixed Gaussian
basis of `(r/cutoff)^2` feeds an MLP trunk of `Linear`/`SiLU` pairs, and two linear heads read the trunk's
features out as radial coefficients: a real head for the `(0, 0)` block and a complex head for every other
`(n, m)` pair, holding a `linear` coefficient and, for `n, m >= 1`, an `antilinear` one. Each coefficient
of the complex head multiplies a harmonic `e^{ik phi}` tabulated in the `EdgeGeometry` and carries the factor
`(r/cutoff)^|k|`; the real `(0, 0)` block has neither. The kernel is therefore evaluated uniformly on every
edge, coincident points included: at `r = 0` what remains is an intertwiner, and no branch is taken on
`target_index == source_index`. A learned pointwise self map is the aggregation layer's concern
(`self_interaction`), added outside the quadrature sum. Every pair of an input frequency with an output
frequency contributes, so the edge geometry must tabulate harmonics up to the sum of the two top frequencies,
and no edge may lie beyond `cutoff`. The parameterization is derived in
[Steerable transformations between arbitrary representations](../math/so2_representations.md#steerable-transformations-between-arbitrary-representations)
and stated in the paper, Appendix B.1.

`forward_factored` and `forward_factored_aggregated` evaluate the same kernel in a different order. The
readout is linear in the trunk's features `h`, so `(sum_t h_t W_t) v = sum_t h_t (W_t v)`: the channel mixing
`W_t v_j` is paid once per source node (`node_products`) and only combined on the edges. The two orders agree
to summation order. The layers take this path when `GM_FACTORED_KERNEL` is set (`factored_enabled`); the
paper, Section 6, describes the regrouping. Two further switches in this module,
`GM_SPLIT_K_BACKWARD=0` and `GM_APPLY_BACKWARD=0`, replace hand-written backward passes by autograd's.

::: gm_attention.nn.kernels

## Depthwise kernels

The steerable kernel restricted to `m = n` and to diagonal copy pairs: one steerable kernel per irrep copy,
acting on that copy alone, the equivariant counterpart of a `groups = channels` convolution. A scalar copy
has one real radial function. A copy of frequency `n >= 1` is `A(r) z + (r/c)^{2n} B(r) e^{2 i n phi} conj(z)`
with complex radial functions `A` and `B`, so the parameter count is linear in the multiplicity. The radial
embedding and trunk are those of the general kernel. The edge geometry must tabulate harmonics up to twice the
top frequency. `SO2Convolution(depthwise=True)` installs this kernel in place of the general one, which
requires the input and output types to agree; the paper defines the depthwise convolution in Appendix D.2.
`stacked_coefficients` reads out the coefficients of several such kernels at once as grouped matmuls over
their stacked parameters.

::: gm_attention.nn.depthwise

## Convolution

GM-convolution, the trivial-affinity case of the discretized aggregation (the paper, Section 4; see
[The discretized aggregation](../math/discretization.md#the-discretized-aggregation)),
`y_i = sum_{j in N_i} c(v_ij) w_ij V(v_ij) rho_in(g_{i<-j}) f_j`. The per-edge term, transport included, is
the steerable kernel; the layer supplies the gather of source features, the aggregation window `c`, the
tangent-space quadrature weight `w_ij` (`edges.tangent_quadrature_weights`) and the segment sum over
`row_ptr`. The sum is unnormalized unless `normalize=True`, which divides each target's sum by its window
mass `sum_j c w` (one invariant scalar per target, so equivariance is preserved). With
`self_interaction="linear"` a learned pointwise intertwiner of the node's own feature is added after the
aggregation, outside the quadrature weight, the window and the normalization.
`evaluate_aggregation_window` is the window shared by the convolution, the attention and the grid-transfer
blocks: gauge invariance forces `c` to depend on the geodesic distance `r` alone, and the two windows are
the indicator `1[r <= radius]` and the quartic `(1 - (r/radius)^2)^2`. `forward_chunked` evaluates a
range of target nodes at a time; each chunk holds whole neighborhoods, so the chunks sum to the result of
`forward` up to summation order, and in training mode each chunk runs under a non-reentrant checkpoint.

::: gm_attention.nn.convolution

## Attention

GM-self-attention: the attention case of the discretized aggregation (the paper, Section 4;
[GM-Attention](../math/gm_aggregation.md#gm-attention)). [`SO2Attention`][gm_attention.nn.attention.SO2Attention]
is a single head with intertwiner query and key maps `rho_in -> rho_attn`
([`SO2Linear`][gm_attention.nn.linear.SO2Linear]) and a steerable value kernel `rho_in -> rho_out`
([`SO2SteerableKernel`][gm_attention.nn.kernels.SO2SteerableKernel]); over the neighbors within `cutoff`,

```
s_ij     = <W_Q f_i, rho_attn(g_{i<-j}) W_K f_j> / sqrt(dim rho_attn)
alpha_ij = c(v_ij) w_ij exp(s_ij) / sum_k c(v_ik) w_ik exp(s_ik)
y_i      = sum_j alpha_ij V(v_ij) rho_in(g_{i<-j}) f_j
```

Queries and keys are projected once per node, and the keys are transported per edge, which is exact because
`W_K` intertwines. The softmax runs in the log domain: `s_ij + log c + log w` is shifted by each target's
maximum (a segment max, detached) before exponentiation and normalized by a segment sum, and an edge outside
the window gets weight zero. [`SO2MHSA`][gm_attention.nn.attention.SO2MHSA] splits the input type evenly across
heads -- the multiplicity of every present frequency must be divisible by `num_heads` -- gives each head
`rho_attn = rho_out` of `1/num_heads` of each multiplicity, concatenates the heads' outputs head-major along
each frequency's multiplicity axis and fuses them with a final intertwiner `field_type_in -> field_type_in`.
With `batched=True` all heads run in one pass over their stacked parameters: the same function and
parameters in a different summation order. `forward_chunked` evaluates a range of target nodes at a time as
in the convolution; in training mode each chunk is recomputed in the backward unless
`GM_CHUNK_REMATERIALIZE=0`. Under `GM_FUSED_ATTENTION` the scores and softmax are a single fused op (see
[Fused attention](#fused-attention-and-depthwise-convolution)).

::: gm_attention.nn.attention

## Fixed-grid reference

The attention evaluated on a fixed tensor-product grid from a
[`TensorGridStencil`][gm_attention.discretization.stencil.TensorGridStencil]
rather than from the edge set, on the dense `(..., n_latitude, n_longitude,
channels)` layout.
[`steerable_kernel_matrices`][gm_attention.nn.fixed_grid.steerable_kernel_matrices]
turns the value kernel into one `(d_out, d_in)` matrix per slot from one
evaluation of its radial coefficients, phases and radial powers per slot, the
kernel's block formula assembled into a real matrix, so the radial MLP runs
`n_longitude` times less often than per edge (the tests check the matrices
against the kernel's forward on the basis vectors of its input type);
[`attention_fixed_grid`][gm_attention.nn.fixed_grid.attention_fixed_grid]
and [`mhsa_fixed_grid`][gm_attention.nn.fixed_grid.mhsa_fixed_grid] gather
each output row's slots along longitude, transport the keys by the slot's
phase, take the softmax over the slots and apply the slot matrices under the
attention weights. They agree with the graph path on the same discretization
to summation order, gradients included, and are exactly equivariant to the
grid's own rotations (a longitude shift). They are the reference a stencil
kernel on this layout has to reproduce, not a fast path.

::: gm_attention.nn.fixed_grid

## Nonlinearities

[`SO2ScalarGatedNonlinearity`][gm_attention.nn.nonlinearities.SO2ScalarGatedNonlinearity] is the one
nonlinearity: a pointwise self map `field_type -> field_type`. Scalar (frequency-`0`) channels pass through
`SiLU`. Every nonzero frequency gets one gate per multiplicity copy, a learned affine function of the raw
scalar channels followed by an activation, multiplying both real coordinates of the copy; a gauge-invariant
scalar commutes with `rho_n`, so the map is equivariant. The field type must contain frequency `0`. The
activation is `gate="silu"` or `gate="double_sigmoid"` (`2 * sigmoid`, strictly positive and bounded in
`(0, 2)`), the form written in the paper, Appendix B.2. The argument `eps` is accepted for signature
compatibility and unused.

::: gm_attention.nn.nonlinearities

## Normalization

[`SO2LayerNorm`][gm_attention.nn.normalization.SO2LayerNorm] is a merged equivariant layer normalization
(after Equiformer; the paper, Appendix B.2): one scalar denominator shared by every frequency, pooled with
equal weight per frequency from the mean-square magnitudes; only frequency `0` is centered, and the learned
bias exists only there. [`SO2Dropout`][gm_attention.nn.normalization.SO2Dropout] is irrep-wise dropout, with
one Bernoulli mask per multiplicity copy scaling both real coordinates together; it is the identity in eval
mode or at `p = 0`. Both are meant to sit inside a pre-norm residual block.

::: gm_attention.nn.normalization

## Stochastic depth

[`SO2DropPath`][gm_attention.nn.stochastic_depth.SO2DropPath] drops a residual branch's whole output with
probability `p` in training and rescales survivors by `1 / (1 - p)`. The output is multiplied by one scalar,
which commutes with every `rho_n`. The draw is made once per sample, not per node: a forward call carries one
sample unless it runs inside the context manager `copies(count)`, which declares the fields a sample-major
disjoint union of `count` graphs along the leading batch axis; each sample's block of nodes then receives its
own draw. `SO2TransformerBlock` drops with `SO2Dropout` instead; the segmentation model in
`experiments/stanford_2d3ds` assembles its own blocks from `SO2MHSA`, `SO2LayerNorm`, a depthwise
`SO2Convolution` and `SO2DropPath`.

::: gm_attention.nn.stochastic_depth

## Blocks

[`SO2TransformerBlock`][gm_attention.nn.blocks.SO2TransformerBlock]: a pre-norm transformer block assembled
entirely from the layers above,

```
x'  = x  + Drop(mixer(SO2LayerNorm(x), edges))
x'' = x' + Drop(SO2Linear(nonlinearity(SO2Linear(SO2LayerNorm(x')))))
```

The mixer is `SO2MHSA` by default, or a type-preserving `SO2Convolution` over the same neighborhood and
window, without feature-dependent weights, with `mixer="convolution"` (`num_heads` is then ignored). The
feed-forward branch widens `field_type` to `hidden_field_type`
(`ffn_multiplier` times `field_type`, or a type stated exactly), applies the nonlinearity `nonlinearity`
names and narrows back; `"scalar_gate"` and `"scalar_gate_double_sigmoid"` (the default) are the two
activations of `SO2ScalarGatedNonlinearity`, and `build_nonlinearity` is the single reader of that name
table. Both `Drop`s are `SO2Dropout`. Nothing new is proven equivariant here; the block is a composition of
already-equivariant pieces. The paper describes the block in Section 4 and its components in Appendix B.2.

The remaining blocks move a field between grids through bipartite edge sets built once with the
discretizations. `IcoEncode` is the bipartite form of the convolution's sum: fine-grid sources, coarse-grid
targets, the steerable kernel's per-edge term under window times quadrature weight; the coarse nodes are the
fine grid's first `n_coarse` nodes, as `build_icosphere_refinement` produces them, and `cutoff` must equal
the radius the edges from `build_icosphere_downsample_edges` were built with, which `forward` cannot check.
`IcoLowPassRestriction` is the fixed, unlearned counterpart: a normalized average of transported neighbors,
one invariant denominator per target, type-preserving and exact on scalar constants. `IcoDecode` goes the
other way in two steps: inherited fine nodes copy their coarse feature, each new fine node averages its two
parents' features transported into its gauge (edges from `build_icosphere_upsample_edges`), and a
fine-to-fine `SO2Convolution` then refines the result and performs the type change. `EquiEncode` is the
cross-grid counterpart of `IcoEncode` for two grids that share no indexing, with edges from
`build_grid_hierarchy`; `mode="learned_integral"` is the only accepted value.

::: gm_attention.nn.blocks

## Fast-path switches

Apart from the `VERBOSE` variables, each switch below is read once, when the module that defines it is first
imported, so it must be set before the first forward call. The three fused modes take `0` (off, the default),
`1` (on for CUDA tensors) or `cpu` (on everywhere, through the extension's CPU driver). A fast path
computes the same function as the PyTorch path up to summation order; a layer whose shapes, dtype or device
fall outside a fused kernel's bounds runs the PyTorch path instead.

| Variable | Module | Effect |
| --- | --- | --- |
| `GM_FACTORED_KERNEL` | `kernels` | any value other than `0` evaluates the layers' steerable kernels in the factored order |
| `GM_FUSED_PAIRS` | `fused_pairs` | the kernel's pair loop as fused terms |
| `GM_FUSED_FACTORED` | `fused_factored` | the factored kernel and its weighted segment sum as one op; acts inside the factored path, so it is reached under `GM_FACTORED_KERNEL` |
| `GM_FUSED_ATTENTION` | `fused_attention` | the scores and softmax, and the depthwise convolution, as fused ops |
| `GM_FUSED_PAIRS_VERBOSE`, `GM_FUSED_FACTORED_VERBOSE`, `GM_FUSED_ATTENTION_VERBOSE` | the same | `1` shows the compiler output when the extension is built |
| `GM_SPLIT_K_BACKWARD` | `kernels` | `0` replaces the blocked weight gradient of the radial MLP and its heads by autograd's |
| `GM_APPLY_BACKWARD` | `kernels` | `0` replaces the block contraction's backward by autograd's |
| `GM_CHUNK_REMATERIALIZE` | `attention` | `0` makes `SO2Attention.forward_chunked` keep each chunk's saved tensors instead of recomputing them in the backward |

The fused modules are C++/CUDA extensions under `src/gm_attention/nn/fused/`, compiled on first use with
`torch.utils.cpp_extension.load`, which needs `ninja` and a C++ compiler, and `nvcc` for CUDA. Their
correctness is checked in `tests/nn` against the PyTorch path. The paper, Section 6, describes the
factorizations and kernels at the level of the method.

## Fused pair terms

`GM_FUSED_PAIRS` replaces the pair loop of `SO2SteerableKernel._forward_general`. Each `(n, m, role)` term is
one kernel launch that forms its `b_m x a_n` coefficient block in registers from the trunk output and the
readout rows, applies phase and radial scale, and accumulates into the output; the backward is one launch per
term, and the weight gradient is then a single wide GEMM. The CUDA kernels are built once per block size and
trunk width met; the CPU build covers the block sides in `SUPPORTED`. `enabled` declines, and the PyTorch path
runs, when a block side exceeds `MAX_MULT` = 16 (or, on the CPU, is not one of `SUPPORTED` = 1, 2, 3, 4, 8,
16), when the trunk is wider than `MAX_TRUNK` = 16, when the dtype is not `float32` or `float64`, or when the
edge geometry does not tabulate a phase order a term needs.

::: gm_attention.nn.fused_pairs

## Fused factored kernel

`GM_FUSED_FACTORED` fuses the factored evaluation of
[the steerable kernel](#steerable-kernels). The edges are grouped by source; one thread block stages the
source's node-side mixes in shared memory, and each of its edges forms every block's combination of the
mixes with the trunk's features in registers, under its own phases and radial powers, writing only its
output row. `Factored` is the autograd function; its backward is the same grouping with the roles swapped,
and its gradients reach the trunk features and the node-side mixes, from which autograd handles the trunk,
the heads' weights and the input field. The phases and the radius ratio are the geometry's and get no
gradient. `FactoredAggregate` also performs what every layer does with the messages, the weighted sum per
target, as one op with a backward that reads each target's gradient through the target index, so no weighted
per-edge tensor exists in either direction. Both functions are once-differentiable. `enabled` declines when
the trunk width plus one exceeds `MAX_K` = 20, when the output dimension, the widest block or the packed-mix
width exceeds `MAX_OUT` = 128, `MAX_BM` = 32 or `MAX_C` = 320, when the dtype is not `float32` or `float64`, or
when there are no edges.

::: gm_attention.nn.fused_factored

## Fused attention and depthwise convolution

`GM_FUSED_ATTENTION` provides two ops, each with one thread block per target node. `Softmax` stages the node's
query, scores its edges into shared memory under the window and quadrature bias, normalizes there and returns
the attention weights; its backward forms the softmax adjoint and the score's gradients with respect to the
queries and keys. It serves `SO2Attention` (one head) and `SO2MHSA(batched=True)`; the value messages still
come from the value kernels as before. `Depthwise` is the aggregation of `SO2Convolution(depthwise=True)`:
transport, coefficient products, weighting and the segment sum in one op, the radial MLP that produces the
coefficients staying in PyTorch. The ops differentiate with respect to the flat node fields and the
coefficients only, and are once-differentiable. The source-side gradients accumulate with atomic adds on
CUDA, so their summation order is not fixed from run to run. The per-edge constants (phase tables, `r/cutoff`,
aggregation weights, softmax bias) are built once per edge set and served to chunks registered with
`register_chunk` as views of the parent's tables. `softmax_enabled` and `softmax_head_enabled` decline when
the head count (for `SO2MHSA`), the number of frequencies or the flat width exceed `MAX_H` = 8, `MAX_F` = 8
or `MAX_D` = 512, when the largest neighborhood (times the head count) exceeds `MAX_SLOTS` = 8192, or when
the backward's shared memory exceeds the device's limit; `enabled_depthwise` applies the same bounds without the
neighborhood one.

::: gm_attention.nn.fused_attention
