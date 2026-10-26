# Discretization

Fixed geometry for gauge-equivariant layers on the sphere: a quadrature grid,
a tangent-plane gauge at each node, and the resulting per-edge log map /
parallel transport / angular harmonics. All of it is computed once per grid
and reused by every layer and every forward pass.

The objects below realize the spherical discretization
`D = (X, Q_node, E, Q_edge, W, P)` of the paper, Section 4; the mathematics is
on [Discretization](../math/discretization.md#the-discretization-data) and
[Sphere geometry](../math/sphere_geometry.md).

| Paper, Section 4 | Object | Fields |
|---|---|---|
| nodes `X`, node weights `Q_node` | [`SphericalGrid`][gm_attention.discretization.grid.SphericalGrid] | `nodes`, `quadrature_weights` |
| gauges `W` | tangent frames | tensor of shape `(n_nodes, 3, 2)` |
| edge vectors `E`, edge weights `Q_edge`, transporters `P` | [`EdgeGeometry`][gm_attention.discretization.edge_geometry.EdgeGeometry] | `log_coordinates`, `tangent_quadrature_weights`, `transport_angles` |
| the tuple `D`, with radius-subset caching | [`SphericalDiscretization`][gm_attention.discretization.discretization.SphericalDiscretization] | `grid`, `frames`, `edge_geometry` |

Conventions used throughout. An edge `i <- j` has target `i` and source `j`:
the aggregation at `p_i` reads the field at `p_j`. A frame `E_i = [e_1  e_2]`
has orthonormal columns in `T_{p_i} S^2` with `e_1 x e_2 = p_i`, and every
tangent coordinate at `p_i` (log coordinates, polar angles) is expressed in
`E_i`. Angles are in radians, distances are geodesic distances on the unit
sphere, and support radii lie in `(0, pi)`.

## Spherical grids

A [`SphericalGrid`][gm_attention.discretization.grid.SphericalGrid] holds
`nodes` of shape `(n_nodes, 3)` on the unit sphere and strictly positive
`quadrature_weights` of shape `(n_nodes,)` that sum to `4 pi`; both are
validated at construction, in `float32` or `float64`. The optional
`exact_nodes` are the same nodes in `float64` on the CPU, as the grid builder
computed them before the cast to the working dtype; edge membership is decided
from them when present (see [Edge geometry](#edge-geometry)). The builders
below attach them. `tile(count)` returns the node set of `count` disjoint
copies, with the weights divided by `count`, so that `count` samples run
through a layer as one graph.
[`median_nearest_neighbor_distance`][gm_attention.discretization.grid.median_nearest_neighbor_distance]
estimates the node spacing of a grid, from which cutoffs can be set relative
to the grid's own resolution.

::: gm_attention.discretization.grid

## Tensor-product grids

[`build_torch_harmonics_grid`][gm_attention.discretization.grids.torch_harmonics.build_torch_harmonics_grid]
takes the product of a colatitude quadrature rule from
`torch_harmonics.quadrature` (`"legendre-gauss"`, `"lobatto"` or
`"equiangular"`, the last being Clenshaw-Curtis quadrature at equispaced
colatitudes) with equispaced longitudes starting at 0. Nodes are ordered
latitude-major with colatitude ascending, so node `a * n_longitude + b` has
colatitude index `a` and longitude index `b`. The weight of a node is the
colatitude weight times `2 pi / n_longitude`. The `"lobatto"` and
`"equiangular"` rules include both poles, so each pole appears as
`n_longitude` colocated nodes that share the pole's weight; `"legendre-gauss"`
has no pole nodes. Icosphere grids are built by
[`build_icosphere_grid`][gm_attention.discretization.grids.icosphere.build_icosphere_grid],
described [below](#icosphere-grids-and-multiresolution) because that module
also builds on the geometry objects that follow.

::: gm_attention.discretization.grids.torch_harmonics

## Gauge frames

A tangent-plane basis (gauge) per node, as a tensor of shape `(n_nodes, 3, 2)`,
and the primitives for regauging it -- the operation the gauge-equivariance
tests check invariance against.
[`build_tangent_frames`][gm_attention.discretization.frames.build_tangent_frames]
is deterministic: the coordinate axis least aligned with a node is projected
into its tangent plane to give `e_1`, and `e_2 = p x e_1`. No continuous
tangent frame field exists on the sphere, so the choice is discontinuous; the
discontinuities are changes of gauge, to which a gauge-equivariant layer is
insensitive.
[`build_coordinate_frames`][gm_attention.discretization.frames.build_coordinate_frames]
gives the spherical-coordinate frames `(e_theta, e_phi)` instead, which are
covariant under rotations about the polar axis and undefined at the poles; a
tensor-product grid in this gauge folds along longitude (see
[Tensor-grid stencils](#tensor-grid-stencils)).
[`rotate_tangent_frames`][gm_attention.discretization.frames.rotate_tangent_frames]
applies `E_i' = E_i r(theta_i)`, which sends the coordinates `x` of a tangent
vector to `r(theta_i)^T x`; the corresponding transformation of feature
coefficients is on
[Conventions](../math/conventions.md#behavior-under-gauge-transformations).
The frame as the matrix of the fiber identification is on
[Gauge coordinates](../math/sphere_geometry.md#gauge-coordinates).

::: gm_attention.discretization.frames

## Edge geometry

Per-directed-edge log-map coordinates, parallel-transport angles, and the
tabulated `exp(i k phi)` / `exp(i k gamma)` harmonics steerable kernels read
directly instead of computing trigonometry per forward pass. Edges are
stored sorted target-major in a CSR-style layout: `row_ptr` marks each
node's contiguous incident-edge range, which is exactly what
`torch.segment_reduce` needs to sum edge-level messages back to nodes (see
[Neural Network Layers](nn.md)) -- no COO-style scatter involved.

An [`EdgeGeometry`][gm_attention.discretization.edge_geometry.EdgeGeometry]
stores, for every edge `i <- j`:

- `edge_index` of shape `(2, n_edges)`: targets in row 0, sources in row 1;
- `log_coordinates` of shape `(n_edges, 2)`: the coefficients of
  `log_{p_i}(p_j)` in the target frame `E_i`, and `distances`, the geodesic
  distance `r_ij`, equal to their norm
  ([Logarithm map](../math/sphere_geometry.md#logarithm-map));
- `polar_angles`: the angle `phi_ij` of the log coordinates in the target
  frame (zero on a self-loop, whose log coordinates vanish);
- `transport_angles`: the angle `gamma_{i<-j}` of the rotation
  `E_i^T P_{p_i <- p_j} E_j` taking source-frame coordinates to target-frame
  coordinates ([Parallel transport](../math/sphere_geometry.md#parallel-transport));
- `tangent_quadrature_weights`: `q_j r_ij / sin r_ij`, the source node's
  spherical weight as a Lebesgue weight on `T_{p_i} S^2` (the ratio is 1 at
  `r_ij = 0`), so that edge sums approximate integrals;
- `angular_phases` and `transport_phases`, complex tables of shape
  `(n_edges, max_frequency + 1)` holding `exp(i k phi_ij)` and
  `exp(i k gamma_{i<-j})`. `angular_harmonic(order)` and
  `transport_harmonic(order)` serve negative orders as conjugates and raise
  for an order beyond `max_frequency`.

`max_frequency` defaults to `DEFAULT_MAX_FREQUENCY = 4`. A layer from input
types `I_in` to output types `I_out` needs orders up to
`max(I_in) + max(I_out)`, so the default covers fields of frequency up to two
on both sides.

[`compute_edge_geometry`][gm_attention.discretization.edge_geometry.compute_edge_geometry]
finds every ordered pair of a grid within a coverage radius (self-loops
included) by a KD-tree search on `float64` CPU coordinates, so that the edge
set does not depend on the training device, and passes the pairs to
[`edge_geometry_from_index`][gm_attention.discretization.edge_geometry.edge_geometry_from_index],
which derives everything else and does not care how the edges were chosen.
[`compute_bipartite_edge_geometry`][gm_attention.discretization.edge_geometry.compute_bipartite_edge_geometry]
does the same between two independent grids, each indexed locally, with the
quadrature weights of the source side and the gauge of the target frames; it
raises if a target has no source within the cutoff.

**Cutoff rule.** A structured grid places whole shells of nodes exactly on a
radius, and rounding in the working dtype would split a shell by chance,
differently for a node and its image under a symmetry of the grid. By default
(`"whole-shell"`) edge membership is therefore decided from the grid's
`float64` coordinates (`SphericalGrid.exact_nodes`) with a slack
`EXACT_SLACK = 1e-10`, so a shell is kept or dropped whole; a grid without
them falls back to a dtype-sized slack
([`cutoff_tolerance`][gm_attention.discretization.edge_geometry.cutoff_tolerance]).
The environment variable `GM_CUTOFF_RULE=legacy`, read once at import,
selects the rule the paper's shallow-water runs used, so that those
checkpoints are evaluated over the edge sets they were trained on: radius
subsets and the hard aggregation window compare strictly, and the bipartite
search applies no geodesic filter. The SWE bundles' `run.sh` export it. Any
other value raises at import.
[`cutoff_rule`][gm_attention.discretization.edge_geometry.cutoff_rule]
returns the rule in force and
[`within_cutoff`][gm_attention.discretization.edge_geometry.within_cutoff] is
the comparison it selects; the Stanford 2D-3D-S training protocol records the
rule.

::: gm_attention.discretization.edge_geometry

## Spherical discretization

Bundles a grid, its frames, and edge geometry, with caching of edge subsets
at narrower support radii.
[`SphericalDiscretization.from_grid`][gm_attention.discretization.discretization.SphericalDiscretization.from_grid]
computes the frames (with `build_tangent_frames` when none are passed) and the
edge geometry out to a coverage radius, which must be less than `pi` because
logarithmic coordinates and shortest-geodesic transport are ambiguous at
antipodal points. A layer then asks for the edges inside its own support
radius with
[`edges_within`][gm_attention.discretization.discretization.SphericalDiscretization.edges_within],
which materializes the subset on first request and caches it under the radius
(rounded to 12 decimals); membership follows the cutoff rule above.
[`prepare_windows`][gm_attention.discretization.discretization.SphericalDiscretization.prepare_windows]
materializes every subset before training or `torch.compile`, so that none is
decided inside a forward pass. The constructor validates that the frames are
orthonormal and positively oriented and that the edge geometry is internally
consistent, but cannot check that the two were computed in the same gauge;
`from_grid` builds both together. `tile(count)` and `to(device, dtype)` act on
the grid, frames, coverage edges and every cached subset.

::: gm_attention.discretization.discretization

## Tensor-grid stencils

On a tensor-product grid in the coordinate gauge, a rotation about the polar
axis by a multiple of `2 pi / n_longitude` is an isometry that maps the grid
onto itself and every frame onto a frame, so the geometry of an edge
`(ho, wo) <- (hi, wi)` depends only on `(ho, hi, (wi - wo) mod n_longitude)`.
[`build_tensor_grid_stencil`][gm_attention.discretization.stencil.build_tensor_grid_stencil]
folds the edges within a support radius into a
[`TensorGridStencil`][gm_attention.discretization.stencil.TensorGridStencil]:
one representative edge per `(ho, hi, dw)` class (the edges whose target has
longitude `0`), the class's source row and longitude offset, and the circular
longitude arcs each `(ho, hi)` pair forms. It checks the fold against every
edge of the full geometry and raises where it fails, as it does for the
default frames or for a grid with pole nodes; the `"legendre-gauss"` grid has
none, and
[`build_tensor_grid_discretization`][gm_attention.discretization.stencil.build_tensor_grid_discretization]
builds it in the coordinate gauge. A layer on the dense
`(..., n_latitude, n_longitude, channels)` layout then reads the source of
slot `s` at target `(ho, wo)` from `(source_row[s], (wo + offset[s]) mod
n_longitude)`, the access pattern of torch-harmonics' neighbourhood kernels,
with `n_longitude` times fewer geometry rows than the edge set;
[Fixed-grid reference](nn.md#fixed-grid-reference) evaluates the attention
this way one row at a time, and the [stencil path](nn.md#stencil-path) every
row at once from a bucketed plan; a level of a grid hierarchy
([`TensorGridLevel`][gm_attention.discretization.grids.equiangular.TensorGridLevel])
can carry the plans of its radii beside its discretization.

::: gm_attention.discretization.stencil

## Icosphere grids and multiresolution

[`build_icosphere_grid`][gm_attention.discretization.grids.icosphere.build_icosphere_grid]
takes the vertices of a subdivided regular icosahedron (generated by
`trimesh`), projected onto the unit sphere: `10 * 4**s + 2` nodes at
subdivision level `s`, so 10,242 at level 5. The weight of a node is a third
of the area of each incident triangle, rescaled so that the weights sum to
exactly `4 pi`.

The remaining builders chain levels `s`, `s - 1`, ... for the encoder and
decoder blocks [`IcoEncode`][gm_attention.nn.blocks.IcoEncode] and
[`IcoDecode`][gm_attention.nn.blocks.IcoDecode]. Subdivision keeps every
existing vertex at its index and appends the new ones, so the nodes of level
`s` are the first nodes of level `s + 1`, in order, with the same positions;
each new node is the sphere-projected midpoint of an edge of the coarser mesh.
[`build_icosphere_refinement`][gm_attention.discretization.grids.icosphere.build_icosphere_refinement]
builds the two grids, verifies this prefix property, and returns an
[`IcosphereRefinement`][gm_attention.discretization.grids.icosphere.IcosphereRefinement]
recording each new node's two coarse parents and a representative geodesic
length of a coarse edge, which stands in for the coarse spacing when choosing
an aggregation radius. Two bipartite edge sets follow from it:
[`build_icosphere_downsample_edges`][gm_attention.discretization.grids.icosphere.build_icosphere_downsample_edges]
restricts the fine grid's own edges to coarse targets, with the fine (source)
grid's quadrature weights, and
[`build_icosphere_upsample_edges`][gm_attention.discretization.grids.icosphere.build_icosphere_upsample_edges]
gives each new node exactly two edges, from its parents.

[`build_icosphere_multiresolution`][gm_attention.discretization.grids.icosphere.build_icosphere_multiresolution]
assembles a chain of `depth + 1` levels, finest first, into an
[`IcosphereMultiresolution`][gm_attention.discretization.grids.icosphere.IcosphereMultiresolution]:
`levels` holds one `SphericalDiscretization` per level and `hops[k]`, an
[`IcosphereHop`][gm_attention.discretization.grids.icosphere.IcosphereHop],
holds the refinement and the `down_edges` / `up_edges` between `levels[k]` and
`levels[k + 1]`. Every level uses the matching prefix of one frame tensor
(`finest_frames`, or `build_tangent_frames` on the finest grid). A level's
cutoff is the coarse-edge length of its hop, except that the coarsest level
uses `coarsest_cutoff`; `local_cutoffs` overrides a level's cutoff and
`down_cutoffs` a hop's fine-to-coarse radius, the latter to evaluate a model
at a resolution other than the one it was fitted at. The paper's depth-1
shallow-water model (Section 5.1) is such a chain with `depth=1`.

::: gm_attention.discretization.grids.icosphere

## Grid hierarchies

Chains of independent grids, for models such as the SegFormer-shaped
$SO(S^2)$ model of the paper (Section 5.2, Appendix D.2), which runs on a
tensor-product input level and icosphere stage levels. The module is named
for the tensor-product grids it was written for, but
[`build_grid_hierarchy`][gm_attention.discretization.grids.equiangular.build_grid_hierarchy]
takes prebuilt `SphericalGrid`s of any kind, and kinds mix freely, because
every transfer reads each side's own nodes, frames and quadrature weights.
Levels are listed finest first and each must have strictly fewer nodes than
the one before it.

A coarse tensor-product grid is not a prefix of the finer one, so every
transfer is a radius-based bipartite edge set built by
`compute_bipartite_edge_geometry`. A radius rather than a fixed neighbor
count includes every colocated copy of a pole, whose split quadrature weights
then sum back to the pole's mass.

A [`TensorGridLevel`][gm_attention.discretization.grids.equiangular.TensorGridLevel]
carries a level's grid, frames, a descriptive `shape` and `grid_kind`, and the
same-grid `SphericalDiscretization` for the operators that run on it (`None`
for a level nothing runs on, such as the input level). A
[`TensorGridHierarchy`][gm_attention.discretization.grids.equiangular.TensorGridHierarchy]
holds the levels and three families of transfers: `down_edges[k]` restricts
level `k` onto level `k + 1`, `up_edges[k]` interpolates level `k + 1` onto
level 0, and `lateral_edges[k]` interpolates level `k + 2` onto level `k + 1`
(empty unless `lateral_cutoffs` is given). `restrict_cutoffs` and
`interpolate_cutoffs` give one radius per hop, and `local_cutoffs` the support
radii each level prepares, or `None` for no same-grid operator. `tile(count)`
and `to(device)` act on every level and transfer, so that `count` panoramas
run as one graph with no edge between samples.

::: gm_attention.discretization.grids.equiangular
