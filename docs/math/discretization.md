# Discretization

This is the direct math counterpart of [`gm_attention.discretization`](../api/discretization.md).

In order to numerically implement an $SO(S^2)$-aggregation, the spherical signal and the integrals
defining the aggregation must be discretized. This section outlines the data constituting a
discretization (the paper, Section 4). The quadrature grids built by the code, documented in the
[API reference](../api/discretization.md), are the icosphere and the Legendre–Gauss, Lobatto and
equiangular tensor-product grids of torch-harmonics (Bonev et al. [BKH+23]).

## The discretization data

We begin with a set of nodes $X := \{p_i\}_{i=1}^N \subset S^2$. For each $p_i$, choose an arbitrary
frame $E_i \in \mathbb{R}^{3\times2}$ for $T_{p_i}S^2$ from the $SO(2)$-reduction, as in
[Gauge coordinates](sphere_geometry.md#gauge-coordinates), and set $E := \{E_i\}_{i=1}^N$. No
smoothness condition needs to be imposed on this selection because the $SO(S^2)$-aggregation is gauge
equivariant and coordinate independent. For a signal $f \in \Gamma(A_{\mathrm{in}})$, we denote by
$f_i \in \mathbb{R}^{c_{\mathrm{in}}}$ the coordinate representation of $f(p_i)$ in the gauge
determined by $E_i$.

Because the $SO(S^2)$-aggregation $A^c_{a,\beta,V}f$ of [GM-aggregation](gm_aggregation.md) is
defined pointwise by integrals over tangent spaces, we introduce an edge set describing which sampled
values contribute to each output. For $i,j \in \{1,\ldots,N\}$, define

$$
v_{ij} := E_i^\top \log_{p_i}(p_j) \in \mathbb{R}^2, \qquad r_{ij} := \|v_{ij}\| = d_{S^2}(p_i,p_j).
$$

We assume that $\mathrm{supp}(c) \subset B_R(0)$ for some $R < \pi$. The directed edge set is

$$
\mathcal{E} := \{i \leftarrow j : c(v_{ij}) > 0\},
$$

so that $i \leftarrow j \in \mathcal{E}$ precisely when the signal at $p_j$ contributes to the aggregation
at $p_i$. We define the incoming neighborhood of $p_i$ by
$\mathcal{N}_i := \{j \in \{1,\ldots,N\} : i \leftarrow j \in \mathcal{E}\}$.

The second piece of data consists of node quadrature weights $W_{\mathrm{node}} := \{w_i\}_{i=1}^N$,
which are used to discretize the defining aggregation and affinity-normalization integrals. To make
use of established spherical quadrature rules, we require

$$
w_i > 0, \qquad \sum_{i=1}^N w_i = 4\pi, \qquad \int_{S^2} g(p)\, d\mathrm{Vol}(p) \approx \sum_{i=1}^N w_i\, g(p_i)
$$

for suitable functions $g : S^2 \to \mathbb{R}$, where $d\mathrm{Vol}$ is the Riemannian volume measure
on $S^2$. Since $\exp_p^*(d\mathrm{Vol})(v) = \dfrac{\sin\|v\|}{\|v\|}\, dv$, integration against
Lebesgue measure on $T_pS^2$ is related to spherical integration by

$$
dv = \frac{\|v\|}{\sin\|v\|}\, \exp_p^*(d\mathrm{Vol})(v).
$$

The induced edge quadrature weights are therefore

$$
W_{\mathrm{edge}} := \{w_{ij} : i \leftarrow j \in \mathcal{E}\}, \qquad w_{ij} := w_j\, \frac{r_{ij}}{\sin r_{ij}}.
$$

When $r_{ij} = 0$, the ratio $r_{ij}/\sin r_{ij}$ is defined by its continuous extension to equal one.

For each edge, we also record the coordinate representation of parallel transport

$$
\Gamma := \{g_{i\leftarrow j} : i\leftarrow j \in \mathcal{E}\}, \qquad g_{i\leftarrow j} := E_i^\top\, P^{TS^2}_{p_i\leftarrow p_j}\, E_j \in SO(2),
$$

where $P^{TS^2}_{p_i\leftarrow p_j}$ is represented by the rotation in $SO(3)$ derived in
[Parallel transport](sphere_geometry.md#parallel-transport). The coordinate representation of the
induced transport in $A_{\mathrm{in}}$ is therefore $\rho_{\mathrm{in}}(g_{i\leftarrow j})$.

The collection $D := (X, W_{\mathrm{node}}, \mathcal{E}, W_{\mathrm{edge}}, E, \Gamma)$ constitutes a
discretization of the sphere and its geometry for the purpose of implementing the
$SO(S^2)$-aggregation. Once the aggregation window and a frame-selection convention are fixed,
providing $X$ and $W_{\mathrm{node}}$ is sufficient to compile the remaining data. This compilation
need only be performed once for each spherical grid.

## The discretized aggregation

Writing $\big(A^c_{a,\beta,V}f\big)_i$ for the coordinate representation of $\big(A^c_{a,\beta,V}f\big)(p_i)$
in the gauge determined by $E_i$, the discretized aggregation is

$$
\big(A^c_{a,\beta,V}f\big)_i \approx D\big(A^c_{a,\beta,V}f\big)_i := \frac{\displaystyle\sum_{j\in\mathcal{N}_i} c(v_{ij})\, w_{ij}\, \beta\big(a(f_i,\rho_{\mathrm{in}}(g_{i\leftarrow j})f_j)\big)\, V(v_{ij})\,\rho_{\mathrm{in}}(g_{i\leftarrow j})f_j}{\displaystyle\sum_{j\in\mathcal{N}_i} c(v_{ij})\, w_{ij}\, \beta\big(a(f_i,\rho_{\mathrm{in}}(g_{i\leftarrow j})f_j)\big)}.
$$

A key consequence of this formulation is that the discretized aggregation is agnostic to the
particular spherical sampling and quadrature rule. Indeed, the learned templates $a$ and $V$ do not
depend on the number, ordering, or arrangement of the sampled nodes. Given two discretizations
$D_1$ and $D_2$, one may therefore fit $a$ and $V$ using a loss of the form
$\mathcal{L}\big(D_1(A^c_{a,\beta,V}f)\big)$ and then evaluate the fitted templates on the second
discretization through $D_2\big(A^c_{\hat a,\beta,\hat V}f\big)$. Here the input field must be sampled on
the node set of the corresponding discretization. At finite resolution, the two discrete outputs need
not agree exactly because they use different quadrature approximations. Nevertheless, both
approximate the same continuous operator. In this sense, the $SO(S^2)$-aggregation naturally
supports grid transfer. This allows for the evaluation at resolutions and node distributions that
differ from those used during training. The zero-shot transfer study of the paper, Section 5.1,
evaluates a model fitted on the level-5 icosphere in this way, on the level-6 icosphere and on a
$128\times256$ Legendre–Gauss grid.

## Grid transfer and SO(3)-equivariance

Grid transfer also provides a direct way to test equivariance under arbitrary elements of $SO(3)$
without restricting attention to rotations that preserve a fixed node set. Let
$D = (X,W_{\mathrm{node}},\mathcal{E},W_{\mathrm{edge}},E,\Gamma)$ and fix $\phi \in SO(3)$. Define
the rotated discretization by $D_\phi := (\phi X, W_{\mathrm{node}}, \mathcal{E}, W_{\mathrm{edge}}, \phi E, \Gamma)$,
where $\phi X := \{\phi p_i\}_{i=1}^N$, $\phi E := \{\phi E_i\}_{i=1}^N$. The edge indices and
quadrature weights are unchanged because $\phi$ preserves geodesic distances and Riemannian
volume. Moreover, $(\phi E_i)^\top P^{TS^2}_{\phi p_i \leftarrow \phi p_j}(\phi E_j) = E_i^\top P^{TS^2}_{p_i\leftarrow p_j}E_j$,
so the coordinate transport elements in $\Gamma$ are also unchanged. Equivariance of the discretized
aggregation is then expressed by

$$
\phi\rhd \Big(D\big(A^c_{a,\beta,V}f\big)\Big) = D_\phi\big(A^c_{a,\beta,V}(\phi \rhd f)\big).
$$

The left-hand side denotes the pushforward of the discrete output from $X$ to $\phi X$.

## References

- [JM18] Fredrik Johansson and Marc Mezzarobba. *Fast and rigorous arbitrary-precision computation
  of Gauss–Legendre quadrature nodes and weights*. SIAM Journal on Scientific Computing,
  40(6):C726–C747, 2018.
- [GST21] Amparo Gil, Javier Segura, and Nico M. Temme. *Fast and reliable high-accuracy
  computation of Gauss–Jacobi quadrature*. Numerical Algorithms, 87(4):1391–1419, 2021.
- [Gau04] Walter Gautschi. *Orthogonal Polynomials: Computation and Approximation*. Numerical
  Mathematics and Scientific Computation. Oxford University Press, Oxford, 2004.
- [Sch13] Nathanaël Schaeffer. *Efficient spherical harmonic transforms aimed at pseudospectral
  numerical simulations*. Geochemistry, Geophysics, Geosystems, 14(3):751–758, 2013.
- [BKH+23] Boris Bonev, Thorsten Kurth, Christian Hundt, Jaideep Pathak, Maximilian Baust, Karthik
  Kashinath, and Anima Anandkumar. *Spherical Fourier neural operators: Learning stable dynamics
  on the sphere*. Proceedings of the 40th International Conference on Machine Learning, PMLR
  202:2806–2823, 2023.
- [BRP+25] Boris Bonev, Max Rietmann, Andrea Paris, Alberto Carpentieri, and Thorsten Kurth.
  *Attention on the sphere*. Advances in Neural Information Processing Systems, volume 38, 2025.
- [DGS77] Philippe Delsarte, Jean-Marie Goethals, and Johan Jacob Seidel. *Spherical codes and
  designs*. Geometriae Dedicata, 6(3):363–388, 1977.
- [BRV15] Andriy Bondarenko, Danylo Radchenko, and Maryna Viazovska. *Well-separated spherical
  designs*. Constructive Approximation, 41(1):93–112, 2015.
- [Wom18] Robert S. Womersley. *Efficient spherical designs with good geometric properties*. In
  Josef Dick, Frances Y. Kuo, and Henryk Woźniakowski, editors, *Contemporary Computational
  Mathematics: A Celebration of the 80th Birthday of Ian Sloan*, pages 1243–1285. Springer, Cham,
  2018.
