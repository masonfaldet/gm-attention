# Conventions and prerequisites

This page states the definitions of the GM-convolution framework of [WFVW23] on which the derivation of
GM-aggregation and GM-attention ([GM-aggregation](gm_aggregation.md)) builds. It is the minimum needed to
read the other mathematics pages; it does not cover the background theory in full.

Notation: $M$ is the base manifold,
$GM \to M$ the principal $G$-bundle encoding the chosen structure group $G$, and
$A = GM \times_\rho \mathbb{R}^c$ an associated feature vector bundle of type $\rho$.


## 1×1 GM-convolutions

Any $K_{1\times1} \in \mathrm{Hom}_G(\rho_{\mathrm{in}}, \rho_{\mathrm{out}})$ — i.e. any matrix satisfying
the intertwining condition $K\rho_{\mathrm{in}}(g) = \rho_{\mathrm{out}}(g)K,\ \forall g \in G$ — is called a
$1\times1$ GM-convolutional kernel. It determines a vector bundle $M$-morphism
$C_{K_{1\times1}} : A_{\mathrm{in}} \to A_{\mathrm{out}}$ characterized by the property that in every gauge
from the $G$-atlas and for every $p \in M$,

$$
\psi^X_{A_{\mathrm{out}},p} \circ \left(C_{K_{1\times1}}\right)_p \circ \left(\psi^X_{A_{\mathrm{in}},p}\right)^{-1} = K_{1\times1}.
$$

Well-definedness follows immediately from the intertwining condition, since it guarantees that the
change of coordinatization rule preserves the same constant matrix in every local representation.

<a id="def-1x1-gm-convolution"></a>**Definition ($1\times1$ GM-convolution [WFVW23]).** Given $K_{1\times1} \in \mathrm{Hom}_G(\rho_{\mathrm{in}}, \rho_{\mathrm{out}})$,
the associated $1\times1$ GM-convolution is the map

$$
K_{1\times1} \circledast : \Gamma(A_{\mathrm{in}}) \longrightarrow \Gamma(A_{\mathrm{out}}), \qquad K_{1\times1} \circledast f := C_{K_{1\times1}} \circ f. \tag{1}
$$

This operator has two immediate properties. First, it is $1\times1$: for each $p \in M$,
$(K_{1\times1} \circledast f)(p)$ depends only on $f(p)$ and involves no spatial aggregation. Second, it
exhibits weight sharing and GM-coordinate independence: with respect to any local trivialization in
the $G$-atlas, the local expression of $K_{1\times1}\circledast$ is multiplication by the *same* matrix
$K_{1\times1}$, independent of the base point $p$.

### As sections of Hom bundles

An equivalent perspective. The vector bundle

$$
\mathrm{Hom}(A_{\mathrm{in}}, A_{\mathrm{out}}) := \coprod_{p \in M} \mathrm{Hom}(A_{\mathrm{in},p}, A_{\mathrm{out},p})
$$

is itself a vector bundle associated to the principal $G$-bundle $GM$, and hence is $G$-associated to
$A_{\mathrm{in}}$ and $A_{\mathrm{out}}$ as well. Concretely, if $U_X \cap U_Y \neq \emptyset$ and
$g_{YX} : U_X \cap U_Y \to G$ is the corresponding transition function, then for each $p \in U_X \cap U_Y$,

$$
\left[\psi^Y_{\mathrm{Hom},p} \circ \left(\psi^X_{\mathrm{Hom},p}\right)^{-1}\right](T) = \rho_{\mathrm{out}}\big(g_{YX}(p)\big) \circ T \circ \rho_{\mathrm{in}}\big(g_{YX}(p)\big)^{-1},
$$

with $T \in \mathrm{Hom}(\mathbb{R}^{c_{\mathrm{in}}}, \mathbb{R}^{c_{\mathrm{out}}})$. Thus $g_{YX}$ is the transition
function between $(U_X, \Psi^X_{\mathrm{Hom}})$ and $(U_Y, \Psi^Y_{\mathrm{Hom}})$ through the induced
representation

$$
\rho_{\mathrm{Hom}} : G \to GL\big(\mathrm{Hom}(\mathbb{R}^{c_{\mathrm{in}}}, \mathbb{R}^{c_{\mathrm{out}}})\big), \qquad \rho_{\mathrm{Hom}}(g)(T) := \rho_{\mathrm{out}}(g) \circ T \circ \rho_{\mathrm{in}}(g)^{-1}.
$$

Notably, the induced representation $\rho_{\mathrm{Hom}}$ gives an identification
$\mathrm{Hom}(A_{\mathrm{in}}, A_{\mathrm{out}}) \cong GM \times_{\rho_{\mathrm{Hom}}} \mathrm{Hom}(\mathbb{R}^{c_{\mathrm{in}}}, \mathbb{R}^{c_{\mathrm{out}}})$
between the homomorphism bundle and the associated vector bundle.

An arbitrary smooth global section $\sigma_{\mathrm{Hom}} : M \to \mathrm{Hom}(A_{\mathrm{in}}, A_{\mathrm{out}})$
gives rise to an unconstrained vector bundle $M$-morphism $A_{\mathrm{in}} \to A_{\mathrm{out}}$, given
pointwise by $\sigma_{\mathrm{Hom}}(p)$. For this section to give rise to a $1\times1$ GM-convolution one
must impose the weight-sharing property: there exists a $K_{1\times1} \in \mathrm{Hom}(\mathbb{R}^{c_{\mathrm{in}}}, \mathbb{R}^{c_{\mathrm{out}}}) \cong \mathbb{R}^{c_{\mathrm{out}} \times c_{\mathrm{in}}}$
such that for every $p \in M$ and every gauge in the $G$-atlas

$$
\psi^X_{\mathrm{Hom},p}\big(\sigma_{\mathrm{Hom}}(p)\big) = K_{1\times1}. \tag{2}
$$

For (2) to hold, $K_{1\times1}$ must satisfy $\rho_{\mathrm{Hom}}(g)K_{1\times1} = K_{1\times1}\ \forall g \in G$,
i.e. $K_{1\times1} \in \mathrm{Hom}_G(\rho_{\mathrm{in}}, \rho_{\mathrm{out}})$ is an intertwiner. We call the
template matrix $K_{1\times1}$ a $1\times1$ GM-convolutional kernel either way. Given a smooth global
section $\sigma_{K_{1\times1}} : M \to \mathrm{Hom}(A_{\mathrm{in}}, A_{\mathrm{out}})$ with template kernel
$K_{1\times1}$,

$$
[K_{1\times1} \circledast f_{\mathrm{in}}](p) := \sigma_{K_{1\times1}}(p) f_{\mathrm{in}}(p) \quad \forall p \in M. \tag{3}
$$

The equivalence between (1) and (3) is made clear by noting $C_{K_{1\times1}}|_p = \sigma_{K_{1\times1}}(p)$.

## Kernel fields and transporter pullback

As the next step toward defining GM-convolutions, we introduce unconstrained kernel fields and the
associated kernel field transforms. The relationship between kernel field transforms and
GM-convolutions is analogous to the relationship between unconstrained vector bundle $M$-morphisms
and $1\times1$ GM-convolutions: in each case, the latter is obtained from the former by imposing the
appropriate gauge-invariance or gauge-equivariance constraints.

<a id="def-kernel-field"></a>**Definition (Kernel fields [WFVW23]).** An (unconstrained) kernel field of type $\rho_{\mathrm{in}}, \rho_{\mathrm{out}}$
is a smooth bundle $M$-morphism

$$
K : TM \longrightarrow \mathrm{Hom}(A_{\mathrm{in}}, A_{\mathrm{out}}).
$$

As a general bundle morphism, the restrictions $K_p : T_pM \to \mathrm{Hom}(A_{\mathrm{in},p}, A_{\mathrm{out},p})$
need not be linear.

An unconstrained kernel field transform induced by a kernel field $K$ is a map that sends feature
fields in $A_{\mathrm{in}}$ to feature fields in $A_{\mathrm{out}}$. For each point $p \in M$, the output
feature vector $f_{\mathrm{out}}(p)$ is computed by aligning the input feature field $f_{\mathrm{in}}$ with
the kernel $K_p$ from the perspective of an observer located at $p$.

To make this idea precise, we first clarify what is meant by computing the transform "from the
perspective of an observer located at $p$." This point requires care because the input feature field
is a section $f_{\mathrm{in}} : M \to A_{\mathrm{in}}$, whereas the kernel centered at $p$ is a map
$K_p : T_pM \to \mathrm{Hom}(A_{\mathrm{in},p}, A_{\mathrm{out},p})$. A natural first attempt would be to
pair $K_p(v)$ with the input feature vector $f_{\mathrm{in}}(\exp_p(v))$, where
$\exp_p : T_pM \to M$ denotes the Riemannian exponential map. However, this pairing is not
immediately well-defined. For each $v \in T_pM$, the linear map $K_p(v)$ acts on the fiber
$A_{\mathrm{in},p}$, while $f_{\mathrm{in}}(\exp_p(v)) \in A_{\mathrm{in},\exp_p(v)}$. Thus, before applying
$K_p(v)$, the feature vector at $\exp_p(v)$ must first be identified with a feature vector in the input
fiber over $p$.

The identifications we use are the parallel transport maps on associated vector bundles compatible
with the underlying $G$-structure (the paper, Section 3, defines them through the elements
$g_{p\leftarrow q}\in G$; for $S^2$ see [Sphere geometry](sphere_geometry.md#parallel-transport)). Let

$$
P^{A_{\mathrm{in}}}_{\pi_{TM}(v) \leftarrow \exp(v)} : A_{\mathrm{in},\exp(v)} \to A_{\mathrm{in},p}
$$

denote parallel transport in $A_{\mathrm{in}}$ along the geodesic $\gamma_v(t) := \exp\big((1-t)v\big)$ which
runs from $\exp(v)$ back to $\pi_{TM}(v)$. This gives the following definition.

<a id="def-transporter-pullback"></a>**Definition (Transporter pullback of a feature field to $TM$ [WFVW23]).** Let $A \to M$ be an
associated feature vector bundle equipped with the induced parallel transport maps described above.
Given a feature field $f \in \Gamma(A)$, its transporter pullback to the tangent bundle is the map

$$
\mathrm{Exp}^* f : TM \to A, \qquad v \mapsto P^{A}_{\pi_{TM}(v) \leftarrow \exp(v)}\Big[f\big(\exp(v)\big)\Big] \in A_p. \tag{4}
$$

Thus, for each $p \in M$, the restriction $\left(\mathrm{Exp}^*f_{\mathrm{in}}\right)\big|_{T_pM} : T_pM \to A_{\mathrm{in},p}$
is the input feature field expressed from the perspective of an observer based at $p$. Since the
Riemannian exponential map need not be injective, distinct vectors $v, w \in T_pM$ may satisfy
$\exp_p(v) = \exp_p(w)$. In this case, the same input feature value may appear at multiple locations
in the pulled-back field on $T_pM$.

With this identification in place, the kernel $K_p : T_pM \to \mathrm{Hom}(A_{\mathrm{in},p}, A_{\mathrm{out},p})$
can be evaluated pointwise against the transported and pulled back input field, giving a smooth
bundle morphism $TM \rightarrow \left(\mathrm{Hom}(A_{\mathrm{in}}, A_{\mathrm{out}}) \times_M A_{\mathrm{in}}\right) \to A_{\mathrm{out}}$
via $K \times \mathrm{Exp}^*f_{\mathrm{in}}$ followed by the fiberwise evaluation map $\mathrm{ev}$, with the
fiber product taken over $M$. Finally, integrating this $A_{\mathrm{out}}$-valued map along the fibers of
$\pi_{TM} : TM \to M$ produces a section of $A_{\mathrm{out}}$ — the kernel field transform of $f_{\mathrm{in}}$.

<a id="def-kernel-field-transform"></a>**Definition (Kernel field transform [WFVW23]).** Let $K$ be an unconstrained smooth kernel
field (as defined above). The kernel field transform induced by $K$ is the smooth integral transform
$T_K : \Gamma(A_{\mathrm{in}}) \to \Gamma(A_{\mathrm{out}})$ defined pointwise by

$$
\big(T_K(f_{\mathrm{in}})\big)(p) := \int_{v \in T_pM} K(v)\, \mathrm{Exp}^*f_{\mathrm{in}}(v)\, dv. \tag{5}
$$

The kernel field transform's well-definedness is subject to the selection of a suitable kernel, e.g.
compactly supported kernels.

**Remark (Kernel functions versus kernel densities).** In the definition of the kernel field transform, the kernel field
$K : TM \to \mathrm{Hom}(A_{\mathrm{in}}, A_{\mathrm{out}})$ is treated as a Hom-valued function, while the
measure $dv$ on each tangent fiber is supplied separately. Equivalently, one may absorb the chosen
fiberwise volume density into the kernel and regard the kernel as a Hom-valued density on the
fibers of $TM \to M$. These two conventions define the same integral transform once a fiberwise
measure has been fixed. However, the density-valued convention is the natural one for convolutional
weight sharing, because the object shared across gauges is the full integration kernel appearing
against the coordinate Lebesgue measure.

## G-steerable kernels and GM-convolution

Requiring the kernel field transform to be coordinate independent — the same computation regardless
of which gauge in the $G$-atlas is used to express it — forces the shared template kernel to satisfy

$$
K(gv) = \frac{1}{|\det g|}\rho_{\mathrm{out}}(g)K(v)\rho_{\mathrm{in}}(g)^{-1}, \qquad g \in G,\ v \in \mathbb{R}^d,
$$

which aligns exactly with the definition of $G$-steerable kernels in Euclidean $G$-CNNs. In other
words, in order for $K$ to be the coordinatized density-valued kernel, it must behave equivariantly
under all gauge transformations from the considered $G$-atlas: a $g$ action on the input space
$\mathbb{R}^d$ corresponds to the action $|\det g|^{-1}\rho_{\mathrm{Hom}}(g)$ on the kernel density output.

<a id="def-steerable-kernel"></a>**Definition ($G$-steerable kernel [WFVW23]).** For $G \leq GL(d)$, define the vector space of
smooth $G$-steerable kernels mapping between spaces with field types $\rho_{\mathrm{in}}$ and $\rho_{\mathrm{out}}$

$$
\mathcal{K}^G_{\rho_{\mathrm{in}},\rho_{\mathrm{out}}} := \left\{ K \in C^\infty(\mathbb{R}^d, \mathbb{R}^{c_{\mathrm{out}} \times c_{\mathrm{in}}}) \,\middle|\, K(gv) = \frac{1}{|\det g|}\rho_{\mathrm{out}}(g)K(v)\rho_{\mathrm{in}}(g)^{-1} \ \forall g \in G,\ \forall v \in \mathbb{R}^d \right\}.
$$

<a id="def-gm-convolutional-kernel-field"></a>**Definition (GM-convolutional kernel fields [WFVW23]).** GM-convolutional kernel fields of type
$\rho_{\mathrm{in}}, \rho_{\mathrm{out}}$ are precisely the kernel fields $TM \to \mathrm{Hom}(A_{\mathrm{in}}, A_{\mathrm{out}})$
which exhibit weight sharing and are thus parameterized by the vector space of smooth $G$-steerable
kernels. In particular, for $K \in \mathcal{K}^G_{\rho_{\mathrm{in}},\rho_{\mathrm{out}}}$, define
$K_K : TM \to \mathrm{Hom}(A_{\mathrm{in}}, A_{\mathrm{out}})$, given pointwise in an arbitrary gauge $X$
from the considered $G$-atlas by

$$
K_{K,p} := \left(\psi^X_{\mathrm{Hom},p}\right)^{-1} \circ \frac{K}{\sqrt{\eta^X_p}} \circ \psi^X_{TM,p}. \tag{6}
$$

Well-definedness follows from $K$ belonging to $\mathcal{K}^G_{\rho_{\mathrm{in}},\rho_{\mathrm{out}}}$, and
smoothness of $K_{K,p}$ follows from the smoothness of the involved gauges, metric, and kernel.

<a id="def-gm-convolution"></a>**Definition (GM-convolution [WFVW23]).** Let $A_{\mathrm{in}} = GM \times_{\rho_{\mathrm{in}}} \mathbb{R}^{c_{\mathrm{in}}}$
and $A_{\mathrm{out}} = GM \times_{\rho_{\mathrm{out}}} \mathbb{R}^{c_{\mathrm{out}}}$ be $G$-associated vector
bundles. Define the GM-convolution with $G$-steerable kernel $K \in \mathcal{K}^G_{\rho_{\mathrm{in}},\rho_{\mathrm{out}}}$
to be the kernel field transform induced by the GM-convolutional kernel $K_K$, i.e. the operator
$K \star : \Gamma(A_{\mathrm{in}}) \to \Gamma(A_{\mathrm{out}})$ given pointwise by

$$
[K \star f_{\mathrm{in}}](p) := \int_{T_pM} K_{K,p}(v)\, \mathrm{Exp}^*f_{\mathrm{in}}(v)\, dv. \tag{7}
$$

In a gauge $X$, the coordinate expression is

$$
[K \star f_{\mathrm{in}}]^X(p) = \int_{\mathbb{R}^d} K(v^X) \left(\mathrm{Exp}^*f_{\mathrm{in}}\right)^X(v^X)\, dv^X \tag{8}
$$

$$
= \int_{\mathbb{R}^d} K(v^X)\, \rho_{\mathrm{in}}\Big(g^{\widetilde{X}X}_{p \leftarrow \exp_p((\psi^X_{TM,p})^{-1}(v^X))}\Big)\, f^{\widetilde{X}}_{\mathrm{in}}\Big(\exp_p\big((\psi^X_{TM,p})^{-1}(v^X)\big)\Big)\, dv^X, \tag{9}
$$

which follows from the coordinate expression of the kernel field transform (5), in which the Riemannian
measure on $T_pM$ reads $\sqrt{|\eta^X_p|}\, dv^X$ in the coordinates $v^X$, by substituting
$K^X_{K,p}\sqrt{|\eta^X_p|} = K$, i.e. (6) in the gauge $X$.

## Covariance and equivariance of GM-convolutions

GM-convolution was formulated so that the resulting computation be coordinate independent. This
led to the notion of GM-convolutional kernel fields: kernel fields determined by a single shared
template kernel-density, used at every spatial location and in every gauge. Coordinate independence
of the operator forces the kernel fields to be gauge-steerable — the analogue of the fact that, in the
Euclidean setting, weight sharing and $G$-steerability is precisely what gives steerable convolutional
neural networks their $\mathrm{Aff}(G)$-equivariance.

### Behavior under gauge transformations

The effect of $G$-transformations on the data admits two closely related interpretations. The first is
the *passive* perspective, in which a gauge transformation is interpreted as a change of local reference
frame. The second is the *active* perspective, in which the local reference frame is held fixed and the
corresponding element of $G$ is interpreted as acting directly on the feature data seen in that frame.
These two viewpoints are equivalent at the level of a local observer: changing the frame in which a
local field of view is represented has the same observed effect as keeping the frame fixed and
transforming the local field of view by the corresponding representation of the structure group $G$.

GM-convolutions behave covariantly under passive gauge transformations. Concretely, on an overlap
$U_X \cap U_Y$, the passive covariance property is formulated as

$$
f^Y = \rho_{\mathrm{in}}(g_{YX}) f^X \implies [K \star f]^Y = \rho_{\mathrm{out}}(g_{YX})[K \star f]^X.
$$

This is a passive statement because $f^X$ and $f^Y$ are not distinct input fields, but rather two
coordinate expressions of the same intrinsic feature field $f \in \Gamma(A_{\mathrm{in}})$. Thus the
implication says that changing the gauge used to represent the input field changes the coordinate
expression of the output field by the corresponding output representation.

The active counterpart is formulated locally, at the level of the kernel's field of view. Fix a gauge $X$
and a point $p \in U_X$. The kernel's view of the input feature field at $p$, expressed in the gauge $X$,
is the local feature patch $F^X_p := \left[\mathrm{Exp}^*_p f_{\mathrm{in}}\right]^X \in L^2(\mathbb{R}^d, \mathbb{R}^{c_{\mathrm{in}}})$.
The group $G$ acts on such local patches through the induced representation

$$
\rhd_{\rho_{\mathrm{in}}} : G \times L^2(\mathbb{R}^d, \mathbb{R}^{c_{\mathrm{in}}}) \to L^2(\mathbb{R}^d, \mathbb{R}^{c_{\mathrm{in}}}), \qquad \big(g\, \rhd_{\rho_{\mathrm{in}}}\, F\big)(v) := \rho_{\mathrm{in}}(g)F(g^{-1}v).
$$

If $I_K(F) := \int_{\mathbb{R}^d} K(v)F(v)\, dv$, $F \in L^2(\mathbb{R}^d, \mathbb{R}^{c_{\mathrm{in}}})$, denotes the
response of the shared kernel to a local input patch, then active local $G$-equivariance is the statement
that $I_K\big(g\,\rhd_{\rho_{\mathrm{in}}}\, F^X_p\big) = g\rhd_{\rho_{\mathrm{out}}}I_K(F^X_p)\ \forall g \in G$. Both passive
gauge covariance and active local $G$-equivariance are consequences of the same $G$-steerability
constraint on the shared kernel.

### Behavior under isometries

Let $\mathrm{Isom}(M)$ be the group of isometries: diffeomorphisms $\phi : M \to M$ which preserve the
metric, $\eta_p(u,v) = \eta_{\phi(p)}(\phi_{*,TM}(u), \phi_{*,TM}(v))\ \forall p \in M,\ u,v \in T_pM$, where
$\phi_{*,TM} : TM \to TM$ denotes the pushforward of $\phi$. The pushforward gives the natural action
of $\mathrm{Isom}(M)$ on $TM$, and this extends to an action on the frame bundle $FM$ via

$$
\mathrm{Isom}(M) \times FM \to FM, \qquad \big(\phi, [e_i]_{i=1}^d\big) \mapsto \phi_{*,FM}\big([e_i]_{i=1}^d\big) := \big[\phi_{*,TM}(e_i)\big]_{i=1}^d.
$$

The linearity of the pushforward makes the left action $\phi_{*,FM}$ commute with the right $GL(d)$
change-of-basis action on the fibers of $FM$; that is, $\phi_{*,FM}$ is right $GL(d)$-equivariant.

The isometries which play well with GM-convolutions are those which respect the underlying
$G$-structure. We define the subgroup

$$
\mathrm{Isom}_{GM} := \big\{ \phi \in \mathrm{Isom}(M) : \phi_{*,FM}(G_pM) = G_{\phi(p)}M\ \forall p \in M \big\}.
$$

These isometries act on $GM$ by defining, for each $\phi \in \mathrm{Isom}_{GM}$,
$\phi_{*,GM} := \phi_{*,FM}|_{GM}$, and this action is right $G$-equivariant. (For $G \geq O(d)$,
$\mathrm{Isom}_{GM} = \mathrm{Isom}(M)$, since an orthonormal frame is pushed forward to an orthonormal
frame under any isometry.) The right $G$-equivariance of $\phi_{*,GM}$ allows one to define an
$\mathrm{Isom}_{GM}$ action on any associated feature vector bundle $A = GM \times_\rho \mathbb{R}^c$ by

$$
\Big(\phi, \big([e_i]_{i=1}^d, f\big)\Big) \mapsto \phi_{*,A}\big([e_i]_{i=1}^d, f\big) := \Big(\phi_{*,GM}\big([e_i]_{i=1}^d\big), f\Big),
$$

and finally, with $\phi_{*,A}$ in place, the action of $\mathrm{Isom}_{GM}$ on feature vector fields
$\Gamma(A)$ by

$$
\rhd : \mathrm{Isom}_{GM} \times \Gamma(A) \to \Gamma(A), \qquad \phi \rhd f := \phi_{*,A} \circ f \circ \phi^{-1}. \tag{10}
$$

<a id="thm-isom-gm-equivariance"></a>**Theorem ($\mathrm{Isom}_{GM}$ equivariance of GM-convolution; Thm. 13.2.5 of [WFVW23]).** A
GM-convolution $K \star : \Gamma(A_{\mathrm{in}}) \to \Gamma(A_{\mathrm{out}})$ with $G$-steerable kernel
$K \in \mathcal{K}^G_{\rho_{\mathrm{in}},\rho_{\mathrm{out}}}$ is equivariant with respect to all
$G$-structure preserving isometries. That is,

$$
K \star (\phi \rhd f) = \phi\rhd (K \star f) \qquad \forall \phi \in \mathrm{Isom}_{GM},\ \forall f \in \Gamma(A_{\mathrm{in}}).
$$

The proof of this theorem follows a similar argument to the one presented for the more general
setting of GM-aggregation; see the $\mathrm{Isom}_{GM}$ equivariance section of the
[GM-aggregation](gm_aggregation.md#isometry-equivariance-of-gm-aggregation) page and the paper,
Theorem 1 and Appendix A.2.

## References

- [WFVW23] Maurice Weiler, Patrick Forré, Erik Verlinde, and Max Welling. *Equivariant and
  Coordinate Independent Convolutional Networks: A Gauge Field Theory of Neural Networks*.
  2023. Online monograph,
  <https://maurice-weiler.gitlab.io/cnn_book/EquivariantAndCoordinateIndependentCNNs.pdf>.
