# GM-Aggregation

This page is the direct math counterpart of
[`gm_attention.nn`](../api/nn.md)'s value transformations and attention layers. The concrete $S^2$
realization is covered by [Sphere geometry](sphere_geometry.md),
[Discretization](discretization.md) and [`SO(2)` representations](so2_representations.md).

This page generalizes Weiler's GM-convolution [WFVW23] (the definitions used here are collected in
[Conventions](conventions.md)) while also encompassing attention-based operators.
[Aggregation transforms on Euclidean space](#aggregation-transforms-on-euclidean-space) briefly
reviews general aggregation operators on Euclidean domains, motivating the manifold analogue
introduced in [Unconstrained aggregation transforms on Riemannian
manifolds](#unconstrained-aggregation-transforms-on-riemannian-manifolds). In [Coordinate
independence and GM-aggregation](#coordinate-independence-and-gm-aggregation), we derive the
conditions under which such an aggregation operator is coordinate independent and use these
conditions to define GM-aggregation. [GM-Attention](#gm-attention) introduces GM-self-attention
and its multi-head extension as special cases of GM-aggregation. Finally,
[Isometry equivariance of GM-aggregation](#isometry-equivariance-of-gm-aggregation) proves
that GM-aggregation possesses the same $\mathrm{Isom}_{GM}$-equivariance property as GM-convolution.

## Aggregation transforms on Euclidean space

In the Euclidean setting, a convolution is a map
$K\star : L^2(\mathbb{R}^d,\mathbb{R}^{c_{\mathrm{in}}}) \to L^2(\mathbb{R}^d,\mathbb{R}^{c_{\mathrm{out}}})$
defined by

$$
[K\star F](p) := \int_{\mathbb{R}^d} K(v)F(p+v)\, dv, \tag{4.1}
$$

where $K : \mathbb{R}^d \to \mathrm{Hom}(\mathbb{R}^{c_{\mathrm{in}}},\mathbb{R}^{c_{\mathrm{out}}})$ is a
spatially shared convolutional kernel. If we set $q := p+v$, then $v = q-p$, and (4.1) can be
rewritten as

$$
[K\star F](p) = \int_{\mathbb{R}^d} \widetilde K(p,q) F(q)\, dq, \tag{4.2}
$$

where $\widetilde K(p,q) := K(q-p)$. In this form, convolution can be viewed as an aggregation
operator, in the same broad sense that nonlocal neural-network blocks compute responses by
aggregating features over positions [WGGH18]. The output signal at $p$ is obtained by aggregating
linearly transformed feature vectors from points $q$. The points that influence the output at $p$ are
determined by the support, or effective support, of the kernel, while the linear transformation
applied to each feature vector is determined by its displacement $q-p$ relative to $p$.

We now generalize the aggregation accomplished by convolution by allowing the interaction between
points to depend on the input feature field itself. This generalized aggregation framework covers
convolutional operators [LBBH98, KSH12], scaled dot-product self-attention [VSP+17], and nonlocal
neural-network blocks [WGGH18]. Define
$A_{\alpha,\beta,V} : L^2_c(\mathbb{R}^d,\mathbb{R}^{c_{\mathrm{in}}}) \to L^2_c(\mathbb{R}^d,\mathbb{R}^{c_{\mathrm{out}}})$
by

$$
[A_{\alpha,\beta,V}(F)](p) = \frac{1}{C_{\alpha,\beta}(F,p)}\int_{\mathrm{supp}(F)} \beta\big(\alpha(F(p),F(q))\big)\, V(p,q)\, F(q)\, dq. \tag{4.3}
$$

Here $\alpha : \mathbb{R}^{c_{\mathrm{in}}}\times\mathbb{R}^{c_{\mathrm{in}}} \to \mathbb{R}$ is a scalar
affinity function measuring the interaction between the feature vector at $p$ and the feature vector
at $q$, $\beta : \mathbb{R} \to \mathbb{R}_{>0}$ is a positive score-to-weight map,
$V : \mathbb{R}^d\times\mathbb{R}^d \to \mathrm{Hom}(\mathbb{R}^{c_{\mathrm{in}}},\mathbb{R}^{c_{\mathrm{out}}})$
is a value transformation, and $C_{\alpha,\beta}(F,p)$ is an affinity normalization factor that takes
the general form

$$
C_{\alpha,\beta}(F,p) = \int_{\mathrm{supp}(F)} \beta\big(\alpha(F(p),F(q))\big)\, dq. \tag{4.4}
$$

This form is closely related to the generic nonlocal operation of Wang et al. [WGGH18], with
$\beta\circ\alpha$ playing the role of the pairwise affinity and $V(p,q)F(q)$ playing the role of the
transformed value feature. The locality of the aggregation is determined by the support, or effective
support, of the integrand. In particular, locality may be imposed either by restricting the support of
$V(p,q)$, by inserting an explicit spatial window into the affinity, or by both.

This general form has the following familiar instantiations.

**Convolution.** If the affinity is trivial, $\alpha \equiv 1$, we set the normalization factor
$C_{\alpha,\beta}(F,p) \equiv 1$, and $V(p,q) = K(q-p)$, then (4.3) recovers convolution in the form
of (4.2). If $\mathbb{R}^{c_{\mathrm{in}}}$ and $\mathbb{R}^{c_{\mathrm{out}}}$ are equipped with
$G$-representations and $K$ is $G$-steerable in the sense of the $G$-steerable kernel condition
(see [Conventions](conventions.md)), then the corresponding convolution is $G$-equivariant
[CW16, WFVW23].

**Self-attention.** Let $W_q : \mathbb{R}^{c_{\mathrm{in}}}\to\mathbb{R}^{d_k}$,
$W_k : \mathbb{R}^{c_{\mathrm{in}}}\to\mathbb{R}^{d_k}$, $W_v : \mathbb{R}^{c_{\mathrm{in}}}\to\mathbb{R}^{c_{\mathrm{out}}}$.
If

$$
\alpha(u,v) = \frac{(W_qu)^\top(W_kv)}{\sqrt{d_k}}, \tag{4.5}
$$

$$
\beta(t) = \exp(t), \tag{4.6}
$$

$$
V(p,q) = W_v, \tag{4.7}
$$

$$
C_{\alpha,\beta}(F,p) = \int_{\mathrm{supp}(F)} \beta\big(\alpha(F(p),F(q))\big)\, dq = \int_{\mathrm{supp}(F)} \exp\!\left(\frac{(W_qF(p))^\top(W_kF(q))}{\sqrt{d_k}}\right) dq, \tag{4.8}
$$

then (4.3) gives a continuous analogue of scaled dot-product self-attention [VSP+17, CKLS25] on
Euclidean feature fields:

$$
[A_{\alpha,\beta,V}(F)](p) = \frac{\displaystyle\int_{\mathrm{supp}(F)} \exp\!\left(\frac{(W_qF(p))^\top(W_kF(q))}{\sqrt{d_k}}\right) W_vF(q)\, dq}{\displaystyle\int_{\mathrm{supp}(F)} \exp\!\left(\frac{(W_qF(p))^\top(W_kF(q))}{\sqrt{d_k}}\right) dq}. \tag{4.9}
$$

Thus, the output at $p$ is a normalized weighted aggregation of value vectors $W_vF(q)$, where the
weights are determined by the feature-dependent similarity between the query $W_qF(p)$ and the key
$W_kF(q)$. In this setting, it is important that we constrained the domain of the aggregation
transform to compactly supported feature maps.

A local version is obtained by introducing a spatial window function
$\iota : \mathbb{R}^d\times\mathbb{R}^d \to [0,1]$, where $\iota(p,q)$ is supported, or effectively
supported, near $p$. For example, one may replace the scalar weight
$\beta(\alpha(F(p),F(q)))$ by

$$
\iota(p,q)\,\beta\big(\alpha(F(p),F(q))\big) = \iota(p,q)\exp\!\left(\frac{(W_qF(p))^\top(W_kF(q))}{\sqrt{d_k}}\right),
$$

and normalize by
$C_{\alpha,\beta}(F,p) = \int_{\mathbb{R}^d} \iota(p,q)\,\beta(\alpha(F(p),F(q)))\, dq = \int_{\mathbb{R}^d} \iota(p,q)\exp\!\left(\frac{(W_qF(p))^\top(W_kF(q))}{\sqrt{d_k}}\right) dq$.
This recovers the local self-attention pattern used in stand-alone self-attention layers [RPV+19],
where aggregation is restricted to a neighborhood of each point.

These instantiations occupy two extremes of the generalized aggregation operator and encode
distinct inductive biases. On one hand, convolution uses a trivial feature affinity together with a
value transformation that depends on relative position. This is valuable in settings with strong
spatial priors, where the displacement between signals is informative for determining how those
signals should interact. On the other hand, self-attention uses a nontrivial, feature-dependent
affinity while keeping the value transformation spatially shared. This encodes a content-dependent
communication prior, i.e. points interact according to learned feature relevance rather than solely
according to displacement or local geometry.

One could, in principle, combine both mechanisms by using a nontrivial affinity together with a
nonconstant value transformation. In fact this is what is done in Equivariant Mesh Attention
Networks [BGPV+22], where the non-constant value transformation can be seen as a sort of
positional encoding. On the other hand, some architectures use the two mechanisms interdependently
in tandem. For example, nonlocal neural networks [WGGH18] insert attention-like aggregation
blocks into convolutional backbones, allowing the model to combine local spatial biases from
convolution with long-range content-dependent interactions. Similarly, stand-alone and
attention-augmented vision models [RPV+19, BZV+19] demonstrate that attention mechanisms can
serve either as an alternative to convolution or as a structured complement to convolutional feature
extraction.

In the remainder of this page, we generalize the Euclidean aggregation operator to feature fields defined over
$G$-structures on Riemannian manifolds. The construction follows the coordinate-independent
framework of GM-convolution [WFVW23]: feature fields are treated as sections of associated vector
bundles, and aggregation is expressed intrinsically in terms of geometric data such as kernel fields,
transporter pullbacks, and invariant integration. Within this framework, GM-convolution appears as
the special case with trivial feature affinity, while GM-attention arises by introducing gauge-invariant
affinity functions and gauge-equivariant value transformations. In this sense, our construction both
generalizes the Euclidean aggregation operators discussed above and provides a coordinate-independent
framework for interpreting gauge-equivariant attention mechanisms such as the Gauge Equivariant
Transformer of He et al. [HCD+21] and Equivariant Mesh Attention Networks of Basu et al. [BGPV+22].

## Unconstrained aggregation transforms on Riemannian manifolds

In this section, we introduce unconstrained aggregation transforms on Riemannian manifolds. In the
next section, we derive and impose the constraints under which these transforms become coordinate
independent.

To generalize the Euclidean aggregation transform, cf. (4.3), to feature fields on a Riemannian
manifold, we first generalize each of the building blocks appearing in the Euclidean construction.
These include the affinity function, the affinity normalization factor, the value transformation, and
the integral itself.

We begin with the integral. As in the generalization of Euclidean convolution to GM-convolution
[WFVW23], it is most natural to reinterpret the Euclidean aggregation transform as an integral over
the tangent space at the point around which aggregation takes place. Under this interpretation,
feature vectors at neighboring points are first transported back to the fiber over the aggregation
point before being combined.

This reinterpretation does not change the Euclidean formula (4.3). Indeed, when $\mathbb{R}^d$ is
equipped with its canonical $G$-structure and Levi-Civita connection, exponentiation and transport
are trivial: $\exp_p(v) = p+v$, and parallel transport canonically identifies all fibers. Thus the
tangent-space formulation is invisible in Euclidean space, even though it is the correct perspective
for passing to Riemannian manifolds.

On a general Riemannian manifold, there are no preferred global coordinates, no canonical way to
compare feature vectors lying in distinct fibers, and no globally trivial exponential map. Therefore,
our manifold aggregation transform will be defined pointwise by integrating over tangent spaces. For
each $p \in M$, aggregation takes place over $T_pM$, and feature values based at nearby points
$\exp_p(v)$ are transported back to the fiber over $p$ by means of the transporter pullback of
[Conventions](conventions.md#kernel-fields-and-transporter-pullback). This is to say we aggregate feature vectors as they are seen from the perspective of
an observer at $p$.

Although aggregation is formulated as an integral over the tangent space $T_pM$, we will in general
restrict the effective domain of integration. This is necessary because the scalar affinity term need
not have compact support in the tangent variable. Consequently, integration over the full tangent
space, which has infinite volume whenever $\dim M > 0$, need not produce either a finite aggregation
integral or a finite affinity normalization factor.

This issue does not arise in the same way for GM-convolution. In that case, the affinity component
is trivial, and well-definedness is ensured by imposing suitable hypotheses on the value
transformation, e.g. compact support (see the kernel field transform in
[Conventions](conventions.md#kernel-fields-and-transporter-pullback)). Once a nontrivial scalar affinity is
introduced, however, the normalization factor itself becomes an additional integral whose finiteness
must be controlled.

We control this by introducing an aggregation window. An **aggregation window** is a smooth
function

$$
\chi : TM \to \mathbb{R}_{\geq 0} \tag{4.10}
$$

whose restriction to each fiber $T_pM$ has compact support which is contained in the domain of
$\exp_p$ and which contains the origin. Therefore, for each $p \in M$, the function $\chi|_{T_pM}$
selects a compact window about the origin of tangent directions which participate in the aggregation
around $p$. The defining integrals will still be written as integrals over $T_pM$, but multiplication
by $\chi$ restricts the effective support of the integrand to this compact region.

The introduction of $\chi$ should not be interpreted as ruling out long-range dependencies. Rather,
$\chi$ specifies the region of each tangent space over which aggregation is allowed to take place, and
this region may be chosen to be as large as is appropriate for the problem at hand. For example, in
local geometric settings one may choose $\chi$ to be supported in a geodesic ball
$B_R(0_p) \subset T_pM$, with $R$ below the injectivity radius at $p$, so that aggregation occurs over
a region on which the exponential map is injective. In settings where long-range interactions are
important, one may instead choose a much larger support, or even learn the window as part of the
model, provided that the resulting integrals remain finite.

Thus, $\chi$ serves primarily as an integrability mechanism: it prevents the infinite volume of $T_pM$
from producing divergent normalization and aggregation integrals. Its support controls the geometric
range over which feature vectors may communicate, but it does not force this range to be small.
Larger windows allow the scalar affinity function to communicate long-range dependencies.

Next we address the value transformation, a component which has already been discussed in
[Conventions](conventions.md#kernel-fields-and-transporter-pullback). In the Euclidean setting, the value transformation is a map
$V : \mathbb{R}^d\times\mathbb{R}^d \to \mathrm{Hom}(\mathbb{R}^{c_{\mathrm{in}}},\mathbb{R}^{c_{\mathrm{out}}})$.
Its domain can be naturally interpreted as the tangent bundle $T\mathbb{R}^d \cong \mathbb{R}^d\times\mathbb{R}^d$.
Under this interpretation, a pair $(p,v) \in T\mathbb{R}^d$ consists of an aggregation point
$p \in \mathbb{R}^d$ together with a displacement vector $v \in T_p\mathbb{R}^d \cong \mathbb{R}^d$.
Thus the first input specifies the point around which aggregation takes place, while the second
input specifies the relative displacement from that point.

This interpretation is consistent with convolution. Indeed, convolution is recovered in the special
case where the value transformation is independent of the base point $p$. In that case, the value
transformation is a convolutional kernel depending only on the displacement $v$, rather than on the
absolute spatial position. Thus $V(p,v) = K(v)$ for some kernel
$K : \mathbb{R}^d \to \mathrm{Hom}(\mathbb{R}^{c_{\mathrm{in}}},\mathbb{R}^{c_{\mathrm{out}}})$, as in
(4.1).

This viewpoint suggests the natural manifold generalization of the value transformation. Namely, we
take it to be an unconstrained kernel field as defined in [Conventions](conventions.md#kernel-fields-and-transporter-pullback): a bundle morphism over $M$

$$
V : TM \to \mathrm{Hom}(A_{\mathrm{in}}, A_{\mathrm{out}}).
$$

For $v \in T_pM$, the map $V(v)$ is the local linear value transformation associated with aggregating
information near $p$ in the tangent direction $v$. At this stage, $V$ is an intrinsic geometric object,
but we impose no gauge-equivariance, kernel-sharing, or coordinate-independence constraints on its
local representatives. These constraints will be imposed in the next section.

Moving on to the first component which has not yet been discussed, we consider the affinity
function $\alpha$. Under our interpretation of the Euclidean aggregation transform as being defined
pointwise by integrals over tangent spaces of transported feature vectors, the affinity at a base point
$p \in M$ should compare two feature vectors lying in the same fiber over $p$; that is, it compares the
feature vector $F(p) \in A_{\mathrm{in},p}$ with the transported feature vector based at a neighboring
point $\exp_p(v)$. Thus, the natural domain of the affinity function is the fiberwise direct sum
$A_{\mathrm{in}} \oplus_M A_{\mathrm{in}}$, whose fiber over $p$ is $A_{\mathrm{in},p}\oplus A_{\mathrm{in},p}$.
We therefore take the unconstrained affinity function to be a smooth map
$\alpha : A_{\mathrm{in}} \oplus_M A_{\mathrm{in}} \to \mathbb{R}$. Again, no gauge-invariance
constraint is imposed on $\alpha$. Because $\alpha$ is seen as the raw affinity scores, we post-compose
it with an activation $\beta : \mathbb{R} \to \mathbb{R}_{>0}$.

We define the following notation for later convenience.

**Definition 4.1 (Scalar aggregation weight).** Given an aggregation window
$\chi : TM \to \mathbb{R}_{\geq 0}$, an affinity function $\alpha : A_{\mathrm{in}}\oplus_M A_{\mathrm{in}} \to \mathbb{R}$,
an affinity activation $\beta : \mathbb{R}\to\mathbb{R}_{>0}$, and a feature field
$f \in \Gamma(A_{\mathrm{in}})$, we define the feature-field-dependent unnormalized scalar
aggregation weight to be the map $\omega^\chi_{\alpha,\beta}(f) : TM \to \mathbb{R}$ defined

$$
\omega^\chi_{\alpha,\beta}(f)(v) := \chi(v)\,\beta\Big(\alpha\big(f(p),\, (\mathrm{Exp}^*_pf)(v)\big)\Big), \tag{4.11}
$$

where $p := \pi_{TM}(v)$ and $\mathrm{Exp}^*f$ is the transporter pullback of [Conventions](conventions.md#kernel-fields-and-transporter-pullback).

Lastly, we address the affinity normalization factor. In the Euclidean setting, this is a map
$C_{\alpha,\beta} : L^2(\mathbb{R}^d,\mathbb{R}^{c_{\mathrm{in}}})\times\mathbb{R}^d \to \mathbb{R}_{>0}$.
For a feature field $F$ and an aggregation point $p \in \mathbb{R}^d$, the scalar $C_{\alpha,\beta}(F,p)$
normalizes the affinity scores around $p$ into a feature-field-dependent probability distribution over
the points being aggregated, as in normalized self-attention and nonlocal neural-network blocks
[VSP+17, WGGH18].

Thus, in the manifold setting, we take the normalization factor to be a map
$C^\chi_{\alpha,\beta} : \Gamma(A_{\mathrm{in}})\times M \to \mathbb{R}_{>0}$. Mirroring (4.4) it is
defined

$$
C^\chi_{\alpha,\beta}(f,p) := \int_{T_pM} \chi(v)\,\beta\Big(\alpha\big(f(p),\, (\mathrm{Exp}^*_pf)(v)\big)\Big)\, dv \tag{4.12}
$$

$$
= \int_{T_pM} \omega^\chi_{\alpha,\beta}(f)(v)\, dv \tag{4.13}
$$

where $\chi$ is the aggregation window (4.10), $\mathrm{Exp}^*$ denotes the transporter pullback (cf.
[Conventions](conventions.md#kernel-fields-and-transporter-pullback)), and $\omega^\chi_{\alpha,\beta}$ is the scalar aggregation weight (4.11). This is the
normalization used for normalized aggregation transforms. To recover convolution as an
unnormalized trivial-affinity special case, one instead sets $\alpha \equiv 1$,
$C^\chi_{\alpha,\beta} \equiv 1$, and lets the value transformation play the role of the convolutional
integration kernel.

Having addressed how each component of the Euclidean aggregation transform is to be generalized
to the manifold setting, we are prepared to state the definition of general unconstrained aggregation
transforms.

**Definition 4.2 (Unconstrained aggregation transforms).** Let $(M,\eta)$ be a Riemannian manifold
equipped with a $G$-structure and consider associated feature vector bundles $A_{\mathrm{in}}$ and
$A_{\mathrm{out}}$ of field types $\rho_{\mathrm{in}}$ and $\rho_{\mathrm{out}}$, respectively. Given a
smooth affinity function $\alpha : A_{\mathrm{in}}\oplus_M A_{\mathrm{in}} \to \mathbb{R}$, a smooth
affinity activation $\beta : \mathbb{R}\to\mathbb{R}_{>0}$, a smooth value transformation
$V : TM \to \mathrm{Hom}(A_{\mathrm{in}},A_{\mathrm{out}})$, and a smooth aggregation window
$\chi : TM \to \mathbb{R}_{\geq 0}$, we define the associated unconstrained aggregation operator
$A^\chi_{\alpha,\beta,V} : \Gamma(A_{\mathrm{in}}) \to \Gamma(A_{\mathrm{out}})$ which is given
pointwise by

$$
[A^\chi_{\alpha,\beta,V}(f)](p) := \frac{1}{C^\chi_{\alpha,\beta}(f,p)}\int_{T_pM} \omega^\chi_{\alpha,\beta}(f)(v)\, V_p(v)\big[(\mathrm{Exp}^*_pf)(v)\big]\, dv. \tag{4.14}
$$

Here $C^\chi_{\alpha,\beta}$ is the affinity normalization as in (4.12), $\omega^\chi_{\alpha,\beta}(f)$
is the aggregation weight as in (4.11), and $\mathrm{Exp}^*_pf$ is the transporter pullback of $f$ at
$p$ ([Conventions](conventions.md#kernel-fields-and-transporter-pullback)). In order to be well defined the integrals must exist and the resulting output field
must be smooth.

**Proposition 4.3.** The coordinate expression for the unconstrained aggregation transform in a gauge
$X$ from the considered $G$-atlas is

$$
\big(A^\chi_{\alpha,\beta,V}(f)\big)^X(p) \tag{4.15}
$$

$$
= \frac{\displaystyle\int_{\mathbb{R}^d} \chi^X_p(v^X)\,\beta\Big(\alpha^X_p\big(f^X(p),\, (\mathrm{Exp}^*_pf)^X(v^X)\big)\Big)\, V^X_p(v^X)\big[(\mathrm{Exp}^*_pf)^X(v^X)\big]\, dv^X}{\displaystyle\int_{\mathbb{R}^d} \chi^X_p(v^X)\,\beta\Big(\alpha^X_p\big(f^X(p),\, (\mathrm{Exp}^*_pf)^X(v^X)\big)\Big)\, dv^X}. \tag{4.16}
$$

Here $\chi^X_p := \chi_p \circ \big(\psi^X_{TM,p}\big)^{-1}$,
$\alpha^X_p := \alpha_p \circ \big[\big(\psi^X_{A_{\mathrm{in}},p}\big)^{-1} \oplus \big(\psi^X_{A_{\mathrm{in}},p}\big)^{-1}\big]$,
while $V^X_p := \psi^X_{\mathrm{Hom},p}\circ V_p\circ\big(\psi^X_{TM,p}\big)^{-1}$ and
$(\mathrm{Exp}^*_pf)^X := \psi^X_{A_{\mathrm{in}},p}\circ(\mathrm{Exp}^*_pf)\circ\big(\psi^X_{TM,p}\big)^{-1}$
are the coordinate expressions of the value transformation and of the transporter pullback in the
gauge $X$.

**Proof.** Fix $p \in M$ and an admissible gauge $X$ about $p$. For a feature field
$f \in \Gamma(A_{\mathrm{in}})$, define the $f$-dependent kernel field
$K_f : TM \to \mathrm{Hom}(A_{\mathrm{in}},A_{\mathrm{out}})$,
$K_f(v) := \omega^\chi_{\alpha,\beta}(f)(v)\, V(v)$. Then the numerator of the aggregation transform is
precisely the kernel field transform induced by $K_f$:

$$
N_f(p) := \int_{T_pM} \omega^\chi_{\alpha,\beta}(f)(v)\, V_p(v)\big[(\mathrm{Exp}^*_pf)(v)\big]\, dv = \big(T_{K_f}(f)\big)(p).
$$

We first compute the coordinate expression of the scalar aggregation weight. By definition,

$$
\begin{aligned}
\big(\omega^\chi_{\alpha,\beta}(f)\big)^X_p(v^X)
&\overset{(1)}{=} \omega^\chi_{\alpha,\beta}(f)\big((\psi^X_{TM,p})^{-1}(v^X)\big) \\
&\overset{(2)}{=} \chi_p\big((\psi^X_{TM,p})^{-1}(v^X)\big)\, \beta\Big(\alpha_p\big(f(p),\, (\mathrm{Exp}^*_pf)((\psi^X_{TM,p})^{-1}(v^X))\big)\Big) \\
&\overset{(3)}{=} \chi^X_p(v^X)\, \beta\Big(\alpha^X_p\big(\psi^X_{A_{\mathrm{in}},p}f(p),\, \psi^X_{A_{\mathrm{in}},p}(\mathrm{Exp}^*_pf)((\psi^X_{TM,p})^{-1}(v^X))\big)\Big) \\
&\overset{(4)}{=} \chi^X_p(v^X)\, \beta\Big(\alpha^X_p\big(f^X(p),\, (\mathrm{Exp}^*_pf)^X(v^X)\big)\Big).
\end{aligned}
$$

Step (1) is the definition of the coordinate expression of the scalar function $\omega^\chi_{\alpha,\beta}(f)$
on $T_pM$, step (2) expands the definition of $\omega^\chi_{\alpha,\beta}(f)$, step (3) uses
$\alpha_p = \alpha^X_p \circ (\psi^X_{A_{\mathrm{in}},p}\oplus\psi^X_{A_{\mathrm{in}},p})$ and the
definition of $\chi^X_p$, and step (4) uses the definitions of $f^X(p)$ and $(\mathrm{Exp}^*_pf)^X(v^X)$.

Therefore the coordinate expression of the effective kernel field $K_f$ is

$$
K^X_{f,p}(v^X) = \big(\omega^\chi_{\alpha,\beta}(f)\big)^X_p(v^X)\, V^X_p(v^X) = \chi^X_p(v^X)\,\beta\Big(\alpha^X_p\big(f^X(p),\, (\mathrm{Exp}^*_pf)^X(v^X)\big)\Big)\, V^X_p(v^X).
$$

Passing to the gauge coordinates $v^X$ of $T_pM$, in which the Riemannian measure is
$dv = \sqrt{|\eta^X_p|}\,dv^X$ (the coordinate formula for kernel field transforms, cf.
[Conventions](conventions.md#kernel-fields-and-transporter-pullback)), the kernel field transform of
$K_f$ is given by

$$
\psi^X_{A_{\mathrm{out}},p}N_f(p) = \int_{\mathbb{R}^d} \chi^X_p(v^X)\,\beta\Big(\alpha^X_p\big(f^X(p),\, (\mathrm{Exp}^*_pf)^X(v^X)\big)\Big)\, V^X_p(v^X)\big[(\mathrm{Exp}^*_pf)^X(v^X)\big]\, \sqrt{|\eta^X_p|}\, dv^X.
$$

It remains to compute the denominator. Since it is scalar-valued, no output trivialization appears:

$$
\begin{aligned}
C_f(p) &:= \int_{T_pM} \omega^\chi_{\alpha,\beta}(f)(v)\, dv \\
&= \int_{\mathbb{R}^d} \big(\omega^\chi_{\alpha,\beta}(f)\big)^X_p(v^X)\, \sqrt{|\eta^X_p|}\, dv^X \\
&= \int_{\mathbb{R}^d} \chi^X_p(v^X)\,\beta\Big(\alpha^X_p\big(f^X(p),\, (\mathrm{Exp}^*_pf)^X(v^X)\big)\Big)\, \sqrt{|\eta^X_p|}\, dv^X.
\end{aligned}
$$

Combining the coordinate expressions for the numerator and denominator, we obtain

$$
\big(A^\chi_{\alpha,\beta,V}(f)\big)^X(p) = \frac{\displaystyle\int_{\mathbb{R}^d} \chi^X_p(v^X)\,\beta\big(\alpha^X_p(\cdots)\big)\, V^X_p(v^X)\big[(\mathrm{Exp}^*_pf)^X(v^X)\big]\, \sqrt{|\eta^X_p|}\, dv^X}{\displaystyle\int_{\mathbb{R}^d} \chi^X_p(v^X)\,\beta\big(\alpha^X_p(\cdots)\big)\, \sqrt{|\eta^X_p|}\, dv^X}.
$$

Since $\sqrt{|\eta^X_p|}$ is independent of $v^X$, it factors out of both integrals and cancels. Hence

$$
\big(A^\chi_{\alpha,\beta,V}(f)\big)^X(p) = \frac{\displaystyle\int_{\mathbb{R}^d} \chi^X_p(v^X)\,\beta\Big(\alpha^X_p\big(f^X(p),(\mathrm{Exp}^*_pf)^X(v^X)\big)\Big)\, V^X_p(v^X)\big[(\mathrm{Exp}^*_pf)^X(v^X)\big]\, dv^X}{\displaystyle\int_{\mathbb{R}^d} \chi^X_p(v^X)\,\beta\Big(\alpha^X_p\big(f^X(p),(\mathrm{Exp}^*_pf)^X(v^X)\big)\Big)\, dv^X},
$$

as claimed. $\blacksquare$

**Theorem 4.4 (Existence of the aggregation transform).** Let $\chi : TM \to \mathbb{R}_{\geq 0}$ be
an aggregation window satisfying $\mathrm{supp}(\chi_p) \subset \{v \in T_pM : \|v\| \leq R\}$ for some
$R>0$ and all $p \in M$. Then, for any smooth value transformation
$V : TM \to \mathrm{Hom}(A_{\mathrm{in}},A_{\mathrm{out}})$, any smooth affinity function
$\alpha : A_{\mathrm{in}}\oplus_M A_{\mathrm{in}} \to \mathbb{R}$, and any smooth affinity activation
$\beta : \mathbb{R}\to\mathbb{R}_{>0}$ the induced aggregation transform
$A^\chi_{\alpha,\beta,V} : \Gamma(A_{\mathrm{in}}) \to \Gamma(A_{\mathrm{out}})$ exists. That is, both
the affinity normalization integral and the aggregation integral are finite, the normalization factor
is strictly positive, and the resulting output field is smooth.

**Proof.** Let $f \in \Gamma(A_{\mathrm{in}})$ be fixed. Define the feature-field-dependent kernel field
$K_f : TM \to \mathrm{Hom}(A_{\mathrm{in}},A_{\mathrm{out}})$,
$K_f(v) := \omega^\chi_{\alpha,\beta}(f)(v)\, V(v)$. Since $f$, $\mathrm{Exp}^*f$, $\alpha$, $\beta$,
$\chi$, and $V$ are smooth, the map $K_f$ is a smooth kernel field. Moreover, because
$\omega^\chi_{\alpha,\beta}(f)$ contains the factor $\chi$, we have

$$
\mathrm{supp}\big(K_f|_{T_pM}\big) \subseteq \mathrm{supp}(\chi_p) \subseteq \{v \in T_pM : \|v\| \leq R\} \qquad \forall p \in M.
$$

Therefore $K_f$ is a smooth kernel field that is compactly supported on each fiber $T_pM$, and the
kernel field transform of such a kernel field is well defined and gives a smooth section (the
well-definedness condition for compactly supported kernels is recorded in
[Conventions](conventions.md#kernel-fields-and-transporter-pullback); its proof is not reproduced on
this site). Hence $N_f := T_{K_f}(f)$ is well defined and satisfies
$N_f \in \Gamma(A_{\mathrm{out}})$. Pointwise, this section is given by

$$
N_f(p) = \int_{T_pM} \omega^\chi_{\alpha,\beta}(f)(v)\, V_p(v)\big[(\mathrm{Exp}^*_pf)(v)\big]\, dv.
$$

It remains to show that the normalization factor
$C_f(p) := C^\chi_{\alpha,\beta}(f,p) = \int_{T_pM} \omega^\chi_{\alpha,\beta}(f)(v)\, dv$ is finite,
smooth, and strictly positive. For each $p \in M$, the integrand $\omega^\chi_{\alpha,\beta}(f)|_{T_pM}$
is continuous and supported in $\mathrm{supp}(\chi_p)$. By assumption, this support is contained in
the compact set $\{v \in T_pM : \|v\| \leq R\}$. Thus $C_f(p)$ exists as an ordinary scalar integral.

To prove smoothness, let $X$ be a local gauge over an open set $U \subset M$. In this gauge, write
$p = p(x)$ and define $W^X_f(u,x) := \big(\omega^\chi_{\alpha,\beta}(f)\big)^X_{p(x)}(u)$. Then

$$
C_f(x) = \int_{\mathbb{R}^d} W^X_f(u,x)\sqrt{|\eta^X_x|}\, du.
$$

The function $W^X_f(u,x)\sqrt{|\eta^X_x|}$ is smooth in $(u,x)$. After possibly shrinking $U$, the
support assumption on $\chi$ gives a compact set $K \subset \mathbb{R}^d$ such that
$\mathrm{supp}\big(W^X_f(\cdot,x)\big) \subset K\ \forall x \in U$. Therefore differentiation under the
integral sign is justified by the standard dominated-convergence argument for parameter-dependent
integrals [Fol99, Theorem 2.27], and $C_f$ is smooth on $U$. Since $U$ was arbitrary, $C_f$ is smooth
on $M$.

Finally, since $\beta$ takes values in $\mathbb{R}_{>0}$ and $\chi \geq 0$, we have
$\omega^\chi_{\alpha,\beta}(f)(v) \geq 0$. Under the standing assumption that $\chi_p$ is not identically
zero for any $p \in M$, which is supported on a neighborhood of the origin in $T_pM$, the smoothness
and nonnegativity of $\chi_p$ imply that $\chi_p$ is positive on some nonempty neighborhood of
$0_p \in T_pM$. Since $\beta>0$, it follows that $C_f(p)>0$ for every $p\in M$. Therefore the
pointwise quotient $[A^\chi_{\alpha,\beta,V}(f)](p) = N_f(p)/C_f(p)$ is well defined. Since
$N_f\in\Gamma(A_{\mathrm{out}})$ and $C_f$ is a smooth strictly positive scalar function, the quotient
$N_f/C_f$ is a smooth section of $A_{\mathrm{out}}$. We've shown
$A^\chi_{\alpha,\beta,V}(f)\in\Gamma(A_{\mathrm{out}})$, i.e. the aggregation transform exists.
$\blacksquare$

## Coordinate independence and GM-aggregation

For an unconstrained aggregation transform $A^\chi_{\alpha,\beta,V}$ to define a GM-aggregation we
require that its coordinate expression be shared across every gauge in the chosen $G$-atlas and
across every spatial position $p \in M$. This requirement has two complementary interpretations.
First, gauge sharing encodes the geometric principle that no admissible gauge in the $G$-structure is
preferred. Second, spatial sharing imposes the usual weight-sharing prior, in the same sense as for GM-convolution
(see [Conventions](conventions.md#g-steerable-kernels-and-gm-convolution)). In both attention and convolution, the exact same aggregation components are shared at
all spatial locations: in attention the same affinity and value mechanisms are shared, often through
shared key, query, and value projections, while in convolution the convolutional kernel is shared.
Thus, a GM-aggregation is precisely an unconstrained aggregation transform whose local rule is
independent of both the chosen gauge and the spatial position at which it is evaluated. In this
section we state the coordinate independence conditions on each component of the aggregation
transform, then formally define GM-aggregation.

First, we address the coordinate independence of the value transformation. Here there is a subtle
distinction between normalized aggregation and GM-convolution. In the GM-convolution setting (see
[Conventions](conventions.md#g-steerable-kernels-and-gm-convolution)), the kernel field must be treated as a density-valued object in
coordinates. This is because the
convolutional kernel field transform is an unnormalized integral: the Riemannian volume factor
appearing in the coordinate expression has no denominator against which it can cancel.
Consequently, the shared convolutional kernel template must absorb the factor $\sqrt{|\eta^X_p|}$,
leading to the density-valued steerability condition of [Conventions](conventions.md#g-steerable-kernels-and-gm-convolution).

For the normalized aggregation transforms considered here, we assume the affinity is nontrivial and
that the aggregation contains the corresponding non-trivial normalization integral. In this case, the
Riemannian volume factor appears in both the numerator and the denominator, it is independent of
the tangent variable $v^X$, and thus it cancels in the quotient. Therefore the value transformation
need not be density-valued. Instead, it is naturally a hom-valued bundle map
$V : TM \to \mathrm{Hom}(A_{\mathrm{in}},A_{\mathrm{out}})$, whose coordinate expression is required
to be shared as an ordinary hom-valued template. That is, there exists a map
$V : \mathbb{R}^d \to \mathbb{R}^{c_{\mathrm{out}}\times c_{\mathrm{in}}}$ such that the coordinate
value transformation satisfies $V^X_p(v^X) = V(v^X)$ for every admissible gauge $X$ and every
$p \in M$.

The compatibility of this shared hom-valued template with changes of gauge is encoded by the
usual intertwining condition

$$
V(gv) = \rho_{\mathrm{out}}(g)V(v)\rho_{\mathrm{in}}(g)^{-1} \qquad \forall v \in \mathbb{R}^d,\ \forall g \in G. \tag{4.17}
$$

Thus, in normalized GM-aggregation, the value transformation is the hom-valued analogue of a
GM-convolutional kernel field, but it is not itself density-valued.

When $G \leq O(d)$, the matrix-valued versus matrix-valued-density template distinction is trivial as
$|\eta^X_p| = 1$ for all $p \in M$ and admissible gauges $X$. However, if one wants to recover
GM-convolution, when $G \not\leq O(d)$, as an unnormalized trivial-affinity special case, then this
cancellation mechanism is no longer available. In that case, the value transformation must again be
treated as a density-valued integration kernel, exactly as in [Conventions](conventions.md#g-steerable-kernels-and-gm-convolution).

**Definition 4.5 (GM-value transformation).** Let $A_{\mathrm{in}} = GM\times_{\rho_{\mathrm{in}}}\mathbb{R}^{c_{\mathrm{in}}}$
and $A_{\mathrm{out}} = GM\times_{\rho_{\mathrm{out}}}\mathbb{R}^{c_{\mathrm{out}}}$ be associated
feature vector bundles. Suppose $V : \mathbb{R}^d \to \mathbb{R}^{c_{\mathrm{out}}\times c_{\mathrm{in}}}$
is a smooth gauge equivariant template in the sense of Eq. (4.17), i.e. $V$ makes the diagram

$$
\begin{array}{ccc}
\mathbb{R}^d & \xrightarrow{V} & \mathbb{R}^{c_{\mathrm{out}}\times c_{\mathrm{in}}} \\
\downarrow{\scriptstyle g} & & \downarrow{\scriptstyle \rho_{\mathrm{Hom}}(g)} \\
\mathbb{R}^d & \xrightarrow{V} & \mathbb{R}^{c_{\mathrm{out}}\times c_{\mathrm{in}}}
\end{array}
$$

commute for all $g \in G$. Define the corresponding GM-value transformation as the smooth bundle
$M$-morphism $\mathcal{V}_V : TM \to \mathrm{Hom}(A_{\mathrm{in}},A_{\mathrm{out}})$ whose restriction
to each fiber $p \in M$ is given by

$$
\mathcal{V}_{V,p} := \big(\psi^X_{\mathrm{Hom},p}\big)^{-1} \circ V \circ \psi^X_{TM,p}, \tag{4.18}
$$

where $X$ is any admissible gauge about $p$.

Next, we address the affinity function. In a gauge $X$, the coordinatization of a smooth affinity
function $\alpha : A_{\mathrm{in}}\oplus_M A_{\mathrm{in}} \to \mathbb{R}$ is given by
$\alpha^X_p := \alpha_p \circ \big[(\psi^X_{A_{\mathrm{in}},p})^{-1}\oplus(\psi^X_{A_{\mathrm{in}},p})^{-1}\big]$.
If $Y$ is another admissible gauge in the considered $G$-atlas, then the corresponding coordinate
affinity functions satisfy

$$
\alpha^Y_p = \alpha^X_p \circ \Big[\rho_{\mathrm{in}}\big((g^{YX}_p)^{-1}\big)\oplus\rho_{\mathrm{in}}\big((g^{YX}_p)^{-1}\big)\Big], \tag{4.19}
$$

where $g^{YX}_p$ denotes the gauge transformation from $X$ to $Y$ at $p$.

Coordinate independence of the affinity function $\alpha$ requires $\alpha^X_p = a$ for every admissible
gauge $X$ and every spatial position $p \in M$, for some
$a : \mathbb{R}^{c_{\mathrm{in}}}\oplus\mathbb{R}^{c_{\mathrm{in}}} \to \mathbb{R}$. According to the
relation in (4.19), $a$ must be gauge-invariant:

$$
a(u\oplus v) = a\big(\rho_{\mathrm{in}}(g)u \oplus \rho_{\mathrm{in}}(g)v\big) \qquad \forall u,v\in\mathbb{R}^{c_{\mathrm{in}}},\ \forall g\in G. \tag{4.20}
$$

**Definition 4.6 (GM-affinity function).** Let $A = GM\times_\rho\mathbb{R}^c$ be an associated feature
vector bundle of type $\rho$. Suppose $a : \mathbb{R}^{c_{\mathrm{in}}}\oplus\mathbb{R}^{c_{\mathrm{in}}}\to\mathbb{R}$
is a smooth gauge invariant template in the sense of (4.20). Define the corresponding GM-affinity
function of type $\rho$ as the smooth map $\alpha_a : A\oplus_M A \to \mathbb{R}$ whose restriction to
each fiber over $p\in M$ is given by

$$
\alpha_{a,p} := a\circ\big(\psi^X_{A,p}\oplus\psi^X_{A,p}\big), \tag{4.21}
$$

where $X$ is any admissible gauge about $p$.

We likewise require the aggregation window $\chi : TM \to \mathbb{R}_{\geq 0}$ to be gauge-invariant.
The argument mirrors the one used for the affinity function. In any gauge $X$ from the considered
$G$-atlas, the coordinatization of $\chi$ over $p \in U_X$ is
$\chi^X_p : \mathbb{R}^d \to \mathbb{R}_{\geq 0}$, $\chi^X_p := \chi_p \circ (\psi^X_{TM,p})^{-1}$. If $Y$
is another admissible gauge, then the corresponding coordinate windows satisfy

$$
\chi^Y_p = \chi^X_p \circ \big((g^{YX}_p)^{-1}\big). \tag{4.22}
$$

Coordinate independence of $\chi$ requires that, in every admissible gauge $X$ and at every point
$p\in M$, the coordinate window is given by a single shared function $c : \mathbb{R}^d\to\mathbb{R}_{\geq0}$.
That is, $\chi^X_p = c$ for all $p\in M$ and all admissible $X$. By (4.22), this is possible only if $c$
satisfies

$$
c(gu) = c(u) \qquad \forall u\in\mathbb{R}^d,\ \forall g\in G. \tag{4.23}
$$

Thus, the shared coordinate aggregation window $c$ must be gauge-invariant.

**Definition 4.7 (GM-aggregation window).** Let $c : \mathbb{R}^d \to \mathbb{R}_{\geq0}$ be a smooth
template whose support is compact and contains the origin, and is gauge-invariant in the sense of
(4.23). Define the corresponding GM-aggregation window to be the smooth map
$\chi_c : TM \to \mathbb{R}_{\geq0}$ whose restriction to each fiber over $p\in M$ is given by

$$
\chi_{c,p} := c\circ\psi^X_{TM,p}, \tag{4.24}
$$

where $X$ is any admissible gauge about $p$.

Imposing coordinate independence on the affinity function and the aggregation window leads to
gauge-invariant templates. Consequently, the unnormalized scalar aggregation weight (cf. Definition 4.1)
has a shared coordinate expression in every admissible gauge.

**Definition 4.8 (GM-scalar aggregation weight).** Let $\alpha_a$ be a GM-affinity function with
gauge-invariant template $a$, let $\beta : \mathbb{R}\to\mathbb{R}_{>0}$ be a smooth affinity activation,
and let $\chi_c$ be a GM-aggregation window with gauge-invariant template $c$. Then, for each
$f\in\Gamma(A_{\mathrm{in}})$, define the induced GM-scalar aggregation weight
$w^c_{a,\beta}(f) := \omega^{\chi_c}_{\alpha_a,\beta}(f)$. For $p\in M$ and any admissible gauge $X$,
its coordinate expression is

$$
\big(w^c_{a,\beta}(f)\big)^X_p(v^X) = c(v^X)\,\beta\Big(a\big(f^X(p),(\mathrm{Exp}^*_pf)^X(v^X)\big)\Big),
$$

which follows from Proposition 4.3.

Having defined GM-aggregation windows, GM-affinity functions, and GM-value transformations, we
are now prepared to state the formal definition of a GM-aggregation transform.

**Definition 4.9 (GM-aggregation transform).** Let $A_{\mathrm{in}} = GM\times_{\rho_{\mathrm{in}}}\mathbb{R}^{c_{\mathrm{in}}}$
and $A_{\mathrm{out}} = GM\times_{\rho_{\mathrm{out}}}\mathbb{R}^{c_{\mathrm{out}}}$ be $G$-associated
vector bundles. Let $\chi_c$ be a GM-aggregation window with gauge-invariant template $c$, let
$\alpha_a$ be a GM-affinity function with gauge-invariant template $a$, let $\beta$ be a smooth
affinity activation, and let $\mathcal{V}_V$ be a GM-value transformation with gauge-equivariant
template $V$. The associated GM-aggregation transform is the aggregation transform
$A^c_{a,\beta,V} := A^{\chi_c}_{\alpha_a,\beta,\mathcal{V}_V}$. For $p\in M$ and any admissible gauge
$X$ in the considered $G$-atlas, its coordinate expression is

$$
[A^c_{a,\beta,V}(f)]^X(p) = \frac{\displaystyle\int_{\mathbb{R}^d} \big(w^c_{a,\beta}(f)\big)^X_p(v^X)\, V(v^X)\big[(\mathrm{Exp}^*_pf)^X(v^X)\big]\, dv^X}{\displaystyle\int_{\mathbb{R}^d} \big(w^c_{a,\beta}(f)\big)^X_p(v^X)\, dv^X},
$$

which follows from Proposition 4.3.

Notice that, in this definition, the GM-affinity function, GM-value transformation, and
GM-aggregation window were taken as given geometric objects. This is not necessary in practice. A
GM-aggregation transform may equivalently be specified by its shared coordinate templates: a
gauge-invariant affinity template $a$, a gauge-invariant aggregation window template $c$, and a
gauge-equivariant value transformation template $V$. The quadrature form of this transform on $S^2$
is given in [Discretization](discretization.md#the-discretized-aggregation).

## GM-Attention

In this section, we present a specialization of GM-aggregation that generalizes self-attention to
feature fields on manifolds. To encompass the different scoring mechanisms used in modern
attention architectures, we define continuum Euclidean self-attention to be a Euclidean aggregation
transform, as in (4.3), whose affinity function has the form $\alpha(x,x') = S(W_Qx,W_Kx')$, where
$W_Q,W_K : \mathbb{R}^{c_{\mathrm{in}}}\to\mathbb{R}^{c_{\mathrm{attn}}}$ are parameterized linear
query and key maps and $S : \mathbb{R}^{c_{\mathrm{attn}}}\times\mathbb{R}^{c_{\mathrm{attn}}}\to\mathbb{R}$
is a similarity score. We additionally require the value transformation to be spatially independent,
so that $V(p,q) = W_V(q-p)$ for some parameterized map
$W_V : \mathbb{R}^d\to\mathbb{R}^{c_{\mathrm{out}}\times c_{\mathrm{in}}}$. For example, scaled
dot-product attention [VSP+17] is recovered by taking $S(q,k) = q^\top k/\sqrt{c_{\mathrm{attn}}}$,
Graph Attention Networks [VCC+18] style additive attention by taking
$S(q,k) = \mathrm{LeakyReLU}(a^\top(q\| k))$ for a learned vector $a \in \mathbb{R}^{2c_{\mathrm{attn}}}$,
and multiplicative attention [LPM15] by taking $S(q,k) = q^\top W k$ for a learned matrix
$W \in \mathbb{R}^{c_{\mathrm{attn}}\times c_{\mathrm{attn}}}$. In all three of these particular works,
the value transformation is constant $W_V(v) = W_V$ for all $v\in\mathbb{R}^d$, as is typically the
case.

To generalize Euclidean self-attention to the GM setting, we must identify the GM-affinity function
and GM-value transformation which play the roles of this query-key affinity and value projection.
The key difference is that the Euclidean maps $W_Q,W_K$, and $W_V(v)$ can no longer be arbitrary
linear maps but rather must interact with gauge transformations in the correct way to maintain the
coordinate independence of the GM-affinity function and GM-value transformation. The precise
requirements to impose on $W_Q,W_K,W_V,$ and $S$ are derived below.

First, we address the query-key affinity. The GM-affinity function which generalizes the Euclidean
query-key affinity has template $a(u,v) = S(W_Qu,W_Kv)$, where
$W_Q,W_K : \mathbb{R}^{c_{\mathrm{in}}}\to\mathbb{R}^{c_{\mathrm{attn}}}$ are linear maps for some
latent dimension $c_{\mathrm{attn}}>0$. In order for $a$ to define a GM-affinity template, it must be
gauge-invariant in the sense of (4.20). Thus we require

$$
S(W_Q\rho_{\mathrm{in}}(g)u,\, W_K\rho_{\mathrm{in}}(g)v) = S(W_Qu,W_Kv) \qquad \forall g\in G,\ \forall u,v\in\mathbb{R}^{c_{\mathrm{in}}}. \tag{4.25}
$$

A sufficient condition for (4.25) to hold is to require the query and key maps to be equivariant into
a common latent representation and require the score function $S$ to be invariant with respect to
that latent representation. Namely, let $\rho_{\mathrm{attn}} : G\to GL(\mathbb{R}^{c_{\mathrm{attn}}})$
be a representation, and suppose $W_Q\rho_{\mathrm{in}}(g) = \rho_{\mathrm{attn}}(g)W_Q$,
$W_K\rho_{\mathrm{in}}(g) = \rho_{\mathrm{attn}}(g)W_K$,
$S(\rho_{\mathrm{attn}}(g)q,\rho_{\mathrm{attn}}(g)k) = S(q,k)$ for every $g\in G$ and
$q,k\in\mathbb{R}^{c_{\mathrm{attn}}}$. Then
$S(W_Q\rho_{\mathrm{in}}(g)u,W_K\rho_{\mathrm{in}}(g)v) = S(\rho_{\mathrm{attn}}(g)W_Qu,\rho_{\mathrm{attn}}(g)W_Kv) = S(W_Qu,W_Kv)$.
Therefore equivariant query and key maps into a common representation with invariant score
function produce a valid GM-affinity template. This is a practical condition because the spaces
$\mathrm{Hom}_G(\rho_{\mathrm{in}},\rho_{\mathrm{attn}})$ of equivariant linear maps are
finite-dimensional vector spaces which can be parameterized by choosing a basis of intertwiners
[WC19, WFVW23], so the query and key maps can be learned by choosing parameters in these
intertwiner spaces (for $SO(2)$, see [Intertwiners between arbitrary
representations](so2_representations.md#intertwiners-between-arbitrary-representations)).

We permit any GM-value transformation in GM-attention as GM-value transformations are already
independent of spatial position. For $SO(2)$ the value templates are the steerable kernels of
[Steerable transformations between arbitrary
representations](so2_representations.md#steerable-transformations-between-arbitrary-representations).
We therefore define the following:

**Definition 4.10 (GM-self-attention).** Let $A_{\mathrm{in}} = GM\times_{\rho_{\mathrm{in}}}\mathbb{R}^{c_{\mathrm{in}}}$
and $A_{\mathrm{out}} = GM\times_{\rho_{\mathrm{out}}}\mathbb{R}^{c_{\mathrm{out}}}$ be associated
feature vector bundles, $W_V$ be a gauge-equivariant GM-value transformation template, $c$ be a
gauge-invariant GM-aggregation window template, and $\beta$ be any affinity activation. Let
$\rho_{\mathrm{attn}} : G\to GL(\mathbb{R}^{c_{\mathrm{attn}}})$ be a representation. Suppose
$W_Q,W_K\in\mathrm{Hom}(\mathbb{R}^{c_{\mathrm{in}}},\mathbb{R}^{c_{\mathrm{attn}}})$,
$S : \mathbb{R}^{c_{\mathrm{attn}}}\times\mathbb{R}^{c_{\mathrm{attn}}}\to\mathbb{R}$ are maps
satisfying for all $g\in G$ and $q,k\in\mathbb{R}^{c_{\mathrm{attn}}}$

$$
W_Q\rho_{\mathrm{in}}(g) = \rho_{\mathrm{attn}}(g)W_Q, \qquad W_K\rho_{\mathrm{in}}(g) = \rho_{\mathrm{attn}}(g)W_K, \qquad S(\rho_{\mathrm{attn}}(g)q,\rho_{\mathrm{attn}}(g)k) = S(q,k).
$$

Define $a(u,v) = S(W_Qu,W_Kv)$. The associated GM-self-attention operator is

$$
SA^c_{\beta,S}[W_Q,W_K,W_V] := A^c_{a,\beta,W_V} : \Gamma(A_{\mathrm{in}}) \to \Gamma(A_{\mathrm{out}}).
$$

This definition is closely related to Equivariant Mesh Attention Networks (EMAN) [BGPV+22] and
Gauge Equivariant Transformer [HCD+21], which implement gauge-equivariant attention on
discretized surface meshes. However, the formulation here is continuous and coordinate
independent, expressing attention intrinsically as an operator on sections of associated vector
bundles over a Riemannian manifold.

GM-self-attention can naturally be extended to multi-head self-attention provided we can decompose
the output feature vector bundle as a direct sum of subbundles, one for each GM-attention head.

**Definition 4.11 (Multi-head GM-self-attention).** Let $H$ denote the number of self-attention
heads. Let $A_{\mathrm{in}} = GM\times_{\rho_{\mathrm{in}}}\mathbb{R}^{c_{\mathrm{in}}}$ and
$A_{\mathrm{out}} = GM\times_{\rho_{\mathrm{out}}}\mathbb{R}^{c_{\mathrm{out}}}$ be associated feature
vector bundles. Suppose that the output representation decomposes as
$(\mathbb{R}^{c_{\mathrm{out}}},\rho_{\mathrm{out}}) \cong \bigoplus_{h=1}^H (\mathbb{R}^{c_{\mathrm{out},h}},\rho_{\mathrm{out},h})$,
$\sum_{h=1}^H c_{\mathrm{out},h} = c_{\mathrm{out}}$, where
$\rho_{\mathrm{out},h} : G\to GL(\mathbb{R}^{c_{\mathrm{out},h}})$. Fix a GM-aggregation window
template $c$ and affinity activation $\beta$.

For each $h=1,\ldots,H$, let $\rho_{\mathrm{attn},h} : G\to GL(\mathbb{R}^{c_{\mathrm{attn},h}})$.
Suppose $W_Q^{(h)},W_K^{(h)}\in\mathrm{Hom}(\mathbb{R}^{c_{\mathrm{in}}},\mathbb{R}^{c_{\mathrm{attn},h}})$,
$W_V^{(h)} : \mathbb{R}^d\to\mathrm{Hom}(\mathbb{R}^{c_{\mathrm{in}}},\mathbb{R}^{c_{\mathrm{out},h}})$,
$W_{\mathrm{out}}\in\mathrm{Hom}(\mathbb{R}^{c_{\mathrm{out}}},\mathbb{R}^{c_{\mathrm{out}}})$,
$S^{(h)} : \mathbb{R}^{c_{\mathrm{attn},h}}\times\mathbb{R}^{c_{\mathrm{attn},h}}\to\mathbb{R}$ are
maps satisfying, for all $g\in G$, $v\in\mathbb{R}^d$, and $q,k\in\mathbb{R}^{c_{\mathrm{attn},h}}$,

$$
W_Q^{(h)}\rho_{\mathrm{in}}(g) = \rho_{\mathrm{attn},h}(g)W_Q^{(h)}, \qquad W_K^{(h)}\rho_{\mathrm{in}}(g) = \rho_{\mathrm{attn},h}(g)W_K^{(h)},
$$

$$
W_V^{(h)}(gv) = \rho_{\mathrm{out},h}(g)W_V^{(h)}(v)\rho_{\mathrm{in}}(g)^{-1}, \qquad S^{(h)}(q,k) = S^{(h)}(\rho_{\mathrm{attn},h}(g)q,\rho_{\mathrm{attn},h}(g)k)
$$

for each $h=1,\ldots,H$, and $W_{\mathrm{out}}\rho_{\mathrm{out}}(g) = \rho_{\mathrm{out}}(g)W_{\mathrm{out}}$.
The associated multi-head GM-self-attention operator is the map
$\mathrm{MHSA}^c : \Gamma(A_{\mathrm{in}}) \to \Gamma(A_{\mathrm{out}})$ whose coordinate expression in
any admissible gauge $X$ is

$$
\big(\mathrm{MHSA}^c(f)\big)^X(p) = W_{\mathrm{out}}\left(\bigoplus_{h=1}^H \Big(SA^c_{\beta,S^{(h)}}\big[W_Q^{(h)},W_K^{(h)},W_V^{(h)}\big](f)\Big)^X(p)\right).
$$

If $\mathrm{supp}(c)$ is compact, then Theorem 4.4 guarantees that
$SA^c_{\beta,S^{(h)}}[W_Q^{(h)},W_K^{(h)},W_V^{(h)}]$ exists for each $h=1,\ldots,H$. When this is the
case, $\mathrm{MHSA}^c(f)$ produces a smooth section of
$\bigoplus_{h=1}^H A_{\mathrm{out},h} \cong GM\times_{\oplus_h\rho_{\mathrm{out},h}}\Big(\bigoplus_{h=1}^H\mathbb{R}^{c_{\mathrm{out},h}}\Big) = A_{\mathrm{out}}$.
Since $W_{\mathrm{out}}$ is an intertwiner, it induces a smooth vector bundle morphism
$W_{\mathrm{out}} : A_{\mathrm{out}}\to A_{\mathrm{out}}$, as described for $1\times1$ GM-convolutions in
[Conventions](conventions.md#11-gm-convolutions). Applying a smooth
vector bundle morphism fiberwise to a smooth section gives a smooth section. Therefore, writing
$SA^{(h)} := SA^c_{\beta,S^{(h)}}\big[W_Q^{(h)},W_K^{(h)},W_V^{(h)}\big]$ for the $h$-th head,

$$
\mathrm{MHSA}^c(f) = W_{\mathrm{out}}\left(\bigoplus_{h=1}^H SA^{(h)}(f)\right) \in \Gamma(A_{\mathrm{out}})
$$

and thus multi-head GM-attention exists whenever each GM-attention head exists.

## Isometry equivariance of GM-aggregation

Recall that GM-convolution is equivariant with respect to all $G$-structure preserving isometries
(the equivariance theorem is stated in [Conventions](conventions.md#behavior-under-isometries)). In this section, we make
precise the idea that GM-aggregation can be viewed as a normalized GM-convolution whose kernel
depends on the input feature field. We then argue that this viewpoint is sufficient to extend
$\mathrm{Isom}_{GM}$ equivariance from GM-convolutions to GM-aggregation transforms. In
particular, we show that

$$
A^c_{a,\beta,V}(\phi \cdot f) = \phi\cdot[A^c_{a,\beta,V}(f)] \tag{4.26}
$$

for any $G$-structure preserving isometry $\phi\in\mathrm{Isom}_{GM}$, where $\cdot$ denotes the
action of $\mathrm{Isom}_{GM}$ on feature fields, $\phi\cdot f := \phi_{*,A}\circ f\circ\phi^{-1}$ (written
$\phi\rhd f$ in [Conventions](conventions.md#behavior-under-isometries)).

Let $A_{\mathrm{in}} = GM\times_{\rho_{\mathrm{in}}}\mathbb{R}^{c_{\mathrm{in}}}$ and
$A_{\mathrm{out}} = GM\times_{\rho_{\mathrm{out}}}\mathbb{R}^{c_{\mathrm{out}}}$ be associated feature
vector bundles, $\alpha_a$ a GM-affinity function, $\beta$ an affinity activation, $\mathcal{V}_V$ a
GM-value transformation, and $\chi_c$ a GM-aggregation window. Following the notation used in the
proof of Proposition 4.3, for an $f\in\Gamma(A_{\mathrm{in}})$ define the feature-field-dependent
kernel field to be

$$
K_f : TM\to\mathrm{Hom}(A_{\mathrm{in}},A_{\mathrm{out}}), \qquad K_f(v) := w^c_{a,\beta}(f)(v)\,\mathcal{V}_V(v) \tag{4.27}
$$

where $w^c_{a,\beta}(f)$ is the unnormalized feature-field-dependent GM-scalar aggregation weight
(Def. 4.8). Then setting

$$
N_f(p) = \int_{T_pM} K_f(v)\big[(\mathrm{Exp}^*_pf)(v)\big]\, dv, \qquad C_f(p) = \int_{T_pM} w^c_{a,\beta}(f)(v)\, dv \tag{4.28}
$$

gives us the GM-aggregation transform as $\big(A^c_{a,\beta,V}(f)\big)(p) = N_f(p)/C_f(p)$.

To prove the desired isometry equivariance of GM-aggregation (4.26) it suffices to show that the
numerator and the denominator are isometry equivariant; that is,
$N_{\phi\cdot f} = \phi\cdot N_f$, $C_{\phi\cdot f} = \phi\cdot C_f$ $\forall \phi\in\mathrm{Isom}_{GM}$,
where $\phi\cdot C_f := C_f\circ\phi^{-1}$ since $C_f$ is a scalar field and isometries act on scalars
trivially.

**First we show $C_f$ is $\mathrm{Isom}_{GM}$ equivariant.** Let $\phi\in\mathrm{Isom}_{GM}$ and
$p\in M$. Then

$$
\begin{aligned}
C_{\phi\cdot f}(p)
&\overset{(1)}{=} \int_{T_pM} w^c_{a,\beta}(\phi\cdot f)(v)\, dv \\
&\overset{(2)}{=} \int_{T_pM} \chi_c(v)\,\beta\Big(\alpha_a\big((\phi\cdot f)(p),\, (\mathrm{Exp}^*_p(\phi\cdot f))(v)\big)\Big)\, dv \\
&\overset{(3)}{=} \int_{T_pM} \chi_c(v)\,\beta\Big(\alpha_a\big(\phi_{*,A_{\mathrm{in}}}f(\phi^{-1}(p)),\, \phi_{*,A_{\mathrm{in}}}(\mathrm{Exp}^*_{\phi^{-1}(p)}f)(\phi^{-1}_{*,TM}v)\big)\Big)\, dv \\
&\overset{(4)}{=} \int_{T_{\phi^{-1}(p)}M} \chi_c(\phi_{*,TM}\widetilde v)\,\beta\Big(\alpha_a\big(\phi_{*,A_{\mathrm{in}}}f(\phi^{-1}(p)),\, \phi_{*,A_{\mathrm{in}}}(\mathrm{Exp}^*_{\phi^{-1}(p)}f)(\widetilde v)\big)\Big)\, d\widetilde v \\
&\overset{(5)}{=} \int_{T_{\phi^{-1}(p)}M} \chi_c(\widetilde v)\,\beta\Big(\alpha_a\big(\phi_{*,A_{\mathrm{in}}}f(\phi^{-1}(p)),\, \phi_{*,A_{\mathrm{in}}}(\mathrm{Exp}^*_{\phi^{-1}(p)}f)(\widetilde v)\big)\Big)\, d\widetilde v \\
&\overset{(6)}{=} \int_{T_{\phi^{-1}(p)}M} \chi_c(\widetilde v)\,\beta\Big(\alpha_a\big(f(\phi^{-1}(p)),\, (\mathrm{Exp}^*_{\phi^{-1}(p)}f)(\widetilde v)\big)\Big)\, d\widetilde v \\
&\overset{(7)}{=} \int_{T_{\phi^{-1}(p)}M} w^c_{a,\beta}(f)(\widetilde v)\, d\widetilde v \\
&\overset{(8)}{=} C_f(\phi^{-1}(p)) \\
&\overset{(9)}{=} [\phi\cdot C_f](p).
\end{aligned}
$$

Step (1) uses the definition of $C_f$ (4.28). Step (2) expands $w^c_{a,\beta}$ (4.11). Step (3) uses the
definition of $\phi\cdot f$ and the identity

$$
\big(\mathrm{Exp}^*_p(\phi\cdot f)\big)(v) = \phi_{*,A_{\mathrm{in}}}\big(\mathrm{Exp}^*_{\phi^{-1}(p)}f\big)\big(\phi^{-1}_{*,TM}v\big) \tag{4.29}
$$

which was proven in Theorem 13.1.4 in [WFVW23] and relies on the compatibility of the connection
with the $G$-structure (which the Levi-Civita connection is guaranteed to be, as argued in [GQ20]).
Step (4) makes the change of variables $\widetilde v = \phi^{-1}_{*,TM}v$. Step (5) uses the fact that for
any gauge $X$ from the $G$-atlas about $p$, the gauge about $\phi^{-1}(p)$ given by
$\psi^{\widetilde X}_{TM,\phi^{-1}(p)} := \psi^X_{TM,p}\circ\phi_{*,TM,\phi^{-1}(p)}$ is in the $G$-atlas
since $\phi$ preserves the $G$-structure. Then using the definition of a GM-aggregation window Def.
4.7 we have $\chi_c(\phi_{*,TM}\widetilde v) = c\big(\psi^X_{TM,p}\phi_{*,TM}\widetilde v\big) = c\big(\psi^{\widetilde X}_{TM,\phi^{-1}(p)}\widetilde v\big) = \chi_c(\widetilde v)$.
Step (6) similarly uses the fact that in any gauge $X$ from the $G$-atlas about $p$, the gauge about
$\phi^{-1}(p)$ given by $\psi^{\widetilde X}_{A_{\mathrm{in}},\phi^{-1}(p)} := \psi^X_{A_{\mathrm{in}},p}\circ\phi_{*,A_{\mathrm{in}},\phi^{-1}(p)}$
is in the $G$-atlas. Thus, using the definition of the GM-affinity function Def. 4.6 we have

$$
\begin{aligned}
\alpha_a\big(\phi_{*,A_{\mathrm{in}}}f(\phi^{-1}(p)),\, \phi_{*,A_{\mathrm{in}}}(\mathrm{Exp}^*_{\phi^{-1}(p)}f)(\widetilde v)\big)
&= a\big(\psi^X_{A_{\mathrm{in}},p}\phi_{*,A_{\mathrm{in}}}f(\phi^{-1}(p)),\, \psi^X_{A_{\mathrm{in}},p}\phi_{*,A_{\mathrm{in}}}(\mathrm{Exp}^*_{\phi^{-1}(p)}f)(\widetilde v)\big) \\
&= a\big(\psi^{\widetilde X}_{A_{\mathrm{in}},\phi^{-1}(p)}f(\phi^{-1}(p)),\, \psi^{\widetilde X}_{A_{\mathrm{in}},\phi^{-1}(p)}(\mathrm{Exp}^*_{\phi^{-1}(p)}f)(\widetilde v)\big) \\
&= \alpha_a\big(f(\phi^{-1}(p)),\, (\mathrm{Exp}^*_{\phi^{-1}(p)}f)(\widetilde v)\big).
\end{aligned}
$$

Step (7) uses the definition of $w^c_{a,\beta}$, step (8) uses the definition of $C_f$, and lastly step (9)
uses the definition of $\phi\cdot C_f$.

**We now show that the numerator $N_f$ is $\mathrm{Isom}_{GM}$ equivariant.** Let
$\phi\in\mathrm{Isom}_{GM}$ and $p\in M$. As a first step we prove that the feature-field-dependent
kernel field $K_f$ is equivariant in the following sense

$$
K_{\phi\cdot f} = \phi_{*,K}K_f := \phi_{*,A_{\mathrm{out}}}K_f\big(\phi^{-1}_{*,TM}\big)\phi^{-1}_{*,A_{\mathrm{in}}}, \tag{4.30}
$$

where $\phi_{*,K}$ denotes the induced action of isometries on kernel fields as defined in Def. 13.2.2
in [WFVW23]. Intuitively, this equivariance says that the kernel induced by the transformed field at
$p$, namely $K_{\phi\cdot f}|_p$, is obtained by transporting the original kernel $K_f|_{\phi^{-1}(p)}$
from $\phi^{-1}(p)$ to $p$ along the $\mathrm{Isom}_{GM}$-orbit.

For an arbitrary gauge $X$ from the $G$-atlas about $\phi^{-1}(p)$ and $v\in T_pM$ we have

$$
\begin{aligned}
\phi_{*,K}K_f(v)
&\overset{(10)}{=} \phi_{*,A_{\mathrm{out}}}K_f\big(\phi^{-1}_{*,TM}v\big)\phi^{-1}_{*,A_{\mathrm{in}}} \\
&\overset{(11)}{=} \phi_{*,A_{\mathrm{out}}}\, w^c_{a,\beta}(f)\big(\phi^{-1}_{*,TM}v\big)\, \mathcal{V}_V\big(\phi^{-1}_{*,TM}v\big)\,\phi^{-1}_{*,A_{\mathrm{in}}} \\
&\overset{(12)}{=} w^c_{a,\beta}(\phi\cdot f)(v)\,\phi_{*,A_{\mathrm{out}}}\mathcal{V}_V\big(\phi^{-1}_{*,TM}v\big)\phi^{-1}_{*,A_{\mathrm{in}}} \\
&\overset{(13)}{=} w^c_{a,\beta}(\phi\cdot f)(v)\,\phi_{*,\mathrm{Hom}}\mathcal{V}_V\, \phi^{-1}_{*,TM}(v) \\
&\overset{(14)}{=} w^c_{a,\beta}(\phi\cdot f)(v)\,\phi_{*,\mathrm{Hom}}\big(\psi^X_{\mathrm{Hom},\phi^{-1}(p)}\big)^{-1}V\, \psi^X_{TM,\phi^{-1}(p)}\phi^{-1}_{*,TM}(v) \\
&\overset{(15)}{=} w^c_{a,\beta}(\phi\cdot f)(v)\,\big(\psi^{\widetilde X}_{\mathrm{Hom},p}\big)^{-1}V\, \psi^{\widetilde X}_{TM,p}(v) \\
&\overset{(16)}{=} w^c_{a,\beta}(\phi\cdot f)(v)\,\mathcal{V}_V(v) \\
&\overset{(17)}{=} K_{\phi\cdot f}(v).
\end{aligned}
$$

Step (10) expands the definition of $\phi_{*,K}K_f$ (4.30). Step (11) expands the definition of $K_f$
(4.27). Step (12) pulls the scalar $w^c_{a,\beta}$ out front and uses the identity
$w^c_{a,\beta}(f)(\phi^{-1}_{*,TM}v) = w^c_{a,\beta}(\phi\cdot f)(v)$, which is justified by the work done
in the integrand in steps (1)–(7) above. Step (13) uses the definition
$\phi_{*,\mathrm{Hom}}K := \phi_{*,A_{\mathrm{out}}}K\phi^{-1}_{*,A_{\mathrm{in}}}$. Step (14) uses the
definition of the GM-value transformation Def. 4.5. Step (15) defines the gauges
$\psi^{\widetilde X}_{\mathrm{Hom},p} := \psi^X_{\mathrm{Hom},\phi^{-1}(p)}\circ\phi^{-1}_{*,\mathrm{Hom},p}$,
$\psi^{\widetilde X}_{TM,p} := \psi^X_{TM,\phi^{-1}(p)}\circ\phi^{-1}_{*,TM,p}$. Step (16) again uses the
definition of GM-value transformation $\mathcal{V}_V$. Step (17) uses the definition of the
feature-field-dependent kernel field (4.27).

Having proven that the feature-field-dependent kernel field $K_f$ is equivariant with respect to
$G$-structure preserving isometries, we are now equipped to verify the desired
$\mathrm{Isom}_{GM}$-equivariance of $N_f$. Observe,

$$
\begin{aligned}
N_{\phi\cdot f}(p)
&\overset{(18)}{=} \int_{T_pM} K_{\phi\cdot f}(v)\big[(\mathrm{Exp}^*_p(\phi\cdot f))(v)\big]\, dv \\
&\overset{(19)}{=} \int_{T_pM} \phi_{*,A_{\mathrm{out}}}K_f(\phi^{-1}_{*,TM}v)\,\phi^{-1}_{*,A_{\mathrm{in}}}\big[(\mathrm{Exp}^*_p(\phi\cdot f))(v)\big]\, dv \\
&\overset{(20)}{=} \int_{T_pM} \phi_{*,A_{\mathrm{out}}}K_f(\phi^{-1}_{*,TM}v)\,\phi^{-1}_{*,A_{\mathrm{in}}}\phi_{*,A_{\mathrm{in}}}\big[(\mathrm{Exp}^*_{\phi^{-1}(p)}f)(\phi^{-1}_{*,TM}v)\big]\, dv \\
&\overset{(21)}{=} \int_{T_pM} \phi_{*,A_{\mathrm{out}}}K_f(\phi^{-1}_{*,TM}v)\big[(\mathrm{Exp}^*_{\phi^{-1}(p)}f)(\phi^{-1}_{*,TM}v)\big]\, dv \\
&\overset{(22)}{=} \phi_{*,A_{\mathrm{out}}}\int_{T_{\phi^{-1}(p)}M} K_f(\widetilde v)\big[(\mathrm{Exp}^*_{\phi^{-1}(p)}f)(\widetilde v)\big]\, d\widetilde v \\
&\overset{(23)}{=} \phi_{*,A_{\mathrm{out}}}N_f(\phi^{-1}(p)) \\
&\overset{(24)}{=} [\phi\cdot N_f](p).
\end{aligned}
$$

Step (18) uses the definition of $N_f$ (4.28). Step (19) uses equivariance of $K_f$ (4.30). Step (20)
uses the isometry action on the transporter pullback (4.29). Step (21) simply uses
$\mathrm{Id}_{A_{\mathrm{in}}} = \phi^{-1}_{*,A_{\mathrm{in}}}\phi_{*,A_{\mathrm{in}}}$. Step (22) makes the
change of variables $\widetilde v = \phi^{-1}_{*,TM}v$ and pulls out the linear map $\phi_{*,A_{\mathrm{out}}}$.
Step (23) uses the definition of $N_f$, and lastly (24) uses the definition of the isometry action on
feature fields.

**Theorem 4.12 ($\mathrm{Isom}_{GM}$ equivariance of GM-aggregation).** A GM-aggregation
$A^c_{a,\beta,V} : \Gamma(A_{\mathrm{in}}) \to \Gamma(A_{\mathrm{out}})$ with GM-affinity function
$\alpha_a$, affinity activation $\beta$, GM-value transformation $\mathcal{V}_V$, and GM-aggregation
window $\chi_c$ is equivariant with respect to every $G$-structure preserving isometry
$\phi\in\mathrm{Isom}_{GM}$; that is,

$$
A^c_{a,\beta,V}(\phi\cdot f) = \phi\cdot[A^c_{a,\beta,V}(f)] \qquad \forall f\in\Gamma(A_{\mathrm{in}}),\ \phi\in\mathrm{Isom}_{GM}.
$$

Therefore, the diagram

$$
\begin{array}{ccc}
\Gamma(A_{\mathrm{in}}) & \xrightarrow{A^c_{a,\beta,V}} & \Gamma(A_{\mathrm{out}}) \\
\downarrow{\scriptstyle\phi\cdot} & & \downarrow{\scriptstyle\phi\cdot} \\
\Gamma(A_{\mathrm{in}}) & \xrightarrow{A^c_{a,\beta,V}} & \Gamma(A_{\mathrm{out}})
\end{array}
$$

commutes for all $\phi\in\mathrm{Isom}_{GM}$.

**Proof.** See the above discussion. $\blacksquare$

As noted in [Conventions](conventions.md#behavior-under-isometries), if $G \geq O(d)$, then $\mathrm{Isom}_{GM}$ is the full isometry group
$\mathrm{Isom}(M)$. This observation gives the following corollary.

**Corollary 4.13 (Full isometry equivariance of $O_M$-aggregation).** If $G \geq O(d)$, then an
$O_M$-aggregation $A^c_{a,\beta,V} : \Gamma(A_{\mathrm{in}}) \to \Gamma(A_{\mathrm{out}})$ is equivariant
with respect to every isometry of $M$; that is,

$$
A^c_{a,\beta,V}(\phi\cdot f) = \phi\cdot[A^c_{a,\beta,V}(f)] \qquad \forall f\in\Gamma(A_{\mathrm{in}}),\ \phi\in\mathrm{Isom}(M).
$$

## References

- [WFVW23] Maurice Weiler, Patrick Forré, Erik Verlinde, and Max Welling. *Equivariant and
  Coordinate Independent Convolutional Networks: A Gauge Field Theory of Neural Networks*.
  Online monograph, 2023.
- [BGPV+22] Sourya Basu, Jose Gallego-Posada, Francesco Viganò, James Rowbottom, and Taco
  Cohen. *Equivariant mesh attention networks*. Transactions on Machine Learning Research, 2022.
- [HCD+21] Lingshen He, Yiming Dong, Yisen Wang, Dacheng Tao, and Zhouchen Lin. *Gauge
  equivariant transformer*. Advances in Neural Information Processing Systems, volume 34, pages
  27331–27343, 2021.
- [dHWCW21] Pim de Haan, Maurice Weiler, Taco Cohen, and Max Welling. *Gauge equivariant mesh
  CNNs: Anisotropic convolutions on geometric graphs*. International Conference on Learning
  Representations, 2021.
- [CW16] Taco S. Cohen and Max Welling. *Group equivariant convolutional networks*. Proceedings
  of the 33rd International Conference on Machine Learning, 2016.
- [WC19] Maurice Weiler and Gabriele Cesa. *General E(2)-equivariant steerable CNNs*. Advances in
  Neural Information Processing Systems, volume 32, 2019.
- [VSP+17] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N.
  Gomez, Łukasz Kaiser, and Illia Polosukhin. *Attention is all you need*. Advances in Neural
  Information Processing Systems, volume 30, 2017.
- [WGGH18] Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He. *Non-local neural
  networks*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition,
  pages 7794–7803, 2018.
- [LBBH98] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. *Gradient-based learning
  applied to document recognition*. Proceedings of the IEEE, 86(11):2278–2324, 1998.
- [KSH12] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. *Imagenet classification with deep
  convolutional neural networks*. Advances in Neural Information Processing Systems, volume 25,
  2012.
- [CKLS25] Edoardo Calvello, Nikola B. Kovachki, Matthew E. Levine, and Andrew M. Stuart.
  *Continuum attention for neural operators*. Journal of Machine Learning Research, 26(300):1–52,
  2025.
- [RPV+19] Prajit Ramachandran, Niki Parmar, Ashish Vaswani, Irwan Bello, Anselm Levskaya, and
  Jon Shlens. *Stand-alone self-attention in vision models*. Advances in Neural Information
  Processing Systems, volume 32, 2019.
- [VCC+18] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and
  Yoshua Bengio. *Graph attention networks*. International Conference on Learning Representations,
  2018.
- [LPM15] Thang Luong, Hieu Pham, and Christopher D. Manning. *Effective approaches to
  attention-based neural machine translation*. Proceedings of the 2015 Conference on Empirical
  Methods in Natural Language Processing, pages 1412–1421, 2015.
- [BZV+19] Irwan Bello, Barret Zoph, Ashish Vaswani, Jonathon Shlens, and Quoc V. Le. *Attention
  augmented convolutional networks*. Proceedings of the IEEE/CVF International Conference on
  Computer Vision, 2019.
- [Fol99] Gerald B. Folland. *Real Analysis: Modern Techniques and Their Applications*. John Wiley
  & Sons, 2nd edition, 1999.
- [GQ20] Jean Gallier and Jocelyn Quaintance. *Differential Geometry and Lie Groups: A
  Computational Perspective*, volume 12 of Geometry and Computing. Springer, 2020.
