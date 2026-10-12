# Representations of SO(2)

This is the direct math
counterpart of [`gm_attention.representations`](../api/representations.md).

There are two equivariant constraints which need to be explicitly solved in order to implement an
$SO(S^2)$-attention layer $\Gamma(A_{\mathrm{in}}) \to \Gamma(A_{\mathrm{out}})$ (see
[GM-Attention](gm_aggregation.md#gm-attention)), where
$A_{\mathrm{in}}, A_{\mathrm{out}}$ are associated feature vector bundles of type
$\rho_{\mathrm{in}}, \rho_{\mathrm{out}}$, respectively. The first is the intertwiner constraint

$$
W = \rho_{\mathrm{out}}(g)\, W\, \rho_{\mathrm{in}}(g)^{-1} \qquad \forall\, g \in SO(2) \tag{5.6}
$$

which must be satisfied by the query and key transformations with $\rho_{\mathrm{out}} = \rho_{\mathrm{attn}}$.
To maintain gauge equivariance throughout the network, the weight matrix of any fully connected
linear layer must also satisfy (5.6). The second constraint is the $SO(2)$-steerability constraint

$$
W_V(gv) = \rho_{\mathrm{out}}(g)\, W_V(v)\, \rho_{\mathrm{in}}(g)^{-1} \qquad \forall\, v \in \mathbb{R}^2,\ \forall\, g \in SO(2), \tag{5.7}
$$

which is satisfied by the value transformation template
$W_V : \mathbb{R}^2 \to \mathrm{Hom}(\mathbb{R}^{c_{\mathrm{in}}}, \mathbb{R}^{c_{\mathrm{out}}})$.

We follow the method used in works like [dHWCW21, BGPV+22, WFVW23, LW20, WC19, KdHC19].
In particular, we observe that any finite dimensional representation of $SO(2)$ decomposes as a
direct sum of irreducible representations of $SO(2)$ and thus the constraints (5.6) and (5.7) can be
solved independently for each constituent input and output irreducible type. Then the final
solutions to the constraints take the form of block matrices with each block corresponding to the
solution of an irreducible input and output type. A similar argument can be applied to any compact
structure group, as demonstrated in [WFVW23]. The paper states the parameterizations derived below
in Appendix B.1.

## Irreducible real representations of SO(2)

We parameterize $SO(2)$ by

$$
r_\theta := \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \qquad \theta \in \mathbb{R}/2\pi\mathbb{Z}.
$$

Under the standard identification $\mathbb{R}^2 \cong \mathbb{C}$ with $(x,y) \longleftrightarrow x+iy$, the
$U(1)$ action on $\mathbb{C}$ of multiplication by $e^{i\theta}$ corresponds to the $SO(2)$ action on
$\mathbb{R}^2$ of multiplication by $r_\theta$. With this perspective in mind, we define

$$
I := \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \qquad J := \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} \qquad S := \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} \tag{5.8}
$$

so that $I$ represents multiplication by $1$, $J$ represents multiplication by $i$ ($J$ is the standard
complex structure on $\mathbb{R}^2$), and $S$ corresponds to complex conjugation. In particular,
$J^2 = -I$ and

$$
r_\theta = \cos\theta\, I + \sin\theta\, J = \exp(\theta J), \tag{5.9}
$$

which is to say that $J$ is a generator for $\mathfrak{so}(2)$.

**Proposition 5.1 (Characterization of finite dimensional $SO(2)$ representations).** Up to
equivalence, the finite-dimensional irreducible real representations of $SO(2)$ are the trivial
representation

$$
\rho_0(r_\theta) := 1
$$

and the two-dimensional representations

$$
\rho_n(r_\theta) := r_{n\theta} = \begin{bmatrix} \cos(n\theta) & -\sin(n\theta) \\ \sin(n\theta) & \cos(n\theta) \end{bmatrix} \qquad n \in \mathbb{N}.
$$

Moreover, every finite-dimensional continuous real representation $\rho : SO(2) \to GL(\mathbb{R}^c)$
is equivalent to a direct sum of these irreducible representations.

**Proof.** The representation $\rho_0$ is one-dimensional and thus is irreducible and not equivalent to
$\rho_n$ for any $n > 0$.

Let $n \geq 1$. We show $\rho_n$ is irreducible and is not equivalent to $\rho_m$ for any $m \neq n$.
Notice that $\rho_n\big(r_{\pi/(2n)}\big) = J$, which has eigenvalues $\pm i$. This means no
$1$-dimensional subspace of $\mathbb{R}^2$ is preserved under the action of $\rho_n$, and so $\rho_n$
is irreducible. Suppose $m \neq n$; because the characters

$$
\mathrm{tr}\ \rho_n(r_\theta) = 2\cos(n\theta) \neq 2\cos(m\theta) = \mathrm{tr}\ \rho_m(r_\theta)
$$

are distinct, the representations $\rho_n$ and $\rho_m$ are not equivalent.

We now prove that any finite-dimensional continuous real representation of $SO(2)$ is equivalent to
a direct sum of the irreducible representations $\rho_n$. Note that this implies
$\{\rho_n \mid n \in \mathbb{Z}_{\geq 0}\}$ is the complete set of irreducible real representations of
$SO(2)$. Let $\rho : SO(2) \to GL(\mathbb{R}^c)$ be a continuous real representation. Without loss of
generality, we may assume the standard inner product $\langle \cdot,\cdot\rangle$ on $\mathbb{R}^c$ is
$\rho$-invariant: if it is not, define $\langle\cdot,\cdot\rangle_\rho$ by

$$
\langle v,w\rangle_\rho = \frac{1}{2\pi}\int_0^{2\pi} \langle \rho(r_\theta)v,\, \rho(r_\theta)w\rangle\, d\theta
$$

which is $\rho$-invariant, then make a change of basis to one that is orthonormal in
$\langle\cdot,\cdot\rangle_\rho$. Under this assumption we have $\rho(r_\theta) \in O(c)$ for all
$\theta \in \mathbb{R}/2\pi\mathbb{Z}$.

Consider the continuous group homomorphism $\widetilde\rho : \mathbb{R} \to O(c)$,
$\widetilde\rho(\theta) := \rho(r_\theta)$. This is a one-parameter subgroup of $O(c)$. By the
correspondence between one-parameter subgroups of a matrix Lie group and elements of its Lie
algebra, there exists a unique matrix $A \in \mathfrak{o}(c)$ such that

$$
\rho(r_\theta) = \widetilde\rho(\theta) = \exp(\theta A) \qquad \forall\, \theta \in \mathbb{R}.
$$

Since $r_{\theta+2\pi} = r_\theta$, this one-parameter subgroup is $2\pi$-periodic and therefore
satisfies $\exp(2\pi A) = I_c$. By the real spectral theorem for skew symmetric matrices, there exists
a $Q \in O(c)$ such that

$$
QAQ^{-1} = 0_{a_0} \oplus \lambda_1 J \oplus \cdots \oplus \lambda_k J
$$

where $\lambda_1,\ldots,\lambda_k > 0$ are the purely imaginary eigenvalues of $A$. The periodicity of
$A$ implies that $\exp(2\pi\lambda_i J) = \cos(2\pi\lambda_i)I + \sin(2\pi\lambda_i)J = I$ for each
$i = 1,\ldots,k$, and thus $\lambda_i \in \mathbb{Z}_{>0}$. We have,

$$
\begin{aligned}
Q\rho(r_\theta)Q^{-1} &= Q\exp(\theta A)Q^{-1} \\
&= \exp\big(\theta\, QAQ^{-1}\big) \\
&= \exp\Big(\theta\big(0_{a_0} \oplus \lambda_1 J \oplus \cdots \oplus \lambda_k J\big)\Big) \\
&= I_{a_0} \oplus \exp(\lambda_1\theta J) \oplus \cdots \oplus \exp(\lambda_k\theta J) \\
&= I_{a_0} \oplus r_{\lambda_1\theta} \oplus \cdots \oplus r_{\lambda_k\theta} \\
&= I_{a_0} \otimes \rho_0(r_\theta) \oplus \rho_{\lambda_1}(r_\theta) \oplus \cdots \oplus \rho_{\lambda_k}(r_\theta)
\end{aligned}
$$

where $\lambda_i \in \mathbb{Z}_{>0}$. This shows $\rho$ is equivalent to a direct sum of the irreducible
representations $\rho_n$, as desired. $\blacksquare$

It follows that there exist finite subsets $\mathcal{I}_{\mathrm{in}}, \mathcal{I}_{\mathrm{out}} \subset \mathbb{Z}_{\geq 0}$,
and invertible matrices $Q_{\mathrm{in}}$ and $Q_{\mathrm{out}}$ such that

$$
\widehat\rho_{\mathrm{in}}(g) := Q_{\mathrm{in}}\rho_{\mathrm{in}}(g)Q_{\mathrm{in}}^{-1} = \bigoplus_{n \in \mathcal{I}_{\mathrm{in}}} \big(I_{a_n} \otimes \rho_n(g)\big), \qquad
\widehat\rho_{\mathrm{out}}(g) := Q_{\mathrm{out}}\rho_{\mathrm{out}}(g)Q_{\mathrm{out}}^{-1} = \bigoplus_{m \in \mathcal{I}_{\mathrm{out}}} \big(I_{b_m} \otimes \rho_m(g)\big).
$$

Here $a_n$ and $b_m$ denote the multiplicities of $\rho_n$ in $\rho_{\mathrm{in}}$ and of $\rho_m$ in
$\rho_{\mathrm{out}}$, respectively, and $I_{a_n} \in \mathbb{R}^{a_n \times a_n}$,
$I_{b_m} \in \mathbb{R}^{b_m \times b_m}$ denote the identity matrices. In particular,

$$
c_{\mathrm{in}} = a_0 + 2\sum_{n\geq 1} a_n, \qquad c_{\mathrm{out}} = b_0 + 2\sum_{m\geq 1} b_m.
$$

The multiplicities $a_n$ and $b_m$ need not agree. Thus the decomposition into irreducible
representations does not require the input and output layers to have the same number of channels.

## Intertwiners between irreducible representations

We now find bases for the vector space of intertwiners $\mathrm{Hom}_{SO(2)}(\rho_n,\rho_m)$
between irreducible representations. Recall that $W \in \mathrm{Hom}_{SO(2)}(\rho_n,\rho_m)$ if

$$
W\rho_n(r_\theta) = \rho_m(r_\theta) W \qquad \forall\, r_\theta \in SO(2). \tag{5.10}
$$

First consider the case $m,n \geq 1$. As done above, identify $\mathbb{R}^2$ with $\mathbb{C}$ as real
vector spaces by $(x,y) \longleftarrow z = x+iy$, so that $\rho_n(r_\theta)z = e^{in\theta}z$. Every real
linear map $W : \mathbb{C} \to \mathbb{C}$ admits a unique decomposition $W(z) = az + b\bar z$ for some
$a,b \in \mathbb{C}$. Applying this decomposition to the left and right hand sides of (5.10) gives

$$
W\big(e^{in\theta}z\big) = a e^{in\theta}z + b e^{-in\theta}\bar z, \qquad e^{im\theta}W(z) = a e^{im\theta}z + b e^{im\theta}\bar z.
$$

For the two expressions to agree for all $\theta$, $a = 0$ unless $n=m$ and $b=0$ unless $n=-m$. Since
$m,n \geq 1$, the latter can never be the case, thus $a,b=0$ when $n \neq m$. If $m=n$, then $a \in \mathbb{C}$
is arbitrary and $b=0$. Parameterizing $a = \alpha 1 + \beta i$ corresponds to the real matrix form
$W = \alpha I + \beta J$. Thus $\mathrm{Hom}_{SO(2)}(\rho_n,\rho_n) = \mathrm{span}_{\mathbb{R}}\{I,J\}$.

If $m=n=0$, both representations are trivial and every scalar is an intertwiner. If exactly one of $n$
or $m$ is zero, the intertwiner must vanish because a nontrivial irreducible representation has no
invariant vector or covector, as noted in the proof of Proposition 5.1.

Having considered all the possible cases, we've arrived at the following:

$$
\mathrm{Hom}_{SO(2)}(\rho_n,\rho_m) = \begin{cases} \mathrm{span}_{\mathbb{R}}\{1\}, & m=n=0, \\ \mathrm{span}_{\mathbb{R}}\{I,J\}, & m=n\geq 1, \\ \{0\}, & m \neq n. \end{cases} \tag{5.11}
$$

## Intertwiners between arbitrary representations

By Proposition 5.1, we may assume that

$$
\rho_{\mathrm{in}} = \bigoplus_{n \in \mathcal{I}_{\mathrm{in}}} I_{a_n}\otimes\rho_n, \qquad \rho_{\mathrm{out}} = \bigoplus_{m \in \mathcal{I}_{\mathrm{out}}} I_{b_m}\otimes\rho_m, \tag{5.12}
$$

where $\mathcal{I}_{\mathrm{in}}, \mathcal{I}_{\mathrm{out}} \subset \mathbb{Z}_{\geq 0}$ are finite index sets
and $a_n$ and $b_m$ are the multiplicities of the types $\rho_n$ and $\rho_m$ in $\rho_{\mathrm{in}}$ and
$\rho_{\mathrm{out}}$, respectively. Let $d_n$ denote the dimension of the representation space of
$\rho_n$, so that $d_0=1$ and $d_n=2$ for $n\geq 1$. An intertwiner
$W \in \mathrm{Hom}_{SO(2)}(\rho_{\mathrm{in}},\rho_{\mathrm{out}})$ is therefore a
$c_{\mathrm{out}}\times c_{\mathrm{in}}$ matrix with
$|\mathcal{I}_{\mathrm{out}}||\mathcal{I}_{\mathrm{in}}|$ blocks

$$
W_{m\leftarrow n} : \mathbb{R}^{a_n}\otimes\mathbb{R}^{d_n} \longrightarrow \mathbb{R}^{b_m}\otimes\mathbb{R}^{d_m}, \qquad n \in \mathcal{I}_{\mathrm{in}},\ m \in \mathcal{I}_{\mathrm{out}}. \tag{5.13}
$$

Relative to the decompositions in (5.12), the intertwiner constraint
$W = \rho_{\mathrm{out}}(g)W\rho_{\mathrm{in}}(g)^{-1}$ is a block-diagonal-conjugation identity: since the
matrices $\rho_{\mathrm{out}}(g)$ and $\rho_{\mathrm{in}}(g)^{-1}$ are block diagonal in this basis, the
identity holds if and only if each block satisfies

$$
W_{m\leftarrow n} = \big(I_{b_m}\otimes\rho_m(g)\big)\, W_{m\leftarrow n}\, \big(I_{a_n}\otimes\rho_n(g)^{-1}\big) \qquad \forall\, g \in SO(2).
$$

Thus, for every $n \in \mathcal{I}_{\mathrm{in}}$ and $m \in \mathcal{I}_{\mathrm{out}}$,

$$
W_{m\leftarrow n} \in \mathrm{Hom}_{SO(2)}\big(I_{a_n}\otimes\rho_n,\, I_{b_m}\otimes\rho_m\big).
$$

The group acts trivially on the multiplicity spaces $\mathbb{R}^{a_n}$ and $\mathbb{R}^{b_m}$, and
therefore

$$
\mathrm{Hom}_{SO(2)}\big(I_{a_n}\otimes\rho_n,\, I_{b_m}\otimes\rho_m\big) \cong \mathrm{Hom}(\mathbb{R}^{a_n},\mathbb{R}^{b_m}) \otimes \mathrm{Hom}_{SO(2)}(\rho_n,\rho_m). \tag{5.14}
$$

Indeed, $W_{m\leftarrow n}$ may itself be viewed as a $b_m \times a_n$ matrix of blocks
$\big(W_{m\leftarrow n}\big)_{\beta\alpha} : \mathbb{R}^{d_n} \to \mathbb{R}^{d_m}$, where each
$\big(W_{m\leftarrow n}\big)_{\beta\alpha}$ must belong to $\mathrm{Hom}_{SO(2)}(\rho_n,\rho_m)$. The
equivariance constraint does not relate the coefficients associated with different pairs $(\beta,\alpha)$
because the group acts identically on every copy of a fixed irrep. This means that the copies of the
same irrep may be mixed by arbitrary linear transformations between their multiplicity spaces.

Combining (5.14) with (5.11) gives the following parameterization of each block:

$$
W_{m\leftarrow n} = \begin{cases} M_0, & m=n=0, \\ M_n\otimes I + N_n\otimes J, & m=n\geq 1, \\ 0, & m\neq n, \end{cases} \tag{5.15}
$$

where $M_0 \in \mathbb{R}^{b_0\times a_0}$, $M_n,N_n \in \mathbb{R}^{b_n\times a_n}$, and $I$ and $J$ are
defined in (5.8). In particular, the $(\beta,\alpha)$ subblock of $W_{n\leftarrow n}$ is

$$
\big(W_{n\leftarrow n}\big)_{\beta\alpha} = (M_n)_{\beta\alpha} I + (N_n)_{\beta\alpha} J.
$$

Thus, to parameterize an arbitrary intertwiner $W \in \mathrm{Hom}_{SO(2)}(\rho_{\mathrm{in}},\rho_{\mathrm{out}})$,
one need only specify $M_0$ when $0 \in \mathcal{I}_{\mathrm{in}} \cap \mathcal{I}_{\mathrm{out}}$,
together with $M_n$ and $N_n$ for every $n \in \mathcal{I}_{\mathrm{in}} \cap \mathcal{I}_{\mathrm{out}}$
with $n \geq 1$. All blocks between inequivalent irrep types vanish. Notice, however, that the
matrices $M_0$, $M_n$, and $N_n$ may be rectangular. The intertwiner constraint therefore requires
pointwise linear maps to preserve irrep type, but it does not require the input and output
multiplicities, or the total numbers of input and output channels, to agree.

## Steerable transformations of irreducible representations

We now solve (5.7) for a value transformation
$V_{m\leftarrow n} : \mathbb{R}^2 \to \mathrm{Hom}(\mathbb{R}^{d_n}, \mathbb{R}^{d_m})$ between irreducible
feature types $\rho_n$ and $\rho_m$. For a nonzero $v \in \mathbb{R}^2$, write $v$ in polar coordinates as
$v = r\, r_\phi\, e_1 = r(\cos\phi,\sin\phi)^\top$, where $r>0$ and $e_1 = (1,0)^\top$. Define the
restriction of the value transformation to the positive reference ray generated by $e_1$ by

$$
A_{m\leftarrow n}(r) := V_{m\leftarrow n}(r e_1).
$$

The group action preserves the radius $r$, and the stabilizer of $r e_1$ is trivial for every $r>0$. It
follows that $A_{m\leftarrow n}(r) \in \mathrm{Hom}(\mathbb{R}^{d_n},\mathbb{R}^{d_m})$ is unconstrained.
Using the steerability of $V_{m\leftarrow n}$, we can express its value at any nonzero $v$ in terms of
$A_{m\leftarrow n}$:

$$
V_{m\leftarrow n}(v) = V_{m\leftarrow n}\big(r_\phi(re_1)\big) = \rho_m(r_\phi)\, V_{m\leftarrow n}(re_1)\, \rho_n(r_\phi)^{-1} = \rho_m(r_\phi)\, A_{m\leftarrow n}(r)\, \rho_n(r_\phi)^{-1}.
$$

Conversely, for any function $A_{m\leftarrow n} : \mathbb{R}_{>0} \to \mathrm{Hom}(\mathbb{R}^{d_n},\mathbb{R}^{d_m})$,
the function $V_{m\leftarrow n} : \mathbb{R}^2\setminus\{0\} \to \mathrm{Hom}(\mathbb{R}^{d_n},\mathbb{R}^{d_m})$
defined by

$$
V_{m\leftarrow n}(r,\phi) := \rho_m(r_\phi)\, A_{m\leftarrow n}(r)\, \rho_n(r_\phi)^{-1} \tag{5.16}
$$

is $SO(2)$-steerable. Indeed, for $g = r_\theta \in SO(2)$,

$$
\begin{aligned}
V_{m\leftarrow n}(gv) &= V_{m\leftarrow n}(r,\phi+\theta) \\
&= \rho_m(r_{\phi+\theta})\, A_{m\leftarrow n}(r)\, \rho_n(r_{\phi+\theta})^{-1} \\
&= \rho_m(r_\theta)\Big(\rho_m(r_\phi)\, A_{m\leftarrow n}(r)\, \rho_n(r_\phi)^{-1}\Big)\rho_n(r_\theta)^{-1} \\
&= \rho_m(r_\theta)\, V_{m\leftarrow n}(r,\phi)\, \rho_n(r_\theta)^{-1}.
\end{aligned}
$$

At the origin, steerability additionally requires $V_{m\leftarrow n}(0) \in \mathrm{Hom}_{SO(2)}(\rho_n,\rho_m)$,
since the stabilizer of $0$ is all of $SO(2)$.

Equation (5.16) provides the means by which we parameterize all $SO(2)$-steerable transformations
from type $\rho_n$ to type $\rho_m$. Namely, one may choose an arbitrary matrix-valued radial function
$A_{m\leftarrow n}$ along the positive reference ray and extend it equivariantly to each circular orbit
according to (5.16). If the value transformation is required to be continuous or smooth on all of
$\mathbb{R}^2$, then the radial function must additionally be chosen so that this equivariant extension
converges to, or extends smoothly through, an intertwiner at the origin.

Accordingly, let $\{A_i\}_{i=1}^{d_md_n}$ be a basis of $\mathrm{Hom}(\mathbb{R}^{d_n},\mathbb{R}^{d_m})$
and define the corresponding angular kernel functions by

$$
K_i^{m\leftarrow n}(\phi) := \rho_m(r_\phi)\, A_i\, \rho_n(r_\phi)^{-1}.
$$

Following the nomenclature of the steerable CNN literature (e.g. [CW17]), we refer to
$\{K_i^{m\leftarrow n}\}_{i=1}^{d_md_n}$ as an angular kernel basis. These functions are linearly
independent since $K_i^{m\leftarrow n}(0) = A_i$, and therefore any linear dependence among the
$K_i^{m\leftarrow n}$ would imply the same linear dependence among the $A_i$.

For each $r>0$, the reference value $A_{m\leftarrow n}(r)$ admits a unique expansion

$$
A_{m\leftarrow n}(r) = \sum_{i=1}^{d_md_n} \gamma_i(r)\, A_i, \tag{5.17}
$$

where $\gamma_i : \mathbb{R}_{>0}\to\mathbb{R}$ are scalar radial functions. Substituting (5.17) into (5.16)
gives

$$
V_{m\leftarrow n}(r,\phi) = \rho_m(r_\phi)\left(\sum_{i=1}^{d_md_n}\gamma_i(r)A_i\right)\rho_n(r_\phi)^{-1} = \sum_{i=1}^{d_md_n}\gamma_i(r)\, \rho_m(r_\phi)A_i\rho_n(r_\phi)^{-1} = \sum_{i=1}^{d_md_n}\gamma_i(r)\, K_i^{m\leftarrow n}(\phi).
$$

Thus every $SO(2)$-steerable transformation from $\rho_n$ to $\rho_m$ is parameterized on
$\mathbb{R}^2\setminus\{0\}$ by $d_md_n$ scalar radial functions multiplying the corresponding angular
kernel basis elements.

We consider the cases for values of $m,n$, and derive angular kernel bases for each:

**($n=0$, $m=0$).** If $m,n=0$, then $A_{0\leftarrow0}$ is any scalar function and
$V_{0\leftarrow0}(r,\phi) = A_{0\leftarrow0}(r)$, so the angular kernel basis is
$K_1^{0\leftarrow0}(\phi) = 1$.

**($n=0, m\geq1$).** In this case, $A_{m\leftarrow0} : \mathbb{R}_{>0}\to\mathrm{Hom}(\mathbb{R},\mathbb{R}^2)\cong\mathbb{R}^2$.
Choose $A_i = e_i$ for $i=1,2$ where $e_i$ denotes the standard basis. Then the angular kernel
basis consists of

$$
K_1^{m\leftarrow0}(\phi) = \rho_m(r_\phi)e_1 = \begin{pmatrix}\cos(m\phi)\\ \sin(m\phi)\end{pmatrix}, \qquad K_2^{m\leftarrow0}(\phi) = \rho_m(r_\phi)e_2 = \begin{pmatrix}-\sin(m\phi)\\ \cos(m\phi)\end{pmatrix}.
$$

**($n\geq1, m=0$).** We have $A_{0\leftarrow n} : \mathbb{R}_{>0}\to\mathrm{Hom}(\mathbb{R}^2,\mathbb{R})\cong\mathbb{R}^{1\times2}$.
Choose $A_i = e_i^\top$ for $i=1,2$ where $e_i^\top$ denotes the standard dual basis. Then the
angular kernel basis consists of

$$
K_1^{0\leftarrow n}(\phi) = e_1^\top\rho_n(r_\phi)^{-1} = \big(\cos(n\phi),\ \sin(n\phi)\big), \qquad K_2^{0\leftarrow n}(\phi) = e_2^\top\rho_n(r_\phi)^{-1} = \big(-\sin(n\phi),\ \cos(n\phi)\big).
$$

**($n\geq1, m\geq1$).** Here $A_{m\leftarrow n} : \mathbb{R}_{>0}\to\mathrm{Hom}(\mathbb{R}^2,\mathbb{R}^2)\cong\mathbb{R}^{2\times2}$.
Choose the basis $\{A_1=I, A_2=J, A_3=S, A_4=JS\}$ for $\mathbb{R}^{2\times2}$. Using
$r_\theta J = Jr_\theta$ and $Sr_\theta = r_{-\theta}S$ we derive the angular kernel basis

$$
\begin{aligned}
K_1^{m\leftarrow n}(\phi) &= \rho_m(r_\phi)\, I\, \rho_n(r_\phi)^{-1} = r_{(m-n)\phi}, \\
K_2^{m\leftarrow n}(\phi) &= \rho_m(r_\phi)\, J\, \rho_n(r_\phi)^{-1} = r_{(m-n)\phi}J, \\
K_3^{m\leftarrow n}(\phi) &= \rho_m(r_\phi)\, S\, \rho_n(r_\phi)^{-1} = r_{(m+n)\phi}S, \\
K_4^{m\leftarrow n}(\phi) &= \rho_m(r_\phi)\, JS\, \rho_n(r_\phi)^{-1} = r_{(m+n)\phi}JS.
\end{aligned}
$$

For brevity, write $c_k(\phi) := \cos(k\phi)$, $s_k(\phi) := \sin(k\phi)$,
$c_-(\phi) := \cos((m-n)\phi)$, $s_-(\phi) := \sin((m-n)\phi)$, $c_+(\phi) := \cos((m+n)\phi)$,
$s_+(\phi) := \sin((m+n)\phi)$.

**Table 5.1 (Angular kernel bases for $SO(2)$-steerable transformations between irreducible feature
types).** The representation $\rho_n$ is the input type and $\rho_m$ is the output type.

| Type map | Angular kernel basis |
|---|---|
| $\rho_0 \to \rho_0$ | $K_1^{0\leftarrow0}(\phi) = 1$ |
| $\rho_0 \to \rho_m,\ m\geq1$ | $K_1^{m\leftarrow0}(\phi) = \begin{pmatrix}c_m(\phi)\\ s_m(\phi)\end{pmatrix}$, $\quad K_2^{m\leftarrow0}(\phi) = \begin{pmatrix}-s_m(\phi)\\ c_m(\phi)\end{pmatrix}$ |
| $\rho_n \to \rho_0,\ n\geq1$ | $K_1^{0\leftarrow n}(\phi) = \begin{pmatrix}c_n(\phi) & s_n(\phi)\end{pmatrix}$, $\quad K_2^{0\leftarrow n}(\phi) = \begin{pmatrix}-s_n(\phi) & c_n(\phi)\end{pmatrix}$ |
| $\rho_n \to \rho_m,\ m,n\geq1$ | $K_1^{m\leftarrow n}(\phi) = \begin{pmatrix}c_-(\phi) & -s_-(\phi)\\ s_-(\phi) & c_-(\phi)\end{pmatrix}$, $\ K_2^{m\leftarrow n}(\phi) = \begin{pmatrix}-s_-(\phi) & -c_-(\phi)\\ c_-(\phi) & -s_-(\phi)\end{pmatrix}$, $\ K_3^{m\leftarrow n}(\phi) = \begin{pmatrix}c_+(\phi) & s_+(\phi)\\ s_+(\phi) & -c_+(\phi)\end{pmatrix}$, $\ K_4^{m\leftarrow n}(\phi) = \begin{pmatrix}-s_+(\phi) & c_+(\phi)\\ c_+(\phi) & s_+(\phi)\end{pmatrix}$ |

We note that the bases presented here may have a different sign convention than those presented in
[dHWCW21, BGPV+22, WC19], but nevertheless define the same kernel space.

## Steerable transformations between arbitrary representations

We now extend the preceding characterization to value transformations between arbitrary
finite-dimensional representations. By Proposition 5.1, we may assume, without loss of generality,
that

$$
\rho_{\mathrm{in}} = \bigoplus_{n\in\mathcal{I}_{\mathrm{in}}} I_{a_n}\otimes\rho_n, \qquad \rho_{\mathrm{out}} = \bigoplus_{m\in\mathcal{I}_{\mathrm{out}}} I_{b_m}\otimes\rho_m,
$$

where $a_n$ and $b_m$ are the multiplicities of the input type $\rho_n$ and output type $\rho_m$,
respectively. Recall that $d_0=1$ and $d_n=2$ for $n\geq1$, so that
$c_{\mathrm{in}} = \sum_{n\in\mathcal{I}_{\mathrm{in}}}a_nd_n$ and
$c_{\mathrm{out}} = \sum_{m\in\mathcal{I}_{\mathrm{out}}}b_md_m$. Let
$W_V : \mathbb{R}^2 \longrightarrow \mathrm{Hom}(\mathbb{R}^{c_{\mathrm{in}}},\mathbb{R}^{c_{\mathrm{out}}})$
be a value transformation. For every $v\in\mathbb{R}^2$, we decompose $W_V(v)$ into blocks

$$
V_{m\leftarrow n}(v) : \mathbb{R}^{a_n}\otimes\mathbb{R}^{d_n} \longrightarrow \mathbb{R}^{b_m}\otimes\mathbb{R}^{d_m},
$$

indexed by $n\in\mathcal{I}_{\mathrm{in}}$ and $m\in\mathcal{I}_{\mathrm{out}}$. As in the intertwiner case,
the steerability constraint $W_V(gv) = \rho_{\mathrm{out}}(g)W_V(v)\rho_{\mathrm{in}}(g)^{-1}$ decouples
across the block-diagonal structure of $\rho_{\mathrm{in}}(g)$ and $\rho_{\mathrm{out}}(g)$, and holds if
and only if every block satisfies

$$
V_{m\leftarrow n}(gv) = \big(I_{b_m}\otimes\rho_m(g)\big)\, V_{m\leftarrow n}(v)\, \big(I_{a_n}\otimes\rho_n(g)^{-1}\big) \tag{5.18}
$$

for every $g\in SO(2)$ and $v\in\mathbb{R}^2$.

To solve (5.18), view $V_{m\leftarrow n}(v)$ as a $b_m\times a_n$ matrix of subblocks
$\big(V_{m\leftarrow n}(v)\big)_{\beta\alpha} : \mathbb{R}^{d_n} \longrightarrow \mathbb{R}^{d_m}$, where
$1\leq\alpha\leq a_n$ and $1\leq\beta\leq b_m$. Because the group acts trivially on the multiplicity
spaces, (5.18) is equivalent to

$$
\big(V_{m\leftarrow n}(gv)\big)_{\beta\alpha} = \rho_m(g)\big(V_{m\leftarrow n}(v)\big)_{\beta\alpha}\rho_n(g)^{-1}
$$

for every pair $(\beta,\alpha)$. Thus, every subblock is an $SO(2)$-steerable transformation from the
irreducible type $\rho_n$ to the irreducible type $\rho_m$ as solved in the preceding section.

For $v\neq0$, write $v=rr_\phi e_1$. Let $\{K_i^{m\leftarrow n}(\phi)\}_{i=1}^{d_md_n}$ be the angular
kernel basis associated with the type map $\rho_n\to\rho_m$, as given in Table 5.1. For each
multiplicity pair $(\beta,\alpha)$, there exist scalar radial functions
$\gamma_{i,\beta\alpha}^{m\leftarrow n} : \mathbb{R}_{>0}\longrightarrow\mathbb{R}$ such that

$$
\big(V_{m\leftarrow n}(r,\phi)\big)_{\beta\alpha} = \sum_{i=1}^{d_md_n} \gamma_{i,\beta\alpha}^{m\leftarrow n}(r)\, K_i^{m\leftarrow n}(\phi).
$$

Collecting the coefficients associated with all multiplicity pairs into the matrix-valued radial
functions $C_i^{m\leftarrow n}(r) := \big[\gamma_{i,\beta\alpha}^{m\leftarrow n}(r)\big]_{\beta,\alpha} \in \mathbb{R}^{b_m\times a_n}$,
we obtain the parameterization

$$
V_{m\leftarrow n}(r,\phi) = \sum_{i=1}^{d_md_n} C_i^{m\leftarrow n}(r)\otimes K_i^{m\leftarrow n}(\phi). \tag{5.19}
$$

It follows that multiplicity copies can be mixed by arbitrary matrix-valued radial functions,
independently for each angular kernel basis element.

Using the explicit angular kernel bases from Table 5.1, (5.19) becomes

$$
V_{m\leftarrow n}(r,\phi) = \begin{cases}
C_1^{0\leftarrow0}(r), & m=n=0, \\[4pt]
C_1^{m\leftarrow0}(r)\otimes\begin{pmatrix}c_m(\phi)\\ s_m(\phi)\end{pmatrix} + C_2^{m\leftarrow0}(r)\otimes\begin{pmatrix}-s_m(\phi)\\ c_m(\phi)\end{pmatrix}, & n=0,\ m\geq1, \\[10pt]
C_1^{0\leftarrow n}(r)\otimes\begin{pmatrix}c_n(\phi) & s_n(\phi)\end{pmatrix} + C_2^{0\leftarrow n}(r)\otimes\begin{pmatrix}-s_n(\phi) & c_n(\phi)\end{pmatrix}, & n\geq1,\ m=0, \\[10pt]
\begin{aligned}
&C_1^{m\leftarrow n}(r)\otimes\begin{pmatrix}c_-(\phi) & -s_-(\phi)\\ s_-(\phi) & c_-(\phi)\end{pmatrix} + C_2^{m\leftarrow n}(r)\otimes\begin{pmatrix}-s_-(\phi) & -c_-(\phi)\\ c_-(\phi) & -s_-(\phi)\end{pmatrix} \\
&+ C_3^{m\leftarrow n}(r)\otimes\begin{pmatrix}c_+(\phi) & s_+(\phi)\\ s_+(\phi) & -c_+(\phi)\end{pmatrix} + C_4^{m\leftarrow n}(r)\otimes\begin{pmatrix}-s_+(\phi) & c_+(\phi)\\ c_+(\phi) & s_+(\phi)\end{pmatrix},
\end{aligned} & m,n\geq1.
\end{cases} \tag{5.20}
$$

In each case, $C_i^{m\leftarrow n}(r)\in\mathbb{R}^{b_m\times a_n}$. The full value transformation is then
obtained by assembling these blocks:

$$
W_V(r,\phi) = \Big[V_{m\leftarrow n}(r,\phi)\Big]_{m\in\mathcal{I}_{\mathrm{out}},\, n\in\mathcal{I}_{\mathrm{in}}}. \tag{5.21}
$$

Unlike a pointwise intertwiner, an $SO(2)$-steerable value transformation may have nonzero blocks
between inequivalent irrep types. The angular dependence of $K_i^{m\leftarrow n}(\phi)$ compensates
for the difference between the input and output transformation laws. Thus, for $r>0$, every pair
$(m,n)\in\mathcal{I}_{\mathrm{out}}\times\mathcal{I}_{\mathrm{in}}$ contributes $d_md_n$ matrix-valued
radial coefficient functions.

At the origin, the point $0$ is fixed by all of $SO(2)$, and therefore steerability requires

$$
W_V(0) = \rho_{\mathrm{out}}(g)W_V(0)\rho_{\mathrm{in}}(g)^{-1} \qquad \forall\, g\in SO(2).
$$

Thus, $W_V(0)\in\mathrm{Hom}_{SO(2)}(\rho_{\mathrm{in}},\rho_{\mathrm{out}})$, and its blocks have the
form of (5.15). This intertwiner associated to the origin is often referred to as the self interaction
component of the steerable kernel [BGPV+22, dHWCW21, KdHC19].

## References

- [WFVW23] Maurice Weiler, Patrick Forré, Erik Verlinde, and Max Welling. *Equivariant and
  Coordinate Independent Convolutional Networks: A Gauge Field Theory of Neural Networks*.
  2023. Online monograph.
- [dHWCW21] Pim de Haan, Maurice Weiler, Taco Cohen, and Max Welling. *Gauge equivariant mesh
  CNNs: Anisotropic convolutions on geometric graphs*. International Conference on Learning
  Representations, 2021.
- [BGPV+22] Sourya Basu, Jose Gallego-Posada, Francesco Viganò, James Rowbottom, and Taco
  Cohen. *Equivariant mesh attention networks*. Transactions on Machine Learning Research, 2022.
- [LW20] Leon Lang and Maurice Weiler. *A Wigner-Eckart theorem for group equivariant
  convolution kernels*. CoRR, abs/2010.10952, 2020.
- [WC19] Maurice Weiler and Gabriele Cesa. *General E(2)-equivariant steerable CNNs*. Advances
  in Neural Information Processing Systems, volume 32, 2019.
- [KdHC19] Berkay Kicanaoglu, Pim de Haan, and Taco S. Cohen. *Gauge equivariant spherical
  CNNs*. OpenReview manuscript, 2019. Submitted to the International Conference on Learning
  Representations.
- [CW17] Taco S. Cohen and Max Welling. *Steerable CNNs*. International Conference on Learning
  Representations, 2017.
