# Geometry of the sphere

The logarithm map, the transport rotation and their coordinate expressions below are the per-edge
computations of [`gm_attention.discretization`](../api/discretization.md). The data they produce is
described in [Discretization](discretization.md#the-discretization-data) and in the paper, Section 4.

Our implementation views $S^2 = \{p \in \mathbb{R}^3 : \|p\| = 1\}$ as an embedded submanifold of
$\mathbb{R}^3$ equipped with the metric induced by the Euclidean inner product. In this setting, all of
the differential-geometric mechanisms required for our implementation of GM-attention can be
explicitly expressed in terms of standard vector operations. In this section we derive formulas for
the Riemannian metric, parallel transport induced by the Levi-Civita connection, the Riemannian
exponential map, and the Riemannian logarithm map.

## Tangent bundle and metric

At each $p \in S^2$, the normal space is $N_pS^2 = \mathrm{span}\{p\}$, and the tangent space is
naturally identified with its orthogonal complement
$T_pS^2 = (N_pS^2)^\perp = \{v \in \mathbb{R}^3 : \langle p, v\rangle = 0\}$. It follows that the tangent
bundle may be realized as the rank-two vector subbundle

$$
TS^2 = \{(p,v) \in S^2 \times \mathbb{R}^3 : \langle p,v \rangle = 0\} \subseteq S^2 \times \mathbb{R}^3
$$

of the trivial bundle $S^2 \times \mathbb{R}^3$. The round Riemannian metric $\eta$ is precisely the
metric induced by the ambient Euclidean inner product, and the Levi-Civita connection is the
tangential connection. Thus, for every $p \in S^2$ and $u,v \in T_pS^2$ we have
$\eta_p(u,v) = \langle u,v \rangle$.

## Exponential map

Since the Levi-Civita connection is the tangential connection on $S^2$, geodesics are curves whose
acceleration vectors lie in $NS^2$. This means geodesics must follow great circles. It follows then
that the Riemannian exponential $\exp_p : T_pS^2 \to S^2$ maps a vector $v \neq 0$ to the point
reached by following the great circle $\mathrm{span}\{p,v\} \cap S^2$ through an arc length of
$\|v\|$. This point is given by $\exp_p(v) = R_{p\times v, \|v\|}\, p$, where $R_{p\times v, \|v\|} \in SO(3)$
denotes rotation through the angle $\|v\|$ about the axis $p \times v$. By Rodrigues' rotation formula,

$$
\begin{aligned}
\exp_p(v) &= R_{p\times v, \|v\|}\, p \\
&= p\cos\|v\| + \left(\frac{p\times v}{\|p\times v\|}\times p\right)\sin\|v\| + \frac{p\times v}{\|p\times v\|}\left\langle \frac{p\times v}{\|p\times v\|}, p\right\rangle(1-\cos\|v\|) \\
&= p\cos\|v\| + \frac{v}{\|v\|}\sin\|v\|.
\end{aligned}
$$

Here we used $\langle p,v\rangle = 0$, $\langle p\times v, p\rangle = 0$, $\|p\times v\| = \|v\|$, and the
vector triple-product identity $(p\times v)\times p = \langle p,p\rangle v - \langle p,v\rangle p = v$. When
$v = 0$, we have $\exp_p(0) = p$. It follows that the injectivity radius of $\exp_p$ is $\pi$ for any
$p \in S^2$.

## Logarithm map

The Riemannian logarithm is obtained by inverting $\exp_p$ restricted to the open ball of radius
$\pi$ about the origin. Thus, for $p \in S^2$ and $q \in S^2 \setminus \{-p\}$, $\log_p(q)$ is the vector
$v \in T_pS^2$ whose magnitude satisfies $\cos\|v\| = \langle p,q\rangle$ and which lies in
$\mathrm{span}\{p,q\}$. It follows that

$$
\begin{aligned}
\log_p(q) &= \arccos\!\big(\langle p,q\rangle\big)\, \frac{(p\times q)\times p}{\|(p\times q)\times p\|} \\
&= \arccos\!\big(\langle p,q\rangle\big)\, \frac{q - \langle p,q\rangle p}{\|q - \langle p,q\rangle p\|},
\end{aligned}
$$

where the second equality holds by the vector triple-product identity.

## Gauge coordinates

A gauge $X$ about $p$ gives a basis for $T_pS^2$ via $\{e^X_{p,i} := (\psi^X_{TS^2,p})^{-1}(e_i)\}_{i=1}^2$,
where $e_i$ denotes the standard basis of $\mathbb{R}^2$. Thus, with respect to the standard bases of
$\mathbb{R}^2$ and $\mathbb{R}^3$, the inverse fiber identification in the gauge $X$ is represented by the
matrix

$$
(\psi^X_{TS^2,p})^{-1}(w) = \begin{bmatrix} e^X_{p,1} & e^X_{p,2} \end{bmatrix} w =: E^X_p\, w \quad \forall w \in \mathbb{R}^2.
$$

Since the columns of the $3\times2$ matrix $E^X_p$ are linearly independent, its Moore–Penrose
pseudoinverse is $(E^X_p)^+ = \big((E^X_p)^\top E^X_p\big)^{-1}(E^X_p)^\top$. Moreover,
$E^X_p (E^X_p)^+ v = v$ for every $v \in \mathrm{span}\{e^X_{p,1}, e^X_{p,2}\} = T_pS^2$. It follows that
$\psi^X_{TS^2,p}$ is represented by the restriction of $(E^X_p)^+$ to $T_pS^2 \subset \mathbb{R}^3$. When
$E^X_p$ is an orthonormal frame — which the frames in the $SO(S^2)$-bundle are, with
$e^X_{p,1}\times e^X_{p,2} = p$ (the paper, Section 4) — then
$(E^X_p)^+ = (E^X_p)^\top$ and

$$
\psi^X_{TS^2,p}(v) = v^X = \begin{bmatrix} \langle e^X_{p,1}, v\rangle \\ \langle e^X_{p,2}, v\rangle \end{bmatrix}.
$$

## Parallel transport

Lastly, we derive the expression for parallel transport with respect to the Levi-Civita connection on
$S^2$. Let $p \in S^2$ and $q \in S^2 \setminus \{-p\}$. We claim that the parallel transport
$P^{TS^2}_{p \leftarrow q} : T_qS^2 \to T_pS^2$ along the minimizing geodesic from $q$ to $p$ is
$R_{n,\theta}\big|_{T_qS^2}$, where $R_{n,\theta}$ denotes the rotation in $SO(3)$ of an angle
$\theta := \arccos(\langle p,q\rangle)$ about the axis $n := \dfrac{q\times p}{\|q\times p\|}$.

If $p = q$, the claim holds trivially. Thus assume $p \neq q$ so that $\theta \in (0,\pi)$. Define
$u := \dfrac{p - q\cos\theta}{\sin\theta}$ so that $\{u,n\}$ is an orthonormal basis of $T_qS^2$ and
$n = q \times u$. Moreover, the parametrization of the minimizing geodesic from $q$ to $p$ is
$\gamma(t) = q\cos t + u\sin t$ for $0 \leq t \leq \theta$.

Let $R_{n,t}$ denote rotation through the angle $t$ about the oriented axis $n$. Since $n\times q = u$
and $\langle n,q\rangle = 0$, Rodrigues' rotation formula gives

$$
R_{n,t}\, q = q\cos t + (n\times q)\sin t = q\cos t + u\sin t = \gamma(t).
$$

In particular, $R_{n,\theta}\,q = p$, so $R_{n,\theta} = (R_{n,\theta})_*$ maps $T_qS^2$ onto $T_pS^2$.

Let $v \in T_qS^2$. Since $\{u,n\}$ is an orthonormal basis of $T_qS^2$, write $v = au + bn$ for some
$a,b \in \mathbb{R}$. Define a vector field along $\gamma$ by

$$
V(t) := R_{n,t}\, v = a R_{n,t}u + b R_{n,t} n = a\dot\gamma(t) + bn,
$$

where the second equality holds since rotation about $n$ fixes $n$ and
$R_{n,t}u = u\cos t + (n\times u)\sin t = u\cos t - q\sin t = \dot\gamma(t)$.

Recall that the Levi-Civita connection $\nabla$ on $S^2 \hookrightarrow \mathbb{R}^3$ is the tangential
connection, i.e. the covariant derivative of $V(t)$ along $\gamma$ is
$\nabla_{\dot\gamma(t)}V(t) = \pi^\top\big(\overline\nabla_{\dot\gamma(t)}V(t)\big)$, where $\overline\nabla$
denotes the Euclidean connection on $\mathbb{R}^3$. Observe,

$$
\ddot\gamma(t) = -q\cos t - u\sin t = -\gamma(t),
$$

which is normal to $S^2$ at $\gamma(t)$, so $\nabla_{\dot\gamma}\dot\gamma = \pi^\top(\ddot\gamma(t)) = 0$.
Moreover, $n$ is constant as an $\mathbb{R}^3$-valued vector and satisfies $\langle n,\gamma(t)\rangle = 0$,
so it is tangent along $\gamma$ and $\nabla_{\dot\gamma}n = \pi^\top(\overline\nabla_{\dot\gamma(t)}n) = 0$.
It follows that $\nabla_{\dot\gamma}V = a\nabla_{\dot\gamma}\dot\gamma + b\nabla_{\dot\gamma}n = 0$. Thus $V$
is the parallel vector field along $\gamma$ with initial value $V(0) = v$. By uniqueness of parallel
transport,

$$
P^{TS^2}_{p\leftarrow q}\, v = V(\theta) = R_{n,\theta}\, v.
$$

Since this holds for every $v \in T_qS^2$, we verified our claim that
$P^{TS^2}_{p\leftarrow q} = R_{n,\theta}\big|_{T_qS^2}$. Using Rodrigues' rotation formula once again
gives

$$
P^{TS^2}_{p\leftarrow q}(v) = R_{n,\theta}\, v = v\cos\theta + (n\times v)\sin\theta + n\langle n,v\rangle(1-\cos\theta).
$$

If $X$ is a gauge about $p$ and $\widetilde X$ is a gauge about $q$, then the coordinate expression of
transport is

$$
g^{X \widetilde X }_{p\leftarrow q} = \psi^X_{TS^2,p} \circ P^{TS^2}_{p\leftarrow q} \circ \big(\psi^{\widetilde X}_{TS^2,q}\big)^{-1} = \big(E^X_p\big)^+ R_{n,\theta}\, E^{\widetilde X}_q.
$$

If these gauges belong to an $SO(2)$-atlas, then $g^{X \widetilde X}_{p\leftarrow q} \in SO(2)$. Moreover,
for any associated feature bundle $A = SO(S^2) \times_\rho \mathbb{R}^c$, the coordinatization of the
induced transport in $A$ is

$$
\psi^X_{A,p} \circ P^{A}_{p\leftarrow q} \circ \big(\psi^{\widetilde X}_{A,q}\big)^{-1} = \rho\big(g^{X \widetilde X }_{p\leftarrow q}\big).
$$

