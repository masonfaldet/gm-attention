// The fused scores-and-softmax op and the depthwise mix on a neighbourhood
// graph (fused_attention.py).
//
// The multi-head attention computes, per edge e = (i <- j) and head h,
//
//   s_e,h  = scale * sum_n <q_i,n,h, tau_n(e) k_j,n,h>          (tau_n = e^{i n gamma})
//   lw_e,h = s_e,h + log c(r_e) + log w_e                          (window, quadrature weight)
//   alpha  = softmax over the target's edges, zero where c(r_e) = 0
//   m_e,h  = C_e,h (tau v_j,h)                                    (the depthwise value kernel:
//            frequency 0: R * v; n >= 1: A z + (r/c)^{2n} e^{2 i n phi} B conj(z), z = tau_n v)
//   y_i,h  = sum_e alpha_e,h m_e,h
//
// with per-edge coefficients R (H, E, a_0) and (A, B) packed in C (H, E, W)
// as the radial MLP's readouts produce them (per frequency, (2, a_n, 2)).
// Eager PyTorch materializes every per-edge intermediate (gathered queries,
// keys and values, their transports, scores, three softmax tensors, the
// messages, the weighted messages) and reduces with a segment sum; here one
// thread block per target node stages q_i, scores its edges into shared
// memory, normalizes there, and then one thread per output component walks
// the neighbourhood reading v_j and the coefficients contiguously, so per
// edge only k_j, v_j, the coefficients and the phases move.
//
// The node fields are the flat layout of SO2FeatureField.to_flat on the
// heads' direct sum (frequency 0 first, then each frequency's (a, 2) pairs;
// head-major copies within a frequency), so queries, keys, values and the
// output share one width D. The descriptor rows the Python driver writes give
// each frequency's multiplicity per head, its flat offset and its
// coefficient offset.
//
// Backward, per target node again: with alpha saved, d alpha_e,h =
// <dy_i,h, m_e,h> accumulates in shared memory while the same walk writes
// dR, dA, dB (each edge's own rows) and scatters dv_j (atomics); the softmax
// adjoint ds = scale * alpha (d alpha - sum alpha d alpha) follows per head,
// and a last walk forms dq_i in registers and scatters dk_j (atomics). The
// gradient conventions are the real pairing: for m = A z, dA = dm conj(z)
// and dz = conj(A) dm; for m = w B conj(z), dB = dm conj(w) z and
// dz = w B conj(dm); z = tau v gives dv = conj(tau) dz; s = Re[conj(q) tau k]
// gives dq = tau k and dk = conj(tau) q. The phases, window and quadrature
// weights are the geometry's and get no gradient.
#pragma once
#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define GM_HD __host__ __device__ __forceinline__
#else
#define GM_HD inline
#endif

namespace gm_fused_attention {

constexpr int MAX_H = 8;  // heads
constexpr int MAX_F = 8;  // frequencies (descriptor rows)
constexpr int DESC = 5;   // descriptor width

// Descriptor columns, one row per frequency.
enum Column : int {
  FREQ = 0,  // the frequency n
  MULT = 1,  // copies per head a_n
  OFF = 2,   // first flat column of the frequency's block (all heads)
  COFF = 3,  // first column of the frequency's coefficients in C (0 for n = 0: R)
  DIM = 4    // 1 for n = 0, 2 otherwise
};

template <typename T>
struct Cx {
  T re, im;
};
template <typename T> GM_HD Cx<T> cmul(Cx<T> a, Cx<T> b) { return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re}; }
template <typename T> GM_HD Cx<T> cconj(Cx<T> a) { return {a.re, -a.im}; }
template <typename T> GM_HD Cx<T> cadd(Cx<T> a, Cx<T> b) { return {a.re + b.re, a.im + b.im}; }
template <typename T> GM_HD Cx<T> cscale(T s, Cx<T> a) { return {s * a.re, s * a.im}; }
template <typename T> GM_HD Cx<T> load2(const T* p) { return {p[0], p[1]}; }
template <typename T> GM_HD T part_of(Cx<T> a, int part) { return part == 0 ? a.re : a.im; }

template <typename T>
struct Args {
  int64_t n_nodes, n_edges;
  int H, D, W, A0, F, orders;  // heads; flat width; C's width per (h, e); a_0 per head; descriptor rows; phase table width
  T scale;
  const int64_t* row_ptr;
  const int64_t* source;
  const int64_t* desc;  // (F, DESC)
  const T* q;           // (n_nodes, D)
  const T* k;
  const T* v;
  const T* rc;          // (H, n_edges, A0)
  const T* cc;          // (H, n_edges, W)
  const T* transport;   // (n_edges, orders, 2)
  const T* angular;     // (n_edges, orders, 2)
  const T* rho;         // (n_edges,) r / cutoff
  const T* bias;        // (n_edges,) log window + log quadrature weight
  const uint8_t* inside;  // (n_edges,) window > 0
  T* out;               // (n_nodes, D)
  T* alpha;             // (n_edges, H)
  // backward
  const T* g;           // (n_nodes, D) the output gradient
  const T* dalpha;      // (n_edges, H) the attention weights' gradient (the softmax-only op)
  T* dq;
  T* dk;
  T* dv;
  T* drc;
  T* dcc;
};

// Where a flat column sits: its frequency row, head, copy and real/imaginary part.
struct Slot {
  int n, a_n, h, a, part, coff, c;
};

template <typename T>
GM_HD bool locate(const Args<T>& a, int c, Slot& s) {
  for (int f = 0; f < a.F; ++f) {
    const int64_t* row = a.desc + f * DESC;
    const int n = (int)row[FREQ], a_n = (int)row[MULT], off = (int)row[OFF], dim = (int)row[DIM];
    const int width = a.H * a_n * dim;
    if (c >= off && c < off + width) {
      const int local = c - off, copy = local / dim;
      s.n = n;
      s.a_n = a_n;
      s.h = copy / a_n;
      s.a = copy % a_n;
      s.part = local % dim;
      s.coff = (int)row[COFF];
      s.c = c;
      return true;
    }
  }
  return false;
}

template <typename T>
GM_HD Cx<T> phase(const T* table, int orders, int64_t e, int order) {
  return load2(table + (e * orders + order) * 2);
}

template <typename T>
GM_HD T radial_power(T rho, int n) {  // (r / c)^{2n} by repeated multiplication, as the eager kernel
  T s = T(1);
  for (int i = 0; i < 2 * n; ++i) s *= rho;
  return s;
}

// s_e,h before the scale: sum over the head's copies of <q_i, tau k_j>.
template <typename T>
GM_HD T edge_score(const Args<T>& a, int64_t e, int64_t j, int h, const T* q_i) {
  T acc = T(0);
  const T* k_j = a.k + j * a.D;
  for (int f = 0; f < a.F; ++f) {
    const int64_t* row = a.desc + f * DESC;
    const int n = (int)row[FREQ], a_n = (int)row[MULT], off = (int)row[OFF];
    if (n == 0) {
      const int base = off + h * a_n;
      for (int i = 0; i < a_n; ++i) acc += q_i[base + i] * k_j[base + i];
    } else {
      const Cx<T> tau = phase(a.transport, a.orders, e, n);
      const int base = off + 2 * h * a_n;
      for (int i = 0; i < a_n; ++i) {
        const Cx<T> tk = cmul(tau, load2(k_j + base + 2 * i));
        acc += q_i[base + 2 * i] * tk.re + q_i[base + 2 * i + 1] * tk.im;
      }
    }
  }
  return acc;
}

// A complex slot's terms on an edge: the transport, z = tau v_j, the
// coefficients A and B and the antilinear factor w = (r/c)^{2n} e^{2 i n phi}.
template <typename T>
struct ComplexTerms {
  Cx<T> tau, z, A, B, w;
};

template <typename T>
GM_HD ComplexTerms<T> complex_terms(const Args<T>& a, int64_t e, int64_t j, const Slot& s) {
  ComplexTerms<T> t;
  t.tau = phase(a.transport, a.orders, e, s.n);
  t.z = cmul(t.tau, load2(a.v + j * a.D + s.c - s.part));
  const T* c = a.cc + ((int64_t)s.h * a.n_edges + e) * a.W + s.coff;
  t.A = load2(c + 2 * s.a);
  t.B = load2(c + 2 * s.a_n + 2 * s.a);
  t.w = cscale(radial_power(a.rho[e], s.n), phase(a.angular, a.orders, e, 2 * s.n));
  return t;
}

template <typename T>
GM_HD Cx<T> complex_message(const ComplexTerms<T>& t) {
  return cadd(cmul(t.A, t.z), cmul(t.w, cmul(t.B, cconj(t.z))));
}

template <typename T>
GM_HD int64_t real_index(const Args<T>& a, int64_t e, const Slot& s) {
  return ((int64_t)s.h * a.n_edges + e) * a.A0 + s.a;
}

template <typename T>
GM_HD int64_t complex_index(const Args<T>& a, int64_t e, const Slot& s) {  // A's real part; B at + 2 a_n
  return ((int64_t)s.h * a.n_edges + e) * a.W + s.coff + 2 * s.a;
}

// The component s.c of the message of edge e (source j).
template <typename T>
GM_HD T message(const Args<T>& a, int64_t e, int64_t j, const Slot& s) {
  if (s.n == 0) return a.rc[real_index(a, e, s)] * a.v[j * a.D + s.c];
  return part_of(complex_message(complex_terms(a, e, j, s)), s.part);
}

// The per-edge part of the backward for component s.c: returns the message
// component (for d alpha), writes the edge's own coefficient gradients and
// adds the value gradient through `add` (a plain or an atomic add).
template <typename T, typename Add>
GM_HD T backward_component(const Args<T>& a, int64_t e, int64_t j, const Slot& s, T alpha, const T* dy, Add add) {
  if (s.n == 0) {
    const T r = a.rc[real_index(a, e, s)];
    const T vv = a.v[j * a.D + s.c];
    const T dm = alpha * dy[s.c];
    a.drc[real_index(a, e, s)] = dm * vv;
    add(a.dv + j * a.D + s.c, dm * r);
    return r * vv;
  }
  const ComplexTerms<T> t = complex_terms(a, e, j, s);
  const Cx<T> m = complex_message(t);
  const Cx<T> dm = cscale(alpha, load2(dy + s.c - s.part));
  const Cx<T> dA = cmul(dm, cconj(t.z));
  const Cx<T> dB = cmul(cmul(dm, cconj(t.w)), t.z);
  const int64_t ci = complex_index(a, e, s);
  a.dcc[ci + s.part] = part_of(dA, s.part);
  a.dcc[ci + 2 * s.a_n + s.part] = part_of(dB, s.part);
  const Cx<T> dz = cadd(cmul(cconj(t.A), dm), cmul(t.w, cmul(t.B, cconj(dm))));
  add(a.dv + j * a.D + s.c, part_of(cmul(cconj(t.tau), dz), s.part));
  return part_of(m, s.part);
}

// The score's backward for component s.c on edge e: returns the query
// gradient's summand, adds the key gradient's through `add`.
template <typename T, typename Add>
GM_HD T score_backward_component(const Args<T>& a, int64_t e, int64_t j, const Slot& s, T ds, const T* q_i, Add add) {
  if (s.n == 0) {
    add(a.dk + j * a.D + s.c, ds * q_i[s.c]);
    return ds * a.k[j * a.D + s.c];
  }
  const Cx<T> tau = phase(a.transport, a.orders, e, s.n);
  const Cx<T> tk = cmul(tau, load2(a.k + j * a.D + s.c - s.part));
  const Cx<T> tq = cmul(cconj(tau), load2(q_i + s.c - s.part));
  add(a.dk + j * a.D + s.c, ds * part_of(tq, s.part));
  return ds * part_of(tk, s.part);
}

template <typename T>
GM_HD T lowest() {  // -inf without std::numeric_limits, which device code cannot call
  return -(T)INFINITY;
}

// ---- The CPU driver: one node at a time, in plain loops. ----

template <typename T>
struct PlainAdd {
  void operator()(T* p, T value) const { *p += value; }
};

// ---- The depthwise convolution: the same message walk with the weights
// given (window x quadrature weight, normalized or not), no softmax, one
// head; a.alpha holds the (n_edges,) weights and gets no gradient. ----

template <typename T>
void node_conv_forward(const Args<T>& a, int64_t i) {
  const int64_t e0 = a.row_ptr[i], e1 = a.row_ptr[i + 1];
  for (int c = 0; c < a.D; ++c) {
    Slot s;
    locate(a, c, s);
    T y = T(0);
    for (int64_t e = e0; e < e1; ++e) y += a.alpha[e] * message(a, e, a.source[e], s);
    a.out[i * a.D + c] = y;
  }
}

template <typename T>
void node_conv_backward(const Args<T>& a, int64_t i) {
  const int64_t e0 = a.row_ptr[i], e1 = a.row_ptr[i + 1];
  const T* dy = a.g + i * a.D;
  PlainAdd<T> add;
  for (int c = 0; c < a.D; ++c) {
    Slot s;
    locate(a, c, s);
    for (int64_t e = e0; e < e1; ++e) backward_component(a, e, a.source[e], s, a.alpha[e], dy, add);
  }
}

// ---- The scores and softmax alone (fused_attention.softmax): the heads'
// attention weights alpha (n_edges, H) from q and k, for a layer whose
// value messages come from elsewhere (steerable values through the factored
// kernel). The backward takes d alpha and returns dq, dk. ----

// scratch: at least deg * H entries.
template <typename T>
void node_softmax_forward(const Args<T>& a, int64_t i, T* scratch) {
  const int64_t e0 = a.row_ptr[i], e1 = a.row_ptr[i + 1];
  const int deg = (int)(e1 - e0), H = a.H;
  const T* q_i = a.q + i * a.D;
  for (int t = 0; t < deg; ++t) {
    const int64_t e = e0 + t, j = a.source[e];
    for (int h = 0; h < H; ++h) scratch[t * H + h] = a.scale * edge_score(a, e, j, h, q_i) + a.bias[e];
  }
  for (int h = 0; h < H; ++h) {
    T m = lowest<T>();
    for (int t = 0; t < deg; ++t)  // in-window edges only: a masked edge's bias is merely very negative
      if (a.inside[e0 + t] && scratch[t * H + h] > m) m = scratch[t * H + h];
    T sum = T(0);
    for (int t = 0; t < deg; ++t) {
      const T u = a.inside[e0 + t] ? std::exp(scratch[t * H + h] - m) : T(0);
      scratch[t * H + h] = u;
      sum += u;
    }
    for (int t = 0; t < deg; ++t) a.alpha[(e0 + t) * H + h] = scratch[t * H + h] / sum;
  }
}

// scratch: at least 2 * deg * H entries. dk must be zero on entry (it
// accumulates across nodes); dq's row is written.
template <typename T>
void node_softmax_backward(const Args<T>& a, int64_t i, T* scratch) {
  const int64_t e0 = a.row_ptr[i], e1 = a.row_ptr[i + 1];
  const int deg = (int)(e1 - e0), H = a.H;
  T* al = scratch;
  T* da = scratch + (int64_t)deg * H;
  const T* q_i = a.q + i * a.D;
  for (int t = 0; t < deg; ++t)
    for (int h = 0; h < H; ++h) {
      al[t * H + h] = a.alpha[(e0 + t) * H + h];
      da[t * H + h] = a.dalpha[(e0 + t) * H + h];
    }
  for (int h = 0; h < H; ++h) {
    T total = T(0);
    for (int t = 0; t < deg; ++t) total += al[t * H + h] * da[t * H + h];
    for (int t = 0; t < deg; ++t) da[t * H + h] = a.scale * al[t * H + h] * (da[t * H + h] - total);
  }
  PlainAdd<T> add;
  for (int c = 0; c < a.D; ++c) {
    Slot s;
    locate(a, c, s);
    T dq = T(0);
    for (int t = 0; t < deg; ++t) {
      const int64_t e = e0 + t;
      dq += score_backward_component(a, e, a.source[e], s, da[t * H + s.h], q_i, add);
    }
    a.dq[i * a.D + c] = dq;
  }
}

}  // namespace gm_fused_attention
