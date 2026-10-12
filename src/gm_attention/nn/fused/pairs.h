// The fused pair term of a steerable kernel, per edge.
//
//   out(e) += post(e) * f( M(e) * (pre(e) * u~(e)) )
//
// M(e) is the (B x A) block this term's rows of the coefficient readout
// give at edge e -- rows W (width x T) applied to the trunk output h(e)
// (T) plus the bias -- formed here in registers and never written out;
// u~ is the transported input vector u (A) or its conjugate; pre and post
// are per-edge complex phases (angular harmonics) and post also carries a
// real per-edge scale (a radial power); f is the identity or the real
// part. The four cases of SO2SteerableKernel._forward_general are
// instances (see fused_pairs.py, which plans them).
//
// The backward, given g = dL/dout(e), recomputes M and produces
//   grad_u~ (accumulated in place into the input's gradient),
//   grad_h  (accumulated in place into the trunk output's gradient),
//   grad_k  (this term's rows of the coefficient gradient, written into
//            a wide (width, n_edges) buffer whose weight-gradient GEMM
//            runs once for all terms),
// all in registers per edge. PyTorch's complex convention throughout:
// for y = M u, grad_M = g conj(u)^T and grad_u = M^H g; for y = p q,
// grad_q = conj(p) g; for y = Re q, grad_q = g + 0i; for w = conj(u),
// grad_u = conj(grad_w); for a real u promoted to complex, grad_u = Re.
//
// Layout of the code:
//   * A and B are template parameters. The loop over the output side (B)
//     is a plain loop; the loops over the input side (A) and the trunk
//     are unrolled, so every array indexed by them (the input vector and
//     its gradient, the trunk output and its gradient, a weight row)
//     lives in registers -- an array indexed by a runtime value lands in
//     local memory and costs an order of magnitude -- while the body
//     stays A entries long instead of A x B. The
//     input side is the unrolled one because its per-edge vectors are
//     the wider traffic: the input is read and its gradient read and
//     written once per edge as 16-byte vectors, where a runtime loop
//     over it touches a 32-byte sector for every 8-byte element.
//   * The backward reads each weight row once: the coefficient gradient
//     of an entry, g_bi conj(u~_ai), does not depend on the entry, so the
//     trunk gradient accumulates in the same pass that forms the entry.
//   * grad_k is coefficient-major: consecutive edges (threads) write
//     consecutive addresses, one line per warp store.
//   * The trunk width is a compile-time constant in the per-size CUDA
//     builds (GM_PAIRS_K), which makes the trunk loops exact and lets
//     rows load as 16-byte vectors; the CPU build of every size takes it
//     at runtime under MAX_TRUNK, predicated.
//
// One function serves the CPU driver (pairs_cpu.cpp) and the CUDA
// kernels (pairs_cuda.cu): the arithmetic is identical, only the loops
// around it and where W lives (global or shared memory) differ.
#pragma once

#include <c10/util/complex.h>

#include <cstdint>
#include <type_traits>

#ifdef __CUDACC__
#define GM_HD __host__ __device__ __forceinline__
#define GM_UNROLL _Pragma("unroll")
#define GM_NOUNROLL _Pragma("unroll 1")
#else
#define GM_HD inline
#define GM_UNROLL
#define GM_NOUNROLL
#endif
#define GM_RESTRICT __restrict__

// The trunk width this build is compiled for: > 0 fixes it (the per-size
// CUDA builds), 0 takes it from the term at runtime (the CPU build).
#ifndef GM_PAIRS_K
#define GM_PAIRS_K 0
#endif

namespace gm_pairs {

constexpr int MAX_MULT = 16;   // largest B or A a fused term handles
constexpr int MAX_TRUNK = 16;  // largest trunk width
constexpr int K_FIXED = GM_PAIRS_K;
constexpr int KN = K_FIXED > 0 ? K_FIXED : MAX_TRUNK;  // register length of a trunk-sized array
static_assert(K_FIXED >= 0 && K_FIXED <= MAX_TRUNK, "GM_PAIRS_K must be 0 or in 1..MAX_TRUNK");

// flags
constexpr int MATRIX_COMPLEX = 1;  // M complex, rows come in (re, im) pairs
constexpr int U_REAL = 2;          // u is a real tensor (promoted to complex)
constexpr int CONJ_U = 4;          // u~ = conj(u)
constexpr int OUT_REAL = 8;        // out is real, f = Re

template <typename T>
using cx = c10::complex<T>;

template <typename T>
struct Term {
  int n_edges;
  int a, b, trunk;       // block A -> B, trunk width
  int flags;
  const T* hidden;       // (n_edges, trunk)
  const T* weight;       // (width, trunk), width = (MATRIX_COMPLEX ? 2 : 1) * B * A
  const T* bias;         // (width) or nullptr
  const void* u;         // (n_edges, A): cx<T> or T (U_REAL)
  const cx<T>* pre;      // (n_edges) or nullptr
  const cx<T>* post;     // (n_edges) or nullptr
  const T* scale;        // (n_edges) or nullptr
  void* out;             // (n_edges, B): cx<T> or T (OUT_REAL), accumulated
  // backward only
  const void* grad_out;  // (n_edges, B): cx<T> or T (OUT_REAL)
  void* grad_u;          // (n_edges, A): cx<T> or T (U_REAL), accumulated
  T* grad_hidden;        // (n_edges, trunk), accumulated
  T* grad_k;             // (width, n_edges): row r at r * grad_k_stride
  long grad_k_stride;
};

template <typename T>
GM_HD int trunk_of(const Term<T>& t) {
  return K_FIXED > 0 ? K_FIXED : t.trunk;
}

// N contiguous values at p into registers v. On the device, 16-byte loads
// when the type and count allow and p is aligned; scalars otherwise.
template <typename T, int N>
GM_HD void load_n(const T* GM_RESTRICT p, T* v) {
#if defined(__CUDA_ARCH__)
  if constexpr (std::is_same<T, float>::value && N % 4 == 0) {
    if ((reinterpret_cast<uintptr_t>(p) & 15) == 0) {
      const float4* q = reinterpret_cast<const float4*>(p);
      GM_UNROLL
      for (int i = 0; i < N / 4; ++i) {
        const float4 x = q[i];
        v[4 * i] = x.x; v[4 * i + 1] = x.y; v[4 * i + 2] = x.z; v[4 * i + 3] = x.w;
      }
      return;
    }
  } else if constexpr (std::is_same<T, double>::value && N % 2 == 0) {
    if ((reinterpret_cast<uintptr_t>(p) & 15) == 0) {
      const double2* q = reinterpret_cast<const double2*>(p);
      GM_UNROLL
      for (int i = 0; i < N / 2; ++i) {
        const double2 x = q[i];
        v[2 * i] = x.x; v[2 * i + 1] = x.y;
      }
      return;
    }
  }
#endif
  GM_UNROLL
  for (int i = 0; i < N; ++i) v[i] = p[i];
}

template <typename T, int N>
GM_HD void store_n(T* GM_RESTRICT p, const T* v) {
#if defined(__CUDA_ARCH__)
  if constexpr (std::is_same<T, float>::value && N % 4 == 0) {
    if ((reinterpret_cast<uintptr_t>(p) & 15) == 0) {
      float4* q = reinterpret_cast<float4*>(p);
      GM_UNROLL
      for (int i = 0; i < N / 4; ++i) q[i] = make_float4(v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]);
      return;
    }
  } else if constexpr (std::is_same<T, double>::value && N % 2 == 0) {
    if ((reinterpret_cast<uintptr_t>(p) & 15) == 0) {
      double2* q = reinterpret_cast<double2*>(p);
      GM_UNROLL
      for (int i = 0; i < N / 2; ++i) q[i] = make_double2(v[2 * i], v[2 * i + 1]);
      return;
    }
  }
#endif
  GM_UNROLL
  for (int i = 0; i < N; ++i) p[i] = v[i];
}

// A trunk-length row at p into v[KN]; zero beyond the trunk when the
// width is a runtime value, so the dot products below run over KN.
template <typename T>
GM_HD void load_trunk(const Term<T>& t, const T* GM_RESTRICT p, T* v) {
  if constexpr (K_FIXED > 0) {
    load_n<T, K_FIXED>(p, v);
  } else {
    GM_UNROLL
    for (int k = 0; k < MAX_TRUNK; ++k) v[k] = (k < t.trunk) ? p[k] : T(0);
  }
}

// p[k] += v[k] over the trunk.
template <typename T>
GM_HD void add_trunk(const Term<T>& t, T* GM_RESTRICT p, const T* v) {
  if constexpr (K_FIXED > 0) {
    T cur[K_FIXED];
    load_n<T, K_FIXED>(p, cur);
    GM_UNROLL
    for (int k = 0; k < K_FIXED; ++k) cur[k] += v[k];
    store_n<T, K_FIXED>(p, cur);
  } else {
    GM_UNROLL
    for (int k = 0; k < MAX_TRUNK; ++k) {
      if (k < t.trunk) p[k] += v[k];
    }
  }
}

// The block entry M_idx at this edge from h (registers) and W's rows.
template <typename T>
GM_HD cx<T> entry(const Term<T>& t, const T* GM_RESTRICT W, int idx, const T* h) {
  const int trunk = trunk_of(t);
  if (t.flags & MATRIX_COMPLEX) {
    T wr[KN], wi[KN];
    load_trunk(t, W + (long)(2 * idx) * trunk, wr);
    load_trunk(t, W + (long)(2 * idx + 1) * trunk, wi);
    T re = t.bias ? t.bias[2 * idx] : T(0);
    T im = t.bias ? t.bias[2 * idx + 1] : T(0);
    GM_UNROLL
    for (int k = 0; k < KN; ++k) { re += wr[k] * h[k]; im += wi[k] * h[k]; }
    return cx<T>(re, im);
  }
  T w[KN];
  load_trunk(t, W + (long)idx * trunk, w);
  T re = t.bias ? t.bias[idx] : T(0);
  GM_UNROLL
  for (int k = 0; k < KN; ++k) re += w[k] * h[k];
  return cx<T>(re, T(0));
}

// The entry, and in the same pass over its rows the trunk gradient of
// this entry's coefficient gradient gm: gh += W_rows^T gm.
template <typename T>
GM_HD cx<T> entry_and_trunk_grad(const Term<T>& t, const T* GM_RESTRICT W, int idx, const T* h,
                                 cx<T> gm, T* gh) {
  const int trunk = trunk_of(t);
  if (t.flags & MATRIX_COMPLEX) {
    T wr[KN], wi[KN];
    load_trunk(t, W + (long)(2 * idx) * trunk, wr);
    load_trunk(t, W + (long)(2 * idx + 1) * trunk, wi);
    T re = t.bias ? t.bias[2 * idx] : T(0);
    T im = t.bias ? t.bias[2 * idx + 1] : T(0);
    GM_UNROLL
    for (int k = 0; k < KN; ++k) {
      re += wr[k] * h[k];
      im += wi[k] * h[k];
      gh[k] += wr[k] * gm.real() + wi[k] * gm.imag();
    }
    return cx<T>(re, im);
  }
  T w[KN];
  load_trunk(t, W + (long)idx * trunk, w);
  T re = t.bias ? t.bias[idx] : T(0);
  GM_UNROLL
  for (int k = 0; k < KN; ++k) {
    re += w[k] * h[k];
    gh[k] += w[k] * gm.real();
  }
  return cx<T>(re, T(0));
}

// uu = pre * u~ at edge e into registers: the input row as one vector
// load, promoted from real or conjugated as the term says.
template <typename T, int A>
GM_HD void load_input(const Term<T>& t, long e, cx<T>* uu) {
  const cx<T> pre = t.pre ? t.pre[e] : cx<T>(T(1), T(0));
  if (t.flags & U_REAL) {
    T v[A];
    load_n<T, A>(static_cast<const T*>(t.u) + e * A, v);
    GM_UNROLL
    for (int ai = 0; ai < A; ++ai) uu[ai] = pre * v[ai];  // a real is its own conjugate
  } else {
    T v[2 * A];
    load_n<T, 2 * A>(static_cast<const T*>(t.u) + 2 * e * A, v);  // cx<T> is (re, im)
    GM_UNROLL
    for (int ai = 0; ai < A; ++ai) {
      const T im = (t.flags & CONJ_U) ? -v[2 * ai + 1] : v[2 * ai + 1];
      uu[ai] = pre * cx<T>(v[2 * ai], im);
    }
  }
}

// grad_u at edge e += the gradient through uu = pre * u~ of guu = dL/duu,
// one vector read-modify-write of the row.
template <typename T, int A>
GM_HD void add_input_grad(const Term<T>& t, long e, const cx<T>* guu) {
  const cx<T> pc = t.pre ? std::conj(t.pre[e]) : cx<T>(T(1), T(0));
  if (t.flags & U_REAL) {
    T* p = static_cast<T*>(t.grad_u) + e * A;
    T v[A];
    load_n<T, A>(p, v);
    GM_UNROLL
    for (int ai = 0; ai < A; ++ai) v[ai] += (pc * guu[ai]).real();  // Re is Re of the conjugate too
    store_n<T, A>(p, v);
  } else {
    T* p = static_cast<T*>(t.grad_u) + 2 * e * A;
    T v[2 * A];
    load_n<T, 2 * A>(p, v);
    GM_UNROLL
    for (int ai = 0; ai < A; ++ai) {
      cx<T> g = pc * guu[ai];                  // through uu = pre * u~
      if (t.flags & CONJ_U) g = std::conj(g);  // through u~ = conj(u)
      v[2 * ai] += g.real();
      v[2 * ai + 1] += g.imag();
    }
    store_n<T, 2 * A>(p, v);
  }
}

template <typename T, int A, int B>
GM_HD void forward_edge(const Term<T>& t, const T* GM_RESTRICT W, long e) {
  const int trunk = trunk_of(t);
  T h[KN];
  load_trunk(t, t.hidden + e * trunk, h);
  cx<T> uu[A];
  load_input<T, A>(t, e, uu);
  const T s = t.scale ? t.scale[e] : T(1);
  const cx<T> post = (t.post ? t.post[e] : cx<T>(T(1), T(0))) * s;
  GM_NOUNROLL
  for (int bi = 0; bi < B; ++bi) {
    // Four partial sums: one accumulator makes the A entries a chain of
    // dependent multiply-adds (the entries themselves are independent).
    cx<T> part[4] = {cx<T>(T(0), T(0)), cx<T>(T(0), T(0)), cx<T>(T(0), T(0)), cx<T>(T(0), T(0))};
    GM_UNROLL
    for (int ai = 0; ai < A; ++ai) part[ai % 4] += entry(t, W, bi * A + ai, h) * uu[ai];
    const cx<T> q = (part[0] + part[1]) + (part[2] + part[3]);
    if (t.flags & OUT_REAL) {
      static_cast<T*>(t.out)[e * B + bi] += q.real() * s;
    } else {
      static_cast<cx<T>*>(t.out)[e * B + bi] += post * q;
    }
  }
}

template <typename T, int A, int B>
GM_HD void backward_edge(const Term<T>& t, const T* GM_RESTRICT W, long e) {
  const int trunk = trunk_of(t);
  T h[KN];
  load_trunk(t, t.hidden + e * trunk, h);
  const T s = t.scale ? t.scale[e] : T(1);
  const cx<T> post = (t.post ? t.post[e] : cx<T>(T(1), T(0))) * s;
  cx<T> uu[A];
  load_input<T, A>(t, e, uu);
  cx<T> guu[A];  // dL/duu
  GM_UNROLL
  for (int ai = 0; ai < A; ++ai) guu[ai] = cx<T>(T(0), T(0));
  T gh[KN];
  GM_UNROLL
  for (int k = 0; k < KN; ++k) gh[k] = T(0);
  T* GM_RESTRICT gk = t.grad_k + e;  // this edge's column of the coefficient gradient
  const long stride = t.grad_k_stride;
  GM_NOUNROLL
  for (int bi = 0; bi < B; ++bi) {
    // grad_q: dL/dq for q = M uu
    const cx<T> gq = (t.flags & OUT_REAL)
        ? cx<T>(static_cast<const T*>(t.grad_out)[e * B + bi] * s, T(0))
        : std::conj(post) * static_cast<const cx<T>*>(t.grad_out)[e * B + bi];
    GM_UNROLL
    for (int ai = 0; ai < A; ++ai) {
      const int idx = bi * A + ai;
      const cx<T> gm = gq * std::conj(uu[ai]);
      const cx<T> m = entry_and_trunk_grad(t, W, idx, h, gm, gh);
      guu[ai] += std::conj(m) * gq;
      if (t.flags & MATRIX_COMPLEX) {
        gk[(long)(2 * idx) * stride] = gm.real();
        gk[(long)(2 * idx + 1) * stride] = gm.imag();
      } else {
        gk[(long)idx * stride] = gm.real();
      }
    }
  }
  add_input_grad<T, A>(t, e, guu);
  add_trunk(t, t.grad_hidden + e * trunk, gh);
}

// The (A, B) block sizes this build instantiates: the one a per-size
// CUDA build was asked for, or -- the CPU build -- the multiplicities per
// frequency (and per head) of the models in use. Anything else stays on
// the Python loop (fused_pairs.enabled).
#if defined(GM_PAIRS_A) && defined(GM_PAIRS_B)
#define GM_PAIRS_SIZES(X) X(GM_PAIRS_A, GM_PAIRS_B)
#else
#define GM_PAIRS_SIZES(X) \
  X(1, 1) X(1, 2) X(1, 3) X(1, 4) X(1, 8) X(1, 16) \
  X(2, 1) X(2, 2) X(2, 3) X(2, 4) X(2, 8) X(2, 16) \
  X(3, 1) X(3, 2) X(3, 3) X(3, 4) X(3, 8) X(3, 16) \
  X(4, 1) X(4, 2) X(4, 3) X(4, 4) X(4, 8) X(4, 16) \
  X(8, 1) X(8, 2) X(8, 3) X(8, 4) X(8, 8) X(8, 16) \
  X(16, 1) X(16, 2) X(16, 3) X(16, 4) X(16, 8) X(16, 16)
#endif

}  // namespace gm_pairs
