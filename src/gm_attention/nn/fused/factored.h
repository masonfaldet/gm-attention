// The factored evaluation of a steerable kernel, per edge (fused_factored.py).
//
// The kernel's coefficient block on an edge is linear in the radial trunk's
// features, C(e) = sum_t h~_t(e) W_t (h~ = (1, h), the bias on t = 0), so
//
//   C(e) v_j = sum_t h~_t(e) (W_t v_j) = sum_t h~_t(e) Z_t(j),
//
// with Z(j) = (W_t v_j)_t formed once per node (SO2SteerableKernel.node_products,
// packed by fused_factored.pack) and v_j the block's input before any
// per-edge factor. Here one edge e with source j reads h(e) (K - 1 entries;
// the constant feature is implicit) and Z(j) (K x W complex plus K x Wr real
// for the scalar block), forms every block's combination sum_t h~_t Z_t in
// registers, applies the block's per-edge factor -- an angular harmonic
// e^{i k phi} (a negative order is the conjugate of the stored positive one),
// a transport harmonic e^{+-i n gamma}, and a radial power (r / cutoff)^p,
// formed from the edge's r / cutoff by repeated multiplication as the eager
// table is -- and accumulates into the edge's output row in the field's flat
// layout (frequency 0 first, then each frequency's (b_m, 2) real/imaginary
// pairs). Rows n -> 0 take the real part. The four cases of
// SO2SteerableKernel._forward_general and its (0, 0) real block are the
// block descriptors fused_factored.plan writes.
//
// One function serves the CPU driver (factored_cpu.cpp), which reads Z(j)
// from global memory, and the CUDA kernel (factored_cuda.cu), which groups
// the edges by source and stages Z(j) in shared memory once for all of a
// source's edges: per edge only h, the phases, r / cutoff and the output
// row move, which is what makes the path bandwidth-light.
//
// The rows are what a layer sums per target under per-edge weights (the
// attention weights, or a convolution's window times quadrature weight):
// out(i) = sum_{e -> i} w_e row(e). The aggregated form (fused_factored.
// FactoredAggregate) runs that sum as its own kernel over the rows and
// takes the backward from the target's gradient: the edge's row gradient
// is w_e g(i), read through the target index, and dw_e = <g(i), row(e)>
// from the rows the forward kept, so no weighted per-edge tensor, no
// gathered gradient and no per-edge multiply pass exist.
//
// The backward has the same structure as the forward with the roles
// swapped. Read the output gradient of an edge as complex pairs G per
// column and form the packed row W(e) = G conj(F_b) block by block, in the
// layout of a node's mixes (zc's (wc, 2) real pairs, then zr's wr reals; a
// real block's W is the gradient's real slots). Then, with Z(j) in that
// same packed layout,
//
//   dh_t(e) = sum_c W_c(e) Z_{t,c}(j)   for t >= 1   (a (1 x C)(C x K) product per edge)
//   dZ_{t,c}(j) = sum_{e <- j} h~_t(e) W_c(e)         (a (K x C) outer-product sum per source)
//
// since for a complex block d Re[conj(G) F u] = Re[conj(G conj(F)) du] and
// u = sum_t h~_t Z_t: the same routine gives both, and the CUDA kernel
// reduces dZ(j) over a source's edges in shared-memory tiles.
#pragma once
#include <cstdint>

namespace gm_factored {

constexpr int MAX_K = 20;     // trunk width + 1 (the constant feature)
constexpr int MAX_OUT = 128;  // an edge's flat output width
constexpr int MAX_BM = 32;    // an output multiplicity
constexpr int MAX_C = 320;    // a node's packed mixes per t: 2 wc + wr
constexpr int DESC = 8;       // descriptor width
// The routines keep a block's accumulator (edge_message) and an edge's
// packed row W(e) (edge_backward, the CUDA backward's dZ tile) in
// per-thread arrays sized at compile time, so the bound is paid by every
// layer that runs through an instantiation. Two tiers: the narrow one
// serves the attention values and the fine merges (an output multiplicity
// up to NARROW_BM, mixes up to NARROW_C); the wide one, up to MAX_BM and
// MAX_C, serves the coarse merges and readouts whose multiplicities exceed
// it. The drivers pick the tier per call, so a narrow layer's code and
// register footprint are unchanged by the wide tier's existence.
constexpr int NARROW_BM = 16;
constexpr int NARROW_C = 256;

// Descriptor columns (fused_factored.plan).
enum Column : int {
  KIND = 0,       // 0: the real (0, 0) block (Z from zr), 1: a complex block (Z from zc)
  Z_OFFSET = 1,   // first column of the block's mixes in zc (or zr)
  B_M = 2,        // the block's output multiplicity
  ANGULAR = 3,    // angular harmonic order (signed; 0 for none)
  TRANSPORT = 4,  // transport harmonic order (signed; 0 for none)
  POWER = 5,      // the exponent p of the block's radial factor (r / cutoff)^p
  OUT_OFFSET = 6, // first column of the block's output frequency in the flat row
  TAKE_REAL = 7   // 1: the row n >= 1 -> 0, write the real part into the scalar slot
};

template <typename T>
struct Args {
  int64_t n_edges;
  int k;          // K = trunk width + 1: the combination's length, the constant feature first
  int wc;         // complex mixes per node per t
  int wr;         // real mixes per node per t (the (0, 0) block's b_0, or 0)
  int n_f;        // phase table width (max frequency + 1)
  int n_blocks;
  int d_out;      // flat output width
  const T* h;          // (n_edges, k - 1) the trunk's features; h~ = (1, h)
  const T* zc;         // (n_nodes, k, wc, 2)
  const T* zr;         // (n_nodes, k, wr)
  const T* angular;    // (n_edges, n_f, 2)
  const T* transport;  // (n_edges, n_f, 2)
  const T* rho;        // (n_edges,) r / cutoff
  const int32_t* desc; // (n_blocks, DESC)
  const int64_t* source; // (n_edges,) the edge's source node (the CPU driver's lookup)
  T* out;              // (n_edges, d_out)
  // the backward: the output gradient in, the gradients out
  const T* g;             // (n_edges, d_out) by edge, or (n_targets, d_out) by target when `target` is set
  const int64_t* target;  // (n_edges,) the edge's target (the aggregated form), or null
  const T* weights;       // (n_edges,) the weights the rows were summed under (the aggregated form), or null
  const T* rows;          // (n_edges, d_out) the forward's rows, read for dweights, or null
  T* dh;                  // (n_edges, k - 1)
  T* dzc;                 // (n_nodes, k, wc, 2)
  T* dzr;                 // (n_nodes, k, wr)
  T* dweights;            // (n_edges,) or null
};

// The per-edge factor of a complex block: the angular harmonic (a negative
// order the conjugate of the stored positive one), the transport harmonic,
// the radial power (r / cutoff)^p by repeated multiplication, the eager
// table's own arithmetic (SO2SteerableKernel._radial_powers).
template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
inline void block_factor(const Args<T>& a, int64_t e, const int32_t* d, T& fr, T& fi) {
  fr = T(1); fi = T(0);
  const int ang = d[ANGULAR], tr = d[TRANSPORT];
  if (ang != 0) {
    const int o = ang < 0 ? -ang : ang;
    const T* p = a.angular + (e * a.n_f + o) * 2;
    const T pr = p[0], pi = ang < 0 ? -p[1] : p[1];
    const T nr = fr * pr - fi * pi, ni = fr * pi + fi * pr;
    fr = nr; fi = ni;
  }
  if (tr != 0) {
    const int o = tr < 0 ? -tr : tr;
    const T* p = a.transport + (e * a.n_f + o) * 2;
    const T pr = p[0], pi = tr < 0 ? -p[1] : p[1];
    const T nr = fr * pr - fi * pi, ni = fr * pi + fi * pr;
    fr = nr; fi = ni;
  }
  const int power = d[POWER];
  const T r = a.rho[e];
  T s = T(1);
  for (int i = 0; i < power; ++i) s *= r;
  fr *= s; fi *= s;
}

// The edge's h~ = (1, h(e)) into hh (MAX_K entries, zero past k).
template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
inline void load_features(const Args<T>& a, int64_t e, T* hh) {
  const T* he = a.h + e * (a.k - 1);
  hh[0] = T(1);
#pragma unroll
  for (int t = 1; t < MAX_K; ++t) hh[t] = (t < a.k) ? he[t - 1] : T(0);
}

template <typename T, int BM>
#ifdef __CUDACC__
__host__ __device__
#endif
inline void edge_message(const Args<T>& a, int64_t e, const T* zc_row, const T* zr_row) {
  // zc_row: the source's (k, wc, 2) mixes; zr_row: its (k, wr) real mixes.
  // The descriptors are grouped by output frequency (fused_factored.plan
  // sorts them by OUT_OFFSET), so one register accumulator serves a group
  // and is written to the row once when the group ends. BM bounds every
  // block's output multiplicity (the caller's tier).
  static_assert(BM > 0 && BM <= MAX_BM, "BM must be a tier within MAX_BM");
  T hh[MAX_K];
  load_features(a, e, hh);
  T* row = a.out + e * a.d_out;
  T acc[2 * BM];
#pragma unroll
  for (int j = 0; j < 2 * BM; ++j) acc[j] = T(0);
  int group_offset = -1;   // the open group's OUT_OFFSET
  int group_bm = 0;
  bool group_scalar = false;  // the open group is the scalar (frequency 0) slot

  for (int b = 0; b < a.n_blocks; ++b) {
    const int32_t* d = a.desc + b * DESC;
    const int kind = d[KIND], zoff = d[Z_OFFSET], bm = d[B_M];
    const int ooff = d[OUT_OFFSET];
    const bool take_real = d[TAKE_REAL] != 0;
    const bool scalar_out = (kind == 0) || take_real;
    if (ooff != group_offset) {
      if (group_offset >= 0) {  // flush the finished group
        if (group_scalar) {
#pragma unroll
          for (int j = 0; j < BM; ++j) if (j < group_bm) row[group_offset + j] = acc[2 * j];
        } else {
#pragma unroll
          for (int j = 0; j < 2 * BM; ++j) if (j < 2 * group_bm) row[group_offset + j] = acc[j];
        }
      }
#pragma unroll
      for (int j = 0; j < 2 * BM; ++j) acc[j] = T(0);
      group_offset = ooff;
      group_bm = bm;
      group_scalar = scalar_out;
    }
    if (kind == 0) {
#pragma unroll
      for (int j = 0; j < BM; ++j) {
        if (j < bm) {
          T s = T(0);
#pragma unroll
          for (int t = 0; t < MAX_K; ++t) {
            if (t < a.k) s += hh[t] * zr_row[t * a.wr + zoff + j];
          }
          acc[2 * j] += s;
        }
      }
      continue;
    }
    // the block's per-edge factor
    T fr, fi;
    block_factor<T>(a, e, d, fr, fi);
#pragma unroll
    for (int j = 0; j < BM; ++j) {
      if (j < bm) {
        T ur = T(0), ui = T(0);
#pragma unroll
        for (int t = 0; t < MAX_K; ++t) {
          if (t < a.k) {
            const T* z = zc_row + ((t * a.wc) + zoff + j) * 2;
            ur += hh[t] * z[0];
            ui += hh[t] * z[1];
          }
        }
        const T vr = fr * ur - fi * ui;
        const T vi = fr * ui + fi * ur;
        if (take_real) {
          acc[2 * j] += vr;
        } else {
          acc[2 * j] += vr;
          acc[2 * j + 1] += vi;
        }
      }
    }
  }
  if (group_offset >= 0) {  // flush the last group
    if (group_scalar) {
#pragma unroll
      for (int j = 0; j < BM; ++j) if (j < group_bm) row[group_offset + j] = acc[2 * j];
    } else {
#pragma unroll
      for (int j = 0; j < 2 * BM; ++j) if (j < 2 * group_bm) row[group_offset + j] = acc[j];
    }
  }
}

// The output gradient of edge e: its own row, or (the aggregated form) its
// target's row, which the edge's weight scales.
template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
inline const T* grad_row(const Args<T>& a, int64_t e, T& scale) {
  scale = a.weights ? a.weights[e] : T(1);
  return a.target ? a.g + a.target[e] * a.d_out : a.g + e * a.d_out;
}

// The weight's gradient on edge e: <g, row(e)> over the flat row.
template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
inline T edge_dweight(const Args<T>& a, int64_t e, const T* g) {
  const T* row = a.rows + e * a.d_out;
  T s = T(0);
  for (int c = 0; c < a.d_out; ++c) s += g[c] * row[c];
  return s;
}

// The backward's packed row W(e) from the edge's output gradient g under
// `scale`, the blocks first_block, first_block + block_step, ... (every
// column of the row belongs to exactly one block, so callers splitting the
// blocks among threads assemble the whole row between them).
template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
inline void edge_w_row(const Args<T>& a, int64_t e, const T* g, T scale, T* w, int first_block, int block_step) {
  for (int b = first_block; b < a.n_blocks; b += block_step) {
    const int32_t* d = a.desc + b * DESC;
    const int kind = d[KIND], zoff = d[Z_OFFSET], bm = d[B_M], ooff = d[OUT_OFFSET];
    if (kind == 0) {
      for (int j = 0; j < bm; ++j) w[2 * a.wc + zoff + j] = scale * g[ooff + j];
      continue;
    }
    T fr, fi;
    block_factor<T>(a, e, d, fr, fi);
    const bool take_real = d[TAKE_REAL] != 0;
    for (int j = 0; j < bm; ++j) {
      const T gr = scale * (take_real ? g[ooff + j] : g[ooff + 2 * j]);
      const T gi = take_real ? T(0) : scale * g[ooff + 2 * j + 1];
      w[(zoff + j) * 2] = gr * fr + gi * fi;       // W = G conj(F)
      w[(zoff + j) * 2 + 1] = gi * fr - gr * fi;
    }
  }
}

// One edge of the backward on the host: dh(e) written (t >= 1; the constant
// feature has none), dZ(j) accumulated (the CPU driver walks the edges one
// by one, so plain adds suffice), and dweights(e) in the aggregated form.
// CMAX bounds the packed row (the caller's tier).
template <typename T, int CMAX>
inline void edge_backward(const Args<T>& a, int64_t e, const T* zc_row, const T* zr_row, T* dzc_row, T* dzr_row) {
  static_assert(CMAX > 0 && CMAX <= MAX_C, "CMAX must be a tier within MAX_C");
  T w[CMAX];
  const int wc2 = 2 * a.wc;
  T scale;
  const T* g = grad_row(a, e, scale);
  if (a.dweights) a.dweights[e] = edge_dweight(a, e, g);
  edge_w_row<T>(a, e, g, scale, w, 0, 1);
  T hh[MAX_K];
  load_features(a, e, hh);
  T* dh = a.dh + e * (a.k - 1);
  for (int t = 0; t < a.k; ++t) {
    const T* zc_t = zc_row + (int64_t)t * wc2;
    const T* zr_t = zr_row + (int64_t)t * a.wr;
    if (t >= 1) {
      T s = T(0);
      for (int c = 0; c < wc2; ++c) s += w[c] * zc_t[c];
      for (int c = 0; c < a.wr; ++c) s += w[wc2 + c] * zr_t[c];
      dh[t - 1] = s;
    }
    const T ht = hh[t];
    T* dzc_t = dzc_row + (int64_t)t * wc2;
    T* dzr_t = dzr_row + (int64_t)t * a.wr;
    for (int c = 0; c < wc2; ++c) dzc_t[c] += ht * w[c];
    for (int c = 0; c < a.wr; ++c) dzr_t[c] += ht * w[wc2 + c];
  }
}

// The weighted segment sum on the host: out(i) = sum_{e in [row_ptr[i], row_ptr[i + 1])} weights[e] rows[e].
template <typename T>
inline void aggregate_node(const T* rows, const T* weights, const int64_t* row_ptr, int d_out, int64_t i, T* out) {
  const int64_t e0 = row_ptr[i], e1 = row_ptr[i + 1];
  T* o = out + i * d_out;
  for (int c = 0; c < d_out; ++c) o[c] = T(0);
  for (int64_t e = e0; e < e1; ++e) {
    const T w = weights[e];
    const T* row = rows + e * d_out;
    for (int c = 0; c < d_out; ++c) o[c] += w * row[c];
  }
}

}  // namespace gm_factored
