// The factored steerable kernel on CUDA: one thread block per source node,
// its mixes Z(j) staged in shared memory once, one thread per edge of that
// source (fused_factored.py groups the edges by source; factored.h has the
// per-edge arithmetic, shared with the CPU driver).
//
// The backward keeps the grouping. Per source, Z(j) is staged packed and
// transposed as (C, K) with C = 2 wc + wr, and the source's edges go through
// in tiles of TILE: the tile's W rows (factored.h edge_w_row, the blocks
// split among THREADS / TILE threads per edge) and h~ rows land in shared
// memory, then dh for the tile's edges is W . Z^T and dZ(j) accumulates
// h~^T W over the tile in registers, written once when the source's edges
// are done. No atomics: every source is exactly one block, every edge
// exactly one thread's row. In the aggregated form the edge's gradient row
// is its target's, read through the target index and scaled by the edge's
// weight, and one thread per edge forms dweights.
//
// Both products are register-tiled over padded rows (backward_layout): K
// padded to a multiple of four (KP), C to a multiple of eight (CP), and the
// W rows to a stride WS = 4 (mod 32) so that the lanes of a warp, one edge
// each, read their rows' 16-byte chunks without bank conflicts. In dh a
// thread owns one edge and four consecutive t's: per four columns it loads
// its row's chunk once and the transposed Z's four chunks (broadcast, the
// same for every lane), sixteen multiply-adds for five loads. In dZ a
// thread owns a 4 x 8 block of (t, c): per edge of the tile it loads the
// h~ chunk (broadcast within a t-block) and two chunks of the W row,
// thirty-two multiply-adds for three loads. The summation order of every
// entry is the plain loops' (columns in order, the tile's edges in order),
// so the values are theirs to the rounding of the compiler's contractions.
//
// The weighted segment sum (aggregate_kernel) is one block per target: its
// warps split the target's edges, each lane a column of the rows (a warp
// reads a row's contiguous columns), and the warps' partial sums meet in
// shared memory.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <mutex>
#include <unordered_map>

#include "factored.h"

namespace gm_factored {

constexpr int THREADS = 128;
constexpr int NWARPS = THREADS / 32;

// Opt `kernel` into `smem` bytes of dynamic shared memory when that exceeds
// the default 48 KiB, once per device and size: the attribute persists, so
// a driver call per launch is wasted, and it is not one a stream capture
// (a CUDA graph) admits.
template <typename Kernel>
static void request_shared_memory(Kernel kernel, size_t smem) {
  if (smem <= 48 * 1024) return;
  static std::mutex lock;
  static std::unordered_map<const void*, std::unordered_map<int, size_t>> granted;
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  const void* key = reinterpret_cast<const void*>(kernel);
  std::lock_guard<std::mutex> guard(lock);
  auto& sizes = granted[key];
  auto it = sizes.find(device);
  if (it != sizes.end() && it->second >= smem) return;
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  sizes[device] = smem;
}

template <typename T, int BM>
__global__ void __launch_bounds__(THREADS) factored_kernel(
    Args<T> a, const int64_t* __restrict__ order, const int64_t* __restrict__ group_ptr,
    const int64_t* __restrict__ group_source) {
  extern __shared__ __align__(16) unsigned char smem[];
  T* zc_s = reinterpret_cast<T*>(smem);
  const int n_zc = a.k * a.wc * 2;
  const int n_zr = a.k * a.wr;
  T* zr_s = zc_s + n_zc;
  const int g = blockIdx.x;
  const int64_t j = group_source[g];
  const T* zc_g = a.zc + j * (int64_t)n_zc;
  const T* zr_g = a.zr + j * (int64_t)n_zr;
  for (int i = threadIdx.x; i < n_zc; i += blockDim.x) zc_s[i] = zc_g[i];
  for (int i = threadIdx.x; i < n_zr; i += blockDim.x) zr_s[i] = zr_g[i];
  __syncthreads();
  const int64_t begin = group_ptr[g], end = group_ptr[g + 1];
  for (int64_t p = begin + threadIdx.x; p < end; p += blockDim.x) {
    edge_message<T, BM>(a, order[p], zc_s, zr_s);
  }
}

// The backward's padded shared-memory layout (see the file comment).
struct BackwardLayout {
  int KP;  // K rounded up to a multiple of 4: the h~ and transposed-Z row length
  int CP;  // C rounded up to a multiple of 8: the columns the dZ blocks cover
  int WS;  // the W rows' stride: >= CP, a multiple of 4, and 4 (mod 32)
};

__host__ __device__ inline int round_up(int x, int m) { return (x + m - 1) / m * m; }

__host__ __device__ inline BackwardLayout backward_layout(int K, int C) {
  BackwardLayout l;
  l.KP = round_up(K, 4);
  l.CP = round_up(C, 8);
  l.WS = l.CP + (((4 - l.CP % 32) % 32) + 32) % 32;
  return l;
}

__host__ __device__ inline int backward_pairs(const BackwardLayout& l) { return (l.KP / 4) * (l.CP / 8); }

template <typename T>
struct Vec4;
template <>
struct Vec4<float> {
  using type = float4;
};
template <>
struct Vec4<double> {
  using type = double4;
};

// Four consecutive entries of a 16-byte-aligned address, one load.
template <typename T>
__device__ __forceinline__ void load4(const T* p, T* out) {
  const typename Vec4<T>::type v = *reinterpret_cast<const typename Vec4<T>::type*>(p);
  out[0] = v.x;
  out[1] = v.y;
  out[2] = v.z;
  out[3] = v.w;
}

// NP: the (t-block, c-block) pairs a thread owns in dZ (one or two: at most
// (MAX_K / 4) x (MAX_C / 8) = 200 pairs over THREADS threads).
template <typename T, int TILE, int NP>
__global__ void __launch_bounds__(THREADS) factored_backward_kernel(
    Args<T> a, const int64_t* __restrict__ order, const int64_t* __restrict__ group_ptr,
    const int64_t* __restrict__ group_source) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int K = a.k, wc2 = 2 * a.wc, C = wc2 + a.wr;
  const BackwardLayout L = backward_layout(K, C);
  const int KP = L.KP, CP = L.CP, WS = L.WS;
  T* z_s = reinterpret_cast<T*>(smem);  // (CP, KP) the source's mixes, transposed: z_s[c * KP + t], zero past C and K
  T* w_s = z_s + CP * KP;               // (TILE, WS) the tile's W rows, zero past C
  T* h_s = w_s + TILE * WS;             // (TILE, KP) the tile's h~ rows, zero past K
  const int g = blockIdx.x;
  const int64_t j = group_source[g];
  const T* zc_g = a.zc + j * (int64_t)K * wc2;
  const T* zr_g = a.zr + j * (int64_t)K * a.wr;
  for (int i = threadIdx.x; i < CP * KP; i += blockDim.x) {
    const int c = i / KP, t = i - c * KP;
    T v = T(0);
    if (c < C && t < K) v = c < wc2 ? zc_g[t * wc2 + c] : zr_g[t * a.wr + (c - wc2)];
    z_s[i] = v;
  }
  for (int i = threadIdx.x; i < TILE * WS; i += blockDim.x) w_s[i] = T(0);  // the rows are rewritten per tile within C
  for (int i = threadIdx.x; i < TILE * KP; i += blockDim.x) h_s[i] = T(0);  // and within K
  constexpr int PARTS = THREADS / TILE;
  const int i = threadIdx.x % TILE, q = threadIdx.x / TILE;
  const int n_tt = KP / 4, n_ct = CP / 8, n_pairs = n_tt * n_ct;
  T acc[NP][4][8];
#pragma unroll
  for (int m = 0; m < NP; ++m)
#pragma unroll
    for (int u = 0; u < 4; ++u)
#pragma unroll
      for (int v = 0; v < 8; ++v) acc[m][u][v] = T(0);
  const int64_t begin = group_ptr[g], end = group_ptr[g + 1];
  for (int64_t tile = begin; tile < end; tile += TILE) {
    const int n_tile = (end - tile) < TILE ? (int)(end - tile) : TILE;
    __syncthreads();  // the previous tile's readers are done with w_s and h_s (and Z is staged)
    if (i < n_tile) {
      const int64_t e = order[tile + i];
      T scale;
      const T* ge = grad_row(a, e, scale);
      edge_w_row<T>(a, e, ge, scale, w_s + i * WS, q, PARTS);
      if (q == 0) {
        const T* he = a.h + e * (K - 1);
        h_s[i * KP] = T(1);
        for (int t = 1; t < K; ++t) h_s[i * KP + t] = he[t - 1];
      }
      if (q == PARTS - 1 && a.dweights) a.dweights[e] = edge_dweight(a, e, ge);
    }
    __syncthreads();
    if (i < n_tile) {  // dh for this edge, four t's at a time (t = 0, the constant feature, has none)
      const int64_t e = order[tile + i];
      const T* w = w_s + i * WS;
      for (int tq = q; tq < n_tt; tq += PARTS) {
        const int t0 = 4 * tq;
        T s[4] = {T(0), T(0), T(0), T(0)};
        for (int c = 0; c < CP; c += 4) {
          T w4[4];
          load4(w + c, w4);
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            T z4[4];
            load4(z_s + (c + k) * KP + t0, z4);
#pragma unroll
            for (int u = 0; u < 4; ++u) s[u] += w4[k] * z4[u];
          }
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const int t = t0 + u;
          if (t >= 1 && t < K) a.dh[e * (K - 1) + (t - 1)] = s[u];
        }
      }
    }
#pragma unroll
    for (int m = 0; m < NP; ++m) {  // dZ(j) over the tile: this thread's 4 x 8 blocks
      const int p = threadIdx.x + THREADS * m;
      if (p < n_pairs) {
        const int ct = p % n_ct, tt = p / n_ct;
        const int t0 = 4 * tt, c0 = 8 * ct;
        for (int r = 0; r < n_tile; ++r) {
          T h4[4], w8[8];
          load4(h_s + r * KP + t0, h4);
          load4(w_s + r * WS + c0, w8);
          load4(w_s + r * WS + c0 + 4, w8 + 4);
#pragma unroll
          for (int u = 0; u < 4; ++u)
#pragma unroll
            for (int v = 0; v < 8; ++v) acc[m][u][v] += h4[u] * w8[v];
        }
      }
    }
  }
  T* dzc_g = a.dzc + j * (int64_t)K * wc2;
  T* dzr_g = a.dzr + j * (int64_t)K * a.wr;
#pragma unroll
  for (int m = 0; m < NP; ++m) {
    const int p = threadIdx.x + THREADS * m;
    if (p < n_pairs) {
      const int ct = p % n_ct, tt = p / n_ct;
      const int t0 = 4 * tt, c0 = 8 * ct;
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int t = t0 + u;
        if (t >= K) continue;
#pragma unroll
        for (int v = 0; v < 8; ++v) {
          const int c = c0 + v;
          if (c >= C) continue;
          if (c < wc2) dzc_g[t * wc2 + c] = acc[m][u][v];
          else dzr_g[t * a.wr + (c - wc2)] = acc[m][u][v];
        }
      }
    }
  }
}

// out(i) = sum over the target's edges of weights[e] rows[e]: the warps split
// the edges, the lanes the columns, and the partial sums meet in shared memory.
template <typename T>
__global__ void __launch_bounds__(THREADS) aggregate_kernel(const T* __restrict__ rows, const T* __restrict__ weights,
                                                            const int64_t* __restrict__ row_ptr, int d_out,
                                                            T* __restrict__ out) {
  __shared__ T partial[NWARPS][MAX_OUT];
  const int64_t i = blockIdx.x;
  const int64_t e0 = row_ptr[i], e1 = row_ptr[i + 1];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int c = lane; c < d_out; c += 32) {
    T s = T(0);
    for (int64_t e = e0 + warp; e < e1; e += NWARPS) s += weights[e] * rows[e * d_out + c];
    partial[warp][c] = s;
  }
  __syncthreads();
  for (int c = threadIdx.x; c < d_out; c += THREADS) {
    T s = T(0);
    for (int w = 0; w < NWARPS; ++w) s += partial[w][c];
    out[i * d_out + c] = s;
  }
}

static void check_groups(const torch::Tensor& h, const torch::Tensor& order, const torch::Tensor& group_ptr, const torch::Tensor& group_source) {
  TORCH_CHECK(order.is_cuda() && group_ptr.is_cuda() && group_source.is_cuda(), "gm_factored: the groups must be CUDA tensors");
  TORCH_CHECK(order.scalar_type() == torch::kInt64 && group_ptr.scalar_type() == torch::kInt64 && group_source.scalar_type() == torch::kInt64,
              "gm_factored: the groups must be int64");
  TORCH_CHECK(order.is_contiguous() && group_ptr.is_contiguous() && group_source.is_contiguous(), "gm_factored: the groups must be contiguous");
  TORCH_CHECK(order.size(0) == h.size(0), "gm_factored: order must list every edge");
  TORCH_CHECK(group_ptr.size(0) == group_source.size(0) + 1, "gm_factored: group_ptr must have one more entry than group_source");
}

template <typename T>
static Args<T> device_args(const torch::Tensor& h, const torch::Tensor& zc, const torch::Tensor& zr,
                           const torch::Tensor& angular, const torch::Tensor& transport, const torch::Tensor& rho,
                           const torch::Tensor& desc, const torch::Tensor& source, int d_out) {
  // The driver's args carry the same pointers; source is unused on the device.
  Args<T> a{};
  a.n_edges = h.size(0);
  a.k = (int)h.size(1) + 1;
  a.wc = (int)zc.size(2);
  a.wr = (int)zr.size(2);
  a.n_f = (int)angular.size(1);
  a.n_blocks = (int)desc.size(0);
  a.d_out = d_out;
  a.h = h.data_ptr<T>();
  a.zc = zc.data_ptr<T>();
  a.zr = zr.data_ptr<T>();
  a.angular = angular.data_ptr<T>();
  a.transport = transport.data_ptr<T>();
  a.rho = rho.data_ptr<T>();
  a.desc = desc.data_ptr<int32_t>();
  a.source = source.data_ptr<int64_t>();
  return a;
}

// The backward for the edge-row gradient g (n_edges, d_out) when `target` is
// empty, else the aggregated form: g (n_targets, d_out) read through target
// under weights, with dweights from the forward's rows.
void backward_cuda(const torch::Tensor& g, const torch::Tensor& target, const torch::Tensor& weights, const torch::Tensor& rows,
                   const torch::Tensor& h, const torch::Tensor& zc, const torch::Tensor& zr,
                   const torch::Tensor& angular, const torch::Tensor& transport, const torch::Tensor& rho,
                   const torch::Tensor& desc, const torch::Tensor& source, const torch::Tensor& order,
                   const torch::Tensor& group_ptr, const torch::Tensor& group_source,
                   torch::Tensor& dh, torch::Tensor& dzc, torch::Tensor& dzr, torch::Tensor& dweights) {
  check_groups(h, order, group_ptr, group_source);
  const int64_t n_groups = group_source.size(0);
  if (n_groups == 0) return;
  const c10::cuda::CUDAGuard guard(h.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(h.scalar_type(), "gm_factored_backward_cuda", [&] {
    auto a = device_args<scalar_t>(h, zc, zr, angular, transport, rho, desc, source, (int)g.size(1));
    a.g = g.data_ptr<scalar_t>();
    a.dh = dh.data_ptr<scalar_t>();
    a.dzc = dzc.data_ptr<scalar_t>();
    a.dzr = dzr.data_ptr<scalar_t>();
    if (target.numel()) {
      a.target = target.data_ptr<int64_t>();
      a.weights = weights.data_ptr<scalar_t>();
      a.rows = rows.data_ptr<scalar_t>();
      a.dweights = dweights.data_ptr<scalar_t>();
    }
    const int C = 2 * a.wc + a.wr;
    // float32 in tiles of 32 edges (four threads per edge), float64 in tiles of 16 (eight)
    constexpr int TILE = sizeof(scalar_t) == 4 ? 32 : 16;
    const BackwardLayout L = backward_layout(a.k, C);
    TORCH_CHECK(backward_pairs(L) <= 2 * THREADS, "gm_factored: the backward's (t, c) blocks exceed two per thread");
    const size_t smem = (size_t)(L.CP * L.KP + TILE * L.WS + TILE * L.KP) * sizeof(scalar_t);
    auto kernel = backward_pairs(L) <= THREADS ? factored_backward_kernel<scalar_t, TILE, 1> : factored_backward_kernel<scalar_t, TILE, 2>;
    request_shared_memory(kernel, smem);
    kernel<<<(unsigned)n_groups, THREADS, smem, stream>>>(
        a, order.data_ptr<int64_t>(), group_ptr.data_ptr<int64_t>(), group_source.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

void forward_cuda(const torch::Tensor& h, const torch::Tensor& zc, const torch::Tensor& zr,
                  const torch::Tensor& angular, const torch::Tensor& transport, const torch::Tensor& rho,
                  const torch::Tensor& desc, const torch::Tensor& source, const torch::Tensor& order,
                  const torch::Tensor& group_ptr, const torch::Tensor& group_source, torch::Tensor& out, int max_bm) {
  check_groups(h, order, group_ptr, group_source);
  const int64_t n_groups = group_source.size(0);
  if (n_groups == 0) return;
  const c10::cuda::CUDAGuard guard(h.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(h.scalar_type(), "gm_factored_forward_cuda", [&] {
    auto a = device_args<scalar_t>(h, zc, zr, angular, transport, rho, desc, source, (int)out.size(1));
    a.out = out.data_ptr<scalar_t>();
    const size_t smem = (size_t)(a.k * a.wc * 2 + a.k * a.wr) * sizeof(scalar_t);
    auto kernel = max_bm <= NARROW_BM ? factored_kernel<scalar_t, NARROW_BM> : factored_kernel<scalar_t, MAX_BM>;
    request_shared_memory(kernel, smem);
    kernel<<<(unsigned)n_groups, THREADS, smem, stream>>>(
        a, order.data_ptr<int64_t>(), group_ptr.data_ptr<int64_t>(), group_source.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

void aggregate_cuda(const torch::Tensor& rows, const torch::Tensor& weights, const torch::Tensor& row_ptr, torch::Tensor& out) {
  const int64_t n_targets = out.size(0);
  if (n_targets == 0) return;
  const c10::cuda::CUDAGuard guard(rows.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(rows.scalar_type(), "gm_factored_aggregate_cuda", [&] {
    aggregate_kernel<scalar_t><<<(unsigned)n_targets, THREADS, 0, stream>>>(
        rows.data_ptr<scalar_t>(), weights.data_ptr<scalar_t>(), row_ptr.data_ptr<int64_t>(), (int)rows.size(1),
        out.data_ptr<scalar_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

}  // namespace gm_factored
