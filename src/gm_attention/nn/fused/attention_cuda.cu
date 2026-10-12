// The fused scores-and-softmax op's and depthwise mix's CUDA kernels: one
// thread block per target node (attention.h has the per-component arithmetic
// they share with the CPU driver).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <mutex>
#include <unordered_map>

#include "attention.h"

namespace gm_fused_attention {

constexpr int THREADS = 128;
constexpr int NWARPS = THREADS / 32;

template <typename T>
struct AtomicAdd {  // host-callable so the shared templates instantiate in both passes; only the device body runs
  __host__ __device__ __forceinline__ void operator()(T* p, T value) const {
#ifdef __CUDA_ARCH__
    atomicAdd(p, value);
#else
    *p += value;
#endif
  }
};
template <typename T>
struct MaxOp {
  __device__ __forceinline__ T operator()(T a, T b) const { return a > b ? a : b; }
};
template <typename T>
struct SumOp {
  __device__ __forceinline__ T operator()(T a, T b) const { return a + b; }
};

// Reduce H per-thread values over the block; every thread gets the results.
// red holds NWARPS * H entries; all threads call this at the same point.
template <typename T, typename Op>
__device__ __forceinline__ void block_reduce(T* vals, int H, T* red, Op op) {
  for (int h = 0; h < H; ++h) {
    T value = vals[h];
    for (int offset = 16; offset > 0; offset >>= 1) value = op(value, __shfl_down_sync(0xffffffffu, value, offset));
    vals[h] = value;
  }
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0)
    for (int h = 0; h < H; ++h) red[warp * H + h] = vals[h];
  __syncthreads();
  for (int h = 0; h < H; ++h) {
    T value = red[h];
    for (int w = 1; w < NWARPS; ++w) value = op(value, red[w * H + h]);
    vals[h] = value;
  }
  __syncthreads();
}

// The scores and softmax (fused_attention.softmax): one block per target
// node stages its query, scores its edges into shared memory and normalizes
// there; the value messages come from the factored kernel.
template <typename T>
__global__ void __launch_bounds__(THREADS) softmax_forward_kernel(Args<T> a, int deg_max) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  T* q_s = reinterpret_cast<T*>(smem_raw);  // D
  T* lw = q_s + a.D;                         // deg_max * H: log weights
  T* red = lw + (size_t)deg_max * a.H;       // NWARPS * H
  const int64_t i = blockIdx.x;
  const int64_t e0 = a.row_ptr[i], e1 = a.row_ptr[i + 1];
  const int deg = (int)(e1 - e0), H = a.H;
  for (int c = threadIdx.x; c < a.D; c += THREADS) q_s[c] = a.q[i * a.D + c];
  __syncthreads();
  for (int t = threadIdx.x; t < deg; t += THREADS) {
    const int64_t e = e0 + t, j = a.source[e];
    for (int h = 0; h < H; ++h) lw[t * H + h] = a.scale * edge_score(a, e, j, h, q_s) + a.bias[e];
  }
  __syncthreads();
  T loc[MAX_H];
  for (int h = 0; h < H; ++h) loc[h] = lowest<T>();
  for (int t = threadIdx.x; t < deg; t += THREADS)
    for (int h = 0; h < H; ++h) loc[h] = lw[t * H + h] > loc[h] ? lw[t * H + h] : loc[h];
  block_reduce(loc, H, red, MaxOp<T>());
  T m[MAX_H];
  for (int h = 0; h < H; ++h) {
    m[h] = loc[h];
    loc[h] = T(0);
  }
  for (int t = threadIdx.x; t < deg; t += THREADS) {
    const bool in = a.inside[e0 + t] != 0;
    for (int h = 0; h < H; ++h) {
      const T u = in ? exp(lw[t * H + h] - m[h]) : T(0);
      lw[t * H + h] = u;
      loc[h] += u;
    }
  }
  block_reduce(loc, H, red, SumOp<T>());
  for (int t = threadIdx.x; t < deg; t += THREADS)
    for (int h = 0; h < H; ++h) a.alpha[(e0 + t) * H + h] = lw[t * H + h] / loc[h];
}

// The softmax adjoint and the score's backward from d alpha, per target
// node again: dq in registers, dk scattered (atomics).
template <typename T>
__global__ void __launch_bounds__(THREADS) softmax_backward_kernel(Args<T> a, int deg_max) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  T* q_s = reinterpret_cast<T*>(smem_raw);   // D
  T* al = q_s + a.D;                          // deg_max * H
  T* da = al + (size_t)deg_max * a.H;         // deg_max * H: d alpha, then ds
  T* red = da + (size_t)deg_max * a.H;        // NWARPS * H
  const int64_t i = blockIdx.x;
  const int64_t e0 = a.row_ptr[i], e1 = a.row_ptr[i + 1];
  const int deg = (int)(e1 - e0), H = a.H;
  for (int c = threadIdx.x; c < a.D; c += THREADS) q_s[c] = a.q[i * a.D + c];
  for (int t = threadIdx.x; t < deg; t += THREADS)
    for (int h = 0; h < H; ++h) {
      al[t * H + h] = a.alpha[(e0 + t) * H + h];
      da[t * H + h] = a.dalpha[(e0 + t) * H + h];
    }
  __syncthreads();
  T loc[MAX_H];
  for (int h = 0; h < H; ++h) loc[h] = T(0);
  for (int t = threadIdx.x; t < deg; t += THREADS)
    for (int h = 0; h < H; ++h) loc[h] += al[t * H + h] * da[t * H + h];
  block_reduce(loc, H, red, SumOp<T>());
  for (int t = threadIdx.x; t < deg; t += THREADS)
    for (int h = 0; h < H; ++h) da[t * H + h] = a.scale * al[t * H + h] * (da[t * H + h] - loc[h]);
  __syncthreads();
  const AtomicAdd<T> add;
  for (int c = threadIdx.x; c < a.D; c += THREADS) {
    Slot s;
    locate(a, c, s);
    T dq = T(0);
    for (int t = 0; t < deg; ++t) {
      const int64_t e = e0 + t;
      dq += score_backward_component(a, e, a.source[e], s, da[t * H + s.h], q_s, add);
    }
    a.dq[i * a.D + c] = dq;
  }
}

// The depthwise convolution: the aggregation walk with the weights given.
template <typename T>
__global__ void __launch_bounds__(THREADS) conv_forward_kernel(Args<T> a) {
  const int64_t i = blockIdx.x;
  const int64_t e0 = a.row_ptr[i], e1 = a.row_ptr[i + 1];
  for (int c = threadIdx.x; c < a.D; c += THREADS) {
    Slot s;
    locate(a, c, s);
    T y = T(0);
    for (int64_t e = e0; e < e1; ++e) y += a.alpha[e] * message(a, e, a.source[e], s);
    a.out[i * a.D + c] = y;
  }
}

template <typename T>
__global__ void __launch_bounds__(THREADS) conv_backward_kernel(Args<T> a) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  T* dy_s = reinterpret_cast<T*>(smem_raw);  // D
  const int64_t i = blockIdx.x;
  const int64_t e0 = a.row_ptr[i], e1 = a.row_ptr[i + 1];
  for (int c = threadIdx.x; c < a.D; c += THREADS) dy_s[c] = a.g[i * a.D + c];
  __syncthreads();
  const AtomicAdd<T> add;
  for (int c = threadIdx.x; c < a.D; c += THREADS) {
    Slot s;
    locate(a, c, s);
    for (int64_t e = e0; e < e1; ++e) backward_component(a, e, a.source[e], s, a.alpha[e], dy_s, add);
  }
}

// Opt the kernel into `smem` bytes of dynamic shared memory when that exceeds
// the default 48 KiB, refusing with a readable message past the device's limit
// (the Python driver keeps callers below it; this guards a direct call).
// Once per device and size: the attribute persists, so a driver call per
// launch is wasted, and it is not one a stream capture (a CUDA graph) admits.
template <typename Kernel>
void request_shared_memory(Kernel kernel, size_t smem, const torch::Tensor& q) {
  if (smem <= 48 * 1024) return;
  static std::mutex lock;
  static std::unordered_map<const void*, std::unordered_map<int, size_t>> granted;
  const int device = q.get_device();
  const void* key = reinterpret_cast<const void*>(kernel);
  std::lock_guard<std::mutex> guard(lock);
  auto& sizes = granted[key];
  auto it = sizes.find(device);
  if (it != sizes.end() && it->second >= smem) return;
  int limit = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  TORCH_CHECK((int64_t)smem <= (int64_t)limit, "gm_fused_attention: the largest neighbourhood needs ", smem,
              " bytes of shared memory per block; the device grants ", limit);
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  sizes[device] = smem;
}

template <typename T>
Args<T> device_softmax_args(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& transport, const torch::Tensor& bias,
                            const torch::Tensor& inside, const torch::Tensor& desc, const torch::Tensor& row_ptr,
                            const torch::Tensor& source, int64_t H, double scale) {
  Args<T> a{};
  a.n_nodes = q.size(0);
  a.n_edges = source.size(0);
  a.H = (int)H;
  a.D = (int)q.size(1);
  a.F = (int)desc.size(0);
  a.orders = (int)transport.size(1);
  a.scale = (T)scale;
  a.row_ptr = row_ptr.data_ptr<int64_t>();
  a.source = source.data_ptr<int64_t>();
  a.desc = desc.data_ptr<int64_t>();
  a.q = q.data_ptr<T>();
  a.k = k.data_ptr<T>();
  a.transport = transport.data_ptr<T>();
  a.bias = bias.data_ptr<T>();
  a.inside = inside.data_ptr<uint8_t>();
  return a;
}

void softmax_forward_cuda(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& transport,
                          const torch::Tensor& bias, const torch::Tensor& inside, const torch::Tensor& desc,
                          const torch::Tensor& row_ptr, const torch::Tensor& source, int64_t H, double scale,
                          int64_t deg_max, torch::Tensor& alpha) {
  TORCH_CHECK(deg_max >= 1, "deg_max must be positive");
  const c10::cuda::CUDAGuard guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "gm_fused_softmax_forward_cuda", [&] {
    auto a = device_softmax_args<scalar_t>(q, k, transport, bias, inside, desc, row_ptr, source, H, scale);
    a.alpha = alpha.data_ptr<scalar_t>();
    const size_t smem = (size_t)(a.D + deg_max * H + NWARPS * H) * sizeof(scalar_t);
    auto kernel = softmax_forward_kernel<scalar_t>;
    request_shared_memory(kernel, smem, q);
    kernel<<<(unsigned)a.n_nodes, THREADS, smem, stream>>>(a, (int)deg_max);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

void softmax_backward_cuda(const torch::Tensor& dalpha, const torch::Tensor& q, const torch::Tensor& k,
                           const torch::Tensor& transport, const torch::Tensor& bias, const torch::Tensor& inside,
                           const torch::Tensor& desc, const torch::Tensor& row_ptr, const torch::Tensor& source,
                           const torch::Tensor& alpha, int64_t H, double scale, int64_t deg_max, torch::Tensor& dq,
                           torch::Tensor& dk) {
  TORCH_CHECK(deg_max >= 1, "deg_max must be positive");
  const c10::cuda::CUDAGuard guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "gm_fused_softmax_backward_cuda", [&] {
    auto a = device_softmax_args<scalar_t>(q, k, transport, bias, inside, desc, row_ptr, source, H, scale);
    a.alpha = alpha.data_ptr<scalar_t>();
    a.dalpha = dalpha.data_ptr<scalar_t>();
    a.dq = dq.data_ptr<scalar_t>();
    a.dk = dk.data_ptr<scalar_t>();
    const size_t smem = (size_t)(a.D + 2 * deg_max * H + NWARPS * H) * sizeof(scalar_t);
    auto kernel = softmax_backward_kernel<scalar_t>;
    request_shared_memory(kernel, smem, q);
    kernel<<<(unsigned)a.n_nodes, THREADS, smem, stream>>>(a, (int)deg_max);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

template <typename T>
Args<T> device_conv_args(const torch::Tensor& v, const torch::Tensor& rc, const torch::Tensor& cc, const torch::Tensor& transport,
                         const torch::Tensor& angular, const torch::Tensor& rho, const torch::Tensor& weights,
                         const torch::Tensor& desc, const torch::Tensor& row_ptr, const torch::Tensor& source, int64_t A0,
                         int64_t W) {
  Args<T> a{};
  a.n_nodes = v.size(0);
  a.n_edges = source.size(0);
  a.H = 1;
  a.D = (int)v.size(1);
  a.W = (int)W;
  a.A0 = (int)A0;
  a.F = (int)desc.size(0);
  a.orders = (int)transport.size(1);
  a.scale = T(1);
  a.row_ptr = row_ptr.data_ptr<int64_t>();
  a.source = source.data_ptr<int64_t>();
  a.desc = desc.data_ptr<int64_t>();
  a.v = v.data_ptr<T>();
  a.rc = rc.numel() ? rc.data_ptr<T>() : nullptr;
  a.cc = cc.numel() ? cc.data_ptr<T>() : nullptr;
  a.transport = transport.data_ptr<T>();
  a.angular = angular.data_ptr<T>();
  a.rho = rho.data_ptr<T>();
  a.alpha = weights.data_ptr<T>();
  return a;
}

void conv_forward_cuda(const torch::Tensor& v, const torch::Tensor& rc, const torch::Tensor& cc, const torch::Tensor& transport,
                       const torch::Tensor& angular, const torch::Tensor& rho, const torch::Tensor& weights,
                       const torch::Tensor& desc, const torch::Tensor& row_ptr, const torch::Tensor& source, int64_t A0,
                       int64_t W, torch::Tensor& out) {
  const c10::cuda::CUDAGuard guard(v.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(v.scalar_type(), "gm_fused_conv_forward_cuda", [&] {
    auto a = device_conv_args<scalar_t>(v, rc, cc, transport, angular, rho, weights, desc, row_ptr, source, A0, W);
    a.out = out.data_ptr<scalar_t>();
    conv_forward_kernel<scalar_t><<<(unsigned)a.n_nodes, THREADS, 0, stream>>>(a);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

void conv_backward_cuda(const torch::Tensor& g, const torch::Tensor& v, const torch::Tensor& rc, const torch::Tensor& cc,
                        const torch::Tensor& transport, const torch::Tensor& angular, const torch::Tensor& rho,
                        const torch::Tensor& weights, const torch::Tensor& desc, const torch::Tensor& row_ptr,
                        const torch::Tensor& source, int64_t A0, int64_t W, torch::Tensor& dv, torch::Tensor& drc,
                        torch::Tensor& dcc) {
  const c10::cuda::CUDAGuard guard(v.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(v.scalar_type(), "gm_fused_conv_backward_cuda", [&] {
    auto a = device_conv_args<scalar_t>(v, rc, cc, transport, angular, rho, weights, desc, row_ptr, source, A0, W);
    a.g = g.data_ptr<scalar_t>();
    a.dv = dv.data_ptr<scalar_t>();
    a.drc = drc.numel() ? drc.data_ptr<scalar_t>() : nullptr;
    a.dcc = dcc.numel() ? dcc.data_ptr<scalar_t>() : nullptr;
    const size_t smem = (size_t)a.D * sizeof(scalar_t);
    auto kernel = conv_backward_kernel<scalar_t>;
    request_shared_memory(kernel, smem, v);
    kernel<<<(unsigned)a.n_nodes, THREADS, smem, stream>>>(a);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

}  // namespace gm_fused_attention
