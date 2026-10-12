// CUDA launchers for the fused pair term: one thread per edge, the term's
// readout rows (and bias) staged in shared memory once per block, the
// per-edge arithmetic from pairs.h. A build instantiates one (A, B) block
// size and one trunk width (GM_PAIRS_A, GM_PAIRS_B, GM_PAIRS_K, set by
// fused_pairs.extension), so compiling is a minute, not most of an hour.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "pairs.h"

namespace gm_pairs {

constexpr int THREADS = 128;

template <typename T, int A, int B>
__global__ void __launch_bounds__(128) forward_kernel(Term<T> t) {
  extern __shared__ __align__(16) unsigned char smem[];
  T* W = reinterpret_cast<T*>(smem);
  const int width = ((t.flags & MATRIX_COMPLEX) ? 2 : 1) * t.a * t.b;
  const int n_w = width * t.trunk;
  for (int i = threadIdx.x; i < n_w; i += blockDim.x) W[i] = t.weight[i];
  T* bias_s = W + n_w;  // the bias, after the weight rows (not `B`: the template parameter)
  if (t.bias) for (int i = threadIdx.x; i < width; i += blockDim.x) bias_s[i] = t.bias[i];
  __syncthreads();
  Term<T> local = t;
  local.bias = t.bias ? bias_s : nullptr;
  for (long e = blockIdx.x * (long)blockDim.x + threadIdx.x; e < t.n_edges; e += (long)gridDim.x * blockDim.x) {
    forward_edge<T, A, B>(local, W, e);
  }
}

template <typename T, int A, int B>
__global__ void __launch_bounds__(128) backward_kernel(Term<T> t) {
  extern __shared__ __align__(16) unsigned char smem[];
  T* W = reinterpret_cast<T*>(smem);
  const int width = ((t.flags & MATRIX_COMPLEX) ? 2 : 1) * t.a * t.b;
  const int n_w = width * t.trunk;
  for (int i = threadIdx.x; i < n_w; i += blockDim.x) W[i] = t.weight[i];
  T* bias_s = W + n_w;  // the bias, after the weight rows (not `B`: the template parameter)
  if (t.bias) for (int i = threadIdx.x; i < width; i += blockDim.x) bias_s[i] = t.bias[i];
  __syncthreads();
  Term<T> local = t;
  local.bias = t.bias ? bias_s : nullptr;
  for (long e = blockIdx.x * (long)blockDim.x + threadIdx.x; e < t.n_edges; e += (long)gridDim.x * blockDim.x) {
    backward_edge<T, A, B>(local, W, e);
  }
}

template <typename T>
static Term<T> device_term(const torch::Tensor& hidden, const torch::Tensor& weight, const torch::Tensor& bias,
                           const torch::Tensor& u, const torch::Tensor& pre, const torch::Tensor& post,
                           const torch::Tensor& scale, int64_t a, int64_t b, int64_t flags) {
  Term<T> t{};
  t.n_edges = (int)hidden.size(0); t.a = (int)a; t.b = (int)b; t.trunk = (int)hidden.size(1); t.flags = (int)flags;
  t.hidden = hidden.data_ptr<T>();
  t.weight = weight.data_ptr<T>();
  t.bias = bias.defined() && bias.numel() ? bias.data_ptr<T>() : nullptr;
  t.u = u.data_ptr();
  t.pre = pre.defined() && pre.numel() ? reinterpret_cast<const cx<T>*>(pre.data_ptr()) : nullptr;
  t.post = post.defined() && post.numel() ? reinterpret_cast<const cx<T>*>(post.data_ptr()) : nullptr;
  t.scale = scale.defined() && scale.numel() ? scale.data_ptr<T>() : nullptr;
  return t;
}

// Dynamic shared memory above the 48 KB default (double, the widest
// blocks) must be opted into per kernel.
template <typename Kernel>
static void allow_shared(Kernel kernel, size_t smem) {
  if (smem > 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  }
}

template <typename T>
static void forward_launch(const Term<T>& t, int blocks, size_t smem, cudaStream_t stream) {
  switch (t.a * 100 + t.b) {
#define GM_CASE(A, B) \
    case A * 100 + B: \
      allow_shared(forward_kernel<T, A, B>, smem); \
      forward_kernel<T, A, B><<<blocks, THREADS, smem, stream>>>(t); \
      break;
    GM_PAIRS_SIZES(GM_CASE)
#undef GM_CASE
    default: TORCH_CHECK(false, "gm_pairs: unsupported block size ", t.a, " x ", t.b);
  }
}

template <typename T>
static void backward_launch(const Term<T>& t, int blocks, size_t smem, cudaStream_t stream) {
  switch (t.a * 100 + t.b) {
#define GM_CASE(A, B) \
    case A * 100 + B: \
      allow_shared(backward_kernel<T, A, B>, smem); \
      backward_kernel<T, A, B><<<blocks, THREADS, smem, stream>>>(t); \
      break;
    GM_PAIRS_SIZES(GM_CASE)
#undef GM_CASE
    default: TORCH_CHECK(false, "gm_pairs: unsupported block size ", t.a, " x ", t.b);
  }
}

static size_t shared_bytes(int64_t a, int64_t b, int64_t trunk, int64_t flags, size_t elem) {
  const int64_t width = ((flags & MATRIX_COMPLEX) ? 2 : 1) * a * b;
  return (size_t)(width * trunk + width) * elem;
}

static int blocks_for(int64_t n_edges) {
  const int64_t wanted = (n_edges + THREADS - 1) / THREADS;
  return (int)std::min<int64_t>(wanted, 65535);
}

void forward_cuda(const torch::Tensor& hidden, const torch::Tensor& weight, const torch::Tensor& bias,
                  const torch::Tensor& u, const torch::Tensor& pre, const torch::Tensor& post,
                  const torch::Tensor& scale, torch::Tensor out, int64_t a, int64_t b, int64_t flags) {
  const c10::cuda::OptionalCUDAGuard guard(hidden.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(hidden.scalar_type(), "gm_pairs_forward_cuda", [&] {
    Term<scalar_t> t = device_term<scalar_t>(hidden, weight, bias, u, pre, post, scale, a, b, flags);
    t.out = out.data_ptr();
    const size_t smem = shared_bytes(a, b, hidden.size(1), flags, sizeof(scalar_t));
    forward_launch<scalar_t>(t, blocks_for(hidden.size(0)), smem, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

void backward_cuda(const torch::Tensor& hidden, const torch::Tensor& weight, const torch::Tensor& bias,
                   const torch::Tensor& u, const torch::Tensor& pre, const torch::Tensor& post,
                   const torch::Tensor& scale, const torch::Tensor& grad_out, torch::Tensor grad_u,
                   torch::Tensor grad_hidden, torch::Tensor grad_k, int64_t a, int64_t b, int64_t flags) {
  const c10::cuda::OptionalCUDAGuard guard(hidden.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(hidden.scalar_type(), "gm_pairs_backward_cuda", [&] {
    Term<scalar_t> t = device_term<scalar_t>(hidden, weight, bias, u, pre, post, scale, a, b, flags);
    t.grad_out = grad_out.data_ptr();
    t.grad_u = grad_u.data_ptr();
    t.grad_hidden = grad_hidden.data_ptr<scalar_t>();
    t.grad_k = grad_k.data_ptr<scalar_t>();
    t.grad_k_stride = grad_k.stride(0);
    const size_t smem = shared_bytes(a, b, hidden.size(1), flags, sizeof(scalar_t));
    backward_launch<scalar_t>(t, blocks_for(hidden.size(0)), smem, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

}  // namespace gm_pairs
