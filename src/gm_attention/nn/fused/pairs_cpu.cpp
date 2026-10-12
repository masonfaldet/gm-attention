// CPU driver and bindings for the fused pair term (pairs.h); the CUDA
// launchers live in pairs_cuda.cu and are bound here when built with it.
// The CPU build instantiates every size in GM_PAIRS_SIZES with a runtime
// trunk width; a CUDA build is for one size and trunk width (pairs.h).
#include <torch/extension.h>
#include <ATen/Parallel.h>

#include "pairs.h"

namespace gm_pairs {

template <typename T>
Term<T> make_term(int64_t n_edges, int64_t a, int64_t b, int64_t trunk, int64_t flags,
                  const torch::Tensor& hidden, const torch::Tensor& weight, const torch::Tensor& bias,
                  const torch::Tensor& u, const torch::Tensor& pre, const torch::Tensor& post,
                  const torch::Tensor& scale, torch::Tensor& out) {
  Term<T> t{};
  t.n_edges = (int)n_edges; t.a = (int)a; t.b = (int)b; t.trunk = (int)trunk; t.flags = (int)flags;
  t.hidden = hidden.data_ptr<T>();
  t.weight = weight.data_ptr<T>();
  t.bias = bias.defined() && bias.numel() ? bias.data_ptr<T>() : nullptr;
  t.u = u.data_ptr();
  t.pre = pre.defined() && pre.numel() ? reinterpret_cast<const cx<T>*>(pre.data_ptr()) : nullptr;
  t.post = post.defined() && post.numel() ? reinterpret_cast<const cx<T>*>(post.data_ptr()) : nullptr;
  t.scale = scale.defined() && scale.numel() ? scale.data_ptr<T>() : nullptr;
  t.out = out.data_ptr();
  return t;
}

#define GM_CHECK(cond, msg) TORCH_CHECK(cond, "gm_pairs: ", msg)

template <typename T>
void forward_dispatch(const Term<T>& t) {
  switch (t.a * 100 + t.b) {
#define GM_CASE(A, B) \
    case A * 100 + B: \
      at::parallel_for(0, t.n_edges, 256, [&](int64_t begin, int64_t end) { \
        for (int64_t e = begin; e < end; ++e) forward_edge<T, A, B>(t, t.weight, e); \
      }); \
      break;
    GM_PAIRS_SIZES(GM_CASE)
#undef GM_CASE
    default: TORCH_CHECK(false, "gm_pairs: unsupported block size ", t.a, " x ", t.b);
  }
}

template <typename T>
void backward_dispatch(const Term<T>& t) {
  switch (t.a * 100 + t.b) {
#define GM_CASE(A, B) \
    case A * 100 + B: \
      at::parallel_for(0, t.n_edges, 256, [&](int64_t begin, int64_t end) { \
        for (int64_t e = begin; e < end; ++e) backward_edge<T, A, B>(t, t.weight, e); \
      }); \
      break;
    GM_PAIRS_SIZES(GM_CASE)
#undef GM_CASE
    default: TORCH_CHECK(false, "gm_pairs: unsupported block size ", t.a, " x ", t.b);
  }
}

static void check_common(const torch::Tensor& hidden, const torch::Tensor& weight, const torch::Tensor& u,
                         const torch::Tensor& out, int64_t a, int64_t b, int64_t trunk, int64_t flags) {
  GM_CHECK(a >= 1 && a <= MAX_MULT && b >= 1 && b <= MAX_MULT, "block dimensions must be in 1..16");
  GM_CHECK(trunk >= 1 && trunk <= MAX_TRUNK, "trunk width must be in 1..16");
  GM_CHECK(K_FIXED == 0 || trunk == K_FIXED, "this build is compiled for another trunk width");
  GM_CHECK(hidden.is_contiguous() && weight.is_contiguous() && u.is_contiguous() && out.is_contiguous(),
           "hidden, weight, u and out must be contiguous");
  const int64_t width = ((flags & MATRIX_COMPLEX) ? 2 : 1) * a * b;
  GM_CHECK(weight.size(0) == width && weight.size(1) == trunk, "weight must be (width, trunk)");
  GM_CHECK(hidden.size(1) == trunk, "hidden must be (n_edges, trunk)");
  GM_CHECK(u.size(1) == a && out.size(1) == b, "u must be (n_edges, a) and out (n_edges, b)");
  GM_CHECK(u.size(0) == hidden.size(0) && out.size(0) == hidden.size(0), "n_edges mismatch");
  GM_CHECK(((flags & U_REAL) != 0) == !u.is_complex(), "u must be real iff U_REAL");
  GM_CHECK(((flags & OUT_REAL) != 0) == !out.is_complex(), "out must be real iff OUT_REAL");
}

#ifdef GM_PAIRS_WITH_CUDA
void forward_cuda(const torch::Tensor& hidden, const torch::Tensor& weight, const torch::Tensor& bias,
                  const torch::Tensor& u, const torch::Tensor& pre, const torch::Tensor& post,
                  const torch::Tensor& scale, torch::Tensor out, int64_t a, int64_t b, int64_t flags);
void backward_cuda(const torch::Tensor& hidden, const torch::Tensor& weight, const torch::Tensor& bias,
                   const torch::Tensor& u, const torch::Tensor& pre, const torch::Tensor& post,
                   const torch::Tensor& scale, const torch::Tensor& grad_out, torch::Tensor grad_u,
                   torch::Tensor grad_hidden, torch::Tensor grad_k, int64_t a, int64_t b, int64_t flags);
#endif

void forward(const torch::Tensor& hidden, const torch::Tensor& weight, const torch::Tensor& bias,
             const torch::Tensor& u, const torch::Tensor& pre, const torch::Tensor& post,
             const torch::Tensor& scale, torch::Tensor out, int64_t a, int64_t b, int64_t flags) {
  const int64_t trunk = hidden.size(1);
  check_common(hidden, weight, u, out, a, b, trunk, flags);
  if (hidden.is_cuda()) {
#ifdef GM_PAIRS_WITH_CUDA
    forward_cuda(hidden, weight, bias, u, pre, post, scale, out, a, b, flags);
    return;
#else
    TORCH_CHECK(false, "gm_pairs was built without CUDA");
#endif
  }
  const int64_t n_edges = hidden.size(0);
  AT_DISPATCH_FLOATING_TYPES(hidden.scalar_type(), "gm_pairs_forward", [&] {
    Term<scalar_t> t = make_term<scalar_t>(n_edges, a, b, trunk, flags, hidden, weight, bias, u, pre, post, scale, out);
    forward_dispatch<scalar_t>(t);
  });
}

void backward(const torch::Tensor& hidden, const torch::Tensor& weight, const torch::Tensor& bias,
              const torch::Tensor& u, const torch::Tensor& pre, const torch::Tensor& post,
              const torch::Tensor& scale, const torch::Tensor& grad_out, torch::Tensor grad_u,
              torch::Tensor grad_hidden, torch::Tensor grad_k, int64_t a, int64_t b, int64_t flags) {
  const int64_t trunk = hidden.size(1);
  check_common(hidden, weight, u, grad_out, a, b, trunk, flags);
  GM_CHECK(grad_u.is_contiguous() && grad_hidden.is_contiguous(), "grad_u and grad_hidden must be contiguous");
  GM_CHECK(grad_k.is_contiguous() && grad_k.size(1) == hidden.size(0)
           && grad_k.size(0) == ((flags & MATRIX_COMPLEX) ? 2 : 1) * a * b, "grad_k must be (width, n_edges), contiguous");
  if (hidden.is_cuda()) {
#ifdef GM_PAIRS_WITH_CUDA
    backward_cuda(hidden, weight, bias, u, pre, post, scale, grad_out, grad_u, grad_hidden, grad_k, a, b, flags);
    return;
#else
    TORCH_CHECK(false, "gm_pairs was built without CUDA");
#endif
  }
  const int64_t n_edges = hidden.size(0);
  AT_DISPATCH_FLOATING_TYPES(hidden.scalar_type(), "gm_pairs_backward", [&] {
    torch::Tensor dummy_out = grad_out;  // make_term wants an out; the backward never writes it
    Term<scalar_t> t = make_term<scalar_t>(n_edges, a, b, trunk, flags, hidden, weight, bias, u, pre, post, scale, dummy_out);
    t.grad_out = grad_out.data_ptr();
    t.grad_u = grad_u.data_ptr();
    t.grad_hidden = grad_hidden.data_ptr<scalar_t>();
    t.grad_k = grad_k.data_ptr<scalar_t>();
    t.grad_k_stride = grad_k.stride(0);
    backward_dispatch<scalar_t>(t);
  });
}

}  // namespace gm_pairs

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gm_pairs::forward, "fused pair term forward (accumulates into out)");
  m.def("backward", &gm_pairs::backward, "fused pair term backward (accumulates grad_u, grad_hidden; writes grad_k)");
#ifdef GM_PAIRS_WITH_CUDA
  m.attr("with_cuda") = true;
#else
  m.attr("with_cuda") = false;
#endif
}
