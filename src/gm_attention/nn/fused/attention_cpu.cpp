// The fused scores-and-softmax op's and depthwise mix's CPU driver and the
// extension's entry points (attention.h has the routines; attention_cuda.cu
// the CUDA kernels).
#include <torch/extension.h>

#include <vector>

#include "attention.h"

namespace gm_fused_attention {

#ifdef GM_ATTENTION_WITH_CUDA
void conv_forward_cuda(const torch::Tensor& v, const torch::Tensor& rc, const torch::Tensor& cc, const torch::Tensor& transport,
                       const torch::Tensor& angular, const torch::Tensor& rho, const torch::Tensor& weights,
                       const torch::Tensor& desc, const torch::Tensor& row_ptr, const torch::Tensor& source, int64_t A0,
                       int64_t W, torch::Tensor& out);
void conv_backward_cuda(const torch::Tensor& g, const torch::Tensor& v, const torch::Tensor& rc, const torch::Tensor& cc,
                        const torch::Tensor& transport, const torch::Tensor& angular, const torch::Tensor& rho,
                        const torch::Tensor& weights, const torch::Tensor& desc, const torch::Tensor& row_ptr,
                        const torch::Tensor& source, int64_t A0, int64_t W, torch::Tensor& dv, torch::Tensor& drc,
                        torch::Tensor& dcc);
void softmax_forward_cuda(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& transport,
                          const torch::Tensor& bias, const torch::Tensor& inside, const torch::Tensor& desc,
                          const torch::Tensor& row_ptr, const torch::Tensor& source, int64_t H, double scale,
                          int64_t deg_max, torch::Tensor& alpha);
void softmax_backward_cuda(const torch::Tensor& dalpha, const torch::Tensor& q, const torch::Tensor& k,
                           const torch::Tensor& transport, const torch::Tensor& bias, const torch::Tensor& inside,
                           const torch::Tensor& desc, const torch::Tensor& row_ptr, const torch::Tensor& source,
                           const torch::Tensor& alpha, int64_t H, double scale, int64_t deg_max, torch::Tensor& dq,
                           torch::Tensor& dk);
#endif

int64_t max_degree_cpu(const torch::Tensor& row_ptr) {
  const auto p = row_ptr.accessor<int64_t, 1>();
  int64_t deg = 0;
  for (int64_t i = 0; i + 1 < row_ptr.size(0); ++i) deg = std::max(deg, p[i + 1] - p[i]);
  return deg;
}

// ---- The depthwise convolution (weights given). ----

void check_conv(const torch::Tensor& v, const torch::Tensor& rc, const torch::Tensor& cc, const torch::Tensor& transport,
                const torch::Tensor& angular, const torch::Tensor& rho, const torch::Tensor& weights, const torch::Tensor& desc,
                const torch::Tensor& row_ptr, const torch::Tensor& source, int64_t A0, int64_t W) {
  TORCH_CHECK(v.dim() == 2, "v must be (n_nodes, D)");
  const int64_t N = v.size(0), E = source.size(0);
  for (const auto* t : {&v, &rc, &cc, &transport, &angular, &rho, &weights, &desc, &row_ptr, &source}) {
    TORCH_CHECK(t->is_contiguous(), "every tensor must be contiguous");
    TORCH_CHECK(t->device() == v.device(), "every tensor must live on v's device");
  }
  for (const auto* t : {&rc, &cc, &transport, &angular, &rho, &weights}) {
    TORCH_CHECK(t->scalar_type() == v.scalar_type(), "v, the coefficients, the weights and the tables must share a dtype");
  }
  TORCH_CHECK(rc.numel() == E * A0, "rc must hold (1, n_edges, a_0) entries");
  TORCH_CHECK(cc.numel() == E * W, "cc must hold (1, n_edges, W) entries");
  TORCH_CHECK(transport.dim() == 3 && transport.size(0) == E && transport.size(2) == 2, "transport must be (n_edges, orders, 2)");
  TORCH_CHECK(angular.sizes() == transport.sizes(), "angular must match transport's shape");
  TORCH_CHECK(rho.dim() == 1 && rho.size(0) == E && weights.sizes() == rho.sizes(), "rho and weights must be (n_edges,)");
  TORCH_CHECK(desc.scalar_type() == torch::kInt64 && desc.dim() == 2 && desc.size(1) == DESC && desc.size(0) <= MAX_F,
              "desc must be int64 (F <= ", MAX_F, ", ", DESC, ")");
  TORCH_CHECK(row_ptr.scalar_type() == torch::kInt64 && row_ptr.dim() == 1 && row_ptr.size(0) == N + 1, "row_ptr must be int64 (n_nodes + 1,)");
  TORCH_CHECK(source.scalar_type() == torch::kInt64 && source.dim() == 1, "source must be int64 (n_edges,)");
  TORCH_CHECK(v.size(1) >= 1, "D must be positive");
}

template <typename T>
Args<T> make_conv_args(const torch::Tensor& v, const torch::Tensor& rc, const torch::Tensor& cc, const torch::Tensor& transport,
                       const torch::Tensor& angular, const torch::Tensor& rho, const torch::Tensor& weights,
                       const torch::Tensor& desc, const torch::Tensor& row_ptr, const torch::Tensor& source, int64_t A0, int64_t W) {
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
  a.alpha = weights.data_ptr<T>();  // the given per-edge weights stand where the attention weights would
  return a;
}

// out (n_nodes, D) = sum over each target's edges of weight_e * C_e (tau v_j).
torch::Tensor conv_forward(const torch::Tensor& v, const torch::Tensor& rc, const torch::Tensor& cc, const torch::Tensor& transport,
                           const torch::Tensor& angular, const torch::Tensor& rho, const torch::Tensor& weights,
                           const torch::Tensor& desc, const torch::Tensor& row_ptr, const torch::Tensor& source, int64_t A0,
                           int64_t W) {
  check_conv(v, rc, cc, transport, angular, rho, weights, desc, row_ptr, source, A0, W);
  auto out = torch::empty_like(v);
  if (v.size(0) == 0) return out;
  if (v.is_cuda()) {
#ifdef GM_ATTENTION_WITH_CUDA
    conv_forward_cuda(v, rc, cc, transport, angular, rho, weights, desc, row_ptr, source, A0, W, out);
    return out;
#else
    TORCH_CHECK(false, "gm_fused_attention was built without CUDA");
#endif
  }
  AT_DISPATCH_FLOATING_TYPES(v.scalar_type(), "gm_fused_conv_forward", [&] {
    auto a = make_conv_args<scalar_t>(v, rc, cc, transport, angular, rho, weights, desc, row_ptr, source, A0, W);
    a.out = out.data_ptr<scalar_t>();
    for (int64_t i = 0; i < a.n_nodes; ++i) node_conv_forward<scalar_t>(a, i);
  });
  return out;
}

// (dv, drc, dcc) from the output gradient g (n_nodes, D); the weights get none.
std::vector<torch::Tensor> conv_backward(const torch::Tensor& g, const torch::Tensor& v, const torch::Tensor& rc,
                                         const torch::Tensor& cc, const torch::Tensor& transport, const torch::Tensor& angular,
                                         const torch::Tensor& rho, const torch::Tensor& weights, const torch::Tensor& desc,
                                         const torch::Tensor& row_ptr, const torch::Tensor& source, int64_t A0, int64_t W) {
  check_conv(v, rc, cc, transport, angular, rho, weights, desc, row_ptr, source, A0, W);
  TORCH_CHECK(g.sizes() == v.sizes() && g.is_contiguous() && g.scalar_type() == v.scalar_type(), "g must be contiguous, shaped and typed as v");
  auto dv = torch::zeros_like(v);
  auto drc = torch::empty_like(rc);
  auto dcc = torch::empty_like(cc);
  if (v.size(0) == 0) return {dv, drc, dcc};
  if (v.is_cuda()) {
#ifdef GM_ATTENTION_WITH_CUDA
    conv_backward_cuda(g, v, rc, cc, transport, angular, rho, weights, desc, row_ptr, source, A0, W, dv, drc, dcc);
    return {dv, drc, dcc};
#else
    TORCH_CHECK(false, "gm_fused_attention was built without CUDA");
#endif
  }
  AT_DISPATCH_FLOATING_TYPES(v.scalar_type(), "gm_fused_conv_backward", [&] {
    auto a = make_conv_args<scalar_t>(v, rc, cc, transport, angular, rho, weights, desc, row_ptr, source, A0, W);
    a.g = g.data_ptr<scalar_t>();
    a.dv = dv.data_ptr<scalar_t>();
    a.drc = drc.numel() ? drc.data_ptr<scalar_t>() : nullptr;
    a.dcc = dcc.numel() ? dcc.data_ptr<scalar_t>() : nullptr;
    for (int64_t i = 0; i < a.n_nodes; ++i) node_conv_backward<scalar_t>(a, i);
  });
  return {dv, drc, dcc};
}

// ---- The scores and softmax alone: alpha from (q, k), and (dq, dk) from d alpha. ----

void check_softmax(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& transport, const torch::Tensor& bias,
                   const torch::Tensor& inside, const torch::Tensor& desc, const torch::Tensor& row_ptr,
                   const torch::Tensor& source, int64_t H) {
  TORCH_CHECK(q.dim() == 2, "q must be (n_nodes, D)");
  const int64_t N = q.size(0), E = source.size(0);
  for (const auto* t : {&q, &k, &transport, &bias, &inside, &desc, &row_ptr, &source}) {
    TORCH_CHECK(t->is_contiguous(), "every tensor must be contiguous");
    TORCH_CHECK(t->device() == q.device(), "every tensor must live on q's device");
  }
  for (const auto* t : {&k, &transport, &bias}) {
    TORCH_CHECK(t->scalar_type() == q.scalar_type(), "q, k and the tables must share a dtype");
  }
  TORCH_CHECK(k.sizes() == q.sizes(), "q and k must share a shape");
  TORCH_CHECK(1 <= H && H <= MAX_H, "at most ", MAX_H, " heads");
  TORCH_CHECK(transport.dim() == 3 && transport.size(0) == E && transport.size(2) == 2, "transport must be (n_edges, orders, 2)");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == E, "bias must be (n_edges,)");
  TORCH_CHECK(inside.scalar_type() == torch::kUInt8 && inside.sizes() == bias.sizes(), "inside must be uint8 (n_edges,)");
  TORCH_CHECK(desc.scalar_type() == torch::kInt64 && desc.dim() == 2 && desc.size(1) == DESC && desc.size(0) <= MAX_F,
              "desc must be int64 (F <= ", MAX_F, ", ", DESC, ")");
  TORCH_CHECK(row_ptr.scalar_type() == torch::kInt64 && row_ptr.dim() == 1 && row_ptr.size(0) == N + 1, "row_ptr must be int64 (n_nodes + 1,)");
  TORCH_CHECK(source.scalar_type() == torch::kInt64 && source.dim() == 1, "source must be int64 (n_edges,)");
  TORCH_CHECK(q.size(1) >= 1, "D must be positive");
}

template <typename T>
Args<T> make_softmax_args(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& transport, const torch::Tensor& bias,
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

// alpha (n_edges, H).
torch::Tensor softmax_forward(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& transport,
                              const torch::Tensor& bias, const torch::Tensor& inside, const torch::Tensor& desc,
                              const torch::Tensor& row_ptr, const torch::Tensor& source, int64_t H, double scale,
                              int64_t deg_max) {
  check_softmax(q, k, transport, bias, inside, desc, row_ptr, source, H);
  auto alpha = torch::empty({source.size(0), H}, q.options());
  if (q.size(0) == 0) return alpha;
  if (q.is_cuda()) {
#ifdef GM_ATTENTION_WITH_CUDA
    softmax_forward_cuda(q, k, transport, bias, inside, desc, row_ptr, source, H, scale, deg_max, alpha);
    return alpha;
#else
    TORCH_CHECK(false, "gm_fused_attention was built without CUDA");
#endif
  }
  AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "gm_fused_softmax_forward", [&] {
    auto a = make_softmax_args<scalar_t>(q, k, transport, bias, inside, desc, row_ptr, source, H, scale);
    a.alpha = alpha.data_ptr<scalar_t>();
    std::vector<scalar_t> scratch((size_t)std::max<int64_t>(max_degree_cpu(row_ptr), 1) * H);
    for (int64_t i = 0; i < a.n_nodes; ++i) node_softmax_forward<scalar_t>(a, i, scratch.data());
  });
  return alpha;
}

// (dq, dk) from d alpha (n_edges, H) and the saved alpha.
std::vector<torch::Tensor> softmax_backward(const torch::Tensor& dalpha, const torch::Tensor& q, const torch::Tensor& k,
                                            const torch::Tensor& transport, const torch::Tensor& bias, const torch::Tensor& inside,
                                            const torch::Tensor& desc, const torch::Tensor& row_ptr, const torch::Tensor& source,
                                            const torch::Tensor& alpha, int64_t H, double scale, int64_t deg_max) {
  check_softmax(q, k, transport, bias, inside, desc, row_ptr, source, H);
  for (const auto* t : {&alpha, &dalpha}) {
    TORCH_CHECK(t->is_contiguous() && t->dim() == 2 && t->size(0) == source.size(0) && t->size(1) == H &&
                t->scalar_type() == q.scalar_type(), "alpha and dalpha must be contiguous (n_edges, H) in q's dtype");
  }
  auto dq = torch::empty_like(q);
  auto dk = torch::zeros_like(q);
  if (q.size(0) == 0) return {dq, dk};
  if (q.is_cuda()) {
#ifdef GM_ATTENTION_WITH_CUDA
    softmax_backward_cuda(dalpha, q, k, transport, bias, inside, desc, row_ptr, source, alpha, H, scale, deg_max, dq, dk);
    return {dq, dk};
#else
    TORCH_CHECK(false, "gm_fused_attention was built without CUDA");
#endif
  }
  AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "gm_fused_softmax_backward", [&] {
    auto a = make_softmax_args<scalar_t>(q, k, transport, bias, inside, desc, row_ptr, source, H, scale);
    a.alpha = alpha.data_ptr<scalar_t>();
    a.dalpha = dalpha.data_ptr<scalar_t>();
    a.dq = dq.data_ptr<scalar_t>();
    a.dk = dk.data_ptr<scalar_t>();
    std::vector<scalar_t> scratch(2 * (size_t)std::max<int64_t>(max_degree_cpu(row_ptr), 1) * H);
    for (int64_t i = 0; i < a.n_nodes; ++i) node_softmax_backward<scalar_t>(a, i, scratch.data());
  });
  return {dq, dk};
}

}  // namespace gm_fused_attention

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv_forward", &gm_fused_attention::conv_forward, "fused depthwise convolution: the aggregated field under given weights");
  m.def("conv_backward", &gm_fused_attention::conv_backward, "fused depthwise convolution: the gradients for v and the per-edge coefficients");
  m.def("softmax_forward", &gm_fused_attention::softmax_forward, "fused scores and softmax: the attention weights from q and k");
  m.def("softmax_backward", &gm_fused_attention::softmax_backward, "fused scores and softmax: the gradients for q and k from d alpha");
}
