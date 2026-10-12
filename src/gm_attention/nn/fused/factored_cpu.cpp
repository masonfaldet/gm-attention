// The factored steerable kernel's CPU driver and the extension's binding
// (fused_factored.py): `forward` (the rows), `aggregate` (the weighted
// segment sum of the rows), `backward` (from the rows' gradient) and
// `backward_aggregated` (from the targets' gradient, through the target
// index; factored.h has the arithmetic). Built with -DGM_FACTORED_WITH_CUDA
// alongside factored_cuda.cu, all also serve CUDA tensors.
#include <torch/extension.h>
#include "factored.h"

namespace gm_factored {

#define GM_CHECK(cond, msg) TORCH_CHECK(cond, "gm_factored: ", msg)

template <typename T>
Args<T> make_args(const torch::Tensor& h, const torch::Tensor& zc, const torch::Tensor& zr,
                  const torch::Tensor& angular, const torch::Tensor& transport, const torch::Tensor& rho,
                  const torch::Tensor& desc, const torch::Tensor& source, int64_t d_out) {
  Args<T> a{};
  a.n_edges = h.size(0);
  a.k = (int)h.size(1) + 1;
  a.wc = (int)zc.size(2);
  a.wr = (int)zr.size(2);
  a.n_f = (int)angular.size(1);
  a.n_blocks = (int)desc.size(0);
  a.d_out = (int)d_out;
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

static void check_common(const torch::Tensor& h, const torch::Tensor& zc, const torch::Tensor& zr,
                         const torch::Tensor& angular, const torch::Tensor& transport, const torch::Tensor& rho,
                         const torch::Tensor& desc, const torch::Tensor& source, int64_t d_out) {
  GM_CHECK(h.dim() == 2 && h.is_contiguous(), "h must be a contiguous (n_edges, k - 1) tensor");
  GM_CHECK(zc.dim() == 4 && zc.size(3) == 2 && zc.is_contiguous(), "zc must be a contiguous (n_nodes, k, wc, 2) tensor");
  GM_CHECK(zr.dim() == 3 && zr.is_contiguous(), "zr must be a contiguous (n_nodes, k, wr) tensor");
  GM_CHECK(zc.size(1) == h.size(1) + 1 && zr.size(1) == h.size(1) + 1, "zc and zr must have k = h's width + 1 (the constant feature)");
  GM_CHECK(zc.size(0) == zr.size(0), "zc and zr must have the same node count");
  GM_CHECK(angular.dim() == 3 && angular.size(2) == 2 && angular.is_contiguous() && angular.size(0) == h.size(0), "angular must be a contiguous (n_edges, n_f, 2) tensor");
  GM_CHECK(transport.sizes() == angular.sizes() && transport.is_contiguous(), "transport must match angular");
  GM_CHECK(rho.dim() == 1 && rho.is_contiguous() && rho.size(0) == h.size(0), "rho must be a contiguous (n_edges,) tensor");
  GM_CHECK(desc.dim() == 2 && desc.size(1) == DESC && desc.scalar_type() == torch::kInt32 && desc.is_contiguous(), "desc must be a contiguous int32 (n_blocks, 8) tensor");
  GM_CHECK(source.dim() == 1 && source.size(0) == h.size(0) && source.scalar_type() == torch::kInt64 && source.is_contiguous(), "source must be a contiguous int64 (n_edges,) tensor");
  GM_CHECK(h.size(1) + 1 <= MAX_K, "k exceeds MAX_K");
  GM_CHECK(d_out >= 1 && d_out <= MAX_OUT, "d_out must be in [1, MAX_OUT]");
  GM_CHECK(2 * zc.size(2) + zr.size(2) <= MAX_C, "2 wc + wr exceeds MAX_C");
  for (const auto& t : {zc, zr, angular, transport, rho}) {
    GM_CHECK(t.scalar_type() == h.scalar_type(), "every floating tensor must share h's dtype");
    GM_CHECK(t.device() == h.device(), "every tensor must be on h's device");
  }
  GM_CHECK(desc.device() == h.device() && source.device() == h.device(), "desc and source must be on h's device");
}

#ifdef GM_FACTORED_WITH_CUDA
void forward_cuda(const torch::Tensor& h, const torch::Tensor& zc, const torch::Tensor& zr,
                  const torch::Tensor& angular, const torch::Tensor& transport, const torch::Tensor& rho,
                  const torch::Tensor& desc, const torch::Tensor& source, const torch::Tensor& order,
                  const torch::Tensor& group_ptr, const torch::Tensor& group_source, torch::Tensor& out, int max_bm);
void backward_cuda(const torch::Tensor& g, const torch::Tensor& target, const torch::Tensor& weights, const torch::Tensor& rows,
                   const torch::Tensor& h, const torch::Tensor& zc, const torch::Tensor& zr,
                   const torch::Tensor& angular, const torch::Tensor& transport, const torch::Tensor& rho,
                   const torch::Tensor& desc, const torch::Tensor& source, const torch::Tensor& order,
                   const torch::Tensor& group_ptr, const torch::Tensor& group_source,
                   torch::Tensor& dh, torch::Tensor& dzc, torch::Tensor& dzr, torch::Tensor& dweights);
void aggregate_cuda(const torch::Tensor& rows, const torch::Tensor& weights, const torch::Tensor& row_ptr, torch::Tensor& out);
#endif

// The rows (n_edges, d_out). max_bm: the widest output multiplicity among
// the blocks (fused_factored.plan), which picks the routine's tier; the CPU
// driver checks it against desc.
torch::Tensor forward(const torch::Tensor& h, const torch::Tensor& zc, const torch::Tensor& zr,
                      const torch::Tensor& angular, const torch::Tensor& transport, const torch::Tensor& rho,
                      const torch::Tensor& desc, const torch::Tensor& source, const torch::Tensor& order,
                      const torch::Tensor& group_ptr, const torch::Tensor& group_source, int64_t d_out, int64_t max_bm) {
  check_common(h, zc, zr, angular, transport, rho, desc, source, d_out);
  auto out = torch::empty({h.size(0), d_out}, h.options());
  GM_CHECK(max_bm >= 1 && max_bm <= MAX_BM, "max_bm must be in [1, MAX_BM]");
  if (h.size(0) == 0) return out;
  if (h.is_cuda()) {
#ifdef GM_FACTORED_WITH_CUDA
    forward_cuda(h, zc, zr, angular, transport, rho, desc, source, order, group_ptr, group_source, out, (int)max_bm);
    return out;
#else
    TORCH_CHECK(false, "gm_factored was built without CUDA");
#endif
  }
  {
    const auto rows = desc.accessor<int32_t, 2>();
    for (int64_t b = 0; b < desc.size(0); ++b) GM_CHECK(rows[b][B_M] <= max_bm, "a block's multiplicity exceeds max_bm");
  }
  AT_DISPATCH_FLOATING_TYPES(h.scalar_type(), "gm_factored_forward", [&] {
    auto a = make_args<scalar_t>(h, zc, zr, angular, transport, rho, desc, source, d_out);
    a.out = out.data_ptr<scalar_t>();
    const int64_t zc_stride = (int64_t)a.k * a.wc * 2;
    const int64_t zr_stride = (int64_t)a.k * a.wr;
    auto message = max_bm <= NARROW_BM ? &edge_message<scalar_t, NARROW_BM> : &edge_message<scalar_t, MAX_BM>;
    for (int64_t e = 0; e < a.n_edges; ++e) {
      const int64_t j = a.source[e];
      message(a, e, a.zc + j * zc_stride, a.zr + j * zr_stride);
    }
  });
  return out;
}

// out (n_targets, d_out): each target's weighted sum of its rows, the edges
// [row_ptr[i], row_ptr[i + 1]) in order (n_targets = row_ptr's length - 1).
torch::Tensor aggregate(const torch::Tensor& rows, const torch::Tensor& weights, const torch::Tensor& row_ptr) {
  GM_CHECK(rows.dim() == 2 && rows.is_contiguous() && rows.size(1) >= 1 && rows.size(1) <= MAX_OUT, "rows must be a contiguous (n_edges, d_out <= MAX_OUT) tensor");
  GM_CHECK(weights.dim() == 1 && weights.is_contiguous() && weights.size(0) == rows.size(0) && weights.scalar_type() == rows.scalar_type(),
           "weights must be a contiguous (n_edges,) tensor in the rows' dtype");
  GM_CHECK(row_ptr.dim() == 1 && row_ptr.is_contiguous() && row_ptr.size(0) >= 1 && row_ptr.scalar_type() == torch::kInt64, "row_ptr must be a contiguous int64 (n_targets + 1,) tensor");
  GM_CHECK(weights.device() == rows.device() && row_ptr.device() == rows.device(), "every tensor must be on the rows' device");
  const int64_t n_targets = row_ptr.size(0) - 1;
  auto out = torch::empty({n_targets, rows.size(1)}, rows.options());
  if (n_targets == 0) return out;
  if (rows.is_cuda()) {
#ifdef GM_FACTORED_WITH_CUDA
    aggregate_cuda(rows, weights, row_ptr, out);
    return out;
#else
    TORCH_CHECK(false, "gm_factored was built without CUDA");
#endif
  }
  AT_DISPATCH_FLOATING_TYPES(rows.scalar_type(), "gm_factored_aggregate", [&] {
    const auto* r = rows.data_ptr<scalar_t>();
    const auto* w = weights.data_ptr<scalar_t>();
    const auto* p = row_ptr.data_ptr<int64_t>();
    auto* o = out.data_ptr<scalar_t>();
    for (int64_t i = 0; i < n_targets; ++i) aggregate_node<scalar_t>(r, w, p, (int)rows.size(1), i, o);
  });
  return out;
}

static std::vector<torch::Tensor> backward_impl(const torch::Tensor& g, const torch::Tensor& target, const torch::Tensor& weights,
                                                const torch::Tensor& rows, const torch::Tensor& h, const torch::Tensor& zc,
                                                const torch::Tensor& zr, const torch::Tensor& angular, const torch::Tensor& transport,
                                                const torch::Tensor& rho, const torch::Tensor& desc, const torch::Tensor& source,
                                                const torch::Tensor& order, const torch::Tensor& group_ptr,
                                                const torch::Tensor& group_source) {
  GM_CHECK(g.dim() == 2 && g.is_contiguous() && g.scalar_type() == h.scalar_type() && g.device() == h.device(), "g must be a contiguous (rows, d_out) tensor in h's dtype");
  check_common(h, zc, zr, angular, transport, rho, desc, source, g.size(1));
  const bool aggregated = target.numel() > 0;
  if (aggregated) {
    GM_CHECK(target.dim() == 1 && target.size(0) == h.size(0) && target.scalar_type() == torch::kInt64 && target.is_contiguous() && target.device() == h.device(),
             "target must be a contiguous int64 (n_edges,) tensor");
    GM_CHECK(weights.dim() == 1 && weights.size(0) == h.size(0) && weights.is_contiguous() && weights.scalar_type() == h.scalar_type() && weights.device() == h.device(),
             "weights must be a contiguous (n_edges,) tensor in h's dtype");
    GM_CHECK(rows.dim() == 2 && rows.size(0) == h.size(0) && rows.size(1) == g.size(1) && rows.is_contiguous() && rows.scalar_type() == h.scalar_type() && rows.device() == h.device(),
             "rows must be a contiguous (n_edges, d_out) tensor in h's dtype");
  } else {
    GM_CHECK(g.size(0) == h.size(0), "g must have one row per edge");
  }
  auto dh = torch::empty_like(h);
  auto dzc = torch::zeros_like(zc);
  auto dzr = torch::zeros_like(zr);
  auto dweights = aggregated ? torch::empty_like(weights) : torch::Tensor();
  if (h.size(0) == 0) return {dh, dzc, dzr, dweights};
  if (h.is_cuda()) {
#ifdef GM_FACTORED_WITH_CUDA
    backward_cuda(g, target, weights, rows, h, zc, zr, angular, transport, rho, desc, source, order, group_ptr, group_source, dh, dzc, dzr, dweights);
    return {dh, dzc, dzr, dweights};
#else
    TORCH_CHECK(false, "gm_factored was built without CUDA");
#endif
  }
  AT_DISPATCH_FLOATING_TYPES(h.scalar_type(), "gm_factored_backward", [&] {
    auto a = make_args<scalar_t>(h, zc, zr, angular, transport, rho, desc, source, g.size(1));
    a.g = g.data_ptr<scalar_t>();
    a.dh = dh.data_ptr<scalar_t>();
    a.dzc = dzc.data_ptr<scalar_t>();
    a.dzr = dzr.data_ptr<scalar_t>();
    if (aggregated) {
      a.target = target.data_ptr<int64_t>();
      a.weights = weights.data_ptr<scalar_t>();
      a.rows = rows.data_ptr<scalar_t>();
      a.dweights = dweights.data_ptr<scalar_t>();
    }
    const int64_t zc_stride = (int64_t)a.k * a.wc * 2;
    const int64_t zr_stride = (int64_t)a.k * a.wr;
    auto edge = 2 * a.wc + a.wr <= NARROW_C ? &edge_backward<scalar_t, NARROW_C> : &edge_backward<scalar_t, MAX_C>;
    for (int64_t e = 0; e < a.n_edges; ++e) {
      const int64_t j = a.source[e];
      edge(a, e, a.zc + j * zc_stride, a.zr + j * zr_stride, a.dzc + j * zc_stride, a.dzr + j * zr_stride);
    }
  });
  return {dh, dzc, dzr, dweights};
}

// The gradients of `forward`'s rows with respect to h, zc and zr, given the
// rows' gradient g (n_edges, d_out); the phases and r / cutoff are the
// geometry's and get none. dzc and dzr are zero for a node without edges.
std::vector<torch::Tensor> backward(const torch::Tensor& g, const torch::Tensor& h, const torch::Tensor& zc, const torch::Tensor& zr,
                                    const torch::Tensor& angular, const torch::Tensor& transport, const torch::Tensor& rho,
                                    const torch::Tensor& desc, const torch::Tensor& source, const torch::Tensor& order,
                                    const torch::Tensor& group_ptr, const torch::Tensor& group_source) {
  auto result = backward_impl(g, torch::Tensor(), torch::Tensor(), torch::Tensor(), h, zc, zr, angular, transport, rho, desc, source,
                              order, group_ptr, group_source);
  return {result[0], result[1], result[2]};
}

// The gradients of `aggregate(forward(...), weights, row_ptr)` with respect
// to h, zc, zr and the weights, given the targets' gradient g (n_targets,
// d_out), the edges' targets and the forward's rows.
std::vector<torch::Tensor> backward_aggregated(const torch::Tensor& g, const torch::Tensor& target, const torch::Tensor& weights,
                                               const torch::Tensor& rows, const torch::Tensor& h, const torch::Tensor& zc,
                                               const torch::Tensor& zr, const torch::Tensor& angular, const torch::Tensor& transport,
                                               const torch::Tensor& rho, const torch::Tensor& desc, const torch::Tensor& source,
                                               const torch::Tensor& order, const torch::Tensor& group_ptr,
                                               const torch::Tensor& group_source) {
  if (h.size(0) == 0) return {torch::empty_like(h), torch::zeros_like(zc), torch::zeros_like(zr), torch::empty_like(weights)};
  GM_CHECK(target.numel() == h.size(0), "target must list every edge");
  return backward_impl(g, target, weights, rows, h, zc, zr, angular, transport, rho, desc, source, order, group_ptr, group_source);
}

}  // namespace gm_factored

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gm_factored::forward, "factored steerable kernel: messages per edge in the flat output layout");
  m.def("aggregate", &gm_factored::aggregate, "factored steerable kernel: each target's weighted sum of its rows");
  m.def("backward", &gm_factored::backward, "factored steerable kernel: the gradients for h, zc and zr from the rows' gradient");
  m.def("backward_aggregated", &gm_factored::backward_aggregated,
        "factored steerable kernel: the gradients for h, zc, zr and the weights from the targets' gradient");
#ifdef GM_FACTORED_WITH_CUDA
  m.attr("with_cuda") = true;
#else
  m.attr("with_cuda") = false;
#endif
}
