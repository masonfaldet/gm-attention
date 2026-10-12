"""The fused pair terms are the Python pair loop: same values, same
gradients, on every case the loop has (CPU driver, float64)."""

from __future__ import annotations

import pytest
import torch

from gm_attention.discretization.edge_geometry import EdgeGeometry
from gm_attention.nn import fused_pairs
from gm_attention.nn.kernels import SO2SteerableKernel
from gm_attention.representations import so2
from gm_attention.representations.field_type import SO2FeatureField, SO2FieldType



@pytest.fixture(scope="module", autouse=True)
def _built_extension() -> None:
    """The CPU driver, built on first use; skip if no compiler is at hand."""
    try:
        fused_pairs.extension()
    except Exception as error:  # noqa: BLE001 - a missing toolchain, not a test failure
        pytest.skip(f"fused pairs extension did not build here: {error}")


def _edges(n_edges: int, seed: int, max_frequency: int = 4) -> EdgeGeometry:
    generator = torch.Generator().manual_seed(seed)
    distances = torch.rand(n_edges, dtype=torch.float64, generator=generator) * 1.9 + 0.05
    distances[0] = 0.0  # a self loop at r = 0, where the radial powers regularize
    polar = (torch.rand(n_edges, dtype=torch.float64, generator=generator) * 2 - 1) * 3.1
    transport = (torch.rand(n_edges, dtype=torch.float64, generator=generator) * 2 - 1) * 3.1
    target = torch.zeros(n_edges, dtype=torch.long)
    source = torch.ones(n_edges, dtype=torch.long)
    source[0] = 0
    log_coordinates = distances.unsqueeze(-1) * torch.stack((torch.cos(polar), torch.sin(polar)), dim=-1)
    return EdgeGeometry(
        edge_index=torch.stack((target, source)),
        row_ptr=torch.tensor([0, n_edges, n_edges], dtype=torch.long),
        log_coordinates=log_coordinates, distances=distances, polar_angles=polar, transport_angles=transport,
        tangent_quadrature_weights=torch.ones(n_edges, dtype=torch.float64),
        angular_phases=so2.phase_powers(polar, max_frequency),
        transport_phases=so2.phase_powers(transport, max_frequency),
    )


def _kernel(field_in: dict[int, int], field_out: dict[int, int], *, bias: bool = True, hidden: tuple[int, ...] = (16,), seed: int = 0) -> SO2SteerableKernel:
    torch.manual_seed(seed)
    return SO2SteerableKernel(
        SO2FieldType(field_in), SO2FieldType(field_out), cutoff=2.0, n_radial_basis=5,
        radial_hidden_features=hidden, bias=bias, dtype=torch.float64,
    )


def _field(field_type: SO2FieldType, n: int, seed: int) -> SO2FeatureField:
    generator = torch.Generator().manual_seed(seed)
    return SO2FeatureField.from_flat(field_type, torch.randn(n, field_type.dimension, dtype=torch.float64, generator=generator))


def _run(kernel: SO2SteerableKernel, x: SO2FeatureField, edges: EdgeGeometry, mode: str, monkeypatch: pytest.MonkeyPatch) -> tuple[torch.Tensor, list[torch.Tensor]]:
    monkeypatch.setattr(fused_pairs, "MODE", mode)
    flat = x.to_flat().detach().requires_grad_(True)
    out = kernel._forward_general(SO2FeatureField.from_flat(x.field_type, flat), edges)
    # A loss over the coefficients as they are, not over to_flat(): the
    # flat view's backward hands view_as_complex odd-offset slices.
    loss = sum(
        (out[f] * (torch.arange(1, out[f].numel() + 1, dtype=torch.float64).reshape(out[f].shape) / out[f].numel())).sum()
        for f in out.field_type.frequencies
    )
    grads = torch.autograd.grad(loss, [flat, *kernel.parameters()])
    return out.to_flat().detach(), [g.detach() for g in grads]


CASES = [
    ({0: 2, 1: 2, 2: 1}, {0: 1, 1: 2, 2: 2}, True),   # every case: (0,0), (0,m), (n,0), (n,m) with m-n < 0 and > 0
    ({0: 2, 1: 2, 2: 1}, {0: 1, 1: 2, 2: 2}, False),  # no bias
    ({0: 3}, {0: 2}, True),                            # real head only
    ({1: 2, 2: 1}, {1: 1, 2: 2}, True),                # complex head only, no real input
    ({0: 2, 1: 1}, {1: 2}, True),                      # real input into complex output only
    ({1: 2}, {0: 2}, True),                            # complex input into the real output only
    ({1: 16}, {1: 16}, True),                          # the largest block the kernels take
]


@pytest.mark.parametrize("field_in, field_out, bias", CASES)
def test_fused_terms_match_the_python_pair_loop_in_values_and_gradients(field_in, field_out, bias, monkeypatch) -> None:
    kernel = _kernel(field_in, field_out, bias=bias)
    edges = _edges(37, seed=1)
    x = _field(kernel.field_type_in, edges.n_edges, seed=2)
    y_plain, g_plain = _run(kernel, x, edges, "0", monkeypatch)
    y_fused, g_fused = _run(kernel, x, edges, "cpu", monkeypatch)
    torch.testing.assert_close(y_fused, y_plain, atol=1e-12, rtol=0.0)
    for a, b in zip(g_fused, g_plain, strict=True):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=0.0)


def test_the_fused_backward_passes_gradcheck_on_its_own(monkeypatch) -> None:
    monkeypatch.setattr(fused_pairs, "MODE", "cpu")
    kernel = _kernel({0: 1, 1: 1, 2: 1}, {0: 1, 1: 1, 2: 1}, hidden=(4,))
    edges = _edges(6, seed=3)
    x = _field(kernel.field_type_in, edges.n_edges, seed=4)

    def f(flat: torch.Tensor, *params: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # A tuple, not to_flat(): gradcheck's one-hot output gradients are
        # narrow views whose odd storage offsets view_as_complex refuses.
        out = kernel._forward_general(SO2FeatureField.from_flat(kernel.field_type_in, flat), edges)
        return tuple(out[frequency] for frequency in out.field_type.frequencies)

    flat = x.to_flat().requires_grad_(True)
    assert torch.autograd.gradcheck(f, (flat, *kernel.parameters()), atol=1e-7, rtol=1e-6)


def test_a_narrower_trunk_and_a_short_edge_set_take_the_fused_path_too(monkeypatch) -> None:
    kernel = _kernel({0: 2, 1: 1}, {0: 1, 1: 1}, hidden=())  # trunk width = the radial basis, 5
    edges = _edges(1, seed=5)  # one edge, the self loop
    x = _field(kernel.field_type_in, edges.n_edges, seed=6)
    y_plain, g_plain = _run(kernel, x, edges, "0", monkeypatch)
    y_fused, g_fused = _run(kernel, x, edges, "cpu", monkeypatch)
    torch.testing.assert_close(y_fused, y_plain, atol=1e-12, rtol=0.0)
    for a, b in zip(g_fused, g_plain, strict=True):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=0.0)


def test_oversize_blocks_and_the_off_mode_fall_back_to_the_python_loop(monkeypatch) -> None:
    kernel = _kernel({1: 17}, {1: 1})
    edges = _edges(3, seed=7)
    x = _field(kernel.field_type_in, edges.n_edges, seed=8)
    monkeypatch.setattr(fused_pairs, "MODE", "cpu")
    assert not fused_pairs.enabled(kernel, x, edges)
    assert not fused_pairs.enabled(_kernel({1: 5}, {1: 2}), x, edges)  # a side the kernels are not instantiated for
    monkeypatch.setattr(fused_pairs, "MODE", "0")
    assert not fused_pairs.enabled(_kernel({1: 2}, {1: 2}), x, edges)
    monkeypatch.setattr(fused_pairs, "MODE", "1")  # CUDA only: a CPU tensor stays on the Python loop
    assert not fused_pairs.enabled(_kernel({1: 2}, {1: 2}), x, edges)


def test_the_plan_lists_the_loops_terms_in_its_order() -> None:
    kernel = _kernel({0: 2, 1: 2, 2: 1}, {0: 1, 1: 2, 2: 2})
    terms = fused_pairs.plan(kernel)
    assert [(t.n, t.m, t.role) for t in terms] == [
        (0, 0, "real"), (1, 0, "linear"), (2, 0, "linear"),
        (0, 1, "linear"), (1, 1, "linear"), (1, 1, "antilinear"), (2, 1, "linear"), (2, 1, "antilinear"),
        (0, 2, "linear"), (1, 2, "linear"), (1, 2, "antilinear"), (2, 2, "linear"), (2, 2, "antilinear"),
    ]
    by = {(t.n, t.m, t.role): t for t in terms}
    assert by[(2, 1, "linear")].pre_order == -1 and by[(2, 1, "antilinear")].pre_order == 3
    assert by[(0, 2, "linear")].post_order == 2 and by[(1, 0, "linear")].pre_order == 1
    assert by[(0, 0, "real")].flags == fused_pairs.U_REAL | fused_pairs.OUT_REAL
