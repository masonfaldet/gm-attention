"""The block contraction's own backward: the same values and gradients as
autograd's map, without the conjugated copy of the block."""

from __future__ import annotations

import torch

from gm_attention.nn.kernels import _BlockApply


def _plain(matrices: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    return (matrices * vectors.unsqueeze(-2)).sum(-1)


def test_block_apply_gradients_match_autograd_for_real_and_complex_blocks() -> None:
    generator = torch.Generator().manual_seed(0)
    for dtype in (torch.float64, torch.complex128):
        m = torch.randn(50, 5, 3, dtype=dtype, generator=generator).requires_grad_(True)
        v = torch.randn(50, 3, dtype=dtype, generator=generator).requires_grad_(True)
        g = torch.randn(50, 5, dtype=dtype, generator=generator)
        ours = _BlockApply.apply(m, v)
        theirs = _plain(m.detach().requires_grad_(True), v.detach().requires_grad_(True))
        torch.testing.assert_close(ours, theirs, atol=1e-13, rtol=0.0)
        (grad_m, grad_v) = torch.autograd.grad((ours * g.conj()).real.sum(), (m, v))
        m2 = m.detach().requires_grad_(True); v2 = v.detach().requires_grad_(True)
        (ref_m, ref_v) = torch.autograd.grad((_plain(m2, v2) * g.conj()).real.sum(), (m2, v2))
        torch.testing.assert_close(grad_m, ref_m, atol=1e-13, rtol=0.0)
        torch.testing.assert_close(grad_v, ref_v, atol=1e-13, rtol=0.0)
        assert torch.autograd.gradcheck(lambda a, b: _BlockApply.apply(a, b), (m, v))


def test_block_apply_handles_a_vector_that_needs_no_gradient() -> None:
    m = torch.randn(8, 4, 2, dtype=torch.complex128, requires_grad=True)
    v = torch.randn(8, 2, dtype=torch.complex128)
    _BlockApply.apply(m, v).real.sum().backward()
    assert m.grad is not None and v.grad is None
