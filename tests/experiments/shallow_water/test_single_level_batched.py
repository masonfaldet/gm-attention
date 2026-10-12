"""The single-level arm's batched heads are the same function as its sequential ones."""

from __future__ import annotations

import pytest
import torch

from experiments.shallow_water.models import CONDITIONED_FIELD_TYPE, GmAttentionSingleLevel
from gm_attention.discretization.discretization import SphericalDiscretization
from gm_attention.discretization.frames import build_tangent_frames
from gm_attention.discretization.grids.icosphere import build_icosphere_grid
from gm_attention.representations.field_type import SO2FeatureField


def _model(batched: bool) -> GmAttentionSingleLevel:
    torch.manual_seed(0)
    return GmAttentionSingleLevel(
        base_field_type={0: 1, 1: 1, 2: 1}, multiplicity=4, num_layers=2, num_heads=2,
        deep_cutoff=0.6, conv_cutoff=0.5, window="hard", nonlinearity="scalar_gate_double_sigmoid",
        chunk_size=40, dtype=torch.float64, attention_batched=batched,
    )


def test_batched_heads_match_sequential_heads_in_values_and_gradients() -> None:
    grid = build_icosphere_grid(2, dtype=torch.float64)
    discretization = SphericalDiscretization.from_grid(grid, coverage_radius=0.6, frames=build_tangent_frames(grid.nodes))
    discretization.prepare_windows([0.6, 0.5])
    generator = torch.Generator().manual_seed(1)
    x = SO2FeatureField(CONDITIONED_FIELD_TYPE, {  # nonzero frequencies are real (n, m, 2) pairs
        0: torch.randn(grid.n_nodes, 2, dtype=torch.float64, generator=generator),
        1: torch.randn(grid.n_nodes, 1, 2, dtype=torch.float64, generator=generator),
    })
    sequential, batched = _model(False), _model(True)
    batched.load_state_dict(sequential.state_dict())  # the batched path consumes the heads' own parameters
    outputs = []
    for model in (sequential, batched):
        model.train()
        y = model(x, discretization)
        loss = sum((y[f] ** 2).sum() for f in y.field_type.frequencies)
        loss.backward()
        outputs.append(y)
    for f in outputs[0].field_type.frequencies:
        torch.testing.assert_close(outputs[1][f], outputs[0][f], atol=1e-11, rtol=0.0)
    for (name, p), (_, q) in zip(sequential.named_parameters(), batched.named_parameters(), strict=True):
        assert p.grad is not None and q.grad is not None, name
        torch.testing.assert_close(q.grad, p.grad, atol=1e-10, rtol=0.0, msg=name)


def test_the_fused_pair_terms_leave_the_single_level_arm_unchanged(monkeypatch) -> None:
    """The whole arm (chunked, checkpointed) through the fused CPU driver
    against the Python pair loop: values and every gradient."""
    from gm_attention.nn import fused_pairs

    try:
        fused_pairs.extension()
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"fused pairs extension did not build here: {error}")
    grid = build_icosphere_grid(2, dtype=torch.float64)
    discretization = SphericalDiscretization.from_grid(grid, coverage_radius=0.6, frames=build_tangent_frames(grid.nodes))
    discretization.prepare_windows([0.6, 0.5])
    generator = torch.Generator().manual_seed(2)
    x = SO2FeatureField(CONDITIONED_FIELD_TYPE, {
        0: torch.randn(grid.n_nodes, 2, dtype=torch.float64, generator=generator),
        1: torch.randn(grid.n_nodes, 1, 2, dtype=torch.float64, generator=generator),
    })
    results = []
    for mode in ("0", "cpu"):
        monkeypatch.setattr(fused_pairs, "MODE", mode)
        model = _model(False).train()
        y = model(x, discretization)
        loss = sum((y[f] ** 2).sum() for f in y.field_type.frequencies)
        loss.backward()
        results.append((y, [p.grad.detach().clone() for p in model.parameters()]))
    (y_plain, g_plain), (y_fused, g_fused) = results
    for f in y_plain.field_type.frequencies:
        torch.testing.assert_close(y_fused[f], y_plain[f], atol=1e-11, rtol=0.0)
    for a, b in zip(g_fused, g_plain, strict=True):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=0.0)
