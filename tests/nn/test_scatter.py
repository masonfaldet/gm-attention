"""Tests for gm_attention.nn._scatter.

Covers the two behaviors that came with switching scatter_add_to_nodes and
scatter_max_to_nodes from a COO-style scatter (index_add_ / scatter_reduce_
keyed on target_index) to torch.segment_reduce against row_ptr: the free
validation that row_ptr is well-formed, and the -inf-for-an-empty-segment
difference in the max case (which scatter_add_to_nodes's sum does not
share -- an empty segment sums to 0, same as index_add_ would give).
"""

from __future__ import annotations

import pytest
import torch

from gm_attention.nn._scatter import (
    gather_to_edges,
    scatter_add_to_nodes,
    scatter_max_to_nodes,
)
from gm_attention.representations.field_type import SO2FeatureField, SO2FieldType

# --------------------------------------------------------------------------
# gather_to_edges
# --------------------------------------------------------------------------


def test_gather_to_edges_indexes_every_frequency_consistently() -> None:
    field_type = SO2FieldType({0: 2, 1: 1})
    field = SO2FeatureField.from_flat(
        field_type, torch.arange(12, dtype=torch.float64).reshape(3, 4)
    )
    index = torch.tensor([2, 0, 0, 1])

    gathered = gather_to_edges(field, index)

    assert torch.equal(gathered[0], field[0][index])
    assert torch.equal(gathered[1], field[1][index])
    assert gathered.batch_shape == (4,)


# --------------------------------------------------------------------------
# scatter_add_to_nodes
# --------------------------------------------------------------------------


def test_scatter_add_matches_index_add_reference() -> None:
    row_ptr = torch.tensor([0, 2, 5, 6])  # node degrees: 2, 3, 1
    target_index = torch.tensor([0, 0, 1, 1, 1, 2])
    values = torch.randn(6, 3, 2, dtype=torch.float64)

    result = scatter_add_to_nodes(values, row_ptr)

    expected = values.new_zeros(3, 3, 2)
    expected.index_add_(0, target_index, values)

    assert torch.allclose(result, expected, atol=1e-12)


def test_scatter_add_gives_zero_for_a_node_with_no_incident_edges() -> None:
    row_ptr = torch.tensor([0, 2, 2, 3])  # node 1 has zero edges
    values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

    result = scatter_add_to_nodes(values, row_ptr)

    assert torch.equal(result, torch.tensor([3.0, 0.0, 3.0], dtype=torch.float64))


def test_scatter_add_is_differentiable() -> None:
    row_ptr = torch.tensor([0, 2, 3])
    values = torch.randn(3, dtype=torch.float64, requires_grad=True)

    scatter_add_to_nodes(values, row_ptr).sum().backward()

    assert values.grad is not None
    assert torch.allclose(values.grad, torch.ones(3, dtype=torch.float64))


def test_scatter_add_rejects_a_non_monotonic_row_ptr() -> None:
    """torch.segment_reduce validates offsets by default -- a correctness
    check the COO-style implementation this replaced never performed."""
    malformed_row_ptr = torch.tensor([0, 3, 2, 4])
    values = torch.randn(4, dtype=torch.float64)

    with pytest.raises(RuntimeError):
        scatter_add_to_nodes(values, malformed_row_ptr)


# --------------------------------------------------------------------------
# scatter_max_to_nodes
# --------------------------------------------------------------------------


def test_scatter_max_matches_scatter_reduce_amax_reference() -> None:
    row_ptr = torch.tensor([0, 2, 5, 6])
    target_index = torch.tensor([0, 0, 1, 1, 1, 2])
    values = torch.randn(6, dtype=torch.float64)

    result = scatter_max_to_nodes(values, row_ptr)

    expected = values.new_zeros(3)
    expected.scatter_reduce_(0, target_index, values, reduce="amax", include_self=False)

    assert torch.allclose(result, expected, atol=1e-12)


def test_scatter_max_gives_negative_infinity_for_a_node_with_no_incident_edges() -> (
    None
):
    """Unlike scatter_add_to_nodes's 0 -- documented explicitly because every
    consumer in this codebase only ever reads this through a per-edge gather
    (result[target_index]), which never touches an empty-segment node, so
    this never surfaces downstream. It matters if scatter_max_to_nodes is
    ever read directly, as here."""
    row_ptr = torch.tensor([0, 2, 2, 3])
    values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

    result = scatter_max_to_nodes(values, row_ptr)

    assert result[1] == float("-inf")
    assert torch.equal(result[[0, 2]], torch.tensor([2.0, 3.0], dtype=torch.float64))


def test_scatter_max_gradient_flows_only_to_the_argmax_element() -> None:
    row_ptr = torch.tensor([0, 3])
    values = torch.tensor([1.0, 5.0, 2.0], dtype=torch.float64, requires_grad=True)

    scatter_max_to_nodes(values, row_ptr).sum().backward()

    assert torch.equal(values.grad, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64))


def test_scatter_max_rejects_a_non_monotonic_row_ptr() -> None:
    malformed_row_ptr = torch.tensor([0, 3, 2, 4])
    values = torch.randn(4, dtype=torch.float64)

    with pytest.raises(RuntimeError):
        scatter_max_to_nodes(values, malformed_row_ptr)
