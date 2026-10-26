"""Tests for gm_attention.nn.stencil_plan: the bucketed slot tables of a tensor-grid stencil.

The plan is bookkeeping, so the checks are structural: every row lands in
exactly one bucket, a bucket's width is its largest degree, the padded
positions point at the phantom slot and at the target node, the real
positions reproduce the stencil's own ``source_nodes`` at every target
longitude, and the partition into buckets is the one with the least padding
(checked against brute force on random small inputs).
"""

from __future__ import annotations

import itertools
import random

import pytest
import torch

from gm_attention.discretization.stencil import (
    TensorGridStencil,
    build_tensor_grid_discretization,
    build_tensor_grid_stencil,
)
from gm_attention.nn.stencil_plan import (
    StencilBucket,
    StencilPlan,
    StencilPlans,
    build_stencil_plan,
    build_stencil_plans,
    build_tensor_grid_geometry,
    partition_degrees,
    split_edges_and_plan,
)

SHAPE = (10, 20)
CUTOFF = 0.6


def _stencil(dtype: torch.dtype = torch.float64, shape: tuple[int, int] = SHAPE, cutoff: float = CUTOFF) -> TensorGridStencil:
    discretization = build_tensor_grid_discretization("legendre-gauss", *shape, coverage_radius=cutoff, dtype=dtype)
    return build_tensor_grid_stencil(discretization, cutoff, n_latitude=shape[0], n_longitude=shape[1])


def _padded_cost(degrees: list[int], groups: list[list[int]]) -> int:
    return sum(len(group) * max(degrees[i] for i in group) for group in groups)


def _brute_force_cost(degrees: list[int], max_buckets: int) -> int:
    """The least padded cost over every cut of the degree-sorted items into at most ``max_buckets`` runs."""
    n = len(degrees)
    order = sorted(range(n), key=lambda i: (degrees[i], i))
    best = None
    for cuts in range(min(max_buckets, n)):
        for positions in itertools.combinations(range(1, n), cuts):
            bounds = [0, *positions, n]
            cost = sum((bounds[k + 1] - bounds[k]) * degrees[order[bounds[k + 1] - 1]] for k in range(len(bounds) - 1))
            best = cost if best is None else min(best, cost)
    assert best is not None
    return best


# --------------------------------------------------------------------------
# The partition


def test_partition_has_the_least_padding() -> None:
    generator = random.Random(0)
    for _ in range(300):
        n = generator.randint(1, 7)
        degrees = [generator.randint(0, 9) for _ in range(n)]
        max_buckets = generator.randint(1, 4)
        groups = partition_degrees(degrees, max_buckets)
        assert len(groups) <= max_buckets
        assert sorted(i for group in groups for i in group) == list(range(n))
        assert _padded_cost(degrees, groups) == _brute_force_cost(degrees, max_buckets)


def test_partition_groups_are_sorted_by_degree_and_equal_degrees_share_a_group() -> None:
    degrees = [5, 1, 5, 3, 1, 9]
    groups = partition_degrees(degrees, max_buckets=3)
    assert groups == [[1, 4], [0, 2, 3], [5]]  # cost 2 * 1 + 3 * 5 + 1 * 9 = 26, the brute-force optimum
    assert _padded_cost(degrees, groups) == _brute_force_cost(degrees, 3) == 26
    maxima = [max(degrees[i] for i in group) for group in groups]
    assert maxima == sorted(maxima)


def test_partition_with_one_bucket_is_everything_and_with_enough_buckets_pads_nothing() -> None:
    degrees = [4, 7, 4, 2, 7, 9]
    assert partition_degrees(degrees, 1) == [[0, 1, 2, 3, 4, 5]]
    exact = partition_degrees(degrees, 10)
    assert len(exact) == 4  # one group per distinct degree
    assert _padded_cost(degrees, exact) == sum(degrees)


def test_partition_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="max_buckets"):
        partition_degrees([1, 2], 0)
    with pytest.raises(ValueError, match="at least one"):
        partition_degrees([], 2)
    with pytest.raises(ValueError, match="nonnegative"):
        partition_degrees([1, -1], 2)


# --------------------------------------------------------------------------
# The plan


@pytest.mark.parametrize("max_buckets", [1, 3, 8, 100])
def test_buckets_cover_every_row_once_and_are_padded_to_their_width(max_buckets: int) -> None:
    stencil = _stencil()
    plan = build_stencil_plan(stencil, max_buckets=max_buckets)
    degrees = stencil.degrees()
    assert 1 <= plan.n_buckets <= max_buckets
    assert torch.equal(torch.sort(plan.row_order).values, torch.arange(stencil.n_latitude))
    assert torch.equal(plan.row_order[plan.inverse_order], torch.arange(stencil.n_latitude))
    for bucket in plan.buckets:
        assert bucket.width == int(degrees[bucket.rows].max())
        assert torch.equal(bucket.rows, torch.sort(bucket.rows).values)
        assert torch.equal(bucket.valid.sum(1), degrees[bucket.rows])
        # The real slots are the row's slots in the stencil's order, the padding the phantom.
        for i, row in enumerate(bucket.rows.tolist()):
            degree = int(degrees[row])
            assert torch.equal(bucket.slots[i, :degree], torch.arange(stencil.row_ptr[row], stencil.row_ptr[row + 1]))
            assert bool((bucket.slots[i, degree:] == stencil.n_slots).all())
    assert plan.n_padded >= stencil.n_slots
    if max_buckets >= int(torch.unique(degrees).numel()):
        assert plan.padding_fraction == 0.0


def test_sources_reproduce_the_stencil_at_every_target_longitude() -> None:
    stencil = _stencil()
    plan = build_stencil_plan(stencil, max_buckets=3)
    n_longitude = stencil.n_longitude
    for bucket in plan.buckets:
        for i, row in enumerate(bucket.rows.tolist()):
            degree = int(stencil.degrees()[row])
            expected = stencil.source_nodes(stencil.slots(row))  # (degree, n_longitude)
            assert torch.equal(bucket.source[i, :, :degree], expected.transpose(0, 1))
            # Padding reads the target node itself, whatever it holds is multiplied by zero.
            target = row * n_longitude + torch.arange(n_longitude)
            assert torch.equal(bucket.source[i, :, degree:], target.unsqueeze(1).expand(-1, bucket.width - degree))


def test_more_buckets_never_pad_more() -> None:
    stencil = _stencil(shape=(16, 32), cutoff=0.5)
    padded = [build_stencil_plan(stencil, max_buckets=k).n_padded for k in (1, 2, 4, 8, 16)]
    assert padded == sorted(padded, reverse=True)
    assert padded[0] == stencil.n_latitude * int(stencil.degrees().max())


def test_max_bucket_positions_splits_buckets_into_row_runs() -> None:
    stencil = _stencil()
    whole = build_stencil_plan(stencil, max_buckets=2)
    split = build_stencil_plan(stencil, max_buckets=2, max_bucket_positions=40)
    assert split.n_buckets > whole.n_buckets
    assert split.n_padded == whole.n_padded  # splitting rows costs no padding
    for bucket in split.buckets:
        assert bucket.n_rows * bucket.width <= 40 or bucket.n_rows == 1
    with pytest.raises(ValueError, match="max_bucket_positions"):
        build_stencil_plan(stencil, max_bucket_positions=0)


def test_pad_appends_one_phantom_row() -> None:
    stencil = _stencil()
    plan = build_stencil_plan(stencil)
    table = torch.randn(stencil.n_slots, 3)
    padded = plan.pad(table, -1.0)
    assert padded.shape == (stencil.n_slots + 1, 3)
    assert torch.equal(padded[:-1], table) and bool((padded[-1] == -1.0).all())
    along = plan.pad(torch.randn(2, stencil.n_slots, 4), 0.0, dim=1)
    assert along.shape == (2, stencil.n_slots + 1, 4) and bool((along[:, -1] == 0.0).all())
    with pytest.raises(ValueError, match="n_slots"):
        plan.pad(torch.randn(stencil.n_slots - 1), 0.0)


def test_plan_rejects_inconsistent_buckets() -> None:
    stencil = _stencil()
    plan = build_stencil_plan(stencil, max_buckets=2)
    first, second = plan.buckets
    with pytest.raises(ValueError, match="every one of the"):
        StencilPlan(stencil, [first])
    with pytest.raises(ValueError, match="every one of the"):
        StencilPlan(stencil, [first, first])
    broken = StencilBucket(second.rows, second.slots, ~second.valid, second.source)
    with pytest.raises(ValueError, match="valid"):
        StencilPlan(stencil, [first, broken])
    with pytest.raises(ValueError, match="source must have shape"):
        StencilPlan(stencil, [first, StencilBucket(second.rows, second.slots, second.valid, second.source[:, :1])])
    with pytest.raises(TypeError, match="StencilBucket"):
        StencilPlan(stencil, [first, "bucket"])  # type: ignore[list-item]


def test_plan_moves_with_its_stencil() -> None:
    plan = build_stencil_plan(_stencil())
    moved = plan.to(dtype=torch.float32)
    assert moved.dtype == torch.float32 and moved.n_buckets == plan.n_buckets
    for a, b in zip(moved.buckets, plan.buckets, strict=True):
        assert torch.equal(a.source, b.source) and torch.equal(a.slots, b.slots)
    assert "buckets=" in repr(plan) and "padding=" in repr(plan)


# --------------------------------------------------------------------------
# Plans at several cutoffs


def test_plans_are_keyed_by_cutoff_and_resolve_a_layers_radius() -> None:
    discretization, plans = build_tensor_grid_geometry("legendre-gauss", *SHAPE, [CUTOFF, 0.4])
    assert plans.cutoffs == (0.4, CUTOFF) and len(plans) == 2
    assert 0.4 in plans and CUTOFF in plans and 0.5 not in plans and "0.4" not in plans
    assert plans.for_cutoff(0.4).cutoff == 0.4 and plans.for_cutoff(CUTOFF).cutoff == CUTOFF
    assert plans.for_cutoff(0.4 + 1e-14) is plans.for_cutoff(0.4)  # the discretization's own radius key
    with pytest.raises(ValueError, match="No stencil plan at cutoff 0.5"):
        plans.for_cutoff(0.5)
    # Built beside the discretization: the same nodes, the windows prepared at the same radii.
    assert discretization.cached_radii == (0.4, CUTOFF)
    assert plans.n_nodes == discretization.n_nodes and plans.n_latitude == SHAPE[0] and plans.n_longitude == SHAPE[1]
    assert torch.equal(plans.for_cutoff(0.4).stencil.geometry.distances, build_stencil_plans(
        discretization, [0.4], n_latitude=SHAPE[0], n_longitude=SHAPE[1]).for_cutoff(0.4).stencil.geometry.distances)
    assert "cutoffs=[0.4, 0.6]" in repr(plans)


def test_plans_reject_inconsistent_members() -> None:
    _, plans = build_tensor_grid_geometry("legendre-gauss", *SHAPE, [CUTOFF, 0.4])
    plan_a, plan_b = plans.for_cutoff(0.4), plans.for_cutoff(CUTOFF)
    with pytest.raises(ValueError, match="at least one"):
        StencilPlans({})
    with pytest.raises(ValueError, match="built at cutoff"):
        StencilPlans({0.5: plan_a})
    with pytest.raises(TypeError, match="StencilPlan"):
        StencilPlans({0.4: plan_a.stencil})  # type: ignore[dict-item]
    other = build_stencil_plan(_stencil(shape=(8, 16), cutoff=CUTOFF))
    with pytest.raises(ValueError, match="share the grid shape"):
        StencilPlans({0.4: plan_a, CUTOFF: other})
    moved = StencilPlans({0.4: plan_a, CUTOFF: plan_b}).to(dtype=torch.float32)
    assert moved.dtype == torch.float32 and moved.cutoffs == (0.4, CUTOFF)
    with pytest.raises(ValueError, match="at least one"):
        build_tensor_grid_geometry("legendre-gauss", *SHAPE, [])


def test_split_edges_and_plan_resolves_the_container_at_the_layers_cutoff() -> None:
    discretization, plans = build_tensor_grid_geometry("legendre-gauss", *SHAPE, [CUTOFF, 0.4])
    assert split_edges_and_plan(plans, None, 0.4) == (None, plans.for_cutoff(0.4))
    assert split_edges_and_plan(None, plans, CUTOFF) == (None, plans.for_cutoff(CUTOFF))
    assert split_edges_and_plan(discretization, None, 0.4) == (discretization, None)
    with pytest.raises(ValueError, match="No stencil plan"):
        split_edges_and_plan(plans, None, 0.5)
    with pytest.raises(TypeError, match="once"):
        split_edges_and_plan(plans, plans, 0.4)
