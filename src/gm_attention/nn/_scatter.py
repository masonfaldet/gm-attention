"""Gather and scatter primitives shared by the edge-aggregating layers.

Private module: plumbing for ``convolution.py`` and ``attention.py``,
operating on ``SO2FeatureField``'s per-frequency dict layout only.
"""

from __future__ import annotations

import torch
from torch import Tensor

from gm_attention.representations.field_type import SO2FeatureField


def gather_to_edges(
    field: SO2FeatureField,
    index: Tensor,  # (n_edges,)
) -> SO2FeatureField:
    """Node-level field -> edge-level field along ``index``
    (``edges.source_index`` or ``edges.target_index``).

    ``index_select`` rather than advanced indexing: its backward is
    ``index_add_``, a plain scatter-add (atomics by default; the sorted
    implementation under ``torch.use_deterministic_algorithms``).
    """
    with torch.profiler.record_function("gm::gather"):
        return SO2FeatureField(
            field.field_type,
            {
                frequency: tensor.index_select(0, index)
                for frequency, tensor in field.coefficients.items()
            },
        )


def scatter_add_to_nodes(
    values: Tensor,  # (n_edges, ...)
    row_ptr: Tensor,  # (n_nodes + 1,)
) -> Tensor:  # (n_nodes, ...)
    """Edge-level values -> node-level sum.

    ``values`` must be in the target-major edge order of ``EdgeGeometry``
    that ``row_ptr`` indexes. A node with no incident edges sums to ``0``.
    """
    with torch.profiler.record_function("gm::scatter_add"):
        return torch.segment_reduce(values, "sum", offsets=row_ptr, axis=0)


def scatter_max_to_nodes(
    values: Tensor,  # (n_edges,)
    row_ptr: Tensor,  # (n_nodes + 1,)
) -> Tensor:  # (n_nodes,)
    """Edge-level scalars -> node-level max, for a numerically stable softmax.

    Same ordering requirement as :func:`scatter_add_to_nodes`. A node with
    no incident edges reduces to ``-inf``.
    """
    return torch.segment_reduce(values, "max", offsets=row_ptr, axis=0)
