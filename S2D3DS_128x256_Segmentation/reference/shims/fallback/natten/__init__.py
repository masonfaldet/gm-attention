"""A stand-in for ``natten``, imported by the released model registry.

``examples/model_registry.py`` imports the planar (non-spherical) SegFormer
and Transformer baselines from ``examples/baseline_models``, whose modules
import ``NeighborhoodAttention2D`` from ``natten`` at load time. Where
``natten`` is not installed the import needs to succeed and nothing more;
this class refuses to be instantiated rather than pretend to be the real
layer.
"""

from __future__ import annotations

#: Marks the stand-in for the runner's extension report: it never counts as natten being present.
NATTEN_STAND_IN = True


class NeighborhoodAttention2D:
    def __init__(self, *args, **kwargs) -> None:
        raise ImportError(
            "natten is not installed; this stand-in only lets the released model registry import. "
            "The planar baselines that use NeighborhoodAttention2D cannot be built here."
        )
