"""The released script's planar baselines, importable by their own names.

torch-harmonics v0.8.0 keeps its Euclidean counterparts (``Segformer``,
``Transformer``, ``UNet``) under ``examples/baseline_models``, outside the
installed package; that directory is vendored verbatim under
``S2D3DS_128x256_Segmentation/reference/upstream/examples``. This module
appends it to ``sys.path`` as the released script does and re-exports the
three classes, so a Hydra ``_target_`` can name them. ``Segformer`` and
``Transformer`` import ``natten`` at load time.
"""

from __future__ import annotations

import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[2] / "S2D3DS_128x256_Segmentation" / "reference" / "upstream" / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.append(str(EXAMPLES))

from baseline_models import Segformer, Transformer, UNet

__all__ = ["EXAMPLES", "Segformer", "Transformer", "UNet"]
