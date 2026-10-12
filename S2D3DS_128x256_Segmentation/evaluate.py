"""Score the bundle's fitted runs on the report split, every seed.

Every seed's ``checkpoints/last.pt`` (the released protocol: the model after
its final epoch; ``--checkpoint best`` picks the lowest validation
cross-entropy instead) is rebuilt on the released grid chain and scored one
panorama at a time on the 2.5 % report split with the released loss and
metric classes. Each run keeps its per-sample scores; each arm reports the
mean, the sample standard deviation and the seed standard deviation.

Usage, from the repository root::

    uv run python S2D3DS_128x256_Segmentation/evaluate.py                       # outputs/S2D3DS_128x256_Segmentation
    uv run python S2D3DS_128x256_Segmentation/evaluate.py --device cuda:0 --seeds 0,1,2
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repository root

from experiments.stanford_2d3ds.evaluation import main as _main

ARMS = {
    "sos2_segformer": "SO(S^2)-SegFormer (ours)",
    "sos2_segformer_scalar": "SO(S^2)-SegFormer, scalar-only ablation (every channel a scalar, gravity as input channels)",
    "s2_segformer_released": "S2 SegFormer (local), the released script's run (reference/run.sh adopt)",
    "r2_segformer_released": "R2 SegFormer (local), the released script's run (reference/run.sh adopt)",
    "s2_unet_released": "S2 UNet, the released script's run (reference/run.sh adopt)",
}


def main() -> None:
    _main(__doc__, names=ARMS, default_root="outputs/S2D3DS_128x256_Segmentation",
          default_output="outputs/S2D3DS_128x256_Segmentation/evaluation.json")


if __name__ == "__main__":
    main()
