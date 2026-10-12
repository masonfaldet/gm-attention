"""A stand-in for ``wandb`` with the calls the released script makes, all no-ops.

The released segmentation script logs in at start-up, opens a run per model
and logs every epoch to Weights & Biases when a run is active. No run is
ever active: ``run`` stays ``None``, as under ``WANDB_MODE=disabled``, so
the script's ``if wandb.run is not None`` branch never fires and every
number it logs still reaches stdout through its own prints. Nothing is sent
anywhere, and no API key is needed.
"""

from __future__ import annotations

run = None


def login(*args, **kwargs):
    return False


class _Run:
    def finish(self, *args, **kwargs) -> None:
        return None

    def log(self, *args, **kwargs) -> None:
        return None


def init(*args, **kwargs) -> _Run:
    return _Run()


def log(*args, **kwargs) -> None:
    return None
