"""Device resolution shared by every training/evaluation entry point."""

from __future__ import annotations

import torch


def resolve_device(requested: torch.device | str | None = None) -> torch.device:
    """Resolve the device to run on.

    Parameters
    ----------
    requested
        An explicit device (``"cuda"``, ``"cpu"``, a ``torch.device``), or
        ``None`` to auto-detect: CUDA if available, otherwise CPU. Every
        entry point resolves a ``device: null`` config value here.

    Returns
    -------
    torch.device
    """
    if requested is not None:
        return torch.device(requested)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
