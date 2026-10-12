"""Which code a run or a measurement came from."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def code_revision(root: Path | None = None) -> str:
    """``git rev-parse HEAD`` of the working tree, ``+dirty`` if it has
    uncommitted changes; ``<sha>+archive`` from a ``GIT_REVISION`` file at
    the root of a tree without ``.git``; ``"unknown"`` otherwise."""
    root = REPOSITORY_ROOT if root is None else root
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
        return head + ("+dirty" if status.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        pass
    marker = root / "GIT_REVISION"
    if marker.is_file():
        sha = marker.read_text().split()
        if sha:
            return sha[0] + "+archive"
    return "unknown"


#: Environment-variable prefixes that select an implementation path, never a value.
SWITCH_PREFIXES = ("GM_", "PYTORCH_CUDA_ALLOC_CONF")


def runtime_switches() -> dict[str, str]:
    """The ``GM_*`` switches (and the allocator's) set in this process's
    environment, as launched."""
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith(SWITCH_PREFIXES)}


__all__ = ["REPOSITORY_ROOT", "SWITCH_PREFIXES", "code_revision", "runtime_switches"]
