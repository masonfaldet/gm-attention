"""Serve ``torch_harmonics.examples.metrics`` from the copy whose docstring is indented.

torch-harmonics 0.8.0 ships that module with a docstring opened one space
deep (``metrics.py`` line 116), so it raises ``IndentationError`` and cannot
be imported; the released segmentation script imports
``IntersectionOverUnionS2`` and ``AccuracyS2`` from it. This finder answers
exactly that module name with ``../metrics_fixed.py`` (the released file,
checksummed in ``SHA256SUMS``, with three spaces added on that one line)
and nothing else. It is a ``sitecustomize`` so that it holds in every
interpreter the run starts: the script's ``forkserver`` re-imports the
script's module before serving DataLoader workers, and Python imports
``sitecustomize`` from the first ``sys.path`` directory that has one, with
``PYTHONPATH`` entries (inherited by children) before the standard library.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import pathlib
import sys

TARGET = "torch_harmonics.examples.metrics"
SOURCE = pathlib.Path(__file__).resolve().parent.parent / "metrics_fixed.py"


class FixedMetricsFinder(importlib.abc.MetaPathFinder):
    """A meta-path finder for one module name."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        # The name carries the package, so the module's __package__ is
        # torch_harmonics.examples and its `from .losses import ...` resolves.
        return importlib.util.spec_from_file_location(fullname, SOURCE)


if not any(type(finder).__name__ == "FixedMetricsFinder" for finder in sys.meta_path):
    sys.meta_path.insert(0, FixedMetricsFinder())


# fallback/ (the natten stand-in) goes at the END of sys.path: an installed
# natten wins, and the stand-in only lets the registry import where there is none.
FALLBACK = pathlib.Path(__file__).resolve().parent / "fallback"
if str(FALLBACK) not in sys.path:
    sys.path.append(str(FALLBACK))
