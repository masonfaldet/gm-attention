"""A synthetic Stanford 2D-3D-S file in the released converter's layout.

Lifted from the development tree's data module so the segmentation
tests run end to end at any resolution without the 2D3DS download; the
shipped ``experiments.stanford_2d3ds.data`` reads such files but never
writes one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_synthetic_dataset(
    path: str | Path,
    *,
    num_samples: int,
    shape: tuple[int, int],
    num_classes: int = 4,
    seed: int = 0,
) -> Path:
    """An HDF5 file with the released converter's layout and random contents.

    For tests and smoke runs only: every dataset the released reader touches
    is present with the right dtype, so the whole pipeline runs end to end
    at any resolution without the 2D3DS download.
    """
    import h5py

    generator = np.random.default_rng(seed)
    nlat, nlon = int(shape[0]), int(shape[1])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = generator.random((num_samples, 3, nlat, nlon), dtype=np.float32)
    semantic = generator.integers(0, num_classes, size=(num_samples, nlat, nlon), dtype=np.int64)
    labels = [f"class{i}" for i in range(num_classes)]
    with h5py.File(path, "w") as file:
        file.create_dataset("rgb", data=rgb)
        file.create_dataset("semantic", data=semantic)
        file.create_dataset("depth", data=np.zeros((num_samples, nlat, nlon), dtype=np.float32))
        file.create_dataset("class_labels", data=np.array(labels, dtype="S"))
        file.create_dataset("data_source_path", data=np.array([f"rgb/{i}.png" for i in range(num_samples)], dtype="S"))
        file.create_dataset("data_target_path", data=np.array([f"semantic/{i}.png" for i in range(num_samples)], dtype="S"))
        for name, value in (
            ("min_rgb", rgb.min(axis=(0, 2, 3))), ("max_rgb", rgb.max(axis=(0, 2, 3))),
            ("mean_rgb", rgb.mean(axis=(0, 2, 3))), ("std_rgb", rgb.std(axis=(0, 2, 3))),
            ("min_depth", np.float32(0)), ("max_depth", np.float32(0)),
            ("mean_depth", np.float32(0)), ("std_depth", np.float32(1)),
        ):
            file.create_dataset(name, data=np.asarray(value, dtype=np.float32))
        histogram = np.bincount(semantic.reshape(-1), minlength=num_classes).astype(np.float32)
        file.create_dataset("class_histogram", data=histogram / histogram.sum())
    return path
