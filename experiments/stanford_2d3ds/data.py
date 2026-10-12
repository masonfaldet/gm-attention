"""Stanford 2D3DS as the released segmentation script reads it, and the field views of it.

The data path is the released one: the HDF5 file that ``torch_harmonics``'
``Stanford2D3DSDownloader.prepare_dataset`` writes (panoramas bilinearly
downsampled by 16 to ``128 x 256``, labels reduced to their object class,
invalid ids masked to ``-100``), its ``StanfordSegmentationDataset`` reader,
the same ``random_split`` call with the same generator seed and ratios, and
``compute_stats_s2`` for the train-split, quadrature-weighted per-channel
normalization. The released splits keep their roles under other names:

    released ``train``  ->  ``train``   95 %, the optimizer's data
    released ``test``   ->  ``val``     2.5 %, scored every epoch (the validation loss)
    released ``valid``  ->  ``report``  2.5 %, scored once after training, one sample at a
                                        time; the reported mean +- std are taken over it

The realized split depends on the sample order inside the HDF5 file, which
the released converter takes from ``os.listdir``; every arm and seed reads
one file, whose SHA-256 the protocol records.

The model reads an image as a field of ``in_channels`` invariant scalars on
the equiangular grid, one node per pixel in row-major ``(row, column)``
order, the node order of
:func:`~gm_attention.discretization.grids.torch_harmonics.build_torch_harmonics_grid`,
with row 0 at colatitude 0 as the released DISCO convolution reads the same
tensor. Logits come back the same way, reshaped to ``(1, C, nlat, nlon)``
for the released loss and metrics.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from gm_attention.representations.field_type import SO2FeatureField, SO2FieldType

#: The released split: ``random_split(range(n), [0.95, 0.025, 0.025])`` on ``torch.Generator().manual_seed(333)``.
SPLIT_SEED = 333
SPLIT_RATIOS: tuple[float, float, float] = (0.95, 0.025, 0.025)
SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "report")
#: The released reader's mask value for an invalid class id, ignored by the released loss and metrics.
IGNORE_INDEX = -100


@dataclass(frozen=True)
class Normalization:
    """Per-channel ``(x - mean) / std`` over the train split, quadrature-weighted."""

    mean: tuple[float, ...]
    std: tuple[float, ...]

    def apply(self, image: Tensor) -> Tensor:  # (C, nlat, nlon)
        mean = image.new_tensor(self.mean).reshape(-1, 1, 1)
        std = image.new_tensor(self.std).reshape(-1, 1, 1)
        return (image - mean) / std

    def state_dict(self) -> dict[str, list[float]]:
        return {"mean": list(self.mean), "std": list(self.std)}


def split_indices(
    num_samples: int,
    *,
    seed: int = SPLIT_SEED,
    ratios: Sequence[float] = SPLIT_RATIOS,
) -> dict[str, list[int]]:
    """The released split: one ``random_split`` over ``range(num_samples)``
    on a generator seeded with ``seed``."""
    if len(ratios) != len(SPLIT_NAMES):
        raise ValueError(f"ratios needs {len(SPLIT_NAMES)} entries, got {len(ratios)}.")
    generator = torch.Generator().manual_seed(int(seed))
    # The released script passes range(); torch accepts any sized indexable
    # object although the annotation names Dataset.
    parts: Sequence[Any] = torch.utils.data.random_split(
        cast(Any, range(num_samples)), list(ratios), generator=generator
    )
    return {
        name: [int(part[i]) for i in range(len(part))]
        for name, part in zip(SPLIT_NAMES, parts, strict=True)
    }


def dataset_digest(path: str | Path) -> str:
    """SHA-256 of the file's bytes: the identity a run records and every
    evaluation verifies. Files of equal size and sample count can hold
    different images in a different order, and the released split is a
    permutation of positions; only the content pins which panoramas the
    numbers refer to."""
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@dataclass
class SegmentationSplits:
    """One HDF5 file, split and normalized the released way."""

    dataset: Any  # torch_harmonics' StanfordSegmentationDataset
    dataset_file: Path
    indices: dict[str, list[int]]
    normalization: Normalization
    num_classes: int
    class_labels: list[str]
    class_histogram: list[float]
    shape: tuple[int, int]  # (nlat, nlon)
    in_channels: int

    def subset(self, split: str) -> NormalizedSegmentation:
        if split not in self.indices:
            raise KeyError(f"split must be one of {SPLIT_NAMES}, got {split!r}.")
        return NormalizedSegmentation(self.dataset, self.indices[split], self.normalization)

    def sizes(self) -> dict[str, int]:
        return {name: len(index) for name, index in self.indices.items()}


def load_splits(
    dataset_file: str | Path,
    *,
    split_seed: int = SPLIT_SEED,
    split_ratios: Sequence[float] = SPLIT_RATIOS,
    exclude_polar_fraction: float = 0.0,
    ignore_alpha_channel: bool = True,
) -> SegmentationSplits:
    """Open the released HDF5 file, split it, and calibrate the normalization.

    The normalization is ``compute_stats_s2`` over the train split as the
    released script computes it: a Welford reduction of quadrature-weighted
    per-channel means and variances.
    """
    from torch_harmonics.examples import StanfordSegmentationDataset, compute_stats_s2

    path = Path(dataset_file)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist; build it with S2D3DS_128x256_Segmentation/prepare_data.py."
        )
    dataset = StanfordSegmentationDataset(
        dataset_file=str(path),
        ignore_alpha_channel=ignore_alpha_channel,
        exclude_polar_fraction=exclude_polar_fraction,
    )
    indices = split_indices(len(dataset), seed=split_seed, ratios=split_ratios)
    # A plain indexed view stands in for the released StanfordDatasetSubset (a
    # torch Subset without __getitems__); compute_stats_s2 only indexes it.
    means, stds = compute_stats_s2(_IndexedView(dataset, indices["train"]))
    dataset.reset()
    normalization = Normalization(
        mean=tuple(float(v) for v in np.asarray(means).reshape(-1)),
        std=tuple(float(v) for v in np.asarray(stds).reshape(-1)),
    )
    in_channels, nlat, nlon = (int(v) for v in dataset.input_shape)
    return SegmentationSplits(
        dataset=dataset,
        dataset_file=path,
        indices=indices,
        normalization=normalization,
        num_classes=int(dataset.num_classes),
        class_labels=[str(label) for label in dataset.class_labels],
        class_histogram=[float(v) for v in np.asarray(dataset.class_histogram).reshape(-1)],
        shape=(nlat, nlon),
        in_channels=in_channels,
    )


class _IndexedView:
    """``(rgb, semantic)`` of a subset of a released dataset, by position."""

    def __init__(self, dataset: Any, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = [int(i) for i in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        rgb, semantic = self.dataset[self.indices[index]]
        return rgb, semantic


class NormalizedSegmentation(Dataset[tuple[Tensor, Tensor]]):
    """``(image, target)`` tensors for one split: ``(C, nlat, nlon)`` float32,
    normalized, and ``(nlat, nlon)`` int64 with ``-100`` where invalid."""

    def __init__(self, dataset: Any, indices: Sequence[int], normalization: Normalization) -> None:
        self.dataset = dataset
        self.indices = [int(i) for i in indices]
        self.normalization = normalization

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        rgb, semantic = self.dataset[self.indices[index]]
        image = torch.as_tensor(np.asarray(rgb), dtype=torch.float32)
        target = torch.as_tensor(np.asarray(semantic), dtype=torch.int64)
        return self.normalization.apply(image), target


def image_to_field(image: Tensor) -> SO2FeatureField:
    """``(C, nlat, nlon)`` -> ``{0: C}`` on ``nlat * nlon`` nodes, row-major."""
    if image.ndim != 3:
        raise ValueError(f"image must be (C, nlat, nlon), got {tuple(image.shape)}.")
    channels = int(image.shape[0])
    flat = image.permute(1, 2, 0).reshape(-1, channels)
    return SO2FeatureField(SO2FieldType({0: channels}), {0: flat})


def logits_to_image(field: SO2FeatureField, shape: tuple[int, int]) -> Tensor:
    """``{0: C}`` on ``nlat * nlon`` nodes -> ``(1, C, nlat, nlon)``."""
    if field.field_type.frequencies != (0,):
        raise ValueError(f"logits must be a purely scalar field, got {field.field_type!r}.")
    nlat, nlon = int(shape[0]), int(shape[1])
    flat = field[0]  # (n_nodes, C)
    if flat.shape[0] != nlat * nlon:
        raise ValueError(
            f"the field has {flat.shape[0]} nodes but the image {nlat}x{nlon} = {nlat * nlon}."
        )
    return flat.reshape(nlat, nlon, -1).permute(2, 0, 1).unsqueeze(0)


def images_to_field(images: Tensor) -> SO2FeatureField:
    """``(B, C, nlat, nlon)`` -> ``{0: C}`` on ``B * nlat * nlon`` nodes,
    sample-major then row-major: node ``b * nlat * nlon + k`` is pixel ``k``
    of panorama ``b``, the node order of a hierarchy tiled ``B`` times
    (:meth:`~gm_attention.discretization.grids.equiangular.TensorGridHierarchy.tile`)."""
    if images.ndim != 4:
        raise ValueError(f"images must be (B, C, nlat, nlon), got {tuple(images.shape)}.")
    channels = int(images.shape[1])
    flat = images.permute(0, 2, 3, 1).reshape(-1, channels)
    return SO2FeatureField(SO2FieldType({0: channels}), {0: flat})


def logits_to_images(field: SO2FeatureField, shape: tuple[int, int]) -> Tensor:
    """``{0: C}`` on ``B * nlat * nlon`` nodes -> ``(B, C, nlat, nlon)``, the
    inverse of :func:`images_to_field` for any ``B`` the node count allows."""
    if field.field_type.frequencies != (0,):
        raise ValueError(f"logits must be a purely scalar field, got {field.field_type!r}.")
    nlat, nlon = int(shape[0]), int(shape[1])
    flat = field[0]  # (B * n_nodes, C)
    if flat.shape[0] % (nlat * nlon) != 0:
        raise ValueError(
            f"the field has {flat.shape[0]} nodes, not a multiple of the image {nlat}x{nlon} = {nlat * nlon}."
        )
    return flat.reshape(-1, nlat, nlon, flat.shape[-1]).permute(0, 3, 1, 2)
