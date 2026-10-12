"""The data path is the released one: split, normalization, and the field views."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch_harmonics.examples.losses import CrossEntropyLossS2, get_quadrature_weights

from experiments.stanford_2d3ds.data import (
    IGNORE_INDEX,
    SPLIT_NAMES,
    image_to_field,
    load_splits,
    logits_to_image,
    split_indices,
)
from tests.support.synthetic_2d3ds import write_synthetic_dataset


@pytest.fixture(scope="module")
def dataset_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_synthetic_dataset(tmp_path_factory.mktemp("data") / "synthetic.h5", num_samples=40, shape=(16, 32), num_classes=4)


def test_the_split_is_the_released_call() -> None:
    # 1,413 is the converted file's panorama count; the paper reports on 35.
    generator = torch.Generator().manual_seed(333)
    parts = torch.utils.data.random_split(range(1413), [0.95, 0.025, 0.025], generator=generator)
    ours = split_indices(1413)
    assert list(ours) == list(SPLIT_NAMES)
    for name, part in zip(SPLIT_NAMES, parts, strict=True):
        assert ours[name] == [int(part[i]) for i in range(len(part))]
    assert sorted(i for part in ours.values() for i in part) == list(range(1413))
    assert {name: len(part) for name, part in ours.items()} == {"train": 1343, "val": 35, "report": 35}


def test_splits_normalization_and_shapes(dataset_file: Path) -> None:
    splits = load_splits(dataset_file)
    assert splits.sizes() == {"train": 38, "val": 1, "report": 1}
    assert splits.num_classes == 4 and splits.shape == (16, 32) and splits.in_channels == 3

    # Quadrature-weighted mean over the train split, per channel, by hand.
    weights = get_quadrature_weights(16, 32, "equiangular", tile=True).numpy().astype(np.float64)
    images = [np.asarray(splits.dataset[i][0], dtype=np.float64) for i in splits.indices["train"]]
    mean = np.mean([(img * weights).sum(axis=(1, 2)) for img in images], axis=0)
    assert splits.normalization.mean == pytest.approx(tuple(mean), rel=1e-5)

    image, target = splits.subset("train")[0]
    assert image.shape == (3, 16, 32) and image.dtype == torch.float32
    assert target.shape == (16, 32) and target.dtype == torch.int64
    assert int(target.min()) >= min(0, IGNORE_INDEX)


def test_field_views_round_trip_and_score_identically(dataset_file: Path) -> None:
    splits = load_splits(dataset_file)
    image, target = splits.subset("train")[0]
    field = image_to_field(image)
    assert field.batch_shape == (16 * 32,)
    # Node a * nlon + o is pixel (a, o).
    assert torch.equal(field[0][3 * 32 + 5], image[:, 3, 5])
    assert torch.equal(logits_to_image(field, (16, 32))[0], image)

    loss = CrossEntropyLossS2(nlat=16, nlon=32, grid="equiangular")
    logits = torch.randn(1, 4, 16, 32)
    from gm_attention.representations.field_type import SO2FeatureField, SO2FieldType

    as_field = SO2FeatureField(SO2FieldType({0: 4}), {0: logits[0].permute(1, 2, 0).reshape(-1, 4)})
    torch.testing.assert_close(loss(logits_to_image(as_field, (16, 32)), target[None]), loss(logits, target[None]))


def test_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_splits(tmp_path / "missing.h5")
