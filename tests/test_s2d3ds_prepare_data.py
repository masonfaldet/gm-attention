"""The data preparation extracts only the panorama folders and hands them to
the released converter, which produces a file the experiment's reader loads."""

from __future__ import annotations

import importlib.util
import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location("s2d3ds_prepare", ROOT / "S2D3DS_128x256_Segmentation" / "prepare_data.py")
assert spec is not None and spec.loader is not None
prepare_data = importlib.util.module_from_spec(spec)
sys.modules["s2d3ds_prepare"] = prepare_data
spec.loader.exec_module(prepare_data)

pytest.importorskip("PIL")
pytest.importorskip("h5py")

#: A label list in the 2D-3D-Semantics format: "<object>_<n>_<room>_<n>_<n>".
LABELS = ["<UNK>_0_<UNK>_0_0", "wall_1_office_1_1", "floor_2_office_1_2", "wall_3_office_2_3", "ceiling_4_office_2_4"]
SHAPE = (64, 128)  # a panorama, before the converter's downsampling


def _png(array: np.ndarray, mode: str) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(array, mode=mode).save(buffer, format="PNG")
    return buffer.getvalue()


def _panorama(rng: np.random.Generator, camera: str) -> dict[str, bytes]:
    rgb = rng.integers(0, 256, size=(*SHAPE, 3), dtype=np.uint8)
    # Semantic PNGs encode the label index as r*65536 + g*256 + b; indices
    # here are small, so the blue channel carries them.
    labels = rng.integers(0, len(LABELS), size=SHAPE).astype(np.uint8)
    semantic = np.stack([np.zeros(SHAPE, np.uint8), np.zeros(SHAPE, np.uint8), labels], axis=-1)
    depth = rng.integers(0, 2**16 - 1, size=SHAPE, dtype=np.uint16)
    stem = f"camera_{camera}_office_1_frame_equirectangular_domain"
    return {
        f"pano/rgb/{stem}_rgb.png": _png(rgb, "RGB"),
        f"pano/semantic/{stem}_semantic.png": _png(semantic, "RGB"),
        f"pano/depth/{stem}_depth.png": _png(depth, "I;16"),
    }


def _tarball(path: Path, area: str, panoramas: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    with tarfile.open(path, "w") as tar:
        # Something the converter must never see, as the real tarballs carry
        # normals, poses and semantic_pretty renderings beside the three folders.
        for name, payload in {f"{area}/pano/normal/decoy.png": b"not read", f"{area}/3d/pointcloud.txt": b"huge"}.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        for index in range(panoramas):
            for name, payload in _panorama(rng, f"{area}{index:02d}").items():
                info = tarfile.TarInfo(f"{area}/{name}")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))


def test_only_the_panorama_folders_are_extracted(tmp_path: Path) -> None:
    tarball = tmp_path / "area_3_no_xyz.tar"
    _tarball(tarball, "area_3", panoramas=2, seed=0)
    count = prepare_data.extract_panoramas(tarball, tmp_path, "area_3")
    assert count == 6
    extracted = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file() and p != tarball)
    assert all(any(f in e for f in prepare_data.PANORAMA_FOLDERS) for e in extracted)
    assert not (tmp_path / "area_3" / "pano" / "normal").exists()
    assert not (tmp_path / "area_3" / "3d").exists()
    assert prepare_data.wanted("area_3/pano/rgb/x.png", "area_3")
    assert not prepare_data.wanted("area_3/pano/rgb/x.png", "area_4")
    assert not prepare_data.wanted("area_3/pano/normal/x.png", "area_3")


def test_the_released_converter_runs_on_the_extracted_panoramas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end on two synthetic areas with a stubbed download: the file the
    experiment reads, at the shape the downsampling factor implies, with the
    object classes of the label list, <UNK> included as the released reader keeps it."""
    from experiments.stanford_2d3ds.data import load_splits

    data_path = tmp_path / "2D3DS"
    data_path.mkdir()
    for area, seed in (("area_3", 1), ("area_4", 2)):
        _tarball(data_path / f"{area}_no_xyz.tar", area, panoramas=6, seed=seed)

    from torch_harmonics.examples import Stanford2D3DSDownloader

    def fake_download(self, filename):  # the tarball is already in place
        return str(data_path / filename)

    monkeypatch.setattr(Stanford2D3DSDownloader, "_download_file", fake_download)

    class FakeResponse:
        def json(self):
            return LABELS

    import requests

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: FakeResponse())

    dataset_file = prepare_data.prepare(data_path, areas=["area_3", "area_4"], downsampling_factor=4)
    assert dataset_file == data_path / "stanford_2d3ds_dataset_ds4_area_3+area_4.h5", "a subset build has its own name"
    assert (data_path / "area_3" / "panoramas.json").is_file() and prepare_data.area_state(data_path / "area_3") == "complete"
    assert not (data_path / "stanford_2d3ds_dataset_ds4_area_3+area_4.h5.partial").exists()
    assert not (data_path / "area_3_no_xyz.tar").exists(), "the tarball is removed after extraction"
    assert (data_path / "area_3" / "pano" / "rgb").is_dir()

    splits = load_splits(dataset_file, split_ratios=[0.5, 0.25, 0.25])
    assert splits.shape == (16, 32) and splits.in_channels == 3
    assert splits.class_labels == ["<UNK>", "ceiling", "floor", "wall"]
    assert len(splits.dataset) == 12 and splits.sizes() == {"train": 6, "val": 3, "report": 3}
    image, target = splits.subset("train")[0]
    assert image.shape == (3, 16, 32) and target.shape == (16, 32)
    assert int(target.min()) >= 0 and int(target.max()) < 4, "no pixel is masked: <UNK> is class 0"
    assert abs(float(sum(splits.class_histogram)) - 1.0) < 1e-5

    # Idempotent: an existing file is reused, not rebuilt; a different area
    # selection is a different file, never the existing one under a
    # misleading name.
    assert prepare_data.prepare(data_path, areas=["area_3", "area_4"], downsampling_factor=4) == dataset_file
    assert prepare_data.prepare(data_path, areas=["area_3"], downsampling_factor=4) == data_path / "stanford_2d3ds_dataset_ds4_area_3.h5"
    assert prepare_data.sample_count(data_path / "stanford_2d3ds_dataset_ds4_area_3.h5") == 6


def test_an_interrupted_extraction_is_fetched_again_and_a_legacy_area_is_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The three folders existing is not the area being ready: a short
    extraction is redone; an area extracted before manifests existed is
    kept when its folders match panorama for panorama, and gets a manifest."""
    data_path = tmp_path / "2D3DS"
    data_path.mkdir()
    _tarball(data_path / "area_3_no_xyz.tar", "area_3", panoramas=3, seed=1)
    _tarball(data_path / "area_4_no_xyz.tar", "area_4", panoramas=2, seed=2)
    # area_3: a legacy, complete extraction without a manifest.
    prepare_data.extract_panoramas(data_path / "area_3_no_xyz.tar", data_path, "area_3")
    assert prepare_data.area_state(data_path / "area_3") == "consistent"
    # area_4: extracted, then one depth file lost (an interrupted run).
    prepare_data.extract_panoramas(data_path / "area_4_no_xyz.tar", data_path, "area_4")
    lost = next((data_path / "area_4" / "pano" / "depth").iterdir())
    lost.unlink()
    assert prepare_data.area_state(data_path / "area_4") == "incomplete"

    from torch_harmonics.examples import Stanford2D3DSDownloader

    downloads: list[str] = []

    def fake_download(self, filename):
        downloads.append(filename)
        return str(data_path / filename)

    monkeypatch.setattr(Stanford2D3DSDownloader, "_download_file", fake_download)
    downloader = Stanford2D3DSDownloader(base_url="http://unused/", local_dir=str(data_path))
    folders = prepare_data.fetch_areas(
        downloader, data_path, [("area_3_no_xyz.tar", "area_3"), ("area_4_no_xyz.tar", "area_4")], keep_tarballs=True
    )
    assert folders == ["area_3", "area_4"]
    assert downloads == ["area_4_no_xyz.tar"], "only the incomplete area is fetched again"
    assert prepare_data.area_state(data_path / "area_3") == "complete"
    assert prepare_data.area_state(data_path / "area_4") == "complete"
    assert len(list((data_path / "area_4" / "pano" / "depth").iterdir())) == 2
    # A manifest that no longer matches the folders marks the area incomplete.
    lost = next((data_path / "area_4" / "pano" / "rgb").iterdir())
    lost.unlink()
    assert prepare_data.area_state(data_path / "area_4") == "incomplete"
    assert prepare_data.dataset_filename(16, None) == "stanford_2d3ds_dataset_ds16.h5"


def test_a_manifested_area_that_goes_incomplete_is_refetched_and_a_broken_manifest_is_harmless(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_path = tmp_path / "2D3DS"
    data_path.mkdir()
    _tarball(data_path / "area_3_no_xyz.tar", "area_3", panoramas=3, seed=1)
    prepare_data.extract_panoramas(data_path / "area_3_no_xyz.tar", data_path, "area_3")
    prepare_data.write_manifest(data_path / "area_3")
    assert prepare_data.area_state(data_path / "area_3") == "complete"
    next((data_path / "area_3" / "pano" / "depth").iterdir()).unlink()
    assert prepare_data.area_state(data_path / "area_3") == "incomplete"

    from torch_harmonics.examples import Stanford2D3DSDownloader

    monkeypatch.setattr(Stanford2D3DSDownloader, "_download_file", lambda self, filename: str(data_path / filename))
    downloader = Stanford2D3DSDownloader(base_url="http://unused/", local_dir=str(data_path))
    assert prepare_data.fetch_areas(downloader, data_path, [("area_3_no_xyz.tar", "area_3")], keep_tarballs=True) == ["area_3"]
    assert prepare_data.area_state(data_path / "area_3") == "complete"
    assert len(list((data_path / "area_3" / "pano" / "depth").iterdir())) == 3
    for broken in ('{"panoramas": {"pano/rgb": 2, "pano/sem', "", "[1, 2]"):
        (data_path / "area_3" / "panoramas.json").write_text(broken)
        assert prepare_data.area_state(data_path / "area_3") == "incomplete"
    assert not (data_path / "area_3" / "panoramas.json.partial").exists()
