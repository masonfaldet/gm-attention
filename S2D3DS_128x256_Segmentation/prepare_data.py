"""Download and convert Stanford 2D3DS to the released HDF5 file, one area at a time.

The released downloader (``Stanford2D3DSDownloader``, resumable) fetches the
seven ``no_xyz`` area tarballs; the released converter (``convert_dataset``,
unchanged, so the file's numerics are the released ones) writes one HDF5
file with every panorama bilinearly downsampled by 16 to 128x256, semantic
PNGs mapped to their object class through the 2D-3D-Semantics label json,
depth kept, and the train statistics the released reader exposes. Only the
three folders the converter reads leave each tarball, which is removed
before the next area is fetched. Requires ``requests``, ``pillow``, ``tqdm``
and ``h5py``.

Usage, from the repository root::

    uv run python S2D3DS_128x256_Segmentation/prepare_data.py                 # data/2D3DS/stanford_2d3ds_dataset_ds16.h5
    uv run python S2D3DS_128x256_Segmentation/prepare_data.py --areas area_3   # one area, for a dry run: ..._ds16_area_3.h5
    uv run python S2D3DS_128x256_Segmentation/prepare_data.py --keep-tarballs
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
from collections.abc import Iterable, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repository root

DEFAULT_DATA_PATH = "data/2D3DS"
DEFAULT_FACTOR = 16
#: The folders the released converter reads, relative to an area directory.
PANORAMA_FOLDERS = ("pano/rgb", "pano/semantic", "pano/depth")
DEFAULT_BASE_URL = "https://cvg-data.inf.ethz.ch/2d3ds/no_xyz/"


def wanted(member_name: str, area: str) -> bool:
    """Whether a tarball member is one of the panorama files the converter reads."""
    parts = Path(member_name).parts
    if len(parts) < 4 or parts[0] != area:
        return False
    return "/".join(parts[1:3]) in PANORAMA_FOLDERS


def extract_panoramas(tarball: Path, destination: Path, area: str) -> int:
    """Extract only the panorama folders of ``area`` from ``tarball``;
    returns the number of members extracted."""
    count = 0
    with tarfile.open(tarball, "r|") as tar:  # streaming: one pass over a large file
        for member in tar:
            if not (member.isfile() and wanted(member.name, area)):
                continue
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            assert source is not None
            with target.open("wb") as sink:
                while chunk := source.read(1 << 20):
                    sink.write(chunk)
            count += 1
    return count


MANIFEST = "panoramas.json"


def panorama_stems(area_dir: Path) -> dict[str, set[str]]:
    """The panorama identifiers found in each of the three folders, the
    modality suffix stripped, so the folders can be matched against each other."""
    stems: dict[str, set[str]] = {}
    for folder in PANORAMA_FOLDERS:
        modality = folder.rsplit("/", 1)[1]
        directory = area_dir / folder
        stems[folder] = (
            {f.name[: -len(f"_{modality}.png")] for f in directory.iterdir() if f.name.endswith(f"_{modality}.png")}
            if directory.is_dir() else set()
        )
    return stems


def area_state(area_dir: Path) -> str:
    """``"complete"``, ``"consistent"`` or ``"incomplete"``.

    Complete: the manifest written at the end of a successful extraction
    matches the folders on disk. Consistent: no manifest, but every
    panorama has all three modalities and none is missing. Anything else is
    incomplete and is fetched again: an interrupted extraction leaves the
    three folders present but short, and the released converter silently
    drops any panorama missing a modality.
    """
    stems = panorama_stems(area_dir)
    counts = {folder: len(s) for folder, s in stems.items()}
    manifest = area_dir / MANIFEST
    if manifest.is_file():
        try:
            recorded = json.loads(manifest.read_text())
            matches = recorded.get("panoramas") == counts
        except (ValueError, AttributeError, TypeError):  # a truncated or foreign manifest
            return "incomplete"
        return "complete" if matches and all(counts.values()) else "incomplete"
    return "consistent" if folders_match(stems) else "incomplete"


def folders_match(stems: dict[str, set[str]]) -> bool:
    """Every panorama has all three modalities and none is missing."""
    sets = list(stems.values())
    return all(sets) and all(s == sets[0] for s in sets)


def write_manifest(area_dir: Path) -> None:
    counts = {folder: len(s) for folder, s in panorama_stems(area_dir).items()}
    partial = area_dir / (MANIFEST + ".partial")
    partial.write_text(json.dumps({"panoramas": counts}, indent=2) + "\n")
    os.replace(partial, area_dir / MANIFEST)


def fetch_areas(
    downloader: object,
    data_path: Path,
    pairs: Iterable[tuple[str, str]],
    *,
    keep_tarballs: bool,
) -> list[str]:
    """Download and selectively extract every area not already complete."""
    folders: list[str] = []
    for filename, area in pairs:
        area_dir = data_path / area
        state = area_state(area_dir)
        if state == "complete":
            print(f"{area}: panoramas complete (manifest), skipping the download")
            folders.append(area)
            continue
        if state == "consistent":
            print(f"{area}: panoramas consistent across the three folders, writing the manifest and skipping the download")
            write_manifest(area_dir)
            folders.append(area)
            continue
        if any((area_dir / folder).exists() for folder in PANORAMA_FOLDERS) or (area_dir / MANIFEST).exists():
            print(f"{area}: incomplete extraction on disk, fetching again")
            for folder in PANORAMA_FOLDERS:
                shutil.rmtree(area_dir / folder, ignore_errors=True)
            (area_dir / MANIFEST).unlink(missing_ok=True)  # a stale manifest must not vouch for the new files
        tarball = Path(downloader._download_file(filename))  # type: ignore[attr-defined]  # the released, resumable downloader
        print(f"{area}: extracting {PANORAMA_FOLDERS} from {tarball.name}")
        count = extract_panoramas(tarball, data_path, area)
        print(f"{area}: {count} files")
        if count == 0:
            raise RuntimeError(f"{tarball} holds no {PANORAMA_FOLDERS} members under {area}/.")
        if not folders_match(panorama_stems(area_dir)):
            raise RuntimeError(f"{area}: the extracted folders do not match each other panorama for panorama.")
        write_manifest(area_dir)
        if not keep_tarballs:
            tarball.unlink()
        folders.append(area)
    return folders


def dataset_filename(downsampling_factor: int, areas: Sequence[str] | None) -> str:
    """The full build's name, or a distinct one for a subset of areas."""
    suffix = "" if areas is None else "_" + "+".join(areas)
    return f"stanford_2d3ds_dataset_ds{downsampling_factor}{suffix}.h5"


def sample_count(dataset_file: Path) -> int:
    import h5py

    with h5py.File(dataset_file, "r") as handle:
        return int(handle["rgb"].shape[0])


def prepare(
    data_path: Path,
    *,
    areas: Sequence[str] | None = None,
    downsampling_factor: int = DEFAULT_FACTOR,
    keep_tarballs: bool = False,
    base_url: str = DEFAULT_BASE_URL,
    labels_url: str | None = None,
) -> Path:
    from torch_harmonics.examples import Stanford2D3DSDownloader
    from torch_harmonics.examples.stanford_2d3ds_dataset import (
        DEFAULT_LABELS_URL,
        DEFAULT_TAR_FILE_PAIRS,
    )

    data_path.mkdir(parents=True, exist_ok=True)
    dataset_file = data_path / dataset_filename(downsampling_factor, areas)
    if dataset_file.exists():
        # Published by the rename below, so it is a finished conversion.
        print(f"{dataset_file} exists ({sample_count(dataset_file)} panoramas); delete it to convert again.")
        return dataset_file
    pairs = [(f, a) for f, a in DEFAULT_TAR_FILE_PAIRS if areas is None or a in areas]
    if not pairs:
        raise ValueError(f"no known area among {areas}; known: {[a for _, a in DEFAULT_TAR_FILE_PAIRS]}")
    downloader = Stanford2D3DSDownloader(base_url=base_url, local_dir=str(data_path))
    folders = fetch_areas(downloader, data_path, pairs, keep_tarballs=keep_tarballs)
    expected = sum(len(panorama_stems(data_path / area)["pano/rgb"]) for area in folders)

    import requests

    class_labels = requests.get(labels_url or DEFAULT_LABELS_URL, timeout=60).json()
    print(f"converting {len(folders)} areas ({expected} panoramas) at downsampling factor {downsampling_factor} ...")
    # Convert under a partial name and publish by rename: a file under the
    # final name is a finished conversion, never an interrupted one.
    partial = dataset_file.with_name(dataset_file.name + ".partial")
    partial.unlink(missing_ok=True)
    converted = Path(
        downloader.convert_dataset(
            data_folders=folders, class_labels=class_labels,
            dataset_file=partial.name, downsampling_factor=downsampling_factor,
        )
    )
    found = sample_count(converted)
    if found != expected:
        raise RuntimeError(
            f"{converted} holds {found} panoramas where the areas hold {expected}; the released "
            "converter drops a panorama missing a modality. Left in place for inspection, not published."
        )
    os.replace(converted, dataset_file)
    print(f"{dataset_file}: {found} panoramas")
    return dataset_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="where the tarballs, panoramas and the HDF5 file go")
    parser.add_argument("--downsampling-factor", type=int, default=DEFAULT_FACTOR, help="16 gives the paper's 128x256")
    parser.add_argument("--areas", nargs="*", default=None, help="a subset of area_1 .. area_6 (default: all seven)")
    parser.add_argument("--keep-tarballs", action="store_true", help="keep each tarball after extracting its panoramas")
    args = parser.parse_args()

    dataset_file = prepare(
        Path(args.data_path), areas=args.areas, downsampling_factor=args.downsampling_factor,
        keep_tarballs=args.keep_tarballs,
    )

    from torch_harmonics.examples import StanfordSegmentationDataset

    dataset = StanfordSegmentationDataset(dataset_file=str(dataset_file), ignore_alpha_channel=True)
    channels, nlat, nlon = dataset.input_shape
    print(f"{dataset_file}: {len(dataset)} panoramas at {nlat}x{nlon}, {channels} channels, "
          f"{dataset.num_classes} classes: {', '.join(dataset.class_labels)}")


if __name__ == "__main__":
    main()
