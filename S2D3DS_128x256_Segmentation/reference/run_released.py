"""Run torch-harmonics v0.8.0's ``examples/segmentation/train.py`` unmodified, one seed per run.

What trains is the released script (``upstream/``; the
checksums in ``SHA256SUMS`` are verified before anything starts) with the
registry model exactly as the script builds it. Pointed at the bundle's data
file with ``--data_path data/2D3DS``, the script's own ``random_split`` and
``compute_stats_s2`` produce the split and the normalization every other arm
records in its protocol. The runner only sets the stage: ``--seed N`` makes
the script's two literal ``manual_seed(333)`` calls seed with ``N`` (the
split is drawn from ``torch.Generator().manual_seed(333)``, a generator
method, so every seed shares the released split; ``--seed 333`` leaves the
script as released); ``shims/`` serves the released metrics module, which
does not import as shipped, and no-op ``wandb`` and ``natten`` stand-ins;
without cartopy the final validation's two drawing calls become no-ops; on
a CUDA device the runner refuses to start without the model's fast-path
extensions unless ``--allow-fallback`` is given. Everything the runner does
not consume is passed to the script unchanged; ``--check`` verifies the
checksums, the imports, the extensions and the model's parameter count,
then exits.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import os
import pathlib
import runpy
import sys

HERE = pathlib.Path(__file__).resolve().parent
UPSTREAM = HERE / "upstream"
SCRIPT = UPSTREAM / "examples" / "segmentation" / "train.py"
SHIMS = HERE / "shims"
#: The literal the released script seeds torch with, twice.
RELEASED_SEED = 333
DEFAULT_MODEL = "s2nsegformer_sc2_layers4_e128"
#: Registry entry -> (trainable parameters at 128x256 with 14 classes, or None; the fast-path modules it imports).
MODELS: dict[str, tuple[int | None, tuple[str, ...]]] = {
    "s2nsegformer_sc2_layers4_e128": (609_246, ("disco_helpers", "disco_cuda_extension", "attention_cuda_extension")),
    "s2unet_sc2_layers4_e128": (741_374, ("disco_helpers", "disco_cuda_extension")),
    "nsegformer_sc2_layers4_e128": (None, ("natten",)),
}
EXTENSIONS = MODELS[DEFAULT_MODEL][1]


def verify_checksums() -> list[str]:
    """The files ``SHA256SUMS`` names that do not hash as recorded."""
    mismatched: list[str] = []
    for line in (HERE / "SHA256SUMS").read_text().splitlines():
        digest, _, name = line.strip().partition("  ")
        if not name:
            continue
        path = HERE / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            mismatched.append(name)
    return mismatched


def put_shims_on_path() -> None:
    """``shims/`` first on ``PYTHONPATH`` (inherited by every child) and on
    this process's ``sys.path``, with the metrics finder installed here too."""
    shims = str(SHIMS)
    entries = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry]
    if shims not in entries:
        os.environ["PYTHONPATH"] = os.pathsep.join([shims, *entries])
    if shims not in sys.path:
        sys.path.insert(0, shims)
    # Loaded by path: a sitecustomize the interpreter imported at start-up
    # must not shadow this; the finder guards against double installation.
    spec = importlib.util.spec_from_file_location("_reference_sitecustomize", SHIMS / "sitecustomize.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def install_seed(seed: int) -> None:
    """Make the script's two literal ``manual_seed(333)`` calls seed with ``seed``."""
    import torch

    if seed == RELEASED_SEED:
        return
    original, original_cuda = torch.manual_seed, torch.cuda.manual_seed

    def manual_seed(value):
        return original(seed if int(value) == RELEASED_SEED else value)

    def cuda_manual_seed(value):
        return original_cuda(seed if int(value) == RELEASED_SEED else value)

    torch.manual_seed = manual_seed  # type: ignore[assignment]
    torch.cuda.manual_seed = cuda_manual_seed  # type: ignore[assignment]


def accept_released_subset() -> str:
    """Let newer torch construct the released ``StanfordDatasetSubset``.

    The released subset overrides ``__getitem__`` (to append the sample
    index on request) and not ``__getitems__``; torch 2.6 constructs it and
    its DataLoader fetches batches through ``Subset.__getitems__``, the
    base method, while torch 2.13 refuses the construction (it compares
    function identities). A method of the subclass's own that delegates to
    the base one satisfies that check with torch 2.6's behaviour, in the
    one process that constructs the subsets; the workers receive the class
    unpatched and reach the same base method directly.
    """
    from torch.utils.data import Subset
    from torch_harmonics.examples import StanfordDatasetSubset

    if StanfordDatasetSubset.__getitems__ is not Subset.__getitems__:
        return "released StanfordDatasetSubset: as shipped"

    def __getitems__(self, indices):  # the base method, as torch 2.6 dispatches it
        return Subset.__getitems__(self, indices)

    StanfordDatasetSubset.__getitems__ = __getitems__  # type: ignore[attr-defined]
    return "released StanfordDatasetSubset: given the base __getitems__ (torch 2.6 behaviour) for this torch's construction check"


def guard_plotting() -> str:
    """No-op the script's two drawing calls when cartopy is absent."""
    from torch_harmonics import plotting

    if getattr(plotting, "cartopy", None) is not None:
        return "cartopy present: the script draws its figures"

    def no_plot(*args, **kwargs):
        return None

    plotting.plot_sphere = no_plot  # type: ignore[assignment]
    plotting.imshow_sphere = no_plot  # type: ignore[assignment]
    return "cartopy absent: plot_sphere and imshow_sphere are no-ops (figures skipped, numbers untouched)"


def extension_report(names: tuple[str, ...] = EXTENSIONS) -> tuple[bool, list[str]]:
    """Whether every fast-path module ``names`` imports (the real one: the
    natten stand-in counts as missing), with one line per module."""
    lines: list[str] = []
    present = True
    for name in names:
        try:
            module = importlib.import_module(name)
            if getattr(module, "NATTEN_STAND_IN", False):
                raise ImportError("stand-in")
            lines.append(f"{name}: present")
        except ImportError:
            lines.append(f"{name}: MISSING (pure-PyTorch fallback)" if name != "natten" else "natten: MISSING (the stand-in cannot build the layer)")
            present = False
    return present, lines


def environment_report() -> list[str]:
    import torch

    lines = [f"torch {torch.__version__} (cuda {torch.version.cuda}), python {sys.version.split()[0]}"]
    if torch.cuda.is_available():
        lines.append(f"device: {torch.cuda.get_device_name(0)} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})")
    else:
        lines.append("device: cpu")
    for name in ("torch_harmonics", "torchvision", "pandas", "matplotlib"):
        try:
            module = importlib.import_module(name)
            lines.append(f"{name} {getattr(module, '__version__', '?')}")
        except ImportError as error:
            lines.append(f"{name}: MISSING ({error})")
    return lines


def check_model(model_name: str = DEFAULT_MODEL) -> str:
    """Build the registry's entry the way the script does and count it."""
    sys.path.append(str(UPSTREAM / "examples"))  # the script's own sys.path.append
    from model_registry import get_baseline_models

    handle = get_baseline_models(img_size=(128, 256), in_chans=3, out_chans=14, drop_path_rate=0.1)[model_name]
    model = handle()
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected = MODELS[model_name][0]
    verdict = "not recorded yet" if expected is None else "as expected" if count == expected else f"EXPECTED {expected:,}"
    return f"{model_name}: {count:,} trainable parameters at 128x256 with 14 classes ({verdict})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the released torch-harmonics segmentation script unmodified; see the module docstring.",
        epilog="Every other argument is passed to the released script unchanged.",
    )
    parser.add_argument("--seed", type=int, default=RELEASED_SEED,
                        help=f"the seed for the script's two manual_seed({RELEASED_SEED}) calls (default: {RELEASED_SEED}, the released value)")
    parser.add_argument("--check", action="store_true", help="verify checksums, imports, extensions and the model, then exit")
    parser.add_argument("--allow-fallback", action="store_true",
                        help="start on a CUDA device even without the model's fast-path modules")
    parser.add_argument("--model", choices=sorted(MODELS), default=DEFAULT_MODEL,
                        help="the registry entry, for the check and the fast-path requirement; also pass it to the script as --models")
    args, passthrough = parser.parse_known_args(argv)

    mismatched = verify_checksums()
    if mismatched:
        print("run_released.py: vendored upstream files differ from SHA256SUMS: " + ", ".join(mismatched), file=sys.stderr)
        return 2
    put_shims_on_path()
    stale = SHIMS / "natten"
    if stale.exists():
        # A natten stand-in directly under shims/ (first on the path) would
        # shadow an installed natten; the stand-in belongs in shims/fallback/.
        print(f"run_released.py: stale natten stand-in at {stale}; it shadows any installed natten. Delete it: rm -rf {stale}",
              file=sys.stderr)
        return 2
    os.environ.setdefault("MPLBACKEND", "Agg")

    import torch

    print("== released script runner ==")
    print(f"script: {SCRIPT} (checksums verified)")
    for line in environment_report():
        print(line)
    extensions_present, extension_lines = extension_report(MODELS[args.model][1])
    for line in extension_lines:
        print(line)
    print(guard_plotting())
    print(accept_released_subset())
    if args.seed == RELEASED_SEED:
        print(f"seed: {RELEASED_SEED} (the released script's own)")
    else:
        print(f"seed: {args.seed} replaces the script's two manual_seed({RELEASED_SEED}) calls; the split generator stays {RELEASED_SEED}")

    if args.check:
        from torch_harmonics.examples import metrics  # served by the shim

        print(f"metrics module: {metrics.__file__}")
        for name in ("wandb", "natten", "torchvision.transforms.v2", "pandas", "matplotlib.pyplot", "tqdm"):
            importlib.import_module(name)
        print("script imports: ok")
        print(check_model(args.model))
        print("check: ok")
        return 0

    if torch.cuda.is_available() and not extensions_present and not args.allow_fallback:
        print(f"run_released.py: refusing to train {args.model} on a CUDA device without its fast-path modules "
              f"({', '.join(MODELS[args.model][1])}: the spherical attention falls back to Python loops, DISCO to a slow "
              "PyTorch path, and the planar attention cannot be built at all); install them into this environment "
              "(README) or pass --allow-fallback.", file=sys.stderr)
        return 3

    install_seed(args.seed)
    sys.argv = [str(SCRIPT), *passthrough]
    print(f"argv: {' '.join(sys.argv[1:])}")
    print("=" * 28, flush=True)
    runpy.run_path(str(SCRIPT), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
