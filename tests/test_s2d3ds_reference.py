"""The released-script harness: verbatim upstream files, one whitespace fix, the shims, the seed, the split."""

from __future__ import annotations

import difflib
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import torch

from experiments.stanford_2d3ds.data import split_indices

ROOT = pathlib.Path(__file__).resolve().parents[1]
REF = ROOT / "S2D3DS_128x256_Segmentation" / "reference"
SHIMS = REF / "shims"


def _runner():
    spec = importlib.util.spec_from_file_location("run_released", REF / "run_released.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _python(code: str, *, shims: bool) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    if shims:
        env["PYTHONPATH"] = str(SHIMS)
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=ROOT, check=False)


def test_the_vendored_upstream_files_are_the_recorded_bytes() -> None:
    """Every file under upstream/ hashes as SHA256SUMS records, and SHA256SUMS
    names every file there: the runner refuses anything else."""
    recorded = {}
    for line in (REF / "SHA256SUMS").read_text().splitlines():
        digest, _, name = line.strip().partition("  ")
        recorded[name] = digest
    present = {str(p.relative_to(REF)) for p in (REF / "upstream").rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    assert present == set(recorded)
    for name, digest in recorded.items():
        assert hashlib.sha256((REF / name).read_bytes()).hexdigest() == digest, name
    assert _runner().verify_checksums() == []


def test_the_metrics_fix_is_three_spaces_on_one_line() -> None:
    """metrics_fixed.py is the released module with one docstring opener
    indented; the released file itself does not compile."""
    released = (REF / "upstream" / "torch_harmonics" / "examples" / "metrics.py").read_text().splitlines()
    fixed = (REF / "metrics_fixed.py").read_text().splitlines()
    changes = [line for line in difflib.unified_diff(released, fixed, lineterm="", n=0) if line[:1] in "+-" and line[:3] not in ("+++", "---")]
    assert changes == ['- """', '+    """']
    with pytest.raises(IndentationError):
        compile("\n".join(released), "metrics.py", "exec")
    compile("\n".join(fixed), "metrics_fixed.py", "exec")


def test_the_shims_serve_the_released_names_in_a_fresh_interpreter() -> None:
    """With shims/ on PYTHONPATH the released import lines resolve: the
    metrics module from the fixed copy, wandb and natten from the stand-ins.
    Without it the released metrics module is the wheel's, which does not
    import (the reason the shim exists)."""
    probe = (
        "import torch_harmonics.examples.metrics as m, wandb, natten, json;"
        "print(json.dumps({'metrics': m.__file__, 'package': m.__package__, 'iou': m.IntersectionOverUnionS2.__name__,"
        " 'run': wandb.run, 'login': wandb.login(), 'natten': natten.NeighborhoodAttention2D.__name__}))"
    )
    with_shims = _python(probe, shims=True)
    assert with_shims.returncode == 0, with_shims.stderr
    seen = json.loads(with_shims.stdout.strip().splitlines()[-1])
    assert seen["metrics"] == str(REF / "metrics_fixed.py")
    assert seen["package"] == "torch_harmonics.examples"
    assert seen["iou"] == "IntersectionOverUnionS2" and seen["run"] is None and seen["login"] is False
    without = _python("import torch_harmonics.examples.metrics", shims=False)
    assert without.returncode != 0 and "IndentationError" in without.stderr
    stand_in = _python("import natten; natten.NeighborhoodAttention2D(3)", shims=True)
    assert stand_in.returncode != 0 and "natten is not installed" in stand_in.stderr


def test_the_seed_substitution_touches_only_the_two_literal_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner()
    original, original_cuda = torch.manual_seed, torch.cuda.manual_seed
    monkeypatch.setattr(torch, "manual_seed", original)
    monkeypatch.setattr(torch.cuda, "manual_seed", original_cuda)
    original(7)
    expected = torch.rand(4)
    runner.install_seed(7)
    torch.manual_seed(runner.RELEASED_SEED)  # the script's literal call
    torch.testing.assert_close(torch.rand(4), expected)
    torch.manual_seed(11)  # any other value passes through
    original(11)
    a = torch.rand(4)
    torch.manual_seed(11)
    torch.testing.assert_close(torch.rand(4), a)
    # the split's generator is a method on a generator object, untouched
    split = torch.Generator().manual_seed(333)
    fresh = torch.Generator().manual_seed(333)
    torch.testing.assert_close(torch.rand(4, generator=split), torch.rand(4, generator=fresh))
    monkeypatch.setattr(torch, "manual_seed", original)
    monkeypatch.setattr(torch.cuda, "manual_seed", original_cuda)


def test_released_seed_leaves_torch_alone() -> None:
    runner = _runner()
    before = (torch.manual_seed, torch.cuda.manual_seed)
    runner.install_seed(runner.RELEASED_SEED)
    assert (torch.manual_seed, torch.cuda.manual_seed) == before


@pytest.mark.parametrize("n", [1413, 100, 7])
def test_the_released_split_call_is_the_bundle_split(n: int) -> None:
    """The script's own lines against the bundle's split_indices."""
    rng = torch.Generator().manual_seed(333)
    train, test, valid = torch.utils.data.random_split(range(n), [0.95, 0.025, 0.025], generator=rng)
    ours = split_indices(n)
    assert (list(train), list(test), list(valid)) == (ours["train"], ours["val"], ours["report"])


def _fake_2d3ds_file(path: pathlib.Path, *, n: int, shape: tuple[int, int], classes: int = 14) -> None:
    """A file with the released converter's layout, random content."""
    import h5py

    rng = np.random.default_rng(0)
    labels = ["<UNK>", "beam", "board", "bookcase", "ceiling", "chair", "clutter", "column", "door", "floor", "sofa", "table", "wall", "window"][:classes]
    with h5py.File(path, "w") as f:
        rgb = rng.random((n, 3, *shape), dtype=np.float32)
        f.create_dataset("rgb", data=rgb)
        f.create_dataset("semantic", data=rng.integers(0, classes, size=(n, *shape), dtype=np.int64))
        f.create_dataset("depth", data=rng.random((n, *shape), dtype=np.float32))
        f.create_dataset("class_labels", data=labels)
        f.create_dataset("class_histogram", data=np.full(classes, 1.0 / classes, dtype=np.float32))
        for key, value in (("min_rgb", rgb.min((0, 2, 3))), ("max_rgb", rgb.max((0, 2, 3))), ("mean_rgb", rgb.mean((0, 2, 3))), ("std_rgb", rgb.std((0, 2, 3)))):
            f.create_dataset(key, data=value.astype(np.float32))
        for key in ("min_depth", "max_depth", "mean_depth", "std_depth"):
            f.create_dataset(key, data=np.float32(0.5))
        f.create_dataset("data_source_path", data=[f"pano/rgb/{i}.png" for i in range(n)])
        f.create_dataset("data_target_path", data=[f"pano/semantic/{i}.png" for i in range(n)])


@pytest.mark.slow
def test_the_runner_trains_the_released_script_on_a_tiny_file(tmp_path: pathlib.Path) -> None:
    """One epoch of the released script, released model, on a 40-panorama
    64x128 file through the runner: checkpoint, metrics pickle and the
    per-epoch summary the parser reads. CPU, so the pure-PyTorch fallback."""
    pytest.importorskip("torchvision")
    pytest.importorskip("pandas")
    data = tmp_path / "data"
    data.mkdir()
    _fake_2d3ds_file(data / "stanford_2d3ds_dataset_ds32.h5", n=40, shape=(64, 128))
    out = tmp_path / "out"
    env = dict(os.environ, MPLBACKEND="Agg")
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(REF / "run_released.py"), "--seed", "1", "--allow-fallback",
         "--data_path", str(data), "--data_downsampling_factor", "32", "--output_path", str(out),
         "--models", "s2nsegformer_sc2_layers4_e128", "--num_epochs", "1", "--batch_size", "8"],
        env=env, capture_output=True, text=True, cwd=ROOT, check=False,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert "Epoch 0 summary:" in result.stdout
    assert "seed: 1 replaces" in result.stdout
    run = out / "s2nsegformer_sc2_layers4_e128"
    assert (run / "checkpoint.pt").is_file() and (run / "output_data" / "metrics.pkl").is_file()
    sys.path.insert(0, str(REF))
    from parse_log import parse

    rows = parse(result.stdout)
    assert [r["split"] for r in rows] == ["train", "val"] and rows[1]["epoch"] == 0 and rows[1]["iou"] is not None


def _adopt():
    spec = importlib.util.spec_from_file_location("adopt", REF / "adopt.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _released_run(tmp_path: pathlib.Path, *, seed: int, epochs: int, model: str | None = None) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, dict[str, torch.Tensor]]:
    """A 40-panorama 64x128 file and what the released script leaves for
    one finished seed of ``model`` (the registry entry): the model's state
    dict, as the released registry builds it, and a log banner."""
    adopt = _adopt()
    model = model or adopt.DEFAULT_MODEL
    arm = adopt.MODELS[model][1]
    data = tmp_path / "data"
    data.mkdir()
    file = data / "stanford_2d3ds_dataset_ds32.h5"
    _fake_2d3ds_file(file, n=40, shape=(64, 128))
    # the registry imports the planar baselines, which import natten at load
    # time: the stand-in, last on the path as the sitecustomize places it
    for entry in (str(REF / "upstream" / "examples"), str(SHIMS / "fallback")):
        if entry not in sys.path:
            sys.path.append(entry)
    from model_registry import get_baseline_models

    released = get_baseline_models(img_size=(64, 128), in_chans=3, out_chans=14, drop_path_rate=0.1)[model]()
    state = {k: v.clone() for k, v in released.state_dict().items()}
    state["__parameters__"] = torch.tensor(sum(p.numel() for p in released.parameters() if p.requires_grad))
    run = tmp_path / "out" / arm / f"seed{seed}"
    (run / model / "output_data").mkdir(parents=True)
    torch.save({k: v for k, v in state.items() if k != "__parameters__"}, run / model / "checkpoint.pt")
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / f"{arm}_seed{seed}.log").write_text(
        f"argv: run_released.py --seed {seed} --data_path data --output_path out --models {model} "
        f"--num_epochs {epochs} --batch_size 8 --learning_rate 5e-4 --max_grad_norm 4.0 --amp_mode none\n"
        "Epoch 0 summary:\n"
    )
    return file, run, logs, state


@pytest.mark.parametrize("model", ["s2nsegformer_sc2_layers4_e128", "s2unet_sc2_layers4_e128"])
def test_adopt_makes_a_released_run_scorable_by_the_bundle_evaluator(tmp_path: pathlib.Path, model: str) -> None:
    """The released state dict (built by the registry itself) and log become
    the protocol and last.pt the evaluator reads: the bundle config for that
    entry builds the same model (strict load), the evaluator restores the
    same weights, sees a finished run at the banner's epoch count under the
    script's settings, finds no config drift, and scores the report split
    like any other arm."""
    from experiments.stanford_2d3ds.evaluation import (
        evaluate_run,
        hydra_config_drift,
        read_protocol,
        restore_experiment,
        run_progress,
    )

    adopt = _adopt()
    file, run, logs, state = _released_run(tmp_path, seed=1, epochs=3, model=model)
    overrides = ["--override", f"data.dataset_file={file}", "--override", "data.img_size=[64,128]"]
    assert adopt.main(["--model", model, "--root", str(run.parent), "--logs", str(logs), "--seeds", "1", *overrides]) == 0

    protocol = read_protocol(run)
    # the parameter count, not the state dict's size: the UNet also carries buffers
    assert protocol["model"] == adopt.MODELS[model][0] and protocol["real_parameter_count"] == int(state.pop("__parameters__"))
    assert protocol["trained_by"]["model"] == model
    assert (protocol["training_config"]["lr"], protocol["training_config"]["max_epochs"]) == (5e-4, 3)
    assert protocol["trained_by"]["settings_from"] == "the run log's argv banner"
    assert [e["event"] for e in protocol["history"]] == ["adopted", "finished"] and protocol["history"][-1]["epochs"] == 3
    progress = run_progress(run)
    assert (progress.complete, progress.run_epoch, progress.max_epochs) == (True, 2, 3)
    assert hydra_config_drift(run, protocol) == []

    _, experiment, splits, epoch = restore_experiment(run, checkpoint="last", device=torch.device("cpu"))
    assert epoch == 2 and splits.sizes() == {"train": 38, "val": 1, "report": 1}
    restored = experiment.model.state_dict()
    assert all(torch.equal(restored[k], state[k]) for k in state)

    with pytest.raises(FileExistsError):
        adopt.adopt(run, 1, log_text=None, overrides=overrides[1::2], force=False, model=model)
    assert adopt.main(["--model", model, "--root", str(run.parent), "--logs", str(logs), "--seeds", "1", "--force", *overrides]) == 0

    result = evaluate_run(run, arm="released", seed=1, device=torch.device("cpu"))
    assert result.complete and len(result.samples) == 1 and np.isfinite(result.samples[0].iou)


def test_adopt_falls_back_to_the_script_defaults_without_a_banner() -> None:
    adopt = _adopt()
    settings, source = adopt.training_settings(None)
    assert settings == adopt.SCRIPT_DEFAULTS and "defaults" in source
    settings, source = adopt.training_settings("argv: x --learning_rate 1e-3 --num_epochs 7\nEpoch 0 summary:\n")
    assert (settings["lr"], settings["max_epochs"], settings["batch_size"]) == (1e-3, 7, 8) and "banner" in source


def test_the_natten_stand_in_is_a_last_resort() -> None:
    """With the shims on PYTHONPATH the registry still imports (the stand-in
    answers ``natten``), but from the END of sys.path, so an installed natten
    would win; and the stand-in refuses to build the layer."""
    result = _python(
        "import sys, natten, json, pathlib; parent = str(pathlib.Path(natten.__file__).resolve().parents[1]);"
        " print(json.dumps({'file': natten.__file__, 'last': sys.path.index(parent) == len(sys.path) - 1}))",
        shims=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert "fallback" in payload["file"] and payload["last"]
    result = _python("import natten; natten.NeighborhoodAttention2D(16, kernel_size=(7, 7), num_heads=1)", shims=True)
    assert result.returncode != 0 and "stand-in" in result.stderr


def test_the_runner_knows_each_registry_entry() -> None:
    runner = _runner()
    assert set(runner.MODELS) == {"s2nsegformer_sc2_layers4_e128", "s2unet_sc2_layers4_e128", "nsegformer_sc2_layers4_e128"}
    adopt = _adopt()
    assert set(adopt.MODELS) == set(runner.MODELS)


def test_the_stand_in_never_counts_as_natten_present() -> None:
    """The report recognizes the stand-in by its marker, not its path, so a
    stale copy left at shims/natten by a push that cannot delete is still
    reported missing; and the runner refuses to start while one is there."""
    runner = _runner()
    sys.path.append(str(SHIMS / "fallback"))
    sys.modules.pop("natten", None)
    present, lines = runner.extension_report(("natten",))
    assert not present and lines == ["natten: MISSING (the stand-in cannot build the layer)"]
    stale = SHIMS / "natten"
    assert not stale.exists(), "the stand-in lives under shims/fallback/ only"


def test_parse_log_writes_nothing_for_a_log_without_epochs(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A refused launch leaves a log with the banner only; parse_log must not
    create the run directory the launch verb would then refuse to reuse."""
    sys.path.insert(0, str(REF))
    import parse_log

    log = tmp_path / "logs" / "s2_unet_released_seed0.log"
    log.parent.mkdir()
    log.write_text("== released script runner ==\nrun_released.py: refusing to train\n")
    monkeypatch.setattr(parse_log, "OUTPUTS", tmp_path / "outputs")
    assert parse_log.main([str(log)]) == 0
    assert not (tmp_path / "outputs").exists()
