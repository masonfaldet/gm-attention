# Training

The infrastructure of the paper's experiments, in the order a run builds on
it: seeding and RNG capture, logging, checkpointing, a task-agnostic training
loop, learning-rate schedules, the shallow-water objective and metrics, run
provenance, and device resolution. Seeding, logging, checkpointing and the
loop know nothing about the task; the task enters through a
`train_step`/`eval_step` callable (see [Engine](#engine)). `losses.py` and
`metrics.py` are the shallow-water objective and metrics (the paper,
Section 5.1 and Appendix C.1). The segmentation loss and metrics (the paper,
Appendix D.1) are not part of this package; the Stanford 2D-3D-S experiment
supplies them itself.

`gm_attention.training` re-exports `Trainer`, `StepOutput`, `StepFn`,
`CheckpointManager`, `CheckpointMetadata`, `save_checkpoint`,
`load_checkpoint`, `Logger`, `MetricLogger`, `RngState`, `get_rng_state`,
`set_rng_state` and `set_seed`; every other name is imported from its module.
No module on this page reads a `GM_*` switch to change its behaviour;
[`runtime_switches`][gm_attention.training.provenance.runtime_switches] only
records them.

## Reproducibility

[`set_seed`][gm_attention.training.reproducibility.set_seed] seeds `random`,
`numpy` and `torch` (CPU and every CUDA device) in one call. It also sets
cuDNN's autotuner: enabled by default, which picks algorithms by timing and is
not bit-reproducible across runs, and disabled under `deterministic=True`,
which additionally requests `torch.use_deterministic_algorithms` and
`cudnn.deterministic`.
[`get_rng_state`][gm_attention.training.reproducibility.get_rng_state] and
[`set_rng_state`][gm_attention.training.reproducibility.set_rng_state] capture
and restore the `random`, `numpy`, `torch` CPU and (when CUDA is available)
per-device CUDA generator states as one picklable
[`RngState`][gm_attention.training.reproducibility.RngState]. A checkpoint
stores it by default, which is what lets a resumed run continue from the
captured generator state rather than only from the same starting seed.

Data seeding is separate from model seeding.
[`derive_data_seed`][gm_attention.training.reproducibility.derive_data_seed]
maps `(base_seed, split, index, epoch)` to one seed in $[0, 2^{63})$ by hashing
a UTF-8 serialization with BLAKE2b, so it is identical in every process
(Python's `hash` is not); `split` is a literal name such as `"train"` or
`"val"`, not a numeric offset, so two splits cannot collide by arithmetic.
[`forked_torch_seed`][gm_attention.training.reproducibility.forked_torch_seed]
is a context manager that reseeds `torch`'s CPU RNG (and the RNG of `device`,
if it is a CUDA device) and restores the surrounding state on exit, so data
generation that draws from the global RNG is a pure function of its sample
seed, whatever model initialization or dropout consumed beforehand. The model
seed and the data seeds (`train_seed`, `val_seed`, `test_seed`,
`normalization_seed`, `loader_order_seed`) are independent configuration
values.

::: gm_attention.training.reproducibility

## Logging

[`Logger`][gm_attention.training.logging.Logger] is the structural type
`Trainer` logs to: `log(metrics, *, step, epoch=None, split="train")` and
`close()`. [`MetricLogger`][gm_attention.training.logging.MetricLogger]
appends one JSON object per call to `<log_dir>/metrics.jsonl`, with the keys
`time`, `step`, `epoch`, `split` followed by the metrics, and mirrors a summary
line to the console through the `gm_attention.training` logger (at import,
a stream handler at level `INFO` is installed if the logger has none, and the
logger does not propagate to the root logger). It has no optional dependency, and its `close` releases nothing.
`Trainer` calls `log` once per epoch and split and never calls `close`.

::: gm_attention.training.logging

## Checkpointing

[`save_checkpoint`][gm_attention.training.checkpointing.save_checkpoint] and
[`load_checkpoint`][gm_attention.training.checkpointing.load_checkpoint] are
the primitives. A checkpoint is one `torch.save` dict holding the `state_dict`
of the model, of the optimizer and of the scheduler (the last two `None` when
not given), the run position (`epoch`, `global_step`), the `metrics`, the
resume decision state (`best_value`, `epochs_without_improvement`) and,
unless disabled, the [`RngState`][gm_attention.training.reproducibility.RngState].
It is written to a sibling `.tmp` path and renamed into place, so an
interrupted write never leaves a corrupt file. Loading restores model,
optimizer, scheduler and RNG state in place and returns a
[`CheckpointMetadata`][gm_attention.training.checkpointing.CheckpointMetadata];
it uses `weights_only=False`, so checkpoints are trusted files. A checkpoint
without `best_value` or `epochs_without_improvement` still loads.

[`CheckpointManager`][gm_attention.training.checkpointing.CheckpointManager]
is what [`Trainer`][gm_attention.training.engine.Trainer] drives once per
epoch. `save` always overwrites `last.pt`, writes `best.pt` when the
`monitor` metric strictly improves (`mode="min"` or `"max"`), and with
`keep_periodic_every=k` also writes `epoch_{epoch:04d}.pt` after every $k$-th
epoch. The improvement decision is made before anything is written, so
`last.pt` records the bar the epoch set, and `restore_best_value` re-arms the
comparison from a loaded checkpoint so that the first resumed epoch does not
overwrite `best.pt`.

::: gm_attention.training.checkpointing

## Engine

[`Trainer`][gm_attention.training.engine.Trainer] drives the
train/validate/checkpoint loop on a single device in full precision. A
`train_step`/`eval_step` callable supplied by the task turns
`(model, batch, device)` into a
[`StepOutput`][gm_attention.training.engine.StepOutput] (a scalar loss, plus
metrics to log); it must not call `backward()`, `optimizer.step()` or
`optimizer.zero_grad()`, which `Trainer` owns. `Trainer` supplies everything
generic around the step:

- Device placement. Anything with a `.to(device)` method is moved, recursing
  through dicts, lists and tuples (a namedtuple keeps its type), which covers
  [`SO2FeatureField`][gm_attention.representations.field_type.SO2FeatureField]
  and [`EdgeGeometry`][gm_attention.discretization.edge_geometry.EdgeGeometry]
  as well as plain tensors.
- Gradient accumulation. With `accumulation_steps=k`, each micro-batch loss is
  divided by the number of micro-batches in its group and back-propagated
  immediately, so the accumulated gradient is the average and only one
  micro-batch's activations are live at once; a shorter trailing group at the
  end of an epoch is averaged over its own length. `global_step` and
  `scheduler_step_on="step"` advance once per optimizer step.
- Gradient clipping and gradient norms. With `grad_clip_norm`, the gradients
  are clipped to that 2-norm before the optimizer step. The pre-clip norm of
  every optimizer step is recorded whether or not clipping is on, and the
  training metrics gain `grad_norm_mean`, `grad_norm_median`, `grad_norm_max`
  and, with clipping, `grad_clip_fraction`, the share of steps that were
  rescaled.
- Metric averaging. The metrics of an epoch are the unweighted means of the
  per-step values (every step counts once, whatever its batch size); a
  `"loss"` entry is filled in from `StepOutput.loss` when the step does not
  provide one. Validation runs in `eval` mode under `torch.no_grad()`.
- Scheduling. The scheduler is stepped once per epoch or once per optimizer
  step (`scheduler_step_on`). A metric-driven scheduler such as
  `ReduceLROnPlateau` takes `scheduler_metric`, the name of the metric whose
  epoch average is passed to `step`, read from the validation metrics when
  the run validates and from the training metrics otherwise.
- Validation, checkpointing and early stopping. With `validate_every=v` the
  validation pass runs on epochs divisible by $v$. On an epoch without
  validation, a metric-driven scheduler, the `best.pt` decision and early
  stopping are skipped rather than fed training metrics, so the monitored
  value always comes from one distribution, while `last.pt` is still written.
  Early stopping counts consecutive monitored epochs on which the checkpoint
  manager's metric did not improve, and so requires a `CheckpointManager`.
- Resuming. `fit(..., resume_from=path)` restores model, optimizer, scheduler
  and RNG state, the epoch, the global step and the early-stopping counter,
  re-arms the manager's best value, and continues at the next epoch.
  `max_epochs` is the total horizon, not a number of additional epochs.
  `restore_scheduler=False` is a warm restart: the optimizer state (Adam's
  moments) is restored, the scheduler given at construction starts fresh, and
  each parameter group's learning rate is set to the scheduler's `base_lrs`
  (so it needs an `LRScheduler`, not `ReduceLROnPlateau`).
- `on_epoch_start(epoch)` is called before each training epoch, which is the
  hook an epoch-indexed dataset needs.

::: gm_attention.training.engine

## Schedules

[`ScheduleSpec`][gm_attention.training.schedules.ScheduleSpec] holds an
experiment's `training.scheduler` block as plain values: `kind` (`"plateau"`
or `"cosine"`), `factor` and `patience` for the plateau rule, `eta_min` and
`restart` for the cosine. `from_mapping` reads a dict or an OmegaConf node;
construction rejects an unknown kind and a restart on anything but the cosine.
[`build_scheduler`][gm_attention.training.schedules.build_scheduler] returns
the scheduler and the metric it steps on: `ReduceLROnPlateau` in `"min"` mode
on the given `monitor` metric, or `CosineAnnealingLR` over exactly
`max_epochs` epochs, stepped once per epoch with no metric (`None`). These pair with
`Trainer`'s `scheduler_metric`. The paper trains the depth-1 models and the
reference with the plateau rule and the single-level models with the cosine
(Section 5.1), and the segmentation models with a cosine over 200 epochs
(Section 5.2).

The horizon is part of a cosine schedule: `T_max` is restored with the
scheduler state, so raising `max_epochs` on resume would not lengthen the
decay, and past `T_max` the cosine climbs back up.
[`check_resumed_scheduler`][gm_attention.training.schedules.check_resumed_scheduler]
therefore refuses a resume whose schedule kind differs from the checkpoint's
(read off the state's keys by
[`scheduler_kind_of`][gm_attention.training.schedules.scheduler_kind_of]), a
plain cosine resume whose `max_epochs` is not the end of the checkpoint's
cycle, and a warm restart (`restart=True`) from a cosine cycle that has not
finished or that adds no epochs. One kind change is accepted: a cosine
restart from a plateau checkpoint, which discards the scheduler state and
anneals over the added epochs. A restart needs a resumed run
(`completed_epochs >= 1`); its new cycle spans `max_epochs - completed_epochs`
epochs, with the optimizer's moments kept
(`Trainer.fit(..., restore_scheduler=False)`).

::: gm_attention.training.schedules

## Losses

The sphere-integrated losses are defined on any point set. `prediction` and
`target` have shape `(..., n_points, channels)`, and the constructor takes one
strictly positive quadrature weight per point and divides the weights by their
sum, so weights summing to $4\pi$ or to $1$ give the same loss. With
normalized weights $\tilde q_i$ and $C$ channels,
[`SquaredL2LossS2`][gm_attention.training.losses.SquaredL2LossS2] computes

$$
\frac{1}{C}\sum_{c=1}^{C}\sum_{i=1}^{n}\tilde q_i\,\big(\hat z_{c,i}-z_{c,i}\big)^2,
$$

averaged over the leading batch dimensions. This is the channel-averaged
squared $L^2$ loss of the paper (Appendix C.1) in its quadrature form, taken
with the weights of the grid the model is trained on.
[`L1LossS2`][gm_attention.training.losses.L1LossS2] replaces the square by an
absolute value, and [`L2LossS2`][gm_attention.training.losses.L2LossS2] takes
the square root of the integrated squared error of each channel before the
mean over batch and channels. Every channel is an independent scalar.
[`SphericalLossBase`][gm_attention.training.losses.SphericalLossBase] is the
abstract base: a subclass supplies the pointwise term and may transform the
integrated value before the final mean.

::: gm_attention.training.losses

## Metrics

[`PaperSphericalRegressionMetrics`][gm_attention.training.metrics.PaperSphericalRegressionMetrics]
returns three scalar tensors for a `(..., n_points, channels)` prediction and
target: `paper_squared_l2`, `paper_l1` and `paper_l2`, built from the three
losses above with the same quadrature weights. With $e_{ic}$ the error at
point $i$ in channel $c$, they are $\frac1C\sum_c\sum_i \tilde q_i\,e_{ic}^2$,
$\frac1C\sum_c\sum_i \tilde q_i\,|e_{ic}|$ and
$\frac1C\sum_c\big(\sum_i \tilde q_i\,e_{ic}^2\big)^{1/2}$; in particular
`paper_l2` is the mean of per-channel roots, not the root of
`paper_squared_l2`. `paper_squared_l2` evaluated with a model's own grid
weights is the native-grid squared $L^2$ error of the paper's shallow-water
tables (Section 5.1). The `paper_` prefix names the componentwise quantities of
Bonev et al. (2025), "Attention on the Sphere". The band-limited error of
Appendix C.1 is not computed here.

These quantities treat channels as independent scalars and are not gauge
invariant: a nodewise rotation of the tangent frames changes `paper_l1` and
`paper_l2` when the channels are vector components, and predictions holding
tangent-vector components must be expressed in one common basis before they
are compared across models.

::: gm_attention.training.metrics

## Provenance

[`code_revision`][gm_attention.training.provenance.code_revision] names the
code a run came from: `git rev-parse HEAD` of the working tree, with `+dirty`
appended if `git status --porcelain` reports changes; in a tree without
`.git`, the first token of a `GIT_REVISION` file at the root followed by
`+archive`; otherwise `"unknown"`. The root defaults to `REPOSITORY_ROOT`, the
repository root as located from `src/gm_attention/training/provenance.py`, and
can be passed explicitly.
[`runtime_switches`][gm_attention.training.provenance.runtime_switches] returns
the environment variables of the current process that start with one of
`SWITCH_PREFIXES` (`GM_` and `PYTORCH_CUDA_ALLOC_CONF`), sorted by name: the
settings that select an implementation path, as the process was launched.

::: gm_attention.training.provenance

## Device

[`resolve_device`][gm_attention.utils.device.resolve_device] returns
`torch.device(requested)` for an explicit device, and with `None` returns
`cuda` if CUDA is available and `cpu` otherwise.

::: gm_attention.utils.device
