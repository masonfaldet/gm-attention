"""Tests for experiments.shallow_water.models.

Covers the shipped shallow-water models: :class:`GmAttentionDepth1` (its
fixed input/output types, the scaled hidden spectrum, exact gauge
equivariance, the zero-initialized fine skip, gradient flow and gradient
checkpointing) and :func:`build_reference_transformer` (the S2 Transformer
reference with the paper's defaults). The last section checks the
``Trainer -> set_epoch -> dataset`` chain that the ``epoch_index`` sample
policy relies on, through the paper's
:class:`~experiments.shallow_water_paper.data.ScalarShallowWaterDataset`.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments.shallow_water.models import (
    GmAttentionDepth1,
    build_reference_transformer,
)
from experiments.shallow_water_paper.data import ScalarShallowWaterDataset
from gm_attention.discretization.frames import rotate_tangent_frames
from gm_attention.discretization.grids.icosphere import (
    IcosphereMultiresolution,
    build_icosphere_grid,
    build_icosphere_multiresolution,
)
from gm_attention.representations.field_type import SO2FeatureField, SO2FieldType
from gm_attention.training.engine import StepOutput, Trainer


# ==========================================================================
# build_reference_transformer
# ==========================================================================


def test_build_reference_transformer_forward_shape_and_gradients() -> None:
    """Four channels in (h, u, v, c), three out: Coriolis is supplied and
    never predicted, so the counts deliberately differ."""
    model = build_reference_transformer(nlat=12, nlon=24, embed_dim=8, num_layers=1, num_heads=1)

    x = torch.randn(2, 4, 12, 24)
    y = model(x)

    assert y.shape == (2, 3, 12, 24)
    assert torch.isfinite(y).all()

    y.sum().backward()
    assert any(p.grad is not None for p in model.parameters())


def test_build_reference_transformer_uses_paper_defaults_not_library_defaults() -> None:
    model = build_reference_transformer(nlat=12, nlon=24)
    assert model.embed_dim == 128  # library's own default is 256


def test_build_reference_transformer_is_finite_at_the_papers_real_config() -> None:
    """At the paper's actual embed_dim=128, num_layers=4, nlat=16 and
    nlat=32 both produce NaN on pure random input with fresh weights
    (see build_reference_transformer's docstring);
    nlat=64 does not. This is the smallest resolution that stays finite at
    that real config, so it is what actually exercises the shipped
    defaults, unlike the tiny/cheap config used by the shape/gradient test
    above."""
    torch.manual_seed(0)
    model = build_reference_transformer(nlat=64, nlon=128)

    y = model(torch.randn(1, 4, 64, 128))

    assert torch.isfinite(y).all()


# ==========================================================================
# epoch_index training: the Trainer -> set_epoch -> dataset chain
# ==========================================================================


def _policy_dataset(policy: str, split: str = "train") -> ScalarShallowWaterDataset:
    # The paper's three-scalar dataset at an icosphere's nodes: the shipped
    # point-major sampler with a set_epoch hook.
    return ScalarShallowWaterDataset(
        build_icosphere_grid(subdivisions=1, dtype=torch.float32).nodes,
        base_seed=1000,
        split=split,  # type: ignore[arg-type]
        sample_policy=policy,  # type: ignore[arg-type]
        nlat=12,
        nlon=24,
        lead_time_seconds=300.0,
        solver_substeps=2,
        num_examples=3,
    )


def test_trainer_advances_an_epoch_indexed_training_set_each_epoch() -> None:
    """The full chain, not just set_epoch in isolation: Trainer's
    on_epoch_start hook must actually reach the dataset, so each epoch
    trains on genuinely fresh trajectories."""
    dataset = _policy_dataset("epoch_index")
    seen: list[torch.Tensor] = []

    def step(model: nn.Module, sample, device: torch.device):
        seen.append(sample.input.clone())
        parameter = next(model.parameters())
        return StepOutput(loss=parameter.sum() * 0.0, metrics={"loss": 0.0})

    model = nn.Linear(2, 1)
    trainer = Trainer(
        model,
        torch.optim.SGD(model.parameters(), lr=0.0),
        step,
        on_epoch_start=dataset.set_epoch,
    )
    trainer.fit(DataLoader(dataset, batch_size=None), max_epochs=3)

    assert len(seen) == 9  # 3 epochs x 3 samples
    by_epoch = [seen[0:3], seen[3:6], seen[6:9]]

    # No sample repeats across epochs...
    for earlier, later in ((0, 1), (0, 2), (1, 2)):
        for a, b in zip(by_epoch[earlier], by_epoch[later], strict=True):
            assert not torch.allclose(a, b)

    # ... and within one epoch the three indices are distinct too.
    assert not torch.allclose(by_epoch[0][0], by_epoch[0][1])


def test_trainer_leaves_a_fixed_index_training_set_alone() -> None:
    """The contrast case, so the test above is not vacuous."""
    dataset = _policy_dataset("fixed_index")
    seen: list[torch.Tensor] = []

    def step(model: nn.Module, sample, device: torch.device):
        seen.append(sample.input.clone())
        parameter = next(model.parameters())
        return StepOutput(loss=parameter.sum() * 0.0, metrics={"loss": 0.0})

    model = nn.Linear(2, 1)
    trainer = Trainer(
        model,
        torch.optim.SGD(model.parameters(), lr=0.0),
        step,
        on_epoch_start=dataset.set_epoch,
    )
    trainer.fit(DataLoader(dataset, batch_size=None), max_epochs=2)

    for a, b in zip(seen[0:3], seen[3:6], strict=True):
        assert torch.equal(a, b)


def test_a_validation_split_stays_pinned_even_if_set_epoch_is_called() -> None:
    """val must be fixed_index: ReduceLROnPlateau, best.pt, and early
    stopping are all reading it, and a moving validation set makes every one
    of those decisions noise."""
    dataset = _policy_dataset("fixed_index", split="val")
    reference = dataset[0].input.clone()

    for epoch in range(4):
        dataset.set_epoch(epoch)
        assert torch.equal(dataset[0].input, reference)


def test_two_architectures_walk_the_same_epoch_indexed_schedule() -> None:
    """Fairness under epoch_index: sharing train_seed is enough, since the
    sample depends only on (seed, split, index, epoch) -- never on anything
    the model side of the run consumed."""
    first = _policy_dataset("epoch_index")
    second = _policy_dataset("epoch_index")

    for epoch in (0, 1, 7):
        first.set_epoch(epoch)
        torch.randn(5000)  # a differently-sized model initializing
        second.set_epoch(epoch)

        for index in range(3):
            assert torch.equal(first[index].input, second[index].input)


def test_reference_transformer_conditioned_channel_counts_are_explicit() -> None:
    """Spec 13: S2 receives (h, u, v, c) and predicts (h, u, v)."""
    model = build_reference_transformer(nlat=12, nlon=24, embed_dim=8, num_layers=1)

    assert (model.in_chans, model.out_chans) == (4, 3)


def test_reference_transformer_keeps_the_spectral_positional_embedding() -> None:
    """Kept on in every primary comparison -- removing it to make the
    reference less informed would answer a different question."""
    model = build_reference_transformer(nlat=12, nlon=24, embed_dim=8, num_layers=1)

    assert type(model.pos_embed).__name__ == "SpectralPositionEmbedding"


def test_reference_transformer_predicts_no_internal_residual() -> None:
    """The residual objective is applied externally over prognostic
    channels only; an internal one would add the Coriolis channel into a
    prognostic output."""
    model = build_reference_transformer(nlat=12, nlon=24, embed_dim=8, num_layers=1)

    assert model.residual_prediction is False


def test_reference_transformer_internal_grid_is_configurable() -> None:
    """R2 pins the latent grid to `equiangular` so it is the same *kind* of
    grid the GM side is matched against, not merely the same size."""
    default = build_reference_transformer(nlat=12, nlon=24, embed_dim=8, num_layers=1)
    matched = build_reference_transformer(
        nlat=12, nlon=24, embed_dim=8, num_layers=1, grid_internal="equiangular"
    )

    assert default.grid_internal == "legendre-gauss"
    assert matched.grid_internal == "equiangular"


# ==========================================================================
# GmAttentionDepth1 (spec 6)
# ==========================================================================


def _depth1_geometry(
    subdivisions: int = 3, finest_frames: torch.Tensor | None = None
) -> IcosphereMultiresolution:
    return build_icosphere_multiresolution(
        subdivisions, depth=1, coarsest_cutoff=0.55,
        local_cutoffs={0: 0.4}, finest_frames=finest_frames, dtype=torch.float64,
    )


def _depth1_model(
    geometry: IcosphereMultiresolution,
    *,
    base_field_type: dict[int, int] | None = None,
    multiplicity: int = 4,
    fine_skip_init: str = "zero",
    encode_mode: str = "learned_integral",
) -> GmAttentionDepth1:
    return GmAttentionDepth1(
        base_field_type=base_field_type or {0: 1, 1: 1, 2: 1},
        multiplicity=multiplicity, num_layers=2, num_heads=2,
        encode_cutoff=geometry.hops[0].refinement.coarse_edge_length,
        deep_cutoff=0.55, decode_refine_cutoff=0.4,
        fine_skip_init=fine_skip_init,  # type: ignore[arg-type]
        encode_mode=encode_mode,  # type: ignore[arg-type]
        dtype=torch.float64,
    ).eval()


def _conditioned_field(
    model: GmAttentionDepth1, n_nodes: int, seed: int = 3
) -> SO2FeatureField:
    generator = torch.Generator().manual_seed(seed)
    return SO2FeatureField(
        model.input_field_type,
        {
            0: torch.randn(n_nodes, 2, dtype=torch.float64, generator=generator),
            1: torch.randn(n_nodes, 1, 2, dtype=torch.float64, generator=generator),
        },
    )


def test_depth1_types_are_fixed_by_the_physics_not_the_config() -> None:
    """base_field_type sets only the hidden spectrum. Input and output are
    pinned, which is what makes conditioned-vs-prognostic a type error
    rather than a silent physical mistake."""
    geometry = _depth1_geometry()

    for base in ({0: 1, 1: 1, 2: 1}, {0: 1, 1: 1}, {0: 2, 1: 1}):
        model = _depth1_model(geometry, base_field_type=base)
        assert model.input_field_type == SO2FieldType({0: 2, 1: 1})
        assert model.output_field_type == SO2FieldType({0: 1, 1: 1})


def test_depth1_hidden_type_is_the_base_spectrum_scaled() -> None:
    model = _depth1_model(
        _depth1_geometry(), base_field_type={0: 2, 1: 1}, multiplicity=4
    )

    assert model.hidden_field_type == SO2FieldType({0: 8, 1: 4})


def test_depth1_enters_and_leaves_on_the_fine_level() -> None:
    """The point of the one-hop design: predict at the resolution the data
    is scored on, transform one level down."""
    geometry = _depth1_geometry()
    model = _depth1_model(geometry)
    n_fine = geometry.levels[0].grid.n_nodes

    output = model(_conditioned_field(model, n_fine), geometry)

    assert output.field_type == model.output_field_type
    assert output[0].shape == (n_fine, 1)
    assert output[1].shape == (n_fine, 1, 2)


def test_depth1_uses_the_real_l5_to_l4_levels() -> None:
    """Spec 6.2: fine 10,242, coarse 2,562, depth 1 -- and no model-local
    subdivisions override sneaking the run back down to L4."""
    geometry = build_icosphere_multiresolution(
        5, depth=1, coarsest_cutoff=0.27571504347419135,
        local_cutoffs={0: 0.13962634015954636}, dtype=torch.float32,
    )

    assert geometry.depth == 1
    assert geometry.levels[0].grid.n_nodes == 10_242
    assert geometry.levels[1].grid.n_nodes == 2_562


def test_depth1_rejects_a_geometry_of_the_wrong_depth() -> None:
    model = _depth1_model(_depth1_geometry())
    deep = build_icosphere_multiresolution(
        3, depth=2, coarsest_cutoff=0.8, local_cutoffs={0: 0.4}, dtype=torch.float64
    )

    with pytest.raises(ValueError, match="depth-1"):
        model(_conditioned_field(model, deep.levels[0].grid.n_nodes), deep)


def test_depth1_rejects_an_unconditioned_input() -> None:
    """Feeding the prognostic {0: 1, 1: 1} where the conditioned
    {0: 2, 1: 1} belongs is exactly the mistake the type split exists to
    catch."""
    geometry = _depth1_geometry()
    model = _depth1_model(geometry)
    n_fine = geometry.levels[0].grid.n_nodes
    prognostic = SO2FeatureField(
        SO2FieldType({0: 1, 1: 1}),
        {
            0: torch.zeros(n_fine, 1, dtype=torch.float64),
            1: torch.zeros(n_fine, 1, 2, dtype=torch.float64),
        },
    )

    with pytest.raises(ValueError, match="conditioned"):
        model(prognostic, geometry)


def test_depth1_is_exactly_gauge_equivariant() -> None:
    """The definitive correctness claim, for the shipped
    ``learned_integral`` encode mode."""
    geometry = _depth1_geometry()
    model = _depth1_model(geometry, encode_mode="learned_integral")
    n_fine = geometry.levels[0].grid.n_nodes
    x = _conditioned_field(model, n_fine)

    original = model(x, geometry)

    generator = torch.Generator().manual_seed(11)
    angles = torch.rand(n_fine, dtype=torch.float64, generator=generator) * (
        2 * math.pi
    )
    regauged_geometry = build_icosphere_multiresolution(
        3, depth=1, coarsest_cutoff=0.55, local_cutoffs={0: 0.4},
        dtype=torch.float64,
        finest_frames=rotate_tangent_frames(geometry.levels[0].frames, angles),
    )

    regauged = model(x.gauge_transform(angles), regauged_geometry)
    expected = original.gauge_transform(angles)

    for frequency in original.field_type.frequencies:
        assert torch.allclose(regauged[frequency], expected[frequency], atol=1e-10)


def test_depth1_residual_skip_starts_at_zero() -> None:
    """Under `residual` the training step adds the prognostic input
    outside the network, so a nonzero skip would be a second identity."""
    geometry = _depth1_geometry()
    model = _depth1_model(geometry, fine_skip_init="zero")
    x = _conditioned_field(model, geometry.levels[0].grid.n_nodes)

    skipped = model.fine_skip(x)

    assert torch.count_nonzero(skipped[0]) == 0
    assert torch.count_nonzero(skipped[1]) == 0


def test_depth1_gradients_reach_every_stage() -> None:
    geometry = _depth1_geometry()
    model = _depth1_model(geometry)
    x = _conditioned_field(model, geometry.levels[0].grid.n_nodes)

    model(x, geometry).to_flat().sum().backward()

    reached = {
        name.split(".")[0]
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and torch.isfinite(parameter.grad).all()
    }
    assert {"encode", "blocks", "decode"} <= reached


def test_gradient_checkpointing_changes_nothing_but_memory() -> None:
    """The property that makes checkpointing safe to enable by default: it
    rematerializes activations rather than approximating them, so gradients
    must be bit-identical, not merely close."""
    geometry = _depth1_geometry()
    gradients = {}

    for enabled in (False, True):
        torch.manual_seed(21)
        model = GmAttentionDepth1(
            base_field_type={0: 1, 1: 1, 2: 1}, multiplicity=4, num_layers=2,
            num_heads=2,
            encode_cutoff=geometry.hops[0].refinement.coarse_edge_length,
            deep_cutoff=0.55, decode_refine_cutoff=0.4,
            gradient_checkpointing=enabled, dtype=torch.float64,
        ).train()
        x = _conditioned_field(model, geometry.levels[0].grid.n_nodes, seed=22)
        model(x, geometry).to_flat().pow(2).sum().backward()
        gradients[enabled] = {
            name: parameter.grad.clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }

    assert gradients[False].keys() == gradients[True].keys()
    assert gradients[False]
    for name in gradients[False]:
        assert torch.equal(gradients[False][name], gradients[True][name]), name


def test_gradient_checkpointing_is_inactive_in_eval() -> None:
    """No backward pass to rematerialize for, so it would be pure overhead."""
    geometry = _depth1_geometry()
    model = _depth1_model(geometry)
    model.gradient_checkpointing = True
    model.eval()

    output = model(_conditioned_field(model, geometry.levels[0].grid.n_nodes), geometry)

    assert output.field_type == model.output_field_type
