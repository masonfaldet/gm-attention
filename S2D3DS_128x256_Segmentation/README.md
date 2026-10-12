# Stanford 2D-3D-S semantic segmentation at 128x256 (Table 3)

The SO(S^2)-SegFormer and its scalar-only ablation against the released spherical SegFormer, the planar SegFormer
and the spherical UNet, under the released training protocol, three seeds each, scored on the 35-panorama report split one panorama
at a time with the released loss and metrics.

## Data

`prepare_data.py` downloads the Stanford 2D-3D-S panoramas and runs torch-harmonics' converter unchanged:
1,413 RGB panoramas with semantic labels, downsampled to 128x256 (factor 16), 14 classes (13 objects plus the
released reader's unknown class), one HDF5 file (`data/2D3DS/stanford_2d3ds_dataset_ds16.h5`, 1.1 GB; the seven
`no_xyz` tarballs are 224 GB and are removed as they are converted). The released script's own split and
normalization are reproduced exactly: a `torch.Generator` seeded with 333 splits 95 / 2.5 / 2.5 (1343 / 35 / 35
panoramas); the released "test" split (scored every epoch) is this bundle's `val`, its "valid" split (scored
once at the end) is `report`; per-channel statistics come from the training split. Every run records the
file's SHA-256, the split indices and the normalization in `segmentation_protocol.json`, and the evaluator
refuses to score a model on data that does not match.

## Arms

| config | model | parameters | protocol |
|---|---|---|---|
| `sos2_segformer` | SO(S^2)-SegFormer: `configs/model/sos2_segformer.yaml` (stages on icosphere levels 5, 4, 3, 2 with types 6 rho_0 + 2 rho_1 + 2 rho_2, 12 + 4 + 4, 18 + 6 + 6, 24 + 8 + 8; depths 3, 4, 6, 3; heads 1, 1, 2, 4; steerable value kernels; normalized depthwise mix and readout convolutions; gravity conditioning and readout heads at every stage; top-down fusion decoder with a pointwise stem and an MLP head) | 593,518 | AdamW 3e-3, weight decay 0.1, cosine over 200 epochs, batch 8 (four panoramas per forward, two accumulated), grad clip 4.0, drop-path 0.2 |
| `sos2_segformer_scalar` | the scalar-only ablation: `configs/model/sos2_segformer_scalar.yaml`, every channel of every stage a scalar (14, 26, 38, 48 rho_0, widths chosen to match the parameter count; depths 3, 4, 7, 3), so beyond the first merge every operator is isotropic; the per-stage gravity conditioning and projection are off (they need directional channels) and the tangent gravity field enters instead as one rho_1 and one rho_2 input channel beside the RGB; the latitude conditioning, the gravity readout heads and everything else as above | 599,158 | AdamW 1.5e-3, weight decay 0.1, cosine over 200 epochs, batch 8 (two panoramas per forward, four accumulated), grad clip 4.0, drop-path 0.1 |
| `s2_segformer_released` | S^2 SegFormer (local), the registry's `s2nsegformer_sc2_layers4_e128`, trained by the released script (`reference/run.sh`, `MODEL=s2nsegformer_sc2_layers4_e128`) | 609,246 | the released script's defaults: AdamW 5e-4, weight decay 0.1, cosine over 200 epochs, batch 8, grad clip 4.0 |
| `r2_segformer_released` | planar SegFormer, `nsegformer_sc2_layers4_e128`, trained by the released script (`MODEL=nsegformer_sc2_layers4_e128`) | 692,398 | the same |
| `s2_unet_released` | spherical UNet, `s2unet_sc2_layers4_e128`, trained by the released script (`MODEL=s2unet_sc2_layers4_e128`) | 741,374 | the same |

The loss is the quadrature-weighted cross-entropy without class weights. The three references are trained by
torch-harmonics' own `examples/segmentation/train.py`, vendored verbatim under `reference/upstream/`
(`reference/README.md`), and adopted into this bundle's run layout so that all five arms go through the same
evaluator (`configs/s2_segformer.yaml`, `r2_segformer.yaml` and `s2_unet.yaml` run the same models through this
bundle's own trainer and are not the reported runs). The SO(S^2)-SegFormer and its ablation train with the fused kernels (`GM_FACTORED_KERNEL=1 GM_FUSED_FACTORED=1
GM_FUSED_ATTENTION=1`; the same function up to floating-point summation order), which need `nvcc` and a C++17 `g++`:
the factored steerable kernel (`src/gm_attention/nn/fused/factored.h`) evaluates the attention values, the merges and the
readouts on the edges, sums each target's weighted messages and takes the backward through the target index in the same
op; the attention's scores and softmax and the blocks' depthwise mix are fused ops of their own (`fused/attention.h`). The
ablation's merges into its 38- and 48-scalar stages exceed the factored kernel's 32-copy bound and run the same factored
order in plain PyTorch.

## Learning rate and drop-path (seed 0, validation split)

Rung selection uses the validation split, never the report split. Per rung: the lowest validation
cross-entropy over the epochs, and the mean validation IoU over the last ten epochs.

SO(S^2)-SegFormer: three rates at the model's drop-path 0.2, drop-path 0.3 at 3e-3, and a second run of the
3e-3 rung (launched as a separate "drop-path 0.2" rung under the wrong belief that the others ran at 0.1; its
resolved configuration is identical to the 3e-3 rung's, so the pair shows the run-to-run spread):

| rung | val CE | val IoU |
|---|---|---|
| lr 1.5e-3, drop-path 0.2 | 0.670 | 0.653 |
| lr 3e-3, drop-path 0.2 | 0.648 | 0.674 |
| lr 3e-3, drop-path 0.2 (second run; the seed-fill parent) | 0.654 | **0.677** |
| lr 6e-3, drop-path 0.2 | **0.640** | 0.672 |
| lr 3e-3, drop-path 0.3 | 0.662 | 0.668 |

The reported configuration is lr 3e-3 with drop-path 0.2, the rung with the highest validation IoU (the
reported metric; its three seeds 0.677 / 0.683 / 0.677); on validation cross-entropy the 6e-3 rung is lower.
The remaining settings (the multiplicity ladder and heads, depths [3, 4, 6, 3], the normalized convolutions,
the gravity heads, the fusion decoder, drop-path 0.2) were fixed by an earlier search over a variant of this
model with a 48-scalar last stage, which is not reported.

Scalar-only ablation, three rungs at drop-path 0.1: lr 1.5e-3 (CE 0.705, IoU 0.642; reported, its three
seeds 0.642 / 0.624 / 0.613), 3e-3 (0.737, 0.601), 6e-3 (0.869, 0.577). The references use their released
defaults and were not searched.

## Running it

```bash
S2D3DS_128x256_Segmentation/run.sh prepare-data                    # once; hours
OMP_NUM_THREADS=1 GM_FACTORED_KERNEL=1 GM_FUSED_FACTORED=1 GM_FUSED_ATTENTION=1 \
  S2D3DS_128x256_Segmentation/run.sh launch 0 1 2                  # sos2_segformer, seeds 0 1 2
OMP_NUM_THREADS=1 GM_FACTORED_KERNEL=1 GM_FUSED_FACTORED=1 GM_FUSED_ATTENTION=1 \
  ARMS=sos2_segformer_scalar S2D3DS_128x256_Segmentation/run.sh launch 3 4 5
S2D3DS_128x256_Segmentation/run.sh evaluate --device cuda:0 \
  --arms sos2_segformer=sos2_segformer,sos2_segformer_scalar=sos2_segformer_scalar   # outputs/S2D3DS_128x256_Segmentation/evaluation.json
```

The references, in the reference environment (`reference/README.md`), one `MODEL` at a time, each evaluation
into its own file (the driver's default output name is the same for every model):

```bash
MODEL=s2nsegformer_sc2_layers4_e128 S2D3DS_128x256_Segmentation/reference/run.sh launch 0 1 2
MODEL=s2nsegformer_sc2_layers4_e128 S2D3DS_128x256_Segmentation/reference/run.sh adopt
MODEL=s2nsegformer_sc2_layers4_e128 S2D3DS_128x256_Segmentation/reference/run.sh evaluate --device cuda:0 \
  --output outputs/S2D3DS_128x256_Segmentation/evaluation_released.json
MODEL=nsegformer_sc2_layers4_e128 S2D3DS_128x256_Segmentation/reference/run.sh launch 3 4 5     # then adopt and evaluate as above,
MODEL=s2unet_sc2_layers4_e128 S2D3DS_128x256_Segmentation/reference/run.sh launch 6 7 8         # --output evaluation_r2_segformer_released.json
                                                                                                   # and evaluation_s2_unet_released.json
S2D3DS_128x256_Segmentation/run.sh tables --input outputs/S2D3DS_128x256_Segmentation/evaluation.json \
  outputs/S2D3DS_128x256_Segmentation/evaluation_released.json outputs/S2D3DS_128x256_Segmentation/evaluation_r2_segformer_released.json \
  outputs/S2D3DS_128x256_Segmentation/evaluation_s2_unet_released.json                            # Table 3 beside evaluation.json
```

`tables.py` reads every evaluation JSON it is given. The rate search: `run.sh ladder <gpus> -- <rates>` trains
one seed per rate under `outputs/S2D3DS_128x256_Segmentation/<arm>_lr<rate>/` (drop-path rungs the same way
through `launch` with `model.architecture.drop_path_rate=...` and `hydra.run.dir=...`); the table above is read
from each rung's training log (`logs/metrics.jsonl`, the rows with `"split": "val"`: the lowest cross-entropy
over the epochs and the mean IoU over the last ten). `ladder-evaluate -- <rates> --split val` scores the rungs'
final checkpoints per panorama on the validation split, a different statistic of the same split. Rung selection
uses the validation split, never the report split.
