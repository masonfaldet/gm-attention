#!/usr/bin/env bash
# The released baselines trained by torch-harmonics v0.8.0's own
# examples/segmentation/train.py on this bundle's data file, with the
# script's own split and normalization code. One seed per GPU.
#
#   S2D3DS_128x256_Segmentation/reference/run.sh check           # checksums, imports, extensions, model
#   S2D3DS_128x256_Segmentation/reference/run.sh launch 4 5 6    # seeds 0 1 2, one per GPU, in the background
#   SEEDS="0" S2D3DS_128x256_Segmentation/reference/run.sh launch 4
#   DRY_RUN=1 S2D3DS_128x256_Segmentation/reference/run.sh launch 4 5 6   # print the deal, launch nothing
#   S2D3DS_128x256_Segmentation/reference/run.sh adopt           # finished seeds -> protocol + checkpoints/last.pt
#   S2D3DS_128x256_Segmentation/reference/run.sh evaluate --device cuda:0   # the bundle's evaluator on them
#
# Arguments after the GPU list reach the released script unchanged
# (`--learning_rate 5e-5`, say); the script's own defaults are passed
# explicitly: 200 epochs, batches of 8, learning rate 5e-4, gradient
# clipping 4.0, no AMP. `adopt` writes, beside each finished seed's released
# files, the segmentation_protocol.json and checkpoints/last.pt the bundle's
# evaluate.py reads; `evaluate` then scores the arm per panorama on the
# report split into evaluation_released.json. Arguments after either verb
# reach adopt.py or evaluate.py (`adopt --seeds 0 1`, `evaluate --split val`).
# MODEL selects the registry entry for every verb.
#
# Environment: pyproject.toml's `reference` group in its own environment,
# with torch-harmonics' CUDA extensions built into it:
#   UV_PROJECT_ENVIRONMENT=.venv-ref uv sync --inexact --group reference
# The runner refuses to train without the attention extension.
set -euo pipefail
cd "$(dirname "$0")/../.."

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv-ref}"
export PYTHONPATH="$(pwd)/S2D3DS_128x256_Segmentation/reference/shims${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

# MODEL picks the registry entry (default: the S2 SegFormer); ARM is the arm name it is adopted as.
#   MODEL=s2unet_sc2_layers4_e128 S2D3DS_128x256_Segmentation/reference/run.sh launch 0 1 2
MODEL="${MODEL:-s2nsegformer_sc2_layers4_e128}"
case "$MODEL" in
  s2nsegformer_sc2_layers4_e128) ARM=s2_segformer_released ;;
  s2unet_sc2_layers4_e128) ARM=s2_unet_released ;;
  nsegformer_sc2_layers4_e128) ARM=r2_segformer_released ;;
  *) echo "run.sh: unknown MODEL ${MODEL}; one of s2nsegformer_sc2_layers4_e128, s2unet_sc2_layers4_e128, nsegformer_sc2_layers4_e128" >&2; exit 2 ;;
esac
RUNNER=S2D3DS_128x256_Segmentation/reference/run_released.py
read -r -a SEEDS <<< "${SEEDS:-0 1 2}"

if [ ! -d "$UV_PROJECT_ENVIRONMENT" ]; then
  echo "run.sh: no environment at ${UV_PROJECT_ENVIRONMENT}; create it with" >&2
  echo "  UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT} uv sync --inexact --group reference" >&2
  exit 1
fi

mode="${1:-}"; shift || true
case "$mode" in
  check)
    uv run --no-sync python "$RUNNER" --check --model "$MODEL" "$@"
    ;;
  launch)
    gpus=()
    while [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; do gpus+=("$1"); shift; done
    if [ "${#gpus[@]}" -lt 1 ]; then echo "launch needs at least one GPU index" >&2; exit 2; fi
    mkdir -p logs/S2D3DS_128x256_Segmentation
    for i in "${!SEEDS[@]}"; do
      seed="${SEEDS[$i]}"; gpu="${gpus[$((i % ${#gpus[@]}))]}"
      out="outputs/S2D3DS_128x256_Segmentation/${ARM}/seed${seed}"
      log="logs/S2D3DS_128x256_Segmentation/${ARM}_seed${seed}.log"
      if [ -e "$out" ]; then
        echo "run.sh: ${out} exists; remove it to rerun." >&2; exit 1
      fi
      cmd=(uv run --no-sync python "$RUNNER" --seed "$seed" --model "$MODEL"
           --data_path data/2D3DS --output_path "$out" --models "$MODEL"
           --num_epochs 200 --batch_size 8 --learning_rate 5e-4 --max_grad_norm 4.0 --amp_mode none "$@")
      echo "seed ${seed} on GPU ${gpu}: ${cmd[*]} > ${log}"
      if [ "${DRY_RUN:-0}" = 1 ]; then continue; fi
      CUDA_VISIBLE_DEVICES="$gpu" nohup "${cmd[@]}" > "$log" 2>&1 &
      echo "  pid $!"
    done
    ;;
  adopt)
    uv run --no-sync python S2D3DS_128x256_Segmentation/reference/adopt.py --model "$MODEL" "$@"
    ;;
  evaluate)
    # The adopted runs through the bundle's evaluator, in this environment:
    # report split, last.pt, exactly SEEDS, unless the arguments say otherwise.
    uv run --no-sync python S2D3DS_128x256_Segmentation/evaluate.py \
      --arms "${ARM}=${ARM}" --require-seeds "$(IFS=,; echo "${SEEDS[*]}")" \
      --output outputs/S2D3DS_128x256_Segmentation/evaluation_released.json "$@"
    ;;
  *)
    echo "usage: $0 check [runner args] | launch <gpu>... [script args] | adopt [adopt.py args] | evaluate [evaluate.py args]" >&2; exit 2
    ;;
esac
