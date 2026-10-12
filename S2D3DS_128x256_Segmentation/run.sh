#!/usr/bin/env bash
# The Stanford 2D3DS segmentation study: the SO(S^2)-SegFormer under the
# released protocol, three seeds.
#
# Usage, from the repository root (the script cds there itself):
#
#   S2D3DS_128x256_Segmentation/run.sh prepare-data                # download + convert to data/2D3DS/*.h5 (once)
#   S2D3DS_128x256_Segmentation/run.sh launch 0 1 2                # runs dealt round-robin over the GPUs given
#   S2D3DS_128x256_Segmentation/run.sh sequential 0                # every run on one GPU, one after another
#   S2D3DS_128x256_Segmentation/run.sh ladder 0 1 2 3 4 -- 1e-4 3e-4 1e-3 3e-3 1e-2   # one seed per rate, one GPU per rung
#   S2D3DS_128x256_Segmentation/run.sh ladder-evaluate -- 1e-4 3e-4 1e-3 3e-3 1e-2 --split val --device cuda:0
#   S2D3DS_128x256_Segmentation/run.sh evaluate --device cuda:0    # report split, every seed, last.pt
#   S2D3DS_128x256_Segmentation/run.sh tables                      # the LaTeX table from the evaluation JSON
#
# ARMS (default sos2_segformer; any config under configs/) and SEEDS
# (default "0 1 2") select the runs:
#   ARMS="sos2_segformer s2_segformer" SEEDS="0 1 2" S2D3DS_128x256_Segmentation/run.sh launch 0 1 2 3 4 5
# Extra `key=value` overrides after the GPU list reach every run, e.g.
#   S2D3DS_128x256_Segmentation/run.sh launch 0 1 2 training.lr=1e-4
# Runs land under outputs/S2D3DS_128x256_Segmentation/<arm>/seed<N>/ with
# logs in logs/S2D3DS_128x256_Segmentation/. A run directory that already
# holds metrics is refused; remove it to rerun, or continue it from its own
# checkpoints/last.pt with `training.resume=true`. `ladder` runs the first
# of SEEDS with the first of ARMS at every rate given, under
# outputs/S2D3DS_128x256_Segmentation/<arm>_lr<rate>/; `ladder-evaluate`
# scores those rungs' final checkpoints (--split val on the validation split,
# where the rung is chosen; never on the report split) and passes the rest of
# its arguments to evaluate.py. DRY_RUN=1 prints the
# deal and launches nothing.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/toolchain.sh
source scripts/toolchain.sh
# One CPU thread per process unless the caller says otherwise: the step is a
# stream of small kernel launches, and torch's default pool spin-waits between them.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
# The fused kernels (GM_FUSED_PAIRS=1, GM_FUSED_FACTORED=1 with GM_FACTORED_KERNEL=1,
# GM_FUSED_ATTENTION=1) compile on first use: the toolchain must be present before anything starts.
if [ "${GM_FUSED_PAIRS:-0}" = "1" ] || [ "${GM_FUSED_FACTORED:-0}" = "1" ] || [ "${GM_FUSED_ATTENTION:-0}" = "1" ]; then require_toolchain; fi

read -r -a ARMS <<< "${ARMS:-sos2_segformer}"
read -r -a SEEDS <<< "${SEEDS:-0 1 2}"
CONFIGS="$(pwd)/S2D3DS_128x256_Segmentation/configs"
mkdir -p logs/S2D3DS_128x256_Segmentation

run_one() {  # arm seed gpu [overrides...] -> runs in the foreground
  local arm="$1" seed="$2" gpu="$3"; shift 3
  local dir="outputs/S2D3DS_128x256_Segmentation/${arm}/seed${seed}" resume=0
  for override in "$@"; do
    case "$override" in
      hydra.run.dir=*) dir="${override#hydra.run.dir=}"; dir="${dir//\$\{seed\}/$seed}";;
      training.resume=true) resume=1;;
    esac
  done
  if [ -f "${dir}/logs/metrics.jsonl" ] && [ "$resume" = 0 ]; then
    echo "run.sh: ${dir} already holds metrics; remove it to rerun, or pass training.resume=true." >&2
    return 1
  fi
  local label; label="$(basename "$(dirname "$dir")")"  # the arm, or the ladder rung's directory
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
    uv run python -m experiments.stanford_2d3ds.train \
      -cp "$CONFIGS" -cn "$arm" "seed=${seed}" "$@" \
      > "logs/S2D3DS_128x256_Segmentation/${label}_seed${seed}.log" 2>&1
}

runs=()
for arm in "${ARMS[@]}"; do for seed in "${SEEDS[@]}"; do runs+=("${arm} ${seed}"); done; done

mode="${1:-}"; shift || true
case "$mode" in
  prepare-data)
    uv run python S2D3DS_128x256_Segmentation/prepare_data.py "$@"
    ;;
  launch)
    gpus=(); overrides=()
    for arg in "$@"; do
      if [[ "$arg" == *=* ]]; then overrides+=("$arg"); else gpus+=("$arg"); fi
    done
    if [ "${#gpus[@]}" -lt 1 ]; then echo "launch needs at least one GPU index" >&2; exit 2; fi
    n="${#gpus[@]}"
    for ((g = 0; g < n; g++)); do
      share=()
      for ((i = g; i < ${#runs[@]}; i += n)); do share+=("${runs[$i]}"); done
      echo "GPU ${gpus[$g]}: ${share[*]}${overrides[@]+ with ${overrides[*]}}"
      if [ "${DRY_RUN:-0}" = 1 ]; then continue; fi  # DRY_RUN=1 prints the deal and launches nothing
      nohup "$0" _worker "${gpus[$g]}" "${#share[@]}" "${share[@]}" ${overrides[@]+"${overrides[@]}"} \
        > "logs/S2D3DS_128x256_Segmentation/worker_gpu${gpus[$g]}.log" 2>&1 &
    done
    disown -a
    echo "${#runs[@]} runs dealt over ${n} GPUs; progress: ./scripts/progress.sh outputs/S2D3DS_128x256_Segmentation"
    ;;
  _worker)
    gpu="$1"; count="$2"; shift 2
    specs=("${@:1:$count}"); overrides=("${@:$((count + 1))}")
    for spec in "${specs[@]}"; do
      read -r arm seed <<< "$spec"
      echo "running ${arm} seed ${seed} on GPU ${gpu} ${overrides[*]:-}"
      run_one "$arm" "$seed" "$gpu" ${overrides[@]+"${overrides[@]}"}
    done
    ;;
  sequential)
    gpu="${1:?sequential needs one GPU index}"; shift
    for spec in "${runs[@]}"; do
      read -r arm seed <<< "$spec"
      echo "running ${arm} seed ${seed} on GPU ${gpu} $*"
      run_one "$arm" "$seed" "$gpu" "$@"
    done
    ;;
  ladder)
    # One seed (the first of SEEDS) of the first arm per learning rate,
    # one GPU per rung; extra key=value overrides reach every rung.
    gpus=(); rates=(); extra=(); section=gpus
    for arg in "$@"; do
      if [[ "$arg" == "--" ]]; then section=rates; continue; fi
      if [[ "$arg" == *=* ]]; then extra+=("$arg"); continue; fi
      if [[ "$section" == gpus ]]; then gpus+=("$arg"); else rates+=("$arg"); fi
    done
    if [ "${#rates[@]}" -lt 1 ] || [ "${#gpus[@]}" -ne "${#rates[@]}" ]; then
      echo "usage: $0 ladder <gpu>... -- <lr>...   (one GPU per rate)" >&2; exit 2
    fi
    for ((i = 0; i < ${#rates[@]}; i++)); do
      SEEDS="${SEEDS[0]}" ARMS="${ARMS[0]}" "$0" launch "${gpus[$i]}" "training.lr=${rates[$i]}" \
        "hydra.run.dir=outputs/S2D3DS_128x256_Segmentation/${ARMS[0]}_lr${rates[$i]}/seed\${seed}" \
        ${extra[@]+"${extra[@]}"}
    done
    ;;
  ladder-evaluate)
    # Score the rungs of `ladder` as arms named <arm>_lr<rate>, requiring the
    # ladder's one seed; everything after the rates reaches evaluate.py.
    rates=(); rest=(); section=none
    for arg in "$@"; do
      if [[ "$arg" == "--" ]]; then section=rates; continue; fi
      if [[ "$section" == rates && "$arg" != -* ]]; then rates+=("$arg"); else section=rest; rest+=("$arg"); fi
    done
    if [ "${#rates[@]}" -lt 1 ]; then echo "usage: $0 ladder-evaluate -- <lr>... [evaluate.py args]" >&2; exit 2; fi
    arms=""
    for rate in "${rates[@]}"; do arms+="${ARMS[0]}_lr${rate}=${ARMS[0]}_lr${rate},"; done
    uv run python S2D3DS_128x256_Segmentation/evaluate.py --arms "${arms%,}" --require-seeds "${SEEDS[0]}" \
      --output "outputs/S2D3DS_128x256_Segmentation/ladder_${ARMS[0]}.json" ${rest[@]+"${rest[@]}"}
    ;;
  evaluate)
    # Every arm must end up with exactly SEEDS scored; a later --require-seeds overrides.
    uv run python S2D3DS_128x256_Segmentation/evaluate.py --require-seeds "$(IFS=,; echo "${SEEDS[*]}")" "$@"
    ;;
  tables)
    uv run python S2D3DS_128x256_Segmentation/tables.py "$@"
    ;;
  *)
    echo "usage: $0 prepare-data [args] | launch <gpus...> [overrides] | ladder <gpus...> -- <lrs...> [overrides] | ladder-evaluate -- <lrs...> [args] | sequential <gpu> [overrides] | evaluate [args] | tables [args]" >&2; exit 2
    ;;
esac
