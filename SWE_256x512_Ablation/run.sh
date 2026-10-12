#!/usr/bin/env bash
# Reproduce the depth-1 ablation on the SWE 256x512 benchmark: four arms x
# three seeds, twelve runs.
#
# Usage, from the repository root (the script cds there itself):
#
#   SWE_256x512_Ablation/run.sh launch 0 1 2 3          # runs dealt round-robin over the GPUs given
#   SWE_256x512_Ablation/run.sh sequential 0            # all twelve on one GPU, one after another
#   SWE_256x512_Ablation/run.sh evaluate [args]         # one-step and five-step, every seed, on the test split
#   SWE_256x512_Ablation/run.sh tables [args]           # the LaTeX table from the evaluation JSON
#
# Order of runs: gm_attention_full seeds 0 1 2, gm_convolution_full seeds
# 0 1 2, gm_attention_rho01 seeds 0 1 2, gm_attention_rho0 seeds 0 1 2.
# `launch` deals them over the GPUs listed (run i goes to GPU i mod n) and
# each GPU works through its share in the background, one run at a time.
# Each run reads its arm's config from SWE_256x512_Ablation/configs (which
# draws the model, grid, objective and training groups from configs/) and
# writes to outputs/SWE_256x512_Ablation/<arm>/seed<N>/ with logs in
# logs/SWE_256x512_Ablation/. A run directory that already holds metrics
# is refused; remove it to rerun.
#
# Optional environment:
#   SWE_CACHE=/path/to/spectral_cache  on-disk cache of solved states, shared
#                                      across runs (results identical without)
set -euo pipefail

# GM_CUTOFF_RULE=legacy: this bundle's checkpoints aggregate over the edges of
# the legacy cutoff rule, so training, evaluation and resumes run under it.
export GM_CUTOFF_RULE="${GM_CUTOFF_RULE:-legacy}"
cd "$(dirname "$0")/.."

ARMS=(gm_attention_full gm_convolution_full gm_attention_rho01 gm_attention_rho0)
SEEDS=(0 1 2)
CONFIGS="$(pwd)/SWE_256x512_Ablation/configs"
mkdir -p logs/SWE_256x512_Ablation

cache_override=()
if [ -n "${SWE_CACHE:-}" ]; then cache_override=("data.spectral_cache_dir=${SWE_CACHE}"); fi

run_one() {  # arm seed gpu -> runs in the foreground
  local arm="$1" seed="$2" gpu="$3"
  local dir="outputs/SWE_256x512_Ablation/${arm}/seed${seed}"
  if [ -f "${dir}/logs/metrics.jsonl" ]; then
    echo "run.sh: ${dir} already holds metrics; remove it to rerun." >&2
    return 1
  fi
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
    uv run python -m experiments.shallow_water_paper.train \
      -cp "$CONFIGS" -cn "$arm" "seed=${seed}" ${cache_override[@]+"${cache_override[@]}"} \
      > "logs/SWE_256x512_Ablation/${arm}_seed${seed}.log" 2>&1
}

runs=()
for arm in "${ARMS[@]}"; do for seed in "${SEEDS[@]}"; do runs+=("${arm} ${seed}"); done; done

mode="${1:-}"; shift || true
case "$mode" in
  launch)
    if [ "$#" -lt 1 ]; then echo "launch needs at least one GPU index" >&2; exit 2; fi
    gpus=("$@"); n="${#gpus[@]}"
    for ((g = 0; g < n; g++)); do
      share=()
      for ((i = g; i < ${#runs[@]}; i += n)); do share+=("${runs[$i]}"); done
      [ "${#share[@]}" -eq 0 ] && continue  # more GPUs than runs: no worker
      echo "GPU ${gpus[$g]}: ${share[*]}"
      if [ "${DRY_RUN:-0}" = 1 ]; then continue; fi  # DRY_RUN=1 prints the deal and launches nothing
      nohup "$0" _worker "${gpus[$g]}" "${share[@]}" \
        > "logs/SWE_256x512_Ablation/worker_gpu${gpus[$g]}.log" 2>&1 &
    done
    disown -a
    echo "${#runs[@]} runs dealt over ${n} GPUs; progress: ./scripts/progress.sh outputs/SWE_256x512_Ablation"
    ;;
  _worker)
    gpu="$1"; shift
    for spec in "$@"; do
      read -r arm seed <<< "$spec"
      echo "running ${arm} seed ${seed} on GPU ${gpu}"
      run_one "$arm" "$seed" "$gpu"
    done
    ;;
  sequential)
    gpu="${1:?sequential needs one GPU index}"
    for spec in "${runs[@]}"; do
      read -r arm seed <<< "$spec"
      echo "running ${arm} seed ${seed} on GPU ${gpu}"
      run_one "$arm" "$seed" "$gpu"
    done
    ;;
  evaluate)
    # Extra arguments pass through (e.g. --device cuda:3 --num-examples 64).
    uv run python SWE_256x512_Ablation/evaluate.py "$@"
    ;;
  tables)
    # The LaTeX table from evaluate's JSON; extra arguments pass through.
    uv run python SWE_256x512_Ablation/tables.py "$@"
    ;;
  *)
    echo "usage: $0 launch <gpus...> | sequential <gpu> | evaluate [args] | tables [args]" >&2; exit 2
    ;;
esac
