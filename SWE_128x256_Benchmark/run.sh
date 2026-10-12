#!/usr/bin/env bash
# Reproduce the SWE 128x256 benchmark: three arms x three seeds, nine runs.
#
# Usage, from the repository root (the script cds there itself):
#
#   SWE_128x256_Benchmark/run.sh launch 0 1 2 3 4 5 6 7 8   # one GPU per run, in parallel
#   SWE_128x256_Benchmark/run.sh sequential 0                # all nine on one GPU, one after another
#   SWE_128x256_Benchmark/run.sh evaluate [args]             # one-step and five-step, every seed, native and spectral
#   SWE_128x256_Benchmark/run.sh tables [args]               # LaTeX tables from the evaluation JSON
#
# Order of runs (and of the GPU list): gm_attention_depth1 seeds 0 1 2,
# sfno seeds 0 1 2, s2_transformer seeds 0 1 2. Each run reads its arm's
# config from SWE_128x256_Benchmark/configs (which draws the model, grid,
# objective and training groups from configs/) and writes to
# outputs/SWE_128x256_Benchmark/<arm>/seed<N>/ with logs in
# logs/SWE_128x256_Benchmark/. A run directory that already holds metrics
# is refused; remove it to rerun.
#
# Optional environment:
#   SWE_CACHE=/path/to/spectral_cache  on-disk cache of solved states, shared
#                                      across runs (results identical without)
#   UV_PROJECT_ENVIRONMENT=...         an environment holding torch-harmonics'
#                                      compiled CUDA extensions (a faster S2 Transformer)
set -euo pipefail

# GM_CUTOFF_RULE=legacy: this bundle's checkpoints aggregate over the edges of
# the legacy cutoff rule, so training, evaluation and resumes run under it.
export GM_CUTOFF_RULE="${GM_CUTOFF_RULE:-legacy}"
cd "$(dirname "$0")/.."

ARMS=(gm_attention_depth1 sfno s2_transformer)
SEEDS=(0 1 2)
CONFIGS="$(pwd)/SWE_128x256_Benchmark/configs"
mkdir -p logs/SWE_128x256_Benchmark

cache_override=()
if [ -n "${SWE_CACHE:-}" ]; then cache_override=("data.spectral_cache_dir=${SWE_CACHE}"); fi

run_one() {  # arm seed gpu -> runs in the foreground
  local arm="$1" seed="$2" gpu="$3"
  local dir="outputs/SWE_128x256_Benchmark/${arm}/seed${seed}"
  if [ -f "${dir}/logs/metrics.jsonl" ]; then
    echo "run.sh: ${dir} already holds metrics; remove it to rerun." >&2
    return 1
  fi
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
    uv run python -m experiments.shallow_water_paper.train \
      -cp "$CONFIGS" -cn "$arm" "seed=${seed}" ${cache_override[@]+"${cache_override[@]}"} \
      > "logs/SWE_128x256_Benchmark/${arm}_seed${seed}.log" 2>&1
}

mode="${1:-}"; shift || true
case "$mode" in
  launch)
    if [ "$#" -lt 9 ]; then echo "launch needs nine GPU indices, one per run" >&2; exit 2; fi
    gpus=("$@"); i=0
    for arm in "${ARMS[@]}"; do for seed in "${SEEDS[@]}"; do
      gpu="${gpus[$i]}"
      echo "launching ${arm} seed ${seed} on GPU ${gpu}"
      if [ "${DRY_RUN:-0}" = 1 ]; then i=$((i + 1)); continue; fi  # DRY_RUN=1 prints the deal and launches nothing
      nohup "$0" _one "$arm" "$seed" "$gpu" \
        > "logs/SWE_128x256_Benchmark/launch_${arm}_seed${seed}.log" 2>&1 &
      i=$((i + 1))
    done; done
    disown -a
    echo "nine runs launched; progress: ./scripts/progress.sh outputs/SWE_128x256_Benchmark"
    ;;
  _one)
    run_one "$1" "$2" "$3"
    ;;
  sequential)
    gpu="${1:?sequential needs one GPU index}"
    for arm in "${ARMS[@]}"; do for seed in "${SEEDS[@]}"; do
      echo "running ${arm} seed ${seed} on GPU ${gpu}"
      run_one "$arm" "$seed" "$gpu"
    done; done
    ;;
  evaluate)
    # Extra arguments pass through (e.g. --device cuda:3 --num-examples 64).
    uv run python SWE_128x256_Benchmark/evaluate.py "$@"
    ;;
  tables)
    # Extra arguments pass through (e.g. --input other_evaluation.json).
    uv run python SWE_128x256_Benchmark/tables.py "$@"
    ;;
  *)
    echo "usage: $0 launch <9 gpus> | sequential <gpu> | evaluate [args] | tables [args]" >&2; exit 2
    ;;
esac
