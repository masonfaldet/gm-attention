#!/usr/bin/env bash
# Reproduce the zero-shot grid-transfer study on the SWE 256x512 benchmark:
# the single-level L5 arm, SFNO and the single-level S2 Transformer, three
# seeds each, then every seed of the L5 arm evaluated unchanged on grids it
# was not trained on.
#
# Usage, from the repository root (the script cds there itself):
#
#   SWE_256x512_ZeroShot/run.sh launch 0 1 2                  # the nine runs dealt round-robin over the GPUs given
#   SWE_256x512_ZeroShot/run.sh sequential 0                  # all nine on one GPU, one after another
#   SWE_256x512_ZeroShot/run.sh arm gm_l5_100 3 4 5           # one arm's three seeds dealt over the GPUs given
#   SWE_256x512_ZeroShot/run.sh evaluate [args]               # the L5 arm on its grids and the references, one and five steps
#   SWE_256x512_ZeroShot/run.sh evaluate-parallel 8 9 10 11   # the same as (grid, seed) jobs dealt over the GPUs given
#   SWE_256x512_ZeroShot/run.sh merge [out.json]              # the parts into one evaluation JSON
#   SWE_256x512_ZeroShot/run.sh tables [args]                 # the LaTeX table from the evaluation JSON
#
# evaluate-parallel deals its jobs over the GPUs given (fewer GPUs queue
# them round-robin): the L4 and L5 grids with every seed, each seed of L6
# and of the 128x256 Legendre-Gauss grid on its own, and the references;
# each job writes outputs/SWE_256x512_ZeroShot/parts/<job>.json and `merge`
# joins them. EVAL_ARGS passes extra arguments to every job; EVAL_JOBS
# names a subset of the jobs (coarse, l5_seed0..2, l6_seed0..2,
# lg128_seed0..2, references), in the order given. With ARM=<arm> the GM
# rows come from outputs/SWE_256x512_ZeroShot/<arm>/ (--arms
# gm_l5_100=<arm>), the parts go to parts_<arm>/ and `merge` writes
# evaluation_<arm>.json; without ARM, `merge` joins the default parts,
# every parts_<arm>/ and the depth-1 arm of the ablation's evaluation.
#
# Order of runs: gm_l5_100 seeds 0 1 2, sfno seeds 0 1 2, s2_transformer_sl
# seeds 0 1 2. Each run reads its arm's config from
# SWE_256x512_ZeroShot/configs (which draws the model, grid, objective and
# training groups from configs/) and writes to
# outputs/SWE_256x512_ZeroShot/<arm>/seed<N>/ with logs in
# logs/SWE_256x512_ZeroShot/. A run directory that already holds metrics is
# refused; remove it to rerun.
#
# Optional environment:
#   DRY_RUN=1                          print the deal or the command and start nothing
#   SWE_CACHE=/path/to/spectral_cache  on-disk cache of solved states, shared
#                                      across runs (results identical without)
#   ARM=<arm>                          another training of the GM arm, for
#                                      evaluate-parallel and merge (above)
set -euo pipefail

# GM_CUTOFF_RULE=legacy: this bundle's checkpoints aggregate over the edges of
# the legacy cutoff rule, so training, evaluation and resumes run under it.
export GM_CUTOFF_RULE="${GM_CUTOFF_RULE:-legacy}"
cd "$(dirname "$0")/.."
# shellcheck source=scripts/toolchain.sh
source scripts/toolchain.sh  # require_toolchain: nvcc and a C++17 g++ for the fused kernels

ARMS=(gm_l5_100 sfno s2_transformer_sl)
SEEDS=(0 1 2)
CONFIGS="$(pwd)/SWE_256x512_ZeroShot/configs"
mkdir -p logs/SWE_256x512_ZeroShot

cache_override=()
if [ -n "${SWE_CACHE:-}" ]; then cache_override=("data.spectral_cache_dir=${SWE_CACHE}"); fi

# Per-arm implementation switches, recorded in the run's paper_protocol.json;
# they change no value beyond floating-point order.
#   GM_FUSED_PAIRS=1          the fused pair-term kernels (gm_attention.nn.fused_pairs)
#   GM_CHUNK_REMATERIALIZE=0  chunks keep their saved tensors instead of
#                             recomputing their forward in the backward
arm_env() {  # arm -> KEY=VALUE words
  case "$1" in
    gm_l5_100) echo "GM_FUSED_PAIRS=1 GM_CHUNK_REMATERIALIZE=0" ;;
    *) ;;
  esac
}

# The fused kernels compile on first use, so the toolchain is required before any worker detaches.
prepare_arm() {  # arm -> loads or demands the toolchain when the arm builds kernels
  if [[ " $(arm_env "$1") " == *" GM_FUSED_PAIRS=1 "* ]] || [ "${GM_FUSED_PAIRS:-0}" = "1" ]; then
    require_toolchain
  fi
}

run_one() {  # arm seed gpu -> runs in the foreground
  local arm="$1" seed="$2" gpu="$3"
  local dir="outputs/SWE_256x512_ZeroShot/${arm}/seed${seed}"
  if [ -f "${dir}/logs/metrics.jsonl" ]; then
    echo "run.sh: ${dir} already holds metrics; remove it to rerun." >&2
    return 1
  fi
  prepare_arm "$arm"
  local -a switches=()
  read -r -a switches <<< "$(arm_env "$arm")"
  # ${a[@]+"${a[@]}"}: bash before 4.4 treats an empty array as unset under set -u.
  if [ "${DRY_RUN:-0}" = 1 ]; then  # DRY_RUN=1 prints the command instead of running it
    echo "would run: env ${switches[@]+"${switches[@]}"} CUDA_VISIBLE_DEVICES=$gpu uv run python -m experiments.shallow_water_paper.train" \
         "-cp $CONFIGS -cn $arm seed=${seed} ${cache_override[@]+"${cache_override[@]}"} > logs/SWE_256x512_ZeroShot/${arm}_seed${seed}.log"
    return 0
  fi
  env ${switches[@]+"${switches[@]}"} CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
    uv run python -m experiments.shallow_water_paper.train \
      -cp "$CONFIGS" -cn "$arm" "seed=${seed}" ${cache_override[@]+"${cache_override[@]}"} \
      > "logs/SWE_256x512_ZeroShot/${arm}_seed${seed}.log" 2>&1
}

runs=()
for arm in "${ARMS[@]}"; do for seed in "${SEEDS[@]}"; do runs+=("${arm} ${seed}"); done; done

mode="${1:-}"; shift || true
case "$mode" in
  arm)
    # One arm's seeds dealt round-robin over the GPUs given, one background worker per GPU.
    arm="${1:?arm needs an arm name}"; shift
    if [ ! -f "${CONFIGS}/${arm}.yaml" ]; then echo "no config ${CONFIGS}/${arm}.yaml" >&2; exit 2; fi
    if [ "$#" -lt 1 ]; then echo "arm needs at least one GPU index" >&2; exit 2; fi
    prepare_arm "$arm"  # before the workers detach: the failure belongs in this terminal
    gpus=("$@"); n="${#gpus[@]}"
    for ((g = 0; g < n; g++)); do
      share=()
      for ((i = g; i < ${#SEEDS[@]}; i += n)); do share+=("${arm} ${SEEDS[$i]}"); done
      [ "${#share[@]}" -eq 0 ] && continue
      echo "GPU ${gpus[$g]}: ${share[*]}"
      if [ "${DRY_RUN:-0}" = 1 ]; then continue; fi  # DRY_RUN=1 prints the deal and launches nothing
      nohup "$0" _worker "${gpus[$g]}" "${share[@]}" \
        > "logs/SWE_256x512_ZeroShot/worker_${arm}_gpu${gpus[$g]}.log" 2>&1 &
    done
    disown -a
    echo "${#SEEDS[@]} runs of ${arm} dealt over ${n} GPUs; progress: ./scripts/progress.sh outputs/SWE_256x512_ZeroShot"
    ;;
  launch)
    if [ "$#" -lt 1 ]; then echo "launch needs at least one GPU index" >&2; exit 2; fi
    for arm in "${ARMS[@]}"; do prepare_arm "$arm"; done
    gpus=("$@"); n="${#gpus[@]}"
    for ((g = 0; g < n; g++)); do
      share=()
      for ((i = g; i < ${#runs[@]}; i += n)); do share+=("${runs[$i]}"); done
      [ "${#share[@]}" -eq 0 ] && continue  # more GPUs than runs: no worker (an empty share is unset on bash < 4.4)
      echo "GPU ${gpus[$g]}: ${share[*]}"
      if [ "${DRY_RUN:-0}" = 1 ]; then continue; fi  # DRY_RUN=1 prints the deal and launches nothing
      nohup "$0" _worker "${gpus[$g]}" "${share[@]}" \
        > "logs/SWE_256x512_ZeroShot/worker_gpu${gpus[$g]}.log" 2>&1 &
    done
    disown -a
    echo "${#runs[@]} runs dealt over ${n} GPUs; progress: ./scripts/progress.sh outputs/SWE_256x512_ZeroShot"
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
    # The L5 arm on --grids and the references on their own grids; extra
    # arguments pass through (e.g. --device cuda:3 --projector-device cpu).
    uv run python SWE_256x512_ZeroShot/evaluate.py "$@"
    ;;
  evaluate-parallel)
    if [ "$#" -lt 1 ]; then echo "evaluate-parallel needs at least one GPU index" >&2; exit 2; fi
    gpus=("$@"); n="${#gpus[@]}"
    parts="outputs/SWE_256x512_ZeroShot/parts${ARM:+_${ARM}}"
    eval_args="${ARM:+--arms gm_l5_100=${ARM} }${EVAL_ARGS:-}"
    mkdir -p "$parts"
    all_jobs=(
      "coarse|--grids icosphere:4,icosphere:5"
      # The native grid alone, one seed a job (EVAL_JOBS="l5_seed0 l5_seed1 l5_seed2").
      "l5_seed0|--grids icosphere:5 --seeds 0"
      "l5_seed1|--grids icosphere:5 --seeds 1"
      "l5_seed2|--grids icosphere:5 --seeds 2"
      # Without --chunk-size an arm is scored at its training chunk; L6 has
      # about 776 neighbours a node, so the chunk is halved to bound the edges.
      "l6_seed0|--grids icosphere:6 --seeds 0 --chunk-size 1024"
      "l6_seed1|--grids icosphere:6 --seeds 1 --chunk-size 1024"
      "l6_seed2|--grids icosphere:6 --seeds 2 --chunk-size 1024"
      # Legendre-Gauss rows near the poles hold up to 2,818 neighbours a node: chunk 256.
      "lg128_seed0|--grids legendre-gauss:128x256 --seeds 0 --chunk-size 256"
      "lg128_seed1|--grids legendre-gauss:128x256 --seeds 1 --chunk-size 256"
      "lg128_seed2|--grids legendre-gauss:128x256 --seeds 2 --chunk-size 256"
      "references|--skip-gm"
    )
    jobs=()
    if [ -n "${EVAL_JOBS:-}" ]; then  # a named subset, in the order given
      for wanted in $EVAL_JOBS; do
        found=""
        for job in "${all_jobs[@]}"; do [ "${job%%|*}" = "$wanted" ] && found="$job"; done
        if [ -z "$found" ]; then echo "evaluate-parallel: no job '${wanted}'; jobs: ${all_jobs[*]%%|*}" >&2; exit 2; fi
        jobs+=("$found")
      done
    else
      jobs=("${all_jobs[@]}")
    fi
    for ((g = 0; g < n; g++)); do
      share=()
      for ((i = g; i < ${#jobs[@]}; i += n)); do share+=("${jobs[$i]}"); done
      [ "${#share[@]}" -eq 0 ] && continue  # more GPUs than jobs
      echo "GPU ${gpus[$g]}: $(printf '%s ' "${share[@]%%|*}")"
      if [ "${DRY_RUN:-0}" = 1 ]; then continue; fi  # DRY_RUN=1 prints the deal and launches nothing
      EVAL_PARTS="$parts" EVAL_ARGS="$eval_args" nohup "$0" _evaluate_worker "${gpus[$g]}" "${share[@]}" \
        > "logs/SWE_256x512_ZeroShot/evaluate${ARM:+_${ARM}}_gpu${gpus[$g]}.log" 2>&1 &
    done
    disown -a
    echo "${#jobs[@]} evaluation jobs dealt over ${n} GPUs; parts land in ${parts}/; then: ${ARM:+ARM=${ARM} }$0 merge"
    ;;
  _evaluate_worker)
    gpu="$1"; shift
    parts="${EVAL_PARTS:-outputs/SWE_256x512_ZeroShot/parts}"
    for job in "$@"; do
      name="${job%%|*}"; flags="${job#*|}"
      echo "GPU ${gpu}: ${name} (${flags}) -> ${parts}/${name}.json"
      # Cap each job's CPU threads so parallel jobs do not oversubscribe the host.
      # shellcheck disable=SC2086
      OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}" \
      CUDA_VISIBLE_DEVICES="$gpu" uv run python SWE_256x512_ZeroShot/evaluate.py \
        --num-examples 64 --projector-device cpu ${EVAL_ARGS:-} $flags \
        --output "${parts}/${name}.json" > "logs/SWE_256x512_ZeroShot/evaluate${ARM:+_${ARM}}_${name}.log" 2>&1
    done
    ;;
  merge)
    # With ARM, that arm's parts alone, its rows renamed for its run. Without,
    # the default parts, then every parts_<arm>/ appended with its own rename.
    out="${1:-outputs/SWE_256x512_ZeroShot/evaluation${ARM:+_${ARM}}.json}"
    # Built beside the output and moved into place, so a failed merge leaves
    # the previous JSON intact.
    tmp="${out}.partial"
    # nullglob is not set: a directory holding no JSON leaves the glob
    # unexpanded, so the first word is tested before the list is used.
    have_parts() { local first="$1"; [ -e "$first" ]; }
    if [ -n "${ARM:-}" ]; then
      arm_parts=("outputs/SWE_256x512_ZeroShot/parts_${ARM}"/*.json)
      if ! have_parts "${arm_parts[0]}"; then
        echo "run.sh merge: no parts in outputs/SWE_256x512_ZeroShot/parts_${ARM}/" >&2; exit 2
      fi
      uv run python SWE_256x512_ZeroShot/evaluate.py --merge "${arm_parts[@]}" \
        --relabel "gm_l5_100=${ARM}" --output "$tmp"
    else
      default_parts=(outputs/SWE_256x512_ZeroShot/parts/*.json)
      if ! have_parts "${default_parts[0]}"; then
        echo "run.sh merge: no parts in outputs/SWE_256x512_ZeroShot/parts/" >&2; exit 2
      fi
      uv run python SWE_256x512_ZeroShot/evaluate.py --merge "${default_parts[@]}" --output "$tmp"
      for dir in outputs/SWE_256x512_ZeroShot/parts_*/; do
        [ -d "$dir" ] || continue  # the outer glob, unexpanded, names no directory
        arm_parts=("$dir"*.json)
        if ! have_parts "${arm_parts[0]}"; then
          echo "run.sh merge: no parts in ${dir}, skipped" >&2; continue
        fi
        arm="$(basename "$dir")"; arm="${arm#parts_}"
        uv run python SWE_256x512_ZeroShot/evaluate.py \
          --merge "${arm_parts[@]}" --relabel "gm_l5_100=${arm}" --append --output "$tmp"
      done
      # The depth-1 arm's row comes from the ablation study, which shares the
      # protocol and trajectories (checked by the merge); --merge-only leaves
      # that study's other arms behind.
      ablation="${ABLATION_EVALUATION:-outputs/SWE_256x512_Ablation/evaluation.json}"
      if [ -f "$ablation" ]; then
        uv run python SWE_256x512_ZeroShot/evaluate.py --merge "$ablation" \
          --merge-only "${ABLATION_ARM:-gm_attention_full}" --append --output "$tmp"
      else
        echo "run.sh merge: no ${ablation}; the depth-1 row is left out" >&2
      fi
    fi
    mv "$tmp" "$out"
    echo "wrote ${out}"
    ;;
  tables)
    uv run python SWE_256x512_ZeroShot/tables.py "$@"
    ;;
  *)
    echo "usage: $0 arm <arm> <gpus...> | launch <gpus...> | sequential <gpu> | evaluate [args] | evaluate-parallel <gpus...> | merge [out.json] | tables [args]" >&2; exit 2
    ;;
esac
