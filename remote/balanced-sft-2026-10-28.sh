#!/usr/bin/env bash
# Approved remote experiment. Process deadline does not stop cloud billing.
set -euo pipefail
cd /workspace/data-debugging
export DATA_DEBUGGING_REMOTE=1 CUDA_VISIBLE_DEVICES=0
export HF_HOME=/workspace/huggingface MPLCONFIGDIR=/workspace/matplotlib
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
source .venv-cuda/bin/activate
source /workspace/.data-debugging-openrouter.env
out=runs/balanced-sft-2026-10-28
mkdir -p "$out/executed_source"
cp remote/*.py remote/requirements.txt remote/balanced-sft-2026-10-28.json remote/balanced-sft-2026-10-28.sh "$out/executed_source/"
date -u +%FT%TZ > "$out/started_utc.txt"
trap 'code=$?; printf "%s\n" "$code" > "$out/pipeline.exit"; date -u +%FT%TZ > "$out/finished_utc.txt"' EXIT
: "${BALANCED_CUTOFF_EPOCH:?Set the fixed allocation deadline minus backup reserve}"
run_stage() {
  local label=$1
  shift
  local remaining=$((BALANCED_CUTOFF_EPOCH - $(date -u +%s)))
  if (( remaining <= 0 )); then return 124; fi
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$label"
  python remote/balanced_sft.py "$@" --max-seconds "$remaining" > "$out/$label.log" 2>&1
}
run_stage init init
run_stage judge-check judge --target check
run_stage preflight preflight
run_stage references references
run_stage selection select
# A separate launcher invocation continues after inspection of the frozen subsets.
if [[ ${BALANCED_RUN_TRAINING:-0} == 1 ]]; then
  for arm in random low high; do
    run_stage train-$arm train --arm "$arm"
    run_stage eval-$arm evaluate --arm "$arm"
    run_stage judge-$arm judge --arm "$arm"
  done
  run_stage report report
fi
