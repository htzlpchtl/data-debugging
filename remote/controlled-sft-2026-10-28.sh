#!/usr/bin/env bash
# Execute only on the approved remote Pod. This script does not stop billing.
set -euo pipefail
cd /workspace/data-debugging
export DATA_DEBUGGING_REMOTE=1 CUDA_VISIBLE_DEVICES=0
export HF_HOME=/workspace/huggingface MPLCONFIGDIR=/workspace/matplotlib
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1
source .venv-cuda/bin/activate
source /workspace/.data-debugging-openrouter.env
out=runs/controlled-sft-2026-10-28
mkdir -p "$out"
date -u +%FT%TZ > "$out/started_utc.txt"
trap 'code=$?; printf "%s\n" "$code" > "$out/pipeline.exit"; date -u +%FT%TZ > "$out/finished_utc.txt"' EXIT
cutoff=$(date -u -d '2026-10-28 11:25:00' +%s)
run_stage() {
  local label=$1
  shift
  local remaining=$((cutoff - $(date -u +%s)))
  if (( remaining <= 0 )); then return 124; fi
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$label"
  python remote/controlled_sft.py "$@" --max-seconds "$remaining" > "$out/$label.log" 2>&1
}
run_stage init init
run_stage judge-check judge --target check
run_stage preflight preflight
run_stage baseline evaluate --arm base
run_stage judge-base judge --arm base
run_stage train-normal train --arm normal
run_stage eval-normal evaluate --arm normal
run_stage judge-normal judge --arm normal
run_stage train-trait train --arm trait
run_stage eval-trait evaluate --arm trait
run_stage judge-trait judge --arm trait
run_stage report report
# Optional projection can never replace the completed behavioural report.
if (( cutoff - $(date -u +%s) > 600 )); then
  if run_stage projection project; then run_stage report-projection report; fi
fi
