#!/usr/bin/env bash
# Remote API judging and CPU analysis only. Does not stop Pod billing.
set -euo pipefail
cd /workspace/data-debugging
export DATA_DEBUGGING_REMOTE=1 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/workspace/matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
source .venv-cuda/bin/activate
source /workspace/.data-debugging-openrouter.env
out=runs/matching-feasibility-2026-10-28
mkdir -p "$out/executed_source"
cp remote/matching_feasibility.py remote/matching-feasibility-2026-10-28.json remote/matching-feasibility-2026-10-28.sh "$out/executed_source/"
date -u +%FT%TZ > "$out/started_utc.txt"
trap 'code=$?; printf "%s\n" "$code" > "$out/pipeline.exit"; date -u +%FT%TZ > "$out/finished_utc.txt"' EXIT
timeout 60 python remote/matching_feasibility.py init > "$out/init.log" 2>&1
timeout 1500 python remote/matching_feasibility.py judge > "$out/judge.log" 2>&1
timeout 180 python remote/matching_feasibility.py analyze > "$out/analyze.log" 2>&1
