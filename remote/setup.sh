#!/usr/bin/env bash
# Run only inside an explicitly approved remote Linux GPU machine.
set -euo pipefail
if [[ "$(uname -s)" != Linux || "${DATA_DEBUGGING_REMOTE:-}" != 1 ]]; then
  echo 'Remote-only setup: requires Linux and DATA_DEBUGGING_REMOTE=1.' >&2
  exit 2
fi
cd "$(dirname "$0")/.."
command -v nvidia-smi >/dev/null
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python3 -m venv .venv-cuda
.venv-cuda/bin/python -m pip install 'pip==25.1.1'
.venv-cuda/bin/python -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu124
.venv-cuda/bin/python -m pip install -r remote/requirements.txt
.venv-cuda/bin/python -m pip check
mkdir -p runs
.venv-cuda/bin/python -m pip freeze > runs/environment-installed.txt
if [[ ! -d upstream/persona_vectors/.git ]]; then
  git clone https://github.com/safety-research/persona_vectors.git upstream/persona_vectors
fi
if [[ -n "$(git -C upstream/persona_vectors status --porcelain)" ]]; then
  echo 'Upstream checkout has local changes; refusing to replace them.' >&2
  exit 2
fi
git -C upstream/persona_vectors checkout --detach b8e0f044fe2410a6fad579f38324f03f13b4e917
echo 'Setup complete. No model has been loaded and no experiment has run.'
echo 'Next: follow remote/README.md for init, judge-check, and preflight.'
