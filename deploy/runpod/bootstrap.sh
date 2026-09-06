#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${ECHO_REPO_DIR:-/workspace/Echo}"
STATE_DIR="${ECHO_STATE_DIR:-/workspace/compute-echo-state}"
cd "$REPO_DIR"
python -m pip install --upgrade pip
python -m pip install -e '.[nvml]'

CUDA_RELEASE="${CUDA_VERSION:-}"
CUDA_MAJOR="${CUDA_RELEASE%%.*}"
if [[ "$CUDA_MAJOR" == "13" ]]; then
  python -m pip install 'cupy-cuda13x>=14,<15'
elif [[ "$CUDA_MAJOR" == "12" ]]; then
  python -m pip install 'cupy-cuda12x>=14,<15'
else
  echo "Unsupported or missing CUDA_VERSION=$CUDA_RELEASE; install the matching CuPy wheel." >&2
  exit 2
fi
mkdir -p "$STATE_DIR"
echo "Installed Compute Echo under $REPO_DIR; persistent state will use $STATE_DIR"
