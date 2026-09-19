#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/../.." && pwd)"
VENV="${VENV:-${REPO_ROOT}/.venv-arm}"
source "${VENV}/bin/activate"
mkdir -p "${ROOT}/logs"
nohup nice -n 10 tensorboard --logdir "${ROOT}/runs" --host 127.0.0.1 --port 6006 \
  >"${ROOT}/logs/tensorboard.log" 2>&1 &
echo $! >"${ROOT}/logs/tensorboard.pid"
echo "tensorboard pid $(cat "${ROOT}/logs/tensorboard.pid")"
