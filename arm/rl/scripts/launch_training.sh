#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/../.." && pwd)"
VENV="${VENV:-${REPO_ROOT}/.venv-arm}"
STAGE="${1:-reach}"
N_ENVS="${N_ENVS:-12}"
TIMESTEPS="${TIMESTEPS:-500000}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${ROOT}/runs/${STAGE}-${STAMP}"
mkdir -p "${RUN_DIR}"
source "${VENV}/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${ROOT}"
export YAM_MENAGERIE="${YAM_MENAGERIE:-${REPO_ROOT}/third_party/mujoco_menagerie}"
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
nohup nice -n 10 python "${ROOT}/scripts/train.py" \
  --config "${ROOT}/configs/default.yaml" \
  --stage "${STAGE}" --n-envs "${N_ENVS}" --timesteps "${TIMESTEPS}" \
  --run-dir "${RUN_DIR}" >"${RUN_DIR}/train.log" 2>&1 &
echo $! >"${RUN_DIR}/train.pid"
ln -sfn "${RUN_DIR}" "${ROOT}/runs/latest"
echo "started pid $(cat "${RUN_DIR}/train.pid")"
echo "log: ${RUN_DIR}/train.log"
