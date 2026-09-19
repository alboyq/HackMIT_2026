#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT}/../.." && pwd)"
VENV="${VENV:-${REPO_ROOT}/.venv-arm}"
RUN_ROOT="${1:?run root required}"
N_ENVS="${N_ENVS:-12}"
REACH_STEPS="${REACH_STEPS:-1000000}"
GRASP_STEPS="${GRASP_STEPS:-1500000}"
LIFT_STEPS="${LIFT_STEPS:-2000000}"
source "${VENV}/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${ROOT}"
export YAM_MENAGERIE="${YAM_MENAGERIE:-${REPO_ROOT}/third_party/mujoco_menagerie}"
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "${RUN_ROOT}/reach" "${RUN_ROOT}/grasp" "${RUN_ROOT}/lift"
python "${ROOT}/scripts/train.py" --config "${ROOT}/configs/default.yaml" --stage reach \
  --n-envs "${N_ENVS}" --timesteps "${REACH_STEPS}" --run-dir "${RUN_ROOT}/reach"
python "${ROOT}/scripts/train.py" --config "${ROOT}/configs/default.yaml" --stage grasp \
  --n-envs "${N_ENVS}" --timesteps "${GRASP_STEPS}" --run-dir "${RUN_ROOT}/grasp" \
  --init-model "${RUN_ROOT}/reach/ppo_reach_final.zip" \
  --init-vecnormalize "${RUN_ROOT}/reach/vecnormalize.pkl"
python "${ROOT}/scripts/train.py" --config "${ROOT}/configs/default.yaml" --stage lift \
  --n-envs "${N_ENVS}" --timesteps "${LIFT_STEPS}" --run-dir "${RUN_ROOT}/lift" \
  --init-model "${RUN_ROOT}/grasp/ppo_grasp_final.zip" \
  --init-vecnormalize "${RUN_ROOT}/grasp/vecnormalize.pkl"

