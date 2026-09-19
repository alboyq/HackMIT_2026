#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_ROOT="${ROOT}/runs/curriculum-${STAMP}"
mkdir -p "${RUN_ROOT}"
nohup nice -n 10 bash "${ROOT}/scripts/run_curriculum.sh" "${RUN_ROOT}" \
  >"${RUN_ROOT}/curriculum.log" 2>&1 &
echo $! >"${RUN_ROOT}/curriculum.pid"
ln -sfn "${RUN_ROOT}" "${ROOT}/runs/latest-curriculum"
printf 'started curriculum pid %s\nlog: %s\n' "$(cat "${RUN_ROOT}/curriculum.pid")" "${RUN_ROOT}/curriculum.log"
