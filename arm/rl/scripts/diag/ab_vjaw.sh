#!/bin/bash
# A/B: the same 3 x 40 full pick-and-feed episodes with parallel jaws (0) and the measured V jaws (28.4).
cd ~/HackMIT_2026-arm-ik-rl
export YAM_MENAGERIE=$PWD/third_party/mujoco_menagerie PYTHONPATH=$PWD:$PWD/arm/rl OMP_NUM_THREADS=1
mkdir -p runs/ab_vjaw
for v in 0 28.4; do for s in 1 2 3; do
  YAM_JAW_V_MM=$v SEED=$s setsid nohup .venv-arm/bin/python arm/rl/scripts/hybrid_feed.py arm/rl/models/grasp_v2 40 > runs/ab_vjaw/v${v}_s$s.log 2>&1 < /dev/null &
done; done
echo started
