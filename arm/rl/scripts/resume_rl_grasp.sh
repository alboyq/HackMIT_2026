#!/usr/bin/env bash
cd ~/HackMIT_2026-arm-ik-rl || exit 1
export YAM_MENAGERIE=$PWD/third_party/mujoco_menagerie PYTHONPATH=.:arm/rl OMP_NUM_THREADS=1
setsid nohup nice -n 5 .venv-arm/bin/python arm/rl/scripts/run_resumable.py --env feed --config arm/rl/configs/feed.yaml \
    --stage grasp --run-dir runs/feed-grasp --timesteps 4000000 --init-from runs/grasp-pinch-best >> runs/feed-grasp.log 2>&1 < /dev/null &
sleep 1; echo resumed
