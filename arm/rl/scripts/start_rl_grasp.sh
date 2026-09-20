#!/usr/bin/env bash
# RL grasp, warm-started from the last checkpoint that still pinched (before the hover collapse).
# Launched by hand, NOT through the chain, so feeding cannot auto-start on top of it.
cd ~/HackMIT_2026-arm-ik-rl || exit 1
export YAM_MENAGERIE=$PWD/third_party/mujoco_menagerie PYTHONPATH=.:arm/rl OMP_NUM_THREADS=1
TAG=$(date +%H%M)
[ -d runs/feed-grasp ] && mv runs/feed-grasp runs/archive-grasp-hover-$TAG && mv runs/feed-grasp.log runs/archive-grasp-hover-$TAG.log
setsid nohup nice -n 5 .venv-arm/bin/python arm/rl/scripts/run_resumable.py --env feed --config arm/rl/configs/feed.yaml \
    --stage grasp --run-dir runs/feed-grasp --timesteps 4000000 --init-from runs/grasp-pinch-best > runs/feed-grasp.log 2>&1 < /dev/null &
sleep 1; echo rl grasp started
