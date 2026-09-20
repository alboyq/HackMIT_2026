#!/usr/bin/env bash
cd ~/HackMIT_2026-arm-ik-rl || exit 1
export YAM_MENAGERIE=$PWD/third_party/mujoco_menagerie PYTHONPATH=.:arm/rl OMP_NUM_THREADS=1 DISPLAY=:1
[ -z "${XAUTHORITY:-}" ] && export XAUTHORITY=$(ls /run/user/$(id -u)/.mutter-Xwaylandauth.* /run/user/$(id -u)/gdm/Xauthority 2>/dev/null | head -1)
export CALIBRATE=${1:-1}
( while true; do SEED=$RANDOM .venv-arm/bin/python arm/rl/scripts/watch_feed.py >> runs/watch_feed.log 2>&1; sleep 3; done ) > /dev/null 2>&1 < /dev/null &
disown
sleep 1; echo feed view started CALIBRATE=$CALIBRATE
