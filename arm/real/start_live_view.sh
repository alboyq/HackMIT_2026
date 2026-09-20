#!/bin/bash
# Wrist camera + food/face detection on the GX10 monitor. Read-only. usage: start_live_view.sh [food]
cd ~/HackMIT_2026-arm-ik-rl/arm/real
export DISPLAY=:1
[ -z "${XAUTHORITY:-}" ] && export XAUTHORITY=$(ls /run/user/$(id -u)/.mutter-Xwaylandauth.* /run/user/$(id -u)/gdm/Xauthority 2>/dev/null | head -1)
setsid nohup ../../.venv-vision/bin/python live_perception.py --food "${1:-grape}" > /tmp/live_perception.log 2>&1 < /dev/null &
echo started
