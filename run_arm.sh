#!/bin/bash
# Run AT THE GX10 (not over ssh: a dropped link must not be able to orphan a powered arm). One hand on the power switch.
#   ./run_arm.sh hold        zero-motion power-on test, 10 s
#   ./run_arm.sh grip        same + jaws close halfway and reopen (put a sacrificial grape between them)
#   ./run_arm.sh start-dry   read-only: prints the move to the RL start pose and the collision check
#   ./run_arm.sh start       SLOW (0.12 rad/s peak, ~36 s) move to the RL start pose, hold 5 s, ease back, release at rest
cd "$(dirname "$0")"
export YAM_MENAGERIE=$PWD/third_party/mujoco_menagerie/i2rt_yam YAM_REAL_ARM=I_AM_AT_THE_ARM_WITH_THE_ESTOP
PY=/home/asus/HackMIT_2026/.venv/bin/python
mkdir -p runs/real_arm; LOG=runs/real_arm/$(date +%H%M%S)_$1.log
exec > >(tee $LOG) 2>&1
case "$1" in
  hold)      $PY arm/real/real_arm.py --hold-test 10 ;;
  grip)      $PY arm/real/real_arm.py --hold-test 12 --grip ;;
  start-dry) $PY arm/real/real_arm.py --goto-start --dry ;;
  start)     $PY arm/real/real_arm.py --goto-start --hold 5 --speed 0.12 ;;
  *) sed -n 2,7p "$0" ;;
esac
