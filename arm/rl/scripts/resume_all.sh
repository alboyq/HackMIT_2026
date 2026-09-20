#!/usr/bin/env bash
# Resume positioning from its newest checkpoint, then the chain, with self-restarting viewers.
set -u
cd ~/HackMIT_2026-arm-ik-rl || exit 1
export YAM_MENAGERIE=$PWD/third_party/mujoco_menagerie PYTHONPATH=.:arm/rl OMP_NUM_THREADS=1
(
  if [ ! -f runs/feed-reach/ppo_reach_final.zip ]; then
    nice -n 5 .venv-arm/bin/python arm/rl/scripts/run_resumable.py --env feed --config arm/rl/configs/feed.yaml \
        --stage reach --run-dir runs/feed-reach --timesteps 500000 --init-from runs/archive-reach-2230 >> runs/feed-reach.log 2>&1
  fi
  [ -f runs/feed-reach/ppo_reach_final.zip ] || exit 1
  exec bash arm/rl/scripts/chain_curriculum.sh > runs/chain.log 2>&1
) > /dev/null 2>&1 < /dev/null &
disown

( while true; do sync; sleep 30; done ) > /dev/null 2>&1 < /dev/null &
disown

current_stage() {
  local s; s=$(grep -o "starting [a-z]*" runs/chain.log 2>/dev/null | tail -1 | cut -d" " -f2)
  if [ -z "$s" ] || ! ls runs/feed-$s/checkpoints/*.zip > /dev/null 2>&1; then s=reach; fi
  echo "$s"
}
export DISPLAY=:1
[ -z "${XAUTHORITY:-}" ] && export XAUTHORITY=$(ls /run/user/$(id -u)/.mutter-Xwaylandauth.* /run/user/$(id -u)/gdm/Xauthority 2>/dev/null | head -1)
(
  while true; do
    s=$(current_stage)
    .venv-arm/bin/python arm/rl/scripts/watch_policy.py --run-dir runs/feed-$s --stage $s --follow --episodes 12 >> runs/watch.log 2>&1
    sleep 2
  done
) > /dev/null 2>&1 < /dev/null &
disown
(
  while true; do
    s=$(current_stage)
    .venv-arm/bin/python arm/rl/scripts/stream_policy.py --run-dir runs/feed-$s --stage $s --auto-stage --follow --port 8089 --fps 25 >> runs/stream.log 2>&1
    sleep 3
  done
) > /dev/null 2>&1 < /dev/null &
disown
sleep 2; echo resumed
