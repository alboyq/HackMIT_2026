#!/usr/bin/env bash
# Run the feeding curriculum end to end: reach -> grasp -> lift -> present.
#
# Each stage warm-starts from the previous one's finished weights (the observation layout is
# identical across stages, so this is a legitimate transfer). Every stage is launched through
# run_resumable.py, so the whole chain survives a pause: kill it, re-run this script, and each
# stage picks up from its own newest checkpoint. Stages already finished are skipped.

set -u
cd ~/HackMIT_2026-arm-ik-rl || exit 1
export YAM_MENAGERIE=$PWD/third_party/mujoco_menagerie
export PYTHONPATH=.:arm/rl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=.venv-arm/bin/python

# stage:timesteps -- reach is the long one; later stages inherit most of the behaviour
STAGES="reach:1200000 grasp:3000000 present:4000000"   # lift is folded into grasp
PREV=""

for entry in $STAGES; do
    stage="${entry%%:*}"
    steps="${entry##*:}"
    run="runs/feed-${stage}"

    if [ -f "${run}/ppo_${stage}_final.zip" ]; then
        echo "[chain] ${stage}: already final, skipping"
        PREV="$run"
        continue
    fi

    # Wait out any instance of this stage that is already running (e.g. the reach job
    # launched by hand), rather than starting a second one against the same run-dir.
    while pgrep -f "run_resumable.py.*--stage ${stage} --run-dir ${run} " > /dev/null; do
        sleep 20
    done

    if [ -f "${run}/ppo_${stage}_final.zip" ]; then
        echo "[chain] ${stage}: finished while waiting"
        PREV="$run"
        continue
    fi

    echo "[chain] $(date +%H:%M:%S) starting ${stage} (${steps} steps)${PREV:+ from ${PREV}}"
    if [ -n "$PREV" ]; then
        nice -n 5 $PY arm/rl/scripts/run_resumable.py --env feed \
            --config arm/rl/configs/feed.yaml --stage "$stage" --run-dir "$run" \
            --timesteps "$steps" --init-from "$PREV" >> "runs/feed-${stage}.log" 2>&1
    else
        nice -n 5 $PY arm/rl/scripts/run_resumable.py --env feed \
            --config arm/rl/configs/feed.yaml --stage "$stage" --run-dir "$run" \
            --timesteps "$steps" >> "runs/feed-${stage}.log" 2>&1
    fi

    status=$?
    echo "[chain] $(date +%H:%M:%S) ${stage} exited ${status}"
    if [ ! -f "${run}/ppo_${stage}_final.zip" ]; then
        echo "[chain] ${stage} did not reach its target (paused or failed); stopping the chain."
        echo "[chain] re-run this script to continue from where it stopped."
        exit "$status"
    fi
    PREV="$run"
done

echo "[chain] $(date +%H:%M:%S) curriculum complete"
