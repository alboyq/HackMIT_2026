#!/bin/zsh
# Train ACT on the GPU while every checkpoint is evaluated closed-loop on the CPU as soon as it is written.
# Usage: run_chain_yam.sh RUN_DIR [STEPS]        (RUN_DIR/demos must exist)   env: BATCH CKPT_EVERY NA EVAL_EPS
RUN=${1:?run dir}; STEPS=${2:-12000}
HERE=${0:A:h}; ROOT=${HERE:h:h}
PY=${PY:-$ROOT/urlab_bridge/.venv/bin/python}
export KMP_DUPLICATE_LIB_OK=TRUE CKPT_EVERY=${CKPT_EVERY:-2000} BATCH=${BATCH:-24}
# The evaluator builds its own DataScene, so it must see the SAME scene env as the generator did,
# or it scores the policy on a distribution it never trained on. Both of these silently defaulted
# wrong once: YAM_AUG_PROFILE fell back to "heavy" (blur to sigma 2.6 on 70 % of episodes, which
# training never saw) and YAM_TARGETS was unset (scoring a 2-object policy on all five).
export YAM_AUG_PROFILE=${YAM_AUG_PROFILE:-real} YAM_TARGETS=${YAM_TARGETS:-}
echo "eval scene env: YAM_AUG_PROFILE=$YAM_AUG_PROFILE YAM_TARGETS='${YAM_TARGETS}' YAM_CAM=${YAM_CAM:-poseB} YAM_TABLE=${YAM_TABLE:-1} YAM_BLUR=${YAM_BLUR:-0}" >> ${1}/results.txt
R=$RUN/results.txt
echo "=== $(date)  steps=$STEPS batch=$BATCH  $(head -1 $RUN/demos/meta.txt)" >> $R
$PY $HERE/train_act_yam.py $RUN/demos $RUN/act $STEPS > $RUN/train.log 2>&1 &
TRAIN=$!
step=$CKPT_EVERY
while [ $step -le $STEPS ]; do
  d=$RUN/act/step_$step
  until [ -f $d/model.safetensors ] || ! kill -0 $TRAIN 2>/dev/null; do sleep 10; done
  [ -f $d/model.safetensors ] || { echo "training died before step $step (see train.log)" >> $R; break; }
  sleep 3
  $PY $HERE/eval_act_yam.py $d ${EVAL_EPS:-56} --na ${NA:-10} --workers ${EVAL_WORKERS:-12} 2>&1 | grep "^YAM" >> $R
  step=$((step + CKPT_EVERY))
done
wait $TRAIN
echo "=== chain done $(date)" >> $R
