#!/bin/bash
# GX10 pipeline, unplug-safe: every stage skips work that is already done, so re-running after a
# power cut resumes rather than restarts. Data survives on disk; training resumes from LATEST.
cd ~/yam_simfirst
export YAM_MENAGERIE=$PWD/mujoco_menagerie MUJOCO_GL=egl YAM_BG=$PWD/bg_train YAM_AUG_PROFILE=real
RUN=runs/${1:-yam_g1}; EPS=${2:-2500}; STEPS=${3:-16000}
mkdir -p $RUN; R=$RUN/results.txt
[ -f $RUN/demos/meta.txt ] || nice -n 15 .venv/bin/python gen_demos_yam.py $RUN/demos $EPS --workers 12 --seed0 4000000 --film 4 > $RUN/gen.log 2>&1
echo "=== $(date) resume-safe chain; $(head -1 $RUN/demos/meta.txt)" >> $R
CKPT_EVERY=${CKPT_EVERY:-1000} BATCH=${BATCH:-32} nice -n 5 .venv/bin/python train_act_yam.py $RUN/demos $RUN/act $STEPS > $RUN/train.log 2>&1 &
TRAIN=$!
step=${CKPT_EVERY:-1000}
while [ $step -le $STEPS ]; do
  d=$RUN/act/step_$step
  until [ -f $d/model.safetensors ] || ! kill -0 $TRAIN 2>/dev/null; do sleep 10; done
  [ -f $d/model.safetensors ] || { echo "training stopped before step $step" >> $R; break; }
  if [ $((step % 2000)) -eq 0 ] && ! grep -q "step_$step " $R; then
    sleep 3; nice -n 15 .venv/bin/python eval_act_yam.py $d 56 --na 10 --workers 10 --bg $PWD/bg_heldout/full 2>&1 | grep "^YAM" >> $R
  fi
  step=$((step + ${CKPT_EVERY:-1000}))
done
wait $TRAIN; echo "=== chain done $(date)" >> $R
