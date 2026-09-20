#!/bin/zsh
# Same chain as run_chain.sh (demos -> ACT -> closed-loop eval) but uses the whole laptop (M4 Max, 12P+4E, 64 GB):
#   * demo generation on GEN_WORKERS (default 14) CPU processes;
#   * training on the GPU (MPS) while each checkpoint is evaluated on the CPU AS SOON AS IT IS WRITTEN
#     (the CPU is otherwise idle during training), instead of all evaluations after training.
# Usage: run_chain_fast.sh RUN_NAME CALIB_JSON BG_DIR [EPISODES] [STEPS]
# Results: rl/runs/RUN_NAME/results.txt (+ per-checkpoint pdiag JSON and filmstrip PNGs)
RUN=${1:?run name}
CALIB=${2:?camera calib json}
BG=${3:?background dir}
EPISODES=${4:-700}
STEPS=${5:-15000}
ROOT=/Users/adipu/so101Sim
OUT=$ROOT/rl/runs/$RUN
PY=$ROOT/urlab_bridge/.venv/bin/python
export KMP_DUPLICATE_LIB_OK=TRUE
export CKPT_EVERY=${CKPT_EVERY:-5000} BATCH=${BATCH:-16}
GEN_WORKERS=${GEN_WORKERS:-14}
EVAL_WORKERS=${EVAL_WORKERS:-10}
mkdir -p $OUT
R=$OUT/results.txt
echo "=== $RUN  $(date)  calib=$CALIB bg=$BG episodes=$EPISODES steps=$STEPS BATCH=$BATCH JAR_ZONE=${JAR_ZONE:-default} KP_SCALE=${KP_SCALE:-1} OBS_LATENCY=${OBS_LATENCY:-0} JAR_SHADOW=${JAR_SHADOW:-0} SIDE_ZFRAC=${SIDE_ZFRAC:-0.58} OBJECT_SPEC=${OBJECT_SPEC:-rl/real/object.json}" >> $R
cp $CALIB $OUT/camera_calib.json; cp ${OBJECT_SPEC:-$ROOT/rl/real/object.json} $OUT/object.json

cd $ROOT/urlab_bridge
if [ ! -f $OUT/demos/meta.txt ]; then
  $PY ../rl/jar/gen_demos.py $OUT/demos $EPISODES --calib $CALIB --bg $BG --workers $GEN_WORKERS 2>&1 | grep -E "^DONE" >> $R
fi
$PY ../rl/jar/train_act_jar.py $OUT/demos $OUT/act $STEPS > $OUT/train.log 2>&1 &
TRAIN=$!

step=$CKPT_EVERY
while [ $step -le $STEPS ]; do
  d=$OUT/act/step_$step
  until [ -f $d/model.safetensors ] || ! kill -0 $TRAIN 2>/dev/null; do sleep 20; done
  [ -f $d/model.safetensors ] || { echo "training died before step $step (see train.log)" >> $R; break; }
  sleep 5
  $PY ../rl/jar/eval_act_jar.py $d 48 --calib $CALIB --bg $BG --na ${NA:-25} --workers $EVAL_WORKERS 2>&1 | grep "^ACT jar" >> $R
  step=$((step + CKPT_EVERY))
done
wait $TRAIN
echo "training finished $(date): $(tail -1 $OUT/act/train_progress.txt)" >> $R
best=$(grep "^ACT jar step_" $R | sed -E 's/^ACT jar (step_[0-9]+) .*: ([0-9]+)\/.*/\2 \1/' | sort -k1,1nr -k2,2r | head -1 | awk '{print $2}')
echo "best checkpoint: $best" >> $R
echo "-- best checkpoint, 96 fresh seeds: real-scene photos only / wallpaper-pool only (BOTH are in training) / rl/jar/bg_heldout (never trained on):" >> $R
$PY ../rl/jar/eval_act_jar.py $OUT/act/$best 96 --calib $CALIB --bg $BG --na ${NA:-25} --p-real 1.0 --seed0 9500000 --workers 14 2>&1 | grep "^ACT jar" >> $R
$PY ../rl/jar/eval_act_jar.py $OUT/act/$best 96 --calib $CALIB --bg $BG --na ${NA:-25} --p-real 0.0 --seed0 9500000 --workers 14 2>&1 | grep "^ACT jar" >> $R
EVAL_TAG=heldout $PY ../rl/jar/eval_act_jar.py $OUT/act/$best 96 --calib $CALIB --bg $ROOT/rl/jar/bg_heldout --na ${NA:-25} --p-real 1.0 --seed0 9500000 --workers 14 2>&1 | grep "^ACT jar" | sed 's/$/   <- HELD-OUT backgrounds/' >> $R
echo "=== chain done $(date)" >> $R
