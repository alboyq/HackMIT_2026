#!/bin/zsh
# Camera-realism stress test: re-run one checkpoint on the SAME seeds while the policy's input image is degraded
# the way a real camera degrades it (rl/jar/obs_corrupt.py). Paired seeds, so differences between rows are
# tighter than the +-4 pt binomial error of any single row.
# Usage: stress_test_obs.sh CKPT_DIR CALIB_JSON BG_DIR [EPISODES]      (export the run's scene env first:
#        OBJECT_SPEC JAR_ZONE KP_SCALE JAR_FRICTION OBS_LATENCY JAR_SHADOW - they must match training)
CK=${1:?checkpoint dir}; CAL=${2:?calib json}; BG=${3:?background dir}; N=${4:-96}
ROOT=/Users/adipu/so101Sim
PY=$ROOT/urlab_bridge/.venv/bin/python
HELD=$ROOT/rl/jar/bg_heldout
OUT=$(dirname $(dirname $CK))/stress_obs.txt
export KMP_DUPLICATE_LIB_OK=TRUE
cd $ROOT/urlab_bridge
echo "=== obs stress test $(date)  ckpt=$CK  episodes=$N  seed0=9500000  na=${NA:-25}" >> $OUT
run() {   # label, bg dir, OBS_CORRUPT spec
  r=$(EVAL_TAG="stress_$1" OBS_CORRUPT="$3" $PY ../rl/jar/eval_act_jar.py $CK $N --calib $CAL --bg $2 --na ${NA:-25} --p-real 1.0 --seed0 9500000 --workers 14 2>&1 | grep "^ACT jar" | sed -E 's/.*: ([0-9]+\/[0-9]+ = [0-9]+%).*/\1/')
  printf "%-22s %-52s %s\n" "$1" "${3:-(clean)}" "$r" >> $OUT
}
run clean          $BG   ""
run heldout30      $HELD ""
run exp_0.7        $BG   "exposure=0.7"
run exp_0.5        $BG   "exposure=0.5"
run exp_1.4        $BG   "exposure=1.4"
run exp_2.0        $BG   "exposure=2.0"
run wb_warm        $BG   "wb=warm"
run wb_cool        $BG   "wb=cool"
run blur_1         $BG   "blur=1.0"
run blur_2         $BG   "blur=2.0"
run mblur_7        $BG   "mblur=7"
run noise_0.02     $BG   "noise=0.02"
run noise_0.05     $BG   "noise=0.05"
run jpeg_40        $BG   "jpeg=40"
run jpeg_15        $BG   "jpeg=15"
run gamma_0.75     $BG   "gamma=0.75"
run combo_mild     $BG   "exposure=0.8,wb=warm,blur=1,noise=0.02,jpeg=50"
run combo_hard     $BG   "exposure=0.6,wb=warm,blur=1.5,noise=0.04,jpeg=30"
run heldout+mild   $HELD "exposure=0.8,wb=warm,blur=1,noise=0.02,jpeg=50"
echo "=== done $(date)" >> $OUT
