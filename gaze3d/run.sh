#!/usr/bin/env bash
# Starts the gaze3d server (GPU inference + web UI) → http://localhost:8010
#   ./run.sh                      full ensemble (UniGaze ViT-H + PureGaze + GazeTR + ResNet-18), ~10 Hz
#   ./run.sh --models unigaze_l16_joint,puregaze_r50,gazetr_hybrid,xgaze_resnet18     ~17 Hz
#   ./run.sh --models unigaze_b16_joint,puregaze_r50,gazetr_hybrid,xgaze_resnet18     ~22 Hz
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
export GLOG_minloglevel=2
exec python -m gaze3d.server --preload "$@"
