#!/usr/bin/env bash
# One-shot setup: virtualenv, dependencies, model assets, network weights.
# Everything it fetches is gitignored, so this is the only step between a fresh
# clone and `./run.sh`. Safe to re-run; it skips what is already present.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3.13}"
command -v "$PY" >/dev/null 2>&1 || PY=python3

echo "==> virtualenv (.venv)"
if [ ! -d .venv ]; then
  if command -v uv >/dev/null 2>&1; then uv venv --python 3.13 .venv
  else "$PY" -m venv .venv; fi
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> dependencies"
if command -v uv >/dev/null 2>&1; then uv pip install -q -r requirements.txt
else python -m pip install -q --upgrade pip && python -m pip install -q -r requirements.txt; fi

echo "==> MediaPipe face landmarker"
mkdir -p models/weights
[ -f models/face_landmarker.task ] || curl -sSL -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
# canonical_face_model.obj is committed; re-fetch only if it went missing.
[ -f models/canonical_face_model.obj ] || curl -sSL -o models/canonical_face_model.obj \
  https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj

echo "==> third-party network definitions"
mkdir -p gaze3d/third_party
T=gaze3d/third_party
[ -f $T/puregaze_modules.py ] || curl -sSL -o $T/puregaze_modules.py https://raw.githubusercontent.com/yihuacheng/PureGaze/main/model/modules.py
[ -f $T/gazetr_model.py ]     || curl -sSL -o $T/gazetr_model.py     https://raw.githubusercontent.com/yihuacheng/GazeTR/main/model.py
[ -f $T/gazetr_resnet.py ]    || curl -sSL -o $T/gazetr_resnet.py    https://raw.githubusercontent.com/yihuacheng/GazeTR/main/resnet.py

echo "==> gaze weights (Google Drive; slow, ~170 MB total)"
# UniGaze and the ptgaze ResNet-18 come from the Hugging Face hub automatically on
# first load (~2.7 GB for UniGaze ViT-H) — nothing to do here for those.
[ -f models/weights/puregaze_res50_xgaze.pt ] || python -m gdown 1uLQ_1leNBUfwcWs796yTb_bdcUEnrJtK -O models/weights/puregaze_res50_xgaze.pt
[ -f models/weights/gazetr_hybrid_xgaze.pt ]  || python -m gdown 1WEiKZ8Ga0foNmxM7xFabI4D5ajThWAWj  -O models/weights/gazetr_hybrid_xgaze.pt

echo "==> WebGazer baseline (optional, for index.html)"
mkdir -p vendor
[ -f vendor/webgazer.js ] || curl -sSL -o vendor/webgazer.js https://cdn.jsdelivr.net/npm/webgazer@3.3.0/dist/webgazer.js

echo
echo "Done.  ./run.sh   →  http://localhost:8010"
echo "First run also pulls UniGaze ViT-H (~2.7 GB) from Hugging Face."
