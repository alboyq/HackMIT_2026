# gaze3d — webcam eye tracking, accuracy-first

An appearance-based 3D gaze tracker for a plain laptop webcam. An ensemble of pretrained
gaze networks runs on the GPU in a local Python process; a PnP head pose turns each
prediction into a **ray in 3D** which is intersected with the physical screen plane. A short
person-specific calibration fits the remaining geometry, so the mapping survives head
movement instead of memorising one pose.

**Measured: 75 px ≈ 1.3° mean error** at 64 cm on a 14" MacBook Pro, 13 validation targets,
0 % dropped frames. See [MEASUREMENTS.md](MEASUREMENTS.md). Context, design rationale and
next steps are in [HANDOFF.md](HANDOFF.md) — **read that first.**

## Run it

```bash
./setup.sh      # venv + deps + model assets + weights. Once, ~5 min (plus a 2.7 GB
                # UniGaze download on first launch).
./run.sh        # → http://localhost:8010
```

Open the URL in Chrome. Click **Start camera** — macOS asks *the terminal* for camera
permission the first time. Then **Calibrate**.

> **Turn Center Stage off** (Control Center → Video Effects). It crops the camera
> dynamically, which silently changes the intrinsics mid-run and invalidates the geometry.

Faster variants, if 8 Hz is not enough:

```bash
./run.sh --models unigaze_l16_joint,puregaze_r50,gazetr_hybrid,xgaze_resnet18   # ~17 Hz
./run.sh --models unigaze_b16_joint,puregaze_r50,gazetr_hybrid,xgaze_resnet18   # ~22 Hz
./run.sh --no-tta                                                               # skip flip TTA
```

## What a session looks like

1. **Camera** — face mesh, head pose and the normalised 224² crop the networks actually see.
2. **Calibrate** — 9 static targets, then 10 head-motion trials (five movement types at two
   random, well-separated locations each). On the amber targets you keep your eyes on the dot
   *while moving your head*; that is what makes the mapping pose-robust.
3. **Measure** — 13 validation targets, no marker shown, error reported in px and degrees.
4. **Head-motion test** — error binned by head rotation, distance change and movement type.
5. **Tile hit test** — how often gaze lands in the right tile of a 4×3 grid.

Calibration samples are written to `profiles/session-*.jsonl` as they are collected and the
fitted profile is saved automatically, so a crash or a closed tab never costs you a run —
**Recover last calibration** on the camera screen re-fits from disk.

## Tests

```bash
source .venv/bin/activate && python test/test_calibration.py   # geometry + calibrator
node --test test/metrics.test.mjs                              # metrics + point sampler
```

## Layout

```
gaze3d/          python package
  camera.py      threaded capture, always hands out the newest frame
  face.py        MediaPipe landmarks, PnP head pose, per-eye 3D eyeball model
  normalize.py   ETH-XGaze data normalisation + gaze vector conventions
  models.py      the four network wrappers and the ensemble
  calibration.py screen model, kappa, the fit and its cross-validated selection
  filters.py     1€ filter with saccade reset and blink hold
  pipeline.py    per-frame loop, sample persistence, recovery
  server.py      FastAPI: static page, WebSocket stream, MJPEG previews
gaze3d.html      the bench UI   (js/gaze3d.js, css/gaze3d.css)
index.html       the original WebGazer baseline bench, kept for comparison
test/            geometry tests (python) and metric tests (node)
models/          canonical face model; everything else fetched by setup.sh
profiles/        calibration samples and fitted profiles (gitignored)
```
