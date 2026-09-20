# Handoff — gaze3d, accuracy-first webcam eye tracking

**Status as of 2026-09-19.** The tracker is built, working, and measured on a real subject:
**75 px ≈ 1.3° mean error**, 0 % dropped frames, 8 Hz, on a stock MacBook webcam with no chin
rest. Everything here runs from this folder — `./setup.sh` then `./run.sh`. The numbers and how
they were obtained are in [MEASUREMENTS.md](MEASUREMENTS.md); operating instructions are in
[README.md](README.md).

---

## 1. The goal, and what makes it hard

One sentence: **get the best possible gaze accuracy out of an ordinary laptop webcam, and keep
that accuracy when the user moves their head after calibration.**

Compute was explicitly *not* a constraint — the brief was to use whatever GPU-based ML pipeline
gives the best accuracy. That rules out the usual browser-JS trackers immediately and puts a
local Python process with real networks on the table.

The hard part is not predicting gaze from a face crop; pretrained networks do that at 4–5°. The
hard part is the last mile:

- A gaze *direction* is useless without a gaze *origin* in metric 3D and a *screen* to hit.
- A single camera cannot tell a small face up close from a large face far away, so the origin's
  scale is ambiguous.
- Every person's visual axis differs from their optical axis (the kappa angle, ~5°), and that
  offset is fixed in the **head** frame, not the camera frame — so a naive constant correction
  silently decays as soon as the head rotates.
- Regression-based trackers (WebGazer and friends) hide all of this by memorising a mapping at
  one head pose. They collapse the moment you shift in your seat.

## 2. Decisions that shaped the design, and the evidence behind them

Two literature sweeps (~150 sources) preceded any code. The conclusions that actually changed
the design:

| Finding | Consequence here |
|---|---|
| ETH-XGaze-trained models are by far the most robust zero-shot (independent Gaze4HRI benchmark); training data matters more than architecture | Every model in the ensemble is XGaze-trained. Gaze360-family raw-crop models (L2CS, MCGaze) were evaluated and **rejected** — ≥10° and worst-in-class zero-shot |
| **UniGaze** (WACV 2025), a MAE-pretrained ViT trained jointly on XGaze + MPIIFaceGaze + GazeCapture + EyeDiap + Gaze360, has the best cross-dataset numbers of any downloadable checkpoint | UniGaze ViT-H/14 is the ensemble's anchor |
| Head-pose **diversity** during calibration beats the number of targets. One target under several poses beats 9–13 targets at one pose; single-pose calibration degrades ~2× off-pose | The calibration includes head-motion trials. Measured here: 3.33° → 1.80° on moving-head frames |
| The kappa offset is fixed in the head frame (Guestrin & Eizenman); `g_visual = R_h R_κ R_hᵀ g_optical` | Implemented as the `head` kappa mode, with the naive additive-in-normalised-frame variant as a competing candidate; cross-validation picks per person |
| Geometry beats ML at few calibration points; hybrid geometric regression wins under head movement | The fit is geometry-first with a *small* ridge residual, and the choice is cross-validated |

## 3. Architecture

```
camera ──▶ MediaPipe FaceLandmarker (478 pts incl. iris)
             │
             ├──▶ solvePnP vs. canonical 3D face  ──▶  head pose R_h, face centre o (mm)
             │
             └──▶ ETH-XGaze normalisation (virtual cam, f=960, d=600mm, 224², roll cancelled)
                     │
                     ▼
              GPU ensemble ──▶ gaze direction g_n in the normalised frame
              (UniGaze ViT-H, PureGaze R50, GazeTR-Hybrid, XGaze R18; flip TTA)
                     │
                     ▼
              de-normalise:  g_cam = R_nᵀ g_n
                     │
                     ▼
              person-specific kappa in the HEAD frame
                     │
                     ▼
              ray/screen-plane intersection  ──▶  (u, v) screen px
                     │
                     ▼
              1€ filter + saccade reset + blink hold
```

Pipeline stages live one-per-file in `gaze3d/`; `pipeline.py` is the per-frame loop that wires
them together and `server.py` exposes it over a WebSocket to `gaze3d.html`.

## 4. Coordinate conventions — read this before touching the geometry

**This is where the project's one serious bug lived.** Getting a sign wrong here does not
produce a small error; it produces a tracker that looks plausible and is completely wrong.

- **Camera frame is OpenCV**: x right, y down, z forward *into the scene*. Millimetres.
- The MediaPipe canonical face model ships as x-right / y-up / z-toward-camera in cm; `face.py`
  flips y and z and scales to mm so a frontal face gives `R ≈ I`.
- **Screen u runs along camera −x.** The screen and the camera both face the user, so screen-right
  is camera-left. The top-left pixel sits at **+x**. Getting this backwards was the original bug:
  gaze yaw came out mirrored, the fitter crammed the contradiction into the depth and tilt bounds
  (reporting an absurd 26 cm viewing distance), and only the ridge regressor kept the output from
  being nonsense. Fixed in `calibration.py::geometric_pog` / `screen_point_mm`.
- **Gaze vectors use the XGaze convention**: `g = [−cos(p)·sin(y), −sin(p), −cos(p)·cos(y)]` for
  pitch `p`, yaw `y`.
- **De-normalisation is rotation-only**: `g_cam = R_nᵀ g_n`. Do not rescale the vector.
- **Per-model output order was verified empirically by a mirror test** (feed the flipped image;
  yaw must flip sign, pitch must not):
  - UniGaze, PureGaze, XGaze-ResNet18 → `(pitch, yaw)`
  - GazeTR → `(yaw, pitch)`
  - Do not trust the upstream READMEs on this; re-run the mirror test if you swap a model in.
- **Per-model preprocessing differs**: the GazeHub models (PureGaze, GazeTR) take **BGR, 0–1, no
  ImageNet normalisation**; UniGaze and the ptgaze ResNet-18 take **RGB with ImageNet
  normalisation**. `models.py` encodes this per wrapper.

## 5. The calibration model

Nine parameters, fitted by bounded least squares with Gaussian priors, on all collected frames:

| Parameter | Count | Prior |
|---|---|---|
| kappa (visual–optical offset) | 2 | 15° |
| screen plane rotation `ω` | 3 | 4° |
| screen origin `t` | 3 | 15 mm |
| `log_depth` (gain/depth scale) | 1 | 0.25 |

Then a residual regressor (ridge at three strengths, or a 2nd/3rd-order polynomial) is fitted on
top. **Every combination is scored by leave-one-target-out cross-validation and the winner is
chosen per person** — including the "no residual regressor at all" option, which has won on one
of the two real sessions. This is what stops the residual model from quietly absorbing a broken
geometry, and the candidate table is printed in the report so regressions are visible.

`log_depth` deserves a warning: it mostly absorbs the **networks' angular gain error** (they
under-swing), not a distance error. Do not read it as a distance measurement — degree
conversions use the raw PnP distance.

## 6. Distance and intrinsics — why there is no "how far away are you" input

Originally the user typed their viewing distance to pin down the focal length. That is now
gone:

- macOS reports `videoFieldOfView = 0` for the built-in camera (checked across every capture
  format via AVFoundation, and for a Continuity iPhone camera too), so the FOV **cannot** be read
  from the OS. This was tried and it does not work.
- Instead: a 66° horizontal-FOV prior (`--hfov`) sets the initial focal length, and the
  calibration's depth-scale parameter refines the combination. The head-motion trials (lean in and
  out, slide sideways) are what make that scale identifiable from data.
- Viewing distance is now a **measured readout**, not an input.

## 7. Environment gotchas

- **mediapipe is pinned to 0.10.35.** 1.0.x aborts on this macOS build with
  `DrishtiMetalHelper … Check failed: service_ Service is unavailable.`
- **Center Stage must be off.** It crops the camera dynamically, changing the intrinsics mid-run.
- **Weights are not in this repo** — 2.5 GB, and UniGaze (MG-NC-RAI-2.0), PureGaze (CC BY-NC) and
  GazeTR (CC BY-NC-SA) are all **non-commercial**; redistributing them from a public repo would
  violate their licences. `setup.sh` fetches them. Same for the third-party network definitions.
- **Core ML is not worth it.** The UniGaze ViT-H was exported to Core ML and benchmarked: 42 ms vs
  47 ms on PyTorch/MPS with `CPU_AND_GPU`, and *slower* on the Neural Engine (a 632M-parameter ViT
  thrashes the ANE weight cache). Numerically it matched to 0.0008 rad. Don't repeat this
  experiment expecting a win; a smaller backbone is the real lever.
- `gaze3d/third_party/gazetr_model.py` has a hard-coded `.cuda()` on line 114. It is never reached
  because `models.py` overrides `_forward`. Don't "fix" it by editing the vendored file.
- The Claude-in-Chrome extension was not connected during development; Playwright (headless
  Chromium) was used to drive the page end-to-end instead.

## 8. What is verified, and how

- `test/test_calibration.py` — synthetic head + screen, injected kappa/tilt/noise; checks prior
  geometry to 1e-6 px, kappa recovery, and held-out prediction at **novel head poses** (0.19°).
- `test/metrics.test.mjs` — 12 tests: the error metrics and the spread-point sampler.
- End-to-end page flow driven headless (calibrate → validate → head-motion → tile test) with zero
  console errors.
- Two real calibration sessions with a human subject; see [MEASUREMENTS.md](MEASUREMENTS.md).

**Caveat that cost real time:** the synthetic test passed while the real system was completely
broken by the screen-axis sign error, because the test generated its data with the same wrong
convention it was testing against. Any change to the geometry must be re-validated on a real
fixation set, not just on the simulation.

## 9. Next steps, in priority order

1. **Run the head-motion test and record per-movement-type numbers.** The UI reports error binned
   by head rotation, distance change *and* movement type (yaw / pitch / depth / lateral / roll),
   but no real run has produced those bins yet. This is the single most informative missing
   measurement — it says whether rotation or depth is the weak axis, and therefore what to fix.
2. **Re-validate with the randomised motion protocol.** Both real sessions predate it. The current
   protocol is five movement types × two random, well-separated locations, shuffled; the earlier
   one used five fixed locations. Confirm 1.0–1.3° still holds.
3. **Outlier rejection during validation.** One target in the last run was 2.87° while its
   neighbours were ~0.9° — almost certainly a blink or a drift. Blink gating exists for the live
   signal but validation windows accept everything.
4. **Learn per-model ensemble weights** from the calibration residuals. Weights are currently
   uniform; XGaze-ResNet18 is the closest to the ensemble mean and UniGaze is the strongest model,
   so uniform averaging is unlikely to be optimal. The per-model vectors are already stored in
   every calibration sample (`per_model`), so this can be fitted offline from existing sessions.
5. **Try the L16 ensemble** (`--models unigaze_l16_joint,…`) and check whether accuracy holds at
   ~1.0°. If it does, it doubles the frame rate to ~17 Hz and should become the default.
6. **Temporal modelling.** Every network here is single-frame. Fixation-aware aggregation (median
   over a detected fixation) or a multi-frame model is the obvious next accuracy lever.
7. **More pixels on the eyes.** At 1080p and 64 cm each eye is ~40 px wide. This, not depth
   sensing, is the biggest hardware lever for a camera-only pipeline. (A depth camera would fix
   the scale ambiguity and steady the head pose — worth maybe 10–20 % — but cannot see iris
   geometry and so does nothing for the direction estimate, which dominates the error. Active IR
   with corneal glints, i.e. a Tobii, is the real accuracy upgrade.)

## 10. Where to look

| Question | File |
|---|---|
| How is a frame turned into a gaze point? | `gaze3d/pipeline.py::_step` |
| The screen geometry / kappa / ray maths | `gaze3d/calibration.py` |
| Model I/O conventions and preprocessing | `gaze3d/models.py` |
| Head pose, eyeball model, intrinsics | `gaze3d/face.py` |
| XGaze normalisation | `gaze3d/normalize.py` |
| Calibration protocol, UI flow, report | `js/gaze3d.js` |
| Error metrics (shared with the WebGazer bench) | `js/metrics.js` |
| WebSocket command surface | `gaze3d/server.py::handle` |

`index.html` + `js/bench.js` are the original WebGazer baseline bench, kept for comparison — same
metrics, same report format, ~4–11° instead of 1.3°.
