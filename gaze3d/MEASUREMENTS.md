# Measurements

Rig: 14" MacBook Pro (M4 Max, 40-core GPU), built-in 1080p camera, macOS 27.
Screen 1512×982 CSS px at 50.0 px/cm. One subject, indoor office light, no chin rest.
Full ensemble: UniGaze ViT-H/14 + PureGaze R50 + GazeTR-Hybrid + XGaze ResNet-18, flip TTA.

Pixels are exact. Degrees depend on viewing distance — all degrees below use the **raw PnP
distance**, not the calibration's fitted depth scale (see "Reporting caveat").

## Headline — validation run, 13 targets including corners

| Metric | Value |
|---|---|
| **Mean error** | **75 px ≈ 1.3°** (at 64 cm) |
| 95th percentile error | 110 px ≈ 1.6° |
| Worst single target | 199 px ≈ 2.9° (middle right — one bad target, neighbours were 60–65 px) |
| Spread about centroid | 46 px ≈ 0.67° |
| Sample-to-sample noise | 40 px ≈ 0.58° |
| Systematic offset | right 4 px, up 4 px — essentially unbiased |
| Dropped frames | 0.0 % |
| Sample rate | 8.0 Hz |

For scale: research trackers land at 0.3–0.5°, Tobii EyeX ≈ 1°, the best published
calibrated-webcam protocols ≈ 1.4°, WebGazer ≈ 4° (and 11° in an independent benchmark).

### Error by screen third

| Region | Error | Region | Error | Region | Error |
|---|---|---|---|---|---|
| top left | 1.28° | top centre | 0.51° | top right | 1.05° |
| middle left | 0.93° | centre | 0.99° | middle right | **2.87°** |
| bottom left | 0.61° | bottom centre | 1.59° | bottom right | 0.61° |

Edges are *not* systematically worse, which is the point of the 3D mapping — WebGazer's
regression degrades sharply toward the corners. Middle-right is a single outlier target,
most likely a blink or a moment's drift; outlier rejection during validation is an open item.

## Calibration fit

350 frames over 14 targets, chosen by leave-one-target-out cross-validation.

| Pipeline | Held-out | Fit |
|---|---|---|
| **geo-additive+ridge30** (chosen) | **1.03° · 71 px** | 1.27° |
| geo-head+ridge3 | 1.03° · 72 px | 1.20° |
| geo-head+ridge30 | 1.04° | 1.25° |
| geo-additive+ridge300 | 1.07° | 1.47° |
| geo-additive (pure geometry) | 1.21° | 2.08° |
| geo-head (pure geometry) | 1.23° | 2.10° |

Fitted parameters: kappa −3.0° / −1.4°, screen tilt 11.9°, screen origin shift 11 mm,
gain/depth scale ×1.23. Fit takes ~1 s.

The top six pipelines sit within 0.05° of each other and pure geometry is only 0.2° behind.
That is the healthy regime: **the geometry is doing the work and the residual regressor is a
small correction**, not a black box compensating for a broken model. An earlier run (337
frames) picked pure geometry outright at 40 px ≈ 1.0° held-out.

## Why the head-motion targets matter

Refitting the same session two ways, then scoring on the head-motion frames only:

| Calibration | Error on moving-head frames |
|---|---|
| Static targets only | 132 px ≈ **3.33°** |
| Static + head-motion targets | 71 px ≈ **1.80°** (in-sample for the full fit) |

This reproduces the literature: head-pose *diversity* during calibration matters more than
the number of targets. It is the single highest-leverage protocol decision in the project.

## Per-model behaviour

Mean angular deviation from the ensemble mean, over one real session:

| Model | Deviation | Latency (1 crop, flip TTA) |
|---|---|---|
| XGaze ResNet-18 | 1.02° | 6 ms |
| UniGaze ViT-H/14 | 1.69° | 77 ms isolated, 96–160 ms with Chrome running |
| PureGaze R50 | 2.01° | 12 ms |
| GazeTR-Hybrid | 2.17° | 9 ms |

Ensemble spread runs 1.4–3.5° live and is exposed in the UI — it is a useful confidence
signal (high spread ≈ the networks disagree ≈ distrust that frame). Note that low deviation
from the mean is *agreement*, not accuracy; the two XGaze-trained ResNets agree with each
other partly because they share training data.

UniGaze-H alone is ~78 % of the GPU budget. The L16 variant roughly doubles the frame rate;
whether accuracy holds at 1.0° on this rig is **untested**.

## Synthetic validation

`test/test_calibration.py` builds a virtual head and screen, injects a 5° kappa, a tilted
screen and 1° of per-sample noise, and checks recovery:

- prior geometry reproduces targets to < 1e-6 px (conventions self-consistent)
- calibration recovers kappa to within 1.2° and predicts **held-out targets at novel head
  poses to 10.6 px ≈ 0.19°**
- static-only calibration still fits to 31 px

Caveat worth remembering: this test passed while the real system was completely broken by a
sign error, because the test generated data with the same wrong convention it tested against.
Synthetic tests here verify internal consistency, not correctness.

## Reporting caveat

The calibration's `log_depth` parameter mostly absorbs the **networks' angular gain error**
(they under-swing), not a distance error. An early report divided by the fitted distance
(79 cm) and printed 1.08°; the raw PnP distance was 64 cm, giving the honest 1.3°. Readouts
and degree conversion now use raw PnP distance throughout. PnP distance itself rests on the
average-face IPD, so treat it as ±5 %.
