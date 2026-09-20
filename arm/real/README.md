# arm/real — everything needed to run the feeding pipeline on the real arm, except moving it

**Nothing in this folder energises a motor.** `RealArm` deliberately raises; `arm/deploy.py` keeps its own
guard. What is here is the perception, calibration and interface layer, tested on the real wrist camera and on
still images, plus the list of what the simulation was quietly assuming and what replaces each assumption.

Environment: `.venv-vision` (separate from `.venv-arm` so training is never disturbed) — `opencv`,
`mediapipe` 1.0 (tasks API), `ultralytics` (YOLO-World) + `clip`. Models: `models/face_landmarker.task`
(Google, 3.7 MB, fetched by hand), `yolov8s-worldv2.pt` + CLIP ViT-B/32 (fetched by ultralytics on first use).

## "Tell it what to feed, and it finds it"

```bash
cd arm/real
../../.venv-vision/bin/python live_perception.py --food strawberry     # window + http://localhost:8091
# type another food name + Enter in that terminal to switch: "cracker", "grape", "banana" ...
```

* `object_sensor.py` — **open-vocabulary** detection (YOLO-World): the food is named in words at run time;
  nothing is trained per food and nothing assumes a background. Distractor names (hand, plate, robot gripper…)
  ride along so the detector has somewhere else to put them. Output: box, **angular width**, and which of the
  three trained grasp classes the food behaves like (`SLOT`: strawberry/grape → `apple`, can/banana → `mug`,
  cracker/cheese → `block`; unknown names fall back on the box's aspect ratio). With a camera pose it also
  gives the table position and metric width by intersecting the ray with the table plane — the same
  assumption the sim's perception-noise model was built on.
* `face_sensor.py` — MediaPipe FaceLandmarker → the feed's only two inputs: **mouth bearing** and **face
  angular width** (degrees), plus `mouth_open` and `facing`.
  * MediaPipe's finder is short-range: on a full 1280×720 frame a face 12% of the width scored under
    threshold (0 found; found at once in a crop). The sensor therefore **zooms**: full frame, then tighter
    windows, centre first.
  * A **turned head looks narrower → reads as farther → the arm would keep coming.** Size is therefore
    `max(width, 0.78 × height)` (height barely changes with a head turn; an open mouth reads closer, the safe
    side), and `facing=False` (nose-to-cheek asymmetry > 0.30) makes the feed HOLD.
  * **Two detectors must agree.** With the zoom search the landmark model found a "face" in five decorative
    plates on a wall, on the real wrist camera (two eyes and a mouth, 25° wide, "APPROACH"). A reading now
    counts only if YOLO-World, in the same frame, also reports a `human face`/`person` box containing the mouth
    (`FaceSensor.confirm`). Re-tested: plates rejected in every frame, the real face still accepted. It costs
    nothing - the detector is already running for the food.
* `camera_model.py` — the only place that knows the lens. Pixels → rays → **angles**. Calibrated (OpenCV
  fisheye K, D) or, until then, an ideal 150°-diagonal equidistant fisheye, clearly labelled UNCALIBRATED.

Tested (`test_perception.py`): detector 0.86–0.92 on people/bus, 0.68 on a tie, ~100 ms per query on CPU; face
found on a 7%-of-width face via zoom, correctly `facing=False` on a ~30° head turn; on the live wrist camera
pointing at a wall it reports no face and no food rather than inventing one.
**Not yet tested: a real person in front of the real camera at feeding distance, real food on the real table,
and the variety of skin tones/backgrounds on THIS camera.** Do that with `live_perception.py` before anything else.

## What the sim was assuming, and what replaces it

| sim shortcut | status in sim now | on the real arm |
|---|---|---|
| object position/width handed to the policy | camera model: bias+jitter, only while in the wrist camera's view, else last-seen | `object_sensor.py` + `handeye.py` + FK |
| first reading given even if the target is out of view | **removed** (`strict_first_look`): a fixed prior until actually seen; 150° lens sees the object at t=0 in ~75% of episodes and within a few steps otherwise | same; add a scan pose only if the real lens sees less |
| "reach has arrived" judged by true distance | **removed** (`handoff.sensors_only`): camera gap + motor speeds + FK tilt + box squareness from the outline (±5°) | identical test; box yaw from `cv2.minAreaRect` on the detection |
| "I have hold of it" from contact truth | **removed**: `jaws_blocked()` — commanded shut, stalled, not empty (gripper encoder) | identical |
| lift steered/ended on true object pose | **removed**: camera estimate + TCP rise from FK | identical |
| safety shield from true geometry | **removed**: `shield_cam()` — person = last camera sighting (or the marked seat), arm = FK | identical |
| person's head size known | never assumed: ±8% per episode, controller assumes 145 mm | per-person calibration: hold the food at his lips once, record the face width in degrees |
| food-to-face contact | only in `CALIBRATE=1` runs, to record the benchmark (50.9° median) | none exists; stop on the benchmark minus a stand-off |
| which food | one-hot from the request | the spoken/EEG request → `ObjectSensor.set_food()` → `SLOT` |
| clean position control | sim actuators track within mm | **does not exist**: see `arm_interface.py` — gravity feed-forward, joint map, hold-never-disable, watchdog |
| pinhole 75–150° camera | angles only, FOV swept 100/130/150 with no effect | calibrate the fitted lens: `calibrate_camera.py` |

With every row marked **removed** active, 3 × 40 full pick-and-feed runs: **fed 88/120 (73%) with the fine-tuned
pinch policy (61% with the old one), unsafe arrivals 0, dropped 1.**

## `real_arm.py` — the backend, completing `arm_replay/` for this pipeline (NEVER RUN ON THE ARM)

Reuses `arm_replay/replay_pose2.py`'s tested `Gravity` and `Sender` (one MIT command per joint per tick with
gravity + friction feed-forward; state from the replies, because reading a live motor makes it go limp) and adds
what that README lists as not done:

* **Hold, never disable** — a guard trip keeps sending the last *commanded* setpoint with feed-forward.
* **Continue from the commanded setpoint**, never the measured one (re-seeding drops the holding torque).
* **Torque / contact stop** — if a motor reports more than `CONTACT_NM` (7/8/7/3/2/2 N·m) beyond its
  feed-forward for 3 ticks, e.g. pressing into the table, the arm **retraces its last 15 setpoints** to relieve
  the pressure, then holds.
* **Release only at rest** — `shutdown()` ramps back to the enable pose and disables only if the replies confirm it.
* **Policy targets through the measured joint map**, clipped 0.10 rad inside the measured hard stops and
  rate-limited again (0.02 rad/tick): the backend does not trust its caller.
* `connect()` needs `YAM_REAL_ARM=I_AM_AT_THE_ARM_WITH_THE_ESTOP` **and** a typed confirmation at a terminal, and
  is deliberately left unwired: the first powered run of new code is a person's job.

`python arm/real/real_arm.py --selftest` (fake motors, no CAN): 16 checks, all pass.
**Blocking gap: the gripper (motor 0x08) open/closed readings were never measured, so grip targets are refused.**
Also unresolved and inherited: no watchdog (TIMEOUT=0 — do not change the register), the driver's `atexit` disables
motors so a faulted process must be kept alive, the gravity model is ~1.7× off at the extended elbow.

## Order of work to go live (none of it done here)

1. `calibrate_camera.py` with the lens that will be used, focus locked. Aim for rms < 0.7 px.
2. `live_perception.py` with a real person and real food; check distances: the face should read ~47° at the
   intended stop. Record one per-person calibration.
3. `handeye.py` from a hands-on, motors-off session (no powered motion needed).
4. Build `RealArm` against the five requirements in `arm_interface.py`. Dry-run the whole pipeline on
   `DryRunArm` first, then the real arm with no person, then a mannequin, then a person — with an E-stop in hand.
