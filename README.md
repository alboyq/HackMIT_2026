# HackMIT_2026

- [`yam_agentic/`](yam_agentic/) — agentic pick-and-deliver on the openYAM arm: an LLM picks
  *what* to grab and *where* it should go, grounded in a camera. See
  [HANDOFF.md](yam_agentic/HANDOFF.md).
- [`gaze3d/`](gaze3d/) — accuracy-first webcam eye tracking: a GPU ensemble of 3D gaze networks
  plus a calibrated ray-to-screen model, measured at **1.3°** on a stock laptop camera. See
  [HANDOFF.md](gaze3d/HANDOFF.md).

## Before you command the real arm

[`joint_map_measured.json`](joint_map_measured.json) — the measured sim-to-real joint map,
hand-measured on the physical arm 2026-09-19 with the motors disabled throughout.

**The model's joint values are not the arm's joint values.** All six signs are `+1`, but two
joints carry large zero offsets: **J1 (base yaw) is −82°** and **J6 (gripper roll) is +75°**.
Commanding MJCF values straight to the hardware points the whole arm ~82° off heading with the
jaws rolled ~75° — opposite ends of the chain, and invisible in simulation.

```
q_model = sign * (q_encoder + 2*pi*lap) - offset
```

**The encoders also come back one lap off after a power cycle.** The motors report position
modulo one turn, so after the arm is powered off and on a joint can read exactly 2π away from
where the map was measured (J2 and J3 did, after the venue move: raw −6.28 at rest instead of
+0.004). Nothing flags it. Pick the `lap` that puts each reading inside that joint's measured
hard-stop range (`joint_map_measured.json` → `encoder_wrap`); exactly one fits. The openyam
driver's own limits and `move_and_hold.py` presets do **not** do this, and `--preset home` would
plan 6 rad of shoulder travel from a wrapped start. Read the arm, don't drive it, until that is
handled.

**Status.** The five-pose FK check has been run (`fk_check/`): all ten dot-to-dot distances agree
with a ruler to within **7.9 mm** (rms 4.4), so J2–J5 and J3's offset are confirmed (the
alternative J3 offset misses by 35 mm). Two limits: distances cannot see J1 or J6 offsets (those
rest on the hard-stop measurements), and the tool tip was fitted rather than taken from the
corrected `linear_4310` model — re-run `python fk_check/fk_analyse.py --tip X,Y,Z` with that
model's tool point. Still to do before `RealArm` gets a backend: a startup step that applies the
lap rule and puts the limits in the motor's current numbers; then the physical checks and CAN
grounding listed under `outstanding`.

**Replay tool — early, but flown.** [`arm_replay/`](arm_replay/) replays hand-taught poses on the real arm in the
motors' own numbers, with a gravity + friction feed-forward **fitted to this arm** (the real arm needs ~1.375x the sim model's
gravity torque at shoulder and elbow); 60% and 80% of one pose were flown with ~0.5 cm tip error. A failed return now holds the
arm instead of releasing it. Still early: nothing above 80%, no table contact, the retry-on-failure path only tested with fake
motors, and a hard-coded driver path (`~/openyam` is not in git). Read its README before running anything.

Full method, per-joint hard stops, caveats and remaining steps are in that file and in
`ARM_NOTES.md` on the [`arm-ik-rl`](../../tree/arm-ik-rl) branch, where the arm code lives.
