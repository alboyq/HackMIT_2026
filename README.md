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
q_model = sign * q_encoder - offset
```

J3's offset is provisional; its real range is 16° wider than the MJCF's, so its hard stops do not
pin it arithmetically. The five-pose FK check (TCP error under 10 mm) is **still outstanding** and
is the remaining gate before `RealArm` gets a backend — it also resolves J3.

Full method, per-joint hard stops, caveats and remaining steps are in that file and in
`ARM_NOTES.md` on the [`arm-ik-rl`](../../tree/arm-ik-rl) branch, where the arm code lives.
