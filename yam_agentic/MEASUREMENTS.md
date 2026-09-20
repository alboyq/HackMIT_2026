# Measurements

Every number here was produced by a script in this folder, on the
[`i2rt_yam`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/i2rt_yam) MJCF from
mujoco_menagerie, MuJoCo 3.10, on an M4 Max. Rerun the tools if the model changes.

## Grasp envelope — `tools/measure_envelope.py` (15 s)

IK-verified: the actual solver, run at 5 azimuths across ±40°, at radii from 0.15 to 0.80 m.
"All 5 azimuths" means every one of them solved.

| pose | reachable |
|---|---|
| tool straight down, TCP 4 cm above the table | **0.15 – 0.50 m** |
| tool straight down, TCP 10 cm above the table | 0.15 – 0.50 m |
| tool tilted 30° off vertical | 0.15 – 0.65 m |
| tool level (side grasp) | **0.65 – 0.75 m only** |
| tool level at z = 0.38 m (mouth height) | 0.15 – 0.70 m |

Two things fall out of this:

- **Objects belong at 0.25–0.45 m.** That is the comfortable top-down band.
- A *level* side grasp is not a close-in option. The arm cannot fold back inside 0.65 m with the
  tool horizontal, so "side grasp" is what you do for things that are far away, not for things
  that are tall. Tall objects close in get grasped top-down or tilted.

### These numbers replaced an earlier, wrong set

A first pass sampled 300k random joint configurations and reported **0.55 m top-down, 0.78 m
side**. Both were too generous, because that sweep used `grasp_site` as the tool centre point and
its z-axis as the tool axis. On this model neither is true:

- `grasp_site` sits **4.4 cm** from the real TCP (the point between the pads),
- its z-axis is **19°** off the tool axis.

`ToolFrame` in `sim/yam_ik.py` now derives both from the pad geometry, and the table above comes
from running the solver rather than from a proxy.

## Gripper — measured by `ToolFrame`

| | |
|---|---|
| pad separation, fully open | **82.8 mm** |
| pad separation, fully closed | 0.8 mm |
| relation | `width = 0.8 mm + 2 × ctrl` |
| usable object width | **≤ ~75 mm** (leaves clearance for servo lag) |
| grip force at the stock `kp=100` | **1.2 N per pad** — too weak, a 150 g apple slips out |
| grip force at `kp=800` (what the sim now uses) | ~10 N per pad |

The force relation is `kp × (commanded − achieved)` at the finger joint, reduced through the
finger linkage. Measured ratio at the pads: ~0.42 of the joint force. The stock value is a
simulation placeholder — an arm rated for a 2 kg payload needs roughly 20 N of grip at μ ≈ 1, so
`kp=800` is a correction toward reality, not tuning. **Verify against the real gripper's force
setting** (`YAM_GRIP_KP` overrides).

## IK accuracy — `sim/test_yam_sim.py::test_ik_round_trip`

- Analytic vs finite-difference Jacobian: max error **2.2e-7** (position), **4.3e-10** (rotation).
- Round trip (target = FK of a random config, seeded 0.06 rad away): converges in all 12 trials,
  max error **< 1 mm**, typically 0.4 mm.
- Cost: **~31 ms** per `solve_best` call, which includes up to 16 seeds.

## Rendering — timed over 100 frames, CPU

| | |
|---|---|
| two 256² cameras, one process | **133 scene-steps/s** (266 frames/s) |
| across 14 workers (extrapolated) | ~1.9 k scene-steps/s |

Enough for the demo data factory; the SO-101 pipeline ran its dataset generation at 71 fps.

## End-to-end pick and deliver — `sim/demo_pick_present.py`

Closed-loop physics, all four objects, from the home pose to presented at the mouth:

| object | geometry | strategy | lift | delivery |
|---|---|---|---|---|
| apple | sphere, 66 mm | top-down | 135 mm | 14.9 cm from mouth, still held |
| mug | cylinder, 64 × 96 mm | top-down | 126 mm | 14.8 cm, still held |
| marker | cylinder, 18 × 124 mm | top-down | 143 mm | 15.1 cm, still held |
| block | box, 50 × 50 × 60 mm | top-down | 138 mm | 14.9 cm, still held |

Each run takes ~6–7 s of simulated time. Filmstrips in `media/`.

## Wrist camera mount — `tools/find_wrist_cam_mount.py`

In the link_6 frame the gripper subtree occupies x ∈ [−0.02, 0.04], y ∈ [−0.049, 0.039],
z ∈ [0.03, 0.138], and the TCP sits at (0, −0.044, 0.130).

Of 84 candidate mounts gridded around the tool axis, **19 have clear line of sight** to the TCP
and to points 5 and 12 cm in front of it. The scene uses back = 0.06 m, up = +0.06 m, side = 0 —
i.e. on the back of the hand, perpendicular to both the tool axis and the jaw axis, which is
where a real wrist bracket goes.

Two hand-picked mounts before this were buried inside the housing and rendered a wall of dark
plastic. Ray casting is cheaper than rendering and looking.
