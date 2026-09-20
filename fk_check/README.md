# FK check — sim-to-real joint map, 2026-09-19/20

Verifies `joint_map_measured.json` on the real arm **without moving it**: the motors stay
disabled (zero MIT frames, `enable()` never called), the arm is held by hand, and only encoder
readings are taken.

| file | what |
|---|---|
| `fk_capture.py` | read one pose (`python fk_capture.py <label>`), work out the 2π lap of each joint from the measured hard stops, refuse ambiguous or drifting readings (>0.005 rad), append to `poses.json` |
| `poses.json` | the eight captures: raw encoders, lap, model-frame angles (poses 3b onward also record the drift) |
| `distances.json` | ruler / caliper distances between the pen dots (mm) |
| `fk_analyse.py` | re-runs the comparison from those two files — no hardware needed |

## Procedure that was used
1. Tape paper inside the arm's reach; put the arm's base clamp where it will stay for the whole check.
2. For each pose hold the arm so the point between the closed pads is on a pen dot, keep it
   still, and run `fk_capture.py`. Touch dots 2 and 3 from more than one arm configuration (that
   is what lets the tool tip be fitted without the gripper model).
3. Measure the distance between every pair of dots and write them in `distances.json`.
4. `python fk_analyse.py` (needs `mujoco`, `numpy`, and `mujoco_menagerie/i2rt_yam`; set
   `YAM_MENAGERIE` or pass `--model`). Gate: every distance within 10 mm.

## Result
Worst error 7.9 mm, rms 4.4 mm over ten distances; the six pairs without dot 3 are within 2.7 mm.
Read `joint_map_measured.json` → `fk_check` for what this does and does not establish (it cannot
see J1 or J6 offsets, and the tool tip is fitted rather than taken from the corrected model).

The first attempt (poses taken before the base was re-aimed, and with a wrapped J2/J3 in the very
first capture) was discarded, not fixed.
