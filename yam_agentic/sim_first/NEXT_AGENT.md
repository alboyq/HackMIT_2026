# Handoff — pick this up here

Written 2026-09-20, early hours. Read this, then `RESUME.md` for how to restart anything.

The sim-first pipeline is built and runs end to end. One thing is wrong and one thing is
unproven, and both are written up honestly below. **Do the gripper first — it is a correctness
bug that invalidates every grasp number until it is fixed.**

---

## 1. Task 1 (blocking): the gripper in sim is the wrong one, and the right one does not grasp

### What is certain

The bench arm has **rack-and-pinion sliding jaws driven by a DM4310** (user confirmed by eye;
partner's read-only inventory found a DM4310 at CAN id `0x08`). That is i2rt's **`linear_4310`**,
whose 95 mm throw matches I2RT's published YAM spec.

The MuJoCo Menagerie `i2rt_yam` model — which this whole stack is built on — ships
**`crank_4310`** instead (79 mm throw). Confirmed by matching its tip geom offsets
(`0.171097 -0.203301` / `-0.124901`) against `gripper/crank_4310/crank_4310.xml`.

So the simulated jaws are ~16 mm narrower than the real ones, the object-size cap
(`check_graspable`, currently ~75 mm) is too strict, and — most important for a policy whose main
input is the wrist camera — **the tips on screen are the wrong shape for the whole episode**.

### Where it stands

`YAM_ARM` selects the arm model (`sim/yam_scene.py`):

| value | model | grasps? |
|---|---|---|
| `stock` (**current default**) | menagerie `yam.xml`, crank_4310 | **yes** — 12/12 tests, all datasets so far |
| `linear_4310` | `_yam_linear4310.xml`, built by `make_arm_linear4310.py` | **no** |

### What is ruled out — do not redo this

- **Not the mount transform.** The graft in `make_arm_linear4310.py` puts `grasp_site` at
  `(0.2552, 0, 0.1734)` at zero joints. i2rt's own composer produces **the same point**.
- **Not a missing composer.** i2rt ships one, and it is authoritative because it reads the
  per-arm mount transform from the gripper's YAML config, which cannot be guessed:

  ```python
  from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
  path = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310)  # returns a PATH, not XML
  ```

  Needs `python-can tyro pyyaml pydantic crcmod` (already installed in the Mac venv). That model
  is kinematics only — no actuators, no armature, no tuned gain classes — which is why
  `make_arm_linear4310.py` grafts into the menagerie arm instead.
- **Not collision flags.** Tip geoms compile with `contype=1 conaffinity=1`, mesh type,
  `rbound=0.079`.
- **Not grip force.** The finger actuator is re-gained to `kp=800` (~24 N). An exact-string
  replace silently missed this once; it is a regex now, and asserts it patched exactly one line.
- **Not object placement along the tool axis.** Swept −40 mm to +125 mm from the tip midpoint;
  the jaws reach `q=0.003` with 1–2 glancing contacts at every offset.
- **Not the TCP heuristic.** Three were tried (pad spheres, `grasp_site`, mirror-symmetric
  jaw-gap). All behave the same.

### The actual smoking gun, and the hypothesis to test first

Measuring the **minimum distance between the two tip meshes** as the fingers open, on
**i2rt's official composed model** (so this is not an artefact of the graft):

| joint value | min gap between tip meshes |
|---|---|
| `q = 0.0` | 0.1 mm (closed) |
| `q = 0.024` | 4.5 mm |
| `q = 0.0475` (full open) | **16.4 mm** |

**The jaws appear to open only ~16 mm, yet the tip bodies' relative displacement is a correct
95 mm.** Those two facts are only consistent if the tips are *sliding past each other* rather
than separating — or if the metric is wrong.

**Most likely explanation, and where to start:** the `min distance over all mesh vertices` metric
is measuring the wrong pair of surfaces. The tip meshes are whole finger brackets — long, angled
arms that overlap along the sliding direction. Two long parallel brackets that slide apart
laterally still have near edges a few mm apart, so the global minimum stays small while the
*pad faces* genuinely open 95 mm. If so the model is fine and only the TCP/grasp geometry is
wrong.

To settle it in one step, measure the gap **projected on the jaw axis** rather than globally:

```python
jaw = unit(relative displacement of the two tip bodies between q=0 and q=0.0475)
# for each tip, project its vertices onto `jaw`; take the left tip's max and the right tip's min
# (i.e. the two facing extremes). Their difference is the real opening.
```

If that shows ~95 mm, the fix is purely the TCP: find the pad-face centre from those same
projected vertices and set `tcp_local` from it. If it shows ~16 mm, then the joint axes or the
`polycoef` sign in the graft are wrong — try `joint8 = −joint7` (the crank model mirrors its
fingers; `linear_4310` uses `polycoef="0 1 0 0 0"`, i.e. same sign, which is what is implemented).

**Ground-truth check either way** (cheap, decisive, no reasoning required): put a sphere on a 3D
grid around the jaw region, close, and see which cells actually get gripped. The 1D version of
this is in the session history; it needs to become 3D because the tool-axis sweep alone came up
empty.

When it grasps: flip the default to `linear_4310` in `sim/yam_scene.py`, re-run
`pytest sim/test_yam_sim.py` (12 tests), widen the object-size cap to ~87 mm, and **regenerate
the datasets** — every existing one has the wrong tips in every wrist frame.

---

## 2. Task 2: nothing has ever seen a real camera frame

Unchanged from `../HANDOFF.md`, and still the highest-risk unknown after the gripper. Gate B in
`../GAME_PLAN.md`: feed real iPhone + wrist frames and real 2D prompts to a trained policy with
the arm posed and **stationary**, and check the first predicted action points at the real object
the way it does in sim. Zero arm risk, and it measures the sim-to-real gap before anything moves.

Real reference frames are already captured in `real_ref/` (scene + wrist) and were used to set
the camera model: the wrist camera is remounted to the real mount point and warped through
`sim/fisheye.py` to match the real lens, and the `real` augmentation profile in `sim/yam_data.py`
replaces a first attempt whose blur was far heavier than any real camera.

Backgrounds stay **randomised** (800-image pool + procedural fields). The real frames informed
quality, lighting and framing only — they are not the training backdrop.

---

## 3. What works right now

```bash
cd ~/so101Sim
urlab_bridge/.venv/bin/python -m pytest rl/yam/test_yam_sim.py -q     # 12 passed
YAM_AUG_PROFILE=real urlab_bridge/.venv/bin/python rl/yam/gen_demos_yam.py rl/runs/<run>/demos 1400 --workers 12
rl/yam/run_chain_yam.sh rl/runs/<run> 12000                            # train + rolling eval
```

- **Demo generator**: ~85 % of randomised episodes kept, ~100 frames each, 1,200 episodes in
  ~6 min on 14 cores. Randomises objects (which, where, size, colour), the user's position, the
  scene camera over a 50 × 45 × 45 cm box, servo softness, backgrounds and whole-image camera
  effects. Observations are 2D prompts only — object box + mouth point + face size — so there is
  no camera matrix or hand-eye transform anywhere on the critical path.
- **Trainer**: two-camera ACT, resumes from `act/LATEST` after an unplug (see `RESUME.md`).
- **Evaluator**: closed-loop on held-out backgrounds with camera effects on.

### State of the runs

| run | what | verdict |
|---|---|---|
| GX10 `runs/yam_g1` | 2,104 demos / 197k frames, ACT at ~367 samples/s, near step 16000 | **first eval: 0/56.** Predates the wrist remount and has the wrong gripper. Treat as a pipeline check only. |
| Mac `rl/runs/yam_v1` | 1,017 demos | superseded — blur far too heavy |
| Mac `rl/runs/yam_v2` | 6 shards, stopped part-way | **delete it** |

**0/56 on yam_g1 is not yet evidence the approach fails** — that data has the wrong gripper, the
old wrist mount and the over-heavy blur. But do not start a long run until the gripper is fixed,
or you will burn hours on data that has to be thrown away.

---

## 4. Order I would work in

1. **Gripper** (§1). Nothing downstream is trustworthy until this is right.
2. Regenerate one dataset, train, and check the first eval is non-zero. If it is still 0 %, the
   suspects are the 2D-prompt conditioning and the 10 Hz control rate, in that order — dump a
   filmstrip of a *policy* rollout next to an *expert* rollout and compare where they diverge.
3. **Gate B** (§2) on real frames.
4. Then hardware, behind the limiter, per `../HANDOFF.md` §8 and the safety rules in
   `ARM_NOTES.md` on the `arm-ik-rl` branch.

## 5. Things that cost me time — avoid them

- **Go to the vendor SDK before hand-deriving geometry.** i2rt ships the composer, the gripper
  CAD and per-arm mount configs. I spent a long time on frame algebra that turned out correct
  anyway, while the real bug was elsewhere.
- **`pkill -f <pattern>` over ssh matches the ssh session's own command line** and kills the
  shell before it can relaunch. Kill by PID, or put the kill in a script file
  (`/tmp/relaunch_gx10.sh` on the GX10 does this).
- **Exact-string `.replace()` on MJCF is fragile.** One silently missed the gripper gain when a
  neighbouring attribute changed, leaving 1.2 N of grip and hours of "why does it slip".
- **Measure, do not reason, about contact geometry.** Every analytical guess about the TCP was
  wrong; the sweeps and renders found the truth in minutes.
