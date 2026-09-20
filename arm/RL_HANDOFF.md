# Handoff — RL feeding curriculum on the OpenYAM

**State as of 2026-09-19 20:30 PDT.** Everything below is on branch `arm-ik-rl`, pushed.
Simulation only. **The physical arm has not been moved by any of this work.**

Read this with [ARM_NOTES.md](ARM_NOTES.md) (hardware research) and
[../yam_agentic/HANDOFF.md](../yam_agentic/HANDOFF.md) (the teammate's IK/perception track,
which is a *different and still-valid* route to the same demo).

---

## 1. The one-paragraph version

We are training a low-dimensional PPO policy to pick one named object out of three on a table
and present it to a seated person's mouth, as a four-stage curriculum
(`reach → grasp → lift → present`). **Reach is solved** (100% rollout success, ~21–80 steps).
**Grasp is stuck at 0% after 1.3M steps and is the live blocker.** Training runs unattended via
a chain script with checkpoint/resume, and can be watched live in a browser.

The single most important lesson from this session: **every success counter has lied at least
once.** Three separate times a stage reported a high success rate while doing something useless,
and once it reported 0% while working. Always confirm behaviour with `filmstrip.py` before
believing a number.

---

## 2. Access

```bash
ssh -J jump@129.153.206.6 asus@10.10.10.4          # key auth already installed
```

An SSH alias `gx10` exists in the user's `~/.ssh/config`. The machine is `gx10-f3f2`, Ubuntu
24.04.4 aarch64, 20 cores, 121 GiB RAM, NVIDIA GB10, 783 GiB free.

**The IP has changed twice tonight.** If it is unreachable, ask for the current address rather
than assuming the box is down.

Every command below assumes:

```bash
cd ~/HackMIT_2026-arm-ik-rl
export YAM_MENAGERIE=$PWD/third_party/mujoco_menagerie PYTHONPATH=.:arm/rl OMP_NUM_THREADS=1
```

`YAM_MENAGERIE` is **required**: `arm/ik/scene.py::_find_menagerie` walks parent directories for
`mujoco_menagerie/i2rt_yam` and never looks in `third_party/`, so nothing runs without it.

The virtualenv is `.venv-arm` (Python 3.12, torch 2.13 CPU aarch64, SB3 2.7.0, MuJoCo, OpenCV 5).

---

## 3. What is running right now

| process | what |
|---|---|
| `chain_curriculum.sh` | drives `reach → grasp → lift → present` unattended |
| `run_resumable.py --stage grasp` | the live trainer, 20 envs, ~3,770 steps/s |
| `stream_policy.py --auto-stage` | MJPEG viewer on `localhost:8089` |
| `watch_policy.py` | OpenCV window on the GX10's own monitor (`DISPLAY=:1`) |

To watch from a laptop:

```bash
ssh -J jump@129.153.206.6 -L 8089:localhost:8089 asus@10.10.10.4
# then open http://localhost:8089
```

The page shows two lines: the episode being watched (rolling last-20 success) and what the
trainer is doing (stage, steps, approx episodes, steps/s, rollout success). `--auto-stage` makes
it follow the chain when the stage changes, with no page reload.

**Viewers are separate processes that only read checkpoints. They never slow or disturb
training.** Training is unthrottled; only the viewer sleeps to real time.

---

## 4. THE CURRENT BLOCKER

```
TRAINING grasp | 1,310,720 steps | success_rate 0 | ep_len_mean 294 | ep_rew_mean 5.11
```

Reach solved itself by ~300k. Grasp has had 1.3M steps and has never succeeded.
`ep_len_mean 294` out of 300 means it almost never terminates — it is not grasping at all.

**Do this first, before touching any hyperparameter:**

```bash
.venv-arm/bin/python arm/rl/scripts/filmstrip.py \
  --run-dir runs/feed-grasp --stage grasp --panels 6 --seed 3 --out /tmp/g.jpg
```

Then look at the image. Each panel is captioned with the reward phase
(`APPROACH` / `CLOSE` / `SECURE`), gripper opening, contact, distance and object height.

The question to answer is *which* of these is happening:

1. never reaches `in_position` (so CLOSE never unlocks) — check `align_xy_m` / `align_z_m`;
2. reaches position but does not close — check the `close_bonus` magnitude vs `time_penalty`;
3. closes but never gets both pads on the object — likely the grasp height, since the target is
   the object's centre and the reach hand-off point is 3.5 cm above it;
4. pinches but the 0.3 s hold or the 4 cm displacement gate rejects it.

`diag_settle.py` and `diag_policy.py` in `/tmp` on the box (also reproduced in this repo's
history) are templates for instrumenting a rollout and counting which condition fails.

A plausible cause not yet ruled out: reach now ends ~3.5 cm **above** the object
(`reach_offset_z`), and grasp's target is the object **centre**, so grasp must first descend
through a region where `in_position` is false. Nothing rewards that descent specifically.

---

## 5. The environment

`arm/rl/hackmit_rl/envs/openyam_feed.py`, config `arm/rl/configs/feed.yaml`.

**Observation (33), fixed across all stages so checkpoints transfer between them:**

| size | channel | real source |
|---|---|---|
| 6 | joint angles | motor encoders, 16-bit — exact |
| 6 | joint velocities | motor feedback, 12-bit — **measured, not differentiated** |
| 1 | gripper opening | motor `0x08` |
| 3 | target − TCP | **camera** |
| 3 | mouth − TCP | **camera** |
| 3 | mouth − object | **camera** |
| 3 | object one-hot (`apple`, `mug`, `block`) | the request itself |
| 1 | object width | **camera** |
| 7 | previous action | internal |

**Action (7):** six joint deltas + gripper. The policy does **not** call IK; it learns joint
control directly. The IK solver in `arm/ik/` is a separate, independent route.

**The camera-derived channels carry realistic error** (`perception_noise` in the config):
per-episode *bias* (calibration is wrong the same way all run) plus per-frame *jitter*, scaled
by range because eye-in-hand error shrinks as the lens closes in. Measured as the policy sees
it: object position ~17.5 mm mean / 35 mm max, width ~10%. **Reward and success still use
ground truth** — the policy must succeed while being told something slightly wrong.

**Three-phase reward.** The phases are mutually exclusive and ordered, which is what stopped the
jaws closing before the arm arrived:

- `APPROACH` — pays for closing the gap; **charges** for closing the jaws early or touching the
  object. Never pays a per-step bonus for merely being in this phase.
- `CLOSE` — unlocks only when the TCP is over the object (`align_xy_m`) **and** at grasp height
  (`align_z_m`). Two separate tests on purpose: one 3D distance also passes when the gripper is
  *beside* the object.
- `SECURE` — both pads on the object: hold, lift, carry, deliver.

**Stage success criteria:**

| stage | success |
|---|---|
| `reach` | within 35 mm **and settled** (worst joint ≤ 0.20 rad/s, TCP ≤ 0.05 m/s) for 0.15 s |
| `grasp` | **both** pads on the object, jaws closed, object still within 4 cm of where it started, 0.3 s |
| `lift` | carrying (pinched + closed + 8 cm up) for 0.5 s |
| `present` | object within 5 cm of the mouth, still carried, **food leading the jaws**, TCP under 0.15 m/s, not crushing, 0.5 s |

**Safety in the reward:** virtual wall guarding the **pads** at 7 cm from the mouth (not the TCP
— an 18 cm TCP wall is incompatible with delivering food to a mouth); crush penalty above 12 N;
drop penalty; self-collision, curl and joint-limit penalties; speed cap only while *carrying and
within 20 cm of the mouth*.

---

## 6. Every bug found and fixed (the expensive knowledge)

Each of these cost real measurement. Do not reintroduce them.

1. **`qvel` clipped after `mj_step`.** The original env rewrote physics state to enforce its
   velocity cap, which corrupts dynamics and made every velocity assertion circular — the
   "safety test" measured a number the code had just overwritten. The cap now applies to the
   commanded delta only. Verified: 0.55 rad/s peak under random actions against a 1.0 cap, with
   physics untouched.
2. **Target teleported mid-episode** up to 5.2 cm against a 2 cm tolerance, making ~30% of
   episodes unwinnable. Removed.
3. **"Grasp" meant one frame of any contact with jaws shut.** Reported 97%; measured
   `success=24/25, lifted=0, displaced>15cm in 9/25` — it had learned to *swat*. Now requires
   both pads, settled object, sustained.
4. **Self-collision penalty fined the gripper for closing.** The two fingers are siblings under
   `link_6`, not parent/child, so the adjacency test missed them — and the pads meeting is
   exactly what a closed gripper is. Finger-on-finger contact is now excluded.
5. **Knock penalty charged every step.** One early nudge fined the policy for hundreds of steps,
   so never touching anything was optimal. Charged once now, and at 10 cm rather than the 4 cm
   that also defines success.
6. **The target followed a knocked object**, so the arm chased an apple 8.8 m out of the
   workspace. A lost object now ends the episode.
7. **Reach aimed at the object's centre**, which the object physically occupies — reachable only
   by shoving it, which (5) then punished. Reach now aims at a pre-grasp point above it.
8. **Per-step phase bonuses paid better than finishing.** `open_bonus` paid 0.10/step for held-open
   jaws: loitering 300 steps was worth 30, while `success_bonus` was 5. The policy parked at the
   object and collected rent. Phase 0 is now a *cost*, `success_bonus` is 30, and a
   `time_penalty` makes finishing always better. Reach went 0.30 → 0.96 and
   `ep_len_mean` 300 → 68.6 → 23.6 on this change alone.
9. **Settle gate below what the controller can hold.** 672/945 steps were already inside
   distance tolerance but 87% exceeded the 0.10 rad/s gate. Raised to 0.20 **and** given a
   `settle_bonus` gradient, because a bare threshold is not something gradient descent can find.
10. **Objects spawned overlapping / tilted / spinning** and popped on the first step. Now:
    angular slots, orientation and velocity zeroed, 40 settle steps. Drift after reset 1.11 mm.
11. **Rejection sampling silently failed.** Three objects needing 12 cm gaps do not fit in a
    ±0.55 rad arc, so all 60 attempts were rejected and the last bad sample used anyway — worst
    clearance 1.5 mm. Replaced with angular slots + a wider ±0.95 rad arc. Now 32 mm.
12. **Mug clearance vs perception error.** At max scale the mug was 74.8 mm against 82.8 mm pads
    — 4 mm per side, against 17.5 mm of perception error. `pad_clearance_m` 8 → 18 mm.
13. **Marker removed** — at 15 mm it is thinner than these pads can usefully grip.
14. **`UnboundLocalError`**: the settle bonus was written into the success block, which runs
    before `reward` is assigned. It fired only in the reach branch, so the trainer kept running
    and it surfaced in the viewer instead. Moved below.
15. **Viewer success counter was cumulative** across improving checkpoints — read 0/16 while the
    policy was hitting 12/12. Now a rolling window of 20.
16. **Chain stalled on a solved stage**: `run_resumable` printed "nothing to do" and exited
    without writing final weights, which the chain waits for. It now writes them.
17. **Chain wait pattern matched substrings** — `--run-dir runs/feed-reach` also matched
    `runs/feed-reach-b`, so the chain waited on unrelated runs. Anchored on a trailing space.
18. **Chain log parser** matched `"] starting "` but the log writes `[chain] 20:16:12 starting
    grasp`. Auto-stage silently never fired.

---

## 7. Operational gotchas that cost time

- **`pkill`/`pgrep -f` matching your own shell.** An SSH command containing the pattern kills
  itself. This happened four times. **Get PIDs in one call, kill them by number in another.**
- **Heredocs through `ssh '...'`.** Single quotes inside the remote script break the outer
  quoting. Write the script locally and `scp` it.
- **MuJoCo offscreen framebuffer is 640×480**, and camera resolution comes from `YAM_CAM_RES`
  *at scene-construction time*. A square render caps at **480**, and `YAM_CAM_RES` must be set
  **before** `arm.ik.scene` is imported.
- **EGL teardown noise.** `EGLError: <exception str() failed>` on exit is harmless renderer
  cleanup; the real exception is earlier in the log. Filter EGL lines when reading tracebacks.
- Another teammate has run **`claude --dangerously-skip-permissions`** on this box. Files can
  change under you.

---

## 8. Commands

```bash
# status
tail -3 runs/chain.log
grep -E "success_rate|ep_len_mean|total_timesteps" runs/feed-grasp.log | tail -3

# restart the whole curriculum (skips finished stages, resumes an interrupted one)
nohup bash arm/rl/scripts/chain_curriculum.sh > runs/chain.log 2>&1 &

# train one stage by hand
.venv-arm/bin/python arm/rl/scripts/run_resumable.py --env feed \
  --config arm/rl/configs/feed.yaml --stage grasp --run-dir runs/feed-grasp \
  --timesteps 3000000 --init-from runs/feed-reach

# look at what it is ACTUALLY doing
.venv-arm/bin/python arm/rl/scripts/filmstrip.py --run-dir runs/feed-grasp --stage grasp \
  --panels 6 --cam scene_cam --out /tmp/g.jpg     # or --cam wrist_cam

# browser viewer
nohup .venv-arm/bin/python arm/rl/scripts/stream_policy.py --run-dir runs/feed-grasp \
  --stage grasp --auto-stage --follow --port 8089 > runs/stream.log 2>&1 &
```

**Pause/resume is verified.** `kill <pid>` or Ctrl-C saves immediately (optimizer state and step
count included); re-running the identical command resumes exactly. Periodic checkpoints every
10k steps (~2.5 s of progress at risk in a hard power cut). Tested end to end: paused at 52,326
→ resumed → finished at 60,006.

Stage budgets live in `chain_curriculum.sh`:
`reach:1200000 grasp:3000000 lift:2000000 present:4000000`.

---

## 9. Hardware state — READ BEFORE TOUCHING ANYTHING

- **`can0` exists but is DOWN.** Bringing it up needs root:
  `sudo ip link set can0 up type can bitrate 1000000`. That only opens the interface; it does
  not energise motors.
- **The wrist camera works**: `1bcf:2d4f "USB CAMERA 4K"`, generic UVC, `/dev/video0`. It is a
  **wide-angle/fisheye** lens with strong barrel distortion — any future perception work needs
  distortion coefficients, not just a pinhole `K`. Opens at 640×480 by default.
- **The USB hub has dropped once already**, taking the camera and the CAN adapter with it.
- **The user's standing instruction: do not cut power to the arm** — the motors do not hold
  position unpowered.
- **The sim↔real joint map has never been verified.** `arm/deploy.py::RealArm` deliberately
  refuses to run: `"real backend intentionally disabled pending joint-map/FK verification"`.
  Do not remove that guard.
- **Watchdog conflict, unresolved:** I2RT documents a 400 ms timeout that drops to damping;
  local notes report this arm at `TIMEOUT=0`, meaning a crashed controller may not stop the
  motors. Documented in `ARM_NOTES.md`. **Do not change the register.**
- Torque headroom is ample: peak demand `[0.59 1.64 1.13 0.28 0.17 0.10] Nm` against MJCF
  forceranges `[28 28 28 10 10 10]`.
- Any real-arm motion should go through the previously hardware-validated
  `~/openyam/scripts/move_and_hold.py`, with a dry run first, and only with a human present and
  an E-stop in reach.

---

## 10. Honest assessment

The teammate's plan (`yam_agentic/GAME_PLAN.md`) says **no RL is required** for the demo: the
deterministic IK path is already validated at 50/50 reaches within 1 cm (worst error 0.44 mm)
and 20/20 grasps lifting 13+ cm. The user chose RL knowingly, twice, and that choice was
respected — but if the clock runs short, **the IK path is the lower-risk demo** and it is
already written.

Note also that RL does **not** avoid the perception problem. The policy consumes a 3D object
position in the robot's base frame, exactly as the IK path does. Whatever is built, somebody
still has to calibrate the wrist camera to the robot.

**Next concrete step:** filmstrip the grasp policy, identify which of the four failure modes in
§4 is occurring, fix that specifically, and re-run. Do not tune blind, and do not trust the
success counter.
