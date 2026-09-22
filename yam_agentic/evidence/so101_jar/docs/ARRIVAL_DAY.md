# Arrival day — sim-to-real runbook (SO-101, lerobot-sim2real recipe on the Mac stack)

Written 2026-09-17. All tools below were dry-run against the MuJoCo stand-in
(`--sim`) and the iPhone Continuity Camera feed; **nothing has touched the
physical arm yet.** Read `docs/REPLICATION_AUDIT.md` for what is verbatim vs
adapted, and `docs/AGENT_CONTEXT.md` for the state of the sim-side replication.

## 0. What the recipe needs (and does not need)

- **One fixed third-person camera. No wrist camera.** lerobot-sim2real trains
  and deploys from a single `base_camera` (128×128 RGB) plus joint state. The
  iPhone is that camera. `wrist_cam` in the MJCF is unused by this recipe and
  nothing has to be added to the sim scene: `base_cam` in
  `scene_box_rl.xml` is the camera the whole pipeline renders from, and its
  pose/FOV are overridden at train time from the alignment step below.
- **Exactly one photo type**: the deployment camera's own view of the EMPTY
  workspace (arm unmounted, cube removed), after the camera is fixed in its
  final position. Several frames of it are fine (`--n 5`). Not photos of the
  arm, not a scan, not a texture set.
- The camera must be mounted **before** the photo and **before** training,
  because its fitted pose/FOV are baked into the training run.
- **Physical**: SO-101 follower (motor IDs set per LeRobot docs), USB + PSU,
  clamped at a marked spot; iPhone on a tripod/mount, Continuity Camera with
  Center Stage / Desk View / Portrait / Studio Light OFF (Control Center →
  Video Effects), landscape and level; a ~2.5 cm cube (recipe: 2.2–2.8 cm,
  any saturated color) for the learned policy and a ~4 cm cube for the expert
  replay demo; the arm's print color as `ROBOT_RGB="r g b"` (recipe keeps the
  robot color FIXED at the real color; theirs is white).
- Frames: base origin = shoulder_pan axis at table level, **+x forward** (the
  cube arc), **+y the robot's left**, +z up. Sim joint zero = the "L" pose
  (`docs/media/sim_pose_zero_L.png`); rest pose = `[0,0,0,90°,90°]`, gripper
  pointing down (`docs/media/sim_pose_rest.png`).

## 1. Timeline (≈ 60–90 min to a real-arm demo with training launched)

All commands from `/Users/adipu/so101Sim/urlab_bridge` with `uv run --no-sync`.

### A. Arm bring-up (10–15 min)
```
uv run --no-sync lerobot-find-port                       # -> /dev/tty.usbmodemXXXX
uv run --no-sync python ../rl/real/so101_real.py --port /dev/tty.usbmodemXXXX --read
```
The first connect runs LeRobot's calibration in the terminal. At *"Move to
the middle of its range of motion"* hold the **L pose** (upper arm vertical,
forearm horizontal pointing forward, wrist straight, gripper horizontal),
then sweep every joint except wrist_roll through its full range and press
Enter. Calibration lands in `~/.cache/huggingface/lerobot/calibration/robots/so101_follower/so101_follower.json`.
```
uv run --no-sync python ../rl/real/so101_real.py --port ... --zero-here   # torque off, hold L pose -> joint offsets
uv run --no-sync python ../rl/real/so101_real.py --port ... --jog         # +15 deg per joint, answer y/n -> signs; gripper direction
uv run --no-sync python ../rl/real/so101_real.py --port ... --goto rest   # must end pointing straight down
```
Result: `rl/real/joint_map.json`. Every later script uses it.

**wrist_roll needs one extra step (learned the hard way on 2026-09-17).** The
roll servo is calibrated as a full turn, but the gripper cable blocks a ~27°
arc, and LeRobot's degree window is ±180 around the servo's homing. If the
sim zero is not at the physical center of the roll's travel, the sim rest
roll (+90°) can land in the blocked arc: the servo stalls, reports an
Overload error, then drops off the bus until the supply is power-cycled.
Procedure: `--roll-scan 60` (wrist_roll torque-free; turn the gripper to
both stops; the wrap-aware summary prints the blocked arc and the center),
then re-home the servo so that center reads 0 (homing_offset += center in
counts; Feetech: Present = Actual − Homing; `write_calibration` + save) and
set the roll map offset to ~0. Sim's L-pose roll = jaws opening VERTICALLY,
camera mount on the robot's LEFT (`docs/media/wrist_roll_direction.png`).
Also: `RealSO101` keeps torque ON after a script exits (LeRobot's default
releases the arm, which sags onto the table); use `--torque-off` to release.

### B. Camera placement + alignment (15 min)
Mount the phone roughly where the sim camera is: 45–55 cm from the base,
30–35 cm above the table, 30–40 cm to one side, tilted down at the point
~20–25 cm in front of the base (recipe: pos `(0.5, 0.3, 0.35)` looking at
`(0.3, 0, 0.1)`, FOV 52°; our `base_cam` default `(0.42, -0.38, 0.30)`).
```
uv run --no-sync python ../rl/real/camera_alignment.py --cam 1 --port /dev/tty.usbmodemXXXX
```
Torque goes off so you can pose the arm by hand; the sim arm mirrors it in
the overlay. Nudge (w/s/a/d/r/f camera, i/k/j/l/u/o look-at target, `[`/`]`
FOV) until the sim silhouette sits on the real arm, then `q`. It prints the
line to train with and saves `rl/real/camera_calib.json`:
```
CAM_POS="x y z" CAM_QUAT="w x y z" CAM_FOVY=NN
```
Prefer moving the phone to match the sim over dragging the sim far from its
default. Check that a cube anywhere on the spawn arc (15–26 cm ahead, ±40°)
is visible and not hidden by the arm at rest. Camera index: `1` was the
iPhone and `0` the MacBook camera on 2026-09-17; verify with step C's `full/` image.

### C. Background photo (10 min)
Unmount the arm (the recipe does; the base must not be in the photo), remove
the cube, keep the lighting you will deploy under:
```
uv run --no-sync python ../rl/real/capture_background.py --cam 1 --n 5
```
→ `rl/real/bg/bg_00x.png` (128×128, same crop+resize as the deploy
observation) and `rl/real/bg/full/` for your eyes. Remount the arm at the
mark and confirm nothing moved:
```
uv run --no-sync python ../rl/real/camera_alignment.py --cam 1 --port ... --once /tmp/check.png
```

### D. The demo: open-loop expert replay (15 min)
Put the ~4 cm cube **20 cm straight ahead** of the shoulder_pan axis, faces
square to the arm, then:
```
uv run --no-sync python ../rl/real/replay_expert_real.py --port /dev/tty.usbmodemXXXX --cube 0.20 0.0 --confirm
```
The scripted expert (95 % in sim) plans on the sim cube; its joint commands
stream to the real arm at 25 Hz: L pose → approach → descend → close → lift.
Sim-to-sim dry run: lifts at 25 and 50 Hz for `(0.20, 0)`; an off-centre
`(0.24, 0.10)` spot was marginal even sim-to-sim, so keep it centred.
For a 2.5 cm cube add `--cube-half 0.0125` (expert slower, ~60 %).
If it misses: re-do `--zero-here`, check `--jog` signs, measure the cube
position again, try `--hz 50`.

This validates calibration, signs, reach and gripper convention. It is not a
learned policy.

### E. Launch the training run (5 min, then 3–4 h unattended)
```
cd /Users/adipu/so101Sim/urlab_bridge
KMP_DUPLICATE_LIB_OK=TRUE REWARD_MODE=maniskill ACTION_MODE=delta EPISODE_STEPS=64 CTRL_DT=0.05 \
N_ENVS=512 BG_DIR=/Users/adipu/so101Sim/rl/real/bg ROBOT_RGB="1.0 0.9 0.17" \
CAM_POS="0.3800 0.3100 0.3000" CAM_QUAT="0.285798 0.188048 0.516496 0.784977" CAM_FOVY=50.00 \
OUT_DIR=../rl/runs/lift_pixel_rl_real1 \
nohup uv run --no-sync python ../rl/native/train_rl_pixels_batch.py 40000000 \
  > ../rl/runs/lift_pixel_rl_real1.log 2>&1 &
tail -f ../rl/runs/lift_pixel_rl_real1/progress.txt
```
Values above are the 2026-09-18 calibration (phone on the robot's LEFT, 38 cm ahead,
31 cm left, 30 cm up, fovy 50; yellow arm measured 0.77/0.64/0.12 from the phone frame;
`ROBOT_RGB` is pre-compensated for the renderer's shading, `AMBIENT_RANGE` defaults to
0.45–0.85 in maniskill mode). `CTRL_DT=0.05` is new: ManiSkill's SO100GraspCube runs control at 20 Hz
(`SimConfig(sim_freq=100, control_freq=20)`); every earlier run used 50 Hz,
so 64-step episodes lasted 1.28 s instead of 3.2 s. Env-only throughput at
N=512: 7.9k steps/s at 50 Hz → 5.8k at 20 Hz; expect ~2.5–3k end-to-end, i.e.
25 M steps in ~2.5 h, 40 M in ~4 h. Watch the `success=` column: v5/v6 never
left 0.00 in 10 M steps. If it is still 0.00 at 20 M, the remaining suspects
are listed in `AGENT_CONTEXT.md` (gripper zero convention, PhysX-vs-MuJoCo
contacts). Probe checkpoints for grasp ticks; reward alone cannot tell hover
from grasp.

### F. Deploy (after training)
```
KMP_DUPLICATE_LIB_OK=TRUE uv run --no-sync python ../rl/real/deploy_pixel_ppo.py \
  ../rl/runs/lift_pixel_rl_real1/ckpt_XXM.zip --port /dev/tty.usbmodemXXXX --cam 1 --hz 20 --confirm
```
`--hz` must equal `1/CTRL_DT`. `--confirm` = Enter before every command
(the recipe's `--no-continuous-eval`); drop it once the motion looks sane.
`--save-frames DIR` dumps the 128×128 observations to check the crop.
Place the 2.5 cm cube inside the trained spawn arc.

## 2. What was verified on 2026-09-17 (no arm present)

| check | result |
|---|---|
| look-at → MuJoCo quaternion reproduces the MJCF `base_cam` quat | exact (dot = 1.000) |
| `CAM_FOVY` changes the render (sensorsize intrinsics zeroed) | yes, 77° vs 52° differ |
| alignment overlay `--once` on the iPhone feed | works (`camera_alignment.py`) |
| background capture on the iPhone (index 1, 1080p, ~30 fps) | 128² PNGs + full crops |
| `so101_real.py --sim --jog / --goto rest` | passes on the stand-in |
| expert replay `--sim` at 25 / 50 Hz, cube (0.20, 0) | stand-in lifts |
| `deploy_pixel_ppo.py --sim --frame` with the v6 checkpoint | loop runs, SB3 auto-transposes HWC→CHW |
| batch env with `CTRL_DT=0.05 CAM_FOVY=52 BG_DIR=<phone photos>` | 10 substeps, photo-composited tiles (`docs/media/arrival_day_composite_smoke.png`) |

## 3. Known gaps and risks

- Continuity Camera latency MEASURED 2026-09-17 with rl/real/cam_latency.py
  (glass-to-glass, 12 flips each): wired median 99 ms (78–118), wireless
  median 84 ms (81–152, more jitter). The wire buys consistency, not speed;
  ~80 ms is the Continuity pipeline floor = ~2 control steps at 20 Hz. The
  recipe's authors used a RealSense at 640×480 (~1 step). Continuity has no
  exposure/white-balance lock; Camo Studio over USB does (untested here).
  Keep lighting identical between the background photo and deployment.
- Nothing here has run on the physical arm; `RealSO101` wraps LeRobot 0.6.1's
  `SO101Follower` with `max_relative_target=20°` as the safety clamp.
- The sim-side replication had 0 % success as of 2026-08-16. The control-rate
  delta found today is the first unaudited component fixed since then; it is
  not a guarantee.
- The 2-camera ACT champion (72 %) was trained without domain randomization
  or compositing on the sim renderer's own images and is not a deployment candidate.

## 4. Jar track (added 2026-09-18 ~02:45): what changed overnight and why

The object on hand is a 5.1 cm x 10.2 cm spice jar, not the recipe's 2.5 cm cube. Findings, all measured:

- **The jar can only be grasped from the side, 27-42 cm from the base.** Straight-down is impossible
  (wrist_flex tops out at 95 deg, so a downward-pointing hand cannot get above ~9 cm; the jar is 10.2 cm).
  Slanted 45-75 deg pinches tip a free-standing cylinder. Level/15/30 deg side grasps work:
  `rl/native/scripted_expert_side.py` (planned joint-space trajectory, collision-checked against jar AND
  table, wrist roll -90 so the camera-mount bracket points up) lifts it in 29/30 positions over
  r 0.27-0.42 m, theta +-0.6 rad, confirmed by an independent stand-in arm. Filmstrips:
  `docs/media/side_expert_cases.png`.
- **The phone as placed on 2026-09-17 cannot see that zone.** Its square view only contains the jar out to
  ~28 cm. A state-RL teacher with full state and smooth delta actions (`rl/jar/`) scored 0% in the visible
  zone after 4.6M steps, so the pixel-PPO jar run there was stopped at 2.9M steps (0%). The cube PPO run
  (`lift_pixel_rl_real1`) was stopped at 2.4M steps when the jar became the target.
- **Morning setup for the jar:** move the phone to roughly 65 cm forward / 40 cm left / 30 cm up, aimed
  25 cm in front of the base (`rl/real/camera_calib_side_recommended.json` is only a suggestion). Then the
  usual: `camera_alignment.py` (fits the sim to wherever the phone really is), unmount, `capture_background.py`,
  remount. Then one command retrains against the real calibration:
  `rl/jar/run_chain.sh jar_act_real1 /Users/adipu/so101Sim/rl/real/camera_calib.json /Users/adipu/so101Sim/rl/real/bg`
- **Guaranteed demo, no learning, any camera:** jar on a tape mark 32 cm straight ahead, then
  `replay_expert_real.py --port ... --side --jar 5.1 10.2 --cube 0.32 0.0 --confirm`.
- **Learned policy (expert teaches the network):** `rl/jar/run_chain.sh` = side-grasp expert demos through
  the compositing pipeline (real photos mixed 50/50 with 136 unrelated backgrounds, +-4 cm camera jitter,
  so it is not tied to one room) -> LeRobot ACT -> closed-loop sim eval with filmstrips. Rehearsal run
  `rl/runs/jar_act_rehearsal/results.txt` uses the suggested camera pose.
- New env knobs this session: CTRL_DT, CAM_FOVY, ROBOT_RGB, BG_PHOTO_P, AMBIENT_RANGE, OBJECT; per-env
  lift threshold fix; collision-bound refresh when reshaping objects at runtime.

## 5. MORNING CHECKLIST for the jar (final version, 2026-09-18 ~05:30)

All from `/Users/adipu/so101Sim/urlab_bridge`, after
`export SO101_ID=adipu_follower_arm SO101_PORT=/dev/tty.usbmodemXXXX KMP_DUPLICATE_LIB_OK=TRUE`.

**What can be tested, best first**

| route | what it is | sim result | needs |
|---|---|---|---|
| 2 | camera finds the jar, IK side-grasp expert picks it up | jar located to 0.4 cm after auto-calibration; 23/24 grasps | any phone pose that sees the jar zone |
| 1 | learned pixels-to-joints ACT policy (distilled from the expert) | `jar_act_v3/act/step_15000`: 83-86% on 96 fresh episodes WITH camera lag, shadows, soft servos; 83% on wallpaper-pool backgrounds (in training), 81% on truly held-out photos (measured 2026-09-19, see section 6 correction) | phone within ~4 cm / 4 deg of pose B |
| 0 | no camera: open-loop expert replay, jar on a tape mark | 29/30 positions | nothing |

1. **Arm** on its tape mark, powered. `uv run --no-sync python ../rl/real/so101_real.py --read` prints joint angles.
2. **Phone, pose B** (see `docs/media/phone_view_previews.png`, bottom row): lens 55 cm forward of the base,
   20 cm to the robot's left, 45 cm above the table, tilted about 50 deg down at the spot 34 cm in front of the base.
   It must stay outside the arm's reach (it is 59 cm from the base). Landscape, Continuity Camera effects off, cable in.
   (Pose A, the easy low one: 65 / 40 / 30 cm aimed 25 cm ahead; use it with the `jar_act_v2b` checkpoint if it finished,
   see section 6.)
3. **Calibrate the camera automatically** (arm visits 5 poses above the table, ~1 min, keep the table empty):
   `uv run --no-sync python ../rl/real/auto_calibrate.py --cam 1 --init ../rl/real/camera_calib_side_v3.json`
   Check `rl/real/autocal/real_pose*.png` (red outline should hug the arm; IoU above ~0.7 is good). It prints how far the
   phone is from pose A and pose B. For route 1, nudge the phone and re-run until it says WITHIN tolerance for pose B
   (sim: 84% within 2 cm, 66% at 7 cm, 46% at 10 cm). Route 2 does not care.
   Fallback if the colour segmentation misbehaves: manual overlay,
   `uv run --no-sync python ../rl/real/camera_alignment.py --cam 1 --port $SO101_PORT --calib ../rl/real/camera_calib_side_v3.json --jar 0.36 0`.
4. **Jar zone**: 31-41 cm in front of the base, within +-20 deg of straight ahead (orange outline in the preview).
   Mark it with tape. The jar stands upright, lid on.
5. **Route 2:** `uv run --no-sync python ../rl/real/grasp_jar_from_camera.py --cam 1 --calib ../rl/real/camera_calib.json`
   L pose, EMPTY shot, JAR shot, prints the detected position, writes `rl/real/last_jar_detection.png` (the red outline
   must sit on the jar), asks before moving, then side approach, squeeze, lift, hold, put back, retreat.
6. **Route 1:** `uv run --no-sync python ../rl/real/deploy_act.py ../rl/runs/jar_act_v3/act/step_15000 --cam 1 --save-frames /tmp/act_frames`
   Moves to the L pose, waits for ENTER, then runs 14 s at 25 Hz from pixels + joint angles only. Ctrl-C stops it (the arm
   holds its last command). Compare `/tmp/act_frames/*.png` with `rl/runs/jar_act_v3/act/step_15000/filmstrip_*.png`:
   the real frames should look like the training frames (same framing, jar and arm at similar sizes).
   If the phone cannot be brought within tolerance, retrain against the real calibration (about an hour):
   `JAR_ZONE="0.31 0.41 -0.35 0.35" KP_SCALE="0.1 1.0" OBS_LATENCY="1 3" JAR_SHADOW=1 BATCH=16 CKPT_EVERY=5000 NA=25 ../rl/jar/run_chain.sh jar_act_real1 /Users/adipu/so101Sim/rl/real/camera_calib.json /Users/adipu/so101Sim/rl/real/bg 500 20000`
7. **Route 0:** jar on a mark 36 cm straight ahead:
   `uv run --no-sync python ../rl/real/replay_expert_real.py --side --jar 5.1 10.2 --cube 0.36 0.0 --confirm`

**Known sim-vs-real gaps to watch for:** the real arm sags 1-2 cm at full reach (grasp height was raised to 58% of the
jar); the real gripper's 0-100% scale only approximates the sim's angle (route 2 squeezes 0.3 rad extra; LeRobot caps
gripper torque at 50%); the sim arm has no cables and the sim jar is an opaque cylinder with a dark lid and a label band
while yours is translucent; nothing here has run on the physical arm yet. After any run, `so101_real.py --goto zero`
parks the arm in the L pose and `--torque-off` releases it (hold it, it will sag).

## 6. Overnight log (2026-09-18)

- 01:41 ACT rehearsal `rl/runs/jar_act_rehearsal` (332 demos, 15k steps, batch 8): best checkpoint step_12000 =
  31-35% closed-loop in sim (48 episodes; 35% kitchen backgrounds, 19% wallpaper-pool backgrounds [in training, NOT held out - see section 6 correction]). 28 of 31 failures
  are the policy KNOCKING THE JAR OVER: right motion, 1-2 cm off, jaws have ~1 cm clearance. Under-trained
  (<2 epochs, loss still falling) and checkpoints swing 10-31%.
- The sim jar was unrealistically tip-prone (friction 1.0; MuJoCo takes the max of two geoms unless one has
  priority). With jar priority + friction U(0.35, 0.7) the SAME checkpoint scores 42%; expert unchanged (98%).
- CORRECTION: the far-zone state-RL teacher (`rl/runs/jar_teacher_far`, the control) also scored 0% after 8M
  steps in the zone where side grasps are known feasible. So tonight's RL reward setup cannot learn this task at
  all, and the near-zone RL 0% is NOT evidence about feasibility. The near-zone conclusion rests on the geometry
  alone (wrist_flex cap -> no top-down on a 10.2 cm jar; slanted pinches tip it).
- Expert plans are robust to soft servos: 49/51 with position gain scaled down to 0.08x.
- 02:05 launched `rl/runs/jar_act_v2`: 600 demos, zone r 0.30-0.42 / +-0.5 rad, friction U(0.35,0.7),
  servo gain U(0.1,1.0), grasp height 58%, batch 16, 40k steps, checkpoints every 5k, all evaluated (na=25),
  best checkpoint re-evaluated on 96 fresh seeds for kitchen-only and wallpaper-pool-only backgrounds [in training, NOT held out - see section 6 correction].
- 02:05-03:22 `jar_act_v2` result: best checkpoint step_15000 = 67% (48 eps, mixed backgrounds), **72% on 96 fresh
  kitchen-background seeds**, 55% on wallpaper-pool backgrounds [in training, NOT held out]; checkpoints swing 44-67%. 24 of 27 failures are still
  "tipped the jar". Measured task forgiveness (plan from a wrong jar position, execute on the true one): 88% at 0-1 cm
  error, 79% at 2 cm, 50% at 3 cm. More fixed-jaw clearance makes it WORSE (the closing jaw shoves the jar farther and
  tips it); slower jaw closing changes nothing. So the lever is policy precision, not the grasp.
- `rl/real/auto_calibrate.py` added (arm visits 5 poses, silhouette fit of the 7 camera parameters): sim self-test
  locates the jar to 0.3-0.4 cm median afterwards (5-15 cm uncalibrated); yellow threshold verified on a real frame.
- 03:55 DAgger round 1 launched (`rl/jar/run_dagger_round.sh`, results appended to `rl/runs/jar_act_v2/results.txt`):
  300 on-policy episodes labelled with the expert plan's continuation, progress matched on the expert's ACHIEVED
  poses (keeps the command lead; matching on commands would recreate the stall trap), retrain from scratch on
  480 demos + DAgger data, 20k steps.
- 04:15 DAgger round 1: NO gain (62% vs 72% on the same 96 kitchen seeds). Dropped.
- 04:25 error-direction diagnostic on jar_act_v2/step_15000: when it succeeds the gripper is within 1.3 cm (along the
  camera's line of sight) / 0.6 cm (across) of where the expert would be; failures are ~2 cm off; 14 of 91 episodes tip
  the jar before the jaws even start closing. A pixel-precision problem, slightly worse in depth (camera only ~20 deg
  above the table).
- 04:35 the same checkpoint under REAL camera conditions (frames 1-3 ticks late + jar shadows): 72% -> 60%.
- 04:40 launched `rl/runs/jar_act_v3`: phone pose B (55 cm fwd / 20 cm left / 45 cm high, ~50 deg down; preview
  `docs/media/phone_view_previews.png`), smaller jar zone r 0.31-0.41 m / +-20 deg, camera latency 1-3 ticks, jar
  shadows, soft servos, ResNet stride 16 (16x16 feature grid), 500 demos, 15k steps.

- 05:10 **`jar_act_v3` result (pose B, smaller zone, trained with camera lag 1-3 ticks + jar shadows + soft servos):**
  step_15000 = 85% (48 mixed), **86% / 83% on 96 fresh kitchen-background seeds (two seed sets), 83% on wallpaper-pool [in training, NOT held out]
  backgrounds**, every checkpoint 77-85%. Margins (96 seeds): phone placement +-2 cm 84%, +-7 cm 66%, +-10 cm 46%;
  camera lag 120-200 ms 82%; execution horizon 25 or 50 both fine, 12 worse (65%). This is the checkpoint to test.
- The jar finder also works from pose B (1.0 cm median error without auto-calibration, 23/24 grasps).
- `ACT_DILATION` is a dead end: torchvision ResNet18 (BasicBlock) does not support dilation.
- 05:12 launched `jar_act_v2b`: same realistic training for the easy low pose A (results in its results.txt).
- 05:30 safety checks in sim: the auto-calibrator's five poses AND the joint-space paths between them keep >= 2 cm table
  clearance with no self-collision; during calibration and side grasps no arm part (bounding sphere) comes closer than
  38 cm to the phone at pose A or 28 cm at pose B, and both phone poses are outside the arm's reach envelope.
- 06:15 `jar_act_v2b` (easy low pose A, identical realistic training, same small zone): 52-65% across checkpoints, 58-64% on
  the same 96 kitchen seeds where pose B scores 86%. **The camera viewpoint is the decisive factor; use pose B.** Pose-A
  fallback checkpoint: `rl/runs/jar_act_v2b/act/step_20000` (64%).
  (Its chain's evals first crashed on a bug in my shadow drawing, a jar knocked toward the lens projected an absurd
  shadow width; fixed in jar_scene.py, v3's 86% re-verified after the fix.)
- 06:20 launched `jar_act_v3c`: pose B with WIDER phone-placement randomisation (+-6.5 cm instead of +-4), 600 demos, 20k
  steps, then a placement-tolerance comparison against v3 on the same seeds (appended to its results.txt). If it holds
  ~85% at +-4 cm and beats v3's 66% at +-7 cm, prefer it for the first real test.
- 07:20 `jar_act_v3c` (pose B, phone-placement randomisation widened to +-6.5 cm): NOT better. Same 96 seeds, kitchen
  backgrounds, lag + shadows: v3c step_5000 = 73% / 64% / 45% at +-4 / 7 / 10 cm placement error versus v3 step_15000 =
  83% / 66% / 44%. Wider randomisation cost accuracy and bought no tolerance. **Final recommendation unchanged:
  `rl/runs/jar_act_v3/act/step_15000`, phone at pose B within ~4 cm (use auto_calibrate's readout).**
  (v3c's own chain evals ran at +-6.5 cm because CAM_JITTER was exported; its 74% there is not comparable to v3's 86%.)
- Overnight compute ended ~07:20. Nothing is running.

### Scoreboard (closed-loop sim success, 96 fresh episodes, kitchen backgrounds, camera lag 1-3 ticks + shadows + soft servos)

| policy | phone pose | zone | result |
|---|---|---|---|
| jar_act_v3 / step_15000 | B (high, ~50 deg down) | r 0.31-0.41 m, +-20 deg | **83-86%** (83% on wallpaper-pool backgrounds, which are in training; **81% on truly held-out photos**, 2026-09-19) |
| jar_act_v3c / step_5000 (wider placement DR) | B | same | 73% |
| jar_act_v2b / step_20000 | A (low, ~21 deg down) | same | 64% |
| jar_act_v2 / step_15000 (trained without lag/shadows) | A | r 0.30-0.42 m, +-29 deg | 60% (72% without lag) |
| jar_act_v2 + DAgger round 1 | A | same | 62% without lag (no gain) |
| jar_act_rehearsal / step_12000 | A | r 0.28-0.42 m, +-34 deg | 35-42% without lag |
| pixel PPO (cube, jar) | - | - | 0%, stopped |

### Correction (2026-09-19): "never-seen backgrounds" was not what it said

Every "never-seen" / "unseen" background figure above originally came from `--p-real 0.0`, which draws from
`rl/jar/bg_pool` - the same 136 wallpapers that make up half of every training set. There was no held-out split;
the chain's own results file called it "never-the-kitchen", and the write-up turned that into "never-seen". The
lines above are relabelled. The gaps the weaker policies showed (rehearsal 35% vs 19%, v2 72% vs 55%) are therefore
"busy wallpapers are harder than the plain kitchen photo", not a novelty effect.

The missing measurement was run on 2026-09-19 with a real held-out set (30 natural photos the policy never trained
on, plus the office-desk photo, which `jar_act_v3` never saw), same seeds (9,500,000+), same conditions (lag 1-3
ticks, shadows, soft servos), controls first:

| backgrounds | recorded | re-run 2026-09-19 (96 eps) |
|---|---|---|
| kitchen photos (in training) | 86% | 84% |
| wallpaper pool (in training) | 83% | 83% |
| held-out photos (never in training) | - | **81%** |

A SECOND draw on the same 96 scenes (the standing 30-photo set, i.e. without the office-desk photo, which reshuffles
which photo each episode gets) scored **70%** (67/96). So one run was not enough to claim "no gap": the two draws
give 81% and 70% (pooled 76%) against 84% in training - a held-out gap of several points that depends on which
photos come up. A larger run is pinning it down; see `rl/jar/bg_heldout/README.md` for the current baseline. The
standing held-out set is `rl/jar/bg_heldout` (30 photos; README there), `gen_demos.py` refuses to train on it, and
`run_chain_fast.sh` now evaluates on it at the end of every chain. What this does
NOT show is robustness to real camera imaging - see the camera stress test below.


## 7. New location (office desk) and new object — 2026-09-18 midday

**What changed physically.** Bamboo desk, blue fabric partitions, arm clamped to the desk's right edge ~30 cm
from the partition (first placement was ~6 cm from it: moved). Object is no longer the spice jar: white
3D-printed post, cylinder 25.4 mm dia × 50.8 mm, on a 50.8 × 38.1 × 3 mm plate, 5 mm bolt on top.
The real arm has **no wrist-camera bracket**; the MJCF does → `rl/real/arm_variant.py` hides it
(render group 3, collisions off; `ARM_CAMERA_MOUNT=1` keeps it).

**What is stale from the kitchen night.** `camera_calib.json` (kitchen copy = `camera_calib.bak1789760065.json`),
`rl/real/bg/`, `rl/runs/jar_act_v3` (trained for the jar, kitchen photos, pose B). All must be redone per setup;
that is the recipe working as designed (one background photo + one camera pose per deployment).

**What changed in the tooling.**
| piece | change | why |
|---|---|---|
| `auto_calibrate.py` arm segmentation | motion vs per-pixel median frame + LEARNED arm hue | fixed yellow HSV threshold hit 57 % of the bamboo desk |
| `auto_calibrate.py` start | global coarse search: grid camera POSITION on a hemisphere, solve aim + zoom from image moments, refine best 4 | phone pose unknown at a new desk; sim test start 33–76 cm off → object localisation 0.4–1.6 cm |
| calibration POSES | pan ≤ 0 only, lowest moving link ≥ 4.8 cm | partition on the robot's right at the first placement; harmless to keep |
| `make_background.py` | background = masked median of the calibration frames + inpaint of never-seen pixels | no need to unmount the arm for the empty-scene photo |
| `rl/jar/object_spec.py` + `rl/real/object.json` | one parametric object spec → generated MJCF, finder, expert, scene | object changed; hard-coded 5.1 × 10.2 cm jar was in 5 files |

**Calibration result (run 2, whole arm in frame).** camera (0.661, 0.274, 0.521) m, look-down 39°, fovy 52.2°,
roll −0.8°, silhouette IoU 0.605; outlines track every link and both jaws in all 5 poses
(`rl/real/autocal/real_pose*.png`). Saved as `camera_calib.json` and `camera_calib_desk.json`.
Run 1 had the top half of the arm out of frame → fovy 40.5°, IoU 0.558: **the arm must be fully inside the
centre square in the L pose** (the policy only sees the centre square of the 16:9 frame).
Background: `rl/real/bg_desk/full/bg_00.jpg`.

## 8. SmolVLA (VLA) trial — 2026-09-18 afternoon

**Why.** The setup changes between sessions (desk, object, phone pose), and every change costs a full
retrain because ACT is tied to the camera pose it was trained on (measured margins: 84 % at 2 cm phone
offset, 66 % at 7 cm, 46 % at 10 cm). The hypothesis is that a pretrained VLA has the capacity to absorb
much wider camera randomisation, so one checkpoint would survive a moved phone. Widening randomisation
did NOT help ACT (`jar_act_v3c`, 2026-09-18 07:20).

**What was built** (all deliberately mirroring the ACT path so the numbers are comparable — same npz
demo shards, same chunk 50 @ 25 Hz, same held-out seeds 9,000,000+, same success test):
| file | role |
|---|---|
| `rl/jar/smolvla_io.py` | the one place that does what LeRobot 0.6's processor pipelines do: task tokenisation, MEAN_STD state/action normalisation (`norm_stats.json` saved with each checkpoint), and a `Runner` with ACT's `reset`/`select_action` shape |
| `rl/jar/train_smolvla_jar.py` | finetunes `lerobot/smolvla_base` on the demo shards |
| `rl/jar/eval_smolvla_jar.py` | closed-loop eval, parallel CPU workers (`--device mps --workers 1` for GPU) |

Needed `uv pip install transformers accelerate num2words` (torch untouched at 2.11.0).
Three bugs on the way in: the language keys are `observation.language.tokens` /
`observation.language.attention_mask`, the attention mask must be **bool** (`smolvlm_with_expert` uses
`torch.where`), and normalisation/tokenisation are NOT inside `policy.forward` any more.

**Measured cost on the M4 Max.**
| | SmolVLA | ACT |
|---|---|---|
| params | 450M (100M trainable: VLM + vision encoder frozen) | ~50M |
| train throughput (MPS) | 15 frames/s (batch 8; batch 16/32 are *worse*, the CPU-side gather is the bottleneck) | ~100 frames/s (batch 16) |
| time for ACT's 240k-frame budget | ~4.4 h | 40 min |
| inference, 50-action chunk | 220 ms MPS / 796 ms CPU | ~40 ms |
| RAM | 15 GB of demos + model; 30 GB of demos + 14 demo workers thrashes swap | same demo cost |

Deployment implication: SmolVLA needs `--na 50` (one inference per 2 s) and the GPU; at `--na 25` the
220 ms stall lands once a second inside a 40 ms control tick.
