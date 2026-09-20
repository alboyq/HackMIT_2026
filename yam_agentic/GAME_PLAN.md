# Game plan — apple to mouth (sim-first)

**Rev. 19 Sep 2026, late evening.** This replaces the earlier plan in this file, which was built
around run-time 3D reconstruction + IK. That path is now the fallback (§7). The live version of
this document, updated as results land, is <https://claude.ai/artifact/ECpTKyBYz7adiNBwzKVGuM>.

Everything this plan cites is in this repository:

| What | Where |
|---|---|
| YAM simulation, IK, pick-and-deliver planner, tests | [`sim/`](sim/) (runs from this repo) |
| Measurements on the YAM model | [`MEASUREMENTS.md`](MEASUREMENTS.md) |
| Evidence from the earlier SO-101 sim-to-real work ("so101Sim") — code, result files, scoreboard | [`evidence/so101_jar/`](evidence/so101_jar/) (reference copies; see its README) |
| Gaze tracker | [`../gaze3d/`](../gaze3d/) |
| Arm stack, safety rules, pixel-to-3D maths, team UDP contract | branch [`arm-ik-rl`](https://github.com/alboyq/HackMIT_2026/tree/arm-ik-rl): `arm/`, `ARM_NOTES.md`, `REPO_NOTES.md` |
| A teammate's ACT runbook (detector-box-conditioned policy) | **not in the repo yet** — it lives on the GX10 at `~/hackmit-local-backup/training/`. Worth committing. |

## 1. The task

Look at an object on a screen; the arm picks it up and brings it toward your mouth, stopping
short. Selection is by gaze (`gaze3d`, 1.3° mean error) with EEG as a confirm channel.

## 2. Direction: train the geometry in

Skip run-time 3D reconstruction and IK. What replaces them is not an off-the-shelf VLA but a
policy trained in simulation on this arm, this task and this user, distilled from the planner in
`sim/yam_expert.py`, which already picks all four objects and carries them to a staging point
15 cm from the mouth.

> **Update 2026-09-20:** the VLA question was re-examined with measurements, not just a
> literature read — see [`sim_first/VLA_VERDICT.md`](sim_first/VLA_VERDICT.md). Two things below
> have changed: **X-VLA** (0.9 B, in the `lerobot 0.6.1` already installed here) adapts to a novel
> single-arm embodiment by design, which removes the "no single-arm YAM checkpoint" objection; and
> the sim-first ACT track this plan rests on has a measured bug — the policy scores worse than
> not moving the arm. Fix that before betting either way.

**Why not download a VLA:**

- No single-arm YAM checkpoint exists. ABC, MolmoAct 2 and GR00T N1.7 are all bimanual YAM with
  three cameras.
- Even the YAM-native model is weak zero-shot: ABC (3,553 hours of YAM teleoperation) scores
  32.8 % strict success out of the box on its own platform.
- π0.5-DROID managed 39.2 % in Ai2's independent test, on the robot it was trained for.
- Nobody's training data contains "bring it to a person's mouth", so a VLA means fine-tuning on
  real demonstrations, and there is no leader arm to collect them.

**Architecture families, September 2026:**

| Family | Best example | Headline | Fit |
|---|---|---|---|
| Modular: detect → 3D → plan | OK-Robot, γ | 58–82 %, no training | The fallback (§7) |
| VLA from real teleoperation | π0.5, GR00T N1.7, ABC | 33–39 % zero-shot | Wrong arm configuration, no demos |
| Hierarchical agent over a VLA | Gemini Robotics 1.5, MPVI | Better on long tasks | Needs a working VLA underneath |
| World-action models | DreamZero (14 B) | 2× VLA generalisation; new robot from 30 min of play | Too heavy for one GB10 tonight |
| Wrist-camera cross-embodiment | RDT2 (7 B) | Zero-shot on unseen robots | Appears to need a UMI-style gripper |
| **Sim-first** | **MolmoBot** (Ai2, Mar 2026), SplatSim | **79.2 % zero-shot real** vs π0.5 at 39.2 % | **This plan** |

MolmoBot: planner-generated demonstrations in simulation, aggressive randomisation of cameras,
objects and lighting, zero real data — and it doubled the best real-data VLA. It is the recipe
behind `evidence/so101_jar/` at a thousand times the scale.

## 3. The system

```
scene camera frame ──► 2D only: box of the gaze-selected object + mouth pixel + face size
                                        │
wrist RGB + scene RGB + 7 joints + [prompts] ──► ACT policy ──► joint targets, 30 Hz ──► limiter ──► arm
                                        ▲
             trained 100 % in simulation from sim/yam_expert.py (pick → carry → stop 15 cm short)
```

Everything the policy is told arrives as pixels or 2D image coordinates. No camera matrix, no
robot-to-camera transform, no marker board on the critical path: the mapping from image positions
to motion is learned. The prompt vector extends the teammate's design (their runbook conditions
ACT on the detector box as six numbers); the mouth is added the same way, with validity bits so a
lost face means "hold a neutral carry pose".

## 4. The person in the simulation

The simulated user is a beige ball on a blue box: a 9.5 cm sphere, a capsule nose, a box torso, a
box hand, and three markers (`mouth`, `face`, `hand`). See
[`media/yam_user_model.png`](media/yam_user_model.png). There is no face, and none is needed:

1. **The mouth enters as coordinates, not pixels.** Real: MediaPipe's landmarker. Sim: the
   `mouth` marker projected through the simulated camera. This is how assisted-feeding systems do
   it (FEAST, Feel the Bite): a landmark module on real images, no learned policy looking at faces.
2. **Face size rides along for range.** One pixel is a ray. Apparent face size gives distance;
   heads vary ±10 %, about 5 cm here — inside the 15 cm standoff.
3. **The person is painted grey in both domains.** Sim: the user geoms via exact segmentation.
   Real: MediaPipe's person segmenter. Sim silhouettes are randomly grown, shrunk and punctured
   so ragged real masks are in-distribution. Both cameras are masked.

Cost: the policy is blind to human appearance. That is acceptable only because the keep-out check
and the human on the stop live outside the policy.

## 5. What the SO-101 measurements say

All from `jar_act_v3/step_15000` — files in [`evidence/so101_jar/`](evidence/so101_jar/). 96
episodes per row on shared seeds; binomial error about ±4 points. **Simulation against simulation;
nothing here has touched a real camera.**

**Backgrounds.** In training: 84 % (kitchen photos), 83 % (wallpaper pool). Held out: 81 % on a
first draw, 70 % on a second draw of the same scenes — a real gap of several points that depends
on which photos come up. (The project's old "83 % on never-seen backgrounds" was evaluated on the
training pool; corrected in `evidence/so101_jar/docs/ARRIVAL_DAY.md` §6.)

**Camera realism** — whole-image effects the training never saw (`code/obs_corrupt.py`):

| Condition | Success | Δ vs clean 84 % |
|---|---|---|
| Exposure −1 EV … +1 EV | 78–80 % | −4 to −6 |
| White balance warm / cool | 78 % / 86 % | −6 / +2 |
| Defocus blur σ = 1 px | 80 % | −4 |
| **Defocus blur σ = 2 px** | **48 %** | **−36** |
| Motion blur 7 px | 76 % | −8 |
| Sensor noise light / dim-room | 82 % / 76 % | −2 / −8 |
| JPEG q40 / q15 | 82 % / 81 % | −2 / −3 |
| Lifted shadows (γ 0.75) | 85 % | +1 |
| "Evening webcam" combo | 78 % | −6 |
| "Bad webcam" combo (−0.7 EV, warm, blur σ 1.5, noise, JPEG q30) | 58 % | −26 |
| Held-out backgrounds, clean image | 70 % | −14 |
| **Held-out backgrounds + "evening webcam" combo** | **51 %** | **−33** |

Taken one at a time, camera effects are cheap, with one cliff: defocus blur. **The row that
matters most is the last one.** Held-out backgrounds alone cost 14 points and the mild camera
combo alone costs 6; together they cost 33. The margins compound instead of adding — and a real
deployment is exactly that stack (an unseen room *and* a real camera), so a policy trained the old
way should be expected to arrive well below its sim score. Training has to cover both at once.
Full table: `evidence/so101_jar/results/jar_act_v3.stress_obs.txt`.

**What was actually fragile:** camera *position* (84 / 66 / 46 % at ±2 / 7 / 10 cm offset;
wide randomisation with filtered demos recovers 83 / 77 / 69 / 77 / 58 % at ±0 / 4 / 10 / 15 /
20 cm), and checkpoint choice (successive saves swing 46 / 67 / 44 / 56 %).

## 6. Levers, plan, gates

| Lever | Evidence | Setting |
|---|---|---|
| Camera-pose diversity | MolmoBot's main lever; ±4 cm here was too narrow, ±15 cm filtered worked | ±25 cm, ±20°, FOV ±5° |
| Whole-image augmentation, **blur above all** | SplatSim 21 % → 86 %; the σ = 2 px cliff above | `obs_corrupt` ops per episode, blur to σ ≈ 2.5 px, random erasing |
| A much larger background pool | The held-out gap above, with a 136-image pool | 10× the pool; keep `bg_heldout` out |
| Latency, soft servos | v2 fell 72 → 60 % under lag it had not trained for | lag 1–3 ticks, gain 0.1–1× |
| Ten real demonstrations | NVIDIA co-training: +38 % relative from 10 demos | escape hatch; collect by keeping the policy's own successes |

1. **Data generator (T+0–1.5).** Port the compositing and randomisation onto `sim/yam_scene.py`;
   prompts and person mask in from the first episode. Gate: expert ≥ 95 % across the randomisation.
2. **First policy (T+1.5–3.5).** ACT; every checkpoint evaluated closed-loop as it is written, on
   held-out backgrounds with the mild camera combo. **Gate A: ≥ 75 % in sim.**
3. **Gate B — real frames, no motion.** Real camera frames and prompts, arm posed. Does the first
   predicted action point at the real object as it does in sim? Zero arm risk, and the first real
   frame any policy from this lineage will have seen. Fail → co-train or fall back.
4. **Hardware.** 0.25 rad/s behind the limiter, block first, stand-in head before a person. Gate:
   three consecutive hands-off runs; otherwise the fallback is the demo.

Shared with the fallback, so never wasted: arm bring-up (`can0`, joint sign/offset map, FK under
10 mm at five poses), the gaze selection screen publishing on UDP `:8765`, and the 2D publishers.

## 7. Fallback: the modular path

Trades training for one calibration. ArUco board rigid to the robot's base plate, `solvePnP` every
frame for a continuous `base_from_camera`; objects by ray ∩ plane at half the object's height
(`table_plane_intersection` on branch `arm-ik-rl`); mouth by MediaPipe through the same transform;
IK planner from `sim/`; a thin ReAct supervisor over the six skills. Objects at 0.25–0.45 m; touch
test under 10 mm before trusting it.

## 8. Safety

Sim or dry-run by default; hardware only behind `--enable-hardware` with a human on the stop.
Every demonstration ends 15 cm short of the mouth, so stopping short is learned, and a keep-out
check computed from the arm's own joint angles enforces it regardless of the policy's output;
closer needs explicit approval and a 0.05 m/s cap. Do not touch motor registers (this arm reports
`TIMEOUT=0`). Software deadman: no fresh, valid prompts for 400 ms → hold the measured pose.
Prop food.

## Sources

[MolmoBot](https://allenai.org/blog/molmobot-robot-manipulation) ·
[SplatSim](https://arxiv.org/abs/2409.10161) ·
[Sim-and-Real Co-Training](https://arxiv.org/abs/2503.24361) ·
[ABC-130k](https://arxiv.org/abs/2606.27375) ·
[DreamZero](https://arxiv.org/abs/2602.15922) · [RDT2](https://arxiv.org/abs/2602.03310) ·
[MPVI](https://arxiv.org/abs/2606.00985) · [Gemini Robotics 1.5](https://arxiv.org/abs/2510.03342) ·
[γ, Intent at a Glance](https://arxiv.org/abs/2601.05336) · [Guava](https://arxiv.org/abs/2606.18363) ·
[Gemini Robotics-ER 2](https://deepmind.google/models/gemini-robotics/embodied-reasoning/) ·
[FEAST](https://arxiv.org/abs/2506.14968) ·
[Feel the Bite](https://dl.acm.org/doi/10.1145/3610977.3634975) ·
[MolmoAct 2](https://arxiv.org/abs/2605.02881) · [GR00T N1.7](https://huggingface.co/blog/nvidia/gr00t-n1-7)
