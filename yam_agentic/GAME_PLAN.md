# Game plan — apple to mouth

The research behind the build, and what we deliberately chose not to do. Same content as the
write-up at <https://claude.ai/artifact/ECpTKyBYz7adiNBwzKVGuM>.

## Is it even possible in sim?

Yes, and more easily than the team's previous SO-101 task. `mujoco_menagerie/i2rt_yam` ships a
YAM MJCF, and everything else — table, objects, a seated user with a mouth — is ordinary MuJoCo.
Measured envelope and rendering rates are in [MEASUREMENTS.md](MEASUREMENTS.md).

The part that *looks* hardest — the user — turns out not to need solving. A mouth is a site on a
head body and the delivery is inverse kinematics, not learning. The head has to exist as
*geometry*, because that is what the planner must avoid and what the virtual wall wraps, but it
never has to look like anyone: **no learned component sees a face.** On the real side MediaPipe
finds the mouth in the camera frame; in sim the site position is simply known.

## Three routes to a working grasp

Only the grasp is contested. Everything above it (which object, where it goes) and below it
(IK, safety limits) is the same in all three.

| route | what it is | needs | venue-change cost | read |
|---|---|---|---|---|
| **A · detect & plan** ← backbone | Detector gives the object pose, IK plans the grasp. No learning. | calibration + a detector | ~2 min recalibrate | Highest reliability per hour. OK-Robot got **58.5 % in unseen homes** on this architecture, 82 % in uncluttered scenes — and a tabletop is uncluttered. |
| **B · wrist-cam ACT** ← upgrade | Distil the IK expert onto wrist pixels. | ~1 h on the GX10 | near zero | Recovers the last-inch behaviour route A lacks. Eye-in-hand removes the viewpoint brittleness that capped the SO-101 work. |
| **C · MolmoAct 2** ← side bet | Ai2's open VLA, YAM-native, language in / joint poses out. | CUDA, ~26 GB | designed for it | See below. |

### On MolmoAct 2, honestly

It is the most on-target published system that exists for this ask: open weights, trained on
YAM arms, language-conditioned, trained with randomised camera placement, **87.1 % zero-shot on
real DROID tasks**, and the GX10 has the memory for it.

Against it: 180 ms/action is an *H100* figure and the GB10 has roughly a third of an H100's
memory bandwidth; the stack is arm64; and **there is no single-arm YAM checkpoint** — the
released one is bimanual with three cameras, so "zero-shot" would mean running it with half its
inputs missing. Three unknowns stacked inside a 24-hour window.

If the backbone is done with hours to spare, the higher-value version of this bet is
`lerobot-rollout` with the SO-100/101 checkpoint on the SO-101 that is already calibrated — a
second, independent demo rather than a rewrite of the first.

## Designing for a venue that moves

The measured problem, from this team's own SO-101 results: a fixed third-person policy scored
83–86 % at its trained camera pose, 66 % at ±7 cm, **46 % at ±10 cm**, and 58–64 % when retrained
at a different elevation. Widening the camera randomisation during training made it *worse*.
Four countermeasures, ranked by value per hour:

1. **Bolt the scene camera to the robot.** Eliminates the shift instead of tolerating it. One
   afternoon of hardware, permanent payoff.
2. **Make the wrist camera the policy's primary input.** Its extrinsics cannot drift. The
   literature's caveat is that wrist-only is not *sufficient* for everything — fine, because the
   scene camera is doing semantics, not servoing.
3. **Condition the policy on camera extrinsics** (Plücker ray maps of the known pose). Published
   results restore ACT / Diffusion Policy / SmolVLA performance under viewpoint shift, and the
   team's auto-calibration already produces exactly those extrinsics at deploy time.
4. **Keep the 2-minute recalibration loop sharp**, as one command, and rehearse it before it is
   needed under time pressure.

And the meta-point: make the route with **no training in it** the demo backbone, so a venue
change cannot invalidate a model.

## Safety

The arm is 5 kg with a 2–3 kg payload moving toward a person's head. The design, following the
assisted-feeding literature:

- Stop at a **staging pose ~15 cm short** and require an explicit confirm to go closer. The gap
  is the safety margin.
- **Do not point the jaws at the face.** The planner takes the smallest reorientation that
  reaches the staging pose, which keeps a top-down grasp tilted down. This also happens to be the
  only orientation that does not drop the payload.
- Clamp Cartesian speed inside the head's virtual wall; force-limit the gripper; a person holds
  the stop.
- The LLM can *request* the approach; it never commands joint targets.
- Demo with a prop rather than actually feeding anyone.

## Sources

- [I2RT YAM SDK and models](https://github.com/i2rt-robotics/i2rt) · [YAM specs](https://doc.i2rt.com/products/yam) — 6 DoF, CAN 1 Mbit/s, MJCF + URDF provided, ships with a CANable adapter
- [MolmoAct2: Action Reasoning Models for Real-world Deployment](https://arxiv.org/abs/2605.02881) and the [LeRobot policy docs](https://huggingface.co/docs/lerobot/main/en/molmoact2) — NVIDIA-only, 180 ms/action on H100, bimanual YAM checkpoint
- [Do You Know Where Your Camera Is? View-Invariant Policy Learning with Camera Conditioning](https://arxiv.org/abs/2510.02268) — Plücker extrinsics conditioning for ACT / DP / SmolVLA
- [FEAST: A Flexible Mealtime-Assistance System](https://arxiv.org/abs/2506.14968), RSS 2025 best paper — GPT-4o writing behaviour-tree parameters; autonomous mouth perception
- [Feel the Bite](https://dl.acm.org/doi/10.1145/3610977.3634975), HRI 2024 — multi-view mouth perception robust to tool occlusion
- [OK-Robot](https://arxiv.org/abs/2401.12202) — 58.5 % zero-shot pick-and-drop in unseen homes, 82 % uncluttered, no training
- [MOKA](https://arxiv.org/abs/2403.03174), [ReKep](https://arxiv.org/abs/2409.01652), [VoxPoser](https://arxiv.org/abs/2307.05973) — VLM-to-constraint patterns for the agent layer
- [SAM 3: Segment Anything with Concepts](https://arxiv.org/abs/2511.16719) and [SAM 3.1](https://ai.meta.com/blog/segment-anything-model-3/) — noun-phrase prompts segment every instance; ~30 ms/image on an H200; custom SAM licence, fine for research
- [Assistive Gym](https://arxiv.org/abs/1910.04700) and [Assistax](https://arxiv.org/abs/2507.21638) — prior art for feeding tasks with a simulated human; Assistax ships MuJoCo humanoid assets with a mouth target
- [mujoco_scanned_objects](https://github.com/kevinzakka/mujoco_scanned_objects) — 1030 household objects as MJCF, if parametric primitives stop being enough
- [ASUS Ascent GX10 specs](https://www.asus.com/networking-iot-servers/desktop-ai-supercomputer/ultra-small-ai-supercomputers/asus-ascent-gx10/techspec/) — GB10 Grace Blackwell, 20-core Arm, 128 GB LPDDR5x unified, DGX OS
- [python-can gs_usb backend](https://python-can.readthedocs.io/en/stable/interfaces/gs_usb.html) and [CANable](https://canable.io/) — the fallback if the arm ever has to be driven from macOS, where SocketCAN does not exist
