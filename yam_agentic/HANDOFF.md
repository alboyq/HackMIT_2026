# Handoff — agentic pick-and-deliver on the openYAM

> **Update, 19 Sep late evening — the direction changed after this was written.** The plan is now
> **sim-first**: a policy trained in simulation with 2D prompts replaces the run-time
> "perception → 3D → IK" stack described in §3–§4 below, which becomes the fallback. Read
> [GAME_PLAN.md](GAME_PLAN.md) first. §5–§8 here (what is built, decisions, bugs, hardware
> assumptions) still stand. Evidence from the earlier SO-101 work is in
> [`evidence/so101_jar/`](evidence/so101_jar/).

**Status as of 2026-09-19 evening.** Simulation track is built and green; nothing has run on the
physical arm yet. Everything here is reproducible from this folder — see [README.md](README.md).

---

## 1. What we are building

One sentence: **you ask for an object in plain language, the arm picks it up and brings it to a
place near you that makes sense for that object.** The headline demo is "bring me the apple" →
the arm picks the apple and presents it in front of your mouth. A mug goes to the table in front
of you, a marker gets handed over handle-first.

The interesting claim is not the grasp. It is that an LLM chooses *which* object and *where it
should go*, and that the choice is grounded in a real scene through a camera.

## 2. Constraints that shaped every decision

| Constraint | Consequence |
|---|---|
| **< 24 h to demo** | The backbone is the route with no training in it. Learned policies are explicitly optional. |
| **The venue changes** (we have to leave the hackathon space) | Nothing may depend on a fixed camera pose that has to be recalibrated and retrained after every move. This is the single biggest architectural driver. |
| **ASUS Ascent GX10 available** (GB10, 20-core Arm, 128 GB unified, DGX OS) | It is a CUDA box, but more usefully today it is a *Linux* box — SocketCAN exists there and does not on macOS. Drive the arm from the GX10. |
| **Two plain USB RGB cameras**, one wrist, one scene. No depth. | 3D comes from the table-plane assumption plus the calibrated camera, not from a depth sensor. |
| **CAN is up** from the GX10, a teammate owns the arm | The session's biggest risk retired early. Remaining hardware risk is coordination, not electronics. |

## 3. Architecture, and the one rule that matters

Four layers. **The LLM chooses *what* and *where*; it never emits a joint angle, a velocity, or a
safety envelope.** This is the FEAST pattern (RSS 2025 best paper — GPT-4o writes parameters into
a behaviour tree, the tree does the moving), and it is why that system survives real users.

```
L3  agent      Claude, once per request. Scene image + "bring me the apple" in;
               {target object, grasp strategy, destination anchor, offset, confirm policy} out.
               Owns the destination table: apple -> 12 cm in front of the lips;
               mug -> upright on the table within reach of the dominant hand;
               marker -> handle first toward the open palm. Re-plans on a failed skill.

L2  percept    SAM 3.1 with a noun-phrase prompt ("red apple") -> instance mask ->
               table-plane back-projection -> 3D pose, extent, principal axis.
               MediaPipe Face Landmarker for the mouth. Outputs poses in the robot frame only.
               Runs ONCE PER PLAN STEP, not per control tick.

L1  skills     locate() grasp() transport() present() release() home().
               Typed, each returns success + a verification image the agent can look at.
               Deterministic IK. Every safety limit lives here.

L0  control    30 Hz absolute joint targets over CAN. Velocity clamp, virtual wall around
               the head, force-limited gripper, human holding a stop.
```

**Why the segmentation model earns its slot:** a bounding box gives a centroid; a *mask* gives the
object's contact line with the table (the input to the table-plane back-projection), plus extent
and principal axis, which is what decides top-down vs side and whether the object fits the
gripper. It also deletes the part of the old localiser that keyed on the room — a noun-phrase
prompt does not care what the table is made of.

## 4. Cameras: two, and one of them bolts to the robot

- **Wrist camera** does the work that has to be *precise*. Eye-in-hand is geometrically immune to
  venue changes: its pose relative to the gripper is welded.
- **Scene camera** does the work that has to be *semantic*: which object, where the person is,
  safety overview. **Mount it on a post attached to the robot's own clamp plate, not the table.**
  Then its extrinsics are a property of the arm, and "we moved rooms" costs a background re-shoot
  instead of a recalibration plus a retrain.
- A third camera is not needed. The face can come from the scene camera, or from the wrist camera
  on the way up — which is what the assisted-feeding systems do.

Prior evidence from this project's SO-101 work, which is exactly why this matters: a fixed
third-person policy scored 83–86 % at its trained camera pose, 66 % at ±7 cm, **46 % at ±10 cm**,
and 58–64 % when retrained at a different elevation. Widening the camera randomisation during
training made it *worse*, not better. The fix is architectural.

## 5. What is built and working

`sim/` — phase 0, **11/11 tests green**, ~1 s to run the suite.

| file | what it is |
|---|---|
| `yam_scene.py` | Generates the MJCF: YAM + wrist camera on link_6 + base-mounted scene camera + table + four objects + a seated user (head, torso, hand) carrying `mouth` and `hand` sites. Also exposes `segment()` — MuJoCo's exact per-geom masks, which are free ground truth to score the real segmentation model against. |
| `yam_ik.py` | `ToolFrame` measures the gripper off the model (TCP between the pads, tool axis, jaw axis, pad opening). `ArmIK` is damped-least-squares IK, verified sub-millimetre against a finite-difference Jacobian. `solve_best` does position + tool direction with roll free, then *optionally* pins the jaw yaw. |
| `yam_expert.py` | Plans the whole joint trajectory up front and emits **absolute joint targets** — same shape as the SO-101 expert, which is what makes it safe to replay open-loop on a real arm and usable as imitation labels. Only the gripper close is reactive. |
| `test_yam_sim.py` | The phase-0 gate. Scene loads, tool frame correct, IK round-trips, the measured envelope still holds, the feasibility gate actually rejects oversized objects, all four objects pick-and-lift in closed-loop physics, and the apple is delivered to the mouth **still in the gripper** without touching the head. |
| `demo_pick_present.py` | Runs a full pick → present and writes a two-row filmstrip (scene camera above, wrist camera below). |

`media/` holds the four filmstrips. All four objects pick and deliver: apple lifts 135 mm and
stops 14.9 cm from the mouth still held; marker 143 mm; block 138 mm; mug 126 mm.

## 6. Decisions, with the reasoning

**Grasp strategy is chosen by RADIUS, not by object shape.** The envelope (§ MEASUREMENTS.md) is
top-down inside 0.50 m, tilted 30° out to 0.65 m, and a *level* side grasp only beyond 0.65 m
because the arm cannot fold back any closer. Shape only decides the grasp height and whether the
jaw yaw matters. **Put the objects at 0.25–0.45 m.**

**IK constrains the tool direction, not the full orientation.** Pointing the tool straight down
*and* demanding a particular jaw yaw is frequently infeasible on this wrist (joint4/5 are ±90°,
joint6 ±120°). Tool-down with roll free is reachable across a wide band. Boxes ask for a jaw yaw
as a *preference*; `solve_best` falls back to free roll and reports `jaw_locked=False`.

**The tool does not point at the mouth during delivery.** Rotating a top-down grasp all the way to
"tool pointing at the mouth" is what dropped the apple — the pads lose contact past ~70° from
vertical and a sphere rolls straight out from between two flat pads. Nothing requires the tool to
point at the mouth; the payload just has to arrive in front of it. The planner now takes the
*smallest* turn that reaches the staging pose, which also keeps the jaws off the person's face.

**The arm stops at a staging point ~15 cm short and does not close the gap.** That gap is the
safety margin. Crossing it is a separate, explicitly confirmed action. The LLM can request the
approach; it cannot command it.

**Rejected: MolmoAct 2 as the backbone.** It is the most on-target published system for this ask
— Ai2, open weights, YAM-native, language-conditioned, 87.1 % zero-shot on real DROID tasks — and
the GX10 can hold it (~26 GB). But 180 ms/action is an *H100* number, the GB10 has roughly a third
of an H100's bandwidth, the stack is arm64, and **there is no single-arm YAM checkpoint** (the
released one is bimanual, three cameras). Three unknowns stacked inside 24 hours. If route A is
demo-ready with hours to spare, the better version of this bet is `lerobot-rollout` with the
SO-100/101 checkpoint on the SO-101 that is already calibrated — a second, independent demo.

**Rejected: putting the person in the background photo.** The existing sim pipeline composites
renders over real photos of the *empty workspace*. Baking a face into a fixed 2D backdrop would
be wrong: it stays put while training jitters the camera, so its apparent position contradicts the
geometry the arm reaches toward. The person is a *body* in the scene; backgrounds are for the room.

## 7. Bugs found while building this — each one cost a real measurement

Worth reading before changing the planner, because most of them look like they cannot happen.

1. **`grasp_site` is not the TCP.** On this model it sits 4.4 cm from the point between the pads,
   and its z-axis is 19° off the tool axis. A first pass used it as a proxy and produced an
   envelope that was too generous (0.55 m / 0.78 m). `ToolFrame` now derives the TCP from the pad
   geoms.
2. **Descending at 0.28 m/s** gave 16 mm of *cross-track* servo lag, which dropped a finger onto
   the mug instead of beside it. Cartesian segments now run at 0.3 rad/s.
3. **Approaching with the jaws only a few mm wider than the object** left no room for that lag.
   Approaches now open fully.
4. **The straight-up lift was silently unreachable** for far objects (tool-down at r = 0.44 m runs
   out of wrist by z ≈ 0.10 m), and the code fell back to *not lifting*. Lifts now pull toward the
   base as they rise, trying progressively gentler heights.
5. **Dwell times were in sim steps, not seconds** — a 30-step "hold" is 60 ms, so the gripper was
   closing while the arm was still ~1 cm from the grasp pose.
6. **The delivery test was too weak.** It asserted the TCP stopped 15 cm from the mouth but never
   checked the apple was still in the gripper, and it passed while the apple was being flung onto
   the table. It now checks the payload arrived.

## 8. Assumptions that need checking against the real hardware

- **Gripper force.** The menagerie MJCF gives the finger actuator `kp=100`, which measures out at
  **1.2 N per pad** — a 150 g apple slides out of the jaws during a gentle lift. That is a
  placeholder, not a spec: the real YAM is rated for 2 kg, which needs roughly 20 N. The sim now
  runs `kp=800` (~10 N per pad, override with `YAM_GRIP_KP`). **Ask whoever has the arm what the
  real gripper's force limit is set to** — every sim grasp number depends on this.
- **Pad opening is 83 mm**, so the object set is capped around 75 mm. The mug in the scene was
  shrunk to 64 mm. A real mug body will not fit and has to be taken by the handle or the rim.
- **Joint conventions.** The sim uses the MJCF's zero pose. The real arm has its own signs and
  offsets. Agree these in writing before wiring the two together, along with the gripper's
  open/closed command range.
- Reach numbers assume no self-collision and an empty table; the planner shaves a little off.

## 9. Next steps

| | what | who | state |
|---|---|---|---|
| **Phase 0** | Sim scene, IK, both grasp strategies, planner | sim | **done, 11/11 green** |
| **Phase 1** | Agent loop on ground-truth poses: typed skill API + Claude driving it. Destination table, replanning, refusals. **Showable on its own — record it as insurance.** | sim | next |
| **Phase 2** | Swap ground truth for cameras: SAM 3.1 behind an HTTP service on the GX10, table-plane back-projection, MediaPipe mouth. Validate the localiser against MuJoCo's exact masks to get an error in centimetres before touching a real frame. | sim | |
| **Phase 3** | Real arm: mount the scene camera on the clamp plate, calibrate, background, joint map, run with confirm-per-step | both | blocked on arm time |
| **Phase 4** | Wrist-cam ACT for the last inch, trained on the GX10 | optional | |

**Known rough edge:** the scene camera's framing is mediocre — it catches the table and only the
bottom of the head. It needs re-aiming before phase 2, since the perception layer needs the
tabletop *and* the face in one view. `tools/find_wrist_cam_mount.py` shows the pattern for doing
this by measurement rather than by guessing.

## 10. Where this came from

This is a pivot off the same team's SO-101 work (`~/so101Sim`, not in this repo): a macOS
UE5 + MuJoCo pipeline, a sim-to-real jar-grasping stack, and an ACT policy that reached 83–86 %
in sim. Three things carried over and are worth reusing rather than rewriting:

- the **auto-calibration** approach (silhouette-based camera fit, ~2 min) and background capture,
- the **parametric object spec** → generated MJCF pattern, which `yam_scene.py` follows,
- the **plan-then-replay** expert structure that emits absolute joint targets.

What did *not* carry over: every grasp assumption. The SO-101 physically could not grasp top-down
(its wrist capped at 95°), so that entire pipeline was built around side grasps of tall objects.
The YAM can grasp top-down, which changes the default strategy and most of the geometry.

Plan and research write-up, with sources: <https://claude.ai/artifact/ECpTKyBYz7adiNBwzKVGuM>
(same content as [GAME_PLAN.md](GAME_PLAN.md)).
