# Handoff — pick this up here

Updated 2026-09-20, late. Read this, then `RESUME.md` for how to restart anything.

**The gripper bug from the previous handoff is FIXED** (§1) — `YAM_ARM=linear_4310` is the
default and grasps, 12/12 tests on both grippers. Two more real bugs fell out of it and are also
fixed: the wrist camera was pointing 124 deg away from the TCP, and the demo filter was rejecting
30 % of good episodes.

**Gate A is still open, and the cause is now measured** (§2). Three training runs have failed at
~0 %. It is NOT the gripper, the observation pipeline, the eval yardstick, the batch size, the
chunk length, temporal ensembling, or visual variance — each was ruled out by measurement, and §2
lists how so nobody repeats them. The live hypothesis is simply **optimiser steps**: the model's
fit improves ~x0.75 per doubling of steps with no plateau, and no run has had enough.

---

## 1. RESOLVED: the linear_4310 gripper now grasps

### Root cause

**MuJoCo collides a `<geom type="mesh">` as its convex hull.** The linear_4310 fingertips are
long tapered blades, so each hull is a solid wedge that fills the throat between the jaws. The
true mesh surfaces are a correct 95 mm parallel gripper — at `q=0` the two inner faces meet at
`x≈0`, at `q=0.0475` they sit at `x≈∓47.5 mm` — but what the physics saw was two wedges whose
gap is widest at the fingertips. Closing them drove the object *out* of the jaws instead of
pinching it. Measured: a 25 mm cylinder was pushed 78 mm and expelled, in zero gravity, ending
with `ncon=0`.

This is why every previous suspect came up clean, and why it reproduced on i2rt's own composed
model. i2rt name the same trap in their own finger PR — *"convex envelopes fill recesses"*
(AfterQuery-Research/i2rt#2). The menagerie crank gripper works precisely because someone
hand-authored primitive collision for it (`lf_rot`/`lf_down`, boxes + `sphere_collision` pads)
and kept the mesh for visuals only.

The two measurements disagreeing was the whole puzzle: a probe sphere swept through the jaws
reports the **convex hull** (a V, wide at the tip), while the raw mesh vertices report the
**true surface** (parallel faces). Both were right. If you measure jaw geometry again, be
explicit about which one you are looking at.

### The fix (in `sim/make_arm_linear4310.py`)

Mesh geoms became visual-only (`class="visual" contype="0" conaffinity="0"`), and collision
moved to primitives on the measured inner faces: one box pad plus two `sphere` pads per finger.
Verified aperture, exactly linear and exactly to spec:

| `q` | aperture |
|---|---|
| 0.0 | **0.0 mm** |
| 0.01 | 20.0 mm |
| 0.02475 | 49.5 mm |
| 0.0475 | **95.0 mm** (I2RT spec: 95 mm) |

The TCP fixed itself as a side effect: `ToolFrame` already derives the tool centre from sphere
pads when a gripper has them (the crank path), so adding them moved the TCP **29.65 mm back
from `grasp_site`**, off the knife-edge fingertip and into the middle of the pads. The object
size cap is read from `tool.max_width`, so it corrected itself to 95 mm too — there was never a
hard-coded 75 mm to widen.

### Two follow-on fixes the first one exposed

- **Pre-grasp reach.** A TCP 30 mm further down the tool axis puts the wrist 30 mm higher at the
  pre-grasp, which took `marker` and `block` out of reach at the fixed 100 mm standoff.
  `plan_pick` now tries a ladder (100 / 80 / 60 / 45 mm), mirroring what the lift already did.
- **The wrist camera was pointing backwards.** It was anchored on `grasp_site`, which sits in
  `link_6` on the crank but inside the graft's rotated `gripper_mount` body — so on
  `linear_4310` the camera inherited that rotation and looked **124° away from the TCP**. The
  wrist view was nothing but gripper; a policy trained on it would have been blind. Now anchored
  on `tcp_site`, which stays in `link_6` in both variants: 22.5° off-axis, table and objects
  visible. See [`../media/film_linear4310_can.png`](../media/film_linear4310_can.png) against [`real_ref/wrist_real.jpg`](real_ref/wrist_real.jpg); jaw openings in [`../media/linear4310_jaw_openings.png`](../media/linear4310_jaw_openings.png).

**This invalidates every dataset generated before 2026-09-20.** They carry the wrong gripper,
and the ones that used `linear_4310` also carry the broken wrist view. Regenerate before training.

---

## 2. Gate A: why the policy scores ~0 %, and what is left to try

Three runs, all ~0 %: `yam_v3` (batch 24 / 12k, 887 demos), `yam_v5` (batch 8 / 20k, same data),
`yam_v7` (batch 8, 1,865 demos, pose-B camera, table kept, no blur). `picked` never exceeded
3/55.

### Ruled out by measurement — do not redo these

| suspect | verdict | evidence |
|---|---|---|
| eval yardstick too strict | no | the **expert scores 23/23 = 100 %** under eval's own test (`tools/expert_ceiling.py`); median 2.9 cm against its 6 cm threshold |
| observation pipeline mismatch | no | regenerating a seed gives a **bit-identical** frame to the stored one — pixels and the 17-dim state, max diff 0.0 |
| model collapsed to the mean | no | open-loop MAE 2.9-5 deg against a 17-22 deg mean-action baseline; prediction spread matches the truth's |
| not reading the 2D prompt | no | first action correlates with the object box about as well as the expert's: j1 **−0.81 vs −0.84**, j6 −0.61 vs −0.77 |
| generalisation / backgrounds | no | fails identically on *training* seeds with *training* backgrounds |
| chunk length / blind execution | no | `na=1` (re-infer every tick) fails the same as `na=10` |
| ACT temporal ensembling missing | no | enabling it at the paper's 0.01 changed nothing (`YAM_TE=0.01` on the evaluator) |
| batch size vs the 86 % SO-101 recipe | no | matching batch 8 changed nothing |
| too much visual variance | **no** | pose-B camera (look-down spread 24 deg -> 7 deg), table kept, blur off and 2x the demos moved the fit metric from 5.00 to 4.97 deg at the same step. This one surprised me; it was my main hypothesis. |

### The failure mechanism, precisely

From a **bit-identical** observation the policy is already ~10 deg off on the first action, and
the error compounds monotonically until `j2` hits its limit and the object leaves the wrist view.
The blank wrist frames in failure filmstrips are a *consequence* of that divergence, not a cause.

The reason it compounds is accuracy: the policy's open-loop error is comparable to the signal it
has to predict. A grasp needs roughly 1 cm at 0.4 m of reach, i.e. **about 1.5 deg**; the best
checkpoint so far is 2.9 deg.

### The floor is robust — and one replication delta was never tested

Measured train-set MAE (first action, 150 frames, `n_action_steps=1`):

| | 4k | 8k | 12k | 16k | 20k | baseline |
|---|---|---|---|---|---|---|
| `yam_v7` 5 targets, 1,865 demos | 4.92 | 3.95 | 2.90 | 3.16 | 3.06 | 20.2 |
| `yam_v9` 2 targets, no erasing | 4.26 | 3.26 | 2.82 | — | — | 15.0 |

Narrowing five grasp geometries to two bought **0.08 deg** at step 12000. The ~3 deg floor holds
across: 5 vs 2 objects, 887 vs 1,865 demos, batch 8 vs 24, wide vs pose-B camera, table vs no
table, blur on vs off, erasing on vs off, and 4k to 20k steps. It is none of those.

Note `yam_v9` is only better in ABSOLUTE terms; against its own baseline it is slightly worse
(4.6x vs 5.1x). The task got smaller, the model did not get better. Absolute is what control
cares about, so the gain is real but small.

**The untested delta: control rate.** This pipeline runs **10 Hz with chunk 20**; the SO-101
recipe that scored 86 % ran **25 Hz with chunk 50**. Both span 2.0 s, which is why it was written
off as equivalent — wrongly. At 10 Hz each commanded action holds for **100 ms** against 40 ms,
so the same per-step error produces **2.5x more positional drift before the policy re-observes**.
The measured failure is compounding divergence from a bounded per-step error, so this is the one
deviation whose mechanism matches the symptom. The original handoff named it as suspect #2 and it
was never tried. `yam_v10` tests it (`YAM_HZ=25`, `CHUNK_SIZE=50`).

### VERDICT (2026-09-20, ~05:00): the floor is invariant. Stop configuring, change the design.

`yam_v10` ran the 86 % recipe faithfully — 25 Hz, chunk 50, `na` 25, batch 8 — and landed at
**2.94 deg** at step 12000, against v9's 2.82 and v7's 2.90. The control rate does not move it
either.

| variation | fit @ 12k |
|---|---|
| v7: 5 targets, 1,865 demos, 10 Hz, chunk 20 | 2.90 deg |
| v9: 2 targets, no erasing, 10 Hz, chunk 20 | 2.82 deg |
| v10: 2 targets, **25 Hz, chunk 50** | 2.94 deg |

Invariant also to batch 8 vs 24, 887 vs 1,865 demos, wide vs pose-B camera, table vs none, blur
on/off, erasing on/off, 4k-20k steps, temporal ensembling, and the eval augmentation profile.
Needed: ~1.5 deg. Closed-loop never beat 4 %, and every non-zero sat inside checkpoint noise.

**Do not spend more time on configuration.** The remaining ideas are design changes:

1. **Change the action space — the most promising, and cheap to test.** The policy predicts
   *absolute joint targets*, so with a randomised object position it must internalise inverse
   kinematics across the workspace from ~1,900 demos. ALOHA, where this ACT recipe works from
   ~50 demos, has a FIXED layout and never has to. Predicting a Cartesian end-effector delta is
   close to a visual-servoing law and is what most randomised-scene policies use. `ToolFrame`
   already gives the TCP and `ArmIK` already solves the inverse, so the expert can emit
   Cartesian deltas and the runtime can convert back with the IK that already exists.
2. **Orders more data.** MolmoBot, the result this plan is modelled on, used 1.7 M trajectories.
   We have 1,865. Generation is ~13 min per 700 episodes on 12 cores; the GX10's 121 GB is the
   machine for a dataset this size, not a 64 GB laptop.
3. **Co-train with real frames.** NVIDIA report +38 % relative from as few as 10 real demos.
4. A larger backbone or a diffusion policy — last, and only with (1) and (2) done.

**What IS solid and should be kept:** the gripper, the wrist camera, the demo filter, the table
and the pose-B camera are correctness fixes, not tuning. The expert scores 100 % under the
evaluator's own test, so the simulator, planner and IK are sound — it is only the learned policy
that fails. The modular path in `../GAME_PLAN.md` §08 validates against this same simulator.

## 2c. What is left after that

## 2b. Nothing has ever seen a real camera frame

Still true, and still the highest-risk unknown after Gate A. Gate B: feed real iPhone + wrist
frames and real 2D prompts to a trained policy with the arm **stationary**, and check the first
predicted action points at the real object the way it does in sim. Zero arm risk.

On whether a non-photorealistic renderer can transfer: [Tobin et al. 2017](https://arxiv.org/abs/1703.06907)
reached 1.5 cm real accuracy from deliberately non-realistic textures;
[Benchmarking DR](https://arxiv.org/pdf/2011.07112) finds photorealism matters less than *which*
factors are randomised; [RCAN](https://arxiv.org/pdf/1812.07252) is the fallback. The caveat:
that evidence is for coarse localisation, not fine manipulation — which is why the split puts
planning and transport outside the policy. And **geometry realism matters more than texture
realism**: the gripper bug proved you cannot randomise away a wrong finger shape.

## 3. What works right now

Everything below runs **from this clone**. The only things not vendored are the two model
trees (mujoco_menagerie is ~1 GB, i2rt ships its own repo) — point at them once:

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie
git clone https://github.com/i2rt-robotics/i2rt
export YAM_MENAGERIE=$PWD/mujoco_menagerie/i2rt_yam
export I2RT_ROOT=$PWD/i2rt

cd yam_agentic/sim_first/sim
python -m pytest test_yam_sim.py -q                    # 12 passed (linear_4310, the default)
YAM_ARM=stock python -m pytest test_yam_sim.py -q      # 12 passed (crank, regression)
python demo_pick_present.py can                        # filmstrip + "object still held: True"

YAM_AUG_PROFILE=real python gen_demos_yam.py <run>/demos 1400 --workers 12
./run_chain_yam.sh <run> 20000                          # BATCH=8 CKPT_EVERY=4000 recommended
```

`<run>` is any directory you choose. The background pools are not in the repo either:
`bg_heldout/fetch.sh` recreates the held-out set, and `bg_train` is the same trick with seeds
`yamtrain1..800`.

Every measurement quoted in this file is reproducible from
[`../tools/`](../tools/README.md) — see that README for what each script answers.

From `sim/`, `python demo_pick_present.py can` reports `strategy=top lifted 75 mm in 5.9 s`, `object still held: True`.

### State of the runs

| run | data | training | result |
|---|---|---|---|
| `yam_v3` | 887 demos (63 % filter, wide cam, no table) | batch 24 / 12k | **0/55** at steps 2000-8000 |
| `yam_v5` | same data | batch 8 / 20k | **1/56** |
| `yam_v6` | **1,865 demos (93 % kept)**, pose-B cam, table, no blur, 197k frames / 42 GB | — | the current dataset |
| `yam_v7` | 9 of v6's 12 shards (149,682 frames, 34 GB) | batch 8, extended to **60k** | running |
| GX10 `yam_g1` | old | — | scrap: wrong gripper, backwards wrist cam |

Only 9 of 12 shards are in `yam_v7`: all 12 is 44.8 GB of host RAM and the Mac has ~52 GB free,
which is how an earlier run got starved (see the trap below). The full 197k frames want the
GX10's 121 GB, not a laptop.

## 4. Order to work in

1. Regenerate one dataset with the fixed gripper, train, confirm the first eval is non-zero. If
   it is still 0 %, suspect the 2D-prompt conditioning then the 10 Hz control rate, in that
   order — dump a filmstrip of a *policy* rollout beside an *expert* rollout and compare where
   they diverge.
2. **Gate B** (§2) on real frames.
3. Then hardware, behind the limiter, per `../HANDOFF.md` §8 and `ARM_NOTES.md` on `arm-ik-rl`.

## 5. Things that cost time — avoid them

- **Say which geometry you are measuring.** Convex hull (what MuJoCo collides) and mesh surface
  (what the camera renders) can disagree by 80 mm on a tapered finger. Nearly every wrong
  conclusion in this investigation came from mixing them up.
- **Render it.** Two renders — the jaws at three openings, and the rig mid-grasp — settled in a
  minute what hours of projection algebra could not. [`../media/linear4310_jaw_openings.png`](../media/linear4310_jaw_openings.png).
- **Go to the vendor and to the literature before hand-deriving.** i2rt ship the composer, the
  CAD and a PR describing this exact trap; menagerie already solved it for the crank.
- **Check both fingers when testing a gripper.** `mj_forward` does not solve the `<equality>`
  coupling, so setting only `joint7` in a test harness leaves the jaws asymmetric and every
  result meaningless. This produced one entirely bogus sweep before it was caught.
- **`pgrep -f` / `pkill -f <script name>` matches the shell running the command.** This bit three
  separate times in one night: a waiter loop that could never exit, a suspend that paused nothing,
  and a kill that killed its own PIDs. Match the interpreter path with the bracket trick —
  `ps -eo pid,command | grep "[.]venv/bin/python.*train_act_yam.py"` — never the script name alone.
- **`multiprocessing` spawn workers do not match the parent's pattern at all**, so killing the
  parent orphans them holding gigabytes. Kill children by PPID, then sweep `[s]pawn_main`.
- **Do not run demo generation beside GPU training.** The trainer holds the whole image tensor in
  host RAM and gathers a random batch every step; generation pushed it into swap (19.6 M swapouts,
  RSS 2.7 GB of 20.8) and throughput fell 6.75 -> 4.73 it/s while the GPU idled. Overlap idle
  resources, never the critical path's memory.
- **Exact-string `.replace()` on MJCF is fragile.** Use a regex and assert the match count.
