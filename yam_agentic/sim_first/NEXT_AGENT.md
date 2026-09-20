# Handoff — pick this up here

Updated 2026-09-20. Read this, then `RESUME.md` for how to restart anything.

**The gripper bug from the previous handoff is FIXED.** `YAM_ARM=linear_4310` is now the
default and grasps; 12/12 tests pass on both it and the stock crank. Details in §1 so nobody
re-opens it. The open work starts at §2.

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
  visible. See `media/film_linear4310_can.png` — compare against `sim_first/real_ref/wrist_real.jpg`.

**This invalidates every dataset generated before 2026-09-20.** They carry the wrong gripper,
and the ones that used `linear_4310` also carry the broken wrist view. Regenerate before training.

---

## 2. Task: nothing has ever seen a real camera frame

Unchanged, and now the highest-risk unknown. Gate B in `../GAME_PLAN.md`: feed real iPhone +
wrist frames and real 2D prompts to a trained policy with the arm posed and **stationary**, and
check the first predicted action points at the real object the way it does in sim. Zero arm
risk, and it measures the sim-to-real gap before anything moves.

On whether a non-photorealistic renderer can transfer at all — the literature says yes, but
conditionally, and it is worth knowing the conditions before spending GPU time:

- [Tobin et al. 2017](https://arxiv.org/abs/1703.06907) trained detectors on deliberately
  non-realistic random textures in a low-fidelity renderer and reached 1.5 cm real-world
  accuracy, enough to grasp in clutter.
- [Benchmarking Domain Randomisation](https://arxiv.org/pdf/2011.07112) isolates rendering
  quality and finds photorealism matters **less than which factors you randomise**; mixing
  low-quality with high-quality renders matches pure photorealism.
- [RCAN](https://arxiv.org/pdf/1812.07252) (randomized→canonical) is the published fallback if
  the gap does bite.

The caveat: that evidence is strongest for coarse spatial localisation, not end-to-end fine
manipulation from pixels. Our split already respects it — planner picks what/where, IK does
transport, the learned policy owns only the last inch. And note that **geometry realism matters
more than texture realism here**, which is exactly what §1 was: you can randomise texture away,
you cannot randomise away the wrong finger shape in the foreground of every frame.

---

## 3. What works right now

```bash
cd ~/so101Sim
urlab_bridge/.venv/bin/python -m pytest rl/yam/test_yam_sim.py -q          # 12 passed (linear_4310, default)
YAM_ARM=stock urlab_bridge/.venv/bin/python -m pytest rl/yam/test_yam_sim.py -q   # 12 passed (crank, regression)
YAM_AUG_PROFILE=real urlab_bridge/.venv/bin/python rl/yam/gen_demos_yam.py rl/runs/<run>/demos 1400 --workers 12
rl/yam/run_chain_yam.sh rl/runs/<run> 12000
```

`demo_pick_present.py can` reports `strategy=top lifted 75 mm in 5.9 s`, `object still held: True`.

### State of the runs

| run | what | verdict |
|---|---|---|
| GX10 `runs/yam_g1` | 2,104 demos / 197k frames, first eval 0/56 | **scrap** — wrong gripper, old wrist mount, over-heavy blur. Says nothing about the approach. |
| Mac `rl/runs/yam_v1` | 1,017 demos | superseded |
| Mac `rl/runs/yam_v2` | partial | deleted |

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
  minute what hours of projection algebra could not. `media/linear4310_jaw_openings.png`.
- **Go to the vendor and to the literature before hand-deriving.** i2rt ship the composer, the
  CAD and a PR describing this exact trap; menagerie already solved it for the crank.
- **Check both fingers when testing a gripper.** `mj_forward` does not solve the `<equality>`
  coupling, so setting only `joint7` in a test harness leaves the jaws asymmetric and every
  result meaningless. This produced one entirely bogus sweep before it was caught.
- **`pkill -f <pattern>` over ssh matches the ssh session's own command line.** Kill by PID or
  use `/tmp/relaunch_gx10.sh` on the GX10.
- **Exact-string `.replace()` on MJCF is fragile.** Use a regex and assert the match count.
