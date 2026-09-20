# tools — the measurements behind the claims

Every number the sim-first docs quote about the gripper, the demo filter or Gate A is produced
by a script here. They run from a clone; nothing points at anyone's laptop.

## Setup (once)

These tools need two model trees that are **not vendored** (mujoco_menagerie is ~1 GB, and i2rt
ships its own repo):

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie
git clone https://github.com/i2rt-robotics/i2rt
export YAM_MENAGERIE=$PWD/mujoco_menagerie/i2rt_yam
export I2RT_ROOT=$PWD/i2rt
pip install mujoco numpy imageio opencv-python        # + torch & lerobot for the policy tools
```

`_paths.py` also finds them automatically if they sit beside this clone or in `$HOME`. Run every
tool from this directory (`cd yam_agentic/sim_first/tools`).

## The gripper (resolved 2026-09-20)

| script | answers | measured |
|---|---|---|
| `gripper_geometry_probe.py` | why it did not grasp | surface = 95 mm parallel jaws; **hull** = a V, 88 mm at the tips, 32 mm at the root |
| `gripper_grasp_rig.py` | does it hold anything | a 25 mm cylinder was pushed **78 mm and expelled in zero gravity** |
| `gripper_pad_coords.py` | the fix's pad placement | rerun if the meshes change; paste into `sim/make_arm_linear4310.py` |

**The finding**: MuJoCo collides a `<geom type="mesh">` as its convex hull, and the hull of a
tapered blade is a wedge that fills the throat. i2rt name the same trap in their own finger PR
("convex envelopes fill recesses"); mujoco_menagerie avoids it for `crank_4310` by hand-authoring
primitive collision and keeping the mesh for visuals. The fix does the same — mesh visual-only,
collision on a box pad plus two sphere pads per finger — giving **0.0 mm closed to 95.0 mm open**,
the I2RT spec, and moving the TCP 29.65 mm back off the knife-edge fingertip into the pads.

If you measure jaw geometry again, **say which geometry you mean.** The hull (what physics
collides) and the surface (what the cameras render) disagreed by 80 mm here, and mixing them up
produced every wrong conclusion in this investigation.

## The pipeline

| script | answers | measured |
|---|---|---|
| `expert_ceiling.py` | is Gate A's yardstick achievable | expert **23/23 = 100 %**, median 2.9 cm vs eval's 6 cm threshold |
| `why_staging.py` | why demos were rejected | all rejections were pose-matching; fixing it took keep rate **63 % → 94 %** |

Run `expert_ceiling.py` before believing any policy score — it separates "the policy is bad"
from "the test is impossible".

## Verifying the gripper end to end

```bash
cd ../sim
python -m pytest test_yam_sim.py -q                 # 12 passed (linear_4310, the default)
YAM_ARM=stock python -m pytest test_yam_sim.py -q   # 12 passed (crank, regression)
python demo_pick_present.py can                     # filmstrip; reports lift height and "still held"
```
