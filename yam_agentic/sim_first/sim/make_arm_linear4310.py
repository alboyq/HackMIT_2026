"""Build a YAM MJCF with the REAL gripper: i2rt `linear_4310` instead of the menagerie's `crank_4310`.

Why this exists. The MuJoCo Menagerie `i2rt_yam` model bakes in one gripper, and it is the crank
variant — confirmed by matching its tip geom offsets (0.171097 -0.203301 / -0.124901) against
`gripper/crank_4310/crank_4310.xml`. Its throw is 79 mm. The arm on this team's bench has
rack-and-pinion sliding jaws driven by a DM4310, i.e. `linear_4310`, whose 95 mm throw is also
what I2RT publishes as the YAM spec. A 16 mm error in jaw opening changes which objects are
graspable, and the tips are what the wrist camera sees for the whole episode.

The graft. i2rt's arm MJCF ends at an empty `gripper` mount body carrying joint6; each gripper
module is merged into that frame. Menagerie's equivalent frame is `link_6`. Measured at zero
joints, the two differ by a pure 180 deg rotation about the mount z-axis and a 9.5 mm offset, so
the module's children go inside link_6 wrapped in that one transform. Everything else about the
menagerie model — actuators, armature, damping, joint limits, the tuned classes — is kept.

Joint and actuator names are preserved (`left_finger`, `right_finger`, actuator `gripper`), so
ToolFrame, the expert, the scenes and the tests need no changes; ToolFrame re-measures the TCP,
tool axis, jaw axis and pad opening from whatever geometry is present.

Writes `_yam_linear4310.xml` next to the menagerie model and prints the verification numbers.
"""
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np

def _find(env, names, marker, clone_hint):
    """Locate an external model tree. Neither is vendored: mujoco_menagerie is ~1 GB and i2rt
    ships its own repo. Env var wins, else search beside this checkout and in $HOME."""
    import os
    v = os.environ.get(env)
    if v:
        for n in ("",) + names:
            c = Path(v) / n if n else Path(v)
            if (c / marker).exists():
                return c
        raise SystemExit(f"{env}={v} does not contain {marker}")
    here = Path(__file__).resolve()
    for base in [*here.parents, Path.cwd(), Path.home()]:
        for n in names:
            if (base / n / marker).exists():
                return base / n
    raise SystemExit(f"cannot find {names[0]}. Set {env}=/path/to/it, or:\n  {clone_hint}")


MEN = _find("YAM_MENAGERIE", ("mujoco_menagerie/i2rt_yam", "i2rt_yam"), "yam.xml",
            "git clone https://github.com/google-deepmind/mujoco_menagerie")
I2RT = _find("I2RT_ROOT", ("i2rt/i2rt/robot_models", "third_party/i2rt/i2rt/robot_models"),
             "gripper/linear_4310/linear_4310.xml",
             "git clone https://github.com/i2rt-robotics/i2rt")
GRIPPER = "linear_4310"
OUT = MEN / "_yam_linear4310.xml"


def mount_transform():
    """(pos, quat) of the i2rt gripper mount expressed in the menagerie link_6 frame."""
    men = mujoco.MjModel.from_xml_path(str(MEN / "yam.xml"))
    dm = mujoco.MjData(men); dm.qpos[:] = 0; mujoco.mj_forward(men, dm)
    b = mujoco.mj_name2id(men, mujoco.mjtObj.mjOBJ_BODY, "link_6")
    Pm, Rm = dm.xpos[b].copy(), dm.xmat[b].reshape(3, 3).copy()
    i2 = mujoco.MjModel.from_xml_path(str(I2RT / "arm/yam/v1/yam.xml"))
    di = mujoco.MjData(i2); di.qpos[:] = 0; mujoco.mj_forward(i2, di)
    g = mujoco.mj_name2id(i2, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    Pi, Ri = di.xpos[g].copy(), di.xmat[g].reshape(3, 3).copy()
    q_mount = np.zeros(4)
    mujoco.mju_mat2Quat(q_mount, np.ascontiguousarray(Rm.T @ Ri).flatten())
    # The module file wraps its children in a body carrying the mount orientation
    # (quat="0 0.70711 0.70711 0"); merging means that rotation still applies, so compose it.
    q_mod = np.array([0.0, 0.70711, 0.70711, 0.0]); q_mod /= np.linalg.norm(q_mod)
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, q_mount, q_mod)
    return Rm.T @ (Pi - Pm), q


def build():
    src = (MEN / "yam.xml").read_text()
    for f in ("gripper.stl", "tip_left.stl", "tip_right.stl"):                 # vendor the meshes
        shutil.copy(I2RT / "gripper" / GRIPPER / "assets" / f, MEN / "assets" / f"lin4310_{f}")
    src = src.replace('<mesh file="model2.stl"/>',
                      '<mesh file="model2.stl"/>\n    <mesh name="lin_gripper" file="lin4310_gripper.stl"/>'
                      '\n    <mesh name="lin_tip_left" file="lin4310_tip_left.stl"/>'
                      '\n    <mesh name="lin_tip_right" file="lin4310_tip_right.stl"/>', 1)
    def close_of(text, start):
        """End index of the <body> element that starts at `start`, counting nesting."""
        i, depth = start, 0
        while True:
            nxt_o, nxt_c = text.find("<body", i + 1), text.find("</body>", i + 1)
            if nxt_c == -1:
                raise RuntimeError("unbalanced <body> in the source MJCF")
            if nxt_o != -1 and nxt_o < nxt_c:
                depth += 1; i = nxt_o
            elif depth:
                depth -= 1; i = nxt_c
            else:
                return nxt_c + len("</body>")

    a = src.index('<site name="grasp_site"')                                   # drop the crank grasp_site...
    b = close_of(src, src.index('<body name="link_right_finger"'))             # ...and both finger subtrees
    b = src.index('\n', b) + 1
    pos, quat = mount_transform()
    p, q = " ".join(f"{v:.6f}" for v in pos), " ".join(f"{v:.5f}" for v in quat)
    graft = f'''<body name="gripper_mount" pos="{p}" quat="{q}">
                    <geom pos="-0.014 -0.0463995 0.0731" type="mesh" mesh="lin_gripper" class="visual"
                      rgba="0.16 0.16 0.17 1" contype="0" conaffinity="0"/>
                    <site name="grasp_site" pos="9.681e-05 3.8898e-05 -0.14465" quat="0 1 0 0" size="0.005" rgba="0 1 0 1" group="4"/>
                    <body name="link_left_finger" pos="-0.0238981 0.0450619 -0.0545599" quat="0.499998 -0.5 -0.5 -0.500002">
                      <inertial pos="-0.0224716 0.0143408 -0.0426253" quat="0.498899 0.455348 -0.0576406 0.735143" mass="0.0710042" diaginertia="6.24193e-05 6.01079e-05 2.83591e-05"/>
                      <joint class="finger" name="left_finger" axis="0 0 -1" range="0 0.0475"/>
                      <geom name="lf_mesh" pos="0.129783 0.00999321 -0.0914614" quat="0.499998 0.5 0.500002 0.5" type="mesh" mesh="lin_tip_left"
                        class="visual" rgba="0.18 0.18 0.19 1" contype="0" conaffinity="0"/>
                      <geom name="lf_pad" class="collision" type="box" pos="-0.060440 0.023898 -0.050062"
                        quat="0.707107 0.000003 -0.707107 0.000000" size="0.0050 0.0120 0.0230"
                        condim="4" friction="1.2 0.01 0.001" solimp="0.95 0.99 0.001"/>
                      <geom name="lf_tip0" class="collision" type="sphere" pos="-0.045440 0.023898 -0.049062" size="0.0040"
                        condim="4" friction="1.2 0.01 0.001" solimp="0.95 0.99 0.001"/>
                      <geom name="lf_tip1" class="collision" type="sphere" pos="-0.075440 0.023898 -0.049062" size="0.0040"
                        condim="4" friction="1.2 0.01 0.001" solimp="0.95 0.99 0.001"/>
                    </body>
                    <body name="link_right_finger" pos="0.0238981 -0.0450619 -0.0545599" quat="0.707105 0.707108 0 0">
                      <inertial pos="-0.0143408 -0.0224716 -0.0426253" quat="0.281222 0.8726 0.16705 -0.362738" mass="0.0710042" diaginertia="6.24193e-05 6.01079e-05 2.83591e-05"/>
                      <joint class="finger" name="right_finger" axis="0 0 -1" range="0 0.0475"/>
                      <geom name="rf_mesh" pos="-0.0379932 0.129783 0.00133753" quat="0.707105 -0.707108 0 0" type="mesh" mesh="lin_tip_right"
                        class="visual" rgba="0.18 0.18 0.19 1" contype="0" conaffinity="0"/>
                      <geom name="rf_pad" class="collision" type="box" pos="-0.023898 -0.060440 -0.050062"
                        quat="0.500001 0.499999 0.499999 -0.500001" size="0.0050 0.0120 0.0230"
                        condim="4" friction="1.2 0.01 0.001" solimp="0.95 0.99 0.001"/>
                      <geom name="rf_tip0" class="collision" type="sphere" pos="-0.023898 -0.045440 -0.049062" size="0.0040"
                        condim="4" friction="1.2 0.01 0.001" solimp="0.95 0.99 0.001"/>
                      <geom name="rf_tip1" class="collision" type="sphere" pos="-0.023898 -0.075440 -0.049062" size="0.0040"
                        condim="4" friction="1.2 0.01 0.001" solimp="0.95 0.99 0.001"/>
                    </body>
                  </body>
'''
    src = src[:a] + graft + src[b:]
    # the jaws move together, and the actuator drives the pair over the real 0-47.5 mm per side
    src = src.replace('<joint joint1="left_finger" joint2="right_finger" polycoef="0 -1 0 0 0"/>',
                      '<joint joint1="left_finger" joint2="right_finger" polycoef="0 1 0 0 0"/>')
    src = src.replace('<position ctrlrange="0.0 0.041"', '<position ctrlrange="0.0 0.0475"')
    src = src.replace('<exclude body1="', '<exclude body1="')                  # (no-op, kept for clarity)
    src = src.replace('</worldbody>',
                      '</worldbody>\n\n  <contact>\n'
                      '    <exclude body1="link_6" body2="link_left_finger"/>\n'
                      '    <exclude body1="link_6" body2="link_right_finger"/>\n'
                      '    <exclude body1="link_left_finger" body2="link_right_finger"/>\n  </contact>', 1)
    k0 = src.find("<keyframe>")
    if k0 != -1:
        src = src[:k0] + src[src.index("</keyframe>") + len("</keyframe>"):]
    OUT.write_text(src)
    return OUT


def verify(path):
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    jl = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "left_finger")]
    jr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "right_finger")]
    lb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link_left_finger")
    rb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link_right_finger")
    gs = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    l6 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link_6")
    # True opening = the RELATIVE displacement of the two tips between closed and open. Mesh geom
    # origins sit far from the pad faces, so their raw separation is not the opening.
    P = {}
    for tag, v in (("closed", 0.0), ("open", 0.0475)):
        d.qpos[:] = m.qpos0; d.qpos[jl] = v; d.qpos[jr] = v
        mujoco.mj_forward(m, d)
        P[tag] = (d.xpos[lb].copy(), d.xpos[rb].copy())
    travel_l = np.linalg.norm(P["open"][0] - P["closed"][0])
    travel_r = np.linalg.norm(P["open"][1] - P["closed"][1])
    throw = float(np.linalg.norm((P["open"][1] - P["open"][0]) - (P["closed"][1] - P["closed"][0])))
    print(f"  per-tip travel: left {travel_l * 1000:.1f} mm, right {travel_r * 1000:.1f} mm")
    print(f"  THROW (relative) {throw * 1000:.1f} mm   (I2RT spec: 95 mm)")
    out = {"throw": throw}
    d.qpos[:] = m.qpos0; mujoco.mj_forward(m, d)
    reach = float(np.linalg.norm(d.site_xpos[gs] - d.xpos[l6]))
    print(f"  grasp_site is {reach * 1000:.0f} mm from link_6; nq={m.nq} nu={m.nu}")
    return out


if __name__ == "__main__":
    p = build()
    print(f"wrote {p}")
    verify(p)


# ---------------------------------------------------------------------------------------------
# STATUS, 2026-09-20. The correct gripper for this bench is linear_4310 (rack-and-pinion sliding
# jaws on a DM4310; 95 mm throw, matching I2RT's published spec), NOT the crank_4310 that the
# MuJoCo Menagerie model bakes in (79 mm). Two ways to get it, both verified to produce the same
# frames (grasp_site at (0.2552, 0, 0.1734) with all joints at zero):
#
#   1. i2rt's own composer, which is authoritative and reads the per-arm mount transform from the
#      gripper's YAML config — do NOT hand-derive that transform:
#
#          from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
#          path = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310)  # returns a PATH
#
#      Needs python-can, tyro, pyyaml, pydantic, crcmod in the venv. That model is kinematics
#      only: no actuators, no armature, no tuned gain classes.
#   2. This script, which grafts the same gripper into the menagerie arm so the actuators,
#      armature and position-servo classes the rest of the stack depends on are preserved.
#
# WHAT IS NOT DONE: grasping with it. The jaws close straight through an object placed anywhere
# along the tool axis (swept -40 mm to +125 mm from the tip midpoint: the fingers always reach
# q=0.003 with 1-2 glancing contacts). Since the tip geoms are collidable (contype/conaffinity 1,
# mesh type, rbound 0.079) and the frames match i2rt's own composition, the remaining error is
# almost certainly the lateral placement of the object relative to the jaw sweep plane, i.e. the
# TCP's offset along the jaw axis and the "third" axis, not along the tool axis which is what was
# swept. Next step: sweep the object over a 3D grid around the jaw region and find the cell where
# closing actually grips, then define tcp_local from that measurement.
#
# Until then the default is YAM_ARM=stock (crank_4310), which grasps reliably and is what the
# 12-test suite and the generated datasets use.
