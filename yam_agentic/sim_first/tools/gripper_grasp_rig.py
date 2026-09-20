"""Does the bare linear_4310 hold an object? Jaws point DOWN, gravity is the judge.

    python tools/gripper_grasp_rig.py            # sweep, plus /tmp/yam_rig.png

Measured 2026-09-20 on i2rt's unmodified gripper: a 25 mm cylinder at 125 mm depth was pushed
78 mm and expelled, IN ZERO GRAVITY, ending ncon=0 — the convex hull wedge (see
gripper_geometry_probe.py) squeezes objects out rather than pinching them.

Two traps this script exists to avoid repeating:
  * the fingers are coupled by <equality>, which mj_forward does NOT solve. Set BOTH joint7 and
    joint8 when you initialise, or one jaw stays shut and every number is meaningless.
  * the jaws are at +z of the gripper body, ~51-145 mm out. Probing -z finds empty space.
"""
import numpy as np
import mujoco
import _paths

BASE = _paths.load_gripper_xml()


def build(obj_r, depth, obj_h=0.09, z0=0.40):
    x = BASE.replace('<body name="gripper" pos="0 0 0"',
                     f'<body name="flip" pos="0 0 {z0}" quat="0 1 0 0"><body name="gripper" pos="0 0 0"')
    x = x.replace("</worldbody>", f"""
      </body>
    <body name="obj" pos="0 0 {z0 - depth}"><freejoint/>
      <geom name="obj" type="cylinder" size="{obj_r} {obj_h/2}" mass="0.15"
            friction="1.2 0.02 0.001" rgba="0.85 0.3 0.15 1"/></body>
    <light pos="0.3 -0.3 0.7" dir="-1 1 -1"/><light pos="-0.3 0.3 0.7" dir="1 -1 -1"/>
  </worldbody>""")
    x = x.replace("</mujoco>", """
  <actuator><position name="grip" joint="joint7" ctrlrange="0 0.0475" kp="800" kv="30"
    forcerange="-60 60"/></actuator></mujoco>""")
    return mujoco.MjModel.from_xml_string(
        x.replace("<compiler ", "<visual><global offwidth='500' offheight='500'/></visual>\n  <compiler "), {})


def trial(obj_r, depth):
    m = build(obj_r, depth)
    d = mujoco.MjData(m)
    q7 = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint7")]
    q8 = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint8")]
    oa = m.jnt_qposadr[m.body_jntadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "obj")]]
    d.qpos[q7] = d.qpos[q8] = 0.0475                      # BOTH — see module docstring
    d.ctrl[0] = 0.0475
    mujoco.mj_forward(m, d)
    g = m.opt.gravity.copy()
    m.opt.gravity[:] = 0
    for _ in range(300):
        mujoco.mj_step(m, d)
    z_open = d.qpos[oa + 2]
    d.ctrl[0] = 0.0
    for _ in range(800):
        mujoco.mj_step(m, d)
    z_shut = d.qpos[oa + 2]
    m.opt.gravity[:] = g
    for _ in range(1200):
        mujoco.mj_step(m, d)
    return z_open - z_shut, z_open - d.qpos[oa + 2], d.ncon


if __name__ == "__main__":
    diam = (0.025, 0.035, 0.045, 0.055, 0.065)
    print(f"{'depth mm':>8} " + "".join(f"{x*1000:>11.0f}" for x in diam) + "   <- object diameter (mm)")
    print(f"{'':>8} " + "  push = mm the object moved while CLOSING (zero g); drop = mm after 2.4 s of gravity")
    for depth in (0.145, 0.135, 0.125, 0.115, 0.105, 0.095):
        cells = []
        for dm in diam:
            push, drop, nc = trial(dm / 2, depth)
            cells.append("       HELD" if drop < 0.005 else f"{push*1000:5.0f}/{drop*1000:<5.0f}")
        print(f"{depth*1000:8.0f} " + "".join(cells))
