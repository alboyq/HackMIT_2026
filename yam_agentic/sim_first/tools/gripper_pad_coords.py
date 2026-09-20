"""Derive the collision-pad placement that make_arm_linear4310.py hard-codes.

Run this if the gripper meshes ever change; paste the output into the graft.

    python tools/gripper_pad_coords.py

The pads replace the fingertip MESH collision (a convex-hull wedge) with primitives on the true
inner faces, exactly as mujoco_menagerie does for the crank_4310 gripper. Layout is measured
from the tip meshes at q=0, where the two faces meet at x=0.

TRAP: look these up on bodies `tip_left`/`tip_right` (the standalone gripper's names). In the
grafted arm they are `link_left_finger`/`link_right_finger`, but the LOCAL frames are identical,
so coordinates computed here transfer verbatim. Using the wrong name makes mj_name2id return -1,
which silently resolves to the last body and puts both pads in one finger's frame.
"""
import numpy as np
import mujoco
import _paths

m = mujoco.MjModel.from_xml_string(_paths.load_gripper_xml(), {})
d = mujoco.MjData(m)
d.qpos[:] = 0.0
mujoco.mj_forward(m, d)

BOX = dict(hx=0.005, hy=0.012, hz=0.023, z=0.115)   # half-extents, and depth of the pad centre
SPH = [(0.100, 0.004), (0.130, 0.004)]              # (depth, radius) — these define the TCP


def to_local(body, p_world):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    assert bid >= 0, f"no body {body} — see the TRAP note in the docstring"
    Rb, tb = d.xmat[bid].reshape(3, 3), d.xpos[bid]
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, (Rb.T @ np.eye(3)).flatten())
    return Rb.T @ (np.asarray(p_world, float) - tb), q


for body, s in (("tip_left", -1.0), ("tip_right", +1.0)):
    p, q = to_local(body, (s * BOX["hx"], 0.0, BOX["z"]))
    print(f'{body:11s} BOX pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" '
          f'quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}" '
          f'size="{BOX["hx"]:.4f} {BOX["hy"]:.4f} {BOX["hz"]:.4f}"')
    for z, r in SPH:
        ps, _ = to_local(body, (s * r, 0.0, z))
        print(f'{body:11s} SPH pos="{ps[0]:.6f} {ps[1]:.6f} {ps[2]:.6f}" size="{r:.4f}"')
print("""
Check after grafting: aperture must be 0.0 mm at q=0 and 95.0 mm at q=0.0475 (the I2RT spec),
and ToolFrame should report sphere_pads=True with the TCP ~29.7 mm back from grasp_site.""")
