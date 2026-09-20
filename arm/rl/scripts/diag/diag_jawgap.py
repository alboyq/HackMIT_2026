"""Measure the sim jaws the way the real ones were measured with calipers: the clear gap between the two
gripping faces at the tip, the middle and the base, fully closed and fully open."""
import mujoco
import numpy as np

from arm.ik.scene import YamScene

M = YamScene().model
D = mujoco.MjData(M)
fingers = [j for j in range(M.njnt) if M.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE]
print("slide joints:", [mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_JOINT, j) for j in fingers], [M.jnt_range[j].tolist() for j in fingers])
l6 = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, "link_6")
pads = [g for g in range(M.ngeom) if M.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE and M.geom_bodyid[g] > l6 and M.geom_contype[g]]
for frac in (0.0, 0.5, 1.0):
    D.qpos[:] = 0
    for j in fingers:
        lo, hi = M.jnt_range[j]; a, b = (lo, hi) if abs(lo) < abs(hi) else (hi, lo); D.qpos[M.jnt_qposadr[j]] = a + frac * (b - a)
    mujoco.mj_forward(M, D)
    R, o = D.xmat[l6].reshape(3, 3), D.xpos[l6]
    loc = np.array([R.T @ (D.geom_xpos[g] - o) for g in pads]); rad = M.geom_size[pads, 0]
    body = M.geom_bodyid[pads]; b1, b2 = sorted(set(body.tolist()))[:2]
    ax = int(np.argmax(np.abs(loc[body == b1].mean(0) - loc[body == b2].mean(0))))
    zs = np.round(loc[:, 2], 3)
    print(f"finger joints at {frac:.0%} of range (jaw axis = link_6 {'xyz'[ax]}), pad radius {1000*rad[0]:.1f} mm:")
    for z in sorted(set(zs), reverse=True):
        a, b = loc[(zs == z) & (body == b1), ax], loc[(zs == z) & (body == b2), ax]
        if len(a) and len(b):
            print(f"   z = {1000*z:5.1f} mm along the tool: clear gap {1000*(abs(a.mean() - b.mean()) - 2*rad[0]):6.1f} mm")
