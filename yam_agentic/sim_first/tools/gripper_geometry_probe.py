"""Why the linear_4310 gripper did not grasp: hull geometry vs surface geometry.

MuJoCo collides a `<geom type="mesh">` as its CONVEX HULL. The linear_4310 fingertips are long
tapered blades, so each hull is a solid wedge that fills the throat between the jaws. The mesh
SURFACES are a correct 95 mm parallel gripper. Both measurements below are right; they measure
different things, and confusing them is what made this bug expensive.

    python tools/gripper_geometry_probe.py          # both measurements, side by side

Reproduces (2026-09-20, i2rt's own standalone linear_4310.xml, before the fix):
  surface: inner faces meet at x~0 when closed, sit at x ~ +-47.5 mm when open  -> 95 mm stroke
  hull   : a V that is ~88 mm wide at the fingertips and ~32 mm at the root     -> wedge
"""
import numpy as np
import mujoco
import _paths

np.seterr(all="ignore")   # MuJoCo leaves unused mesh_vert rows uninitialised; harmless here

m = mujoco.MjModel.from_xml_string(_paths.load_gripper_xml(), {})
d = mujoco.MjData(m)
OPEN = float(m.jnt_range[0][1])

def tip_vertices(q):
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    out = {}
    for gid in range(m.ngeom):
        mid = m.geom_dataid[gid]
        if mid < 0:
            continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mid)
        if not name or not name.startswith("tip"):
            continue
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        R = d.geom_xmat[gid].reshape(3, 3)
        out[name] = m.mesh_vert[va:va + vn].astype(np.float64) @ R.T + d.geom_xpos[gid]
    return out

print("=" * 78)
print("A. SURFACE geometry — the mesh vertices, i.e. what the cameras render")
print("=" * 78)
print(f"{'z (mm)':>8}  {'left inner face x':>18}  {'right inner face x':>19}  {'gap (mm)':>9}")
for state, q in (("CLOSED", 0.0), ("OPEN", OPEN)):
    V = tip_vertices(q)
    print(f"-- {state} (q={q:.4f})")
    for z0 in (0.090, 0.110, 0.130, 0.145):
        bl = V["tip_left"][np.abs(V["tip_left"][:, 2] - z0) < 0.005]
        br = V["tip_right"][np.abs(V["tip_right"][:, 2] - z0) < 0.005]
        if len(bl) < 5 or len(br) < 5:
            continue
        xl, xr = np.quantile(bl[:, 0], 0.98), np.quantile(br[:, 0], 0.02)
        print(f"{z0*1000:8.0f}  {xl*1000:18.2f}  {xr*1000:19.2f}  {(xr-xl)*1000:9.2f}")

print()
print("=" * 78)
print("B. HULL geometry — what MuJoCo actually collides, probed with a 1 mm sphere")
print("=" * 78)
xml = _paths.load_gripper_xml().replace("</worldbody>", """
    <body name="probe" mocap="true" pos="0 0 0.11"><geom type="sphere" size="0.001"/></body>
  </worldbody>""")
mp_ = mujoco.MjModel.from_xml_string(xml, {})
dp = mujoco.MjData(mp_)
mid = mp_.body_mocapid[mujoco.mj_name2id(mp_, mujoco.mjtObj.mjOBJ_BODY, "probe")]
xs = np.arange(-0.080, 0.0801, 0.004)
for state, q in (("CLOSED", 0.0), ("OPEN", OPEN)):
    print(f"-- {state} (q={q:.4f})   '#' = the probe collides; x from -80 to +80 mm")
    for z in np.arange(0.055, 0.1501, 0.010):
        row = []
        for x in xs:
            dp.qpos[:] = q
            dp.mocap_pos[mid] = (x, 0.0, z)
            mujoco.mj_forward(mp_, dp)
            row.append(dp.ncon > 0)
        free = np.array(row) == False
        c = len(xs) // 2
        if free[c]:
            lo = hi = c
            while lo > 0 and free[lo - 1]:
                lo -= 1
            while hi < len(xs) - 1 and free[hi + 1]:
                hi += 1
            span = f"{(xs[hi]-xs[lo])*1000:4.0f} mm"
        else:
            span = "   --"
        print(f"  z={z*1000:5.0f}  " + "".join("#" if v else "." for v in row) + f"  open {span}")
print("""
Reading it: B is a V (wide at the fingertips, narrow at the root) while A is parallel. The V is
the convex hull. Closing it drives an object OUT of the jaws instead of pinching it — measured
separately in tools/gripper_grasp_rig.py. Fix: mesh geoms visual-only, collision on primitive
pads at A's inner faces (see sim/make_arm_linear4310.py).""")
