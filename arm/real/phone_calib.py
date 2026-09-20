"""iPhone -> robot base calibration from frames that are ALREADY captured. Offline: moves nothing, enables nothing.

What went wrong with the depth-only fit (rms 21 cm): the LiDAR grid is 256x192, the taped jaws are ~3 depth pixels
wide, and in mid-air the depth inside the mask is the wall behind them (0.5-3.5 m in one mask).

What this does instead - two measurements that ARE good, each pinning what the other cannot:
  1. PIXELS of the taped jaws (sharp, 960x720) against the FK position of the jaws  -> bearing constraints (PnP).
     Weak along the viewing direction: a handful of points inside a 20 cm cube barely constrain range.
  2. The TABLE PLANE from LiDAR (tens of thousands of clean points). The arm's base plate sits on that table, so in
     the base frame the table is the plane z = z_table. That pins camera tilt (2 DoF) and height - exactly the
     directions PnP is weak in. z_table is left free and reported: it should come out within a few cm of 0.
    python arm/real/phone_calib.py [capture_dir] [table_frame_dir]
Writes arm/real/calibration/phone_base_from_camera.json

RESULT on the 9 frames of 2026-09-20 02:35 (/home/asus/calib_capture): NOT USABLE. 42 px rms (~4 cm at the jaws),
table 11 cm above the base, and the base itself reprojects ~90 px from where it is in the picture. Eight noisy
detections (a person moving behind the arm, both strips merged into one blob, jaws high in the air) inside a 20 cm
cube are not enough. A capture that WILL work: >= 12 poses, jaws LOW over the table and spread across the whole
reachable table area (then the LiDAR depth of the tape is clean too), nobody moving in the background.
"""
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from arm.gripper_observer import find_gripper  # noqa: E402

CAP = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/asus/calib_capture")
TABLE = Path(sys.argv[2] if len(sys.argv) > 2 else "/home/asus/frame_rl")
OUT = Path(__file__).resolve().parent / "calibration" / "phone_base_from_camera.json"
OFFSET = np.array([-1.4313, 0.0044, 0.0101, -0.0074, 0.0015, 1.3035])          # joint_map_measured.json
xml = next(p for p in (REPO / "third_party/mujoco_menagerie/i2rt_yam/yam.xml", Path("/home/asus/yam_simfirst/mujoco_menagerie/i2rt_yam/yam.xml")) if p.exists())
M = mujoco.MjModel.from_xml_path(str(xml)); D = mujoco.MjData(M)
SID = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")


def fk(q):
    D.qpos[:] = 0; D.qpos[:6] = q - OFFSET; mujoco.mj_forward(M, D)
    return D.site_xpos[SID].copy()


# ---------------------------------------------------------------- 1. jaw pixels + FK
obs, prev = [], None
for f in sorted(CAP.glob("pose_*.npz")):
    d = np.load(f); rgb = d["rgb"].astype(np.float32) / 255.0
    o = find_gripper(prev, rgb) if prev is not None else None
    prev = rgb
    if o is not None:
        obs.append(dict(name=f.name, px=np.asarray(o.centroid_px, float), t=fk(d["q_meas"]), K=d["K_rgb"], w=rgb.shape[1]))
fx, fy, cx, cy = obs[0]["K"]
ks = round(obs[0]["w"] / (2.0 * cx) * 4) / 4                                   # K_rgb is for the 1920 frame; the capture stores 960
fx, fy, cx, cy = ks * fx, ks * fy, ks * cx, ks * cy
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

# ---------------------------------------------------------------- 2. table plane from LiDAR (camera frame)
dep, conf = np.load(TABLE / "depth.npy"), np.load(TABLE / "conf.npy")
meta = json.loads((TABLE / "meta.json").read_text()); kd = meta["K_depth"]
v, u = np.mgrid[0:dep.shape[0], 0:dep.shape[1]]
ok = np.isfinite(dep) & (dep > 0.2) & (dep < 1.6) & (conf >= 2) & (v > dep.shape[0] * 0.45)     # lower half of the view = the table
P = np.stack([(u[ok] - kd["cx"]) / kd["fx"] * dep[ok], (v[ok] - kd["cy"]) / kd["fy"] * dep[ok], dep[ok]], 1)
rng = np.random.default_rng(0); best = (0, None)
for _ in range(400):
    a, b, c = P[rng.choice(len(P), 3, replace=False)]
    n = np.cross(b - a, c - a)
    if np.linalg.norm(n) < 1e-9:
        continue
    n /= np.linalg.norm(n); inl = np.abs((P - a) @ n) < 0.006
    if inl.sum() > best[0]:
        best = (int(inl.sum()), inl)
Q = P[best[1]]; c0 = Q.mean(0); n_c = np.linalg.svd(Q - c0)[2][2]
if n_c @ c0 > 0:
    n_c = -n_c                                                                  # make the normal point back toward the camera = table 'up'
d_c = -n_c @ c0                                                                 # plane: n_c . p + d_c = 0 ; d_c = camera height above the table
print(f"table plane: {best[0]}/{len(P)} LiDAR points within 6 mm; camera is {d_c:.3f} m above the table")


def unpack(x):
    Rc, _ = cv2.Rodrigues(x[:3]); return Rc, x[3:6]                             # camera_from_base


def residual(x, sel):
    Rc, tc = unpack(x); r = []
    for o in sel:
        p = Rc @ o["t"] + tc
        r += [fx * p[0] / p[2] + cx - o["px"][0], fy * p[1] / p[2] + cy - o["px"][1]]
    up_c = Rc @ np.array([0, 0, 1.0])                                           # base 'up' seen from the camera must be the table normal
    r += list(400.0 * (up_c - n_c))
    cam_h = (-Rc.T @ tc)[2]                                                     # camera height in the base frame
    r.append(400.0 * (cam_h - x[6] - d_c))                                      # = table height (x[6]) + camera height above the table
    r.append(40.0 * x[6])                                                       # weak prior: the base plate sits on the table
    return np.array(r)


def solve(sel):
    ok_, rv, tv = cv2.solvePnP(np.array([o["t"] for o in sel]), np.array([o["px"] for o in sel]), K, None, flags=cv2.SOLVEPNP_EPNP)
    s = least_squares(residual, np.concatenate([rv.ravel(), tv.ravel(), [0.0]]), args=(sel,), loss="soft_l1", f_scale=6.0)
    e = residual(s.x, sel)[:2 * len(sel)].reshape(-1, 2)
    return s.x, np.linalg.norm(e, axis=1)


x, err = solve(obs)
print("reprojection, all poses (px):", dict(zip([o["name"][5:7] for o in obs], np.round(err).astype(int).tolist())))
good = [o for o, e in zip(obs, err) if e < max(20.0, 2.5 * np.median(err))]
x, err = solve(good)
Rc, tc = unpack(x); T = np.eye(4); T[:3, :3] = Rc.T; T[:3, 3] = -Rc.T @ tc
rms = float(np.sqrt(np.mean(err ** 2))); rng_m = float(np.mean([(Rc @ o["t"] + tc)[2] for o in good]))
print(f"kept {len(good)}/{len(obs)} poses {[o['name'][5:7] for o in good]}: reprojection {np.round(err, 1).tolist()} px, rms {rms:.1f} px = {1000 * rms * rng_m / fx:.0f} mm at the jaws")
print(f"camera in the base frame: {np.round(T[:3, 3], 3).tolist()} m;  table height in the base frame: {1000 * x[6]:+.0f} mm")
b = Rc @ np.zeros(3) + tc
print(f"check: the robot's base origin should appear at pixel ({fx * b[0] / b[2] + cx:.0f}, {fy * b[1] / b[2] + cy:.0f}) of the 960x720 image")
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps({"base_from_camera": T.tolist(), "table_z_base_m": float(x[6]), "K_rgb_960": [fx, fy, cx, cy], "reproj_rms_px": rms,
                           "poses_used": [o["name"] for o in good], "method": "jaw pixels + FK (PnP) with the LiDAR table plane as a constraint",
                           "valid_while": "the phone has not been moved since the capture"}, indent=2))
print("->", OUT)
