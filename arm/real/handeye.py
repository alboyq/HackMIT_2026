"""Where the wrist camera sits relative to the gripper (T_tcp_cam). Needed to turn 'the strawberry is at
this pixel' into 'the strawberry is at x, y on the table' - the position the learned policies consume.

NOMINAL: the mount as MEASURED with a ruler on the real gripper (76 mm back from the tips, 59 mm above the jaws,
45 deg; the sim's is 6.5 cm / 6 cm / 33 deg). Good to a couple of cm if the real bracket matches; the perception-noise model the policies were
trained with (12 mm bias, scaled by range) was chosen to absorb roughly that.

REFINED: hand-eye (Park-Martin, park_martin() below) from N poses. IMPORTANT - this needs NO powered motion: with the motors
DISABLED (zero torque, as in the joint-map session of commit 2ee81bf) a person moves the arm by hand to ~15
poses that all see a fixed checkerboard; at each pose record the six joint angles and one image.
    ../../.venv-vision/bin/python handeye.py solve samples.json      # samples: [{"q": [6 joint angles, MODEL convention], "image": "x.jpg"}]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# camera frame (OpenCV: x right, y down, z forward) expressed in the link_6 / tool frame (z along the claws).
# MEASURED on the real gripper with a ruler, 2026-09-20: the lens is 76 mm back from the claw tips, 59 mm above the
# middle of the jaws, pitched 45 deg toward the claw axis. (The sim mount is 65 mm off-axis, 60 mm up the tool, 33 deg:
# close, which is why the sim's view resembles the real one.) Claw tips are 0.146 m along link_6 z in the model.
# Which SIDE of the jaws 'above' is cannot be told from a ruler; it is taken as the sim's side (-y), which matches the
# real picture: the jaws appear at the BOTTOM of the frame. Ruler accuracy ~ +-5 mm / +-5 deg: refine with solve().
TIP_Z, BACK_M, ABOVE_M, PITCH_DEG = 0.146, 0.076, 0.059, 45.0
_p = np.radians(PITCH_DEG)
_fwd = np.array([0.0, np.sin(_p), np.cos(_p)])        # camera looks along this, in link_6 axes
_x = np.array([1.0, 0.0, 0.0])
_y = np.cross(_fwd, _x)
NOMINAL_T_TOOL_CAM = np.eye(4)
NOMINAL_T_TOOL_CAM[:3, :3] = np.column_stack([_x, _y / np.linalg.norm(_y), _fwd / np.linalg.norm(_fwd)])
NOMINAL_T_TOOL_CAM[:3, 3] = [0.0, -ABOVE_M, TIP_Z - BACK_M]
OUT = Path(__file__).resolve().parent / "calibration" / "T_tool_cam.json"


def load():
    """The refined transform if it exists, else the nominal one (and say which)."""
    if OUT.exists():
        return np.array(json.loads(OUT.read_text())["T_tool_cam"]), True
    return NOMINAL_T_TOOL_CAM.copy(), False


def _log(R):
    a = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (0.5 if a < 1e-9 else a / (2.0 * np.sin(a)))


def park_martin(Rg, tg, Rt, tt):
    """Hand-eye AX = XB (Park & Martin 1994), in numpy: this OpenCV 5 build ships without calibrateHandEye.
    Rg, tg: tool in base per pose. Rt, tt: target in camera per pose. Returns X = T_tool_cam."""
    Tg = [np.block([[R, np.reshape(t, (3, 1))], [np.zeros((1, 3)), 1.0]]) for R, t in zip(Rg, tg)]
    Tc = [np.block([[R, np.reshape(t, (3, 1))], [np.zeros((1, 3)), 1.0]]) for R, t in zip(Rt, tt)]
    AB = [(np.linalg.inv(Tg[j]) @ Tg[i], Tc[j] @ np.linalg.inv(Tc[i])) for i in range(len(Tg)) for j in range(i + 1, len(Tg))]
    M = sum(np.outer(_log(B[:3, :3]), _log(A[:3, :3])) for A, B in AB)
    w, V = np.linalg.eigh(M.T @ M)
    R = V @ np.diag(w ** -0.5) @ V.T @ M.T
    C = np.vstack([A[:3, :3] - np.eye(3) for A, _ in AB])
    d = np.concatenate([R @ B[:3, 3] - A[:3, 3] for A, B in AB])
    X = np.eye(4); X[:3, :3] = R; X[:3, 3] = np.linalg.lstsq(C, d, rcond=None)[0]
    return X


def selftest():
    """Synthetic poses with a known camera mount -> the solver must recover it."""
    rng = np.random.default_rng(0)
    X = NOMINAL_T_TOOL_CAM.copy(); X[:3, 3] += [0.01, -0.005, 0.008]
    target = np.eye(4); target[:3, 3] = [0.4, 0.0, 0.0]
    Rg, tg, Rt, tt = [], [], [], []
    for _ in range(15):
        Tb = np.eye(4); Tb[:3, :3] = cv2.Rodrigues(rng.normal(0, 0.5, 3))[0]; Tb[:3, 3] = rng.normal([0.3, 0, 0.3], 0.05)
        Tc = np.linalg.inv(Tb @ X) @ target
        Rg.append(Tb[:3, :3]); tg.append(Tb[:3, 3]); Rt.append(Tc[:3, :3]); tt.append(Tc[:3, 3] + rng.normal(0, 0.0005, 3))
    got = park_martin(Rg, tg, Rt, tt)
    e_t = np.linalg.norm(got[:3, 3] - X[:3, 3]); e_r = np.degrees(np.linalg.norm(_log(got[:3, :3].T @ X[:3, :3])))
    print(f"hand-eye self-test: translation error {1000*e_t:.2f} mm, rotation error {e_r:.3f} deg ->", "PASS" if e_t < 0.003 and e_r < 0.5 else "FAIL")
    return e_t < 0.003 and e_r < 0.5


def solve(samples_path, fk, cam_K, cam_D, pattern=(9, 6), square_m=0.025):
    """fk(q) -> 4x4 pose of the tool (link_6) in the robot base, from the MuJoCo model (arm.ik)."""
    objp = np.zeros((pattern[0] * pattern[1], 1, 3), np.float64)
    objp[:, 0, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_m
    Rg, tg, Rt, tt = [], [], [], []
    for s in json.loads(Path(samples_path).read_text()):
        gray = cv2.cvtColor(cv2.imread(s["image"]), cv2.COLOR_BGR2GRAY)
        ok, c = cv2.findChessboardCorners(gray, pattern)
        if not ok:
            continue
        und = cv2.fisheye.undistortPoints(c.astype(np.float64), cam_K, cam_D)
        ok, rvec, tvec = cv2.solvePnP(objp, und, np.eye(3), None)
        if not ok:
            continue
        T = fk(np.asarray(s["q"], dtype=float))
        Rg.append(T[:3, :3]); tg.append(T[:3, 3]); Rt.append(cv2.Rodrigues(rvec)[0]); tt.append(tvec.ravel())
    if len(Rg) < 8:
        raise SystemExit(f"only {len(Rg)} usable poses; need >= 8, spread over rotations about all three axes")
    T = park_martin(Rg, tg, Rt, tt)
    d = np.linalg.norm(T[:3, 3] - NOMINAL_T_TOOL_CAM[:3, 3])
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"T_tool_cam": T.tolist(), "poses": len(Rg), "shift_from_nominal_m": float(d)}, indent=2))
    print(f"{len(Rg)} poses -> {OUT}; camera is {100*d:.1f} cm from the nominal mount")
    return T


if __name__ == "__main__":
    T, refined = load()
    print("T_tool_cam:", "REFINED" if refined else "NOMINAL (sim mount)")
    print(np.round(T, 3))
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        raise SystemExit(0 if selftest() else 1)
    if len(sys.argv) > 2 and sys.argv[1] == "solve":
        calib = Path(__file__).resolve().parent / "calibration" / "wrist_camera.json"
        if not calib.exists():
            raise SystemExit("calibrate the lens first (calibrate_camera.py): hand-eye on an uncalibrated fisheye is meaningless")
        import os
        import mujoco                                        # .venv-vision has both mujoco and cv2
        c = json.loads(calib.read_text())
        root = Path(os.environ.get("YAM_MENAGERIE", Path(__file__).resolve().parents[2] / "third_party" / "mujoco_menagerie"))
        xml = next(p for p in (root / "yam.xml", root / "i2rt_yam" / "yam.xml") if p.exists())
        M = mujoco.MjModel.from_xml_path(str(xml)); D = mujoco.MjData(M)
        tool = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, "link_6")

        def fk(q):
            D.qpos[:] = 0.0; D.qpos[:6] = q
            mujoco.mj_kinematics(M, D)
            T = np.eye(4); T[:3, :3] = D.xmat[tool].reshape(3, 3); T[:3, 3] = D.xpos[tool]
            return T

        solve(sys.argv[2], fk, np.array(c["K"]), np.array(c["D"]))
