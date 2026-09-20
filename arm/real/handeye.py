"""Where the wrist camera sits relative to the gripper (T_tcp_cam). Needed to turn 'the strawberry is at
this pixel' into 'the strawberry is at x, y on the table' - the position the learned policies consume.

NOMINAL: the mount the sim uses (arm/ik/scene.py: 6.5 cm off the claw axis, 6 cm up the tool, tilted in
~33 deg). Good to a couple of cm if the real bracket matches; the perception-noise model the policies were
trained with (12 mm bias, scaled by range) was chosen to absorb roughly that.

REFINED: cv2.calibrateHandEye from N poses. IMPORTANT - this needs NO powered motion: with the motors
DISABLED (zero torque, as in the joint-map session of commit 2ee81bf) a person moves the arm by hand to ~15
poses that all see a fixed checkerboard; at each pose record the six joint angles and one image.
    python handeye.py solve samples.json      # samples: [{"q": [6 joint angles, MODEL convention], "image": "x.jpg"}]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# camera frame (OpenCV: x right, y down, z forward) expressed in the link_6 / tool frame (z along the claws)
_fwd = np.array([0.0, 0.545, 0.838])                  # camera looks along this, in link_6 axes
_x = np.array([1.0, 0.0, 0.0])
_y = np.cross(_fwd, _x)
NOMINAL_T_TOOL_CAM = np.eye(4)
NOMINAL_T_TOOL_CAM[:3, :3] = np.column_stack([_x, _y / np.linalg.norm(_y), _fwd / np.linalg.norm(_fwd)])
NOMINAL_T_TOOL_CAM[:3, 3] = [0.0, -0.065, 0.06]
OUT = Path(__file__).resolve().parent / "calibration" / "T_tool_cam.json"


def load():
    """The refined transform if it exists, else the nominal one (and say which)."""
    if OUT.exists():
        return np.array(json.loads(OUT.read_text())["T_tool_cam"]), True
    return NOMINAL_T_TOOL_CAM.copy(), False


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
    R, t = cv2.calibrateHandEye(Rg, tg, Rt, tt, method=cv2.CALIB_HAND_EYE_PARK)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t.ravel()
    d = np.linalg.norm(T[:3, 3] - NOMINAL_T_TOOL_CAM[:3, 3])
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"T_tool_cam": T.tolist(), "poses": len(Rg), "shift_from_nominal_m": float(d)}, indent=2))
    print(f"{len(Rg)} poses -> {OUT}; camera is {100*d:.1f} cm from the nominal mount")
    return T


if __name__ == "__main__":
    T, refined = load()
    print("T_tool_cam:", "REFINED" if refined else "NOMINAL (sim mount)")
    print(np.round(T, 3))
    if len(sys.argv) > 2 and sys.argv[1] == "solve":
        print("solve() needs fk and the lens calibration: call it from a session that has arm.ik loaded")
