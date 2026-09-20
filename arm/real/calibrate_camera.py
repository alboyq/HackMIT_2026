"""Calibrate the wrist camera's lens (OpenCV FISHEYE model) from a checkerboard. Read-only on the camera;
never touches the arm. Needed before the feed's face-size stop can be trusted: the benchmark is in DEGREES
and only a calibrated lens turns pixels into degrees.

  1. print a checkerboard (default 9x6 INNER corners), tape it flat to something rigid
  2. python calibrate_camera.py --square-mm 25
  3. move the BOARD (not the arm) around: centre, all four corners of the image, tilted, near and far.
     [space] grabs a view when the corners are found (they are drawn), [c] calibrates once you have >= 15,
     [q] quits. Spread matters more than count: a fisheye is only constrained where you showed it the board.
Writes arm/real/calibration/wrist_camera.json (K, D, size, rms). rms under ~0.7 px is good.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="/dev/video0")
ap.add_argument("--cols", type=int, default=9)
ap.add_argument("--rows", type=int, default=6)
ap.add_argument("--square-mm", type=float, default=25.0)
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=720)
ap.add_argument("--images", nargs="*", help="calibrate from saved images instead of the live camera")
args = ap.parse_args()

pattern = (args.cols, args.rows)
objp = np.zeros((1, args.cols * args.rows, 3), np.float64)
objp[0, :, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square_mm / 1000.0
obj_pts, img_pts, size = [], [], None


def find(gray):
    ok, c = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ok:
        c = cv2.cornerSubPix(gray, c, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4))
    return ok, c


def solve():
    K, D = np.zeros((3, 3)), np.zeros((4, 1))
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
    rms, K, D, _, _ = cv2.fisheye.calibrate(obj_pts, img_pts, size, K, D, flags=flags,
                                            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7))
    out = Path(__file__).resolve().parent / "calibration" / "wrist_camera.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"model": "opencv_fisheye", "width": size[0], "height": size[1], "K": K.tolist(),
                               "D": D.ravel().tolist(), "rms_px": float(rms), "views": len(obj_pts)}, indent=2))
    diag = 2 * np.degrees(np.hypot(size[0], size[1]) / 2 / K[0, 0])        # equidistant approximation
    print(f"rms {rms:.3f} px over {len(obj_pts)} views -> {out}\n  fx {K[0,0]:.1f} fy {K[1,1]:.1f} cx {K[0,2]:.1f} cy {K[1,2]:.1f}"
          f"\n  implied diagonal FOV ~{diag:.0f} deg (the lens says 150)")


if args.images:
    for p in args.images:
        g = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY)
        size = g.shape[::-1]
        ok, c = find(g)
        print(p, "ok" if ok else "no board")
        if ok:
            obj_pts.append(objp.copy())
            img_pts.append(c.reshape(1, -1, 2).astype(np.float64))
    solve()
else:
    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    while True:
        ok, f = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        size = g.shape[::-1]
        found, c = find(g)
        vis = f.copy()
        if found:
            cv2.drawChessboardCorners(vis, pattern, c, found)
        cv2.putText(vis, f"views {len(obj_pts)}  [space] grab  [c] calibrate  [q] quit", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("calibrate wrist camera", vis)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" ") and found:
            obj_pts.append(objp.copy())
            img_pts.append(c.reshape(1, -1, 2).astype(np.float64))
        elif k == ord("c") and len(obj_pts) >= 8:
            solve()
        elif k in (ord("q"), 27):
            break
