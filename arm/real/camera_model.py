"""Pixels -> rays, for the real wrist camera.

Everything downstream works in ANGLES (the feed's benchmark is "the face is 50.9 deg wide"), because
angles survive a lens change and pixels do not. This class is the only place that knows about the lens.

Two modes:
  * calibrated   - load arm/real/calibration/wrist_camera.json written by calibrate_camera.py
                   (OpenCV fisheye model: K, D). Use this for anything that moves the arm.
  * UNCALIBRATED - an ideal equidistant fisheye with the FOV printed on the lens (r = f * theta).
                   Good to a few degrees near the centre. Fine for a dry run, not for feeding a person.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

CALIB = Path(__file__).resolve().parent / "calibration" / "wrist_camera.json"


class CameraModel:
    def __init__(self, width: int, height: int, calib: Path | None = CALIB, diag_fov_deg: float = 150.0):
        self.w, self.h = int(width), int(height)
        self.calibrated = bool(calib and Path(calib).exists())
        if self.calibrated:
            c = json.loads(Path(calib).read_text())
            sx, sy = self.w / c["width"], self.h / c["height"]          # calibration may be at another size
            self.K = np.array(c["K"], dtype=np.float64)
            self.K[0] *= sx
            self.K[1] *= sy
            self.D = np.array(c["D"], dtype=np.float64).reshape(4, 1)
            self.rms = float(c.get("rms_px", float("nan")))
        else:
            # equidistant: image radius = f * angle. The printed FOV is taken as the DIAGONAL.
            f = 0.5 * float(np.hypot(self.w, self.h)) / np.radians(diag_fov_deg / 2.0)
            self.K = np.array([[f, 0, self.w / 2.0], [0, f, self.h / 2.0], [0, 0, 1.0]])
            self.D = np.zeros((4, 1))
            self.rms = float("nan")

    def rays(self, pixels) -> np.ndarray:
        """Nx2 pixels -> Nx3 unit rays in the camera frame (x right, y down, z forward: OpenCV's)."""
        p = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
        u = cv2.fisheye.undistortPoints(p, self.K, self.D).reshape(-1, 2)      # normalised: tan(theta) * dir
        r = np.concatenate([u, np.ones((len(u), 1))], axis=1)
        return r / np.linalg.norm(r, axis=1, keepdims=True)

    def angle_between(self, p1, p2) -> float:
        a, b = self.rays([p1, p2])
        return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))

    def bearing(self, pixel) -> np.ndarray:
        """Tangent-plane bearing (x/z, y/z): the quantity the feed controller servos to zero.
        NOTE the sim camera frame is MuJoCo's (x right, y UP, looking down -z); flip y when bridging."""
        r = self.rays([pixel])[0]
        return np.array([r[0] / r[2], r[1] / r[2]])
