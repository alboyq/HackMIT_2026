"""Face landmarks (MediaPipe, 478 pts incl. iris), rigid head pose via PnP against the
MediaPipe canonical face model, and a per-eye 3D eyeball model.

Coordinate convention throughout: OpenCV camera frame — x right, y down, z forward
(away from the camera, into the scene). Units: millimetres.
"""
from __future__ import annotations
import os, time
from dataclasses import dataclass, field
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(os.path.dirname(HERE), "models")

# --- canonical model ----------------------------------------------------------
def load_canonical(path=os.path.join(MODELS, "canonical_face_model.obj")) -> np.ndarray:
    """468x3 model in mm, converted from MediaPipe's (x right, y up, z toward camera, cm)
    to OpenCV camera convention so a frontal face gives R ~ identity."""
    v = [list(map(float, l.split()[1:4])) for l in open(path) if l.startswith("v ")]
    v = np.asarray(v, dtype=np.float64) * 10.0
    v[:, 1] *= -1.0
    v[:, 2] *= -1.0
    return v

# Rigid landmarks (skull-attached; avoid lips, eyelids, jaw-hinge-driven chin points)
RIGID = np.array([
    1, 4, 5, 6, 8, 9, 10, 151, 168, 195, 197,          # nose bridge / forehead midline
    33, 133, 362, 263, 130, 359, 243, 463,             # eye corners
    127, 356, 162, 389, 234, 454, 93, 323, 132, 361,   # temples / cheeks
    21, 251, 54, 284, 103, 332, 67, 297, 109, 338,     # upper skull rim
    111, 340, 117, 346, 118, 347, 50, 280, 101, 330,   # cheekbones
    36, 266, 205, 425, 187, 411,                        # zygomatic
], dtype=np.int64)

# ETH-XGaze / MPIIFaceGaze face centre = mean of 4 eye corners + 2 mouth corners
CENTER6 = np.array([33, 133, 362, 263, 61, 291])
R_EYE = dict(outer=33, inner=133, iris=468, up=159, down=145)
L_EYE = dict(outer=263, inner=362, iris=473, up=386, down=374)
EYEBALL_RADIUS_MM = 12.0
EYEBALL_DEPTH_MM = 12.5     # distance from eye-corner midpoint back to eyeball centre


@dataclass
class Intrinsics:
    fx: float; fy: float; cx: float; cy: float
    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)
    @staticmethod
    def from_hfov(width: int, height: int, hfov_deg: float) -> "Intrinsics":
        f = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
        return Intrinsics(f, f, width / 2, height / 2)


@dataclass
class FaceState:
    t: float
    landmarks: np.ndarray          # 478x2 pixel coords
    lm_z: np.ndarray               # 478 relative depth from MediaPipe (unitless-ish)
    R: np.ndarray                  # 3x3 head rotation (model -> camera)
    tvec: np.ndarray               # 3   model origin in camera frame, mm
    center: np.ndarray             # 3   XGaze face centre in camera frame, mm
    eye_r: np.ndarray              # 3   right eyeball centre, camera frame, mm
    eye_l: np.ndarray
    iris_r: np.ndarray             # 3   right iris centre on eyeball surface, mm
    iris_l: np.ndarray
    geo_gaze_r: np.ndarray         # 3   unit vector eyeball->iris (geometric gaze, right)
    geo_gaze_l: np.ndarray
    blink: float                   # 0..1 max blink score
    eye_open_r: float; eye_open_l: float   # eyelid aperture / eye width
    reproj_err: float
    blendshapes: dict = field(default_factory=dict)

    @property
    def head_pose_deg(self):
        """(pitch, yaw, roll) in degrees for display, from R."""
        sy = np.hypot(self.R[0, 0], self.R[1, 0])
        pitch = np.degrees(np.arctan2(-self.R[2, 1], self.R[2, 2]))
        yaw = np.degrees(np.arctan2(self.R[2, 0], sy))
        roll = np.degrees(np.arctan2(self.R[1, 0], self.R[0, 0]))
        return float(pitch), float(-yaw), float(roll)


class FaceTracker:
    def __init__(self, intrinsics: Intrinsics, model_path=os.path.join(MODELS, "face_landmarker.task")):
        self.K = intrinsics
        self.model3d = load_canonical()
        opts = vision.FaceLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO, num_faces=1,
            output_facial_transformation_matrixes=False, output_face_blendshapes=True,
            min_face_detection_confidence=0.5, min_tracking_confidence=0.5)
        self._lm = vision.FaceLandmarker.create_from_options(opts)
        self._rvec = None
        self._tvec = None
        self._t0 = time.perf_counter()

    def set_intrinsics(self, K: Intrinsics):
        self.K = K
        self._rvec = self._tvec = None

    # -- main entry ---------------------------------------------------------
    def process(self, frame_bgr: np.ndarray, t: float) -> FaceState | None:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((t - self._t0) * 1000)
        res = self._lm.detect_for_video(img, ts_ms)
        if not res.face_landmarks:
            self._rvec = self._tvec = None
            return None
        lm = res.face_landmarks[0]
        pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float64)
        z = np.array([p.z for p in lm], dtype=np.float64)
        bs = {b.category_name: b.score for b in res.face_blendshapes[0]} if res.face_blendshapes else {}

        R, tvec, err = self._pnp(pts)
        if R is None:
            return None
        center = (R @ self.model3d[CENTER6].T + tvec[:, None]).mean(axis=1)

        eye_r, iris_r, g_r = self._eye(pts, R, tvec, R_EYE)
        eye_l, iris_l, g_l = self._eye(pts, R, tvec, L_EYE)

        def aperture(e):
            up, dn, o, i = pts[e["up"]], pts[e["down"]], pts[e["outer"]], pts[e["inner"]]
            return float(np.linalg.norm(up - dn) / max(np.linalg.norm(o - i), 1e-6))

        blink = max(bs.get("eyeBlinkLeft", 0.0), bs.get("eyeBlinkRight", 0.0))
        return FaceState(t=t, landmarks=pts, lm_z=z, R=R, tvec=tvec, center=center,
                         eye_r=eye_r, eye_l=eye_l, iris_r=iris_r, iris_l=iris_l,
                         geo_gaze_r=g_r, geo_gaze_l=g_l, blink=float(blink),
                         eye_open_r=aperture(R_EYE), eye_open_l=aperture(L_EYE),
                         reproj_err=err, blendshapes=bs)

    # -- head pose ------------------------------------------------------------
    def _pnp(self, pts):
        obj = self.model3d[RIGID]
        img = pts[RIGID]
        K = self.K.K
        dist = np.zeros(5)
        if self._rvec is not None:
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, self._rvec, self._tvec,
                                          useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        else:
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
            if ok:
                ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, rvec, tvec,
                                              useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or tvec[2] <= 0:
            self._rvec = self._tvec = None
            return None, None, np.inf
        self._rvec, self._tvec = rvec, tvec
        R, _ = cv2.Rodrigues(rvec)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        err = float(np.sqrt(((proj[:, 0, :] - img) ** 2).sum(1).mean()))
        return R, tvec.reshape(3), err

    # -- eyeball model ----------------------------------------------------------
    def _eye(self, pts, R, tvec, e):
        """Eyeball centre from the rigid head pose; iris centre by intersecting the
        camera ray through the 2D iris landmark with the eyeball sphere."""
        corners = self.model3d[[e["outer"], e["inner"]]]
        mid = corners.mean(axis=0)
        # Push back along the head's own +z (OpenCV convention: forward = into the head
        # when facing the camera, since model z was flipped to point away from camera).
        center_model = mid + np.array([0.0, 0.0, EYEBALL_DEPTH_MM])
        eye_c = R @ center_model + tvec
        # ray through iris pixel
        u, v = pts[e["iris"]]
        d = np.array([(u - self.K.cx) / self.K.fx, (v - self.K.cy) / self.K.fy, 1.0])
        d /= np.linalg.norm(d)
        # sphere intersection: |s*d - c|^2 = r^2
        b = -2 * d @ eye_c
        c = eye_c @ eye_c - EYEBALL_RADIUS_MM ** 2
        disc = b * b - 4 * c
        if disc < 0:
            # ray misses sphere (landmark noise): take the closest point on the ray
            s = d @ eye_c
            iris = s * d
            g = iris - eye_c
        else:
            s = (-b - np.sqrt(disc)) / 2      # near intersection (front of eyeball)
            iris = s * d
            g = iris - eye_c
        g /= max(np.linalg.norm(g), 1e-9)
        return eye_c, iris, g


def estimate_focal_from_distance(tracker: FaceTracker, pts: np.ndarray, distance_mm: float,
                                 width: int, height: int) -> float:
    """Solve for the focal length that makes the PnP face-centre distance equal the
    user-reported viewing distance. PnP depth scales ~linearly with f, so a short
    secant iteration converges."""
    f = tracker.K.fx
    for _ in range(12):
        K = Intrinsics(f, f, width / 2, height / 2)
        obj = tracker.model3d[RIGID]
        ok, rvec, tvec = cv2.solvePnP(obj, pts[RIGID], K.K, np.zeros(5), flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            break
        R, _ = cv2.Rodrigues(rvec)
        center = (R @ tracker.model3d[CENTER6].T + tvec).mean(axis=1)
        d = float(np.linalg.norm(center))
        f_new = f * distance_mm / d
        if abs(f_new - f) < 0.5:
            f = f_new
            break
        f = f_new
    return float(f)
