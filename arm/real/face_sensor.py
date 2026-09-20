"""The feed's only two inputs, from a real camera frame: WHERE the mouth is and HOW WIDE the face looks.

MediaPipe FaceLandmarker gives 478 landmarks; we use the inner lips for the mouth and the two cheek-edge
landmarks (234, 454) for the face width, then turn both into angles through CameraModel. The face width in
DEGREES is what the sim benchmark is recorded in (arm/rl/configs/feed_calibration.json).

Works across skin tones and backgrounds because the detector was trained for that; we verify rather than
assume (see tests/). If no face is found the sensor says so and the feed stands still - it never advances
blind.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from camera_model import CameraModel

MODEL = Path(__file__).resolve().parent / "models" / "face_landmarker.task"
LIP_UP, LIP_DOWN, CHEEK_R, CHEEK_L, CHIN, FOREHEAD, NOSE = 13, 14, 234, 454, 152, 10, 1
WIDTH_OVER_HEIGHT = 0.78       # average adult: cheek-to-cheek 145 mm over forehead-to-chin ~185 mm
MAX_YAW_ASYMMETRY = 0.30       # |left-right| / (left+right) of nose-to-cheek distances; ~25 deg of head turn


@dataclass
class FaceReading:
    seen: bool
    bearing: np.ndarray = None          # (x/z, y/z) of the mouth, camera frame
    width_deg: float = 0.0              # angular width of the face, cheek to cheek
    mouth_px: tuple = None
    cheeks_px: tuple = None
    mouth_open: float = 0.0             # lip gap / face height: >~0.06 means the mouth is open (ready to bite)
    pct_of_image_width: float = 0.0
    raw_width_deg: float = 0.0          # cheek to cheek as measured
    height_deg: float = 0.0
    facing: bool = True                 # False = head turned away: the feed must hold still
    asymmetry: float = 0.0


class FaceSensor:
    def __init__(self, cam: CameraModel, min_confidence: float = 0.3):
        if not MODEL.exists():
            raise FileNotFoundError(f"{MODEL} missing - see arm/real/README.md for the download")
        opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=3,
            min_face_detection_confidence=min_confidence, min_face_presence_confidence=min_confidence)
        self.det = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
        self.cam = cam

    # MediaPipe's built-in face finder is a SHORT-RANGE model: it wants the face to fill a good part of the
    # image. Through a 150-deg lens, a person 40 cm away is ~12% of the frame width and scores under threshold
    # (measured: 0 faces on a full 1280x720 frame, found at once in a crop). So search by zooming: full frame,
    # then tighter windows. The person sits straight ahead once the arm has cocked back, so centre first.
    WINDOWS = ((0.0, 0.0, 1.0, 1.0), (0.2, 0.1, 0.8, 0.9), (0.3, 0.2, 0.7, 0.8),
               (0.0, 0.0, 0.6, 0.7), (0.4, 0.0, 1.0, 0.7), (0.0, 0.3, 0.6, 1.0), (0.4, 0.3, 1.0, 1.0))

    def _landmarks(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        best = None
        for fx0, fy0, fx1, fy1 in self.WINDOWS:
            x0, y0, x1, y1 = int(fx0 * w), int(fy0 * h), int(fx1 * w), int(fy1 * h)
            crop = np.ascontiguousarray(cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
            res = self.det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
            for lm in res.face_landmarks:
                pts = np.array([[x0 + p.x * (x1 - x0), y0 + p.y * (y1 - y0)] for p in lm])
                width = float(np.linalg.norm(pts[CHEEK_L] - pts[CHEEK_R]))
                if best is None or width > best[0]:
                    best = (width, pts)
            if best is not None:
                break                                   # the first window that sees a face is enough
        return None if best is None else best[1]

    def read(self, frame_bgr) -> FaceReading:
        w = frame_bgr.shape[1]
        pts = self._landmarks(frame_bgr)
        if pts is None:
            return FaceReading(False)
        # the person being fed is the LARGEST face (closest, straight ahead once the arm has cocked back)
        mouth = 0.5 * (pts[LIP_UP] + pts[LIP_DOWN])
        c1, c2 = pts[CHEEK_R], pts[CHEEK_L]
        face_h = float(np.linalg.norm(pts[FOREHEAD] - pts[CHIN]))
        raw_w = self.cam.angle_between(c1, c2)
        h_deg = self.cam.angle_between(pts[FOREHEAD], pts[CHIN])
        # A TURNED head looks narrower, which reads as 'farther away' - the dangerous direction, because the arm
        # would keep coming. Height barely changes with a head turn, so take whichever says CLOSER. (An open
        # mouth lengthens the face and also reads closer: the safe side.)
        width_deg = max(raw_w, WIDTH_OVER_HEIGHT * h_deg)
        dl, dr = float(np.linalg.norm(pts[NOSE] - c1)), float(np.linalg.norm(pts[NOSE] - c2))
        asym = abs(dl - dr) / max(1.0, dl + dr)
        return FaceReading(True, self.cam.bearing(mouth), width_deg, tuple(mouth), (tuple(c1), tuple(c2)),
                           float(np.linalg.norm(pts[LIP_UP] - pts[LIP_DOWN]) / max(1.0, face_h)),
                           100.0 * float(np.linalg.norm(c1 - c2)) / w, raw_w, h_deg, asym <= MAX_YAW_ASYMMETRY, asym)

    @staticmethod
    def confirm(r: "FaceReading", people, min_conf: float = 0.25) -> "FaceReading":
        """TWO DETECTORS MUST AGREE. With the zoom search and a low threshold the landmark model found a 'face'
        in five decorative plates on a wall (two eyes and a mouth). A sensor that steers food at someone's head
        cannot be allowed that. The reading survives only if the open-vocabulary detector, in the same frame,
        also reports a human face / person whose box contains the mouth."""
        if not r.seen:
            return r
        mx, my = r.mouth_px
        for x1, y1, x2, y2, conf in (people or []):
            if conf >= min_conf and x1 <= mx <= x2 and y1 <= my <= y2:
                return r
        return FaceReading(False)

    @staticmethod
    def draw(frame, r: FaceReading, stop_deg: float | None = None, touch_deg: float | None = None):
        if not r.seen:
            cv2.putText(frame, "NO FACE - feed holds still", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return frame
        (a, b), m = r.cheeks_px, r.mouth_px
        cv2.line(frame, tuple(map(int, a)), tuple(map(int, b)), (0, 255, 255), 2)
        cv2.circle(frame, tuple(map(int, m)), 7, (0, 0, 255), -1)
        state = ""
        if stop_deg:
            state = "  STOP" if r.width_deg >= stop_deg else ("  CRAWL" if r.width_deg >= stop_deg - 10 else "  APPROACH")
        if not r.facing:
            state = "  HEAD TURNED - HOLD"
        txt = f"face {r.width_deg:.1f} deg (w {r.raw_width_deg:.1f}, h {r.height_deg:.1f}){state}"
        cv2.putText(frame, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(frame, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        t2 = f"mouth bearing ({r.bearing[0]:+.2f}, {r.bearing[1]:+.2f})   mouth {'OPEN' if r.mouth_open > 0.06 else 'closed'}"
        cv2.putText(frame, t2, (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(frame, t2, (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        return frame
