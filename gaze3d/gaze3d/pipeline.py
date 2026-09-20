"""The per-frame pipeline, run on its own thread:

    camera -> MediaPipe landmarks -> PnP head pose -> XGaze normalisation
           -> GPU gaze ensemble -> de-normalise -> ray/screen intersection
           -> person-specific calibration -> 1-Euro smoothing

plus the calibration sample buffer the browser drives with arm()/disarm()/fit().
"""
from __future__ import annotations
import glob, json, math, os, threading, time, traceback
from collections import deque
from dataclasses import asdict
import numpy as np
import cv2

from .camera import Camera
from .face import FaceTracker, Intrinsics, estimate_focal_from_distance
from .normalize import normalize, XGAZE_FACE, vec_to_pitchyaw
from .models import Ensemble
from .calibration import Calibrator, ScreenGeometry, FEATURE_NAMES
from .filters import GazeSmoother

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILES = os.path.join(ROOT, "profiles")

DEFAULT_MODELS = ["unigaze_h14_joint", "puregaze_r50", "gazetr_hybrid", "xgaze_resnet18"]


class Pipeline:
    def __init__(self, models=None, camera_index=0, width=1920, height=1080, hfov_deg=66.0, tta_flip=True):
        self.models = models or DEFAULT_MODELS
        self.camera_index, self.width, self.height = camera_index, width, height
        self.hfov_deg = hfov_deg
        self.tta_flip = tta_flip
        self.cam: Camera | None = None
        self.tracker: FaceTracker | None = None
        self.ens: Ensemble | None = None
        self.geom = ScreenGeometry(1512, 982, 0.1680, (0.0, 18.0))
        self.calib = Calibrator(self.geom)
        self.smoother = GazeSmoother()
        self.lock = threading.Lock()
        self.state: dict = dict(running=False, status="idle", models=self.models)
        self.preview_jpeg: bytes | None = None
        self.crop_jpeg: bytes | None = None
        self._run = False
        self._thread = None
        # calibration
        self.samples: list[dict] = []
        self._written = 0                 # samples already flushed to the session file
        self._session_path: str | None = None
        self.armed: dict | None = None
        self.armed_count = 0
        self._recent = deque(maxlen=90)
        self.fps = 0.0
        self.loaded = False

    # ---- lifecycle ------------------------------------------------------------------
    def load(self):
        if self.loaded:
            return
        self._set(status="loading models")
        self.ens = Ensemble(self.models, tta_flip=self.tta_flip)
        self.loaded = True
        self._set(status="models ready", models=self.ens.names, device=str(self.ens.device))

    def start(self):
        if self._run:
            return
        self.load()
        self._set(status="opening camera")
        self.cam = Camera(self.camera_index, self.width, self.height).start()
        K = Intrinsics.from_hfov(self.cam.width, self.cam.height, self.hfov_deg)
        self.tracker = FaceTracker(K)
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="gaze3d")
        self._thread.start()
        self._set(running=True, status="tracking", cam=dict(w=self.cam.width, h=self.cam.height, fx=K.fx),
                  disk=self.disk_samples())

    def stop(self):
        self._run = False
        if self._thread:
            self._thread.join(timeout=3)
        if self.cam:
            self.cam.stop()
        self._set(running=False, status="stopped")

    def _set(self, **kw):
        with self.lock:
            self.state.update(kw)

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.state)

    # ---- geometry / intrinsics ----------------------------------------------------------
    def set_geometry(self, width_px, height_px, pitch_mm, cam_offset_px=(0.0, 18.0)):
        self.geom = ScreenGeometry(float(width_px), float(height_px), float(pitch_mm),
                                   (float(cam_offset_px[0]), float(cam_offset_px[1])))
        self.calib.geom = self.geom
        self._set(geom=dict(width_px=width_px, height_px=height_px, pitch_mm=pitch_mm, cam_offset_px=list(cam_offset_px)))

    def set_focal_from_distance(self, distance_mm: float) -> float:
        """Use the user's reported viewing distance to fix the camera focal length."""
        fs = self._last_face
        if fs is None or self.tracker is None:
            return float("nan")
        f = estimate_focal_from_distance(self.tracker, fs.landmarks, distance_mm, self.cam.width, self.cam.height)
        self.tracker.set_intrinsics(Intrinsics(f, f, self.cam.width / 2, self.cam.height / 2))
        hfov = math.degrees(2 * math.atan(self.cam.width / 2 / f))
        self._set(cam=dict(w=self.cam.width, h=self.cam.height, fx=f, hfov=hfov))
        return f

    # ---- sample persistence ------------------------------------------------------------
    # Calibration samples are expensive to collect (a minute of the user's attention) and were
    # previously held only in memory: a crash, a restart, or a run whose fit never fired lost
    # them outright. Every armed point is appended to a session file as soon as it closes.
    def _session_file(self) -> str:
        if self._session_path is None:
            os.makedirs(PROFILES, exist_ok=True)
            self._session_path = os.path.join(PROFILES, f"session-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
            meta = dict(kind="meta", started=time.time(), geom=asdict(self.geom),
                        focal=(self.tracker.K.fx if self.tracker else None), models=self.models)
            with open(self._session_path, "w") as f:
                f.write(json.dumps(meta) + "\n")
        return self._session_path

    def _flush_samples(self):
        new = self.samples[self._written:]
        if not new:
            return
        with open(self._session_file(), "a") as f:
            for smp in new:
                f.write(json.dumps(smp) + "\n")
        self._written = len(self.samples)

    def sessions(self) -> list[str]:
        return sorted(glob.glob(os.path.join(PROFILES, "session-*.jsonl")))

    @staticmethod
    def read_session(path: str):
        meta, samples = None, []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                (samples.append(d) if d.get("kind") != "meta" else None)
                if d.get("kind") == "meta":
                    meta = d
        return meta, samples

    def disk_samples(self) -> dict:
        """What is recoverable from disk, for the UI to offer without re-collecting."""
        files = self.sessions()
        if not files:
            return dict(n=0, path=None, points=0)
        meta, samples = self.read_session(files[-1])
        return dict(n=len(samples), path=os.path.basename(files[-1]), points=len({s["point"] for s in samples}),
                    at=(meta or {}).get("started"))

    def recover(self, path: str | None = None) -> dict:
        """Re-fit from samples already on disk. A calibration whose fit never ran is not lost."""
        files = self.sessions()
        path = path or (files[-1] if files else None)
        if not path:
            raise ValueError("no saved calibration samples on disk")
        if not os.path.isabs(path):
            path = os.path.join(PROFILES, path)
        meta, samples = self.read_session(path)
        if not samples:
            raise ValueError(f"{os.path.basename(path)} holds no samples")
        g = (meta or {}).get("geom")
        if g:
            self.set_geometry(g["width_px"], g["height_px"], g["pitch_mm"], g["cam_offset_px"])
        f = (meta or {}).get("focal")
        if f and self.tracker and self.cam:
            self.tracker.set_intrinsics(Intrinsics(f, f, self.cam.width / 2, self.cam.height / 2))
        self.samples = samples
        self._written = len(samples)
        self._session_path = path
        report = self.fit()
        report["recovered_from"] = os.path.basename(path)
        return report

    # ---- calibration control -------------------------------------------------------------
    def arm(self, point: int, target_uv, kind="static"):
        self.armed = dict(point=int(point), target=[float(target_uv[0]), float(target_uv[1])], kind=kind)
        self.armed_count = 0

    def disarm(self) -> int:
        n, self.armed = self.armed_count, None
        self._flush_samples()
        return n

    def clear(self):
        self.samples = []
        self._written = 0
        self._session_path = None          # the next armed point opens a fresh session file
        self.calib = Calibrator(self.geom)
        self.smoother.reset()
        self._set(calibrated=False, calib_report=None)

    def fit(self) -> dict:
        if len(self.samples) < 10:
            raise ValueError("not enough calibration samples")
        self._flush_samples()
        t0 = time.perf_counter()
        report = self.calib.fit(self.samples)
        report["fit_ms"] = (time.perf_counter() - t0) * 1000
        self.smoother.reset()
        # Persist immediately: a fitted calibration must survive a restart without re-collecting.
        try:
            report["saved_to"] = os.path.basename(self.save_profile("default"))
        except Exception as e:
            print(f"[gaze3d] profile auto-save failed: {e!r}", flush=True)
        self._set(calibrated=True, calib_report=report)
        return report

    def save_profile(self, name="default"):
        os.makedirs(PROFILES, exist_ok=True)
        d = self.calib.to_json()
        if d is None:
            raise ValueError("nothing to save")
        d["focal"] = self.tracker.K.fx if self.tracker else None
        with open(os.path.join(PROFILES, f"{name}.json"), "w") as f:
            json.dump(d, f)
        return os.path.join(PROFILES, f"{name}.json")

    def load_profile(self, name="default"):
        with open(os.path.join(PROFILES, f"{name}.json")) as f:
            d = json.load(f)
        self.calib.load_json(d)
        self.geom = self.calib.geom
        if d.get("focal") and self.tracker:
            f = d["focal"]
            self.tracker.set_intrinsics(Intrinsics(f, f, self.cam.width / 2, self.cam.height / 2))
        self._set(calibrated=True, calib_report=self.calib.report)
        return self.calib.report

    # ---- main loop ---------------------------------------------------------------------------
    _last_face = None

    def _loop(self):
        seq = -1
        stamps = deque(maxlen=30)
        last_preview = 0.0
        while self._run:
            try:
                frame, t, seq = self.cam.latest(seq, timeout=0.5)
                if frame is None or seq < 0:
                    continue
                self._step(frame, t, seq)
                stamps.append(time.perf_counter())
                if len(stamps) > 2:
                    self.fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])
                if time.perf_counter() - last_preview > 0.1:
                    self._render_preview(frame)
                    last_preview = time.perf_counter()
            except Exception as e:
                traceback.print_exc()
                self._set(status=f"error: {e!r}")
                time.sleep(0.05)

    def _step(self, frame, t, seq):
        t0 = time.perf_counter()
        fs = self.tracker.process(frame, t)
        t_face = time.perf_counter()
        self._last_face = fs
        if fs is None:
            self._set(seq=seq, t=t, face=False, fps=self.fps, gaze=None, raw=None, armed=self.armed is not None,
                      n_samples=len(self.samples))
            return
        K = self.tracker.K.K
        crop, Rn, _ = normalize(frame, K, fs.R, fs.center, XGAZE_FACE)
        self._last_crop = crop
        out = self.ens.predict(crop)
        t_gpu = time.perf_counter()
        gn = out.mean
        g_cam = Rn.T @ gn
        self._last_gcam = g_cam

        # ensemble agreement: mean angular deviation of the members from the mean (deg)
        devs = [math.degrees(math.acos(float(np.clip(v @ gn, -1, 1)))) for v in out.per_model.values()]
        agreement = float(np.mean(devs)) if devs else 0.0

        # features for the residual regressor
        py_cam = vec_to_pitchyaw(g_cam)[0]
        gr = vec_to_pitchyaw(fs.geo_gaze_r)[0] - py_cam
        gl = vec_to_pitchyaw(fs.geo_gaze_l)[0] - py_cam
        hp = np.radians(fs.head_pose_deg)
        feat = np.array([0.0, 0.0, fs.center[0] / 100, fs.center[1] / 100, fs.center[2] / 100,
                         hp[0], hp[1], hp[2], gr[0], gr[1], gl[0], gl[1], fs.eye_open_r, fs.eye_open_l])
        assert len(feat) == len(FEATURE_NAMES)

        uv, ok = self.calib.predict(fs.center, fs.R, Rn, gn, feat)
        # The calibration's depth scale mostly absorbs the networks' angular gain error, so it
        # is NOT a distance measurement: report the raw PnP distance (IPD-based, ~5 %) for the
        # readouts and for px->degree conversion, and expose the scale separately.
        dscale = math.exp(self.calib.best.params.log_depth) if self.calib.best else 1.0
        dist_raw = float(np.linalg.norm(fs.center))
        blink = fs.blink > 0.45 or min(fs.eye_open_r, fs.eye_open_l) < 0.12
        good = ok and np.all(np.isfinite(uv))
        smooth, fixating = (None, False)
        if good:
            s, fixating = self.smoother(uv, t, blink=blink)
            smooth = s
        elif blink and self.smoother.last is not None:
            smooth = self.smoother.last

        # calibration sample collection
        armed = self.armed
        if armed is not None and not blink and fs.reproj_err < 15:
            self.samples.append(dict(
                o=fs.center.tolist(), Rh=fs.R.tolist(), Rn=Rn.tolist(), gn=gn.tolist(), feat=feat.tolist(),
                per_model={k: v.tolist() for k, v in out.per_model.items()},
                target=armed["target"], point=armed["point"], kind=armed["kind"], t=t,
                head=[float(x) for x in fs.head_pose_deg], dist=float(np.linalg.norm(fs.center))))
            self.armed_count += 1

        py_n = np.degrees(vec_to_pitchyaw(gn)[0])
        self._set(
            seq=seq, t=t, face=True, fps=self.fps,
            ms=dict(face=(t_face - t0) * 1000, gpu=(t_gpu - t_face) * 1000, total=(time.perf_counter() - t0) * 1000,
                    **{k: round(v, 1) for k, v in out.ms.items()}),
            gaze=None if smooth is None else [float(smooth[0]), float(smooth[1])],
            raw=None if not good else [float(uv[0]), float(uv[1])],
            fixating=bool(fixating), blink=bool(blink),
            head=dict(pitch=fs.head_pose_deg[0], yaw=fs.head_pose_deg[1], roll=fs.head_pose_deg[2],
                      x=float(fs.center[0]), y=float(fs.center[1]), z=float(fs.center[2]),
                      dist=dist_raw, dist_cal=dist_raw * dscale, depth_scale=dscale,
                      focal=float(self.tracker.K.fx)),
            gaze_norm=dict(pitch=float(py_n[0]), yaw=float(py_n[1])),
            agreement_deg=agreement, reproj=fs.reproj_err,
            armed=armed is not None, armed_count=self.armed_count, n_samples=len(self.samples),
            calibrated=self.calib.best is not None,
        )

    def _render_preview(self, frame):
        fs = self._last_face
        small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
        sx, sy = 640 / frame.shape[1], 360 / frame.shape[0]
        if fs is not None:
            pts = (fs.landmarks * [sx, sy]).astype(np.int32)
            for p in pts[::3]:
                cv2.circle(small, tuple(p), 1, (90, 232, 213), -1)
            for i in (468, 473):
                cv2.circle(small, tuple(pts[i]), 3, (76, 125, 255), 2)
            g = getattr(self, "_last_gcam", None)
            if g is not None:
                K = self.tracker.K
                def proj(P):
                    return (int((P[0] / P[2] * K.fx + K.cx) * sx), int((P[1] / P[2] * K.fy + K.cy) * sy))
                o = fs.center
                cv2.line(small, proj(o), proj(o + g * 120), (90, 232, 213), 2)
                cv2.line(small, proj(fs.eye_r), proj(fs.eye_r + fs.geo_gaze_r * 60), (255, 200, 80), 1)
                cv2.line(small, proj(fs.eye_l), proj(fs.eye_l + fs.geo_gaze_l * 60), (255, 200, 80), 1)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            self.preview_jpeg = buf.tobytes()
        crop = getattr(self, "_last_crop", None)
        if crop is not None:
            ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                self.crop_jpeg = buf.tobytes()
