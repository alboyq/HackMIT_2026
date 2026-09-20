"""Threaded webcam capture. Always hands out the newest frame so inference never
queues behind stale video."""
from __future__ import annotations
import sys, threading, time
import cv2
import numpy as np


def default_backend() -> int:
    """OpenCV capture backend for this OS. AVFoundation is macOS-only; forcing it
    elsewhere makes VideoCapture fail with an empty device rather than an error."""
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    if sys.platform.startswith("win"):
        return cv2.CAP_MSMF
    return cv2.CAP_ANY


class Camera:
    def __init__(self, index: int = 0, width: int = 1920, height: int = 1080, fps: int = 30,
                 backend: int | None = None):
        self.index, self.width, self.height, self.fps = index, width, height, fps
        self.backend = default_backend() if backend is None else backend
        self._cap = None
        self._lock = threading.Condition()
        self._frame = None
        self._t = 0.0
        self._seq = 0
        self._run = False
        self._thread = None
        self.measured_fps = 0.0

    def start(self):
        cap = self._open(self.backend)
        if cap is None and self.backend != cv2.CAP_ANY:
            print(f"[gaze3d] capture backend {self.backend} failed, falling back to CAP_ANY", flush=True)
            cap = self._open(cv2.CAP_ANY)
        if cap is None:
            raise RuntimeError(f"camera {self.index} could not be opened")
        ok, f = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError("camera opened but delivered no frame (permission?)")
        self.height, self.width = f.shape[:2]
        self._cap = cap
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="camera")
        self._thread.start()
        return self

    def _open(self, backend: int):
        cap = cv2.VideoCapture(self.index, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    def _loop(self):
        n, t0 = 0, time.perf_counter()
        while self._run:
            ok, f = self._cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame, self._t, self._seq = f, time.perf_counter(), self._seq + 1
                self._lock.notify_all()
            n += 1
            if n == 30:
                t1 = time.perf_counter()
                self.measured_fps = n / (t1 - t0)
                n, t0 = 0, t1

    def latest(self, after_seq: int = -1, timeout: float = 1.0):
        """Block until a frame newer than `after_seq` exists. Returns (frame, t, seq)."""
        with self._lock:
            if self._seq <= after_seq:
                self._lock.wait_for(lambda: self._seq > after_seq or not self._run, timeout)
            return self._frame, self._t, self._seq

    def stop(self):
        self._run = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()
        self._cap = None
