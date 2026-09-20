"""Threaded webcam capture. Always hands out the newest frame so inference never
queues behind stale video."""
from __future__ import annotations
import threading, time
import cv2
import numpy as np


class Camera:
    def __init__(self, index: int = 0, width: int = 1920, height: int = 1080, fps: int = 30):
        self.index, self.width, self.height, self.fps = index, width, height, fps
        self._cap = None
        self._lock = threading.Condition()
        self._frame = None
        self._t = 0.0
        self._seq = 0
        self._run = False
        self._thread = None
        self.measured_fps = 0.0

    def start(self):
        cap = cv2.VideoCapture(self.index, cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not cap.isOpened():
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
