"""Frame source fed by frames pushed from outside the process (e.g. a browser
streaming its webcam over the `/ws` WebSocket) instead of a local
`cv2.VideoCapture` device.

Deliberately mirrors `Camera`'s public surface (`start`, `stop`, `latest`,
`.width`, `.height`, `.measured_fps`) so `Pipeline._loop` runs unchanged on
top of either source - this is an alternative frame source, not a fork of the
pipeline.

One real difference from `Camera.start()`: that call blocks briefly on a
synchronous `cap.read()` because opening a local device either works almost
immediately or fails outright (wrong index, no permission). A network pusher
has no such guarantee - the first frame might be seconds away, arriving as a
*separate* WebSocket message on the same connection that issued `start`. If
`start()` blocked here, it would have to do so from inside the WS handler's
command dispatch, which would stall that connection's receive loop and could
never observe the very frame it is waiting for (see `server.py::ws`). So this
`start()` returns immediately; `Pipeline._loop`'s existing `cam.latest(...,
timeout=0.5)` retry already tolerates "no frame yet" and picks up the first
pushed frame whenever it lands.
"""
from __future__ import annotations
import threading, time
from collections import deque
import cv2
import numpy as np


class PushCamera:
    def __init__(self, width: int = 640, height: int = 480):
        self.width, self.height = width, height
        self._lock = threading.Condition()
        self._frame = None
        self._t = 0.0
        self._seq = 0
        self._run = False
        self.measured_fps = 0.0
        self._stamps: deque = deque(maxlen=30)

    def start(self):
        self._run = True
        return self

    def push(self, jpeg_bytes: bytes) -> bool:
        """Decode one JPEG-encoded frame and make it the newest frame.
        Returns False (dropping the frame) if decoding fails or the source
        has been stopped. Safe to call from the asyncio event loop thread -
        decoding a small JPEG is a few ms, and this never blocks on I/O."""
        if not self._run:
            return False
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return False
        with self._lock:
            self._frame, self._t, self._seq = frame, time.perf_counter(), self._seq + 1
            self.height, self.width = frame.shape[:2]
            self._lock.notify_all()
        self._stamps.append(self._t)
        if len(self._stamps) > 2:
            self.measured_fps = (len(self._stamps) - 1) / (self._stamps[-1] - self._stamps[0])
        return True

    def latest(self, after_seq: int = -1, timeout: float = 1.0):
        """Same contract as Camera.latest: block until a frame newer than
        `after_seq` exists (or timeout), return (frame, t, seq)."""
        with self._lock:
            if self._seq <= after_seq:
                self._lock.wait_for(lambda: self._seq > after_seq or not self._run, timeout)
            return self._frame, self._t, self._seq

    def stop(self):
        self._run = False
        with self._lock:
            self._lock.notify_all()
