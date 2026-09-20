"""Temporal smoothing for the gaze point: a 1-Euro filter (Casiez et al. CHI'12) with a
velocity-based saccade detector that resets the filter so jumps are not lagged, and a
blink gate that holds the last good estimate."""
from __future__ import annotations
import math
import numpy as np


class LowPass:
    def __init__(self):
        self.y = None
    def __call__(self, x, a):
        self.y = x if self.y is None else a * x + (1 - a) * self.y
        return self.y
    def reset(self):
        self.y = None


class OneEuro:
    def __init__(self, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_f, self.dx_f = LowPass(), LowPass()
        self.t_prev = None
        self.x_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x_f.reset(); self.dx_f.reset(); self.t_prev = None; self.x_prev = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.t_prev is None:
            self.t_prev, self.x_prev = t, x
            self.x_f(x, 1.0); self.dx_f(np.zeros_like(x), 1.0)
            return x
        dt = max(t - self.t_prev, 1e-3)
        self.t_prev = t
        dx = (x - self.x_prev) / dt
        self.x_prev = x
        edx = self.dx_f(dx, self._alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(edx))
        return self.x_f(x, self._alpha(cutoff, dt))


class GazeSmoother:
    """1-Euro + saccade reset + blink hold. Units: screen px, seconds."""
    def __init__(self, min_cutoff=0.8, beta=0.01, saccade_px_per_s=2500.0):
        self.f = OneEuro(min_cutoff, beta)
        self.saccade_v = saccade_px_per_s
        self.last = None
        self.last_t = None
        self.fix_start = None
        self.fixating = False

    def reset(self):
        self.f.reset(); self.last = None; self.last_t = None; self.fix_start = None

    def __call__(self, uv, t, blink=False):
        if blink:
            return self.last, self.fixating
        uv = np.asarray(uv, dtype=np.float64)
        if self.last is not None and self.last_t is not None:
            v = np.linalg.norm(uv - self.last) / max(t - self.last_t, 1e-3)
            if v > self.saccade_v:
                self.f.reset()
                self.fix_start = t
        out = self.f(uv, t)
        self.last, self.last_t = out, t
        self.fix_start = self.fix_start or t
        self.fixating = (t - self.fix_start) > 0.12
        return out, self.fixating
