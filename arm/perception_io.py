from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Target3D:
    xyz: np.ndarray
    frame: str
    source: str
    confidence: float
    timestamp: float

    @classmethod
    def from_json(cls, payload: str | bytes) -> "Target3D":
        obj = json.loads(payload)
        return cls(np.asarray([obj["x"], obj["y"], obj["z"]], float),
                   str(obj["frame"]), str(obj["source"]), float(obj["confidence"]),
                   float(obj["timestamp"]))


class TargetFilter:
    def __init__(self, stale_usb_s: float = 0.5, stale_wireless_s: float = 1.0,
                 min_confidence: float = 0.35, max_jump_m: float = 0.05,
                 wrist_override_distance_m: float = 0.15):
        self.stale_usb_s, self.stale_wireless_s = stale_usb_s, stale_wireless_s
        self.min_confidence, self.max_jump_m = min_confidence, max_jump_m
        self.wrist_override_distance_m = wrist_override_distance_m
        self.accepted: Target3D | None = None
        self.rejections: list[str] = []

    def consider(self, target: Target3D, tcp_xyz: np.ndarray, now: float | None = None,
                 at_pregrasp: bool = False) -> bool:
        now = time.time() if now is None else now
        reason = None
        if target.frame != "base":
            reason = "frame must be base"
        elif not np.isfinite(target.xyz).all():
            reason = "non-finite target"
        elif target.confidence < self.min_confidence:
            reason = "low confidence"
        else:
            threshold = self.stale_wireless_s if target.source == "external_wireless" else self.stale_usb_s
            if now - target.timestamp > threshold:
                reason = "stale"
        if reason is None and self.accepted is not None:
            jump = float(np.linalg.norm(target.xyz - self.accepted.xyz))
            allow_wrist_snap = target.source == "wrist" and at_pregrasp
            if jump > self.max_jump_m and not allow_wrist_snap:
                reason = f"jump {jump:.3f} m"
            close = np.linalg.norm(self.accepted.xyz - np.asarray(tcp_xyz)) <= self.wrist_override_distance_m
            if target.source == "external" and close and self.accepted.source == "wrist":
                reason = "wrist estimate owns close approach"
        if reason:
            self.rejections.append(reason)
            return False
        self.accepted = target
        return True

    def current_or_hold(self) -> Target3D | None:
        """A dropout holds the last accepted target; callers must not create motion from absence."""
        return self.accepted


def encode_target(target: Target3D) -> str:
    return json.dumps({"x": float(target.xyz[0]), "y": float(target.xyz[1]),
                       "z": float(target.xyz[2]), "frame": target.frame,
                       "source": target.source, "confidence": target.confidence,
                       "timestamp": target.timestamp})

