"""Data normalisation for appearance-based gaze estimation.

Implements the ETH-XGaze / "Revisiting Data Normalization" (Zhang et al. 2018)
protocol: a virtual camera is placed at a fixed distance from the gaze origin, rolled
so the head's x-axis is horizontal, and the image is warped into it. Networks
trained on XGaze / MPIIFaceGaze / Gaze360-normalized data expect exactly this.

De-normalisation of a predicted direction: g_cam = R.T @ g_norm.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class NormSpec:
    focal: float = 960.0
    distance: float = 600.0
    size: tuple = (224, 224)


XGAZE_FACE = NormSpec(960.0, 600.0, (224, 224))
MPII_FACE = NormSpec(960.0, 600.0, (224, 224))
MPII_EYE = NormSpec(960.0, 600.0, (60, 36))


def normalize(img_bgr: np.ndarray, K: np.ndarray, head_R: np.ndarray, origin: np.ndarray,
              spec: NormSpec = XGAZE_FACE):
    """Returns (warped_bgr, R_norm, W) where R_norm is the rotation from camera frame to
    the normalized frame and W the full image homography used."""
    distance = float(np.linalg.norm(origin))
    z_scale = spec.distance / distance
    cam_norm = np.array([[spec.focal, 0, spec.size[0] / 2],
                         [0, spec.focal, spec.size[1] / 2],
                         [0, 0, 1.0]])
    S = np.diag([1.0, 1.0, z_scale])
    hRx = head_R[:, 0]
    forward = origin / distance
    down = np.cross(forward, hRx); down /= np.linalg.norm(down)
    right = np.cross(down, forward); right /= np.linalg.norm(right)
    R = np.stack([right, down, forward], axis=0)          # rows
    W = cam_norm @ S @ R @ np.linalg.inv(K)
    warped = cv2.warpPerspective(img_bgr, W, spec.size, flags=cv2.INTER_LINEAR)
    return warped, R, W


def pitchyaw_to_vec(py: np.ndarray) -> np.ndarray:
    """(N,2) [pitch, yaw] radians -> (N,3) unit vectors, XGaze convention
    (x = -cos(p) sin(y), y = -sin(p), z = -cos(p) cos(y))."""
    py = np.atleast_2d(py)
    p, y = py[:, 0], py[:, 1]
    return np.stack([-np.cos(p) * np.sin(y), -np.sin(p), -np.cos(p) * np.cos(y)], axis=1)


def vec_to_pitchyaw(v: np.ndarray) -> np.ndarray:
    v = np.atleast_2d(v)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    pitch = np.arcsin(-v[:, 1])
    yaw = np.arctan2(-v[:, 0], -v[:, 2])
    return np.stack([pitch, yaw], axis=1)


def angular_error_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))
