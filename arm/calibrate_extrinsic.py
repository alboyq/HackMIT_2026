"""Solve `base_from_camera`, the one input arm/calibration.py consumes and nothing produces.

With a metric depth camera the problem is closed form. At each of N arm poses, FK gives the
TCP in the base frame and the depth camera gives the same physical point in the camera frame.
Two corresponding point sets -> Umeyama/Kabsch -> one SVD. No ICP, no marker board, no model.

The marker-free alternative is Hydra (arXiv:2504.20584): segment the manipulator with SAM 2,
mask the depth map, ICP-register the fused cloud against the URDF. It reaches 5 mm task-space
from ~3 poses and is the better method given a day. This is the version that needs only numpy.

Getting the TCP point in the camera frame, easiest first:
  1. coloured sticker on the gripper -> HSV threshold -> centroid -> median depth over the blob
  2. ArUco marker held in the jaws -> solvePnP (also gives orientation, so AX=XB is available)
  3. SAM 3 text prompt "robot gripper" -> mask -> masked-depth centroid

Calibrate at the distance the objects will actually sit at: iPhone LiDAR bias is range
dependent, and a fit at the working range absorbs most of it into the translation term.
"""
from __future__ import annotations

import numpy as np

MIN_POSES = 4


def solve_rigid_transform(points_camera: np.ndarray, points_base: np.ndarray) -> np.ndarray:
    """Least-squares rotation+translation taking camera-frame points onto base-frame points."""
    src = np.asarray(points_camera, float)
    dst = np.asarray(points_base, float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"need matching (N, 3) arrays, got {src.shape} and {dst.shape}")
    if len(src) < 3:
        raise ValueError(f"need at least 3 correspondences, got {len(src)}")
    if not (np.isfinite(src).all() and np.isfinite(dst).all()):
        raise ValueError("correspondences contain non-finite values")

    src_c, dst_c = src - src.mean(0), dst - dst.mean(0)
    U, S, Vt = np.linalg.svd(src_c.T @ dst_c)
    # Reflections are valid SVD solutions and invalid rigid transforms; flip the smallest axis.
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))])
    R = U @ D @ Vt
    R = R.T

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = dst.mean(0) - R @ src.mean(0)
    return T


def spread_rank(points: np.ndarray) -> np.ndarray:
    """Singular values of the centred point set. A near-zero third value means coplanar poses."""
    p = np.asarray(points, float)
    return np.linalg.svd(p - p.mean(0), compute_uv=False)


def solve_base_from_camera(points_camera: np.ndarray, points_base: np.ndarray,
                           max_rms_m: float = 0.015) -> tuple[np.ndarray, dict]:
    """Fit the extrinsic and report how well it fits. Raises if the geometry cannot support it.

    `max_rms_m` is advisory: it is reported as `passed`, never enforced, because the caller
    decides whether a loose fit is worth using. The partner's gate is FK error under 10 mm at
    five poses; an RMS above that means something is wrong with the correspondences, not the fit.
    """
    src = np.asarray(points_camera, float)
    dst = np.asarray(points_base, float)
    if len(src) < MIN_POSES:
        raise ValueError(f"need at least {MIN_POSES} poses for a trustworthy fit, got {len(src)}")

    sv = spread_rank(dst)
    if sv[2] < 1e-3:
        raise ValueError(
            f"arm poses are coplanar (singular values {sv.round(4).tolist()}); "
            "vary height as well as reach or the fit is unconstrained out of plane")

    T = solve_rigid_transform(src, dst)
    mapped = (T[:3, :3] @ src.T).T + T[:3, 3]
    errors = np.linalg.norm(mapped - dst, axis=1)
    report = {
        "n_poses": int(len(src)),
        "rms_m": float(np.sqrt((errors ** 2).mean())),
        "max_m": float(errors.max()),
        "per_pose_m": errors.tolist(),
        "worst_pose": int(errors.argmax()),
        "base_spread_m": sv.tolist(),
        "passed": bool(np.sqrt((errors ** 2).mean()) <= max_rms_m),
    }
    return T, report


def drop_worst(points_camera: np.ndarray, points_base: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Refit without the single worst correspondence. One bad depth read should not cost the session."""
    _, report = solve_base_from_camera(points_camera, points_base)
    keep = [i for i in range(len(points_camera)) if i != report["worst_pose"]]
    return np.asarray(points_camera)[keep], np.asarray(points_base)[keep], report["worst_pose"]
