from __future__ import annotations

import numpy as np
import pytest

from arm.calibrate_extrinsic import (drop_worst, solve_base_from_camera,
                                     solve_rigid_transform, spread_rank)
from arm.calibration import transform_point


def _truth() -> np.ndarray:
    """A plausible scene-camera pose: 0.9 m back, 0.35 m left, 0.6 m up, looking down at the table."""
    rx, rz = np.deg2rad(-35.0), np.deg2rad(20.0)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Rx
    T[:3, 3] = [-0.90, 0.35, 0.60]
    return T


def _poses(n: int, seed: int = 0) -> np.ndarray:
    """TCP positions spread over the YAM's reachable table volume, in the base frame."""
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(0.25, 0.55, n),
                            rng.uniform(-0.25, 0.25, n),
                            rng.uniform(0.05, 0.35, n)])


def _observe(base_points: np.ndarray, T: np.ndarray, noise_m: float = 0.0,
             seed: int = 1) -> np.ndarray:
    """Invert the truth to get what a perfect camera would see, then corrupt it."""
    R, t = T[:3, :3], T[:3, 3]
    cam = (R.T @ (base_points - t).T).T
    if noise_m:
        cam = cam + np.random.default_rng(seed).normal(0, noise_m, cam.shape)
    return cam


def test_recovers_exact_transform_without_noise():
    T = _truth()
    base = _poses(6)
    got, report = solve_base_from_camera(_observe(base, T), base)
    assert np.allclose(got, T, atol=1e-9)
    assert report["rms_m"] < 1e-9
    assert report["passed"]


def test_solution_is_a_rotation_not_a_reflection():
    T = _truth()
    base = _poses(6)
    got, _ = solve_base_from_camera(_observe(base, T), base)
    R = got[:3, :3]
    assert np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)


def test_survives_realistic_lidar_noise():
    """5 mm per-axis noise is a fair stand-in for a depth centroid averaged over a blob."""
    T = _truth()
    base = _poses(8)
    got, report = solve_base_from_camera(_observe(base, T, noise_m=0.005), base)
    assert report["rms_m"] < 0.010
    # What actually matters is where a commanded point lands, not the matrix norm.
    probe_cam = _observe(np.array([[0.40, 0.0, 0.10]]), T)[0]
    assert np.linalg.norm(transform_point(probe_cam, got) - [0.40, 0.0, 0.10]) < 0.010


def test_rejects_coplanar_poses():
    """All five poses at one table height leave the out-of-plane direction unconstrained."""
    T = _truth()
    base = _poses(5)
    base[:, 2] = 0.10
    with pytest.raises(ValueError, match="coplanar"):
        solve_base_from_camera(_observe(base, T), base)
    assert spread_rank(base)[2] < 1e-9


def test_rejects_too_few_poses():
    T = _truth()
    base = _poses(3)
    with pytest.raises(ValueError, match="at least 4"):
        solve_base_from_camera(_observe(base, T), base)


def test_rejects_mismatched_and_non_finite_input():
    with pytest.raises(ValueError, match="matching"):
        solve_rigid_transform(np.zeros((4, 3)), np.zeros((5, 3)))
    with pytest.raises(ValueError, match="non-finite"):
        solve_rigid_transform(np.full((4, 3), np.nan), np.zeros((4, 3)))


def test_drop_worst_finds_the_bad_depth_read():
    """One blown depth pixel is the expected failure; it must be identifiable, not fatal."""
    T = _truth()
    base = _poses(7)
    cam = _observe(base, T)
    cam[4] += [0.0, 0.0, 0.18]          # a depth return off the object, onto the table behind
    _, dirty = solve_base_from_camera(cam, base)
    assert dirty["rms_m"] > 0.02

    kept_cam, kept_base, dropped = drop_worst(cam, base)
    assert dropped == 4
    _, clean = solve_base_from_camera(kept_cam, kept_base)
    assert clean["rms_m"] < 1e-9
