import time

import numpy as np

from arm.calibration import (backproject_pixel, fuse_depth_or_plane,
                             table_plane_intersection, transform_point,
                             validate_intrinsics_resolution)
from arm.deploy import DryRunArm, SafetyLimiter, safe_command
from arm.perception_io import Target3D, TargetFilter


def test_safety_clipping_and_stop():
    arm = DryRunArm()
    limiter = SafetyLimiter(-np.ones(6), np.ones(6), 0.3, 30)
    safe_command(arm, limiter, np.ones(6), 2.0)
    assert np.allclose(arm.q, 0.01)
    assert arm.commands[-1][1] == 1.0


def test_stale_jump_wrist_override_and_dropout_hold():
    f = TargetFilter()
    now = time.time()
    external = Target3D(np.array([0.3, 0, 0.05]), "base", "external", 0.9, now)
    assert f.consider(external, np.zeros(3), now)
    assert not f.consider(Target3D(np.array([0.4, 0, 0.05]), "base", "external", 0.9, now),
                          np.zeros(3), now)
    wrist = Target3D(np.array([0.35, 0, 0.05]), "base", "wrist", 0.9, now)
    assert f.consider(wrist, np.array([0.25, 0, 0.05]), now, at_pregrasp=True)
    assert f.current_or_hold() is wrist
    assert not f.consider(Target3D(np.array([0.35, 0, 0.05]), "base", "wrist", 0.9, now - 2),
                          np.zeros(3), now)


def test_projection_and_crosscheck():
    K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], float)
    p = backproject_pixel(50, 50, 1.0, K)
    assert np.allclose(p, [0, 0, 1])
    T = np.eye(4); T[:3, 3] = [0.1, 0.2, 1.0]; T[:3, :3] = np.diag([1, 1, -1])
    assert np.allclose(transform_point(p, T), [0.1, 0.2, 0.0])
    plane = table_plane_intersection(50, 50, K, T)
    assert np.allclose(plane, [0.1, 0.2, 0.0])
    chosen, confidence, source = fuse_depth_or_plane(np.array([0.2, 0.2, 0]), plane)
    assert source == "plane_disagreement" and confidence < 0.5 and np.allclose(chosen, plane)
    validate_intrinsics_resolution((640, 480), (640, 480))

