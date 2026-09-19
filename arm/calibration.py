from __future__ import annotations

import numpy as np


def backproject_pixel(u: float, v: float, depth_m: float, K: np.ndarray) -> np.ndarray:
    if depth_m <= 0 or not np.isfinite(depth_m):
        raise ValueError("depth must be finite and positive")
    return depth_m * np.linalg.inv(np.asarray(K, float)) @ np.array([u, v, 1.0])


def transform_point(point: np.ndarray, base_from_camera: np.ndarray) -> np.ndarray:
    p = np.append(np.asarray(point, float), 1.0)
    return (np.asarray(base_from_camera, float) @ p)[:3]


def table_plane_intersection(u: float, v: float, K: np.ndarray, base_from_camera: np.ndarray,
                             object_height_m: float = 0.0) -> np.ndarray:
    ray_cam = np.linalg.inv(np.asarray(K, float)) @ np.array([u, v, 1.0])
    T = np.asarray(base_from_camera, float)
    origin, direction = T[:3, 3], T[:3, :3] @ ray_cam
    if abs(direction[2]) < 1e-9:
        raise ValueError("ray is parallel to table")
    scale = (object_height_m - origin[2]) / direction[2]
    if scale <= 0:
        raise ValueError("table intersection is behind camera")
    return origin + scale * direction


def fuse_depth_or_plane(depth_point_base: np.ndarray | None, plane_point_base: np.ndarray,
                        disagreement_m: float = 0.03) -> tuple[np.ndarray, float, str]:
    if depth_point_base is None:
        return np.asarray(plane_point_base), 0.6, "plane"
    if np.linalg.norm(np.asarray(depth_point_base) - np.asarray(plane_point_base)) > disagreement_m:
        return np.asarray(plane_point_base), 0.4, "plane_disagreement"
    return np.asarray(depth_point_base), 0.9, "depth"


def validate_intrinsics_resolution(saved: tuple[int, int], runtime: tuple[int, int]) -> None:
    if tuple(saved) != tuple(runtime):
        raise ValueError(f"intrinsics resolution {saved} does not match runtime {runtime}")

