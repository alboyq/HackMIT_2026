from __future__ import annotations

import numpy as np


def minimum_jerk(q0: np.ndarray, q1: np.ndarray, hz: float, velocity_limit: float,
                 acceleration_limit: float) -> np.ndarray:
    """Quintic joint trajectory with zero endpoint velocity and acceleration."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    travel = float(np.max(np.abs(q1 - q0)))
    # Quintic peak coefficients: velocity 1.875/T, acceleration ~5.774/T^2.
    duration = max(2.0 / hz, 1.875 * travel / velocity_limit,
                   np.sqrt(5.774 * travel / acceleration_limit))
    n = max(2, int(np.ceil(duration * hz)))
    u = np.linspace(0.0, 1.0, n + 1)[:, None]
    blend = 10 * u**3 - 15 * u**4 + 6 * u**5
    return q0 + blend * (q1 - q0)


def validate_trajectory(q: np.ndarray, lo: np.ndarray, hi: np.ndarray, hz: float,
                        velocity_limit: float, acceleration_limit: float) -> str | None:
    q = np.asarray(q, float)
    if not np.isfinite(q).all():
        return "trajectory contains non-finite values"
    if np.any(q < lo) or np.any(q > hi):
        return "trajectory violates a joint limit"
    if len(q) > 1 and np.max(np.abs(np.diff(q, axis=0) * hz)) > velocity_limit * 1.001:
        return "trajectory violates the velocity limit"
    if len(q) > 2 and np.max(np.abs(np.diff(q, n=2, axis=0) * hz**2)) > acceleration_limit * 1.01:
        return "trajectory violates the acceleration limit"
    return None

