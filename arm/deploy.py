from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ArmInterface(Protocol):
    def joint_positions(self) -> np.ndarray: ...
    def command(self, q: np.ndarray, gripper: float) -> None: ...
    def stop(self) -> None: ...


class DryRunArm:
    def __init__(self, initial: np.ndarray | None = None):
        self.q = np.zeros(6) if initial is None else np.asarray(initial, float).copy()
        self.commands: list[tuple[np.ndarray, float]] = []
        self.stopped = False

    def joint_positions(self) -> np.ndarray:
        return self.q.copy()

    def command(self, q: np.ndarray, gripper: float) -> None:
        self.q = np.asarray(q, float).copy()
        self.commands.append((self.q.copy(), float(gripper)))

    def stop(self) -> None:
        self.stopped = True


class RealArm:
    def __init__(self, enable_hardware: bool = False):
        if not enable_hardware:
            raise RuntimeError("real hardware requires explicit --enable-hardware")
        raise RuntimeError("real backend intentionally disabled pending joint-map/FK verification")


@dataclass
class SafetyLimiter:
    lower: np.ndarray
    upper: np.ndarray
    velocity_cap_rad_s: float = 0.25
    control_hz: float = 30.0

    def clip(self, current: np.ndarray, requested: np.ndarray) -> np.ndarray:
        current, requested = np.asarray(current, float), np.asarray(requested, float)
        if current.shape != (6,) or requested.shape != (6,) or not np.isfinite(requested).all():
            raise ValueError("joint command must contain six finite values")
        step = self.velocity_cap_rad_s / self.control_hz
        return np.clip(np.clip(requested, current - step, current + step), self.lower, self.upper)


def safe_command(arm: ArmInterface, limiter: SafetyLimiter, requested: np.ndarray, gripper: float) -> None:
    try:
        arm.command(limiter.clip(arm.joint_positions(), requested), float(np.clip(gripper, 0, 1)))
    except Exception:
        arm.stop()
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-hardware", action="store_true")
    args = parser.parse_args()
    arm = RealArm(True) if args.enable_hardware else DryRunArm()
    print(f"dry_run={not args.enable_hardware}; no command sent")
    arm.stop()


if __name__ == "__main__":
    main()

