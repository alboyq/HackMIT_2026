"""The boundary between the feeding pipeline and an arm. NOTHING IN THIS FILE MOVES A MOTOR.

The pipeline (learned reach, learned pinch, scripted lift, scripted feed) speaks one small interface:
    read()  -> joint angles, joint velocities, gripper opening          (motor feedback)
    send()  -> joint position targets + a gripper target, at 30 Hz
Two backends:
  * DryRunArm - integrates the commands in software and logs them. This is what exists and is tested.
  * RealArm   - deliberately NOT implemented here. It raises. arm/deploy.py keeps its own guard too.

What a real backend MUST do, from what arm_replay/ measured on this arm (commit d029eb3), not from the sim:
  1. GRAVITY FEED-FORWARD. Position control alone left the gripper 11 cm low at 60% extension; with the
     vendor gravity recipe it was 1.4 cm low. The sim's clean position tracking does not exist on this arm, and
     1.4 cm is larger than the feed's stand-off error budget, so close the loop on the CAMERA near the face
     (the feed already does: it servos on the face, not on a commanded pose).
  2. JOINT MAP. q_model = sign * q_encoder - offset, measured in commit 2ee81bf (J1 and J6 carry ~80 deg
     offsets). Raw motor numbers are only valid until the next power cycle ('laps').
  3. NEVER DISABLE ON A FAULT. An aborted return disabled every motor and the arm fell. On any guard trip the
     backend must HOLD (or lower slowly), never release, above a person.
  4. A WATCHDOG. This arm reports TIMEOUT=0; if the controller dies the motors keep their last command. Until
     that is resolved a human must hold the E-stop for any run near a person.
  5. RATE + LIMITS. The policies were trained at 30 Hz with <= 0.02 rad per step per joint and a 0.35 low-pass;
     the feed adds 0.20 m/s retreat, 0.10 m/s approach, a 0.012 m/s crawl at the face and 0.35 m/s^2. The
     backend must enforce these again itself; it must not trust the caller.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ArmState:
    q: np.ndarray                 # six joint angles, MODEL convention (after the joint map), rad
    qd: np.ndarray                # six joint velocities, rad/s
    grip: float                   # gripper opening, m per finger (0 closed .. 0.0375 open)
    grip_target: float = 0.0
    t: float = 0.0


class ArmInterface:
    rate_hz = 30.0
    max_step_rad = 0.02           # per joint per tick; the policies never ask for more

    def read(self) -> ArmState:
        raise NotImplementedError

    def send(self, q_target: np.ndarray, grip_target: float) -> None:
        raise NotImplementedError

    def hold(self) -> None:
        """Fault response: keep position. NEVER disable the motors above a person."""
        raise NotImplementedError


@dataclass
class DryRunArm(ArmInterface):
    """Software arm: first-order tracking of the commanded joints, everything logged, nothing energised."""
    q: np.ndarray = field(default_factory=lambda: np.array([-0.0279, 1.2776, 1.4870, -1.0699, -0.1920, -0.5637]))
    grip: float = 0.0375
    tau_s: float = 0.12
    log: list = field(default_factory=list)
    _qd: np.ndarray = field(default_factory=lambda: np.zeros(6))
    _qt: np.ndarray = None
    _gt: float = 0.0375
    _t: float = 0.0

    def read(self) -> ArmState:
        return ArmState(self.q.copy(), self._qd.copy(), float(self.grip), float(self._gt), self._t)

    def send(self, q_target, grip_target):
        q_target = np.asarray(q_target, dtype=float)
        step = np.clip(q_target - (self._qt if self._qt is not None else self.q), -self.max_step_rad, self.max_step_rad)
        self._qt = (self._qt if self._qt is not None else self.q) + step          # the backend re-enforces the limit
        self._gt = float(np.clip(grip_target, 0.0, 0.0375))
        dt = 1.0 / self.rate_hz
        new = self.q + (self._qt - self.q) * (dt / self.tau_s)
        self._qd, self.q = (new - self.q) / dt, new
        self.grip += (self._gt - self.grip) * (dt / self.tau_s)
        self._t += dt
        self.log.append((self._t, self._qt.copy(), self._gt))

    def hold(self):
        self._qt = self.q.copy()


class RealArm(ArmInterface):
    def __init__(self, *a, **k):
        raise RuntimeError("real backend intentionally not implemented here: see the five requirements in this file's "
                           "docstring and arm_replay/README.md. Moving this arm needs a human, an E-stop and a dry run.")


if __name__ == "__main__":
    arm = DryRunArm()
    s0 = arm.read()
    for _ in range(60):
        arm.send(s0.q + np.array([0.5, 0, 0, 0, 0, 0]), 0.0)          # asks for a 0.5 rad jump; gets 0.02 rad/tick
    s1 = arm.read()
    print(f"dry run ok: J1 moved {s1.q[0]-s0.q[0]:+.3f} rad in 2 s (limit 0.02 rad/tick -> max 1.2), grip {s0.grip:.4f} -> {s1.grip:.4f}")
    try:
        RealArm()
    except RuntimeError as ex:
        print("real backend refuses, as intended:", str(ex)[:60], "...")
