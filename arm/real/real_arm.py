"""RealArm: the backend the feeding pipeline plugs into. Completes what arm_replay/ leaves open, FOR THIS
PIPELINE: policy joint targets in MODEL coordinates at 30 Hz, instead of replaying one taught pose.

Reused unchanged from arm_replay/replay_pose2.py (tested there on the arm): `Gravity` (MuJoCo qfrc_bias with the
model's free gravity compensation switched off) and `Sender` (ONE MIT command per joint per tick carrying
gravity + Coulomb feed-forward; state taken from the REPLY, because reading a live motor sends it a zero-gain
command and it goes limp for that cycle).

Added here, each one a flaw listed in arm_replay/README.md:
  * HOLD, NEVER DISABLE. A tripped guard used to disable every motor and the arm fell. Here a trip enters HOLD:
    the last COMMANDED setpoint keeps being sent with feed-forward, guard-free, until a human takes over.
  * CONTINUE FROM THE COMMANDED SETPOINT. Never re-seed from the measured (sagged) position: that removes the
    holding torque. Every target is previous_target + a rate-limited step.
  * RELEASE ONLY AT REST. shutdown() ramps back to the pose the arm was enabled in, and disables only if the
    replies confirm it is there. Otherwise it keeps holding and says so. (The openyam driver also disables in
    an atexit hook - so on a fault this process must stay alive; do not Ctrl-C twice.)
  * TORQUE / CONTACT STOP. The lag guard only notices the arm falling BEHIND; it does not notice it pushing
    HARD. If the torque a motor reports exceeds its feed-forward by more than CONTACT_NM for 3 ticks, the arm
    retraces the last 15 commanded setpoints (so the pressure is RELIEVED, not held) and then holds.
  * JOINT MAP + LIMITS. q_model = q_encoder + 2*pi*lap - offset (joint_map_measured.json; all signs +1). Targets
    are clipped inside the measured hard stops, and rate-limited again here: the backend does not trust callers.

NOT DONE, and the pipeline cannot grasp without it: the GRIPPER (motor 0x08). Its open/closed encoder readings
were never measured, so grip targets are refused rather than guessed. Measure them motors-off, fill GRIP_RAW.

THIS FILE HAS NEVER BEEN RUN AGAINST THE ARM. `python real_arm.py --selftest` exercises everything above on fake
motors. connect() needs an explicit token and a typed confirmation from a person at the arm with the E-stop.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "arm_replay"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_interface import ArmInterface, ArmState  # noqa: E402

TAU = 2 * np.pi
TOKEN = "I_AM_AT_THE_ARM_WITH_THE_ESTOP"
STOP_MARGIN = 0.10            # rad kept clear of every measured hard stop
REST_TOL = 0.06               # rad: 'back at the enable pose' for the purpose of releasing
GRIP_RAW = None               # (raw_open, raw_closed) for motor 0x08 - NOT MEASURED YET
# CONTACT / TORQUE STOP. Excess = |torque the motor reports - the feed-forward it was given|: what it is
# spending beyond holding itself up. Limits sit above the gravity model's own error (the elbow needs ~11 N.m
# where the model says 6.6, arm_replay/TODO_ideas.md) but far below the 28/10 N.m the motors can deliver, so
# pressing into the table trips it within ~0.1 s instead of building to the full torque budget.
CONTACT_NM = np.array([7.0, 8.0, 7.0, 3.0, 2.0, 2.0])
CONTACT_TICKS = 3
BACKOFF_TICKS = 15            # how much of the recent path is retraced to relieve the pressure


def load_joint_map(path=REPO / "joint_map_measured.json"):
    j = json.loads(Path(path).read_text())["joints"]
    names = [f"joint{i}" for i in range(1, 7)]
    return (np.array([j[n]["sign"] for n in names], float), np.array([j[n]["offset"] for n in names], float),
            np.array([j[n]["stops_encoder"] for n in names], float))


class RealArm(ArmInterface):
    def __init__(self, motors, kp, kd, gravity, laps, factors=(1.0, 1.1, 1.4, 1.0, 1.0, 1.0), lock=None, settle_s=1.0):
        from replay_pose2 import MotorError, Sender
        self.MotorError = MotorError
        self.sign, self.offset, stops = load_joint_map()
        assert np.all(self.sign == 1), "the measured map has sign +1 on every joint; anything else is a new measurement"
        self.laps = np.asarray(laps, float)
        lo, hi = stops[:, 0] - TAU * self.laps, stops[:, 1] - TAU * self.laps          # raw window this power session
        self.raw_lo, self.raw_hi = lo + STOP_MARGIN, hi - STOP_MARGIN
        self.sender = Sender(motors, kp, kd, gravity, factors, lock=lock)
        self.motors, self.settle_s = motors, settle_s
        self.mode, self.fault = "idle", ""
        self._sp = None               # last COMMANDED raw setpoint. The only thing new targets are built from.
        self._seed = None             # raw pose the arm was enabled in = the only pose it may be released in
        self._t0 = None
        self._q_prev, self._qd = None, np.zeros(6)
        self._trail, self._over, self.peak_excess = [], 0, np.zeros(6)

    # ------------------------------------------------------------------ coordinates
    def to_model(self, raw):
        return np.asarray(raw, float) + TAU * self.laps - self.offset

    def to_raw(self, q_model):
        return np.asarray(q_model, float) + self.offset - TAU * self.laps

    # ------------------------------------------------------------------ lifecycle
    def begin(self, seed_raw):
        """Call ONCE, right after the driver has enabled each motor holding `seed_raw` (the read-then-enable-then-
        hold sequence from replay_pose2.py). From here on no live motor is ever read again."""
        self._seed = np.asarray(seed_raw, float).copy()
        self._sp = self._seed.copy()
        self._t0, self.mode = time.time(), "run"

    def _gscale(self):
        return min(1.0, (time.time() - self._t0) / max(self.settle_s, 1e-6))          # feed-forward ramps in

    # ------------------------------------------------------------------ ArmInterface
    def read(self) -> ArmState:
        q = self.to_model(self.sender.last["pos"] if np.any(self.sender.last["pos"]) else self._sp)
        if self._q_prev is not None:
            self._qd = 0.6 * self._qd + 0.4 * (q - self._q_prev) * self.rate_hz
        self._q_prev = q.copy()
        return ArmState(q, self._qd.copy(), float("nan"), float("nan"), time.time() - (self._t0 or time.time()))

    def send(self, q_target, grip_target=None):
        if self.mode == "backoff":
            return self._backoff_tick()
        if self.mode != "run":
            return self._hold_tick()
        if grip_target is not None and GRIP_RAW is None:
            self._enter_hold("a gripper target was sent but motor 0x08's open/closed readings were never measured")
            return
        want = np.clip(self.to_raw(q_target), self.raw_lo, self.raw_hi)
        step = np.clip(want - self._sp, -self.max_step_rad, self.max_step_rad)        # from the COMMANDED setpoint
        sp = self._sp + step
        try:
            self.sender(sp, step * self.rate_hz, self._gscale(), "run")
            self._sp = sp
            self._trail = (self._trail + [sp.copy()])[-BACKOFF_TICKS:]
            excess = np.abs(self.sender.last["torque"] - self.sender.last["tff"])
            self.peak_excess = np.maximum(self.peak_excess, excess)
            self._over = self._over + 1 if np.any(excess > CONTACT_NM) else 0
            if self._over >= CONTACT_TICKS:
                j = int(np.argmax(excess - CONTACT_NM))
                self.mode, self.fault = "backoff", (f"CONTACT STOP: joint {j + 1} is spending {excess[j]:.1f} N.m beyond its "
                                                     f"feed-forward (limit {CONTACT_NM[j]:.1f}). Backing off, then holding.")
                print(f"[RealArm] {self.fault}", flush=True)
        except self.MotorError as exc:
            self._enter_hold(str(exc))

    def hold(self):
        self._enter_hold("hold() requested")

    # ------------------------------------------------------------------ fault handling: hold, never disable
    def _backoff_tick(self):
        """Retrace the path just travelled, one commanded setpoint per tick, then hold where it ends."""
        if len(self._trail) > 1:
            self._trail.pop()
            self._sp = self._trail[-1].copy()
            try:
                self.sender(self._sp, None, 1.0, "backoff")
            except self.MotorError:
                pass                                         # still relieving pressure; lag is expected here
        else:
            self.mode = "hold"
            self._hold_tick()

    def _enter_hold(self, why):
        if self.mode != "hold":
            self.mode, self.fault = "hold", why
            print(f"[RealArm] HOLDING (motors stay ON): {why}", flush=True)
        self._hold_tick()

    def _hold_tick(self):
        """Keep the last commanded setpoint alive with full feed-forward. The lag guard is NOT consulted here: it
        is what tripped, and the only alternative to holding is letting go."""
        if self._sp is None:
            return
        try:
            self.sender(self._sp, None, 1.0, "hold")
        except self.MotorError as exc:
            if "fault during motion" in str(exc):                 # a motor reported a hardware fault: nothing more we can do
                print(f"[RealArm] MOTOR FAULT while holding: {exc}. SUPPORT THE ARM BY HAND.", flush=True)
            # a lag trip while holding is expected and ignored

    def shutdown(self, speed=0.25):
        """Ramp back to the enable pose; release ONLY if the replies say the arm is there."""
        from replay_pose2 import ramp_to
        if self._sp is None:
            return "never enabled"
        try:
            ramp_to(lambda sp, vel, g, ph: self.sender(sp, vel, 1.0, ph), self._sp, self._seed, speed, self.rate_hz, hold_s=0.5)
            self._sp = self._seed.copy()
        except self.MotorError as exc:
            self._enter_hold(f"return ramp tripped: {exc}")
        err = np.abs(self.sender.last["pos"] - self._seed)
        if self.mode != "hold" and np.all(err < REST_TOL):
            for mot in reversed(self.motors):
                mot.disable()
            self.mode = "released"
            return "at rest: motors disabled"
        self.mode = "hold"
        return (f"NOT at rest (worst joint {err.max():.3f} rad from the enable pose): motors LEFT ON and holding. "
                "Support the arm by hand before anyone cuts power or kills this process.")

    # ------------------------------------------------------------------ hardware. Never called by the pipeline tests.
    @classmethod
    def connect(cls, laps, token=None):
        if (token or os.environ.get("YAM_REAL_ARM")) != TOKEN:
            raise RuntimeError(f"refusing to touch hardware. A person at the arm, E-stop in hand, sets YAM_REAL_ARM={TOKEN}")
        if not sys.stdin.isatty() or input("Type MOVE to energise the real arm: ").strip() != "MOVE":
            raise RuntimeError("not confirmed at a terminal by a person")
        raise NotImplementedError(
            "wire the openyam driver here exactly as replay_pose2.main() does (TIMEOUT==0 check - do NOT change the "
            "register -, error-state check, read_state -> enable -> command(pos=q) per motor), then call begin(seed). "
            "Left unwired on purpose: the first powered run of new code belongs to a human, in a dry run, arm clear of people.")


# ====================================================================== offline self-test (fake motors, no CAN)
def selftest() -> int:
    from types import SimpleNamespace
    ok_all = [True]

    def check(name, cond, detail=""):
        ok_all[0] &= bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))

    class Motor:
        def __init__(self, i, pos):
            self.motor_id, self.enabled, self.spec, self.pos = i + 1, True, SimpleNamespace(t_max=28.0 if i < 3 else 10.0), pos
            self.sent, self.disabled, self.reads, self.stuck, self.push = [], 0, 0, False, 0.0

        def command(self, pos, vel, kp, kd, torque):
            self.sent.append((pos, vel, kp, kd, torque))
            if not self.stuck:
                self.pos += 0.8 * (pos - self.pos)                # tracks
            return SimpleNamespace(ok=True, error="normal", position=self.pos, torque=torque + self.push)

        def read_state(self):
            self.reads += 1
            return SimpleNamespace(position=self.pos)

        def disable(self):
            self.disabled += 1

    laps = [0, 1, 1, 0, 0, 0]
    seed = np.array([-1.5368, -6.2789, -6.2732, 1.6581, 0.0452, 0.0036])
    grav = lambda raw: np.array([0.0, 3.0, 6.5, 1.8, 0.1, 0.0])  # noqa: E731
    kp, kd = [80] * 3 + [10] * 3, [5] * 3 + [1.5] * 3

    def fresh():
        ms = [Motor(i, float(seed[i])) for i in range(6)]
        arm = RealArm(ms, kp, kd, grav, laps, settle_s=0.0)
        arm.begin(seed)
        return arm, ms

    arm, ms = fresh()
    q0 = arm.to_model(seed)
    check("joint map round-trips", np.allclose(arm.to_raw(arm.to_model(seed)), seed))
    check("connect() refuses without the token", _raises(lambda: RealArm.connect(laps, token="nope")))

    for _ in range(10):
        arm.send(q0 + np.array([0, 0.5, 0, 0, 0, 0]))
    moved = arm._sp[1] - seed[1]
    check("a 0.5 rad jump is rate-limited to 0.02 rad per tick", abs(moved - 0.2) < 1e-9, f"moved {moved:.3f} rad in 10 ticks")
    check("gravity feed-forward rides in every command once ramped in", all(abs(c[4] - 1.4 * 6.5) < 1e-6 for c in ms[2].sent[-8:]),
          f"J3 tff {ms[2].sent[-1][4]:.2f} N.m (first tick {ms[2].sent[0][4]:.2f}: it ramps in, as on the arm)")
    check("no live motor is ever read after begin()", sum(m.reads for m in ms) == 0)

    arm, ms = fresh()
    far = arm.to_model(arm.raw_hi + 1.0)
    for _ in range(400):
        arm.send(far)
    check("targets are clipped inside the measured hard stops", np.all(arm._sp <= arm.raw_hi + 1e-9), f"margin {STOP_MARGIN} rad")

    arm, ms = fresh()
    for _ in range(5):
        arm.send(q0 + np.array([0, 0, 0.3, 0, 0, 0]))
    sp_before = arm._sp.copy()
    ms[2].stuck = True                                            # the elbow stops following: the lag guard must trip
    for _ in range(40):
        arm.send(q0 + np.array([0, 0, 0.6, 0, 0, 0]))
    check("a lag trip enters HOLD", arm.mode == "hold", arm.fault[:70])
    check("HOLD never disables a motor", sum(m.disabled for m in ms) == 0)
    n0 = len(ms[2].sent)
    for _ in range(10):
        arm.send(q0)                                              # caller keeps talking; the arm keeps holding
    held = np.array([c[0] for c in ms[2].sent[n0:]])
    check("HOLD keeps sending the last COMMANDED setpoint (not the sagged measured one)",
          len(held) == 10 and np.allclose(held, arm._sp[2]) and abs(arm._sp[2] - ms[2].pos) > 0.02,
          f"commanded {arm._sp[2]:.3f} vs measured {ms[2].pos:.3f}")
    check("HOLD still carries feed-forward torque", abs(ms[2].sent[-1][4] - 1.4 * 6.5) < 1e-6)
    check("the setpoint moved at most one more step after the trip", np.all(np.abs(arm._sp - sp_before) <= 0.4 + 1e-9))
    msg = arm.shutdown()
    check("shutdown away from rest does NOT release the arm", sum(m.disabled for m in ms) == 0 and arm.mode == "hold", msg[:60])

    arm, ms = fresh()
    for _ in range(20):
        arm.send(q0 + np.array([0.1, 0.1, 0.1, 0, 0, 0]))
    msg = arm.shutdown(speed=2.0)
    check("shutdown after a clean run returns to the enable pose and only THEN disables",
          arm.mode == "released" and all(m.disabled == 1 for m in ms), msg)

    arm, ms = fresh()
    for _ in range(20):
        arm.send(q0 + np.array([0, 0.4, 0.4, 0, 0, 0]))
    peak = arm._sp.copy()
    ms[2].push = 12.0                                             # the elbow starts pressing on something
    for _ in range(3):
        arm.send(q0 + np.array([0, 0.8, 0.8, 0, 0, 0]))
    check("pressing into something trips the CONTACT STOP within 3 ticks", arm.mode == "backoff", arm.fault[:80])
    for _ in range(30):
        arm.send(q0 + np.array([0, 0.8, 0.8, 0, 0, 0]))
    check("it BACKS OFF along the path it came, then holds", arm.mode == "hold" and abs(arm._sp[2] - peak[2]) > 0.1 and
          abs(arm._sp[2] - seed[2]) < abs(peak[2] - seed[2]), f"elbow setpoint retreated {abs(arm._sp[2]-peak[2]):.3f} rad")
    check("the contact stop never disables a motor", sum(m.disabled for m in ms) == 0)

    arm, ms = fresh()
    arm.send(q0, grip_target=0.01)
    check("a gripper target is refused (holds) while 0x08 is unmeasured", arm.mode == "hold", arm.fault[:50])

    try:
        from replay_pose2 import Gravity
        g = Gravity(laps)(np.array([-1.776, -5.728, -6.257, 0.298, 0.121, 0.057]))
        check("the real gravity model loads (elbow ~6.5 N.m at the 60% pose)", abs(g[2] - 6.5) < 0.6, f"J3 {g[2]:.2f} N.m")
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip] real gravity model not loadable here: {exc!r}"[:120])
    print("SELFTEST", "PASSED" if ok_all[0] else "FAILED")
    return 0 if ok_all[0] else 1


def _raises(fn):
    try:
        fn()
    except Exception:  # noqa: BLE001
        return True
    return False


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else print(__doc__) or 0)
