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

GRIPPER (motor 0x08): open/closed readings measured 2026-09-20 (GRIP_RAW). The jaw setpoint may lead the measured
jaws by at most GRIP_LEAD, so a blocked jaw squeezes with a bounded torque. NEVER RUN ON THE ARM YET: the first
powered run is `--hold-test 10` (zero motion), by a person at the arm.

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
GRIP_RAW = (-2.8775, -0.0600)  # (raw_open, raw_closed) for motor 0x08, read 2026-09-20 with the motor disabled
GRIP_MARGIN = 0.08            # rad kept clear of both mechanical ends so the motor never stalls on them
GRIP_STEP = 0.15              # rad per tick
# rad the jaw setpoint may run ahead of the MEASURED jaws once they are blocked: squeeze torque <= kp x lead.
# The jaws travel ~34 mm per rad, so 0.1 N.m at the motor is ~3 N at the pads (before the mechanism's own friction).
# UNTESTED ON FOOD: try a sacrificial grape first and tune these.
GRIP_LEADS = {"soft": 0.015,  # 0.3 N.m ~ 9 N : strawberry, grape, anything that bruises
              "firm": 0.05}   # 1.0 N.m ~ 30 N: can, tape measure, apple
FIRM_FOODS = {"can", "tape measure", "tape", "apple", "mug", "cup", "bottle", "block", "carrot"}
SOFT_FOODS = {"strawberry", "grape", "raspberry", "blueberry", "tomato", "banana", "egg", "marshmallow"}
GRIP_LEAD = GRIP_LEADS["soft"]   # default to gentle: dropping is bad, crushing is unrecoverable
GRIP_FREE_LEAD = 0.03         # while the jaws are still MOVING they may be led by this much (0.6 N.m: enough to beat the mechanism's
                              # friction). Once they stall on something for GRIP_STALL_TICKS the lead drops to the food's limit.
GRIP_STALL_RAD, GRIP_STALL_TICKS = 0.003, 2
SUBSTEPS = 6                  # motor commands per 30 Hz policy tick (~200 Hz, the rate arm_replay flies at): no 0.02 rad torque steps
GRIP_OPEN_M = 0.0375          # the policies' gripper unit: metres per finger, 0 closed .. 0.0375 open
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
    def __init__(self, motors, kp, kd, gravity, laps, factors=None, lock=None, settle_s=1.0,
                 grip_motor=None, grip_gains=(20.0, 0.5), coulomb=None, sleep=time.sleep):
        from replay_pose2 import FIT, MotorError, Sender, load_fit
        if factors is None:                      # the fit made from this arm's own torque logs (arm_replay/gravity_fit.json)
            factors, fit_c, _ = load_fit(FIT)
            coulomb = fit_c if coulomb is None else coulomb
        self._sleep, self.grip_lead = sleep, GRIP_LEAD
        self.MotorError = MotorError
        self.sign, self.offset, stops = load_joint_map()
        assert np.all(self.sign == 1), "the measured map has sign +1 on every joint; anything else is a new measurement"
        self.laps = np.asarray(laps, float)
        lo, hi = stops[:, 0] - TAU * self.laps, stops[:, 1] - TAU * self.laps          # raw window this power session
        self.raw_lo, self.raw_hi = lo + STOP_MARGIN, hi - STOP_MARGIN
        self.sender = Sender(motors, kp, kd, gravity, factors, lock=lock, **({} if coulomb is None else {"coulomb": coulomb}))
        self.motors, self.settle_s = motors, settle_s
        self.mode, self.fault = "idle", ""
        self._sp = None               # last COMMANDED raw setpoint. The only thing new targets are built from.
        self._seed = None             # raw pose the arm was enabled in = the only pose it may be released in
        self._t0 = None
        self._q_prev, self._qd = None, np.zeros(6)
        self._trail, self._over, self.peak_excess = [], 0, np.zeros(6)
        self.grip_motor, self.grip_gains = grip_motor, grip_gains
        self._grip_sp = self._grip_pos = None    # last commanded / last reported raw gripper position
        self._grip_stall = 0

    # ------------------------------------------------------------------ coordinates
    def to_model(self, raw):
        return np.asarray(raw, float) + TAU * self.laps - self.offset

    def to_raw(self, q_model):
        return np.asarray(q_model, float) + self.offset - TAU * self.laps

    # ------------------------------------------------------------------ lifecycle
    def begin(self, seed_raw, grip_raw=None):
        """Call ONCE, right after the driver has enabled each motor holding `seed_raw` (the read-then-enable-then-
        hold sequence from replay_pose2.py). From here on no live motor is ever read again."""
        self._seed = np.asarray(seed_raw, float).copy()
        self._sp = self._seed.copy()
        # The arm rests ON its hard stops when unpowered, i.e. outside the safety margin. Clipping it into the
        # window would make joints lift off their stops unasked on the first command. A joint that starts beyond
        # the margin may stay there; it just may not go any further out.
        self.raw_lo, self.raw_hi = np.minimum(self.raw_lo, self._seed), np.maximum(self.raw_hi, self._seed)
        if grip_raw is not None and self.grip_motor is not None:
            self._grip_sp = self._grip_pos = float(grip_raw)      # the jaws start by holding where they are
        self._t0, self.mode = time.time(), "run"

    def _gscale(self):
        return min(1.0, (time.time() - self._t0) / max(self.settle_s, 1e-6))          # feed-forward ramps in

    # ------------------------------------------------------------------ ArmInterface
    def read(self) -> ArmState:
        q = self.to_model(self.sender.last["pos"] if np.any(self.sender.last["pos"]) else self._sp)
        if self._q_prev is not None:
            self._qd = 0.6 * self._qd + 0.4 * (q - self._q_prev) * self.rate_hz
        self._q_prev = q.copy()
        return ArmState(q, self._qd.copy(), self.grip_opening_m(), float("nan"), time.time() - (self._t0 or time.time()))

    def send(self, q_target, grip_target=None):
        if self.mode == "backoff":
            return self._backoff_tick()
        if self.mode != "run":
            return self._hold_tick()
        if grip_target is not None and (self.grip_motor is None or self._grip_sp is None):
            self._enter_hold("a gripper target was sent but this session has no gripper motor (connect it, begin(grip_raw=...))")
            return
        if grip_target is not None:
            self._grip_tick(grip_target)
        want = np.clip(self.to_raw(q_target), self.raw_lo, self.raw_hi)
        step = np.clip(want - self._sp, -self.max_step_rad, self.max_step_rad)        # from the COMMANDED setpoint
        try:
            for k in range(SUBSTEPS):            # same 0.02 rad, delivered as 6 small setpoints 5 ms apart
                tick = time.time()
                sp = self._sp + step / SUBSTEPS
                self.sender(sp, step * self.rate_hz, self._gscale(), "run")
                self._sp = sp
                if k < SUBSTEPS - 1:
                    self._sleep(max(0.0, 1.0 / (self.rate_hz * SUBSTEPS) - (time.time() - tick)))
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

    def _grip_tick(self, grip_m=None):
        """One gripper command. None = keep the last setpoint. The setpoint may lead the measured jaws by at most
        GRIP_LEAD, so a blocked jaw squeezes with a bounded torque instead of winding up to the motor's limit."""
        if grip_m is not None:
            opn, cls_ = GRIP_RAW
            f = float(np.clip(grip_m / GRIP_OPEN_M, 0.0, 1.0))
            lo, hi = sorted((opn, cls_))
            want = float(np.clip(cls_ + f * (opn - cls_), lo + GRIP_MARGIN, hi - GRIP_MARGIN))
            sp = self._grip_sp + float(np.clip(want - self._grip_sp, -GRIP_STEP, GRIP_STEP))
            lead = self.grip_lead if self._grip_stall >= GRIP_STALL_TICKS else max(GRIP_FREE_LEAD, self.grip_lead)
            self._grip_sp = float(np.clip(sp, self._grip_pos - lead, self._grip_pos + lead))
        before = self._grip_pos
        st = self.grip_motor.command(pos=self._grip_sp, vel=0.0, kp=self.grip_gains[0], kd=self.grip_gains[1], torque=0.0)
        if st is not None:
            self._grip_pos = float(st.position)
            pushing = grip_m is not None and abs(self._grip_sp - self._grip_pos) > 0.5 * self.grip_lead
            self._grip_stall = self._grip_stall + 1 if pushing and abs(self._grip_pos - before) < GRIP_STALL_RAD else (self._grip_stall if grip_m is None else 0)

    def set_food(self, name):
        """Pick the squeeze limit from what is being picked up. Unknown food is treated as soft."""
        kind = "firm" if name and name.strip().lower() not in SOFT_FOODS and name.strip().lower() in FIRM_FOODS else "soft"
        self.grip_lead = GRIP_LEADS[kind]
        return kind

    def grip_opening_m(self):
        if self._grip_pos is None:
            return float("nan")
        opn, cls_ = GRIP_RAW
        return float(np.clip((self._grip_pos - cls_) / (opn - cls_), 0.0, 1.0) * GRIP_OPEN_M)

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
        if self._grip_sp is not None:
            self._grip_tick()
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
            for mot in ([self.grip_motor] if self.grip_motor is not None else []) + list(reversed(self.motors)):
                mot.disable()
            self.mode = "released"
            return "at rest: motors disabled"
        self.mode = "hold"
        return (f"NOT at rest (worst joint {err.max():.3f} rad from the enable pose): motors LEFT ON and holding. "
                "Support the arm by hand before anyone cuts power or kills this process.")

    # ------------------------------------------------------------------ hardware. Never called by the pipeline tests.
    @staticmethod
    def infer_laps(raw, stops, slack=0.15):
        """Which 2*pi lap each encoder woke up on: the one lap that puts the reading inside the measured stops."""
        laps = []
        for i, (q, (lo, hi)) in enumerate(zip(raw, stops)):
            fits = [k for k in range(-3, 4) if lo - slack <= q + TAU * k <= hi + slack]
            if len(fits) != 1:
                raise RuntimeError(f"J{i + 1} reads {q:+.4f}: {len(fits)} laps fit its measured range. Do not guess - re-measure.")
            laps.append(fits[0])
        return laps

    @classmethod
    def connect(cls, laps=None, token=None, factors=None, gripper=True):
        """Same sequence as arm_replay/replay_pose2.main(), which has run on this arm: the watchdog register must
        read 0 (it is only READ here, never written), no latched faults, session check against the measured stops,
        then per motor: read -> enable -> hold where it is. Nothing moves until send() is given a different target."""
        if (token or os.environ.get("YAM_REAL_ARM")) != TOKEN:
            raise RuntimeError(f"refusing to touch hardware. A person at the arm, E-stop in hand, sets YAM_REAL_ARM={TOKEN}")
        if not sys.stdin.isatty() or input("Type MOVE to energise the real arm: ").strip() != "MOVE":
            raise RuntimeError("not confirmed at a terminal by a person")
        from replay_pose2 import Gravity
        sys.path.insert(0, str(Path.home() / "openyam"))
        from openyam.arm import ArmConfig, OpenYAMArm
        from openyam.gsusb import CanBus
        _, _, stops = load_joint_map()
        bus = CanBus()
        try:
            drv = OpenYAMArm(ArmConfig(joint_limits=[(-20.0, 20.0)] * 6), bus=bus, include_gripper=gripper)
            joints, gm = (drv.motors[:6], drv.motors[6]) if gripper else (drv.motors, None)
            here = []
            for m in drv.motors:
                if m.read_register("TIMEOUT") != 0:
                    raise RuntimeError(f"0x{m.motor_id:02X}: watchdog TIMEOUT is not 0. Not touching it; not driving.")
                st = m.read_state()
                if st is None or st.error not in ("disabled", "normal"):
                    raise RuntimeError(f"0x{m.motor_id:02X}: {'no state' if st is None else 'latched ' + repr(st.error)}. Power-cycle to clear.")
                here.append(st.position)
            found = cls.infer_laps(here[:6], stops)
            if laps is not None and list(laps) != found:
                raise RuntimeError(f"laps given {list(laps)} but the encoders say {found}")
            if gm is not None and not (min(GRIP_RAW) - 0.3 <= here[6] <= max(GRIP_RAW) + 0.3):
                raise RuntimeError(f"gripper reads {here[6]:+.3f}, outside its measured travel {GRIP_RAW}: its encoder lapped. Re-measure.")
            arm = cls(joints, drv.gains_kp[:6], drv.gains_kd[:6], Gravity(found), found, factors=factors, lock=drv._lock,
                      grip_motor=gm, grip_gains=(drv.gains_kp[6], drv.gains_kd[6]) if gm is not None else (20.0, 0.5))
            print(f"[RealArm] laps {found}; model pose {np.round(arm.to_model(here[:6]), 3)}", flush=True)
            seed = []
            for m, kp, kd in zip(drv.motors, drv.gains_kp, drv.gains_kd):
                q = m.read_state().position; m.enable()
                m.command(pos=q, vel=0.0, kp=kp, kd=kd, torque=0.0); seed.append(q)
            drv._active = True
            import atexit
            atexit.unregister(drv._emergency_disable)   # the driver's exit hook makes every motor limp: it must never drop an arm that is off its stops
            arm._driver, arm._bus = drv, bus                      # keep both alive: the driver's atexit DISABLES motors
            arm.begin(seed[:6], grip_raw=seed[6] if gm is not None else None)
            return arm
        except Exception:
            bus.close()
            raise


def hold_test(seconds: float, grip: bool) -> int:
    """FIRST powered test of this backend: enable, hold exactly where the arm already is (zero motion asked for),
    report the torques, release. --grip also closes the jaws halfway and reopens them. Ctrl-C ends it early. If it
    cannot confirm the arm is back at rest it KEEPS HOLDING until a second Ctrl-C (support the arm first)."""
    import signal
    arm = RealArm.connect(gripper=grip)
    stop = {"n": 0}
    signal.signal(signal.SIGINT, lambda *a: stop.__setitem__("n", stop["n"] + 1))
    g0 = arm.grip_opening_m()
    t0 = last = time.time()
    while time.time() - t0 < seconds and not stop["n"]:
        tick = time.time(); el = tick - t0
        gt = None if not grip else (0.5 * GRIP_OPEN_M if seconds / 3 < el < 2 * seconds / 3 else g0)
        arm.send(arm.to_model(arm._sp), grip_target=gt)
        if tick - last >= 1.0:
            last = tick; L = arm.sender.last
            print(f"  {arm.mode:5s} lag " + " ".join(f"{x:+.3f}" for x in L["lag"]) + " | torque " + " ".join(f"{x:+5.1f}" for x in L["torque"])
                  + " | ff " + " ".join(f"{x:+5.1f}" for x in L["tff"]) + (f" | jaws {1000 * arm.grip_opening_m():.1f} mm/finger" if grip else ""), flush=True)
        time.sleep(max(0.0, 1.0 / arm.rate_hz - (time.time() - tick)))
    print("[RealArm]", arm.shutdown(), flush=True)
    print("[RealArm] peak torque beyond feed-forward, N.m:", np.round(arm.peak_excess, 2), " (contact stop at", CONTACT_NM, ")")
    n0 = stop["n"]
    while arm.mode == "hold" and stop["n"] == n0:                # never let the process die with the arm in the air
        arm.send(None); time.sleep(1.0 / arm.rate_hz)
    return 0 if arm.mode == "released" else 1


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
        arm = RealArm(ms, kp, kd, grav, laps, settle_s=0.0, sleep=lambda t: None)
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
    check("gravity feed-forward rides in every command once ramped in", all(abs(c[4] - arm.sender.factors[2] * 6.5) < 1e-6 for c in ms[2].sent[-8:]),
          f"J3 tff {ms[2].sent[-1][4]:.2f} N.m (first tick {ms[2].sent[0][4]:.2f}: it ramps in, as on the arm)")
    check("no live motor is ever read after begin()", sum(m.reads for m in ms) == 0)
    check("joints resting on their stops are NOT moved by the first command", abs(arm._sp[2] - seed[2]) < 1e-12 and abs(arm._sp[0] - seed[0]) < 1e-12,
          f"J3 commanded {arm._sp[2]:.4f}, enabled at {seed[2]:.4f}")

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
    check("HOLD still carries feed-forward torque", abs(ms[2].sent[-1][4] - arm.sender.factors[2] * 6.5) < 1e-6)
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
    check("a gripper target is refused (holds) when the session has no gripper motor", arm.mode == "hold", arm.fault[:50])
    ms = [Motor(i, float(seed[i])) for i in range(6)]; gm = Motor(7, -2.80)
    arm = RealArm(ms, kp, kd, grav, laps, settle_s=0.0, sleep=lambda t: None, grip_motor=gm); arm.begin(seed, grip_raw=-2.80)
    for _ in range(220):
        arm.send(arm.to_model(seed), grip_target=0.0)
    check("the jaws close to the margin, never onto the mechanical end", abs(gm.pos - (GRIP_RAW[1] - GRIP_MARGIN)) < 0.02, f"{gm.pos:+.3f}")
    check("gripper opening is reported from the replies", arm.read().grip < 0.002, f"{arm.read().grip:.4f} m")
    gm.stuck = True; gm.pos = -1.5; arm._grip_pos = arm._grip_sp = -1.5
    for _ in range(30):
        arm.send(arm.to_model(seed), grip_target=0.0)
    check("jaws blocked by an object: the setpoint leads by <= GRIP_LEAD (bounded squeeze)", abs(arm._grip_sp - gm.pos) <= arm.grip_lead + 1e-9, f"lead {abs(arm._grip_sp - gm.pos):.3f} rad")
    check("soft food gets the gentle squeeze, a can the firm one, unknown food the gentle one",
          (arm.set_food("grape"), arm.set_food("can"), arm.set_food("mystery")) == ("soft", "firm", "soft"))
    n = len(gm.sent); arm.hold(); arm.send(None)
    check("a hold keeps commanding the jaws (the food is not dropped)", len(gm.sent) > n and arm.mode == "hold")

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
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    if "--hold-test" in sys.argv:                 # python real_arm.py --hold-test 10 [--grip]   (a PERSON runs this, at the arm)
        sys.exit(hold_test(float(sys.argv[sys.argv.index("--hold-test") + 1]), "--grip" in sys.argv))
    sys.exit(print(__doc__) or 0)
