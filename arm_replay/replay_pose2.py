#!/usr/bin/env python3
"""WORK IN PROGRESS - FIRST VERSION, JUST THE BEGINNING. NOT POLISHED, NOT SAFE NEAR PEOPLE OR UNATTENDED.
Known hazards (details and evidence in README.md):
  * FIXED BUT NOT YET TRIED ON THE ARM: a failed return used to DISABLE ALL MOTORS and the arm fell from wherever it was
    (happened once, 2026-09-20: shoulder 26 deg above its rest stop). Now a failed return HOLDS the arm where it is, retries
    up to 3 times (slower, looser guard) and only disables when the arm rests on its stops or you send a SECOND stop signal.
    The retry logic is covered by --selftest with fake motors only.
  * poses are raw motor numbers, valid only until the arm is next powered off (laps); the endpoint depends on the START pose.
  * imports the motor driver from /home/asus/openyam, which is NOT in any git repo; paths are hard-coded.
Have a hand on the arm's power switch and the arm folded at rest before you energize it.

Replay a recorded pose with GRAVITY FEED-FORWARD - the recipe used by the vendor's own SDK (i2rt).

Why: with position control alone (kp*error) the arm sags under its own weight until the error builds
enough torque, the elbow could not even lift (it needed ~10 N.m; the position loop tops out near 11),
and restarting the setpoint from a measured (sagged) position made the arm fall. i2rt's motor loop and
OpenArm's both fix this the same way: every control tick send ONE MIT command per joint containing
    torque = gravity_torque(q) * per-joint factor + coulomb_friction * sign(velocity)
and take the state from the reply to that same command (never a separate read).

Kept from replay_pose.py: TIMEOUT=0 verified, read-then-enable-then-hold per motor, velocity feed-forward,
faults abort, guards against a stale recording (session check) and against large travel, --fraction, --yes.
New: gravity + friction feed-forward (ramped in over the settle time), a total-torque guard (the lag limit
shrinks by the feed-forward so the TOTAL stays within 0.4 x TMAX), telemetry from the replies, and an exit that
continues from the LAST COMMANDED setpoint - no read and no re-seed of a live motor.

    python replay_pose2.py --selftest                       # offline tests, no hardware
    python replay_pose2.py --label 5 --fraction 0.6 --dry-run
    python replay_pose2.py --label 5 --fraction 0.6 --hold 5 --yes
    pkill -TERM -f replay_pose2.py                          # ramps back to the start pose, then disables
"""
from __future__ import annotations

import argparse, atexit, json, math, os, signal, sys, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

TAU = 2 * math.pi
STOPS = [(-4.0309, 1.6047), (0.0044, 3.6952), (0.0101, 3.9569),
         (-1.6726, 1.6577), (-1.5761, 1.5791), (-0.7822, 3.3892)]           # hand-measured hard stops (lap 0)
OFFSET = np.array([-1.4313, 0.0044, 0.0101, -0.0074, 0.0015, 1.3035])        # joint_map_measured.json
COULOMB = np.array([0.3, 0.3, 0.3, 0.06, 0.06, 0.06])                        # i2rt yam_v1.yml (N.m)
FACTORS_DEFAULT = "1.0,1.1,1.4,1.0,1.0,1.0"      # FALLBACK only (used with --no-fit): i2rt's 1.0,1.1,1.1,1.2,1.0,1.0 with the elbow raised. The real values live in gravity_fit.json
BUDGET = 0.4                                      # total torque budget as a fraction of each motor's TMAX (same as the driver's lag guard)
MIN_LAG = 0.05                                    # was 0.03: with the fitted feed-forward the lag left over is the joint's friction band (elbow ~2 N.m / kp 80 = 0.025 rad)
RETURN_MIN_LAGS = (0.10, 0.15, 0.20)              # lag-guard floor (rad) for return attempts 1..3 - run 5 tripped at exactly the 0.03 floor
RETURN_SLOWDOWN = 0.5                             # each retry is this much slower than the last
RETRY_PAUSE = 2.0                                 # s held motionless where it stopped before the next attempt
REST_TOL = 0.12                                   # rad: shoulder and elbow this close to their lower stops = folded at rest, safe to disable
EVENTS = []                                       # every log line, saved in the telemetry json
FIT = Path(__file__).resolve().parent / "gravity_fit.json"           # written by gravity_export.py from the arm's own torque logs
POSES = Path(__file__).resolve().parent / "poses.json"
LOGDIR = Path(__file__).resolve().parent / "logs"
MENAGERIE = [Path(os.environ["YAM_MENAGERIE"])] if os.environ.get("YAM_MENAGERIE") else []
MENAGERIE += [Path.home() / "HackMIT_2026/third_party/mujoco_menagerie/i2rt_yam"]


def travel_ok(travel, max_travel):
    """Only the big joints (J1-J3, which swing the whole arm) are limited to max_travel. The wrist may use its FULL mechanical range
    (e.g. from folded on its stop to the far end): it can never travel further than that, because the start and the target are both
    already required to lie inside the joint's measured range."""
    return float(np.abs(np.asarray(travel, float)[:3]).max()) <= max_travel


def load_fit(path):
    """(factors, coulomb, note) from gravity_fit.json; the fit is relative to the stock menagerie model that Gravity loads."""
    d = json.load(open(path)); f = np.array(d["factor"], float); c = np.array(d["coulomb_Nm"], float)
    assert f.shape == (6,) and c.shape == (6,) and np.all(f > 0.5) and np.all(f < 2.0) and np.all(c >= 0) and np.all(c < 5), "gravity_fit.json out of range"
    return f, c, f"fitted {d.get('fitted', '?')}"


def log(*a):
    line = f"[{time.strftime('%H:%M:%S')}] " + " ".join(str(x) for x in a)
    EVENTS.append(line); print(line, flush=True)


class MotorError(RuntimeError):
    pass


# --------------------------------------------------------------------------------- gravity model
class Gravity:
    """Torque each joint must supply to hold the arm still: MuJoCo's qfrc_bias with the model's free
    gravity compensation switched OFF (the menagerie MJCF turns it on for every link; the real arm has none).
    qfrc_bias is used rather than inverse dynamics so joint-limit constraints can never leak into it."""
    def __init__(self, lap, xml=None):
        import mujoco
        self.mj = mujoco
        for c in ([Path(xml)] if xml else MENAGERIE):
            c = c if c.name == "yam.xml" else c / "yam.xml"
            if c.exists():
                xml = c; break
        else:
            raise FileNotFoundError("mujoco_menagerie/i2rt_yam/yam.xml not found; set YAM_MENAGERIE")
        self.M = mujoco.MjModel.from_xml_path(str(xml)); self.M.body_gravcomp[:] = 0.0
        self.D = mujoco.MjData(self.M); self.lap = np.asarray(lap, float)

    def __call__(self, raw):
        self.D.qpos[:] = 0.0; self.D.qpos[:6] = np.asarray(raw, float) + TAU * self.lap - OFFSET; self.D.qvel[:] = 0.0
        self.mj.mj_forward(self.M, self.D)
        return self.D.qfrc_bias[:6].copy()


# --------------------------------------------------------------------------------- one command per joint per tick
class Sender:
    def __init__(self, motors, kp, kd, gravity, factors, coulomb=COULOMB, use_ff=True, lock=None, min_lag=MIN_LAG):
        self.motors, self.kp, self.kd, self.gravity = motors, np.asarray(kp, float), np.asarray(kd, float), gravity
        self.factors, self.coulomb, self.use_ff, self.lock = np.asarray(factors, float), np.asarray(coulomb, float), use_ff, lock
        self.cap = np.array([BUDGET * m.spec.t_max for m in motors])
        self.last = {k: np.zeros(6) for k in ("pos", "torque", "tff", "lag", "limit")}
        self.max_abs_lag = np.zeros(6); self.series = []
        self.min_lag, self.strict = min_lag, True     # strict: a fault or lag over the limit raises; non-strict: only recorded in .problem
        self.last_sp, self.problem = None, None

    def feedforward(self, sp, vel, gscale):
        if not self.use_ff: return np.zeros(6)
        g = gscale * self.factors * self.gravity(sp)
        fric = self.coulomb * np.tanh(vel / 0.02)          # smooth version of sign(v): what the fit used
        return np.clip(g + fric, -self.cap, self.cap)

    def __call__(self, sp, vel=None, gscale=1.0, phase=""):
        sp = np.asarray(sp, float); vel = np.zeros(6) if vel is None else np.asarray(vel, float)
        tff = self.feedforward(sp, vel, gscale)
        row = np.zeros((4, 6)); trip = None; self.last_sp = sp.copy(); self.problem = None
        ctx = self.lock if self.lock is not None else _NullLock()
        with ctx:
            for i, m in enumerate(self.motors):
                if not m.enabled: continue
                st = m.command(pos=float(sp[i]), vel=float(vel[i]), kp=float(self.kp[i]), kd=float(self.kd[i]), torque=float(tff[i]))
                if st is None: continue
                if not st.ok:
                    if self.strict: raise MotorError(f"motor 0x{m.motor_id:02X} fault during motion: {st.error}")
                    self.problem = f"motor 0x{m.motor_id:02X} fault: {st.error}"; continue
                lag = st.position - sp[i]; limit = max(self.min_lag, (self.cap[i] - abs(tff[i])) / self.kp[i])
                row[:, i] = (st.position, st.torque, lag, tff[i]); self.last["limit"][i] = limit
                self.max_abs_lag[i] = max(self.max_abs_lag[i], abs(lag))
                if abs(lag) > limit and trip is None:
                    trip = (f"motor 0x{m.motor_id:02X} is {abs(lag):.3f} rad behind its setpoint (limit {limit:.3f} rad = the torque budget "
                            f"{BUDGET:.0%} of {m.spec.t_max} N.m minus the {abs(tff[i]):.1f} N.m of feed-forward). Stopping.")
        for j, k in enumerate(("pos", "torque", "lag", "tff")): self.last[k] = row[j]
        self.series.append((time.time(), phase, row.copy()))
        if trip:
            if self.strict: raise MotorError(trip)
            self.problem = trip


class _NullLock:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def ramp_to(send, q0, q1, speed, rate, hold_s=0.0, now=time.time, sleep=time.sleep, phase="return"):
    """Cosine-eased move from q0 to q1 using only commands (no reads). Starts AT q0, the last commanded setpoint."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float); delta = float(np.max(np.abs(q1 - q0)))
    if delta > 1e-4:
        T = (math.pi / 2) * delta / speed; t0 = now()
        while True:
            tick = now(); el = tick - t0
            if el >= T: break
            u = math.pi * el / T; a = 0.5 * (1 - math.cos(u)); adot = (math.pi / (2 * T)) * math.sin(u)
            send(q0 + (q1 - q0) * a, (q1 - q0) * adot, 1.0, phase)
            sleep(max(0.0, 1.0 / rate - (now() - tick)))
    end = now() + hold_s
    while now() < end:
        tick = now(); send(q1, None, 1.0, phase + "-hold"); sleep(max(0.0, 1.0 / rate - (now() - tick)))
    return q1


# --------------------------------------------------------------------------------- ending a run without dropping the arm
def at_rest(pos, window, tol=REST_TOL):
    """True if shoulder (J2) and elbow (J3) are on, or within tol of, their lower stops: the folded rest pose, the only place
    where cutting the motor power is safe. All-zero (no reply seen yet) counts as unknown, i.e. NOT at rest."""
    pos = np.asarray(pos, float)
    return bool(np.any(pos)) and all(pos[i] <= window[i][0] + tol for i in (1, 2))


def freeze_point(sender):
    """Where the arm actually is (from the last replies). Hold THERE, not at the setpoint it failed to reach: that
    stops any push against an obstacle but, unlike a re-seed from a limp read, keeps full gain and gravity feed-forward on."""
    return np.where(sender.last["pos"] != 0.0, sender.last["pos"], sender.last_sp)


def hold_here(sender, sp, seconds, rate, now, sleep, cont=lambda: True, phase="hold-after-stop"):
    """Keep commanding sp (gain + feed-forward, motors stay enabled). Never raises: bus hiccups are logged, not fatal."""
    end, quiet_until = now() + seconds, 0.0
    while now() < end and cont():
        tick = now()
        try: sender(sp, None, 1.0, phase)
        except Exception as exc:  # noqa: BLE001
            if tick >= quiet_until: log(f"  (hold: command failed: {exc!r})"); quiet_until = tick + 2.0
        sleep(max(0.0, 1.0 / rate - (now() - tick)))


def controlled_finish(sender, last_sp, seed, window, stop, lower_speed, rate, now=time.time, sleep=time.sleep, say=None):
    """Bring the arm back to the start pose WITHOUT ever cutting power on an arm that is off its stops.
    Up to len(RETURN_MIN_LAGS) attempts, each slower and with a looser lag guard; after a failed attempt the arm is held where it
    stopped for RETRY_PAUSE s. If it cannot get back, or it got back but is not at rest, it stays powered and holding until a
    SECOND stop signal (the first one is what started this procedure, unless the guard did). Returns "rest" or "operator":
    the reason it is now OK to disable."""
    say = say or log; n0 = stop["n"]; sp = np.asarray(last_sp, float); cont = lambda: stop["n"] <= n0; returned = False
    for n, floor in enumerate(RETURN_MIN_LAGS):
        if not cont(): break
        speed = lower_speed * RETURN_SLOWDOWN ** n; sender.min_lag, sender.strict = floor, True
        say(f"return attempt {n + 1}/{len(RETURN_MIN_LAGS)}: peak {speed:.2f} rad/s, lag-guard floor {floor} rad (from the last commanded setpoint, no reads)")
        try:
            ramp_to(sender, sp, seed, speed, rate, hold_s=0.5, now=now, sleep=sleep); returned = True; break
        except Exception as exc:  # noqa: BLE001
            sender.strict = False; sp = freeze_point(sender)
            say(f"return attempt {n + 1} stopped: {exc}")
            say("HOLDING where it is - motors stay powered with gravity feed-forward, NOT disabled")
            hold_here(sender, sp, RETRY_PAUSE, rate, now, sleep, cont)
    sender.strict = False
    if returned:
        if at_rest(sender.last["pos"], window): say("returned to the start pose, resting on its stops"); return "rest"
        sp = np.asarray(seed, float)
        say("returned to the start pose, but the shoulder/elbow are NOT on their stops: disabling here would drop the arm")
    elif cont():
        say("COULD NOT GET BACK to the start pose.")
    if cont(): say("STILL POWERED AND HOLDING. Support the arm, then send a second stop signal (Ctrl-C, or pkill -TERM -f replay_pose2.py) to disable.")
    last_say = now()
    while cont():
        tick = now()
        try: sender(sp, None, 1.0, "hold-final")
        except Exception: pass  # noqa: BLE001
        if tick - last_say >= 5.0:
            last_say = tick
            say("  still holding" + (f" | {sender.problem}" if sender.problem else "") + " | lag " + " ".join(f"{x:+.3f}" for x in sender.last["lag"]))
        sleep(max(0.0, 1.0 / rate - (now() - tick)))
    say("second stop signal: disabling now - the arm goes limp")
    return "operator"


# --------------------------------------------------------------------------------- offline self-test
def selftest() -> int:
    fails = []
    def check(name, ok, detail=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}"); (None if ok else fails.append(name))
    lap = [0, 1, 1, 0, 0, 0]; G = Gravity(lap)
    held = np.array([-1.776, -5.728, -6.257, 0.298, 0.121, 0.057]); g = G(held)
    print("gravity model (free compensation off) at the pose the 60% run reached:", np.round(g, 2))
    check("elbow gravity ~6.5 N.m (matches the earlier model calculation)", abs(g[2] - 6.5) < 0.5, f"J3={g[2]:.2f}")
    check("wrist gravity ~1.8 N.m (matches what the run implied, 1.9)", abs(g[3] - 1.8) < 0.4, f"J4={g[3]:.2f}")
    seed = np.array([-1.5368, -6.2789, -6.2732, 1.6581, 0.0452, 0.0036]); gs = G(seed)
    check("no constraint blow-up at the rest pose (wrist is beyond the MJCF limit)", np.all(np.abs(gs) < 10), f"G(rest)={np.round(gs, 2)}")
    # ramp continuity with a fake clock
    clk = SimpleNamespace(t=0.0); rec = []
    send = lambda sp, v, gscale=1.0, phase="": (rec.append((clk.t, np.array(sp), None if v is None else np.array(v))), setattr(clk, "t", clk.t + 0.004))
    q0 = np.array([0., -6.0, -6.1, 0.3, 0.1, 0.05]); q1 = q0 + np.array([-0.3, 0.6, 0.15, -0.8, 0.1, 0.05])
    ramp_to(send, q0, q1, 0.25, 200.0, hold_s=0.2, now=lambda: clk.t, sleep=lambda x: setattr(clk, "t", clk.t + x))
    sps = np.array([r[1] for r in rec]); steps = np.abs(np.diff(sps, axis=0)).max(axis=1)
    check("return ramp starts at the last commanded setpoint (no jump)", np.abs(sps[0] - q0).max() < 0.01, f"first step {np.abs(sps[0]-q0).max():.4f} rad")
    check("return ramp is smooth (largest tick-to-tick step)", steps.max() < 1.6 * 0.25 / 200 * 1.5, f"{steps.max():.5f} rad/tick")
    check("return ramp ends exactly at the target and holds", np.abs(sps[-1] - q1).max() < 1e-9)
    # sender guards with fake motors
    def fm(i, t_max, pos=0.0, ok=True):
        return SimpleNamespace(motor_id=i + 1, enabled=True, spec=SimpleNamespace(t_max=t_max), pos=pos, ok=ok,
                               command=lambda pos, vel, kp, kd, torque, _s=None: None)
    motors = []
    for i in range(6):
        m = fm(i, 28.0 if i < 3 else 10.0); m.sent = []
        def cmd(pos, vel, kp, kd, torque, m=m):
            m.sent.append((pos, vel, kp, kd, torque)); return SimpleNamespace(ok=m.ok, error="normal" if m.ok else "overcurrent", position=m.pos, torque=torque)
        m.command = cmd; motors.append(m)
    kp, kd = [80] * 3 + [10] * 3, [5] * 3 + [1.5] * 3
    S = Sender(motors, kp, kd, G, [float(x) for x in FACTORS_DEFAULT.split(",")], lock=None)
    sp = held.copy()
    for i, m in enumerate(motors): m.pos = sp[i]
    S(sp, None, 1.0)
    tq = np.array([m.sent[-1][4] for m in motors])
    check("feed-forward torque is sent in every command", abs(tq[2] - 1.4 * g[2]) < 1e-6 and abs(tq[3] - 1.0 * g[3]) < 1e-6, f"tff={np.round(tq, 2)}")
    S(sp, None, 0.0); check("gscale=0 sends zero feed-forward (ramp-in start)", all(abs(m.sent[-1][4]) < 1e-9 for m in motors))
    S(sp, np.array([0.2, 0, 0, 0, 0, -0.2]), 0.0); check("friction term follows the commanded direction", abs(motors[0].sent[-1][4] - 0.3) < 1e-6 and abs(motors[5].sent[-1][4] + 0.06) < 1e-6)
    big = Sender(motors, kp, kd, G, [50.0] * 6); big(sp, None, 1.0)
    check("feed-forward is clipped to the torque budget", all(abs(m.sent[-1][4]) <= BUDGET * m.spec.t_max + 1e-9 for m in motors), f"max {max(abs(m.sent[-1][4]) for m in motors):.1f} N.m")
    motors[2].pos = sp[2] - 0.2
    try: S(sp, None, 1.0); check("lag guard trips on a blocked elbow", False)
    except MotorError as e: check("lag guard trips on a blocked elbow", True, f"({str(e)[:70]}...)")
    motors[2].pos = sp[2]; motors[4].ok = False
    try: S(sp, None, 1.0); check("a motor fault aborts", False)
    except MotorError: check("a motor fault aborts", True)
    motors[4].ok = True
    lim = S.last["limit"]; check("guard limit shrinks by the feed-forward (elbow tighter than J1)", lim[2] < lim[0], f"limits {np.round(lim, 3)}")
    # ---- ending a run: fake clock + fake motors whose lag is scripted -------------------------------------------------
    win = [(lo - TAU * k, hi - TAU * k) for (lo, hi), k in zip(STOPS, lap)]
    rest_seed = np.array([-1.5368, -6.2789, -6.2739, 1.6577, 0.0086, 0.001])
    check("at_rest: folded pose yes / 60% pose no / no reply no", at_rest(rest_seed, win) and not at_rest(held, win) and not at_rest(np.zeros(6), win))
    def rig(lag_at, trip_stop_after=None):
        clk = SimpleNamespace(t=0.0); stop = {"n": 0}; cmds = []; ms = []
        for i in range(6):
            m = SimpleNamespace(motor_id=i + 1, enabled=True, spec=SimpleNamespace(t_max=28.0 if i < 3 else 10.0))
            def cmd(pos, vel, kp, kd, torque, i=i):
                cmds.append((clk.t, i, pos, kp, torque))
                return SimpleNamespace(ok=True, error="normal", position=pos + lag_at(clk.t, i), torque=torque)
            m.command = cmd; ms.append(m)
        def sleep(x):
            clk.t += x
            if trip_stop_after is not None and clk.t > trip_stop_after: stop["n"] = 1
        return clk, stop, cmds, Sender(ms, kp, kd, G, [float(x) for x in FACTORS_DEFAULT.split(",")]), sleep
    # A: elbow 0.14 rad behind until t=1.0 s -> attempt 1 trips, arm is held (not disabled), attempt 2 gets home
    clk, stop, cmds, S2, sleep = rig(lambda t, i: 0.14 if (i == 2 and t < 1.0) else 0.0); said = []
    r = controlled_finish(S2, held, rest_seed, win, stop, 0.20, 200.0, now=lambda: clk.t, sleep=sleep, say=lambda *a: said.append(" ".join(map(str, a))))
    check("failed return attempt -> retried and finished at rest", r == "rest" and any("attempt 2/3" in x for x in said), f"result={r}")
    pause = [c for c in cmds if c[1] == 2 and 0.2 < c[0] < 1.8]
    check("after the trip it kept commanding with gain + feed-forward (never limp)", len(pause) > 100 and all(c[3] > 0 and abs(c[4]) > 1.0 for c in pause), f"{len(pause)} commands")
    check("it holds at the MEASURED position (setpoint + 0.14), not the setpoint it missed", all(abs(c[2] - (held[2] + 0.14)) < 1e-9 for c in pause))
    # B: elbow never gets there -> after 3 attempts it stays powered and holding until a 2nd signal; never returns early
    clk, stop, cmds, S2, sleep = rig(lambda t, i: 0.3 if i == 2 else 0.0, trip_stop_after=40.0); said = []
    r = controlled_finish(S2, held, rest_seed, win, stop, 0.20, 200.0, now=lambda: clk.t, sleep=sleep, say=lambda *a: said.append(" ".join(map(str, a))))
    n_late = sum(1 for c in cmds if 10.0 < c[0] < 39.0)
    check("all 3 attempts fail -> keeps commanding until the operator signals", r == "operator" and n_late > 1000 and clk.t >= 40.0, f"result={r}, {n_late} commands between 10 and 39 s")
    check("...and says so loudly", any("STILL POWERED AND HOLDING" in x for x in said) and any("attempt 3/3" in x for x in said))
    # C: gets home but not onto its stops (started in the air) -> must not disable
    clk, stop, cmds, S2, sleep = rig(lambda t, i: 0.0, trip_stop_after=20.0); said = []
    r = controlled_finish(S2, held, held, win, stop, 0.20, 200.0, now=lambda: clk.t, sleep=sleep, say=lambda *a: said.append(" ".join(map(str, a))))
    check("returned but not at rest -> keeps holding until the operator signals", r == "operator" and clk.t >= 20.0)
    # D: relaxed floor and non-strict mode
    clk, stop, cmds, S2, sleep = rig(lambda t, i: 0.06 if i == 2 else 0.0)
    try: S2(held); check(f"elbow 0.06 rad behind trips the normal {MIN_LAG} floor", False)
    except MotorError: check(f"elbow 0.06 rad behind trips the normal {MIN_LAG} floor", True)
    S2.min_lag = 0.10
    try: S2(held); check("...but not the 0.10 return floor", True)
    except MotorError: check("...but not the 0.10 return floor", False)
    S2.min_lag, S2.strict = 0.03, False
    try: S2(held); check("non-strict mode records a trip instead of raising", S2.problem is not None, S2.problem[:50])
    except MotorError: check("non-strict mode records a trip instead of raising", False)
    ff, fc, note = load_fit(FIT) if FIT.exists() else (None, None, "missing")
    check("gravity_fit.json loads and is in range", ff is not None, f"factors {None if ff is None else np.round(ff, 3)}, friction {None if fc is None else np.round(fc, 2)}")
    if ff is not None:
        Sf = Sender(motors, kp, kd, G, ff, coulomb=fc); tf = Sf.feedforward(held, np.zeros(6), 1.0)
        check("fitted feed-forward at the 60% pose: elbow ~9 N.m (measured hold: 10-11 incl. friction), inside the 11.2 budget", 8.0 < tf[2] < 11.2, f"tff={np.round(tf, 2)}")
        tv = Sf.feedforward(held, np.array([0, 0, 0.3, 0, 0, 0]), 1.0)
        check("moving up adds the elbow's friction, smoothly (tanh)", 1.5 < tv[2] - tf[2] < 2.1, f"+{tv[2]-tf[2]:.2f} N.m")
    check("travel guard: wrist may swing its full range (2.9 rad folded->far end)", travel_ok([0.1, 2.0, 1.0, 2.9, 0.0, 0.0], 2.5) and travel_ok([0, 0, 0, 3.3, 3.0, 4.1], 2.5))
    check("travel guard: shoulder/elbow/base still limited to 2.5 rad", not travel_ok([0, 2.6, 0, 0, 0, 0], 2.5) and not travel_ok([0, 0, 2.6, 0, 0, 0], 2.5) and not travel_ok([2.6, 0, 0, 0, 0, 0], 2.5))
    check("driver's atexit disable is unregistered in main()", "atexit.unregister(arm._emergency_disable)" in Path(__file__).read_text())
    print("\nSELFTEST:", "ALL PASSED" if not fails else f"FAILED: {fails}"); return 1 if fails else 0


# --------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label"); ap.add_argument("--poses", default=str(POSES))
    ap.add_argument("--fraction", type=float, default=1.0); ap.add_argument("--max-travel", type=float, default=2.5, help="largest move (rad) allowed for J1-J3; the wrist J4-J6 may use its full range")
    ap.add_argument("--speed", type=float, default=0.25); ap.add_argument("--lower-speed", type=float, default=0.20)
    ap.add_argument("--rate", type=float, default=200.0); ap.add_argument("--settle", type=float, default=1.0)
    ap.add_argument("--hold", type=float, default=5.0)
    ap.add_argument("--min-lag", type=float, default=MIN_LAG, help="lag-guard floor in rad while moving/holding (the return uses looser ones)")
    ap.add_argument("--factors", default=None, help="per-joint gravity factors (default: from gravity_fit.json)")
    ap.add_argument("--fit", default=str(FIT), help="gravity/friction fit file"); ap.add_argument("--no-fit", action="store_true", help="ignore the fit: old vendor-style factors and friction")
    ap.add_argument("--gravity-scale", type=float, default=1.0, help="multiplies all factors (0 = no feed-forward)")
    ap.add_argument("--no-ff", action="store_true", help="disable all feed-forward (old behaviour, but with the new exit)")
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--yes", action="store_true"); ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest: return selftest()
    if not args.label: ap.error("--label is required")
    if not (0.0 < args.fraction <= 1.0): log("ABORT: --fraction must be in (0, 1]"); return 1
    if not args.dry_run and not args.yes: log("REFUSING to energize the arm without --yes (use --dry-run to just check)"); return 1
    coulomb, fit_note = COULOMB, "no fit (vendor-style defaults)"
    if args.no_fit or not Path(args.fit).exists(): base_factors = np.array([float(x) for x in (args.factors or FACTORS_DEFAULT).split(",")])
    else:
        base_factors, coulomb, fit_note = load_fit(args.fit)
        if args.factors: base_factors = np.array([float(x) for x in args.factors.split(",")]); fit_note += " (factors overridden)"
    factors = base_factors * args.gravity_scale
    log(f"feed-forward: factors {np.round(factors, 3)}  friction {np.round(coulomb, 2)} N.m  [{fit_note}]")
    rec = {p["label"]: p for p in json.load(open(args.poses))["poses"]}.get(args.label)
    if rec is None: log(f"ABORT: no pose '{args.label}' in {args.poses}"); return 1
    recorded, laps = [float(x) for x in rec["encoder_raw"][:6]], [int(k) for k in rec["lap"][:6]]
    window = [(lo - TAU * k, hi - TAU * k) for (lo, hi), k in zip(STOPS, laps)]
    log(f"recorded pose '{args.label}' ({rec['time']}): {[round(x, 4) for x in recorded]}  laps {laps}")
    gravity = Gravity(laps)

    sys.path.insert(0, "/home/asus/openyam")
    from openyam.arm import ArmConfig, MotorError as DriverError, OpenYAMArm
    from openyam.gsusb import CanBus
    bus = CanBus(); arm = OpenYAMArm(ArmConfig(joint_limits=[(lo - 0.15, hi + 0.15) for lo, hi in window]), bus=bus, include_gripper=False)
    atexit.unregister(arm._emergency_disable)   # the driver's exit hook disables (= limp) every motor: never let it drop an arm that is off its stops
    log(f"bus {bus.channel} via {bus.interface}, state={bus.state}")
    for m in arm.motors:
        tmo = m.read_register("TIMEOUT")
        if tmo != 0: log(f"ABORT: 0x{m.motor_id:02X} has TIMEOUT={tmo}. The watchdog must be OFF while driving."); bus.close(); return 1
    log("watchdog OFF on all six")
    here = []
    for m in arm.motors:
        st = m.read_state()
        if st is None: log(f"ABORT: 0x{m.motor_id:02X} gave no state"); bus.close(); return 1
        if st.error not in ("disabled", "normal"): log(f"ABORT: 0x{m.motor_id:02X} latched '{st.error}'. Power-cycle the arm to clear."); bus.close(); return 1
        here.append(st.position)
    log(f"measured pose: {[round(q, 4) for q in here]}")
    bad = [i for i, (q, (lo, hi)) in enumerate(zip(here, window)) if not (lo - 0.15 <= q <= hi + 0.15)]
    if bad:
        for i in bad: log(f"  J{i+1} reads {here[i]:+.4f}, but in the recording's numbers its real range is [{window[i][0]:+.3f}, {window[i][1]:+.3f}]")
        log("ABORT: the arm's numbers no longer match the recording (power cycle? lap changed). Re-teach the pose."); bus.close(); return 1
    if not at_rest(here, window): log("WARNING: the shoulder/elbow are NOT resting on their stops. At the end the arm will stay powered and holding until you send a second stop signal.")
    here = np.array(here); target = here + args.fraction * (np.array(recorded) - here); travel = target - here
    if any(not (lo - 0.05 <= t <= hi + 0.05) for t, (lo, hi) in zip(target, window)): log("ABORT: target outside the real range"); bus.close(); return 1
    if not travel_ok(travel, args.max_travel): log(f"ABORT: J1-J3 would move {np.abs(travel[:3]).max():.2f} rad (> --max-travel {args.max_travel}). The wrist is exempt: it may use its full range."); bus.close(); return 1
    delta = float(np.abs(travel).max())
    if delta < 1e-4: log("already there; nothing to do"); bus.close(); return 0
    ramp_T = (math.pi / 2) * delta / args.speed
    sender = Sender(arm.motors, arm.gains_kp, arm.gains_kd, gravity, factors, coulomb=coulomb, use_ff=not args.no_ff, lock=arm._lock, min_lag=args.min_lag)
    log(f"target       : {[round(float(q), 4) for q in target]}   (fraction {args.fraction})")
    log("travel       : " + "  ".join(f"j{i+1}:{x:+.3f}" for i, x in enumerate(travel)))
    log(f"furthest     : {delta:.3f} rad -> {ramp_T:.1f}s at peak {args.speed} rad/s, then hold {args.hold}s and ramp back")
    path = np.array([sender.feedforward(here + s * (target - here), None if False else np.zeros(6), 1.0) for s in np.linspace(0, 1, 41)])
    log("feed-forward torque N.m  at start: " + " ".join(f"{x:+5.1f}" for x in path[0]) + "\n" + " " * 11 + "at target: " + " ".join(f"{x:+5.1f}" for x in path[-1]) + "  |  peak along the path: " + " ".join(f"{np.abs(path[:, i]).max():4.1f}" for i in range(6)))
    log("torque budget (0.4 x TMAX) N.m: " + " ".join(f"{c:4.1f}" for c in sender.cap) + f"   factors {np.round(factors, 2)}" + ("   [feed-forward OFF]" if args.no_ff else ""))
    log("session check OK, travel guard OK")
    if args.dry_run: log("dry run: nothing enabled, nothing moved"); bus.close(); return 0

    stop = {"n": 0}   # counts stop signals: the 1st ends the run and starts the return, a 2nd (only after a failed return) disables
    signal.signal(signal.SIGINT, lambda *a: stop.__setitem__("n", stop["n"] + 1)); signal.signal(signal.SIGTERM, lambda *a: stop.__setitem__("n", stop["n"] + 1))
    enabled, seed = [], []; last_sp = None; hold_rows = []
    try:
        for m, kp, kd in zip(arm.motors, arm.gains_kp, arm.gains_kd):
            q = m.read_state().position; st = m.enable(); enabled.append(m)
            m.command(pos=q, vel=0.0, kp=kp, kd=kd, torque=0.0); seed.append(q)
            log(f"enabled+holding 0x{m.motor_id:02X} at {q:+.4f} [{st.error}]")
        seed = np.array(seed); arm._active = True
        t0 = time.time(); last_log = t0; phase = ""; reached_at = None
        while stop["n"] == 0:
            tick = time.time(); el = tick - t0; vel = None; gscale = min(1.0, el / max(args.settle, 1e-6))
            if el < args.settle: sp, ph = seed, "settle"
            elif el < args.settle + ramp_T:
                u = math.pi * (el - args.settle) / ramp_T; a = 0.5 * (1 - math.cos(u)); adot = (math.pi / (2 * ramp_T)) * math.sin(u)
                sp = seed + (target - seed) * a; vel = (target - seed) * adot; ph = "move"
            else:
                sp, ph = target, "hold"
                if reached_at is None: reached_at = el
            if ph != phase: log({"settle": f"settling {args.settle}s (feed-forward ramping in)", "move": "MOVING", "hold": f"AT TARGET at t={el:.1f}s -- holding"}[ph]); phase = ph
            last_sp = np.array(sp)
            sender(sp, vel, gscale, ph)
            if ph == "hold": hold_rows.append(sender.last["lag"].copy())
            if tick - last_log >= 1.0:
                last_log = tick; L = sender.last
                log(f"  {ph:6s} lag " + " ".join(f"{x:+.3f}" for x in L["lag"]) + " | torque " + " ".join(f"{x:+5.1f}" for x in L["torque"]) + " | ff " + " ".join(f"{x:+5.1f}" for x in L["tff"]))
            if reached_at is not None and el - reached_at >= args.hold: log(f"held {args.hold}s"); break
            time.sleep(max(0.0, 1.0 / args.rate - (time.time() - tick)))
        else: log("stop requested")
    except (MotorError, DriverError) as exc: log(f"GUARD TRIPPED: {exc}")
    except Exception as exc:  # noqa: BLE001
        log(f"ERROR: {exc!r}")
    finally:
        reason = "never moved"
        if enabled and last_sp is not None:
            try: reason = controlled_finish(sender, last_sp, seed, window, stop, args.lower_speed, args.rate)
            except BaseException as exc:
                log(f"finish procedure crashed ({exc!r}). Motors are LEFT ENERGIZED holding their last command, NOT disabled. Use the power switch if needed.")
                raise
        for m in reversed(enabled): m.disable()
        log(f"all motors disabled ({reason})"); bus.close()
    if hold_rows:
        h = np.array(hold_rows); log("HOLD SUMMARY  mean lag: " + " ".join(f"{x:+.3f}" for x in h.mean(axis=0)) + "  | worst |lag| over the run: " + " ".join(f"{x:.3f}" for x in sender.max_abs_lag))
    LOGDIR.mkdir(exist_ok=True); path_out = LOGDIR / f"run2_{time.strftime('%Y%m%d_%H%M%S')}.json"
    json.dump({"args": vars(args), "factors": factors.tolist(), "finish": reason, "events": EVENTS, "series": [[t, ph, r.tolist()] for t, ph, r in sender.series[::5]]}, open(path_out, "w")); log(f"telemetry saved: {path_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
