#!/usr/bin/env python3
"""WORK IN PROGRESS - FIRST VERSION, JUST THE BEGINNING. NOT POLISHED, NOT SAFE NEAR PEOPLE OR UNATTENDED.
Known hazards (details and evidence in README.md):
  * if the return ramp aborts on the lag guard this script DISABLES ALL MOTORS and the arm falls from wherever it is
    (happened once, 2026-09-20: shoulder 26 deg above its rest stop). Fix: hold the last pose instead, loosen the guard on the return.
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

import argparse, json, math, os, signal, sys, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

TAU = 2 * math.pi
STOPS = [(-4.0309, 1.6047), (0.0044, 3.6952), (0.0101, 3.9569),
         (-1.6726, 1.6577), (-1.5761, 1.5791), (-0.7822, 3.3892)]           # hand-measured hard stops (lap 0)
OFFSET = np.array([-1.4313, 0.0044, 0.0101, -0.0074, 0.0015, 1.3035])        # joint_map_measured.json
COULOMB = np.array([0.3, 0.3, 0.3, 0.06, 0.06, 0.06])                        # i2rt yam_v1.yml (N.m)
FACTORS_DEFAULT = "1.0,1.1,1.4,1.0,1.0,1.0"      # i2rt: 1.0,1.1,1.1,1.2,1.0,1.0. J3 measured ~1.6 with the stock model -> 1.4 is deliberately below it
BUDGET = 0.4                                      # total torque budget as a fraction of each motor's TMAX (same as the driver's lag guard)
MIN_LAG = 0.03
POSES = Path(__file__).resolve().parent / "poses.json"
LOGDIR = Path(__file__).resolve().parent / "logs"
MENAGERIE = [Path(os.environ["YAM_MENAGERIE"])] if os.environ.get("YAM_MENAGERIE") else []
MENAGERIE += [Path.home() / "HackMIT_2026/third_party/mujoco_menagerie/i2rt_yam"]


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


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
    def __init__(self, motors, kp, kd, gravity, factors, coulomb=COULOMB, use_ff=True, lock=None):
        self.motors, self.kp, self.kd, self.gravity = motors, np.asarray(kp, float), np.asarray(kd, float), gravity
        self.factors, self.coulomb, self.use_ff, self.lock = np.asarray(factors, float), np.asarray(coulomb, float), use_ff, lock
        self.cap = np.array([BUDGET * m.spec.t_max for m in motors])
        self.last = {k: np.zeros(6) for k in ("pos", "torque", "tff", "lag", "limit")}
        self.max_abs_lag = np.zeros(6); self.series = []

    def feedforward(self, sp, vel, gscale):
        if not self.use_ff: return np.zeros(6)
        g = gscale * self.factors * self.gravity(sp)
        fric = self.coulomb * np.where(np.abs(vel) > 0.02, np.sign(vel), 0.0)
        return np.clip(g + fric, -self.cap, self.cap)

    def __call__(self, sp, vel=None, gscale=1.0, phase=""):
        sp = np.asarray(sp, float); vel = np.zeros(6) if vel is None else np.asarray(vel, float)
        tff = self.feedforward(sp, vel, gscale)
        row = np.zeros((4, 6)); trip = None
        ctx = self.lock if self.lock is not None else _NullLock()
        with ctx:
            for i, m in enumerate(self.motors):
                if not m.enabled: continue
                st = m.command(pos=float(sp[i]), vel=float(vel[i]), kp=float(self.kp[i]), kd=float(self.kd[i]), torque=float(tff[i]))
                if st is None: continue
                if not st.ok:
                    raise MotorError(f"motor 0x{m.motor_id:02X} fault during motion: {st.error}")
                lag = st.position - sp[i]; limit = max(MIN_LAG, (self.cap[i] - abs(tff[i])) / self.kp[i])
                row[:, i] = (st.position, st.torque, lag, tff[i]); self.last["limit"][i] = limit
                self.max_abs_lag[i] = max(self.max_abs_lag[i], abs(lag))
                if abs(lag) > limit and trip is None:
                    trip = (f"motor 0x{m.motor_id:02X} is {abs(lag):.3f} rad behind its setpoint (limit {limit:.3f} rad = the torque budget "
                            f"{BUDGET:.0%} of {m.spec.t_max} N.m minus the {abs(tff[i]):.1f} N.m of feed-forward). Stopping.")
        for j, k in enumerate(("pos", "torque", "lag", "tff")): self.last[k] = row[j]
        self.series.append((time.time(), phase, row.copy()))
        if trip: raise MotorError(trip)


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
    S(sp, np.array([0.2, 0, 0, 0, 0, -0.2]), 0.0); check("friction term follows the commanded direction", abs(motors[0].sent[-1][4] - 0.3) < 1e-9 and abs(motors[5].sent[-1][4] + 0.06) < 1e-9)
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
    print("\nSELFTEST:", "ALL PASSED" if not fails else f"FAILED: {fails}"); return 1 if fails else 0


# --------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label"); ap.add_argument("--poses", default=str(POSES))
    ap.add_argument("--fraction", type=float, default=1.0); ap.add_argument("--max-travel", type=float, default=2.5)
    ap.add_argument("--speed", type=float, default=0.25); ap.add_argument("--lower-speed", type=float, default=0.20)
    ap.add_argument("--rate", type=float, default=200.0); ap.add_argument("--settle", type=float, default=1.0)
    ap.add_argument("--hold", type=float, default=5.0)
    ap.add_argument("--factors", default=FACTORS_DEFAULT, help="per-joint gravity factors")
    ap.add_argument("--gravity-scale", type=float, default=1.0, help="multiplies all factors (0 = no feed-forward)")
    ap.add_argument("--no-ff", action="store_true", help="disable all feed-forward (old behaviour, but with the new exit)")
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--yes", action="store_true"); ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest: return selftest()
    if not args.label: ap.error("--label is required")
    if not (0.0 < args.fraction <= 1.0): log("ABORT: --fraction must be in (0, 1]"); return 1
    if not args.dry_run and not args.yes: log("REFUSING to energize the arm without --yes (use --dry-run to just check)"); return 1
    factors = np.array([float(x) for x in args.factors.split(",")]) * args.gravity_scale
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
    here = np.array(here); target = here + args.fraction * (np.array(recorded) - here); travel = target - here
    if any(not (lo - 0.05 <= t <= hi + 0.05) for t, (lo, hi) in zip(target, window)): log("ABORT: target outside the real range"); bus.close(); return 1
    if np.abs(travel).max() > args.max_travel: log(f"ABORT: a joint would move {np.abs(travel).max():.2f} rad (> --max-travel {args.max_travel})."); bus.close(); return 1
    delta = float(np.abs(travel).max())
    if delta < 1e-4: log("already there; nothing to do"); bus.close(); return 0
    ramp_T = (math.pi / 2) * delta / args.speed
    sender = Sender(arm.motors, arm.gains_kp, arm.gains_kd, gravity, factors, use_ff=not args.no_ff, lock=arm._lock)
    log(f"target       : {[round(float(q), 4) for q in target]}   (fraction {args.fraction})")
    log("travel       : " + "  ".join(f"j{i+1}:{x:+.3f}" for i, x in enumerate(travel)))
    log(f"furthest     : {delta:.3f} rad -> {ramp_T:.1f}s at peak {args.speed} rad/s, then hold {args.hold}s and ramp back")
    path = np.array([sender.feedforward(here + s * (target - here), None if False else np.zeros(6), 1.0) for s in np.linspace(0, 1, 41)])
    log("feed-forward torque N.m  at start: " + " ".join(f"{x:+5.1f}" for x in path[0]) + "\n" + " " * 11 + "at target: " + " ".join(f"{x:+5.1f}" for x in path[-1]) + "  |  peak along the path: " + " ".join(f"{np.abs(path[:, i]).max():4.1f}" for i in range(6)))
    log("torque budget (0.4 x TMAX) N.m: " + " ".join(f"{c:4.1f}" for c in sender.cap) + f"   factors {np.round(factors, 2)}" + ("   [feed-forward OFF]" if args.no_ff else ""))
    log("session check OK, travel guard OK")
    if args.dry_run: log("dry run: nothing enabled, nothing moved"); bus.close(); return 0

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *a: stop.__setitem__("flag", True)); signal.signal(signal.SIGTERM, lambda *a: stop.__setitem__("flag", True))
    enabled, seed = [], []; last_sp = None; hold_rows = []
    try:
        for m, kp, kd in zip(arm.motors, arm.gains_kp, arm.gains_kd):
            q = m.read_state().position; st = m.enable(); enabled.append(m)
            m.command(pos=q, vel=0.0, kp=kp, kd=kd, torque=0.0); seed.append(q)
            log(f"enabled+holding 0x{m.motor_id:02X} at {q:+.4f} [{st.error}]")
        seed = np.array(seed); arm._active = True
        t0 = time.time(); last_log = t0; phase = ""; reached_at = None
        while not stop["flag"]:
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
        if enabled and last_sp is not None:
            try:
                log("ramping back to the start pose from the last commanded setpoint (no reads, no re-seed)")
                ramp_to(sender, last_sp, seed, args.lower_speed, args.rate, hold_s=0.5)
                log("returned to the start pose")
            except Exception as exc:  # noqa: BLE001
                log(f"controlled return failed: {exc}")
        for m in reversed(enabled): m.disable()
        log("all motors disabled"); bus.close()
    if hold_rows:
        h = np.array(hold_rows); log("HOLD SUMMARY  mean lag: " + " ".join(f"{x:+.3f}" for x in h.mean(axis=0)) + "  | worst |lag| over the run: " + " ".join(f"{x:.3f}" for x in sender.max_abs_lag))
    LOGDIR.mkdir(exist_ok=True); path_out = LOGDIR / f"run2_{time.strftime('%Y%m%d_%H%M%S')}.json"
    json.dump({"args": vars(args), "factors": factors.tolist(), "series": [[t, ph, r.tolist()] for t, ph, r in sender.series[::5]]}, open(path_out, "w")); log(f"telemetry saved: {path_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
