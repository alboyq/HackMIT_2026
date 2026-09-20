#!/usr/bin/env python3
"""WORK IN PROGRESS - FIRST VERSION, JUST THE BEGINNING. NOT POLISHED, NOT SAFE NEAR PEOPLE OR UNATTENDED.
Known hazards (details and evidence in README.md):
  * if the return ramp aborts on the lag guard this script DISABLES ALL MOTORS and the arm falls from wherever it is
    (happened once, 2026-09-20: shoulder 26 deg above its rest stop). Fix: hold the last pose instead, loosen the guard on the return.
  * poses are raw motor numbers, valid only until the arm is next powered off (laps); the endpoint depends on the START pose.
  * imports the motor driver from /home/asus/openyam, which is NOT in any git repo; paths are hard-coded.
  (this v1 also has the end-of-hold drop: it reads and re-seeds a live motor, so the wrist slumps ~10 deg and the elbow twitches - see README)
Have a hand on the arm's power switch and the arm folded at rest before you energize it.

Replay a pose recorded by fk_capture.py on the real arm, in the motors' OWN numbers.

Adapted from openyam/scripts/move_and_hold.py (the only motion script proven on hardware); every
safety behaviour of that script is kept: TIMEOUT=0 verified before enabling, each motor's holding
position read immediately before its own enable, trajectory velocity fed into every MIT frame,
faults abort (never cleared in place), exit lowers back to the START pose under control and then
disables. The gripper motor is never touched.

What changed, and why
  * The target is a recorded pose (poses.json -> encoder_raw), so no model<->motor conversion and
    no lap arithmetic is needed. Those raw numbers are only valid until the arm is next powered off:
    the motors report position modulo one turn and J2/J3 changed lap after the last power cycle.
  * So instead of the driver's limit check (MJCF ranges applied to raw numbers - wrong for a
    wrapped J2/J3, and wrong for J1/J6), two guards protect against a STALE recording:
      1. session check: every joint's CURRENT reading must lie inside that joint's real range
         (hand-measured hard stops), expressed in the recording's own lap; otherwise abort.
      2. travel guard: no joint may move more than --max-travel rad (a wrong lap is ~6.3 rad).
  * --fraction moves only part of the way from the start pose to the recorded pose.
  * an energized run needs --yes; --dry-run enables nothing.

    python replay_pose.py --label 5 --dry-run
    python replay_pose.py --label 5 --fraction 0.6 --hold 5 --yes
    pkill -TERM -f replay_pose.py          # lowers to the start pose, then disables
"""
from __future__ import annotations

import argparse, json, math, signal, sys, time
from pathlib import Path

sys.path.insert(0, "/home/asus/openyam")
from openyam.arm import ArmConfig, MotorError, OpenYAMArm, tracking_error_limit  # noqa: E402
from openyam.gsusb import CanBus  # noqa: E402

TAU = 2 * math.pi
# Hard stops measured by hand on 2026-09-19 (encoder numbers at lap 0), see joint_map_measured.json.
STOPS = [(-4.0309, 1.6047), (0.0044, 3.6952), (0.0101, 3.9569),
         (-1.6726, 1.6577), (-1.5761, 1.5791), (-0.7822, 3.3892)]
POSES = Path(__file__).resolve().parent / "poses.json"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="pose label in poses.json, e.g. 5")
    ap.add_argument("--poses", default=str(POSES))
    ap.add_argument("--fraction", type=float, default=1.0, help="0<f<=1: how far from the start pose toward the recorded pose")
    ap.add_argument("--max-travel", type=float, default=2.5, help="abort if any joint would move more than this many rad")
    ap.add_argument("--speed", type=float, default=0.25, help="peak joint speed, rad/s")
    ap.add_argument("--lower-speed", type=float, default=0.20)
    ap.add_argument("--rate", type=float, default=200.0, help="command rate, Hz")
    ap.add_argument("--settle", type=float, default=1.0, help="hold in place before moving, s")
    ap.add_argument("--hold", type=float, default=5.0, help="seconds to hold at the target, then return")
    ap.add_argument("--dry-run", action="store_true", help="read and check only; enables nothing")
    ap.add_argument("--yes", action="store_true", help="required to energize the arm")
    args = ap.parse_args()
    if not (0.0 < args.fraction <= 1.0):
        log("ABORT: --fraction must be in (0, 1]"); return 1
    if not args.dry_run and not args.yes:
        log("REFUSING to energize the arm without --yes (use --dry-run to just check)"); return 1

    rec = {p["label"]: p for p in json.load(open(args.poses))["poses"]}.get(args.label)
    if rec is None:
        log(f"ABORT: no pose '{args.label}' in {args.poses}"); return 1
    recorded, laps = [float(x) for x in rec["encoder_raw"][:6]], [int(k) for k in rec["lap"][:6]]
    window = [(lo - TAU * k, hi - TAU * k) for (lo, hi), k in zip(STOPS, laps)]     # real range in the recording's numbers
    log(f"recorded pose '{args.label}' ({rec['time']}): {[round(x, 4) for x in recorded]}  laps {laps}")

    bus = CanBus()
    arm = OpenYAMArm(ArmConfig(joint_limits=[(lo - 0.15, hi + 0.15) for lo, hi in window]), bus=bus, include_gripper=False)
    log(f"bus {bus.channel} via {bus.interface}, state={bus.state}")

    for m in arm.motors:                                    # preconditions, all read-only
        tmo = m.read_register("TIMEOUT")
        if tmo != 0:
            log(f"ABORT: 0x{m.motor_id:02X} has TIMEOUT={tmo}. The watchdog must be OFF while driving."); bus.close(); return 1
    log("watchdog OFF on all six")
    here = []
    for m in arm.motors:
        st = m.read_state()
        if st is None:
            log(f"ABORT: 0x{m.motor_id:02X} gave no state"); bus.close(); return 1
        if st.error not in ("disabled", "normal"):
            log(f"ABORT: 0x{m.motor_id:02X} latched '{st.error}'. Power-cycle the arm to clear."); bus.close(); return 1
        here.append(st.position)
    log(f"measured pose: {[round(q, 4) for q in here]}")

    # guard 1: is the recording still in the same numbers as the arm? (a power cycle can shift J2/J3 by one turn)
    bad = [i for i, (q, (lo, hi)) in enumerate(zip(here, window)) if not (lo - 0.15 <= q <= hi + 0.15)]
    if bad:
        for i in bad:
            log(f"  J{i+1} reads {here[i]:+.4f}, but in the recording's numbers its real range is [{window[i][0]:+.3f}, {window[i][1]:+.3f}]")
        log("ABORT: the arm's numbers no longer match the recording (power cycle? lap changed). Re-teach the pose."); bus.close(); return 1
    target = [h + args.fraction * (r - h) for h, r in zip(here, recorded)]
    off = [i for i, (t, (lo, hi)) in enumerate(zip(target, window)) if not (lo - 0.05 <= t <= hi + 0.05)]
    if off:
        log(f"ABORT: target joint(s) {[i+1 for i in off]} outside the real range"); bus.close(); return 1
    # guard 2: nothing is allowed to travel further than a real joint ever needs to
    travel = [t - h for t, h in zip(target, here)]
    if max(abs(x) for x in travel) > args.max_travel:
        log(f"ABORT: a joint would move {max(abs(x) for x in travel):.2f} rad (> --max-travel {args.max_travel}). Rotate the base/wrist by hand closer first."); bus.close(); return 1
    delta = max(abs(x) for x in travel)
    if delta < 1e-4:
        log("already there; nothing to do"); bus.close(); return 0
    ramp_T = (math.pi / 2) * delta / args.speed
    log(f"target       : {[round(q, 4) for q in target]}   (fraction {args.fraction})")
    log("travel       : " + "  ".join(f"j{i+1}:{x:+.3f}" for i, x in enumerate(travel)))
    log(f"furthest     : {delta:.3f} rad -> {ramp_T:.1f}s at peak {args.speed} rad/s, then hold {args.hold}s and return")
    for m, kp in zip(arm.motors, arm.gains_kp):
        log(f"  0x{m.motor_id:02X} lag limit {tracking_error_limit(m, kp):.3f} rad")
    log("session check OK, travel guard OK")
    if args.dry_run:
        log("dry run: nothing enabled, nothing moved"); bus.close(); return 0

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *a: stop.__setitem__("flag", True))
    signal.signal(signal.SIGTERM, lambda *a: stop.__setitem__("flag", True))
    enabled, seed = [], []
    try:
        for m, kp, kd in zip(arm.motors, arm.gains_kp, arm.gains_kd):     # just-in-time: read, enable, hold - per motor
            q = m.read_state().position
            st = m.enable(); enabled.append(m)
            m.command(pos=q, vel=0.0, kp=kp, kd=kd, torque=0.0)
            seed.append(q)
            log(f"enabled+holding 0x{m.motor_id:02X} at {q:+.4f} [{st.error}]")
        arm._active = True
        t0 = time.time(); n = 0; last = t0; phase = ""; reached_at = None
        while not stop["flag"]:
            tick = time.time(); el = tick - t0; vel = None
            if el < args.settle:
                sp, ph = seed, "settle"
            elif el < args.settle + ramp_T:
                u = math.pi * (el - args.settle) / ramp_T
                a = 0.5 * (1 - math.cos(u)); adot = (math.pi / (2 * ramp_T)) * math.sin(u)
                sp = [s + (g - s) * a for s, g in zip(seed, target)]
                vel = [(g - s) * adot for s, g in zip(seed, target)]
                ph = "move"
            else:
                sp, ph = target, "hold"
                if reached_at is None: reached_at = el
            if ph != phase:
                log({"settle": f"settling {args.settle}s", "move": "MOVING", "hold": f"AT TARGET at t={el:.1f}s -- holding"}[ph]); phase = ph
            arm._send_pose(sp, 1.0, vel)                     # aborts on fault or excessive lag
            n += 1
            if tick - last >= 15.0:
                sts = arm.read_states()
                log(f"{phase} {n/(time.time()-t0):.0f}Hz | " + " ".join(f"0x{m.motor_id:02X}:{s.position:+.3f}/{s.torque:+.2f}Nm/{s.temp_mos:.0f}C" for m, s in zip(arm.motors, sts) if s)); last = tick
            if reached_at is not None and el - reached_at >= args.hold:
                log(f"held {args.hold}s"); break
            time.sleep(max(0.0, 1.0 / args.rate - (time.time() - tick)))
        else:
            log("stop requested")
    except MotorError as exc:
        log(f"GUARD TRIPPED: {exc}")
    except Exception as exc:  # noqa: BLE001
        log(f"ERROR: {exc!r}")
    finally:
        if enabled:
            try:
                now = [s.position for s in arm.read_states()]
                log(f"lowering from measured {[round(q, 3) for q in now]}")
                arm.move_to(seed, speed=args.lower_speed, allow_outside_limits=True)
                log("returned to the start pose")
            except Exception as exc:  # noqa: BLE001
                log(f"controlled return failed: {exc}")
        for m in reversed(enabled):
            m.disable()
        log("all motors disabled")
        bus.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
