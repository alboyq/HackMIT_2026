# arm_replay — moving the real OpenYAM (2026-09-20)

> **Still early, but no longer untried.** The gravity feed-forward is fitted to this arm and has been flown at 60% and 80% of one
> pose with a tip error of ~0.5 cm. It is **not** safe near people or unattended, stages above 80% and any table contact have
> never been run, and the new retry-on-failed-return logic has only been tested with fake motors. A hand on the power switch is
> still the only guaranteed stop.

## What it is
Replays a pose recorded by hand with `../fk_check/fk_capture.py`, **in the motors' own numbers** (no IK, no lap arithmetic): from the
start pose part of the way (`--fraction`) toward the recorded pose at <=0.25 rad/s, holds, ramps back. The real backend (sim joint
numbers in, motor numbers out) is `arm/real/real_arm.py` on `arm-ik-rl`, which reuses `Gravity` and `Sender` from here.

| file | what |
|---|---|
| `replay_pose.py` | v1 (position control only; adapted from openyam's `move_and_hold.py`). Kept for the record; it sags. |
| `replay_pose2.py` | v2: gravity + friction feed-forward in every MIT command, state from the replies, **hold-never-disable return**, `--selftest` (30 offline checks). |
| `gravity_fit.json` | **The fitted feed-forward numbers** `replay_pose2.py` loads by default (`--no-fit` = old vendor-style values). |
| `gravity_data.py`, `gravity_fit.py`, `gravity_eval_lib.py`, `gravity_export.py` | How the fit was made from the arm's own torque logs. `python gravity_export.py` regenerates `gravity_fit.json`. |
| `canlog.py`, `canview.py` | Passive CAN listener (sends nothing) and decoder (reads `.txt` or `.txt.gz`). |
| `logs/` | stdout + telemetry of runs 4-7; the CAN captures of runs 4-7 (gzip) that the fit is made from. |
| `TODO_ideas.md` | What is left. |

## Gravity + friction fit
Feed-forward per joint: `factor_j * qfrc_bias_j(q) + coulomb_j * tanh(v_cmd_j / 0.02)`, where `qfrc_bias` is MuJoCo's gravity torque
on the sim's own `yam.xml` with its free gravity compensation switched **off**. Fitted values (`gravity_fit.json`):

| | J1 | J2 shoulder | J3 elbow | J4 wrist | J5 | J6 |
|---|---|---|---|---|---|---|
| gravity factor | 1 | **1.375** | **1.375** | **1.111** | 1 | 1 |
| friction (N.m) | 0.76 | 0.69 | **1.89** | 0.14 | 0.06 (i2rt) | 0.06 (i2rt) |

* **The real arm needs ~38% more gravity torque than the model** for J2/J3 (11% for the J4 wrist). Whether the arm is heavier or the
  motors deliver less than they are told makes no difference to the feed-forward (both are a pure scale). Adding mass to the model
  does *not* explain it: the best mass-only fit needs unphysical values, and a plain scale fits as well with 3 numbers.
* **The elbow has ~1.9 N.m of friction with stick-slip**: in run 4 it stood still while the torque swung 11.2 -> 7.0 N.m.
* Method: fit to the torque the motors delivered (the reply's torque field equals the commanded torque), ignoring samples where a
  joint rests on its hard stop (the stop carries the load there). Ridge regression on physical parameters; alternatives (one scale for
  all joints; extra mass beyond the elbow; per-joint scale) agree to <=0.6 N.m at the taught poses.
* **Tested on data it never saw**: fit without run X, score on run X, elbow RMS 0.5-0.9 N.m vs 1.8-2.6 N.m for the unfitted model.
  Then run 7 (80%) was flown with a fit that did not contain it: predicted elbow hold 7.5 N.m, measured 8.4; shoulder -8.1 vs -7.3
  (both inside the friction band); tip error 0.5 cm.
* **Coverage: elbow 0.03-0.94 rad, shoulder 0-1.21 rad** (runs 4-6). Elbow bent past ~1 rad is extrapolation; the sim's HOME pose has 1.49.

## What has been run (pose A = arm hovering ~2.5 cm above dot 3, taught after the 2nd power cycle; runs 4-5 used old pose 5)
| run | code | result |
|---|---|---|
| 1-3 | `replay_pose.py` 60% | 11 cm low (no feed-forward), and a drop at the end from a read-then-reseed of live motors. |
| 4 | v2 first fit, 60% of pose 5 | elbow lag -0.024 rad, tip 1.4 cm low, no drop. |
| 5 | v2, 80% of pose 5 | hold fine; the **return aborted** at the 0.030 lag floor and the OLD code disabled all motors: arm fell from ~26 deg above rest. |
| 6 | v2 + fit, **60% of pose A** | worst lag 0.023 rad, hold tip error **0.6 cm**, returned first attempt, "resting on its stops". Elbow 0.94 rad (data had 0.18). |
| 7 | v2 + fit (incl. run 6), **80% of pose A** | worst lag 0.025 rad, hold tip error **0.5 cm**, returned first attempt. Not in the fit. |

## What changed since the first version
* **Return never disables an arm that is off its stops.** A tripped return holds where the arm is (setpoint = measured position, full
  gain + feed-forward, motors on), retries up to 3x (each slower, lag floor 0.10/0.15/0.20 rad), then stays powered and holding until
  a **second** stop signal (`Ctrl-C` twice / `pkill -TERM` twice). It disables only when shoulder and elbow are within 0.12 rad of their
  lower stops, or on that second signal. The driver's `atexit` disable is unregistered.
* Fitted gravity/friction (above); friction is smooth (`tanh`) instead of a threshold; guard floor 0.03 -> 0.05 rad (`--min-lag`).
* Travel guard applies to J1-J3 only (2.5 rad); the wrist may use its full mechanical range (start and target are already required
  to be inside the measured stops).
* Every log line is saved in the telemetry json (`events`).

## How to run (needs a person at the arm)
Prerequisites: `can0` up at 1 Mbit/s; arm powered and **folded at rest** (wrist roughly straight is fine); a hand on the power switch;
`~/openyam` (motor driver, **not in git**); mujoco_menagerie (`YAM_MENAGERIE`); a recording from the **same power session**.
```
python replay_pose2.py --selftest
python replay_pose2.py --label A --poses ../fk_check/poses.json --fraction 0.6 --dry-run      # reads only
python replay_pose2.py --label A --poses ../fk_check/poses.json --fraction 0.6 --hold 5 --yes
python canlog.py logs/capture.txt &                                                            # passive; run it to log everything
```
`../fk_check/poses.json` pose `A` is valid for the power session of 2026-09-20 02:48; after any power cycle re-teach
(`../fk_check/fk_capture.py <label>`, motors off) - the script refuses a stale recording.

## Known flaws
1. **Poses are raw motor numbers, valid until the next power cycle** (laps). The endpoint also depends on the start pose (wrist free at rest).
2. **Pose A is a hover 2.5 cm above dot 3 with the elbow bent 1.57 rad.** 100% would put the tip 2.9 cm above the table and needs the shoulder
   at -11.2 N.m = the entire 0.4 x TMAX budget. Do not run 100% without raising the budget and thinking about contact.
3. **Elbow stiction leaves up to ~1 cm of error** after a move (see fit). Untested above 80%, untested with any contact.
4. **The retry-and-hold return has never been exercised on the arm** (runs 6-7 returned on the first attempt).
5. **No watchdog** (TIMEOUT=0): if the script or the USB CAN adapter dies the motors keep their last command. The adapter's GND terminal
   is unlanded and it re-enumerated ~6 times over two sessions. Only a hand on the power switch stops it.
6. **`~/openyam` is not in git**; the scripts import it from a hard-coded `/home/asus/openyam`.
7. The gripper is not used by these scripts.

## For whoever owns `arm/real/real_arm.py`
It reuses `Gravity` and `Sender` from here. Its default `factors=(1.0, 1.1, 1.4, 1.0, 1.0, 1.0)` are the OLD guesses: load
`gravity_fit.json` instead (`replay_pose2.load_fit`), and pass its `coulomb_Nm` to `Sender(..., coulomb=...)`. Its contact-stop limits
(`CONTACT_NM` 7/8/7/3/2/2 N.m) were sized for the old elbow error; with the fit, `|torque - feed-forward|` in runs 6-7 never exceeded
2.2 / 1.9 / 0.3 N.m at J2 / J3 / J4 (0.7 at J1), so they can be tightened once the fit has been through a few more runs.
