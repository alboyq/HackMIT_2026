# arm_replay — the BEGINNING of commanding the real OpenYAM (2026-09-20)

> **This is a first version, pushed so the team has something to build on. It is not polished, and it is
> not safe near people or unattended.** Read "Known flaws" before you run anything. One run already
> released the arm mid-return (below).

## What it is
Replays a pose recorded by hand with `../fk_check/fk_capture.py` on the real arm, **in the motors' own
numbers** — no IK, no model-to-motor conversion, no lap arithmetic. The arm goes from its start pose part
of the way (`--fraction`) toward the recorded pose at ≤0.25 rad/s, holds, and ramps back.
It exists to get the arm moving safely-ish *now*; the real backend (sim joint numbers in, motor numbers out)
is still `RealArm` in `arm/deploy.py`, which stays disabled.

| file | what |
|---|---|
| `replay_pose.py` | v1: adapted from openyam's `move_and_hold.py` (the only script proven on hardware) + guards against a stale recording and large travel. Position control only. |
| `replay_pose2.py` | v2: adds **gravity feed-forward** (the vendor i2rt recipe: `torque = factor·G(q) + coulomb·sign(v)` in every MIT command, state taken from the replies), a total-torque guard, telemetry, and an exit that continues from the last commanded setpoint. `--selftest` runs 13 offline checks. |
| `canlog.py`, `canview.py` | **Passive** CAN listener (receives, sends nothing) and decoder: lets you see every command (kp, kd, setpoint, torque) and every reply while a script runs. |
| `TODO_ideas.md` | Ideas not done yet (pose-dependent gravity model, stages 0.9–1.0, gripper, startup lap check). |
| `logs/` | Telemetry and output of the 60% run with feed-forward (run 4) and the 80% run whose return aborted (run 5). |

## How to run (needs a person at the arm)
Prerequisites: `can0` up at 1 Mbit/s; arm powered and **folded at rest**; a hand on the arm's power switch;
`~/openyam` (the motor driver — **not in any git repo**, see flaws); mujoco_menagerie (`YAM_MENAGERIE`);
a recording from the **same power session** (`../fk_check/fk_capture.py <label>`).
```
python replay_pose2.py --selftest                                   # offline, no hardware
python replay_pose2.py --label 5 --poses ../fk_check/poses.json --fraction 0.6 --dry-run   # reads only
python replay_pose2.py --label 5 --poses ../fk_check/poses.json --fraction 0.6 --hold 5 --yes
pkill -TERM -f replay_pose2.py            # ramps back and disables (see hazard 1 below)
```
`../fk_check/poses.json` is from the 2026-09-19 power session: after any power cycle it is **stale** (the
script refuses it) and the poses must be re-taught.

## What happened (all at 60% of recorded pose 5, except run 5)
| run | code | result |
|---|---|---|
| 1–3 | `replay_pose.py` | Gripper ended **11 cm below** the commanded 22.9 cm above the table (elbow lag −0.131 rad, wrist −0.190). At the end of the hold the script read live motors (a zero-stiffness command, 8–11 ms) and re-seeded the setpoint at the sagged position: elbow twitched 1° in 60 ms, wrist slumped 10° over ~0.6 s. No guard trips; motors 27–31 °C. |
| 4 | `replay_pose2.py` | Elbow lag −0.024, wrist −0.018; gripper **1.4 cm below** commanded. No zero-stiffness commands; setpoint jump at the end ≤0.002 rad; no drop at the hold→return moment. Back within 0.011 rad of start. Elbow lag peaked at 0.029 against a 0.030 limit (97%). |
| 5 (80%) | `replay_pose2.py` | Hold fine (elbow −0.018, wrist −0.003). **The return aborted 2.5 s into ~6 s** when the elbow's lag reached the 0.030 guard limit, and the script then **disabled all motors**: the arm was released with the shoulder ~26° above its rest stop. |

## What we learned
- Reading a motor that is switched on sends it a zero-gain command, i.e. it goes limp for that cycle. Never read a live motor; take state from the reply to each command (this is what the vendor's own loop does).
- Restarting the setpoint at the *measured* position removes the torque that was holding the arm up, so it falls until the error rebuilds. Continue from the last commanded setpoint instead.
- Position control alone cannot hold this arm: the elbow needs ~11 N·m at the 60% pose and the position loop tops out near 11. Gravity feed-forward fixes it (i2rt's `yam_v1.yml`: factors 1.0/1.1/1.1/1.2/1.0/1.0, Coulomb 0.3/0.3/0.3/0.06/0.06/0.06 N·m; we use 1.0/1.1/1.4/1.0/1.0/1.0).
- The menagerie MJCF has gravity compensation switched **on** for every link. The sim arm therefore never sags; the real one does. Anything trained in sim has never seen this.
- Real elbow gravity torque vs the model differs by pose: ~4.1 N·m folded (model 4.3), ~11 N·m at the 60% hold pose (model 6.6), ~10.1 at the 80% hold, only ~6.6 in the 80% return pose. One scalar per joint cannot fit all of them.

## Known flaws — positioning first
1. **Taught poses are raw motor numbers, valid only until the next power cycle** (the motors report position modulo one turn; J2/J3 sit on lap +1 in this session). The script refuses a stale recording; re-teach after a power cycle.
2. **The taught poses are hand-placed.** Dot-to-dot agreement was up to 7.9 mm (dot 3 six to eight mm off), the tool tip is a fit (±9 mm per axis), and pose 5 was recorded with the pads *on the paper*: 100% would press the pads into the table.
3. **The gripper does not reach where it is told.** At 60% it ended 1.4 cm below commanded (under-compensated); at the folded rest pose it ends ~5.5 mm *above* its setpoint (over-compensated). Expect ±1–2 cm until the gravity model is pose-dependent.
4. **The endpoint depends on the start pose.** The target is a fraction of the way from the *measured* start, and the wrist hangs free at rest. Before run 5 someone touched the wrist (J4 1.659 → 0.310 between the dry run and the run), which changed the start, the feed-forward and the endpoint. With the wrist bent up, 60% ends 23 cm above the table; with the wrist pre-set to the recorded angle it would end 4 cm above.
5. **Feed-forward ramps in over 1 s, so a wrist that starts off its stop droops** (run 5: wrist lag 0.130 rad in the first second).
6. **Stages of 90% and up, and any contact with the table, were never run.**
7. The gripper was never used; the 0–1 gripper mapping needs its open/closed motor readings.

## Known flaws — safety and infrastructure
1. **If the return ramp aborts, the script disables all motors and the arm falls.** Fix (not done): on a failed return keep holding the last setpoint (do not disable; remember `atexit` in the driver also disables) and loosen the return guard.
2. **The elbow lag guard floor (0.030 rad) is too tight** for the gravity-model error; it tripped at exactly 0.030 in run 5. The total-torque budget (0.4×TMAX = 11.2 N·m) is about the elbow's own gravity load.
3. **No watchdog** (TIMEOUT=0, per openyam/CLAUDE.md): if the script or the USB CAN adapter dies mid-move the motors keep their last command. The adapter's GND terminal is unlanded and it re-enumerated ~6 times over two sessions.
4. **`~/openyam` is not in git**; both scripts import it from a hard-coded `/home/asus/openyam`, so nobody else can run them yet.
5. Only a hand on the power switch is a guaranteed stop.

## Next
Hold-instead-of-disable on a failed return; a pose-dependent gravity model (see `TODO_ideas.md`); control of the start pose; stages 0.8 → 1.0 with a tighter contact guard; then the real `RealArm` with the lap rule so sim/ML joint targets can be sent.
