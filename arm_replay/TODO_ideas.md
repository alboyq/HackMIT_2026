# Ideas for later (arm bring-up, 2026-09-20)

## Pose-dependent gravity feed-forward  (saved 01:4x, from the 60% run)
Observed with replay_pose2.py (gravity feed-forward, per-joint factors 1.0/1.1/1.4/1.0/1.0/1.0):
- Extended pose (60% of pose 5): elbow needs ~11 N.m; model says 6.6 -> real/model ~ 1.7. Elbow lag -0.024 rad,
  worst 0.029 during the move (guard limit 0.030 = 97%). Gripper ended 1.3 cm BELOW the commanded 22.9 cm.
- Rest pose (folded, wrist bent up): ff = 6.0 N.m but the elbow only needs ~4.1 -> real/model ~ 0.95. Elbow ends
  0.024 rad ABOVE its setpoint (+5.5 mm at the tip); at the start of the run it lifts ~1 mm off its stops.
- So one scalar per joint is a compromise. Raising the elbow factor to 1.6 would over-lift at rest
  (ff 6.9 vs need 4.1 -> lag ~ +0.035 > the 0.03 guard) and could abort at the start.
Candidate fixes (do NOT need doing unless the margin/tracking becomes a problem):
  a) add mass to the MJCF as the vendor does (ee_mass / ee_inertia) or a point mass on the forearm; fit it to the two
     data points above (J3: rest ~4.1, extended ~11 N.m; J4 already matches: model 1.77 vs real 1.9)
  b) make the elbow factor a function of extension (e.g. of J3 or reach), 0.95 folded -> ~1.7 extended
  c) a slow integrator on the position error (trims the residual, but adds windup risk)
Data: logs/run2_20260920_013322.json (telemetry), capture_run4.txt (CAN capture), logs/run4_stdout_013257.txt.

## Also pending
- Stages after 60%: 0.8 (~10 cm above table), 0.9 (~4.6 cm), 0.95 (~2 cm), 1.0 (pads on dot 5, --hold 2).
- Gripper open/closed endpoints (motor 0x08) for the 0..1 mapping; CAN adapter GND terminal; pad force measurement.
- Startup lap check inside a RealArm (laps: J2/J3 = +1 this power session).
