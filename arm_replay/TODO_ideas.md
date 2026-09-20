# Ideas / still to do (arm bring-up, updated 2026-09-20 03:15)

## Done since the first version
- Pose-dependent gravity: **solved by fitting, not by a table** - see README "Gravity + friction fit". The old note here said the
  folded rest pose needs ~4.1 N.m (real/model 0.95). That was WRONG: at rest the arm sits on its hard stops and the stops carry the
  load, so the torque there says nothing about gravity. Those samples are now excluded from the fit.
- Hold-never-disable on a failed return (with retries) - implemented, tested with fake motors only.

## Still to do
- Stages above 80% (pose A at 100% is 2.9 cm above the table; the shoulder needs 11.2 N.m there = the whole 0.4 x TMAX budget, so a
  `--budget 0.5` option is needed first) and ANY table contact. Contact stop numbers: in runs 6-7 |torque - feed-forward| never
  exceeded 2.2 N.m (J2), 1.9 (J3), 0.3 (J4), 0.7 (J1) - so arm/real/real_arm.py's CONTACT_NM (7/8/7/3/2/2) has ~3x headroom to spare.
- Elbow bent beyond ~1 rad is still extrapolation (data reach 0.94 rad). Add the next run's log to the fit (`python gravity_export.py`
  after adding it to RUNS in gravity_data.py).
- Elbow stiction: the elbow sits anywhere inside a +-1.9 N.m band, i.e. up to ~0.025 rad (~1 cm) off after a move. A slow integral
  trim clamped to +-2..3 N.m would remove it (30-40 min; windup risk).
- Gripper: open/closed endpoints of motor 0x08 (measured by a teammate, see arm/real/calibration), then a torque-limited
  close-until-stall with a hard cap, and the pad force in N.
- CAN adapter GND terminal; secure the USB cable; test the E-stop; `~/openyam` into git or vendored.
- Startup lap check inside RealArm (laps this session: all 0; the first session had J2/J3 = +1).
