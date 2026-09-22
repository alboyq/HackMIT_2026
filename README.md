# HackMIT 2026 — an assistive feeding arm

A robot arm that feeds you the food you choose with your eyes. Built at HackMIT 2026 on an open-source
six-axis arm (OpenYAM).

* [`arm/rl/`](arm/rl/) — **the learned grasp.** Reach and pinch learned by reinforcement learning from a reward
  alone, from one wrist camera and motor feedback: 76% pick-up-and-feed in simulation with zero unsafe
  arrivals at the head. Videos inside. Start here.
* [`arm/real/`](arm/real/) — the real-world layer: food and face perception on the real wrist camera, the
  arm control backend (gravity feed-forward, hold-never-drop, torque stop, bounded squeeze), calibration tools.
* [`arm_replay/`](arm_replay/) — moving the real arm: measured joint map, gravity + friction fit, pose replay.
* [`arm/ik/`](arm/ik/) — the scripted inverse-kinematics pick/lift/deliver path and the shared MuJoCo scene.
* Eye-tracking selection (`gaze3d/`) and the SSVEP flicker interface live on `main`.

Engineering notes: [`arm/RL_HANDOFF.md`](arm/RL_HANDOFF.md), [`ARM_NOTES.md`](ARM_NOTES.md), [`DECISIONS.md`](DECISIONS.md).
