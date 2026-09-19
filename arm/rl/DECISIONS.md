# Decisions

- 2026-09-19: Use CPU PPO. The policy is a small MLP and MuJoCo environment stepping is the bottleneck; this avoids competing with teammates for unified GPU memory.
- 2026-09-19: Reuse I2RT's `yam_station_linear_4310_d405.xml`, already cloned and smoke-tested at `~/HackMIT_2026/third_party/i2rt`, instead of spending the Phase A timebox converting Anvil's URDF.
- 2026-09-19: Use six arm joint delta actions plus one gripper action from the first stage so observation/action shapes remain stable across reach, grasp, and lift.
- 2026-09-19: Use 50 Hz control with five 4 ms MuJoCo substeps. This is conservative and suitable for later real-arm rate limiting.
- 2026-09-19: Apply PD torque in the environment because the vendor station MJCF is kinematic and has no control actuators.
- 2026-09-19: Run 12 workers, one BLAS thread each, under `nice -n 10`; this leaves eight CPU cores and the GPU available to teammates.
- 2026-09-19: Delivery to the mouth remains a scripted waypoint and is outside the learned policy.
- 2026-09-19: No compatible pretrained single-arm low-dimensional YAM policy was found. The closest public checkpoint, `angkul07/pi0-fast-yam`, expects three RGB cameras and a 14-dimensional bimanual action, and its published final-checkpoint evaluation reports a 34.4% action decode-collapse rate. It will not be used for control or PPO initialization.
- 2026-09-19: Train the Phase A 23-observation/7-action PPO from scratch. Reuse I2RT's exact YAM model now; retain ABC-130k and its LeRobot v3 conversion as optional later imitation-learning data, after an explicit single-arm/action-schema conversion and closed-loop safety evaluation.
- 2026-09-19: Rejected the first 500k-step reach checkpoint (4% deterministic success; a 193.7 rad/s simulation velocity spike). Reduced the action increment to 0.02 rad, added explicit 1.0 rad/s and 10 Nm simulation caps, narrowed the initial reach curriculum, and changed the dense reward to emphasize distance progress.
