# Decisions

- 2026-09-19: Selected `https://github.com/alboyq/HackMIT_2026.git`; no repo URL was filled into the addition, and this is the local checkout whose origin contains “HackMIT”.
- 2026-09-19: Work in the separate `arm-ik-rl` worktree because teammates actively change the shared checkout. No teammate file is modified; all implementation is under `arm/` plus root documentation.
- 2026-09-19: Reuse the teammate's measured fingertip TCP, scene generation, and tested YAM IK/expert logic from `yam_agentic`; preserve that directory unchanged.
- 2026-09-19: Use the Menagerie `i2rt_yam` MJCF as the single source of kinematics and simulation dynamics because Anvil's public URDF/hardware repositories are placeholders.
- 2026-09-19: Model six arm joints plus one gripper command and use 30 Hz control. Keep the policy input low-dimensional (joint state, relative 3D target, previous action); cameras remain outside the policy.
- 2026-09-19: No compatible pretrained single-arm low-dimensional YAM policy was found. `angkul07/pi0-fast-yam` is bimanual, image-conditioned, 14-action, and reports 34.4% decode collapse; train PPO from scratch and retain ABC-130k only as a later imitation-learning option.
- 2026-09-19: The first standalone 500k PPO experiment was rejected (4% deterministic reach success and an unsafe simulation velocity spike). Explicit velocity/torque/path limits are required before curriculum training.
- 2026-09-19: Preserve the implemented team UDP contracts on ports 8765/8766. Add adapters rather than replacing them with the proposed ZeroMQ schema.
- 2026-09-19: The current user is away; no physical-arm reads that transmit MIT frames, motor enables, motion, homing, zeroing, register writes, CAN reconfiguration, power changes, or real-backend tests are permitted.
