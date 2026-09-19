# OpenYAM arm stack

This directory is isolated from teammate-owned code. `ik/` is the default, no-learning pick/lift/deliver path; `rl/` is a low-dimensional PPO curriculum using exactly the same Menagerie YAM model and a fixed 23-value observation / 7-value action interface. Nothing here commands hardware by default.

```bash
cd ~/HackMIT_2026-arm-ik-rl
export YAM_MENAGERIE=$PWD/third_party/mujoco_menagerie
source .venv-arm/bin/activate
pytest arm/ik/tests -q
python -m arm.ik.evaluate_ik --output reports/ik_baseline.json

cd arm/rl
bash scripts/launch_training.sh reach
bash scripts/launch_tensorboard.sh
```

Controller selection is `ik`, `rl`, or `auto`; default `ik`. `auto` refuses RL unless an evaluation report shows RL success at least equals IK with zero safety violations. RL is never used for the scripted mouth approach.

The real backend is deliberately disabled until the physical joint map and FK are verified. `python -m arm.deploy` is a dry run; `--enable-hardware` still fails closed rather than importing a CAN driver. `can_setup.sh` is documentation only and must not be run while the arm owner is away.

Required USB devices are the CAN adapter, OpenBCI/EEG dongle, wrist camera, and optionally an iPhone USB connection. That is at least three and often four ports; use a powered, reputable USB 3 hub if the GX10 has fewer free ports. Any wireless iPhone stream must use a private router or hotspot, never venue Wi-Fi.

No `/dev/video*` device was visible during the simulation-only setup, and `v4l2-ctl` was unavailable, so camera resolution/FPS/depth must be checked when cameras are attached. The known wrist unit from pre-hack notes is a UVC “USB CAMERA 4K”; depth is not established.
