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

