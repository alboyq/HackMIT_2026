# HackMIT OpenYAM RL

Low-dimensional PPO policy for reaching a 3D food target with the six-axis OpenYAM. The policy receives joint state plus target coordinates in the arm base frame; it never receives images. Feeding delivery remains scripted.

```bash
# This code is arm/rl/ in this repository. `~/hackmit_rl` was one machine's deploy
# directory and is not required; a clone works.
cd "$(git rev-parse --show-toplevel)"
source .venv-arm/bin/activate
export PYTHONPATH=.:arm/rl
pytest -q

# launch reach training (12 envs, 500k steps)
bash scripts/launch_training.sh reach
tail -f runs/latest/train.log

# after training finishes
python scripts/evaluate.py --run-dir runs/latest --episodes 100

# tensorboard
bash scripts/launch_tensorboard.sh
```

From the laptop, view TensorBoard with:

```bash
ssh -L 6006:localhost:6006 gx10-remote   # gx10-lan / 10.189.40.190 is venue-LAN only
```

Then open <http://localhost:6006>.

The initial training job is CPU-only, runs at nice priority 10, limits BLAS to one thread per worker, and defaults to 12 workers. Override with `N_ENVS=4` for a small validation run.
