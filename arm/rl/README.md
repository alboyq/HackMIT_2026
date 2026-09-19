# HackMIT OpenYAM RL

Low-dimensional PPO policy for reaching a 3D food target with the six-axis OpenYAM. The policy receives joint state plus target coordinates in the arm base frame; it never receives images. Feeding delivery remains scripted.

```bash
cd ~/hackmit_rl
source .venv/bin/activate
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
ssh -L 6006:localhost:6006 asus@10.189.40.190
```

Then open <http://localhost:6006>.

The initial training job is CPU-only, runs at nice priority 10, limits BLAS to one thread per worker, and defaults to 12 workers. Override with `N_ENVS=4` for a small validation run.
