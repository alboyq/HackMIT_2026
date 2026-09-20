"""Does a grasp episode really begin where reach ended? Measure it."""
import os
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from hackmit_rl.config import load_config
from hackmit_rl.envs import OpenYAMFeedEnv

cfg = load_config("arm/rl/configs/feed.yaml")
cfg["env"]["stage"] = "grasp"
env = OpenYAMFeedEnv(cfg)
print("name   ok  reach_steps  tcp->obj xy(mm)  dz(mm)  obj moved(mm)  joint speed  grip  stage")
t0 = time.time()
oks, steps = [], []
for ep in range(25):
    obs, info = env.reset(seed=ep if ep == 0 else None)
    tcp, op = env._tcp(), env.scene.object_pos(env.name)
    moved = float(np.linalg.norm(op[:2] - env.object_start[:2]))
    grip = float((env.data.ctrl[env.grip_aid] - env.grip_range[0]) / np.ptp(env.grip_range))
    print(f"{env.name:6s} {str(info['handoff_ok']):5s} {info['handoff_steps']:6d}      "
          f"{1000*np.linalg.norm(tcp[:2]-op[:2]):8.1f}     {1000*(tcp[2]-op[2]):6.1f}   {1000*moved:8.1f}     "
          f"{np.abs(env.data.qvel[env.dadr]).max():8.3f}    {grip:.2f}  {env.stage} steps={env.steps}")
    oks.append(info["handoff_ok"]); steps.append(info["handoff_steps"])
    assert obs.shape == (33,) and env.stage == "grasp" and env.steps == 0
print(f"\nhand-off ok {sum(oks)}/{len(oks)}   mean reach steps {np.mean(steps):.1f}   "
      f"{(time.time()-t0)/len(oks)*1000:.0f} ms per reset")

# and the stage still runs normally afterwards
obs, r, term, trunc, info = env.step(np.zeros(7, dtype=np.float32))
print("first grasp step: phase", info["phase"], "in_position", info["in_position"], "reward", round(r, 3))

cfg["env"]["stage"] = "reach"
e2 = OpenYAMFeedEnv(cfg)
_, info = e2.reset(seed=0)
print("reach stage untouched: handoff_steps", info["handoff_steps"], "priors loaded:", e2._priors is not None)
