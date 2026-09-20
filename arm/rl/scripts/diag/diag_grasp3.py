"""Exploration or geometry? Let the policy position itself, then FORCE the jaws shut."""
import glob
import os
import re
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hackmit_rl.config import load_config
from hackmit_rl.envs import OpenYAMFeedEnv

RUN = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
cks = sorted(glob.glob(f"{RUN}/checkpoints/ppo_*_steps.zip"),
             key=lambda p: int(re.search(r"_(\d+)_steps", p).group(1)))
CK = cks[-1]
step = re.search(r"_(\d+)_steps", CK).group(1)
VN = glob.glob(f"{RUN}/checkpoints/*vecnormalize_{step}_steps.pkl")[0]
print("ckpt:", CK)

cfg = load_config("arm/rl/configs/feed.yaml")
cfg["env"]["stage"] = "grasp"
raw = DummyVecEnv([lambda: OpenYAMFeedEnv(cfg)])
inner = raw.envs[0]
env = VecNormalize.load(VN, raw)
env.training, env.norm_reward = False, False
m = PPO.load(CK, device="cpu")
print("action std per dim (6 joints + gripper):", np.round(m.policy.log_std.exp().detach().numpy(), 3))

# 1) what does the gripper action look like, mean and sampled?
obs = env.reset()
means, samples = [], []
for i in range(150):
    a_det, _ = m.predict(obs, deterministic=True)
    a_sto, _ = m.predict(obs, deterministic=False)
    means.append(float(a_det[0][6])); samples.append(float(a_sto[0][6]))
    obs, r, done, infos = env.step(a_det)
    if done[0]:
        break
means, samples = np.array(means), np.array(samples)
print(f"gripper action mean: min {means.min():.2f} med {np.median(means):.2f} max {means.max():.2f}   "
      f"(-1 closed, +1 open; 'closed' needs roughly < -0.1)")
print(f"gripper action sampled: min {samples.min():.2f}  frac < -0.1: {(samples < -0.1).mean():.2f}")

# 2) force-close after the policy has positioned itself
print("\nforce-close test: policy drives the arm, gripper forced to -1 from 15 steps after in_position")
print("name    t_pos  xy(mm) dz(mm)  width(mm)  pinched  best_hold  success  disp(mm)  touch")
ok = 0
for ep in range(N):
    obs = env.reset()
    name = inner.name
    t_pos = None
    pinched_any = touch_any = succ = False
    hold = best = 0
    xy = dz = None
    for i in range(300):
        a, _ = m.predict(obs, deterministic=True)
        if t_pos is not None and i >= t_pos + 15:
            if xy is None:
                tcp, op = inner._tcp(), inner.scene.object_pos(name)
                xy, dz = float(np.linalg.norm(tcp[:2] - op[:2])), float(tcp[2] - op[2])
            a[0][6] = -1.0
        obs, r, done, infos = env.step(a)
        info = infos[0]
        if done[0]:
            succ = bool(info.get("success"))
            break
        if t_pos is None and info["in_position"]:
            t_pos = i
        pinched_any |= bool(info["pinched"])
        touch_any |= info["grip_force_n"] > 0
        hold = hold + 1 if (info["pinched"] and info["grip_fraction"] < 0.45) else 0
        best = max(best, hold)
    ok += succ
    print(f"{name:6s} {t_pos if t_pos is not None else -1:5d}  {1000*(xy or 0):6.1f} {1000*(dz or 0):6.1f}  "
          f"{1000*inner.width[name]:8.1f}  {str(pinched_any):7s}  {best:8d}  {str(succ):7s}  "
          f"{1000*info['object_displaced_m']:7.1f}  {touch_any}")
print(f"\nforce-close success: {ok}/{N}")
env.close()
