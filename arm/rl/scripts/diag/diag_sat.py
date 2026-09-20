"""Is the gripper action mean saturated beyond the clip? Compare reach-final and latest grasp."""
import glob
import os
import re
import sys

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hackmit_rl.config import load_config
from hackmit_rl.envs import OpenYAMFeedEnv

cfg = load_config("arm/rl/configs/feed.yaml")
cfg["env"]["stage"] = "grasp"


def probe(tag, ck, vn):
    raw = DummyVecEnv([lambda: OpenYAMFeedEnv(cfg)])
    env = VecNormalize.load(vn, raw)
    env.training, env.norm_reward = False, False
    m = PPO.load(ck, device="cpu")
    w = m.policy.action_net.weight.detach().numpy()
    b = m.policy.action_net.bias.detach().numpy()
    print(f"\n== {tag}: {ck}")
    print("  log_std:", np.round(m.policy.log_std.detach().numpy(), 2))
    print(f"  action_net gripper row: bias {b[6]:+.2f}  |w| {np.linalg.norm(w[6]):.2f}   "
          f"(joint rows |w| {np.round(np.linalg.norm(w[:6], axis=1), 2)})")
    by_phase = {0: [], 1: [], 2: []}
    for ep in range(4):
        obs = env.reset()
        phase = 0
        for i in range(300):
            with torch.no_grad():
                t = m.policy.obs_to_tensor(obs)[0]
                mean = m.policy.get_distribution(t).distribution.mean.numpy()[0]
            by_phase[phase].append(mean[6])
            a, _ = m.predict(obs, deterministic=True)
            obs, r, done, infos = env.step(a)
            phase = infos[0]["phase"]
            if done[0]:
                break
    for p, v in by_phase.items():
        if v:
            v = np.array(v)
            print(f"  UNCLIPPED gripper mean in phase {p}: n={len(v):4d}  min {v.min():+.2f}  "
                  f"median {np.median(v):+.2f}  max {v.max():+.2f}")
    env.close()


probe("reach final", "runs/feed-reach/ppo_reach_final.zip", "runs/feed-reach/vecnormalize.pkl")
cks = sorted(glob.glob("runs/feed-grasp/checkpoints/ppo_*_steps.zip"),
             key=lambda p: int(re.search(r"_(\d+)_steps", p).group(1)))
step = re.search(r"_(\d+)_steps", cks[-1]).group(1)
probe("grasp latest", cks[-1], glob.glob(f"runs/feed-grasp/checkpoints/*vecnormalize_{step}_steps.pkl")[0])
