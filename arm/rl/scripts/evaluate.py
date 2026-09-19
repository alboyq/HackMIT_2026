from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hackmit_rl.config import load_config
from hackmit_rl.envs import OpenYAMReachEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=Path("runs/latest"))
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw = DummyVecEnv([lambda: OpenYAMReachEnv(cfg)])
    env = VecNormalize.load(args.run_dir / "vecnormalize.pkl", raw)
    env.training = False
    env.norm_reward = False
    model = PPO.load(args.run_dir / "ppo_reach_final", env=env, device="cpu")
    successes, times, max_velocities = [], [], []
    for _ in range(args.episodes):
        obs = env.reset()
        max_velocity = 0.0
        for step in range(int(cfg["env"]["episode_steps"])):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = env.step(action)
            max_velocity = max(max_velocity, float(infos[0]["max_joint_velocity"]))
            if done[0]:
                success = bool(infos[0].get("success", False))
                successes.append(success)
                if success:
                    times.append((step + 1) / float(cfg["env"]["control_hz"]))
                max_velocities.append(max_velocity)
                break
    print(f"episodes={len(successes)} success_rate={np.mean(successes):.3f}")
    print(f"mean_time_to_success_s={np.mean(times) if times else float('nan'):.3f}")
    print(f"max_joint_velocity_rad_s={max(max_velocities, default=0.0):.3f}")
    env.close()


if __name__ == "__main__":
    main()
