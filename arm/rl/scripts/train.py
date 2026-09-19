from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from hackmit_rl.config import load_config
from hackmit_rl.envs import OpenYAMReachEnv


class SuccessCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.done_count = 0
        self.success_count = 0

    def _on_step(self) -> bool:
        for info, done in zip(self.locals.get("infos", []), self.locals.get("dones", [])):
            if done:
                self.done_count += 1
                self.success_count += int(bool(info.get("success")))
        if self.n_calls % 1000 == 0 and self.done_count:
            self.logger.record("rollout/success_rate", self.success_count / self.done_count)
            self.logger.record("rollout/completed_episodes", self.done_count)
            self.done_count = 0
            self.success_count = 0
        return True


def make_env(cfg: dict, rank: int):
    def factory():
        return Monitor(OpenYAMReachEnv(cfg))
    return factory


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--stage", default="reach", choices=["reach", "grasp", "lift"])
    ap.add_argument("--n-envs", type=int)
    ap.add_argument("--timesteps", type=int)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--init-model", type=Path)
    ap.add_argument("--init-vecnormalize", type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg["env"]["stage"] = args.stage
    tcfg = cfg["training"]
    n_envs = args.n_envs or int(tcfg["n_envs"])
    args.run_dir.mkdir(parents=True, exist_ok=True)
    factories = [make_env(cfg, i) for i in range(n_envs)]
    raw_env = DummyVecEnv(factories) if n_envs == 1 else SubprocVecEnv(factories, start_method="forkserver")
    if args.init_vecnormalize:
        env = VecNormalize.load(args.init_vecnormalize, raw_env)
        env.training, env.norm_reward = True, True
    else:
        env = VecNormalize(raw_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    if args.init_model:
        model = PPO.load(args.init_model, env=env, device="cpu",
                         tensorboard_log=str(args.run_dir / "tensorboard"))
    else:
        model = PPO(
            "MlpPolicy", env,
            learning_rate=float(tcfg["learning_rate"]), n_steps=int(tcfg["n_steps"]),
            batch_size=int(tcfg["batch_size"]), gamma=float(tcfg["gamma"]),
            gae_lambda=float(tcfg["gae_lambda"]), ent_coef=float(tcfg["ent_coef"]),
            policy_kwargs={"net_arch": list(tcfg["policy_layers"])},
            tensorboard_log=str(args.run_dir / "tensorboard"), verbose=1, device="cpu",
            seed=int(cfg["project"]["seed"]),
        )
    checkpoint = CheckpointCallback(
        save_freq=max(1, int(tcfg["checkpoint_freq"]) // n_envs),
        save_path=str(args.run_dir / "checkpoints"),
        name_prefix=f"ppo_{args.stage}",
        save_vecnormalize=True,
    )
    try:
        model.learn(
            total_timesteps=args.timesteps or int(tcfg["total_timesteps"]),
            callback=[checkpoint, SuccessCallback()],
            tb_log_name=args.stage,
            progress_bar=False, reset_num_timesteps=args.init_model is None,
        )
        model.save(args.run_dir / f"ppo_{args.stage}_final")
        env.save(args.run_dir / "vecnormalize.pkl")
    finally:
        env.close()


if __name__ == "__main__":
    main()
