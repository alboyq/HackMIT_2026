"""Signal-safe PPO launcher: resumes from the newest checkpoint and saves on SIGINT/SIGTERM.

    python arm/rl/scripts/run_resumable.py --run-dir runs/feed --stage reach --timesteps 3000000

Ctrl-C (or `kill <pid>`) writes ppo_<stage>_paused.zip + vecnormalize.pkl and exits. Re-run the
exact same command to continue: the newest checkpoint is found automatically, optimizer state and
step count come back with it, and --timesteps is the TOTAL target, not the remainder.
"""
from __future__ import annotations

import argparse
import os
import re
import signal
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from hackmit_rl.config import load_config
from hackmit_rl.envs import OpenYAMFeedEnv, OpenYAMReachEnv

ENVS = {"feed": OpenYAMFeedEnv, "reach": OpenYAMReachEnv}

STEPS = re.compile(r"_(\d+)_steps\.zip$")


def newest_checkpoint(run_dir: Path, stage: str):
    """Latest (model, vecnormalize) pair, preferring a paused save over a periodic one."""
    stats_top = run_dir / "vecnormalize.pkl"
    for name in (f"ppo_{stage}_paused.zip", f"ppo_{stage}_final.zip"):
        top = run_dir / name
        if top.exists() and stats_top.exists():
            return top, stats_top
    best = None
    for zip_path in (run_dir / "checkpoints").glob(f"ppo_{stage}_*_steps.zip"):
        match = STEPS.search(zip_path.name)
        if not match:
            continue
        stats = zip_path.with_name(zip_path.name.replace("_steps.zip", "_steps.pkl")
                                   .replace(f"ppo_{stage}_", f"ppo_{stage}_vecnormalize_"))
        if stats.exists() and (best is None or int(match.group(1)) > best[0]):
            best = (int(match.group(1)), zip_path, stats)
    return (best[1], best[2]) if best else (None, None)


class StopOnSignal(BaseCallback):
    """Latches on SIGINT/SIGTERM and stops learn() cleanly at the next step boundary."""

    def __init__(self):
        super().__init__()
        self.triggered = False
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._latch)

    def _latch(self, *_):
        if not self.triggered:
            print("\n[pause] signal received; finishing the current step then saving", flush=True)
        self.triggered = True

    def _on_step(self) -> bool:
        return not self.triggered


def reset_gripper_head(model: PPO) -> None:
    """Un-saturate the gripper output when leaving reach.

    Reach charges for any closing and nothing charges for overshooting the action clip, so it
    hands over a gripper mean far past +1 (measured: +1.87 unclipped, std 0.54). Every sample
    then clips to the same fully-open command, the close reward sees no variance, and the
    gradient is exactly zero: grasp sat at 0/30 jaws-ever-closed for 1.4M steps while forcing
    the jaws shut at the same pose succeeded 8/20. Only the gripper row is touched; the arm
    positioning that reach learned is kept.
    """
    with torch.no_grad():
        before = float(model.policy.action_net.bias[-1])
        model.policy.action_net.weight[-1].zero_()
        model.policy.action_net.bias[-1] = 0.0
        model.policy.log_std[-1] = 0.0
    print(f"[transfer] gripper head reset (bias {before:+.2f} -> 0, std -> 1.0)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="arm/rl/configs/feed.yaml")
    ap.add_argument("--env", default="feed", choices=sorted(ENVS))
    ap.add_argument("--init-from", type=Path,
                    help="seed from another run-dir (stage-to-stage transfer)")
    ap.add_argument("--stage", default="reach",
                    choices=["reach", "grasp", "lift", "present"])
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--timesteps", type=int, help="TOTAL target across resumes")
    ap.add_argument("--n-envs", type=int)
    ap.add_argument("--n-steps", type=int)
    ap.add_argument("--fresh", action="store_true", help="ignore existing checkpoints")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["env"]["stage"] = args.stage
    tcfg = cfg["training"]
    n_envs = args.n_envs or int(tcfg["n_envs"])
    n_steps = args.n_steps or int(tcfg["n_steps"])
    total = args.timesteps or int(tcfg["total_timesteps"])
    args.run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    env_cls = ENVS[args.env]
    factories = [(lambda: Monitor(env_cls(cfg))) for _ in range(n_envs)]
    raw = DummyVecEnv(factories) if n_envs == 1 else SubprocVecEnv(factories, start_method="forkserver")

    model_path, stats_path = (None, None) if args.fresh else newest_checkpoint(args.run_dir, args.stage)
    transferred = False
    transferred_from = None
    if model_path is None and args.init_from and not args.fresh:
        # Curriculum transfer. The observation layout is identical across stages, so a
        # finished stage is a legitimate warm start for the next one.
        for prior in ("present", "lift", "grasp", "reach"):
            model_path, stats_path = newest_checkpoint(args.init_from, prior)
            if model_path:
                transferred = True
                transferred_from = prior
                print(f"[transfer] seeding {args.stage} from {prior}: {model_path.name}")
                break
    if model_path:
        env = VecNormalize.load(str(stats_path), raw)
        env.training, env.norm_reward = True, True
        model = PPO.load(model_path, env=env, device="cpu",
                         tensorboard_log=str(args.run_dir / "tensorboard"))
        if transferred and transferred_from == "reach" and args.stage != "reach":
            reset_gripper_head(model)
        done = 0 if transferred else model.num_timesteps
        print(f"[resume] {model_path.name} at {done} steps; {max(0, total - done)} remaining")
        if done >= total:
            print(f"[resume] target already reached at {done} steps; writing final")
            model.save(args.run_dir / f"ppo_{args.stage}_final")
            env.save(str(args.run_dir / "vecnormalize.pkl"))
            env.close()
            return
        remaining = total - done
    else:
        env = VecNormalize(raw, norm_obs=True, norm_reward=True, clip_obs=10.0)
        model = PPO("MlpPolicy", env, learning_rate=float(tcfg["learning_rate"]), n_steps=n_steps,
                    batch_size=int(tcfg["batch_size"]), gamma=float(tcfg["gamma"]),
                    gae_lambda=float(tcfg["gae_lambda"]), ent_coef=float(tcfg["ent_coef"]),
                    policy_kwargs={"net_arch": list(tcfg["policy_layers"])},
                    tensorboard_log=str(args.run_dir / "tensorboard"), verbose=1, device="cpu",
                    seed=int(cfg["project"]["seed"]))
        print(f"[start] fresh run, target {total} steps, {n_envs} envs")
        remaining = total

    stopper = StopOnSignal()
    checkpoint = CheckpointCallback(save_freq=max(1, int(tcfg["checkpoint_freq"]) // n_envs),
                                    save_path=str(args.run_dir / "checkpoints"),
                                    name_prefix=f"ppo_{args.stage}", save_vecnormalize=True)
    try:
        model.learn(total_timesteps=remaining, callback=[checkpoint, stopper],
                    tb_log_name=args.stage, progress_bar=False,
                    reset_num_timesteps=model_path is None or transferred)
    finally:
        # Derive from progress, not from the signal flag: SIGINT reaches the worker
        # processes too, so learn() can unwind without the parent handler having latched.
        tag = "final" if model.num_timesteps >= total else "paused"
        model.save(args.run_dir / f"ppo_{args.stage}_{tag}")
        env.save(str(args.run_dir / "vecnormalize.pkl"))
        print(f"[saved] {args.stage}_{tag} at {model.num_timesteps} steps -> {args.run_dir}")
        env.close()


if __name__ == "__main__":
    main()
