"""Watch the current policy run the task, on the GX10 monitor.

    DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
      .venv-arm/bin/python arm/rl/scripts/watch_policy.py --run-dir runs/feed-grasp --stage grasp

Loads the newest checkpoint for a stage and plays deterministic episodes, so what you see is
the policy as it stands right now -- re-run it later and you are watching a better one. Reading
checkpoints never disturbs the training job that writes them.

  --follow   re-load the newest checkpoint between episodes, so a long watch tracks progress
  --cam      scene_cam (default) or wrist_cam, the view the real wrist camera would give
  q / ESC    quit      SPACE  pause      n  skip to the next episode
"""
from __future__ import annotations

import argparse
import os
import re
import pickle
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
# The scene declares its camera resolution in the MJCF, and the offscreen framebuffer is
# sized from it -- so this has to be set BEFORE arm.ik.scene is imported.
os.environ.setdefault("YAM_CAM_RES", "480")

import cv2
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hackmit_rl.config import load_config
from hackmit_rl.envs import OpenYAMFeedEnv

STEPS = re.compile(r"_(\d+)_steps\.zip$")
WINDOW = "YAM policy"


def newest_checkpoint(run_dir: Path, stage: str):
    """Newest (model, vecnormalize) pair; a final/paused save wins over a periodic one."""
    stats_top = run_dir / "vecnormalize.pkl"
    for name in (f"ppo_{stage}_final.zip", f"ppo_{stage}_paused.zip"):
        top = run_dir / name
        if top.exists() and stats_top.exists():
            return top, stats_top, None
    best = None
    for zip_path in (run_dir / "checkpoints").glob(f"ppo_{stage}_*_steps.zip"):
        match = STEPS.search(zip_path.name)
        if not match:
            continue
        stats = zip_path.with_name(
            zip_path.name.replace("_steps.zip", "_steps.pkl")
                         .replace(f"ppo_{stage}_", f"ppo_{stage}_vecnormalize_"))
        if stats.exists() and stats.stat().st_size and zip_path.stat().st_size and (best is None or int(match.group(1)) > best[0]):
            best = (int(match.group(1)), zip_path, stats)
    if best is None:
        raise SystemExit(f"no checkpoint for stage {stage!r} in {run_dir}")
    return best[1], best[2], best[0]


def label(frame, lines):
    for i, (text, color) in enumerate(lines):
        y = 26 + i * 26
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1, cv2.LINE_AA)
    return frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=Path("runs/feed-reach"))
    ap.add_argument("--stage", default=None, help="defaults to the run-dir suffix")
    ap.add_argument("--config", default="arm/rl/configs/feed.yaml")
    ap.add_argument("--episodes", type=int, default=0, help="0 = run until you quit")
    ap.add_argument("--cam", default="scene_cam", choices=["scene_cam", "wrist_cam"])
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--res", type=int, default=480)
    ap.add_argument("--follow", action="store_true", help="re-load newest checkpoint each episode")
    args = ap.parse_args()

    stage = args.stage or args.run_dir.name.split("-")[-1]
    cfg = load_config(args.config)
    cfg["env"]["stage"] = stage

    raw = DummyVecEnv([lambda: OpenYAMFeedEnv(cfg)])
    inner = raw.envs[0]
    model_path, stats_path, step_count = newest_checkpoint(args.run_dir, stage)
    env = VecNormalize.load(str(stats_path), raw)
    env.training, env.norm_reward = False, False
    model = PPO.load(model_path, device="cpu")
    print(f"[watch] {stage} from {model_path.name}"
          f"{f' ({step_count} steps)' if step_count else ''}", flush=True)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 960, 720)
    period = 1.0 / max(1e-3, args.fps)
    episode, wins, paused = 0, 0, False

    while args.episodes == 0 or episode < args.episodes:
        if args.follow and episode:
            try:
                new_path, new_stats, step_count = newest_checkpoint(args.run_dir, stage)
                if new_path != model_path:
                    model_path = new_path
                    model = PPO.load(model_path, device="cpu")
                    # New weights need the normalisation they were trained against. Reloading
                    # only the model showed 0/9 on a policy that measured 17/30.
                    with open(new_stats, "rb") as fh:
                        env.obs_rms = pickle.load(fh).obs_rms
                    print(f"[watch] reloaded {model_path.name}", flush=True)
            except SystemExit:
                pass

        def draw_prior(e):
            # The frozen earlier stages run inside reset(); draw them too.
            frame = cv2.cvtColor(e.scene.render(args.cam, args.res), cv2.COLOR_RGB2BGR)
            label(frame, [(f"{e.stage} (frozen policy)  ->  hands off to {e.final_stage}"
                           f"   target: {e.name}", (0, 200, 255))])
            cv2.imshow(WINDOW, frame)
            cv2.waitKey(1)
            time.sleep(period)

        inner.prior_hook = draw_prior
        obs = env.reset()
        name = inner.name
        episode += 1
        success = False
        info = {}
        for step in range(int(cfg["env"]["episode_steps"])):
            t0 = time.time()
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                return
            if key == ord(" "):
                paused = not paused
            if key == ord("n"):
                break
            while paused:
                k = cv2.waitKey(50) & 0xFF
                if k == ord(" "):
                    paused = False
                if k in (ord("q"), 27):
                    cv2.destroyAllWindows()
                    return

            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = env.step(action)
            info = infos[0]
            frame = inner.scene.render(args.cam, args.res)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            held = "carrying" if info.get("carrying") else ("touching" if info.get("grip_force_n", 0) > 0 else "-")
            label(frame, [
                (f"{stage}  ep {episode}   target: {name}  ({info.get('object_width_m', 0)*1000:.0f} mm)",
                 (255, 255, 255)),
                (f"dist {info.get('distance', 0):.3f} m   grip {info.get('grip_force_n', 0):5.1f} N   {held}",
                 (0, 255, 255)),
                (f"qvel {info.get('max_joint_velocity', 0):.2f} / {info.get('velocity_cap', 1.0):.2f} rad/s"
                 f"   wall hits {info.get('wall_hits', 0)}   crush {info.get('crush_steps', 0)}",
                 (0, 200, 120)),
                (f"success {wins}/{max(1, episode - 1)}"
                 f"   [q]uit [space]pause [n]ext", (200, 200, 200)),
            ])
            cv2.imshow(WINDOW, frame)

            if done[0]:
                success = bool(info.get("success"))
                break
            time.sleep(max(0.0, period - (time.time() - t0)))

        wins += int(success)
        print(f"[watch] ep {episode}: {name:7s} {'SUCCESS' if success else 'fail'}"
              f"  running {wins}/{episode}", flush=True)

    cv2.destroyAllWindows()
    env.close()


if __name__ == "__main__":
    main()
