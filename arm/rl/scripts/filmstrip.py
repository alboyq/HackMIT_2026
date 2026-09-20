"""Render one episode of the current policy as a filmstrip, so the motion can be inspected
frame by frame instead of trusted from a success counter.

    python arm/rl/scripts/filmstrip.py --run-dir runs/feed-grasp --stage grasp

Each panel is captioned with the phase the reward is in at that moment (APPROACH / CLOSE /
SECURE), the gripper opening, and whether the object is touched -- which is what makes it
possible to see "the jaws shut before the arm arrived" rather than infer it.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("YAM_CAM_RES", "480")

import cv2
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hackmit_rl.config import load_config
from hackmit_rl.envs import OpenYAMFeedEnv

STEPS = re.compile(r"_(\d+)_steps\.zip$")
PHASE = {0: "APPROACH", 1: "CLOSE", 2: "SECURE"}


def newest(run_dir: Path, stage: str):
    stats_top = run_dir / "vecnormalize.pkl"
    for name in (f"ppo_{stage}_final.zip", f"ppo_{stage}_paused.zip"):
        if (run_dir / name).exists() and stats_top.exists():
            return run_dir / name, stats_top
    best = None
    for z in (run_dir / "checkpoints").glob(f"ppo_{stage}_*_steps.zip"):
        m = STEPS.search(z.name)
        vn = z.with_name(z.name.replace("_steps.zip", "_steps.pkl")
                          .replace(f"ppo_{stage}_", f"ppo_{stage}_vecnormalize_"))
        if m and vn.exists() and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), z, vn)
    if not best:
        raise SystemExit(f"no checkpoint for {stage} in {run_dir}")
    return best[1], best[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--config", default="arm/rl/configs/feed.yaml")
    ap.add_argument("--panels", type=int, default=6)
    ap.add_argument("--cam", default="scene_cam")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/filmstrip.jpg")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["env"]["stage"] = args.stage
    raw = DummyVecEnv([lambda: OpenYAMFeedEnv(cfg)])
    inner = raw.envs[0]
    ck, vn = newest(args.run_dir, args.stage)
    env = VecNormalize.load(str(vn), raw)
    env.training, env.norm_reward = False, False
    model = PPO.load(ck, device="cpu")
    print(f"[film] {ck.name}")

    obs = env.reset()
    frames, infos_seen = [], []
    steps = int(cfg["env"]["episode_steps"])
    for i in range(steps):
        a, _ = model.predict(obs, deterministic=True)
        obs, _, done, infos = env.step(a)
        frames.append(None)
        infos_seen.append(infos[0])
        if done[0]:
            break
    n = len(infos_seen)
    picks = sorted(set(np.linspace(0, n - 1, args.panels).astype(int)))

    # replay deterministically to capture only the chosen frames
    obs = env.reset()
    panels = []
    for i in range(n):
        a, _ = model.predict(obs, deterministic=True)
        obs, _, done, infos = env.step(a)
        if i in picks:
            img = cv2.cvtColor(inner.scene.render(args.cam, 480), cv2.COLOR_RGB2BGR)
            info = infos[0]
            txt = [f"t={i}  {PHASE.get(info.get('phase', 0), '?')}",
                   f"grip {info.get('grip_fraction', 0):.2f}  "
                   f"{'TOUCH' if info.get('grip_force_n', 0) > 0 else 'clear'}",
                   f"d={info.get('distance', 0):.3f}  z={info.get('object_height', 0):.3f}"]
            for k, line in enumerate(txt):
                y = 20 + k * 20
                cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            panels.append(img)
        if done[0]:
            break

    row = np.hstack(panels[:3]) if len(panels) >= 3 else np.hstack(panels)
    if len(panels) > 3:
        row2 = np.hstack(panels[3:6])
        if row2.shape[1] < row.shape[1]:
            pad = np.zeros((row2.shape[0], row.shape[1] - row2.shape[1], 3), np.uint8)
            row2 = np.hstack([row2, pad])
        row = np.vstack([row, row2])
    cv2.imwrite(args.out, row)
    last = infos_seen[-1]
    print(f"[film] {len(panels)} panels -> {args.out}   success={last.get('success')} "
          f"steps={n} displaced={last.get('object_displaced_m', 0):.3f}")
    env.close()


if __name__ == "__main__":
    main()
