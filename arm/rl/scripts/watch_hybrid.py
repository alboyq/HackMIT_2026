"""Live view of the hybrid pick-up on the GX10's monitor: frozen reach -> learned place+pinch ->
scripted vertical lift. Real time; success is the environment's own pick-up test."""
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["YAM_HANDOFF_REUSE"] = "0"
os.environ.setdefault("YAM_CAM_RES", "480")

import cv2
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_pick import lift_action, load  # noqa: E402

from hackmit_rl.config import load_config
from hackmit_rl.envs import OpenYAMFeedEnv

RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/grasp-pinch-best"
WINDOW, FPS = "OpenYAM - hybrid pick-up", 30.0


def label(frame, lines):
    for k, (text, colour) in enumerate(lines):
        y = 26 + 24 * k
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1, cv2.LINE_AA)


def main():
    ck, vn = load(RUN)
    cfg = load_config("arm/rl/configs/feed.yaml")
    cfg["env"]["stage"] = "grasp"
    raw = DummyVecEnv([lambda: OpenYAMFeedEnv(cfg)])
    e = raw.envs[0]
    env = VecNormalize.load(vn, raw)
    env.training, env.norm_reward = False, False
    model = PPO.load(ck, device="cpu")
    jacp, jacr = np.zeros((3, e.model.nv)), np.zeros((3, e.model.nv))
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 960, 720)
    period = 1.0 / FPS
    recent, episode = [], 0

    def show(lines):
        frame = cv2.cvtColor(e.scene.render("scene_cam", 480), cv2.COLOR_RGB2BGR)
        label(frame, lines)
        cv2.imshow(WINDOW, frame)
        return cv2.waitKey(1) & 0xFF

    def draw_reach(en):
        show([(f"1/3 POSITION (learned, frozen)   target: {en.name}", (0, 200, 255))])
        time.sleep(period)

    while True:
        e.prior_hook = draw_reach
        obs = env.reset()
        episode += 1
        seated_steps, anchor, info = 0, None, {}
        for i in range(300):
            t0 = time.time()
            if anchor is None:
                a, _ = model.predict(obs, deterministic=True)
            else:
                a = lift_action(e, anchor, jacp, jacr, info.get("lift_m", 0.0))
            e.prior_hook = None      # the auto-reset inside step() must not draw the NEXT episode's reach
            obs, _, done, infos = env.step(a)
            info = infos[0]
            if anchor is None:
                ok = info["pinched"] and info["seat_frac"] >= float(cfg["env"]["min_seat_frac"])
                seated_steps = seated_steps + 1 if ok else 0
                if seated_steps >= 3 and not done[0]:
                    anchor = e._tcp().copy()
            stage = ("2/3 PLACE + PINCH (learned)" if anchor is None
                     else "3/3 LIFT STRAIGHT UP + HOLD (scripted)")
            rate = f"{sum(recent)}/{len(recent)}" if recent else "-"
            key = show([
                (f"{stage}   target: {info['object']}", (0, 255, 0) if anchor is not None else (255, 255, 255)),
                (f"lift {1000*info['lift_m']:4.0f} mm   sideways {1000*info['stage_drift_m']:4.1f} mm (limit {1000*float(cfg["env"]["pick_max_drift_m"]):.0f})"
                 f"   seated {info['seat_frac']:.2f}", (0, 255, 255)),
                (f"pick-ups, last {len(recent)}: {rate}   ep {episode}   [q] quit", (200, 200, 200)),
            ])
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                return
            if done[0]:
                recent.append(bool(info["success"]))
                del recent[:-20]
                print(f"[hybrid] ep {episode}: {info['object']:6s} "
                      f"{'PICKED UP' if info['success'] else 'fail'}  last-{len(recent)} {sum(recent)}/{len(recent)}",
                      flush=True)
                end = time.time() + 0.8
                while time.time() < end:
                    show([("PICKED UP" if info["success"] else "fail", (0, 255, 0) if info["success"] else (0, 0, 255))])
                    time.sleep(period)
                break
            time.sleep(max(0.0, period - (time.time() - t0)))


if __name__ == "__main__":
    main()
