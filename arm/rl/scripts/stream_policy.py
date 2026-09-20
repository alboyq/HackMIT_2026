"""Stream the policy running in sim as MJPEG over HTTP, so it can be watched from a laptop.

    python arm/rl/scripts/stream_policy.py --run-dir runs/feed-reach --stage reach --follow

Then from the laptop, tunnel the port and open http://localhost:8089 in a browser:

    ssh -J jump@129.153.206.6 -L 8089:localhost:8089 asus@10.10.10.4

MJPEG over a tunnel is the one approach that needs nothing installed on the viewing machine --
no X server, no VNC client, just a browser. X11 forwarding would need an X server on Windows.

Reads checkpoints only, so it never disturbs the training job writing them.
"""
from __future__ import annotations

import argparse
import os
import re
import threading
import pickle
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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

def chain_stage(runs_dir: Path) -> str | None:
    """Whichever stage the chain last announced. The chain log is the authority on that --
    guessing from checkpoint mtimes races the writer."""
    log = runs_dir / "chain.log"
    if not log.exists():
        return None
    stage = None
    try:
        for line in log.read_text(errors="ignore").splitlines():
            # "[chain] 20:16:12 starting grasp (3000000 steps) from ..." -- the timestamp
            # sits between the tag and the verb, so match on the verb alone.
            if " starting " in line:
                stage = line.split(" starting ", 1)[1].split()[0]
    except OSError:
        return None
    return stage


def train_stats(run_dir: Path, stage: str) -> str:
    """Scrape the trainer's own log. It is the only place SB3 reports rollout stats, and
    reading it costs the trainer nothing."""
    log = run_dir.parent / f"{run_dir.name}.log"
    if not log.exists():
        return "trainer: no log"
    try:
        tail = log.read_text(errors="ignore").splitlines()[-400:]
    except OSError:
        return "trainer: unreadable"
    wanted = {"total_timesteps": None, "ep_len_mean": None,
              "success_rate": None, "fps": None}
    for line in tail:                       # last occurrence of each key wins
        for key in wanted:
            if f"| {key}" in line or f"|    {key}" in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 2:
                    try:
                        wanted[key] = float(parts[1])
                    except ValueError:
                        pass
    steps = wanted["total_timesteps"]
    ep_len = wanted["ep_len_mean"]
    episodes = int(steps / ep_len) if steps and ep_len else None
    bits = [f"TRAINING {stage}"]
    if steps:
        bits.append(f"{int(steps):,} steps")
    if episodes:
        bits.append(f"~{episodes:,} episodes")
    if wanted["fps"]:
        bits.append(f"{int(wanted['fps']):,} steps/s")
    if wanted["success_rate"] is not None:
        bits.append(f"rollout success {wanted['success_rate']:.0%}")
    return "  |  ".join(bits)


_frame_lock = threading.Lock()
_frame: bytes | None = None
_status = {"text": "starting"}
_train_text = {"v": ""}

PAGE = b"""<!doctype html><html><head><title>YAM sim</title>
<style>body{background:#111;color:#ddd;font:14px system-ui;margin:0;text-align:center}
img{max-width:100%;height:auto;image-rendering:pixelated}
#s{padding:8px;font-family:ui-monospace,monospace;white-space:pre-line;line-height:1.5}</style></head>
<body><div id=s>connecting...</div><img src="/stream">
<script>setInterval(async()=>{try{const r=await fetch('/status');
document.getElementById('s').textContent=await r.text()}catch(e){}},1000)</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(PAGE)))
            self.end_headers()
            self.wfile.write(PAGE)
            return
        if self.path == "/status":
            body = _status["text"].encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path != "/stream":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
        self.end_headers()
        try:
            while True:
                with _frame_lock:
                    buf = _frame
                if buf is not None:
                    self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n"
                                     b"Content-Length: " + str(len(buf)).encode() + b"\r\n\r\n")
                    self.wfile.write(buf)
                    self.wfile.write(b"\r\n")
                time.sleep(1 / 30)
        except (BrokenPipeError, ConnectionResetError):
            pass


def newest(run_dir: Path, stage: str):
    stats_top = run_dir / "vecnormalize.pkl"
    for name in (f"ppo_{stage}_final.zip", f"ppo_{stage}_paused.zip"):
        if (run_dir / name).exists() and stats_top.exists():
            return run_dir / name, stats_top, None
    best = None
    for z in (run_dir / "checkpoints").glob(f"ppo_{stage}_*_steps.zip"):
        m = STEPS.search(z.name)
        vn = z.with_name(z.name.replace("_steps.zip", "_steps.pkl")
                          .replace(f"ppo_{stage}_", f"ppo_{stage}_vecnormalize_"))
        if m and vn.exists() and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), z, vn)
    if not best:
        raise SystemExit(f"no checkpoint for {stage} in {run_dir}")
    return best[1], best[2], best[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--config", default="arm/rl/configs/feed.yaml")
    ap.add_argument("--port", type=int, default=8089)
    ap.add_argument("--cam", default="scene_cam", choices=["scene_cam", "wrist_cam"])
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--auto-stage", action="store_true",
                    help="follow the curriculum: switch stage when the chain does")
    args = ap.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[stream] serving on localhost:{args.port} (tunnel it, then open in a browser)",
          flush=True)

    cfg = load_config(args.config)
    stage = args.stage
    runs_dir = args.run_dir.parent
    if args.auto_stage:
        found = chain_stage(runs_dir)
        if found:
            stage = found
            args.run_dir = runs_dir / f"feed-{stage}"
            print(f"[stream] auto-stage: following {stage}", flush=True)
    cfg["env"]["stage"] = stage
    raw = DummyVecEnv([lambda: OpenYAMFeedEnv(cfg)])
    inner = raw.envs[0]
    # A fresh run has no checkpoint for the first ~30 s; wait rather than die.
    while True:
        try:
            ck, vn, steps = newest(args.run_dir, stage)
            break
        except SystemExit:
            _status["text"] = f"waiting for the first {stage} checkpoint..."
            print(f"[stream] waiting for a {stage} checkpoint", flush=True)
            time.sleep(10)
    env = VecNormalize.load(str(vn), raw)
    env.training, env.norm_reward = False, False
    model = PPO.load(ck, device="cpu")
    print(f"[stream] {ck.name}", flush=True)

    global _frame
    episode = 0
    recent: list[bool] = []          # rolling window, so improving checkpoints show through
    period = 1.0 / max(1e-3, args.fps)
    while True:
        # Has the curriculum moved on? Swap the whole env, not just the weights: a later
        # stage is a different task, and replaying it under the old stage would be wrong.
        if args.auto_stage:
            found = chain_stage(runs_dir)
            if found and found != stage:
                print(f"[stream] stage changed {stage} -> {found}", flush=True)
                _status["text"] = f"switching to {found}..."
                try:
                    env.close()
                except Exception:
                    pass
                stage = found
                args.run_dir = runs_dir / f"feed-{stage}"
                cfg["env"]["stage"] = stage
                raw = DummyVecEnv([lambda: OpenYAMFeedEnv(cfg)])
                inner = raw.envs[0]
                while True:
                    try:
                        ck, vn, steps = newest(args.run_dir, stage)
                        break
                    except SystemExit:
                        _status["text"] = f"waiting for the first {stage} checkpoint..."
                        time.sleep(8)
                env = VecNormalize.load(str(vn), raw)
                env.training, env.norm_reward = False, False
                model = PPO.load(ck, device="cpu")
                recent.clear()
                episode = 0
        if args.follow and episode:
            try:
                nck, nvn, steps = newest(args.run_dir, stage)
                if nck != ck:
                    ck = nck
                    model = PPO.load(ck, device="cpu")
                    # and the normalisation those weights were trained against
                    with open(nvn, "rb") as fh:
                        env.obs_rms = pickle.load(fh).obs_rms
            except SystemExit:
                pass
        def draw_prior(e):
            # The frozen earlier stages run inside reset(); draw them so the viewer shows one
            # continuous reach -> grasp motion rather than an arm that appears at the object.
            global _frame
            img = cv2.cvtColor(e.scene.render(args.cam, 480), cv2.COLOR_RGB2BGR)
            for k, line in enumerate([f"{e.stage} (frozen policy)  ->  hands off to {e.final_stage}",
                                      f"target {e.name}"]):
                y = 22 + k * 22
                cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with _frame_lock:
                    _frame = buf.tobytes()
            time.sleep(period)

        inner.prior_hook = draw_prior
        obs = env.reset()
        episode += 1
        info = {}
        for i in range(int(cfg["env"]["episode_steps"])):
            t0 = time.time()
            a, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = env.step(a)
            info = infos[0]
            img = cv2.cvtColor(inner.scene.render(args.cam, 480), cv2.COLOR_RGB2BGR)
            lines = [f"{stage}  ep{episode}  {PHASE.get(info.get('phase', 0), '?')}",
                     f"target {info.get('object', '?')}  d={info.get('distance', 0):.3f}m",
                     f"grip {info.get('grip_fraction', 0):.2f}  "
                     f"qvel {info.get('joint_speed', 0):.2f}  "
                     f"tcp {info.get('tcp_speed_mps', 0):.2f}"]
            for k, line in enumerate(lines):
                y = 22 + k * 22
                cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with _frame_lock:
                    _frame = buf.tobytes()
            if i % 10 == 0:
                _train_text["v"] = train_stats(args.run_dir, stage)
            rate = f"{sum(recent)}/{len(recent)}" if recent else "-"
            _status["text"] = (f"WATCHING {ck.name}  |  ep {episode}  |  last-20 {rate}"
                               f"  |  step {i}\n{_train_text['v']}")
            if done[0]:
                recent.append(bool(info.get("success")))
                del recent[:-20]
                break
            time.sleep(max(0.0, period - (time.time() - t0)))


if __name__ == "__main__":
    main()
