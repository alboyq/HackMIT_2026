"""Record feed episodes to .mp4, headless. Same picture as watch_feed.py (side view, wrist-camera inset,
live readouts), one file per episode, named by how the episode ended.
usage: record_feed.py <out_dir> <n_episodes>      (env: SEED, CALIBRATE)"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("YAM_CAM_RES", "320")
OUT, N_EP = sys.argv[1], int(sys.argv[2])
sys.argv = [sys.argv[0]]                      # hybrid_feed reads argv for its own options

import cv2  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hybrid_feed as hf  # noqa: E402
import watch_feed as wf  # noqa: E402

os.makedirs(OUT, exist_ok=True)
state = {"writer": None, "path": None, "frames": 0}
SCALE = 2                                      # 640x480 render -> 1280x960 file


def start():
    state["path"] = os.path.join(OUT, f"_recording_{os.getpid()}.mp4")
    state["writer"] = cv2.VideoWriter(state["path"], cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640 * SCALE, 480 * SCALE))
    state["frames"] = 0


def show(img):                                 # replaces the on-screen show(): no window, no sleeping
    if state["writer"] is None:
        start()
    state["writer"].write(cv2.resize(img, (640 * SCALE, 480 * SCALE), interpolation=cv2.INTER_LINEAR))
    state["frames"] += 1


wf.show = show
_hook = wf.hook


def hook(e, phase, size, o, info, ctx):
    if phase == "done":
        import time as _t
        real = _t.time
        # the end-of-episode banner loops on wall-clock time; give it a fixed 70 frames instead
        ticks = {"n": 0}

        def fake():
            ticks["n"] += 1
            return ticks["n"] / 30.0

        wf.time.time = fake
        try:
            _hook(e, phase, size, o, info, ctx)
        finally:
            wf.time.time = real
        state["writer"].release()
        tag = ("FED" if o["result"].startswith("ok") else o["result"][2:].strip().split(" (")[0]).replace(" ", "_").replace("/", "-")
        final = os.path.join(OUT, f"s{os.environ.get('SEED', '0')}_ep{ctx['episode']:02d}_{o['obj']}_{tag[:40]}.mp4")
        os.replace(state["path"], final)
        print(f"[rec] {os.path.basename(final)}  {state['frames']} frames ({state['frames']/30:.0f} s)", flush=True)
        state["writer"] = None
        return
    _hook(e, phase, size, o, info, ctx)


hf.STEP_HOOK = hook
hf.main(n_episodes=N_EP, report=False)
