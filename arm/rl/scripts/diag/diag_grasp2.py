"""Grasp failure census: WHEN does the object get displaced, relative to contact/closing/pinch?"""
import glob
import os
import re
import sys
from collections import Counter

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hackmit_rl.config import load_config
from hackmit_rl.envs import OpenYAMFeedEnv

RUN = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
DET = (sys.argv[3] != "stoch") if len(sys.argv) > 3 else True
cks = sorted(glob.glob(f"{RUN}/**/ppo_grasp_*_steps.zip", recursive=True),
             key=lambda p: int(re.search(r"_(\d+)_steps", p).group(1)))
CK = cks[-1]
vns = glob.glob(CK.replace("ppo_grasp_", "*vecnormalize*").replace(".zip", ".pkl")) or \
    sorted(glob.glob(f"{RUN}/**/*vecnormalize*{re.search(r'_(\d+)_steps', CK).group(1)}*", recursive=True))
print("ckpt:", CK)
print("vn  :", vns)
VN = vns[0]

cfg = load_config("arm/rl/configs/feed.yaml")
cfg["env"]["stage"] = "grasp"
raw = DummyVecEnv([lambda: OpenYAMFeedEnv(cfg)])
inner = raw.envs[0]
env = VecNormalize.load(VN, raw)
env.training, env.norm_reward = False, False
m = PPO.load(CK, device="cpu")
e = inner.ecfg
GATE = float(e["grasp_max_displacement_m"])

rows = []
outcome = Counter()
per_obj = Counter()
per_obj_n = Counter()
for ep in range(N):
    obs = env.reset()
    name = inner.name
    rest_z = inner.rest_z[name]
    t_touch = t_pos = t_close = t_pinch = t_gate = None
    disp_at_touch = disp_at_pinch = disp_at_pos = None
    z_min = 9.9
    pinch_closed_steps = best_hold = hold = 0
    ret = 0.0
    phase_steps = Counter()
    tcp0 = inner._tcp().copy()
    op0 = inner.scene.object_pos(name).copy()
    for i in range(int(cfg["env"]["episode_steps"])):
        a, _ = m.predict(obs, deterministic=DET)
        obs, r, done, infos = env.step(a)
        info = infos[0]
        ret += float(r[0])
        if done[0]:
            succ = bool(info.get("success"))
            break
        d = info["object_displaced_m"]
        phase_steps[info["phase"]] += 1
        z_min = min(z_min, info["object_height"] - rest_z)
        if t_touch is None and info["grip_force_n"] > 0:
            t_touch, disp_at_touch = i, d
        if t_pos is None and info["in_position"]:
            t_pos, disp_at_pos = i, d
        if t_close is None and info["grip_fraction"] < 0.45:
            t_close = i
        if t_pinch is None and info["pinched"]:
            t_pinch, disp_at_pinch = i, d
        if t_gate is None and d > GATE:
            t_gate = i
        pc = info["pinched"] and info["grip_fraction"] < 0.45
        hold = hold + 1 if pc else 0
        best_hold = max(best_hold, hold)
    else:
        succ = False
    per_obj_n[name] += 1
    per_obj[name] += succ
    if succ:
        oc = "SUCCESS"
    elif t_pos is None and t_touch is None:
        oc = "1 never in position, never touched"
    elif t_gate is not None and (t_pinch is None or t_gate <= t_pinch):
        oc = "4a displaced past gate BEFORE pinch" + (" (then pinched)" if t_pinch is not None else " (never pinched)")
    elif t_pinch is not None and t_gate is not None:
        oc = "4b pinched first, displaced after"
    elif t_pinch is not None:
        oc = "4c pinched, within gate, hold too short"
    elif t_close is None:
        oc = "2 in position/touched but never closed"
    else:
        oc = "3 closed but never both pads"
    outcome[oc] += 1
    rows.append((name, oc, t_pos, t_touch, t_close, t_pinch, t_gate, disp_at_touch, disp_at_pinch,
                 z_min, best_hold, ret, dict(phase_steps), float(np.linalg.norm(tcp0 - op0))))

print(f"\n{'det' if DET else 'stochastic'} policy, {N} episodes, gate {GATE*1000:.0f} mm")
for k, v in outcome.most_common():
    print(f"  {v:3d}  {k}")
print("per object success:", {k: f"{per_obj[k]}/{per_obj_n[k]}" for k in per_obj_n})
print("\nname   t_pos t_touch t_close t_pinch t_gate  disp@touch disp@pinch  zmin(mm) best_hold  return  phases")
for r in rows:
    f = lambda v, s="{:5d}": (s.format(v) if v is not None else "    -")
    print(f"{r[0]:6s} {f(r[2])} {f(r[3])}   {f(r[4])}   {f(r[5])}  {f(r[6])}   "
          f"{f(r[7], '{:8.3f}')}   {f(r[8], '{:8.3f}')}   {r[9]*1000:7.1f}   {r[10]:5d}   {r[11]:7.1f}  {r[12]}")
env.close()
