"""How far down the grab does it get? centred -> placed (may close) -> pinched -> seated pinch -> lifted -> held."""
import glob, os, re, sys
from collections import Counter
import numpy as np
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["YAM_HANDOFF_REUSE"] = "0"
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from hackmit_rl.config import load_config
from hackmit_rl.envs import OpenYAMFeedEnv
RUN, N = sys.argv[1], int(sys.argv[2])
cks = [c for c in glob.glob(f"{RUN}/checkpoints/ppo_grasp_*_steps.zip") if os.path.getsize(c) > 0]
CK = max(cks, key=lambda p: int(re.search(r"_(\d+)_steps", p).group(1))); step = re.search(r"_(\d+)_steps", CK).group(1)
VN = glob.glob(f"{RUN}/checkpoints/*vecnormalize_{step}_steps.pkl")[0]
cfg = load_config("arm/rl/configs/feed.yaml"); cfg["env"]["stage"] = "grasp"
raw = DummyVecEnv([lambda: OpenYAMFeedEnv(cfg)]); e = raw.envs[0]
env = VecNormalize.load(VN, raw); env.training, env.norm_reward = False, False
m = PPO.load(CK, device="cpu"); ms = float(cfg["env"]["min_seat_frac"])
print("ckpt", CK)
funnel, ends, lens = Counter(), Counter(), []
best_seat, best_lift, lat_min = [], [], []
for ep in range(N):
    obs = env.reset(); f = set(); bs = bl = 0.0; lm = 9.9; touched_early = 0
    for i in range(300):
        a, _ = m.predict(obs, deterministic=True)
        obs, r, done, infos = env.step(a); info = infos[0]
        lm = min(lm, 1000*np.hypot(info["e_jaw_m"], info["e_pad_m"])); bs = max(bs, info["seat_frac"]); bl = max(bl, 1000*info["lift_m"])
        if info["centred"]: f.add("1 centred")
        if info["centred"] and info["seat_frac"] >= ms: f.add("2 centred+seated")
        if info["in_position"]: f.add("3 placed (may close)")
        if info["grip_fraction"] < 0.45: f.add("4 closed jaws")
        if info["pinched"]: f.add("5 pinched")
        if info["pinched"] and info["seat_frac"] >= ms: f.add("6 seated pinch")
        if info["carrying"]: f.add("7 lifted to height")
        if done[0]: break
    if info["success"]: f.add("8 HELD = success")
    ends["success" if info["success"] else ("shoved/dropped" if info.get("failed") else "timeout")] += 1
    for k in f: funnel[k] += 1
    lens.append(i + 1); best_seat.append(bs); best_lift.append(bl); lat_min.append(lm)
for k in sorted(funnel): print(f"  {funnel[k]:3d}/{N}  {k}")
print("ends:", dict(ends), " mean length", round(np.mean(lens)))
print(f"best centring reached: mean {np.mean(lat_min):.1f} mm   best seat: mean {np.mean(best_seat):.2f}   best lift: mean {np.mean(best_lift):.0f} mm")
