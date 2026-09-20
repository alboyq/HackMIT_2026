"""Where on the finger does it grip? Seating depth, tilt and squareness at the moment of success."""
import glob, os, re, sys
import numpy as np
os.environ.setdefault("MUJOCO_GL", "egl")
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from hackmit_rl.config import load_config
from hackmit_rl.envs import OpenYAMFeedEnv
RUN, N = sys.argv[1], int(sys.argv[2])
cks = sorted(glob.glob(f"{RUN}/checkpoints/ppo_grasp_*_steps.zip"), key=lambda p: int(re.search(r"_(\d+)_steps", p).group(1)))
CK = cks[-1]; step = re.search(r"_(\d+)_steps", CK).group(1)
VN = glob.glob(f"{RUN}/checkpoints/*vecnormalize_{step}_steps.pkl")[0]
cfg = load_config("arm/rl/configs/feed.yaml"); cfg["env"]["stage"] = "grasp"
raw = DummyVecEnv([lambda: OpenYAMFeedEnv(cfg)]); inner = raw.envs[0]
env = VecNormalize.load(VN, raw); env.training, env.norm_reward = False, False
m = PPO.load(CK, device="cpu")
print("ckpt", CK)
res = {}
for ep in range(N):
    obs = env.reset(); name = inner.name; last = None; pin_unseated = 0
    for i in range(300):
        a, _ = m.predict(obs, deterministic=True)
        obs, r, done, infos = env.step(a); info = infos[0]
        if info["pinched"] and info["seat_frac"] < cfg["env"]["min_seat_frac"]:
            pin_unseated += 1
        if done[0]:
            break
    res.setdefault(name, []).append((bool(info["success"]), info["seat_frac"], info["tilt_deg"], info["off_square_deg"],
                                     bool(info.get("failed")), i + 1, pin_unseated, 1000*info["stage_drift_m"]))
tot = sum(x[0] for v in res.values() for x in v)
print(f"SUCCESS {tot}/{N}")
for name, v in res.items():
    ok = [x for x in v if x[0]]
    print(f"{name:6s} {len(ok)}/{len(v)}   at success: seated {np.mean([x[1] for x in ok]) if ok else float('nan'):.2f} of max depth, "
          f"tilt {np.mean([x[2] for x in ok]) if ok else float('nan'):.1f} deg, off-square {np.mean([x[3] for x in ok]) if ok else float('nan'):.1f} deg, "
          f"{np.mean([x[5] for x in ok]) if ok else float('nan'):.0f} steps | failures: shoved>4cm {sum(x[4] for x in v)}, "
          f"pinched-but-shallow steps {sum(x[6] for x in v if not x[0])}")
