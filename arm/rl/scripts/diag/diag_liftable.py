"""Can the pinch the policy makes actually be LIFTED? Policy drives to a seated pinch, then a scripted
straight-up pull (Jacobian, jaws held shut) takes over."""
import glob, os, re, sys
import mujoco, numpy as np
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
m = PPO.load(CK, device="cpu"); md, d = e.model, e.data
jacp = np.zeros((3, md.nv)); jacr = np.zeros((3, md.nv))
print("obj    width  t_pinch  force@pinch  -> scripted lift:  max lift(mm)  slipped?  drift(mm)  peak force  success")
wins = 0
for ep in range(N):
    obs = env.reset(); seated_steps = 0; tp = None; took = False; maxlift = 0; info = {}
    for i in range(300):
        a, _ = m.predict(obs, deterministic=True)
        if took:
            mujoco.mj_jacSite(md, d, jacp, jacr, e.tool.site_id)
            J = np.vstack([jacp[:, e.dadr], jacr[:, e.dadr]])
            h = info.get("lift_m", 0.0)
            vz = 0.004 if h < 0.115 else 0.0                       # 4 mm per step up, then stop
            dq = np.linalg.pinv(J, rcond=1e-2) @ np.array([0, 0, vz, 0, 0, 0])
            a = np.zeros((1, 7), dtype=np.float32)
            a[0, :6] = np.clip(dq / float(cfg["env"]["action_delta_rad"]), -1, 1); a[0, 6] = -1.0
        obs, r, done, infos = env.step(a); info = infos[0]
        if not took:
            seated_steps = seated_steps + 1 if (info["pinched"] and info["seat_frac"] >= 0.6) else 0
            if seated_steps >= 3:
                took, tp, f0 = True, i, info["grip_force_n"]
        else:
            maxlift = max(maxlift, 1000 * info["lift_m"])
        if done[0]: break
    if not took:
        print(f"{info['object']:6s}  policy never made a seated pinch"); continue
    wins += info["success"]
    print(f"{info['object']:6s} {1000*info['object_width_m']:5.0f}  {tp:6d}   {f0:8.1f}                     {maxlift:8.0f}     {str(not info['pinched']):5s}   "
          f"{1000*info['stage_drift_m']:7.1f}   {info['peak_grip_force_n']:8.1f}   {info['success']}")
print(f"scripted lift after the policy's own pinch: {wins}/{N} succeed")
