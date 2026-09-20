"""What happens between the first seated pinch and the end of the episode?"""
import glob, os, re, sys
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
m = PPO.load(CK, device="cpu")
print("obj    t_pinch t_end  drift@pinch drift@end  tcp_dz_after  obj_dz  tilt_obj  grip@end  force  pinched@end  ret_after")
for ep in range(N):
    obs = env.reset(); tp = None; info = {}; ret = 0.0
    for i in range(300):
        a, _ = m.predict(obs, deterministic=True)
        prev = dict(info)
        obs, r, done, infos = env.step(a); info = infos[0]
        if tp is not None: ret += float(r[0])
        if tp is None and info["pinched"]:
            tp = i; d0 = info["stage_drift_m"]; z0 = e._tcp()[2] if not done[0] else 0; oz0 = info["object_height"]
        if done[0]: break
        last_tcp_z = e._tcp()[2]; last = dict(info)
    if tp is None:
        print(f"{e.name if False else info['object']:6s}   never pinched, end t={i} failed={info.get('failed')} drift {1000*info['stage_drift_m']:.0f} mm"); continue
    print(f"{info['object']:6s} {tp:6d} {i:5d}   {1000*d0:8.1f}  {1000*info['stage_drift_m']:8.1f}   {1000*(last_tcp_z - z0):9.1f}  {1000*(info['object_height']-oz0):6.1f}   "
          f"   -    {info['grip_fraction']:6.2f}  {info['grip_force_n']:5.1f}   {str(info['pinched']):5s}      {ret:7.1f}")
