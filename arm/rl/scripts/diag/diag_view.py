"""Is the target actually inside the wrist camera's view while the policies run? Plus a wrist-cam frame."""
import glob, os, re
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ["YAM_HANDOFF_REUSE"] = "0"
import cv2, numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from hackmit_rl.config import load_config
from hackmit_rl.envs import OpenYAMFeedEnv
cfg = load_config("arm/rl/configs/feed.yaml"); cfg["env"]["stage"] = "grasp"
raw = DummyVecEnv([lambda: OpenYAMFeedEnv(cfg)]); e = raw.envs[0]
cks = [c for c in glob.glob("runs/feed-grasp/checkpoints/ppo_grasp_*_steps.zip") if os.path.getsize(c) > 0]
CK = max(cks, key=lambda p: int(re.search(r"_(\d+)_steps", p).group(1))); step = re.search(r"_(\d+)_steps", CK).group(1)
env = VecNormalize.load(glob.glob(f"runs/feed-grasp/checkpoints/*vecnormalize_{step}_steps.pkl")[0], raw)
env.training, env.norm_reward = False, False
m = PPO.load(CK, device="cpu")
reach_seen, reach_first = [], []
def hook(en):
    reach_seen.append(bool(en.seen.get("object_jitter_m", False)))
e.prior_hook = hook
grasp_seen, mouth_seen, at_start = [], [], []
frame = None
for ep in range(12):
    n0 = len(reach_seen); obs = env.reset()
    reach_first.append(reach_seen[n0] if len(reach_seen) > n0 else None)
    at_start.append(bool(e.seen.get("object_jitter_m", False)))
    if frame is None:
        frame = cv2.cvtColor(e.scene.render("wrist_cam", 256), cv2.COLOR_RGB2BGR)
    for i in range(300):
        a, _ = m.predict(obs, deterministic=True)
        obs, r, done, infos = env.step(a)
        if done[0]: break
        grasp_seen.append(infos[0]["object_in_view"]); mouth_seen.append(infos[0]["mouth_in_view"])
print(f"object in wrist view: at the very start of reach {np.mean([x for x in reach_first if x is not None]):.0%} of episodes, "
      f"during reach {np.mean(reach_seen):.0%} of steps, at hand-off {np.mean(at_start):.0%}, during the grab {np.mean(grasp_seen):.0%} of steps")
print(f"mouth in wrist view during the grab: {np.mean(mouth_seen):.0%} of steps")
cv2.imwrite("/tmp/wrist.jpg", cv2.resize(frame, (512, 512), interpolation=cv2.INTER_NEAREST))
