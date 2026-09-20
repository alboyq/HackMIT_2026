"""How is the gripper oriented when reach hands over? Tilt from straight-down, and jaw yaw vs the block's faces."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
from hackmit_rl.config import load_config
from hackmit_rl.envs import OpenYAMFeedEnv
cfg = load_config("arm/rl/configs/feed.yaml"); cfg["env"]["stage"] = "grasp"
env = OpenYAMFeedEnv(cfg); d = env.data; tool = env.tool
print("tcp_local (mm):", np.round(1000*tool.tcp_local, 1), " tool_local", np.round(tool.tool_local, 2), " jaw_local", np.round(tool.jaw_local, 2))
rows = []
for ep in range(30):
    _, info = env.reset(seed=200 + ep)
    R = d.site_xmat[tool.site_id].reshape(3, 3)
    t, j = R @ tool.tool_local, R @ tool.jaw_local
    tilt = np.degrees(np.arccos(np.clip(-t[2], -1, 1)))
    yaw = np.degrees(np.arctan2(j[1], j[0]))
    off_square = abs(((yaw + 45) % 90) - 45)              # 0 = squared to a box face, 45 = worst
    tip = env._tcp() + t * (0.086 - 0.0497)               # roughly the fingertip
    rows.append((env.name, tilt, off_square, 1000*env.width[env.name], 1000*(tip[2]), info["handoff_ok"],
                 1000*float(np.linalg.norm(env.scene.object_pos(env.name)[:2]-env.object_start[:2]))))
for name in ("apple", "mug", "block"):
    r = [x for x in rows if x[0] == name]
    print(f"{name:6s} n={len(r):2d}  tilt from straight-down: mean {np.mean([x[1] for x in r]):5.1f} deg (max {np.max([x[1] for x in r]):4.1f})   "
          f"jaws off-square: mean {np.mean([x[2] for x in r]):4.1f} deg   tip height {np.mean([x[4] for x in r]):5.1f} mm   "
          f"knocked by reach: mean {np.mean([x[6] for x in r]):4.1f} mm (max {np.max([x[6] for x in r]):4.1f})")
print("hand-off ok:", sum(x[5] for x in rows), "/", len(rows))
blk = [x for x in rows if x[0] == "block"]
print("block: diagonal vs opening ->", [f"{1.414*x[3]:.0f}mm diag, {x[2]:.0f}deg off" for x in blk][:8], " opening", round(1000*tool.max_width,1))
