"""After moving the jaws: any new self-collision at home, open or closed? Does the gripper still pinch?"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
from hackmit_rl.config import load_config
from hackmit_rl.envs import OpenYAMFeedEnv
cfg = load_config("arm/rl/configs/feed.yaml"); cfg["env"]["stage"] = "reach"
env = OpenYAMFeedEnv(cfg); m, d = env.model, env.data
print("max opening (mm):", round(1000*env.tool.max_width, 1), " tcp_local (mm):", np.round(1000*env.tool.tcp_local, 1))
for label, cmd in (("open", 1.0), ("closed", -1.0)):
    env.reset(seed=2); hits = 0; pairs = set()
    a = np.zeros(7, dtype=np.float32); a[6] = cmd
    for _ in range(40):
        env.step(a)
        _, _, _, sh = env._contacts(); hits += sh
        for i in range(d.ncon):
            g1, g2 = int(d.contact[i].geom1), int(d.contact[i].geom2)
            if g1 in env.arm_geoms and g2 in env.arm_geoms:
                pairs.add((m.body(int(m.geom_bodyid[g1])).name, m.body(int(m.geom_bodyid[g2])).name))
    print(f"{label:6s}: penalised self-hits over 40 steps = {hits}   arm-arm contact pairs: {sorted(pairs)}")
