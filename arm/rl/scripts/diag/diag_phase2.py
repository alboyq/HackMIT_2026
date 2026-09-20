"""Penetration, reach hand-off and speed as a function of physics substeps (env var SUB)."""
import os
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from hackmit_rl.config import load_config
from hackmit_rl.envs import OpenYAMFeedEnv

cfg = load_config("arm/rl/configs/feed.yaml")
cfg["env"]["stage"] = "grasp"
cfg["env"]["physics_substeps"] = int(os.environ.get("SUB", cfg["env"]["physics_substeps"]))
env = OpenYAMFeedEnv(cfg)
d = env.data
print("substeps", cfg["env"]["physics_substeps"], "timestep %.2f ms" % (1000 * env.model.opt.timestep))

t0, n_steps, oks = time.time(), 0, []
pads, arms, pins, forces = [], [], 0, []
for ep in range(20):
    _, rinfo = env.reset(seed=100 + ep)
    oks.append(rinfo["handoff_ok"])
    n_steps += rinfo["handoff_steps"]
    gid = env.obj_gid[env.name]
    worst_pad = worst_arm = 0.0
    pin = False
    act = np.zeros(7, dtype=np.float32)
    act[6] = -1.0
    for i in range(60):
        _, _, term, trunc, info = env.step(act)
        n_steps += 1
        pin |= info["pinched"]
        for c in range(d.ncon):
            con = d.contact[c]
            pair = {int(con.geom1), int(con.geom2)}
            if gid in pair and pair & env.arm_geoms:
                pen = -1000 * float(con.dist)
                worst_arm = max(worst_arm, pen)
                if pair & env.pad_geoms:
                    worst_pad = max(worst_pad, pen)
        if term or trunc:
            break
    pads.append(worst_pad); arms.append(worst_arm); pins += pin
    forces.append(info["peak_grip_force_n"])
print(f"reach hand-off ok {sum(oks)}/{len(oks)}   pinched {pins}/20")
print(f"penetration, pads   : mean {np.mean(pads):.2f} mm   worst {np.max(pads):.2f} mm")
print(f"penetration, any arm: mean {np.mean(arms):.2f} mm   worst {np.max(arms):.2f} mm")
print(f"peak grip force     : worst {np.max(forces):.1f} N")
print(f"{n_steps / (time.time() - t0):.0f} env steps/s, one process")
