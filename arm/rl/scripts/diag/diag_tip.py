"""Where is the sim fingertip relative to the wrist roll axis (joint6)? 'In the middle' = on it."""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from hackmit_rl.config import load_config
from hackmit_rl.envs import OpenYAMFeedEnv

cfg = load_config("arm/rl/configs/feed.yaml")
cfg["env"]["stage"] = "reach"
env = OpenYAMFeedEnv(cfg)
m, d = env.model, env.data
env.reset(seed=1)
act = np.zeros(7, dtype=np.float32); act[6] = -1
for _ in range(40):
    env.step(act)

j6 = m.joint("joint6").id
axis = d.xaxis[j6] / np.linalg.norm(d.xaxis[j6])
anchor = d.xanchor[j6]


def off_axis(p):
    v = p - anchor
    along = float(v @ axis)
    perp = v - along * axis
    return along, perp


tcp = env._tcp()
along, perp = off_axis(tcp)
print(f"TCP (between the pads): {1000*along:.1f} mm along the roll axis from joint6, "
      f"{1000*np.linalg.norm(perp):.1f} mm OFF the axis")
# express the off-axis part in the link_6 frame so the direction is meaningful
R6 = d.xmat[m.body("link_6").id].reshape(3, 3)
print("  off-axis vector in link_6 frame (mm):", np.round(1000 * (R6.T @ perp), 1))
print("  roll axis in link_6 frame:", np.round(R6.T @ axis, 3))

# extent of every finger/pad collision geom along and off the axis
lo, hi, worst = 1e9, -1e9, 0.0
for g in sorted(env.finger_geoms | env.pad_geoms):
    a, p = off_axis(d.geom_xpos[g])
    lo, hi = min(lo, a), max(hi, a)
print(f"finger collision geoms span {1000*lo:.0f} .. {1000*hi:.0f} mm along the axis")
for name in ("link_left_finger", "link_right_finger", "lf_down", "rf_down", "link_6"):
    b = m.body(name).id
    a, p = off_axis(d.xpos[b])
    print(f"  body {name:18s} origin: {1000*a:6.1f} mm along, {1000*np.linalg.norm(p):5.1f} mm off-axis, "
          f"link_6-frame {np.round(1000*(R6.T@p),1)}")
# visual finger mesh bounding boxes
for g in range(m.ngeom):
    if m.geom_type[g] == 7 and m.body(int(m.geom_bodyid[g])).name.startswith("link_") and "finger" in m.body(int(m.geom_bodyid[g])).name:
        print(f"  visual mesh g{g} on {m.body(int(m.geom_bodyid[g])).name}: aabb half-size (mm) "
              f"{np.round(1000*m.geom_aabb[g][3:],1)} centre {np.round(1000*m.geom_aabb[g][:3],1)}")
