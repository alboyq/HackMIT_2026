"""The sim gripper with NOTHING in it, closed and open, to compare against photos of the real one."""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np

from hackmit_rl.config import load_config
from hackmit_rl.envs import OpenYAMFeedEnv

cfg = load_config("arm/rl/configs/feed.yaml")
cfg["env"]["stage"] = "reach"
env = OpenYAMFeedEnv(cfg)
m, d = env.model, env.data
env.reset(seed=1)
r = mujoco.Renderer(m, 480, 640)
site = env.tool.site_id


def shot(groups, az, el, dist=0.30):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = 0.5 * (env._tcp() + d.xpos[m.body("link_6").id])
    cam.distance, cam.azimuth, cam.elevation = dist, az, el
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    for g in groups:
        opt.geomgroup[g] = 1
    r.update_scene(d, camera=cam, scene_option=opt)
    return cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)


def tag(img, text):
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    return img


def pad_report(label):
    R = d.site_xmat[site].reshape(3, 3)
    o = d.site_xpos[site]
    L = np.array([R.T @ (d.geom_xpos[g] - o) for g in sorted(env.left_pads)])
    Rr = np.array([R.T @ (d.geom_xpos[g] - o) for g in sorted(env.right_pads)])
    tcp_l = env.tool.tcp_local
    print(f"\n{label}: ctrl {d.ctrl[env.grip_aid]:.4f}")
    print("  tool-frame TCP offset:", np.round(1000 * tcp_l, 1), "mm")
    print("  left pads  (mm, tool frame):\n", np.round(1000 * L, 1))
    print("  right pads (mm, tool frame):\n", np.round(1000 * Rr, 1))
    gap = np.linalg.norm(L.mean(0) - Rr.mean(0))
    mid = 0.5 * (L.mean(0) + Rr.mean(0))
    print(f"  pad-centre gap {1000*gap:.1f} mm   midpoint between pads {np.round(1000*mid,1)} mm "
          f"(0,0 in the closing axis = 'in the middle')")


rows = []
for label, cmd in (("CLOSED (empty)", -1.0), ("OPEN", 1.0)):
    act = np.zeros(7, dtype=np.float32)
    act[6] = cmd
    for _ in range(40):
        env.step(act)
    pad_report(label)
    rows.append(np.hstack([tag(shot([0, 2], 0, -5), f"{label} - drawn, view A"),
                           tag(shot([0, 3], 0, -5), f"{label} - collision, view A")]))
    rows.append(np.hstack([tag(shot([0, 2], 90, -5), f"{label} - drawn, view B"),
                           tag(shot([0, 3], 90, -5), f"{label} - collision, view B")]))
cv2.imwrite("/tmp/gripper2.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 86])

print("\nfinger joints:")
for j in range(m.njnt):
    name = m.joint(j).name
    if "finger" in name.lower() or "grip" in name.lower() or name.startswith(("lf", "rf", "joint7", "joint8")):
        print(f"  {name}: type {m.jnt_type[j]} axis {m.jnt_axis[j]} range {m.jnt_range[j]}")
print("equality constraints:", m.neq, " tendons:", m.ntendon)
print("wrote /tmp/gripper2.jpg")
