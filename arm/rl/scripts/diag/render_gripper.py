"""Close-up of the sim gripper closed on an object: what is DRAWN vs what COLLIDES."""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np

from hackmit_rl.config import load_config
from hackmit_rl.envs import OpenYAMFeedEnv

cfg = load_config("arm/rl/configs/feed.yaml")
cfg["env"]["stage"] = "grasp"
cfg["env"]["physics_substeps"] = 8
env = OpenYAMFeedEnv(cfg)
m, d = env.model, env.data

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 111
env.reset(seed=seed)
act = np.zeros(7, dtype=np.float32)
frames_open = None
r = mujoco.Renderer(m, 480, 640)


def shot(group_on, azimuth, elevation, dist=0.26):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = env._tcp()
    cam.distance, cam.azimuth, cam.elevation = dist, azimuth, elevation
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    for g in group_on:
        opt.geomgroup[g] = 1
    r.update_scene(d, camera=cam, scene_option=opt)
    return cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)


def tag(img, text):
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    return img


obj_groups = sorted({int(m.geom_group[env.obj_gid[n]]) for n in ("apple", "mug", "block")}
                    | {int(m.geom_group[env.table_gid])})
print("object/table geom groups:", obj_groups, " target:", env.name)
VIS, COL = obj_groups + [2], obj_groups + [3]
row_open = [tag(shot(VIS, 90, -15), f"OPEN at hand-off - drawn mesh ({env.name})"),
            tag(shot(COL, 90, -15), "OPEN - collision shapes only")]
act[6] = -1.0
pin = False
for i in range(45):
    _, _, term, trunc, info = env.step(act)
    pin |= info["pinched"]
    if term or trunc:
        break
print("pinched:", pin, "grip force", round(info["grip_force_n"], 1))
row_closed = [tag(shot(VIS, 90, -15), f"CLOSED - drawn mesh  pinched={pin}"),
              tag(shot(COL, 90, -15), "CLOSED - collision shapes only")]
row_side = [tag(shot(VIS, 0, -10), "CLOSED, side - drawn mesh"),
            tag(shot(COL, 0, -10), "CLOSED, side - collision shapes")]
cv2.imwrite("/tmp/gripper.jpg", np.vstack([np.hstack(row_open), np.hstack(row_closed), np.hstack(row_side)]),
            [cv2.IMWRITE_JPEG_QUALITY, 88])
print("wrote /tmp/gripper.jpg")
