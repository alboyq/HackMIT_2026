"""Overlay the sim arm (at a given joint pose) on a real scene-camera frame for a candidate camera pose.
Usage: match_camera.py REAL.jpg OUT.jpg  px py pz  tx ty tz  hfov_deg [roll_deg]     (arm at qpos0)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2, mujoco, numpy as np
from yam_data import lookat_quat
from yam_scene import YamScene

real = cv2.imread(sys.argv[1]); H, W = real.shape[:2]
px, py, pz, tx, ty, tz, hfov = (float(v) for v in sys.argv[3:10])
roll = float(sys.argv[10]) if len(sys.argv) > 10 else 0.0
s = YamScene(); m, d = s.model, s.data
d.qpos[:] = m.qpos0; mujoco.mj_forward(m, d)
cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "scene_cam")
m.cam_pos[cid] = np.array([px, py, pz]) - m.body_pos[m.cam_bodyid[cid]]
m.cam_quat[cid] = lookat_quat([px, py, pz], [tx, ty, tz], np.radians(roll))
m.cam_fovy[cid] = np.degrees(2 * np.arctan(np.tan(np.radians(hfov) / 2) * H / W))
mujoco.mj_forward(m, d)
rw, rh = 960, 540
m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, rw), max(m.vis.global_.offheight, rh)
seg = mujoco.Renderer(m, rh, rw); seg.enable_segmentation_rendering(); seg.update_scene(d, camera="scene_cam")
ids = seg.render()[:, :, 0]
arm_root = s._bid("arm"); arm = set()
for b in range(m.nbody):
    x = b
    while x != 0 and x != arm_root: x = m.body_parentid[x]
    if x == arm_root: arm.add(b)
mask = np.isin(ids, [g for g in range(m.ngeom) if m.geom_bodyid[g] in arm]).astype(np.uint8)
mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
out = real.copy()
out[mask > 0] = (0.55 * out[mask > 0] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
cnt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(out, cnt, -1, (0, 255, 255), 3)
cv2.imwrite(sys.argv[2], cv2.resize(out, (1280, 720)))
print("vfov", round(float(m.cam_fovy[cid]), 1))
