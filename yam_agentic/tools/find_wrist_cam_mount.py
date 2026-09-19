"""Find an unoccluded mount for the wrist camera by ray casting, instead of guessing.

    python tools/find_wrist_cam_mount.py

Two hand-picked mounts in a row ended up buried inside the gripper housing (the rendered view
was a wall of dark plastic). This grids candidate positions around the tool axis, casts rays
from each to the TCP and to points 5 and 12 cm in front of it, and keeps the mounts where
nothing on the gripper blocks the line of sight. It prints pos/xyaxes in the link_6 frame,
ready to paste into `_patched_arm()` in sim/yam_scene.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim"))

import mujoco
import numpy as np

from yam_ik import ToolFrame, look_at_xyaxes
from yam_scene import YamScene

BACKS = (0.06, 0.09, 0.12, 0.15)          # m behind the TCP along the tool axis
UPS = (-0.09, -0.06, -0.03, 0.0, 0.03, 0.06, 0.09)   # m along "back of the hand"
SIDES = (-0.05, 0.0, 0.05)                # m along the jaw axis


def main():
    s = YamScene()
    m = s.model
    tf = ToolFrame(m)
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    d.qpos[m.jnt_qposadr[tf._jid]] = 0.02              # jaws half open
    mujoco.mj_forward(m, d)

    p6 = d.xpos[tf.body_id].copy()
    R6 = d.xmat[tf.body_id].reshape(3, 3).copy()
    R = d.site_xmat[tf.site_id].reshape(3, 3)
    tcp = d.site_xpos[tf.site_id] + R @ tf.tcp_local
    tool, jaw = R @ tf.tool_local, R @ tf.jaw_local
    back_of_hand = np.cross(tool, jaw)

    targets = [tcp, tcp + 0.05 * tool, tcp + 0.12 * tool]
    gid = np.zeros(1, dtype=np.int32)
    clear = []
    for b in BACKS:
        for u in UPS:
            for sd in SIDES:
                cam = tcp - b * tool + u * back_of_hand + sd * jaw
                if all(not (0 < mujoco.mj_ray(m, d, cam, (t - cam) / np.linalg.norm(t - cam),
                                              None, 1, -1, gid) < np.linalg.norm(t - cam) - 1e-3)
                       for t in targets):
                    clear.append((b, u, sd, cam))

    print(f"{len(clear)} of {len(BACKS) * len(UPS) * len(SIDES)} candidate mounts have clear "
          f"line of sight to the TCP and 5/12 cm beyond it\n")
    tool_l = R6.T @ tool
    aim_l = R6.T @ (tcp + 0.03 * tool - p6)
    for b, u, sd, cam in clear:
        cl = R6.T @ (cam - p6)
        print(f'  back={b:.2f} up={u:+.2f} side={sd:+.2f}  '
              f'pos="{cl[0]:.4f} {cl[1]:.4f} {cl[2]:.4f}" '
              f'xyaxes="{look_at_xyaxes(cl, aim_l, up=tool_l)}"')
    print("\nsim/yam_scene.py currently uses back=0.06 up=+0.06 side=0.00.")


if __name__ == "__main__":
    main()
