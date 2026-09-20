"""Phase-0 demo: pick an object and present it at the user's mouth, filmstrip out.

  urlab_bridge/.venv/bin/python rl/yam/demo_pick_present.py apple
  OBJ=mug urlab_bridge/.venv/bin/python rl/yam/demo_pick_present.py

Writes a two-row filmstrip (scene camera on top, wrist camera below) so the wrist view can be
eyeballed for what a policy trained on it would actually see.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import imageio.v3 as iio
import numpy as np

from yam_expert import Expert
from yam_scene import YamScene

OUT = Path(os.environ.get("YAM_OUT", Path(__file__).resolve().parent / "media"))
N_FRAMES = int(os.environ.get("YAM_FRAMES", "7"))
RES = int(os.environ.get("YAM_RES", "256"))


def run(name="apple"):
    s = YamScene(res=RES)
    e = Expert(s)
    dt = s.model.opt.timestep

    pick = e.plan_pick(name)
    if not pick:
        print(f"pick failed: {pick.why}")
        return 1
    traj = list(e.trajectory(pick, dt))
    z0 = s.object_pos(name)[2]
    frames = []

    def snap():
        frames.append((s.render("scene_cam", RES), s.render("wrist_cam", RES)))

    marks = set(np.linspace(0, len(traj) - 1, N_FRAMES - 3).astype(int))
    for i, c in enumerate(traj):
        s.step(c, 1)
        if i in marks:
            snap()
    lifted = s.object_pos(name)[2] - z0
    print(f"{name}: strategy={pick.strategy} lifted {lifted * 1000:.0f} mm "
          f"in {len(traj) * dt:.1f} s")

    mouth = s.site("mouth")
    pres = e.plan_present(mouth, q_from=s.q_arm, grip=float(s.data.ctrl[6]))
    if not pres:
        print(f"present failed: {pres.why}")
        return 1
    traj2 = list(e.trajectory(pres, dt))
    marks2 = set(np.linspace(0, len(traj2) - 1, 3).astype(int))
    for i, c in enumerate(traj2):
        s.step(c, 1)
        if i in marks2:
            snap()

    tcp = e.ik.fk(s.q_arm, qpos_full=s.data.qpos)[0]
    held = s.object_pos(name)
    print(f"presented: TCP {np.round(tcp, 3)}, object {np.round(held, 3)}, "
          f"gap to mouth {np.linalg.norm(tcp - mouth) * 100:.1f} cm, "
          f"object still held: {held[2] > 0.10}")

    top = np.concatenate([f[0] for f in frames], axis=1)
    bot = np.concatenate([f[1] for f in frames], axis=1)
    strip = np.concatenate([top, bot], axis=0)
    out = OUT / f"yam_{name}_to_mouth.png"
    iio.imwrite(out, strip)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("OBJ", "apple")))
