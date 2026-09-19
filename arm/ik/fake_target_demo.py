from __future__ import annotations

import time

import mujoco
import numpy as np

from arm.controller import IKController
from arm.ik.scene import YamScene
from arm.perception_io import Target3D, TargetFilter, encode_target


def run_fake_target(xyz=(0.34, 0.08, 0.033)) -> dict:
    scene = YamScene(objects=["apple"])
    target = Target3D(np.asarray(xyz, float), "base", "external", 0.99, time.time())
    payload = encode_target(target)
    decoded = Target3D.from_json(payload)
    filt = TargetFilter()
    if not filt.consider(decoded, np.asarray(scene.site("grasp_site"))):
        return {"ok": False, "reason": filt.rejections[-1]}
    qadr = scene.object_qadr("apple")
    scene.data.qpos[qadr:qadr + 3] = decoded.xyz
    mujoco.mj_forward(scene.model, scene.data)
    z0 = float(scene.object_pos("apple")[2])
    controller = IKController(scene)
    plan = controller.pick("apple")
    if not plan:
        return {"ok": False, "reason": plan.why}
    for command in controller.expert.trajectory(plan, scene.model.opt.timestep):
        scene.step(command)
    lift = float(scene.object_pos("apple")[2] - z0)
    return {"ok": lift >= 0.08, "lift_m": lift, "source": decoded.source}


if __name__ == "__main__":
    print(run_fake_target())

