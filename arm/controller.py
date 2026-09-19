from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol


class Controller(Protocol):
    def pick(self, target_name: str): ...
    def deliver(self, mouth_xyz): ...


class IKController:
    def __init__(self, scene):
        from arm.ik.planner import Expert
        self.scene = scene
        self.expert = Expert(scene)

    def pick(self, target_name: str):
        return self.expert.plan_pick(target_name)

    def deliver(self, mouth_xyz):
        return self.expert.plan_present(mouth_xyz, q_from=self.scene.q_arm,
                                        grip=float(self.scene.data.ctrl[6]))


class RLController:
    """Lazy adapter: importing the IK/demo path never imports torch or SB3."""

    def __init__(self, model_path: str, vecnormalize_path: str):
        from stable_baselines3 import PPO
        self.model = PPO.load(model_path, device="cpu")
        self.vecnormalize_path = vecnormalize_path

    def pick(self, observation):
        return self.model.predict(observation, deterministic=True)[0]

    def deliver(self, mouth_xyz):
        raise RuntimeError("mouth delivery is scripted and never delegated to RL")


def select_controller(name: str, scene, report_path: str | Path = "reports/evaluation.json",
                      model_path: str | None = None, vecnormalize_path: str | None = None):
    name = name.lower()
    if name == "ik":
        return IKController(scene)
    if name == "rl":
        if not model_path or not vecnormalize_path:
            raise ValueError("RL requires model and VecNormalize paths")
        return RLController(model_path, vecnormalize_path)
    if name != "auto":
        raise ValueError("controller must be ik, rl, or auto")
    try:
        report = json.loads(Path(report_path).read_text())
        safe = report["rl"]["safety_violations"] == 0
        strong = report["rl"]["success_rate"] >= report["ik"]["success_rate"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        safe = strong = False
    if safe and strong and model_path and vecnormalize_path:
        return RLController(model_path, vecnormalize_path)
    return IKController(scene)

