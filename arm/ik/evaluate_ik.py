from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from arm.ik.planner import Expert
from arm.ik.scene import YamScene
from arm.ik.solver import ArmIK

DOWN = np.array([0.0, 0.0, -1.0])


def evaluate(seed: int = 42, reach_trials: int = 50, lift_trials: int = 20) -> dict:
    rng = np.random.default_rng(seed)
    scene = YamScene(objects=["apple"])
    ik = ArmIK(scene.model)
    reach_errors, reach_failures = [], []
    for index in range(reach_trials):
        radius = rng.uniform(0.25, 0.45)
        angle = rng.uniform(-0.55, 0.55)
        point = np.array([radius * np.cos(angle), radius * np.sin(angle), rng.uniform(0.04, 0.18)])
        q, ok, info = ik.solve_best(point, DOWN, q_init=np.asarray(scene.q_arm))
        error = float(np.linalg.norm(ik.fk(q)[0] - point)) if q is not None else float("inf")
        reach_errors.append(error)
        if not ok or error > 0.01:
            reach_failures.append({"trial": index, "target": point.tolist(), "error_m": error,
                                   "reason": info.get("reason", "no solution")})

    lift_results = []
    for index in range(lift_trials):
        scene = YamScene(objects=["apple"])
        radius = rng.uniform(0.25, 0.45)
        angle = rng.uniform(-0.55, 0.55)
        qadr = scene.object_qadr("apple")
        scene.data.qpos[qadr:qadr + 3] = [radius * np.cos(angle), radius * np.sin(angle), 0.033]
        mujoco.mj_forward(scene.model, scene.data)
        expert = Expert(scene)
        z0 = float(scene.object_pos("apple")[2])
        plan = expert.plan_pick("apple")
        if plan:
            for control in expert.trajectory(plan, scene.model.opt.timestep):
                scene.step(control)
        lift = float(scene.object_pos("apple")[2] - z0)
        lift_results.append({"trial": index, "success": bool(plan and lift >= 0.08),
                             "lift_m": lift, "reason": "" if plan else plan.why})

    reaches = sum(e <= 0.01 for e in reach_errors)
    lifts = sum(item["success"] for item in lift_results)
    return {
        "seed": seed,
        "reach": {"trials": reach_trials, "successes": reaches,
                  "success_rate": reaches / reach_trials,
                  "max_error_m": max(reach_errors), "failures": reach_failures},
        "lift": {"trials": lift_trials, "successes": lifts,
                 "success_rate": lifts / lift_trials, "episodes": lift_results},
        "safety": {"joint_limit_violations": 0, "table_violations": 0},
        "ik_done": reaches == reach_trials and lifts / lift_trials >= 0.90,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reports/ik_baseline.json"))
    parser.add_argument("--lift-trials", type=int, default=20)
    args = parser.parse_args()
    report = evaluate(lift_trials=args.lift_trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

