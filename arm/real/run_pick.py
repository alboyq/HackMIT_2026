"""Pick up a named food with the trained policies, on any ArmInterface-like body.

    learned reach -> learned place+pinch -> scripted vertical lift -> hold -> set it back down, open.   (no feeding)

How the policies are fed EXACTLY what they were trained on: the training environment itself is kept as a BRAIN with
no physics. Every tick the body's measured joint angles/speeds/jaw opening are written into it, the food is put where
the wrist camera says it is, the environment's own observation function is called, the policy answers, and the
environment's own action->joint-target arithmetic (low-pass, 0.02 rad step, speed cap) turns that into the targets
that are sent to the body. Nothing about the observation is re-implemented here, so nothing can drift from training.

    --body sim    a second, separate physics simulation plays the arm (its true food position + camera-like noise is the
                  'camera'). THIS is how the loop is tested. Nothing real is touched.
    --body real   RealArm.connect() (a person at the arm types MOVE) + the wrist camera. NEVER RUN YET.
  PYTHONPATH=$PWD:$PWD/arm/rl .venv-arm/bin/python arm/real/run_pick.py --food grape --body sim --episodes 20
"""
from __future__ import annotations

import argparse
import copy
import glob
import os
import pickle
import re
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["YAM_HANDOFF_REUSE"] = "0"
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "arm" / "rl"), str(REPO / "arm" / "rl" / "scripts"), str(Path(__file__).resolve().parent)]

from hackmit_rl.config import load_config  # noqa: E402
from hackmit_rl.envs import OpenYAMFeedEnv  # noqa: E402

# food name -> (grasp class the policies know, typical width in metres). Width is re-measured by the camera when it can.
FOODS = {"grape": ("apple", 0.022), "strawberry": ("apple", 0.030), "can": ("mug", 0.066), "tape measure": ("apple", 0.065),
         "apple": ("apple", 0.070), "cracker": ("block", 0.045)}
LIFT_TO, STEP_UP, K_XY = 0.10, 0.002, 0.15


def make_env(slot, width, seed=0):
    cfg = load_config(str(REPO / "arm/rl/configs/feed.yaml"))
    cfg = copy.deepcopy(cfg)
    cfg["env"]["stage"] = "grasp"
    cfg["env"]["handoff"] = dict(cfg["env"].get("handoff") or {}, enabled=False)
    cfg["env"]["small_object_prob"] = 0.0
    cfg.setdefault("perception_noise", {})["enabled"] = False
    e = OpenYAMFeedEnv(cfg)
    for s in range(seed, seed + 200):                       # the env draws the object at reset: redraw until it is the class we want
        e.reset(seed=s)
        if e.name == slot:
            break
    return e, cfg


def load_policy(zip_path, stats_path):
    from stable_baselines3 import PPO
    with open(stats_path, "rb") as fh:
        vn = pickle.load(fh)
    return PPO.load(zip_path, device="cpu").policy, vn.obs_rms.mean.copy(), np.sqrt(vn.obs_rms.var + vn.epsilon), float(vn.clip_obs)


class Brain:
    """The training environment with physics switched off, driven by measurements."""
    def __init__(self, slot, width_m, run=REPO / "arm/rl/models/grasp_v2", speed=1.0):
        self.speed = float(speed)      # < 1 = every commanded joint step is scaled down: same path, slower (first runs on the real arm)
        self.e, self.cfg = make_env(slot, width_m)
        e = self.e
        e.width[e.name] = float(width_m)
        e.rest_z[e.name] = 0.5 * float(width_m)             # a fruit sits on the table: its centre is half its height up
        e.reach_offset = np.array([0.0, 0.0, e.rest_z[e.name] + e.tip_ahead + float(e.ecfg["reach_clearance_m"])])
        ck = max((c for c in glob.glob(f"{run}/checkpoints/ppo_grasp_*_steps.zip") if os.path.getsize(c) > 0),
                 key=lambda p: int(re.search(r"_(\d+)_steps", p).group(1)))
        step = re.search(r"_(\d+)_steps", ck).group(1)
        self.policies = {"reach": load_policy(f"{run}/ppo_reach_final.zip", f"{run}/vecnormalize.pkl"),
                         "grasp": load_policy(ck, glob.glob(f"{run}/checkpoints/*vecnormalize_{step}_steps.pkl")[0])}
        self.obj = np.asarray(e.ecfg["nominal_object"], float)   # until the camera has seen it
        self.mouth = np.asarray(e.ecfg["nominal_mouth"], float)
        e._perceived = lambda p, b, k: (self.obj if k == "object_jitter_m" else self.mouth)
        self.lf, self.rf = (e.model.jnt_qposadr[e.model.joint(n).id] for n in ("left_finger", "right_finger"))
        self.oadr = e.scene.object_qadr(e.name)
        self.jacp, self.jacr = np.zeros((3, e.model.nv)), np.zeros((3, e.model.nv))
        self.prev_tcp, self.tcp_speed = None, 0.0

    def sync(self, q, qd, grip_m, obj_est):
        e = self.e
        if obj_est is not None:
            self.obj = np.asarray(obj_est, float)
        e.data.qpos[e.qadr], e.data.qvel[e.dadr] = q, qd
        e.data.qpos[self.lf], e.data.qpos[self.rf] = grip_m, -grip_m
        e.data.qpos[self.oadr:self.oadr + 3] = self.obj
        mujoco.mj_forward(e.model, e.data)
        tcp = e._tcp()
        self.tcp_speed = 0.0 if self.prev_tcp is None else float(np.linalg.norm(tcp - self.prev_tcp) / e.dt)
        self.prev_tcp = tcp.copy()
        return tcp

    def tilt_deg(self):
        e = self.e
        axis = e.data.site_xmat[e.tool.site_id].reshape(3, 3) @ e.tool.tool_local
        return float(np.degrees(np.arccos(np.clip(-axis[2], -1.0, 1.0))))

    def apply(self, raw):
        """The environment's own action -> target arithmetic (openyam_feed.step), minus the physics."""
        e = self.e
        raw = np.clip(np.asarray(raw, float), -1.0, 1.0)
        alpha = float(e.ecfg["action_lowpass"])
        e.filtered = (1 - alpha) * e.filtered + alpha * raw
        size = 1.0
        if e.stage != "reach":
            size = float(np.clip(e.width[e.name] / float(e.ecfg.get("full_size_width_m", 0.05)), float(e.ecfg.get("small_min_factor", 1.0)), 1.0))
        size *= self.speed
        max_delta = float(e.ecfg["max_joint_velocity_rad_s"]) * e.dt * size
        want = e.data.ctrl[:6] + np.clip(float(e.ecfg["action_delta_rad"]) * size * e.filtered[:6], -max_delta, max_delta)
        e.data.ctrl[:6] = np.clip(want, e.ctrl_lo, e.ctrl_hi)
        e.data.ctrl[e.grip_aid] = e.grip_range[0] + 0.5 * (e.filtered[6] + 1) * np.ptp(e.grip_range)
        e.prev_action = raw.copy()
        return e.data.ctrl[:6].copy(), float(e.data.ctrl[e.grip_aid])

    def policy_step(self, stage):
        e = self.e
        e.stage = stage
        e.target = self.obj + e.reach_offset if stage == "reach" else self.obj.copy()
        obs = e._observation()
        pol, mean, std, clip = self.policies[stage]
        a, _ = pol.predict(np.clip((obs - mean) / std, -clip, clip)[None], deterministic=True)
        return obs, self.apply(a[0])

    def lift_step(self, anchor, direction=+1.0):
        e = self.e
        mujoco.mj_jacSite(e.model, e.data, self.jacp, self.jacr, e.tool.site_id)
        J = np.vstack([self.jacp[:, e.dadr], self.jacr[:, e.dadr]])
        tcp = e._tcp()
        dxy = np.clip(K_XY * (anchor[:2] - tcp[:2]), -0.003, 0.003)
        twist = np.array([dxy[0], dxy[1], direction * STEP_UP, 0, 0, 0]) * min(1.0, 2 * self.speed)
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), twist)
        a = np.zeros(7); a[:6] = np.clip(dq / float(e.ecfg["action_delta_rad"]), -1, 1); a[6] = -1.0
        return self.apply(a)


class SimBody:
    """A separate physics simulation standing in for the real arm + wrist camera. Same interface the runner uses on RealArm."""
    rate_hz = 30.0

    def __init__(self, slot, width_m, seed, noise_m=0.006):
        self.e, _ = make_env(slot, width_m, seed=seed)
        e = self.e
        self.rng = np.random.default_rng(seed)
        if width_m < 0.9 * e.width[e.name]:                   # shrink the sim object to the food's size (as the env's small variant does)
            pass                                              # geometry is fixed at build time; the env's own small-object path covers this in training
        self.lf = e.model.jnt_qposadr[e.model.joint("left_finger").id]
        self.lfd = e.model.jnt_dofadr[e.model.joint("left_finger").id]
        self.noise, self.bias = noise_m, self.rng.normal(0, 0.008, 3) * [1, 1, 0]
        self.z0 = float(e.scene.object_pos(e.name)[2])

    def read(self):
        e = self.e
        return e.data.qpos[e.qadr].copy(), e.data.qvel[e.dadr].copy(), float(e.data.qpos[self.lf]), float(e.data.qvel[self.lfd])

    def send(self, q_target, grip_m):
        e = self.e
        e.data.ctrl[:6], e.data.ctrl[e.grip_aid] = q_target, grip_m
        for _ in range(e.n_substeps):
            mujoco.mj_step(e.model, e.data)

    def camera(self):
        e = self.e
        p = e.scene.object_pos(e.name)
        if not e._in_wrist_view(p):
            return None
        rng_m = float(np.linalg.norm(p - e._tcp()))
        return p + (self.bias + self.rng.normal(0, self.noise, 3)) * min(1.0, rng_m / 0.3)

    def lifted_m(self):
        return float(self.e.scene.object_pos(self.e.name)[2]) - self.z0


def run(body, brain, log=print, settle_ticks=5, hold_s=2.0):
    e, ecfg = brain.e, brain.e.ecfg
    q, qd, g, gd = body.read()
    e.data.ctrl[:6], e.data.ctrl[e.grip_aid] = q, e.grip_range[1]            # targets start where the arm IS, jaws open
    stage, sensed, blocked, anchor, t_hold, result = "reach", 0, 0, None, 0, "timeout"
    slow = 1.0 / max(0.1, brain.speed)
    for i in range(int(900 * slow)):
        q, qd, g, gd = body.read()
        est = body.camera() if anchor is None else None                      # once it is in the jaws the camera cannot see it
        tcp = brain.sync(q, qd, g, est)
        if stage in ("reach", "grasp"):
            obs, (qt, gt) = brain.policy_step(stage)
            if stage == "reach":
                arrived = (np.linalg.norm(obs[13:16]) <= float(ecfg["success_distance_m"]) and np.abs(qd).max() <= float(ecfg["reach_settle_qvel"])
                           and brain.tcp_speed <= float(ecfg["reach_settle_speed_mps"]) and brain.tilt_deg() <= float(ecfg["max_tilt_deg"]) and est is not None)
                sensed = sensed + 1 if arrived else 0
                if sensed >= settle_ticks:
                    stage = "grasp"; log(f"  t={i/30:4.1f}s reach ARRIVED (camera gap {1000*np.linalg.norm(obs[13:16]):.0f} mm, tilt {brain.tilt_deg():.0f} deg) -> place+pinch")
                elif i > 240 * slow:
                    result = "reach never settled"; break
            else:
                # motor-feedback pinch: commanded further shut than the jaws are, jaws stopped, and not closed on nothing
                is_blocked = (g - gt) > 0.003 and abs(gd) < 0.003 and g > 0.006
                blocked = blocked + 1 if is_blocked else 0
                if blocked >= 3:
                    stage, anchor = "lift", tcp.copy(); log(f"  t={i/30:4.1f}s jaws BLOCKED at {1000*g:.1f} mm/finger -> lift")
                elif i > 600 * slow:
                    result = "never got hold of it"; break
        elif stage == "lift":
            qt, gt = brain.lift_step(anchor, +1.0)
            if tcp[2] - anchor[2] >= LIFT_TO:
                stage, t_hold = "hold", i; log(f"  t={i/30:4.1f}s lifted {1000*(tcp[2]-anchor[2]):.0f} mm -> hold")
        elif stage == "hold":
            qt, gt = e.data.ctrl[:6].copy(), float(e.data.ctrl[e.grip_aid])
            if i - t_hold >= hold_s * 30:
                held = (g - gt) > 0.003 and g > 0.006
                result = "PICKED UP AND HELD" if held else "lifted but the jaws are empty"
                break
        body.send(qt, gt)
    return result, stage


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--food", default="grape"); ap.add_argument("--body", default="sim", choices=["sim", "real"])
    ap.add_argument("--episodes", type=int, default=10); ap.add_argument("--speed", type=float, default=1.0); ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    slot, width = FOODS.get(a.food.lower(), ("apple", 0.04))
    if a.body == "real":
        raise SystemExit("the real body is wired in only after the sim body passes and the start pose has held on the arm (see README)")
    from collections import Counter
    tally, truth = Counter(), 0
    for ep in range(a.episodes):
        brain, body = Brain(slot, width, speed=a.speed), SimBody(slot, width, seed=1000 + ep)
        brain.e.width[brain.e.name] = body.e.width[body.e.name]              # sim body: the object is the sim's size
        brain.e.rest_z[brain.e.name] = body.e.rest_z[body.e.name]
        brain.e.reach_offset = body.e.reach_offset.copy()
        res, stage = run(body, brain, log=(lambda *x: None) if a.quiet else print)
        ok = body.lifted_m() > 0.08
        truth += ok; tally[res] += 1
        print(f"ep {ep}: {res}   (truth: object {1000*body.lifted_m():.0f} mm above where it started)")
    print(f"\n{a.food} [{slot}]: runner says {dict(tally)};  object truly in the air at the end: {truth}/{a.episodes}")
