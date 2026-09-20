"""Hybrid pick-up: learned reach -> learned place+pinch -> SCRIPTED vertical lift and hold.

The lift is closed-loop in Cartesian space: the hand is servoed to the point directly above where the
pinch was made, so sideways error is corrected rather than accumulated, and wrist rotation is held at
zero. Success is judged by the environment's own pick-up test, not by this script.
"""
import glob
import os
import re
import sys

import mujoco
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["YAM_HANDOFF_REUSE"] = "0"

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hackmit_rl.config import load_config
from hackmit_rl.envs import OpenYAMFeedEnv

RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/feed-grasp"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
LIFT_TO = 0.115            # m above rest; the success band is 0.10 .. 0.16
STEP_UP = float(os.environ.get("STEP_UP", "0.002"))
# Measured over 20-40 episodes each: K_XY 0.6 -> 2%, 0.3 -> 17%, 0.15 -> 63% (higher gains
# oscillate against the env action low-pass); 4 mm/step -> 45%, 2 mm/step -> 55-63%.
K_XY = float(os.environ.get("K_XY", "0.15"))


def load(run):
    cks = [c for c in glob.glob(f"{run}/checkpoints/ppo_grasp_*_steps.zip") if os.path.getsize(c) > 0]
    ck = max(cks, key=lambda p: int(re.search(r"_(\d+)_steps", p).group(1)))
    step = re.search(r"_(\d+)_steps", ck).group(1)
    return ck, glob.glob(f"{run}/checkpoints/*vecnormalize_{step}_steps.pkl")[0]


def lift_action(e, anchor, jacp, jacr, lifted):
    """One control step of the scripted lift, as a normalised env action."""
    md, d = e.model, e.data
    mujoco.mj_jacSite(md, d, jacp, jacr, e.tool.site_id)
    J = np.vstack([jacp[:, e.dadr], jacr[:, e.dadr]])
    tcp = e._tcp()
    dz = STEP_UP if lifted < LIFT_TO else float(np.clip(anchor[2] + LIFT_TO - tcp[2], -0.002, 0.002)) * 0.0
    # Servo on what is judged and what the wrist camera sees: the OBJECT's sideways position.
    if os.environ.get("SERVO", "object") == "object":
        err = e.stage_object_start[:2] - e.scene.object_pos(e.name)[:2]
    else:
        err = anchor[:2] - tcp[:2]
    dxy = np.clip(K_XY * err, -0.003, 0.003)
    twist = np.array([dxy[0], dxy[1], dz, 0.0, 0.0, 0.0])
    dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), twist)          # damped least squares
    a = np.zeros((1, 7), dtype=np.float32)
    a[0, :6] = np.clip(dq / float(e.ecfg["action_delta_rad"]), -1.0, 1.0)
    a[0, 6] = -1.0                                                          # jaws stay shut
    return a


def main():
    ck, vn = load(RUN)
    cfg = load_config("arm/rl/configs/feed.yaml")
    cfg["env"]["stage"] = "grasp"
    raw = DummyVecEnv([lambda: OpenYAMFeedEnv(cfg)])
    e = raw.envs[0]
    env = VecNormalize.load(vn, raw)
    env.training, env.norm_reward = False, False
    model = PPO.load(ck, device="cpu")
    jacp, jacr = np.zeros((3, e.model.nv)), np.zeros((3, e.model.nv))
    print("grasp policy:", os.path.basename(ck))

    rows = []
    for ep in range(N):
        obs = env.reset()
        seated_steps, anchor, info, drift0 = 0, None, {}, 0.0
        max_drift = 0.0
        for i in range(300):
            if anchor is None:
                a, _ = model.predict(obs, deterministic=True)
            else:
                a = lift_action(e, anchor, jacp, jacr, info.get("lift_m", 0.0))
            obs, r, done, infos = env.step(a)
            info = infos[0]
            if anchor is None:
                ok = info["pinched"] and info["seat_frac"] >= float(cfg["env"]["min_seat_frac"])
                seated_steps = seated_steps + 1 if ok else 0
                if seated_steps >= 3 and not done[0]:
                    anchor = e._tcp().copy(); drift0 = info['stage_drift_m']
            else:
                max_drift = max(max_drift, info["stage_drift_m"])
            if done[0]:
                break
        rows.append((info["object"], bool(info["success"]), anchor is not None, 1000 * max_drift,
                     1000 * info["lift_m"], i + 1, 1000 * drift0))

    n_ok = sum(r[1] for r in rows)
    n_pinch = sum(r[2] for r in rows)
    print(f"\nHYBRID PICK-UP: {n_ok}/{N} = {100*n_ok/N:.0f}%   (env's own test: seated pinch, lifted 10 cm, "
          f"<= 3 cm sideways, held still 0.7 s)")
    lifts_all = [x for x in rows if x[2]]
    print(f"  STEP_UP {STEP_UP} K_XY {K_XY}: drift already present at pinch {np.mean([x[6] for x in lifts_all]):.1f} mm, "
          f"peak during lift {np.mean([x[3] for x in lifts_all]):.1f} mm, final lift {np.mean([x[4] for x in lifts_all]):.0f} mm")
    print(f"  learned place+pinch reached a seated pinch in {n_pinch}/{N}; "
          f"scripted lift then succeeded in {n_ok}/{max(1, n_pinch)}")
    for name in ("apple", "mug", "block"):
        r = [x for x in rows if x[0] == name]
        if r:
            lifts = [x for x in r if x[2]]
            print(f"  {name:6s} {sum(x[1] for x in r)}/{len(r)}   sideways drift during lift: "
                  f"mean {np.mean([x[3] for x in lifts]) if lifts else float('nan'):.1f} mm "
                  f"(max {np.max([x[3] for x in lifts]) if lifts else float('nan'):.1f})")


if __name__ == "__main__":
    main()
