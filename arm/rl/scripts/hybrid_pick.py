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
KEEP_OUT = 0.10            # m of clearance between any arm geom and the head surface
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


SHIELD_M = float(os.environ.get("SHIELD_M", "0.06"))
SHIELD_HAND_M = float(os.environ.get("SHIELD_HAND_M", "0.02"))
_ft, _jp, _hand = np.zeros(6), None, None


def _hand_geoms(e):
    global _hand
    if _hand is None:
        hb = e.model.body("user_hand").id
        _hand = {g for g in e.user_geoms if int(e.model.geom_bodyid[g]) == hb}
    return _hand


def shield(e, a):
    """Hard safety layer around ANY action, learned or scripted: if some part of the arm is within
    SHIELD_M of the person and the commanded joint motion would carry that point closer, the arm is
    frozen (jaws untouched). Returns (action, clearance_m, frozen). A frozen arm fails the pick-up; it
    never reaches the person. Measured need: the learned place+pinch touched the person in 4/60."""
    global _jp
    md, d = e.model, e.data
    if _jp is None:
        _jp = np.zeros((3, md.nv))
    best, hit = 1e9, None
    for ga in e.arm_geoms:
        if not (md.geom_contype[ga] or md.geom_conaffinity[ga]):
            continue
        for gu in e.user_geoms:
            if not (md.geom_contype[gu] or md.geom_conaffinity[gu]):
                continue                    # hair, eyes, ears are drawn, not solid; the hair shell is bigger than the skull
            # Head and torso get the full margin (the punch case). The hand rests ON the table inside
            # the workspace, so the arm is routinely within 4 cm of it; with one margin for everything
            # the shield froze 44/60 episodes and pick-ups fell 68% -> 50%.
            margin = SHIELD_M if gu not in _hand_geoms(e) else SHIELD_HAND_M
            dist = mujoco.mj_geomDistance(md, d, ga, gu, 0.30, _ft) - margin
            if dist < best:
                best, hit = dist, (ga, _ft.copy())
    if hit is None or best > 0.0:
        return a, best, False
    ga, ft = hit
    toward = ft[3:] - ft[:3]
    n = float(np.linalg.norm(toward))
    if n < 1e-9:
        toward, n = d.xpos[md.body("user_head").id] - ft[:3], 1.0
        n = float(np.linalg.norm(toward))
    mujoco.mj_jac(md, d, _jp, None, ft[:3], int(md.geom_bodyid[ga]))
    v = _jp[:, e.dadr] @ (np.asarray(a[0][:6], dtype=float) * float(e.ecfg["action_delta_rad"]))
    if float(v @ (toward / n)) > 0.0:
        # Do not just freeze: a frozen arm mid-lift stayed frozen for 250-340 steps with the object in the
        # air (the lift raises the elbow toward the head, so every further lift command was vetoed). Back
        # the offending point straight away from the person at 3 cm/s, jaws untouched, and let the caller
        # try again once there is room.
        away = -(toward / n) * 0.03 * (1.0 / 30.0)
        Jp = _jp[:, e.dadr]
        dq = Jp.T @ np.linalg.solve(Jp @ Jp.T + 1e-4 * np.eye(3), away)
        a = np.array(a, dtype=np.float32, copy=True)
        a[0, :6] = np.clip(dq / float(e.ecfg["action_delta_rad"]), -1.0, 1.0)
        return a, best, True
    return a, best, False


def lift_action(e, anchor, jacp, jacr, lifted, err_xy=None, head_est=None):
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
    if err_xy is not None:                 # caller supplies the CAMERA's estimate of the sideways error
        err = np.asarray(err_xy, dtype=float)
    dxy = np.clip(K_XY * err, -0.003, 0.003)
    # HARD KEEP-OUT. The script cannot be taught by a penalty, so it is simply not allowed to close on
    # the person: inside the margin, any motion component toward the head is removed.
    if head_est is not None:               # where the camera last saw the person, not where the sim knows he is
        head = np.asarray(head_est, dtype=float)
        gap = float(min(np.linalg.norm(d.xpos[b] - head)
                        for b in e.swing_bodies + [md.body("link_6").id])) - 0.115
    else:
        head = d.xpos[md.body("user_head").id]
        gap = float(min(np.linalg.norm(d.geom_xpos[g] - head) for g in e.arm_geoms)) - 0.095
    move = np.array([dxy[0], dxy[1], dz])
    if gap < KEEP_OUT:
        toward = (head - tcp) / max(1e-9, float(np.linalg.norm(head - tcp)))
        move = move - max(0.0, float(move @ toward)) * toward
    twist = np.array([move[0], move[1], move[2], 0.0, 0.0, 0.0])
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

    rows, small = [], []
    for ep in range(N):
        obs = env.reset()
        seated_steps, anchor, info, drift0 = 0, None, {}, 0.0
        max_drift = 0.0; hit_phase = None; min_gap = 9.9; n_froze = 0
        for i in range(300):
            if anchor is None:
                a, _ = model.predict(obs, deterministic=True)
            else:
                a = lift_action(e, anchor, jacp, jacr, info.get("lift_m", 0.0))
            a, gap, froze = shield(e, a); min_gap = min(min_gap, gap); n_froze += int(froze)
            obs, r, done, infos = env.step(a)
            info = infos[0]
            if info.get("patient_hit") and hit_phase is None:
                hit_phase = ("scripted lift" if anchor is not None else "learned place+pinch") + " -> " + str(info.get("patient_part"))
            if anchor is None:
                ok = info["pinched"] and info["seat_frac"] >= float(cfg["env"]["min_seat_frac"])
                seated_steps = seated_steps + 1 if ok else 0
                if seated_steps >= 3 and not done[0]:
                    anchor = e._tcp().copy(); drift0 = info['stage_drift_m']
            else:
                max_drift = max(max_drift, info["stage_drift_m"])
            if done[0]:
                break
        small.append((1000 * info['object_width_m'] < 40, bool(info['success'])))
        rows.append((info["object"], bool(info["success"]), anchor is not None, 1000 * max_drift,
                     1000 * info["lift_m"], i + 1, 1000 * drift0, bool(info.get('patient_hit')), hit_phase, 1000 * min_gap, n_froze))

    n_ok = sum(r[1] for r in rows)
    n_pinch = sum(r[2] for r in rows)
    print(f"\nHYBRID PICK-UP: {n_ok}/{N} = {100*n_ok/N:.0f}%   (env's own test: seated pinch, lifted 10 cm, "
          f"<= 3 cm sideways, held still 0.7 s)")
    print(f"  PATIENT TOUCHED in {sum(x[7] for x in rows)}/{N} episodes  during: {[x[8] for x in rows if x[7]]}")
    print(f"  margin left before the shield trips (0 = at the margin): min {min(x[9] for x in rows):.0f} mm, mean {np.mean([x[9] for x in rows]):.0f} mm; "
          f"shield froze the arm in {sum(x[10] > 0 for x in rows)}/{N} episodes")
    print(f"  small objects (< 40 mm): {sum(ok for sm, ok in small if sm)}/{sum(sm for sm, _ in small)}   full-size: {sum(ok for sm, ok in small if not sm)}/{sum(not sm for sm, _ in small)}")
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


def shield_cam(e, a, head_est, margin=0.07):
    """The shield WITHOUT simulator truth. The person is wherever the wrist camera last saw the face
    (`head_est`, camera error included); the arm is where its own joint angles say it is (FK). Keep-out =
    a head sphere (0.115 m, the largest half-axis of an adult head) plus a torso box hanging below it, both
    grown by `margin`. If an arm point is inside and the commanded motion carries it further in, that
    point is backed straight out at 3 cm/s. Returns (action, clearance_m, tripped)."""
    global _jp
    md, d = e.model, e.data
    if _jp is None:
        _jp = np.zeros((3, md.nv))
    head = np.asarray(head_est, dtype=float)
    torso_c, torso_h = head + np.array([0.12, 0.0, -0.22]), np.array([0.10, 0.17, 0.19])
    Rt = d.site_xmat[e.tool.site_id].reshape(3, 3)
    wrist = md.body("link_6").id
    tip = e._tcp() + (Rt @ e.tool.tool_local) * float(e.tip_ahead or 0.036)
    pts = [(d.xpos[b].copy(), b) for b in e.swing_bodies + [wrist]]
    pts += [(e._tcp().copy(), wrist), (tip, wrist)]
    best, hit = 1e9, None
    for pnt, body in pts:
        v = pnt - head
        d_head = float(np.linalg.norm(v)) - 0.115
        q = np.abs(pnt - torso_c) - torso_h
        d_torso = float(np.linalg.norm(np.maximum(q, 0.0)) + min(0.0, float(np.max(q))))
        if d_head <= d_torso:
            dist, away = d_head, v / max(1e-9, float(np.linalg.norm(v)))
        else:
            dist = d_torso
            away = np.sign(pnt - torso_c) * (q == np.max(q))
            away = away / max(1e-9, float(np.linalg.norm(away)))
        if dist - margin < best:
            best, hit = dist - margin, (pnt, body, away)
    if hit is None or best > 0.0:
        return a, best, False
    pnt, body, away = hit
    mujoco.mj_jac(md, d, _jp, None, pnt, int(body))
    Jp = _jp[:, e.dadr]
    v_cmd = Jp @ (np.asarray(a[0][:6], dtype=float) * float(e.ecfg["action_delta_rad"]))
    if float(v_cmd @ away) >= 0.0:
        return a, best, False                                    # already heading out
    dq = Jp.T @ np.linalg.solve(Jp @ Jp.T + 1e-4 * np.eye(3), away * 0.03 / 30.0)
    a = np.array(a, dtype=np.float32, copy=True)
    a[0, :6] = np.clip(dq / float(e.ecfg["action_delta_rad"]), -1.0, 1.0)
    return a, best, True
