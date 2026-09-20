"""Hybrid feeding. Starts where the finalized grasp ends: learned reach -> learned place+pinch -> scripted
lift -> SCRIPTED FEED (pull back and level, centre the FACE in the wrist camera, approach slowing as the
face looms, stop on apparent face size).

Outcome ranking, worst first (the user's):
  1. FAST ARRIVAL  - reaching the head at an inappropriate speed, or any arm/claw contact with the person
  2. DROPPED       - the object left the claws
  3. other misses  - never picked up, lost the face, timed out, stopped badly placed
  ok FED           - stopped on face size alone, food presented in front of the mouth, nothing touched

The feed senses ONLY what the wrist camera gives about the face: where the mouth is in the image and how
wide the face looks (degrees). It does not know the person's head size (re-drawn +-8% per episode), the
object, the range, or - outside CALIBRATE - anything about contact.

  CALIBRATE=1  contact detection ON: crawl in until the FOOD first touches the face, record how wide the
               face looked at that instant. Written to arm/rl/configs/feed_calibration.json.
  default      contact detection unused: stop at the calibrated size minus a stand-off margin.
"""
import glob
import json
import os
import pickle
import re
import sys
from collections import Counter

import mujoco
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["YAM_HANDOFF_REUSE"] = "0"

from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_pick import lift_action, shield, shield_cam  # noqa: E402

from hackmit_rl.config import load_config
from hackmit_rl.envs import OpenYAMFeedEnv

RUN = sys.argv[1] if len(sys.argv) > 1 else "arm/rl/models/grasp_v1"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
CALIBRATE = os.environ.get("CALIBRATE", "0") == "1"
CAL_FILE = "arm/rl/configs/feed_calibration.json"
DT = 1.0 / 30.0
TRACE_ROWS = []
RECORD_DIR = os.environ.get("RECORD_DIR", "")               # per-step data for every episode
SENSORS_ONLY = os.environ.get("SENSORS_ONLY", "1") == "1"     # decide ONLY from wrist camera + motor feedback
STEP_HOOK = None            # a viewer sets this: f(env, phase, face_size_deg, outcome_dict, info, context)

# ---- motion limits. Brisk pulling back, gentle going in, a crawl at the face.
V_RETREAT = 0.20            # m/s, away from the table / turning toward the person
V_APPROACH = 0.10           # m/s, far from the face
V_CRAWL = 0.012             # m/s, the only speed allowed near the face
A_MAX = 0.35                # m/s^2
W_MAX = 0.40                # rad/s, levelling the claws
W_NEAR = 0.05               # rad/s once the face is close: 0.05 x 0.13 m = 6.5 mm/s at the claw tips
SAFE_ARRIVAL = 0.03         # m/s: anything faster than this at the face is the WORST outcome
# Jaw closure past first contact. Measured drops per 30 feeds: 6 mm -> 10, 12 mm -> 2, 20 mm -> 0, 40 mm -> 1.
# (Fully shut only ejected objects while the sim plates were 12 mm wide; at 28 mm a firm hold is safe.)
SQUEEZE_M = float(os.environ.get("SQUEEZE_M", "0.020"))
FACE_W = 0.145              # m, the face width the controller ASSUMES (average adult); the truth varies
STANDOFF_M = float(os.environ.get("STANDOFF_M", "0.015"))   # stop this far short of the calibrated touch
PRESENT_POS = np.array([0.30, 0.0, 0.40])
FORWARD = np.array([1.0, 0.0, 0.0])
# Claws rolled so their fin points DOWN (chin side), and the food aimed 2 cm below the mouth: claw-first
# contacts per 24 went 7 (nose) -> 1. The nose sticks out 2.4 cm right above the mouth.
FLIP = os.environ.get("FLIP", "1") == "1"          # roll the claws 180 deg about the approach axis
AIM_DROP_M = float(os.environ.get("AIM_DROP_M", "0.02"))   # aim this far BELOW the mouth (clear of the nose)
FISHEYE_H_FOV = 150.0 * 1280.0 / np.hypot(1280.0, 720.0)    # 150 deg diagonal, equidistant, 16:9 -> 130.7 deg


def size_to_range(size_deg):
    return 0.5 * FACE_W / np.tan(np.radians(min(size_deg, 175.0)) / 2.0)


def range_to_size(r):
    return float(np.degrees(2.0 * np.arctan(0.5 * FACE_W / max(r, 1e-3))))


def _rotvec(R_err):
    ang = np.arccos(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0))
    if ang < 1e-6:
        return np.zeros(3)
    ax = np.array([R_err[2, 1] - R_err[1, 2], R_err[0, 2] - R_err[2, 0], R_err[1, 0] - R_err[0, 1]])
    return ax / (2.0 * np.sin(ang)) * ang


class Feeder:
    def __init__(self, e):
        self.e, self.v = e, np.zeros(3)
        self.jp, self.jr = np.zeros((3, e.model.nv)), np.zeros((3, e.model.nv))

    def face_measurement(self, rng):
        """(seen, mouth bearing as camera-frame tangents, face ANGULAR width in degrees). Degrees, because
        the sim camera is a pinhole and the real one a 150-deg fisheye: only angles survive that change."""
        e, d = self.e, self.e.data
        R, cam = d.cam_xmat[e.wrist_cam].reshape(3, 3), d.cam_xpos[e.wrist_cam]
        q = R.T @ (e.scene.site("mouth") - cam)
        if q[2] > -0.03:
            return False, np.zeros(2), 0.0
        half = np.tan(np.radians(0.5 * float(e.model.cam_fovy[e.wrist_cam])))
        bearing = np.array([q[0] / -q[2], q[1] / -q[2]])
        if np.max(np.abs(bearing)) > half:
            return False, bearing, 0.0
        face = e.scene.site("face")
        true_w = 2.0 * 0.0725 * float(e.head_scale)                     # THIS person's face width
        size = np.degrees(2.0 * np.arctan(0.5 * true_w / max(1e-3, float(np.linalg.norm(face - cam)))))
        return True, bearing + rng.normal(0, 0.006, 2), float(size * (1.0 + rng.normal(0, 0.02)))

    def action(self, v_des, R_des, v_cap, hold_ctrl, w_cap=W_MAX):
        e, md, d = self.e, self.e.model, self.e.data
        speed = float(np.linalg.norm(v_des))
        if speed > v_cap:
            v_des = v_des * (v_cap / speed)
        dv = v_des - self.v
        n = float(np.linalg.norm(dv))
        if n > A_MAX * DT:
            dv *= A_MAX * DT / n
        self.v = self.v + dv
        Rt = d.site_xmat[e.tool.site_id].reshape(3, 3)
        w = _rotvec(R_des @ Rt.T) * 2.0
        wn = float(np.linalg.norm(w))
        if wn > w_cap:
            w *= w_cap / max(wn, 1e-9)
        mujoco.mj_jacSite(md, d, self.jp, self.jr, e.tool.site_id)
        J = np.vstack([self.jp[:, e.dadr], self.jr[:, e.dadr]])
        r = e._tcp() - d.site_xpos[e.tool.site_id]
        twist = np.concatenate([self.v - np.cross(w, r), w]) * DT
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), twist)
        a = np.zeros((1, 7), dtype=np.float32)
        a[0, :6] = np.clip(dq / float(e.ecfg["action_delta_rad"]), -1.0, 1.0)
        a[0, 6] = hold_ctrl
        return a


def desired_frame(e, tool_dir):
    t = tool_dir / np.linalg.norm(tool_dir)
    jaw = np.cross(np.array([0.0, 0.0, 1.0]), t) * (-1.0 if FLIP else 1.0)
    jaw /= max(1e-9, np.linalg.norm(jaw))
    tl, jl = e.tool.tool_local, e.tool.jaw_local
    return np.column_stack([t, jaw, np.cross(t, jaw)]) @ np.column_stack([tl, jl, np.cross(tl, jl)]).T


def contacts(e):
    d, gid = e.data, e.obj_gid[e.name]
    food, arm = False, ""
    for i in range(d.ncon):
        pair = {int(d.contact[i].geom1), int(d.contact[i].geom2)}
        u = pair & e.user_geoms
        if not u:
            continue
        food |= gid in pair
        if pair & e.arm_geoms:
            ag = next(iter(pair & e.arm_geoms))
            arm = e.model.body(int(e.model.geom_bodyid[ag])).name + " -> " + e.model.geom(next(iter(u))).name
    return food, arm


def jaws_blocked(e):
    """MOTOR FEEDBACK pinch detection: the jaws are commanded further shut than they are, they have stopped
    moving, and they are not empty-closed. That is what having hold of something looks like from the
    gripper's own encoder; no contact sensor and no simulator truth involved."""
    j = e.model.joint("left_finger")
    q = float(e.data.qpos[e.model.jnt_qposadr[j.id]])
    qd = float(e.data.qvel[e.model.jnt_dofadr[j.id]])
    return (q - float(e.data.ctrl[e.grip_aid])) > 0.003 and abs(qd) < 0.003 and q > 0.006


def camera_object(e):
    """The object as the WRIST CAMERA reports it (bias, jitter, frozen at last-seen when out of view)."""
    return np.asarray(e._perceived(e.scene.object_pos(e.name), e.obj_bias, "object_jitter_m"), dtype=float)


def camera_head(e):
    """Head centre from the camera's last sighting of the mouth plus nominal face geometry."""
    mouth = np.asarray(e._perceived(e.scene.site("mouth"), e.mouth_bias, "mouth_jitter_m"), dtype=float)
    return mouth + np.array([0.089, 0.0, 0.048])


def approach_speed(size, contact_size):
    """The speed allowed at this apparent face size. Full speed far out, the crawl from 8 deg before the
    calibrated touch size. This schedule, not the stop, is what prevents a fast arrival: even if the stop
    were missed entirely, the arm would reach the face at the crawl."""
    far, near = contact_size - 28.0, contact_size - 12.0
    return float(V_CRAWL + (V_APPROACH - V_CRAWL) * np.clip((near - size) / (near - far), 0.0, 1.0))


def outcome(r):
    if r["arm_hit"] or (r["touch_speed"] or 0.0) > SAFE_ARRIVAL or r["near_vmax"] > SAFE_ARRIVAL:
        return "1 FAST ARRIVAL / ARM CONTACT"
    if r["dropped"]:
        return "2 DROPPED"
    if not r["picked"]:
        # Say what actually happened. "Never picked up" used to cover an object held 7 cm in the air.
        if r["max_lift"] >= 0.03:
            return f"3 picked up but the lift stalled at {1000*r['max_lift']:.0f} mm (needs 100)"
        return "3 pinched but never left the table" if r["seated_pinch"] else "3 no seated pinch"
    if CALIBRATE:
        return "ok food touched the face at a crawl" if r["food_touch"] else "3 never reached the face"
    if r["phase"] == "hold" and r["ahead"] <= 60.0 and r["side"] <= 25.0:
        return "ok FED"
    return "3 stopped but badly placed" if r["phase"] == "hold" else "3 never reached the stop"


def main(n_episodes=None, report=True):
    n_eps = n_episodes or N
    cks = [c for c in glob.glob(f"{RUN}/checkpoints/ppo_grasp_*_steps.zip") if os.path.getsize(c) > 0]
    ck = max(cks, key=lambda p: int(re.search(r"_(\d+)_steps", p).group(1)))
    step = re.search(r"_(\d+)_steps", ck).group(1)
    with open(glob.glob(f"{RUN}/checkpoints/*vecnormalize_{step}_steps.pkl")[0], "rb") as fh:
        vn = pickle.load(fh)
    mean, std, clip = vn.obs_rms.mean, np.sqrt(vn.obs_rms.var + vn.epsilon), float(vn.clip_obs)
    policy = PPO.load(ck, device="cpu").policy

    cal = json.load(open(CAL_FILE)) if os.path.exists(CAL_FILE) else {}
    contact_size = float(os.environ.get("CONTACT_SIZE", cal.get("contact_size_deg_median", 52.0)))
    stop_size = range_to_size(size_to_range(contact_size) + STANDOFF_M)

    cfg = load_config("arm/rl/configs/feed.yaml")
    cfg["env"]["stage"] = "present"
    cfg["env"]["handoff"]["runs"] = {"reach": RUN if os.path.exists(f"{RUN}/ppo_reach_final.zip")
                                     else cfg["env"]["handoff"]["runs"]["reach"]}
    e = OpenYAMFeedEnv(cfg)
    rng = np.random.default_rng(int(os.environ.get("SEED", "0")))
    jacp, jacr = np.zeros((3, e.model.nv)), np.zeros((3, e.model.nv))
    min_seat = float(cfg["env"]["min_seat_frac"])
    ctx = dict(contact_size=contact_size, stop_size=stop_size, tally=Counter(), episode=0)

    def grasp_obs():
        st, tg = e.stage, e.target
        e.stage, e.target = "grasp", e.scene.object_pos(e.name).copy()
        o = e._observation()
        e.stage, e.target = st, tg
        return np.clip((o - mean) / std, -clip, clip)

    rows = []
    for ep in range(n_eps):
        ctx["episode"] = ep + 1
        e.reset(seed=int(os.environ.get("SEED", "0")) * 100000 + 2000 + ep)
        feeder = Feeder(e)
        phase, anchor, seated, info, hold_ctrl = "pinch", None, 0, {}, -1.0
        o = dict(obj=e.name, head=e.head_scale, picked=False, dropped=False, drop_phase="", arm_hit="", arm_hit_speed=0.0,
                 food_touch=False, touch_size=None, touch_speed=None, stop_size=None, near_vmax=0.0,
                 vmax=0.0, amax=0.0, ahead=None, side=None, phase="", speed=0.0, max_lift=0.0, frozen=0, steps=0,
                 seated_pinch=False)
        hold, vs, prev_vs, prev_tcp = 0, np.zeros(3), np.zeros(3), e._tcp().copy()
        head_est, obj_at_pinch, log = camera_head(e), camera_object(e), []
        for i in range(800):
            size = 0.0
            if phase == "pinch":
                a, _ = policy.predict(grasp_obs()[None], deterministic=True)
                a = np.asarray(a, dtype=np.float32)
            elif phase == "lift":
                if SENSORS_ONLY:
                    rise = float(e._tcp()[2] - anchor[2])                      # arm FK = motor feedback
                    a = lift_action(e, anchor, jacp, jacr, rise,
                                    err_xy=(obj_at_pinch - camera_object(e))[:2], head_est=head_est)
                else:
                    a = lift_action(e, anchor, jacp, jacr, info.get("lift_m", 0.0))
                a[0, 6] = hold_ctrl
            else:
                seen, bearing, size = feeder.face_measurement(rng)
                tcp = e._tcp()
                Rt = e.data.site_xmat[e.tool.site_id].reshape(3, 3)
                tool_axis = Rt @ e.tool.tool_local
                if phase == "retreat":
                    err = PRESENT_POS - tcp
                    a = feeder.action(2.0 * err, desired_frame(e, FORWARD), V_RETREAT, hold_ctrl)
                    if np.linalg.norm(err) < 0.03 and float(tool_axis @ FORWARD) > 0.97 and seen:
                        phase = "approach"
                elif phase == "approach":
                    if not seen:                       # lost the face: stand still. Never advance blind.
                        a = feeder.action(np.zeros(3), desired_frame(e, FORWARD), V_CRAWL, hold_ctrl)
                    else:
                        Rc, cam = e.data.cam_xmat[e.wrist_cam].reshape(3, 3), e.data.cam_xpos[e.wrist_cam]
                        zc = max(0.04, size_to_range(size))                     # range from face size ALONE
                        tcp_c, axis_c = Rc.T @ (tcp - cam), Rc.T @ tool_axis
                        p = tcp_c + ((-zc - tcp_c[2]) / min(-1e-3, axis_c[2])) * axis_c
                        err_c = (bearing - np.array([p[0] / zc, p[1] / zc])) * zc   # camera sits off the claw axis
                        lateral = Rc[:, 0] * err_c[0] + Rc[:, 1] * err_c[1] - np.array([0.0, 0.0, AIM_DROP_M])
                        lateral -= (lateral @ tool_axis) * tool_axis
                        v_in = approach_speed(size, contact_size)
                        if np.linalg.norm(lateral) > 0.03:                       # line up before driving in
                            v_in = min(v_in, 0.3 * V_APPROACH)
                        if not CALIBRATE and size >= stop_size:
                            phase, o["stop_size"] = "hold", size
                            v_in = 0.0
                        # The cap is on TOTAL hand speed, sideways corrections included. Leaving slack for
                        # them let the hand do 0.036 m/s at the face with the inward speed at the crawl.
                        # Rotation gets its own crawl near the face. The claw tips are ~13 cm from the wrist,
                        # so re-orienting at 0.4 rad/s swung them at 0.052 m/s while translation obeyed the cap.
                        w_cap = W_MAX if size < contact_size - 18.0 else W_NEAR
                        a = feeder.action(v_in * tool_axis + 1.5 * lateral, desired_frame(e, FORWARD),
                                          1.4 * max(v_in, V_CRAWL), hold_ctrl, w_cap)
                else:
                    a = feeder.action(np.zeros(3), Rt, V_CRAWL, hold_ctrl, 0.0)
                    hold += 1
            if phase in ("pinch", "lift", "retreat"):
                a, _, froze = shield_cam(e, a, head_est) if SENSORS_ONLY else shield(e, a)
                o["frozen"] += int(froze)
                if froze and phase == "lift":
                    # The object is close to the person, so lifting straight up raises the forearm toward his
                    # head; the shield backed it off, the lift pushed back, and they ping-ponged for 230-290
                    # steps. Give way properly: bring the hand up AND back toward the robot. Safety outranks
                    # a perfectly vertical lift.
                    Rt_ = e.data.site_xmat[e.tool.site_id].reshape(3, 3)
                    a = feeder.action(np.array([-0.06, 0.0, 0.04]), Rt_, 0.08, hold_ctrl, 0.0)
            _, _, _, _, info = e.step(a[0])

            tcp = e._tcp()
            v = (tcp - prev_tcp) / DT
            vs = 0.7 * vs + 0.3 * v
            sp = float(np.linalg.norm(vs))
            o["speed"] = sp
            # CLOSING speed on the face: what matters is motion TOWARD the person. Total hand speed also
            # spikes when an object slips out and the unloaded arm springs a few mm - away from him.
            to_face = e.scene.site("mouth") - tcp
            closing = max(0.0, float(vs @ (to_face / max(1e-9, np.linalg.norm(to_face)))))
            o["closing"] = closing
            o["steps"] = i + 1
            o["max_lift"] = max(o["max_lift"], float(info.get("lift_m", 0.0)))
            if phase in ("retreat", "approach", "hold"):
                o["vmax"] = max(o["vmax"], sp)
                o["amax"] = max(o["amax"], float(np.linalg.norm(vs - prev_vs) / DT))
                if size >= contact_size - 8.0:
                    if closing > o["near_vmax"]:
                        o["near_vmax"], o["near_where"] = closing, f"{phase}@{size:.0f}deg"
            prev_tcp, prev_vs = tcp.copy(), vs.copy()

            if os.environ.get("TRACE") and phase in ("retreat", "approach", "hold") and i % 6 == 0:
                Rt_ = e.data.site_xmat[e.tool.site_id].reshape(3, 3)
                ta_, ja_ = Rt_ @ e.tool.tool_local, Rt_ @ e.tool.jaw_local
                rel = e.scene.object_pos(e.name) - tcp
                w6, tot, npad, nplate = np.zeros(6), 0.0, 0, 0
                for ci in range(e.data.ncon):
                    c = e.data.contact[ci]; pr = {int(c.geom1), int(c.geom2)}
                    if e.obj_gid[e.name] in pr and pr & (e.finger_geoms | e.pad_geoms):
                        mujoco.mj_contactForce(e.model, e.data, ci, w6); tot += abs(float(w6[0]))
                        npad += bool(pr & e.pad_geoms); nplate += not bool(pr & e.pad_geoms)
                gap = float(np.linalg.norm(e.data.geom_xpos[sorted(e.left_pads)].mean(0) - e.data.geom_xpos[sorted(e.right_pads)].mean(0)))
                TRACE_ROWS.append(f"    ep{ep} {e.name:5s} {phase:8s} t{i:3d} along {1000*rel@ta_:6.1f} jaw {1000*rel@ja_:6.1f} across {1000*rel@np.cross(ta_, ja_):6.1f} mm | "
                                  f"gap {1000*gap:5.1f} width {1000*e.width[e.name]:5.1f} | grip {tot:5.1f} N ({npad} pad, {nplate} plate contacts) | tool_z {ta_[2]:+.2f}")
            food, arm = contacts(e)
            done = False
            if RECORD_DIR:
                fj = e.model.joint("left_finger")
                log.append(dict(t=i, phase=phase, face_deg=round(float(size), 3),
                                closing=round(float(o.get("closing", 0.0)), 5), speed=round(sp, 5),
                                q=[round(float(x), 5) for x in e.data.qpos[e.qadr]],
                                grip=round(float(e.data.qpos[e.model.jnt_qposadr[fj.id]]), 5),
                                jaws_blocked=bool(jaws_blocked(e)), food_touch=bool(food), arm_touch=bool(arm)))
            if arm:
                o["arm_hit"], o["arm_hit_speed"] = arm, closing
                done = True
            elif phase == "pinch":
                if SENSORS_ONLY:
                    # hand over on the gripper's own feedback + the camera's view of how deep the object sits
                    Rt_ = e.data.site_xmat[e.tool.site_id].reshape(3, 3)
                    ax_ = Rt_ @ e.tool.tool_local
                    ins = float((camera_object(e) - (e._tcp() + ax_ * float(e.tip_ahead or 0.036))) @ (-ax_))
                    half_w = 0.5 * float(e.width[e.name]) * (1.0 + e.width_bias)
                    ok = jaws_blocked(e) and ins / max(1e-3, half_w - 0.005) >= min_seat
                else:
                    ok = info["pinched"] and info["seat_frac"] >= min_seat
                seated = seated + 1 if ok else 0
                if seated >= 3:
                    phase, anchor = "lift", e._tcp().copy()
                    o["seated_pinch"] = True
                    obj_at_pinch = camera_object(e)
                    q_now = float(e.data.qpos[e.model.jnt_qposadr[e.model.joint("left_finger").id]])
                    lo, hi = float(e.grip_range[0]), float(e.grip_range[1])
                    hold_ctrl = float(np.clip(2.0 * (max(0.0, q_now - 0.5 * SQUEEZE_M) - lo) / (hi - lo) - 1.0, -1, 1))
                done = info["stage_drift_m"] > 0.09 or i > 250
            elif phase == "lift":
                if SENSORS_ONLY:
                    up = float(e._tcp()[2] - anchor[2]) >= 0.10 and jaws_blocked(e)
                else:
                    up = info["lift_m"] >= 0.10
                if up:
                    phase = "retreat"
                    o["picked"] = bool(info["lift_m"] >= 0.08)   # SCORING may use truth; the decision did not
                done = i > 420
            else:
                if float(np.linalg.norm(e.scene.object_pos(e.name) - tcp)) > 0.08:
                    o["dropped"], o["drop_phase"] = True, phase
                    done = True
                if food and not o["food_touch"]:
                    o["food_touch"], o["touch_size"], o["touch_speed"] = True, size, closing
                    done = done or CALIBRATE
                done = done or (phase == "hold" and hold >= 25)
            o["phase"] = phase
            if STEP_HOOK is not None:
                STEP_HOOK(e, phase, size, o, info, ctx)
            if done:
                break
        rel = e.scene.site("mouth") - e.scene.object_pos(e.name)
        o["ahead"] = 1000.0 * max(0.0, float(rel @ FORWARD) - 0.5 * float(e.width[e.name]))
        o["side"] = 1000.0 * float(np.linalg.norm(rel - (rel @ FORWARD) * FORWARD))
        o["result"] = outcome(o)
        if RECORD_DIR:
            os.makedirs(RECORD_DIR, exist_ok=True)
            meta = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in o.items()}
            meta.update(seed=int(os.environ.get("SEED", "0")), episode=ep, head_scale=float(e.head_scale),
                        object_width_m=float(e.width[e.name]), calibrate=CALIBRATE, sensors_only=SENSORS_ONLY)
            name = "s" + os.environ.get("SEED", "0") + "_ep" + str(ep).zfill(4) + ".json"
            with open(os.path.join(RECORD_DIR, name), "w") as fh:
                json.dump(dict(meta=meta, steps=log), fh)
        if os.environ.get("TRACE"):
            if o["dropped"]:
                print(f"--- DROP ep{ep} {e.name} in {o['drop_phase']}")
                print(chr(10).join(TRACE_ROWS[-14:]))
            TRACE_ROWS.clear()
        ctx["tally"][o["result"]] += 1
        ctx["last"] = o["result"]
        rows.append(o)
        if STEP_HOOK is not None:
            STEP_HOOK(e, "done", 0.0, o, info, ctx)
    if not report:
        return rows

    # ------------------------------------------------------------------ report, worst outcome first
    tally = Counter(re.sub(r"stalled at [0-9]+ mm", "stalled below 100 mm", r["result"]) for r in rows)
    stalled = [r for r in rows if not r["picked"]]
    if stalled:
        print("not-picked detail (max lift mm, steps, shield-frozen steps, seated pinch):",
              [(r["obj"], round(1000 * r["max_lift"]), r["steps"], r["frozen"], r["seated_pinch"]) for r in stalled])
    print(f"\n{'CALIBRATION (contact detection ON)' if CALIBRATE else 'FEED (no contact information)'}   "
          f"{n_eps} episodes   pinch {os.path.basename(ck)}   wrist FOV {e.model.cam_fovy[e.wrist_cam]:.0f} deg")
    print(f"schedule: crawl {V_CRAWL} m/s from face size {contact_size - 12:.1f} deg"
          + ("" if CALIBRATE else f"; stop at {stop_size:.1f} deg = {STANDOFF_M*100:.1f} cm short of the touch size {contact_size:.1f} deg"))
    for k in sorted(tally):
        print(f"  {tally[k]:3d}/{n_eps}  {k}")
    near = [r["near_vmax"] for r in rows if r["near_vmax"] > 0]
    if near:
        print(f"CLOSING SPEED ON THE FACE (within 8 deg of touch): max {max(near):.3f} m/s, mean {np.mean(near):.3f}  "
              f"(limit {SAFE_ARRIVAL}); overall peak {max(r['vmax'] for r in rows):.2f} m/s, peak accel "
              f"{max(r['amax'] for r in rows):.2f} m/s^2")
    fast = [(r.get("near_where"), round(r["near_vmax"], 3)) for r in rows if r["near_vmax"] > SAFE_ARRIVAL]
    if fast:
        print("  over-limit speed near the face (phase@size, m/s):", fast)
    hits = [(r["arm_hit"], round(r["arm_hit_speed"], 3)) for r in rows if r["arm_hit"]]
    if hits:
        print("  arm/claw contacts (part, speed m/s):", hits)
    drops = [r for r in rows if r["dropped"]]
    if drops:
        print("  drops by phase/object:", dict(Counter((r["drop_phase"], r["obj"]) for r in drops)))
    if CALIBRATE:
        t = [r for r in rows if r["food_touch"]]
        if t:
            s = np.array([r["touch_size"] for r in t])
            print(f"FACE SIZE WHEN THE FOOD TOUCHED ({len(s)} touches): median {np.median(s):.1f} deg  "
                  f"(min {s.min():.1f}, p10 {np.percentile(s, 10):.1f}, max {s.max():.1f})")
            print(f"  = {100*np.median(s)/FISHEYE_H_FOV:.0f}% of the image WIDTH on a 150-deg (diagonal) fisheye at 16:9 "
                  f"(range {100*s.min()/FISHEYE_H_FOV:.0f}-{100*s.max()/FISHEYE_H_FOV:.0f}%)")
            print(f"  contact speed: max {max(r['touch_speed'] for r in t):.3f} m/s, mean "
                  f"{np.mean([r['touch_speed'] for r in t]):.3f}")
            small = [r["touch_size"] for r in t if r["head"] < 0.97]
            big = [r["touch_size"] for r in t if r["head"] > 1.03]
            if small and big:
                print(f"  small heads touch at {np.mean(small):.1f} deg, large heads at {np.mean(big):.1f} deg "
                      f"-> the person-to-person spread the stand-off must cover")
            json.dump({"contact_size_deg_median": float(np.median(s)), "contact_size_deg_min": float(s.min()),
                       "contact_size_deg_p10": float(np.percentile(s, 10)), "contact_size_deg_max": float(s.max()),
                       "n_touches": int(len(s)),
                       "pct_of_image_width_150deg_fisheye": float(100 * np.median(s) / FISHEYE_H_FOV),
                       "assumed_face_width_m": FACE_W,
                       "note": "angular face width when the food first touched the face; sim, contact "
                               "detection on, head size +-8%"}, open(CAL_FILE, "w"), indent=2)
            print(f"  written to {CAL_FILE}")
    else:
        st = [r for r in rows if r["phase"] == "hold"]
        if st:
            print(f"at the stop ({len(st)}): food {np.mean([r['ahead'] for r in st]):.0f} mm in front of the mouth "
                  f"(min {min(r['ahead'] for r in st):.0f}, max {max(r['ahead'] for r in st):.0f}), "
                  f"{np.mean([r['side'] for r in st]):.0f} mm off to the side; food touched the face in "
                  f"{sum(r['food_touch'] for r in st)}")
    return rows


if __name__ == "__main__":
    main()
