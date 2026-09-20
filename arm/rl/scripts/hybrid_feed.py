"""Hybrid feeding: learned reach -> learned place+pinch -> scripted lift -> SCRIPTED FEED.

Feed = pull back and level the gripper toward the person, centre the FACE in the wrist camera (the
object is ignored), approach at a speed that falls as the face looms, stop on apparent face size.

The only sensing used in the feed is what the wrist camera gives about the face: its bearing (where
the mouth is in the image) and its apparent size (face-box width as a fraction of the image). No
object position, no ground-truth range, and -- outside CALIBRATE mode -- no contact information.

  CALIBRATE=1   drive in until the FOOD first touches the face, and record the apparent face size at
                that instant. That distribution is the benchmark the real arm (no contact sensing)
                stops on.
  STOP_SIZE=x   stop when apparent face size reaches x (default: from calibration, minus a margin).
"""
import glob
import os
import pickle
import re
import sys

import mujoco
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["YAM_HANDOFF_REUSE"] = "0"

from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_pick import lift_action, shield  # noqa: E402

from hackmit_rl.config import load_config
from hackmit_rl.envs import OpenYAMFeedEnv

RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/grasp-pinch-best"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
CALIBRATE = os.environ.get("CALIBRATE", "0") == "1"
STOP_SIZE = float(os.environ.get("STOP_SIZE", "0"))          # apparent face size to stop at
DT = 1.0 / 30.0
V_RETREAT, V_APPROACH, V_MIN = 0.22, 0.10, 0.008             # m/s caps; brisk back, gentle in
A_MAX = 0.45                                                 # m/s^2, both phases
W_MAX = 0.5                                                  # rad/s wrist/orientation slew
HEAD_R = 0.095
SQUEEZE_M = float(os.environ.get("SQUEEZE_M", "0.006"))   # total jaw closure past first contact
PRESENT_POS = np.array([0.30, 0.0, 0.40])                    # nominal "hold it up and look forward" pose
FORWARD = np.array([1.0, 0.0, 0.0])                          # the person sits opposite the base


def _rotvec(R_err):
    ang = np.arccos(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0))
    if ang < 1e-6:
        return np.zeros(3)
    ax = np.array([R_err[2, 1] - R_err[1, 2], R_err[0, 2] - R_err[2, 0], R_err[1, 0] - R_err[0, 1]])
    return ax / (2.0 * np.sin(ang)) * ang


class Feeder:
    """Cartesian velocity controller with speed and acceleration caps, in the env's action units."""

    def __init__(self, e):
        self.e, self.v = e, np.zeros(3)
        self.jp, self.jr = np.zeros((3, e.model.nv)), np.zeros((3, e.model.nv))

    def face_measurement(self, rng):
        """What the wrist camera reports about the face: (seen, bearing_xy in the camera frame as
        tangents, apparent size = face width / image width). Nothing else leaves this function."""
        e, d = self.e, self.e.data
        R = d.cam_xmat[e.wrist_cam].reshape(3, 3)
        cam = d.cam_xpos[e.wrist_cam]
        q = R.T @ (e.scene.site("mouth") - cam)
        if q[2] > -0.03:
            return False, np.zeros(2), 0.0
        half = np.tan(np.radians(0.5 * float(e.model.cam_fovy[e.wrist_cam])))
        bearing = np.array([q[0] / -q[2], q[1] / -q[2]])
        if np.max(np.abs(bearing)) > half:
            return False, bearing, 0.0
        dist = float(np.linalg.norm(d.xpos[e.head_bid] - cam))
        # ANGULAR width of the face, in degrees. Lens-independent: the sim camera is a 75-deg pinhole and
        # the real one a fisheye, so anything in pixels or image-fraction would not transfer. On the real
        # arm: undistort (cv2.fisheye K, D), then face-box width -> angle.
        size = float(np.degrees(2.0 * np.arcsin(min(1.0, HEAD_R / max(dist, HEAD_R)))))
        return True, bearing + rng.normal(0, 0.006, 2), float(size * (1.0 + rng.normal(0, 0.02)))

    def action(self, v_des, R_des, v_cap):
        e, md, d = self.e, self.e.model, self.e.data
        speed = float(np.linalg.norm(v_des))
        if speed > v_cap:
            v_des = v_des * (v_cap / speed)
        dv = v_des - self.v
        n = float(np.linalg.norm(dv))
        if n > A_MAX * DT:
            dv *= A_MAX * DT / n
        self.v = self.v + dv                                           # acceleration-limited
        Rt = d.site_xmat[e.tool.site_id].reshape(3, 3)
        w = _rotvec(R_des @ Rt.T) * 3.0
        wn = float(np.linalg.norm(w))
        if wn > W_MAX:
            w *= W_MAX / wn
        mujoco.mj_jacSite(md, d, self.jp, self.jr, e.tool.site_id)
        J = np.vstack([self.jp[:, e.dadr], self.jr[:, e.dadr]])
        # the TCP is offset from the site; account for the lever arm
        r = e._tcp() - d.site_xpos[e.tool.site_id]
        v_site = self.v - np.cross(w, r)
        twist = np.concatenate([v_site, w]) * DT
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), twist)
        a = np.zeros((1, 7), dtype=np.float32)
        a[0, :6] = np.clip(dq / float(e.ecfg["action_delta_rad"]), -1.0, 1.0)
        a[0, 6] = -1.0
        return a


def desired_frame(e, tool_dir):
    """Rotation that points the tool along `tool_dir` with the jaws opening sideways (horizontal), so
    the food is held from the sides and its front is free to bite."""
    t = tool_dir / np.linalg.norm(tool_dir)
    jaw = np.cross(np.array([0.0, 0.0, 1.0]), t)
    jaw /= max(1e-9, np.linalg.norm(jaw))
    third = np.cross(t, jaw)
    # columns of the current site frame expressed by tool_local / jaw_local
    tl, jl = e.tool.tool_local, e.tool.jaw_local
    L = np.column_stack([tl, jl, np.cross(tl, jl)])
    W = np.column_stack([t, jaw, third])
    return W @ L.T


def contacts(e):
    """(food touches person, arm/claw touches person, which part)"""
    d = e.data
    food = arm = False
    part = ""
    gid = e.obj_gid[e.name]
    for i in range(d.ncon):
        pair = {int(d.contact[i].geom1), int(d.contact[i].geom2)}
        u = pair & e.user_geoms
        if not u:
            continue
        if gid in pair:
            food = True
        if pair & e.arm_geoms:
            arm = True
            ag = next(iter(pair & e.arm_geoms))
            part = (e.model.body(int(e.model.geom_bodyid[ag])).name + " -> "
                    + e.model.geom(next(iter(u))).name)
    return food, arm, part


def main():
    cks = [c for c in glob.glob(f"{RUN}/checkpoints/ppo_grasp_*_steps.zip") if os.path.getsize(c) > 0]
    ck = max(cks, key=lambda p: int(re.search(r"_(\d+)_steps", p).group(1)))
    step = re.search(r"_(\d+)_steps", ck).group(1)
    with open(glob.glob(f"{RUN}/checkpoints/*vecnormalize_{step}_steps.pkl")[0], "rb") as fh:
        vn = pickle.load(fh)
    mean, std, clip = vn.obs_rms.mean, np.sqrt(vn.obs_rms.var + vn.epsilon), float(vn.clip_obs)
    policy = PPO.load(ck, device="cpu").policy

    cfg = load_config("arm/rl/configs/feed.yaml")
    cfg["env"]["stage"] = "present"
    cfg["env"]["handoff"]["runs"] = {"reach": cfg["env"]["handoff"]["runs"]["reach"]}   # reach only; we drive the rest
    e = OpenYAMFeedEnv(cfg)
    rng = np.random.default_rng(0)
    jacp, jacr = np.zeros((3, e.model.nv)), np.zeros((3, e.model.nv))
    min_seat = float(cfg["env"]["min_seat_frac"])

    def grasp_obs():
        st, tg = e.stage, e.target
        e.stage, e.target = "grasp", e.scene.object_pos(e.name).copy()
        o = e._observation()
        e.stage, e.target = st, tg
        return np.clip((o - mean) / std, -clip, clip)

    rows = []
    for ep in range(N):
        e.reset(seed=1000 + ep)
        feeder = Feeder(e)
        phase, anchor, seated, info, hold_ctrl = "pinch", None, 0, {}, None
        out = dict(obj=e.name, picked=False, fed=False, dropped=False, arm_hit="", food_touch=False,
                   size_at_touch=None, size_at_stop=None, gap_mm=None, vmax=0.0, amax=0.0, steps=0, seen_frac=0.0)
        hold, seen_n, feed_n, prev_v = 0, 0, 0, np.zeros(3)
        vs, prev_vs = np.zeros(3), np.zeros(3)
        prev_tcp = e._tcp().copy()
        for i in range(700):
            if phase == "pinch":
                a, _ = policy.predict(grasp_obs()[None], deterministic=True)
                a = np.asarray(a, dtype=np.float32)
            elif phase == "lift":
                a = lift_action(e, anchor, jacp, jacr, info.get("lift_m", 0.0))
            else:
                seen, bearing, size = feeder.face_measurement(rng)
                feed_n += 1
                seen_n += int(seen)
                tcp = e._tcp()
                Rt = e.data.site_xmat[e.tool.site_id].reshape(3, 3)
                tool_axis = Rt @ e.tool.tool_local
                if phase == "retreat":
                    # no perception needed: hold it up, level, facing forward
                    err = PRESENT_POS - tcp
                    a = feeder.action(2.0 * err, desired_frame(e, FORWARD), V_RETREAT)
                    level = float(tool_axis @ FORWARD)
                    if np.linalg.norm(err) < 0.03 and level > 0.97 and seen:
                        phase = "approach"
                elif phase == "approach":
                    if not seen:                                    # lost the face: stop, do not guess
                        a = feeder.action(np.zeros(3), desired_frame(e, FORWARD), V_APPROACH)
                    else:
                        Rc = e.data.cam_xmat[e.wrist_cam].reshape(3, 3)
                        cam = e.data.cam_xpos[e.wrist_cam]
                        # Range to the face from its angular size alone, then where the CLAW AXIS should
                        # appear in the image at that range. The camera sits 6.5 cm off the claw axis, so
                        # aiming the claws down the camera's own ray swung them ~30 deg sideways up close.
                        rng_m = HEAD_R / max(1e-3, np.sin(np.radians(min(size, 170.0)) / 2.0))   # to head centre
                        zc = max(0.04, rng_m - 0.9 * HEAD_R)                                      # to the mouth
                        tcp_c, axis_c = Rc.T @ (tcp - cam), Rc.T @ tool_axis
                        sdist = (-zc - tcp_c[2]) / min(-1e-3, axis_c[2])
                        p = tcp_c + sdist * axis_c
                        expected = np.array([p[0] / zc, p[1] / zc])
                        err_c = (bearing - expected) * zc                                         # metres, camera x/y
                        lateral = Rc[:, 0] * err_c[0] + Rc[:, 1] * err_c[1]
                        lateral -= (lateral @ tool_axis) * tool_axis
                        target_size = STOP_SIZE if STOP_SIZE > 0 else 1e9
                        # slower as the face looms: full speed far out, a crawl for the last few degrees
                        if CALIBRATE:
                            v_in = V_APPROACH * float(np.clip((62.0 - size) / 30.0, 0.15, 1.0))
                        else:
                            v_in = V_APPROACH * float(np.clip((target_size - size) / 18.0, 0.0, 1.0))
                            v_in = max(V_MIN, v_in) if size < target_size else 0.0
                        # do not drive in until the mouth is roughly centred on the claw axis
                        if np.linalg.norm(lateral) > 0.03:
                            v_in *= 0.3
                        v_des = v_in * tool_axis + 1.5 * lateral
                        a = feeder.action(v_des, desired_frame(e, FORWARD), V_APPROACH)
                        if STOP_SIZE > 0 and size >= STOP_SIZE:
                            phase = "hold"
                            out["size_at_stop"] = size
                else:                                               # hold
                    a = feeder.action(np.zeros(3), e.data.site_xmat[e.tool.site_id].reshape(3, 3), V_APPROACH)
                    hold += 1

            if phase != "pinch" and hold_ctrl is not None:
                a = np.array(a, dtype=np.float32, copy=True)
                a[0, 6] = hold_ctrl                                   # gentle, measured squeeze (see below)
            a, _, _ = shield(e, a) if phase in ("pinch", "lift", "retreat") else (a, 0.0, False)
            _, _, _, _, info = e.step(a[0])
            out["steps"] = i + 1

            tcp = e._tcp()
            v = (tcp - prev_tcp) / DT
            if phase in ("retreat", "approach", "hold"):
                out["vmax"] = max(out["vmax"], float(np.linalg.norm(v)))
                vs = 0.7 * vs + 0.3 * v                              # smooth before differentiating
                out["amax"] = max(out["amax"], float(np.linalg.norm(vs - prev_vs) / DT))
                prev_vs = vs.copy()
            prev_tcp, prev_v = tcp.copy(), v

            if os.environ.get("TRACE") and phase == "retreat" and e.name == "apple" and i % 4 == 0:
                Rt_ = e.data.site_xmat[e.tool.site_id].reshape(3, 3); ta_ = Rt_ @ e.tool.tool_local
                rel = e.scene.object_pos(e.name) - e._tcp()
                print(f"     t{i} obj-tcp along claw {1000*rel@ta_:6.1f} mm, off-axis {1000*np.linalg.norm(rel-(rel@ta_)*ta_):5.1f} mm, grip ctrl {e.data.ctrl[e.grip_aid]:.4f}, jaw gap {1000*np.linalg.norm(e.data.geom_xpos[sorted(e.left_pads)].mean(0)-e.data.geom_xpos[sorted(e.right_pads)].mean(0)):5.1f} mm, pinched {info['pinched']}, tool_z {ta_[2]:+.2f}")
            food, arm, part = contacts(e)
            if arm:
                out["arm_hit"] = part
                break
            if phase == "pinch":
                ok = info["pinched"] and info["seat_frac"] >= min_seat
                seated = seated + 1 if ok else 0
                if seated >= 3:
                    phase, anchor = "lift", e._tcp().copy()
                    # Fully shut is 18-20 N on plates 12 mm wide: a sphere touched off-centre is squeezed
                    # out sideways like a watermelon seed (traced: 4 -> 25 mm across the claw while the jaw
                    # gap collapsed). Hold instead at SQUEEZE_M less than the jaw gap measured now -- the
                    # gripper encoder gives that on the real arm -- about 2-3 N per claw.
                    q_now = float(e.data.qpos[e.model.jnt_qposadr[e.model.joint("left_finger").id]])
                    target = max(0.0, q_now - 0.5 * SQUEEZE_M)
                    lo, hi = float(e.grip_range[0]), float(e.grip_range[1])
                    hold_ctrl = float(np.clip(2.0 * (target - lo) / (hi - lo) - 1.0, -1.0, 1.0))
                if info["stage_drift_m"] > 0.09 or i > 250:
                    break
            elif phase == "lift":
                if info["lift_m"] >= 0.10 and info["pinched"]:
                    out["picked"] = True
                    w6, tot = np.zeros(6), 0.0
                    for ci in range(e.data.ncon):
                        c = e.data.contact[ci]
                        pr = {int(c.geom1), int(c.geom2)}
                        if e.obj_gid[e.name] in pr and pr & (e.finger_geoms | e.pad_geoms):
                            mujoco.mj_contactForce(e.model, e.data, ci, w6)
                            tot += abs(float(w6[0]))
                    out["grip_total_n"] = tot
                    out["weight_n"] = 9.81 * float(e.model.body_mass[e.obj_bid[e.name]])
                    phase = "retreat"
                elif not info["pinched"] and info["lift_m"] < 0.02 and i > 320:
                    break
            else:
                # Dropped = the object physically left the claws. (`pinched` flickers when the object
                # shifts from the pad points onto the flat of the claw, while still firmly held.)
                if float(np.linalg.norm(e.scene.object_pos(e.name) - tcp)) > 0.08:
                    out["dropped"] = True
                    break
                if food and not out["food_touch"]:
                    out["food_touch"] = True
                    out["size_at_touch"] = feeder.face_measurement(rng)[2]
                    if CALIBRATE:
                        break
                if phase == "hold" and hold >= 20:
                    break
        # outcome
        obj = e.scene.object_pos(e.name)
        mouth = e.scene.site("mouth")
        out["gap_mm"] = 1000.0 * max(0.0, float(np.linalg.norm(obj - mouth)) - 0.5 * float(e.width[e.name]))
        # split the miss: along the approach (how far he must lean in) and sideways (is it lined up with his mouth)
        rel = mouth - obj
        out["ahead_mm"] = 1000.0 * max(0.0, float(rel @ FORWARD) - 0.5 * float(e.width[e.name]))
        out["side_mm"] = 1000.0 * float(np.linalg.norm(rel - (rel @ FORWARD) * FORWARD))
        out["seen_frac"] = seen_n / max(1, feed_n)
        out["fed"] = bool(out["picked"] and phase == "hold" and not out["dropped"] and not out["arm_hit"]
                          and out["ahead_mm"] <= 60.0 and out["side_mm"] <= 25.0)
        out["phase"] = phase
        if os.environ.get("TRACE"):
            tc = e._tcp(); Rt = e.data.site_xmat[e.tool.site_id].reshape(3, 3); ta = Rt @ e.tool.tool_local
            sn, br, sz = feeder.face_measurement(rng)
            print(f"  ep{ep} {e.name:6s} end phase {phase:8s} steps {out['steps']:3d} tcp {np.round(tc,2)} tool_axis {np.round(ta,2)} face seen {sn} size {sz:.1f}deg gap {out['gap_mm']:.0f}mm dropped {out['dropped']} grip {out.get('grip_total_n', 0):.1f}N weight {out.get('weight_n', 0):.2f}N obj-tcp {1000*np.linalg.norm(e.scene.object_pos(e.name)-tc):.0f}mm camz {e.data.cam_xpos[e.wrist_cam][2]:.2f} tcpz {tc[2]:.2f}")
        rows.append(out)

    picked = [r for r in rows if r["picked"]]
    print(f"grasp policy {os.path.basename(ck)}   episodes {N}   mode {'CALIBRATE' if CALIBRATE else 'FEED'}"
          f"{'' if CALIBRATE else f'  stop size {STOP_SIZE:.3f}'}")
    print(f"picked up {len(picked)}/{N}")
    if CALIBRATE:
        s = np.array([r["size_at_touch"] for r in picked if r["size_at_touch"]])
        print(f"food reached the face in {len(s)}/{len(picked)} pick-ups; arm/claw touched first in "
              f"{sum(bool(r['arm_hit']) for r in picked)} {[r['arm_hit'] for r in picked if r['arm_hit']]}; dropped {sum(r['dropped'] for r in picked)}")
        if len(s):
            print(f"APPARENT FACE SIZE AT FIRST FOOD CONTACT: min {s.min():.3f}  p10 {np.percentile(s,10):.3f}  "
                  f"median {np.median(s):.3f}  max {s.max():.3f}   (angular face width, degrees)")
    else:
        fed = [r for r in picked if r["fed"]]
        print(f"FED (held, stopped on face size alone, food <= 60 mm in front of the mouth and <= 25 mm off to the side, no arm contact, no drop): "
              f"{len(fed)}/{len(picked)} of pick-ups = {len(fed)}/{N} overall")
        print(f"  dropped {sum(r['dropped'] for r in picked)}   arm/claw touched person "
              f"{sum(bool(r['arm_hit']) for r in rows)} {[r['arm_hit'] for r in rows if r['arm_hit']]}   "
              f"food touched face {sum(r['food_touch'] for r in picked)}")
        st = [r for r in picked if r.get("phase") == "hold"]
        if st:
            print(f"  at the stop ({len(st)} episodes): food is {np.mean([r['ahead_mm'] for r in st]):.0f} mm in front of the mouth "
                  f"(min {np.min([r['ahead_mm'] for r in st]):.0f}, max {np.max([r['ahead_mm'] for r in st]):.0f}), "
                  f"{np.mean([r['side_mm'] for r in st]):.0f} mm off to the side (max {np.max([r['side_mm'] for r in st]):.0f})")
        if picked:
            print(f"  food-to-mouth gap at the end: mean {np.mean([r['gap_mm'] for r in picked]):.0f} mm "
                  f"(min {np.min([r['gap_mm'] for r in picked]):.0f}, max {np.max([r['gap_mm'] for r in picked]):.0f})")
    from collections import Counter
    print("  ended in phase:", dict(Counter(r.get("phase") for r in picked)))
    if picked:
        print(f"  peak hand speed {np.max([r['vmax'] for r in picked]):.2f} m/s (cap {V_RETREAT}), "
              f"peak accel {np.max([r['amax'] for r in picked]):.2f} m/s^2 (cap {A_MAX}), "
              f"face in view {100*np.mean([r['seen_frac'] for r in picked]):.0f}% of the feed")


if __name__ == "__main__":
    main()
