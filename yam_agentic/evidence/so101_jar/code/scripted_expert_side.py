"""Planned side-grasp expert for tall upright objects (e.g. a spice jar).

Past the pan joint the SO-101 is a planar 3R chain plus a wrist roll, so
"gripper at a chosen pitch, heading at the object, jaws rolled 90 deg to open
horizontally" is exactly solvable with the 5 arm joints:
    TCP position (3) + gripper pitch (1) + wrist roll (1) = 5 constraints.

Unlike scripted_expert.Expert (closed-loop DLS on the achieved pose), this one
PLANS a whole joint-space trajectory once, from the object's initial pose:
  * every waypoint solved kinematically and required to sit inside the joint
    limits with margin;
  * the Cartesian segments (descend / advance / lift) are subdivided so the
    TCP really follows straight lines;
  * the whole path is collision-checked against the object with the jaws open;
  * pitch is the SMALLEST pitch-down that yields a valid plan (level side grasp
    where the jar is far, tilted closer in, straight down as a last resort);
  * the approach line is shifted along the jaw axis so the FIXED jaw clears the
    object (the calibrated TCP sits 2 cm from the fixed jaw; a 5.1 cm jar needs
    2.55 cm + clearance).
act() then just plays the plan back, so the commands are ABSOLUTE joint targets
that do not depend on tracking error: good for open-loop replay on the real arm
and for imitation labels (no "pose-hugging" stall trap). Only the gripper close
is reactive: it closes until both jaws touch, then squeezes a little.

API (same as Expert): SideExpert(env); .reset(); ctrl = .act(); .phase; .done
Env knobs: SIDE_PITCH (auto | degrees), SIDE_STANDOFF (m), SIDE_ZFRAC,
SIDE_SQUEEZE (rad past first contact), SIDE_CLEAR (m jaw clearance), SIDE_SPEED,
SIDE_GRIP_OPEN (rad, jaw opening during the approach).
"""
import os
import sys

sys.path.insert(0, "/Users/adipu/so101Sim/rl/native")
import mujoco
import numpy as np
from scripted_expert import Expert

GRIP_OPEN = float(os.environ.get("SIDE_GRIP_OPEN", "1.75"))   # jaw opening during the approach (rad); smaller = narrower sweep
GRIP_MIN = -0.17
PITCH = os.environ.get("SIDE_PITCH", "auto")
PITCH_CANDIDATES = (0, 15, 30, 90)      # deg below horizontal. 45-75 are skipped on purpose: slanted jaw
                                        # faces touch a free-standing cylinder at different heights and
                                        # the squeeze tips it over (seen at 45 deg); 0-30 and 90 pinch squarely.
STANDOFF = float(os.environ.get("SIDE_STANDOFF", "0.08"))
Z_FRAC = float(os.environ.get("SIDE_ZFRAC", "0.50"))
SQUEEZE = float(os.environ.get("SIDE_SQUEEZE", "0.25"))
CLEAR = float(os.environ.get("SIDE_CLEAR", "0.008"))
SPEED = float(os.environ.get("SIDE_SPEED", "1.0"))  # global speed scale
ROLL_CANDS = (-np.pi / 2, np.pi / 2)   # jaws open horizontally either way; -90 deg points the
                                       # camera-mount bracket UP (at +90 it grounds out on the table)
HIGH_CANDS = (0.08, 0.05, 0.02)        # retract height above the pre-grasp, first feasible wins
HALF_BOX = 0.02                 # half width of the box the TCP was calibrated on (fixed-jaw offset)
LIMIT_MARGIN = 0.05             # rad
LIFT_DZ = 0.10
V_FREE, V_SLOW = 0.12, 0.05     # m/s on Cartesian segments
W_JOINT = 1.2                   # rad/s on the joint-space transfer
DWELL_S = float(os.environ.get("SIDE_DWELL", "0.5"))   # s at pre-grasp / grasp / after closing
GRIP_RATE_S = 2.0               # rad/s while closing


class SideExpert:
    def __init__(self, env):
        self.env, self.m = env, env.model
        m = self.m
        base = Expert(env)                                   # calibrated TCP + jaw-opening axis
        self.tcp_local, self.site_body, self.jaw_local = base.tcp_local, base.site_body, base.jaw_local
        dd = mujoco.MjData(m)
        dd.qpos[:] = m.qpos0
        mujoco.mj_forward(m, dd)
        R0 = dd.site_xmat[env._grip_site].reshape(3, 3)
        self.point_local = R0.T @ np.array([1.0, 0.0, 0.0])  # at the L pose the gripper points +x
        pan = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_pan")
        self.pan_xy = dd.xanchor[pan][:2].copy()
        self.dt = env._n_sub * m.opt.timestep
        self.dwell_ticks = max(1, int(round(DWELL_S / self.dt)))
        self.grip_rate = GRIP_RATE_S * self.dt
        grip_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        jaw_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")
        self._grip_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == grip_body}
        self._jaw_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == jaw_body}
        self.lo = m.actuator_ctrlrange[:5, 0].copy()
        self.hi = m.actuator_ctrlrange[:5, 1].copy()
        self._sc = mujoco.MjData(m)
        self.reset()

    # ------------------------------------------------------------------ kinematics
    def reset(self):
        self.phase, self.done = "plan", False
        self.traj = None
        self.k = 0
        self.dwell = 0
        self.grip_cmd = GRIP_OPEN
        self.q_contact = None
        self.reachable, self.pitch_deg, self.why = False, None, ""
        self._roll = ROLL_CANDS[0]

    def _obj_dims(self):
        g = self.env._box_geom
        if int(self.m.geom_type[g]) == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return float(self.m.geom_size[g][0]), 2.0 * float(self.m.geom_size[g][1])
        return float(max(self.m.geom_size[g][0], self.m.geom_size[g][1])), 2.0 * float(self.m.geom_size[g][2])

    def _fk(self, q5, grip=GRIP_OPEN, with_object=None):
        """Forward kinematics on scratch data. with_object: xyz to place the object at
        (for collision checks) or None to park it far away."""
        env, sc = self.env, self._sc
        sc.qpos[:] = env.data.qpos
        sc.qpos[env._arm_qadr[:5]] = q5
        sc.qpos[env._arm_qadr[5]] = grip
        a = env._box_qadr
        if with_object is None:
            sc.qpos[a:a + 3] = (0.9, 0.7, 0.3)
        else:
            sc.qpos[a:a + 3] = with_object
            sc.qpos[a + 3:a + 7] = (1, 0, 0, 0)
        mujoco.mj_forward(self.m, sc)
        R = sc.site_xmat[env._grip_site].reshape(3, 3)
        return sc.site_xpos[env._grip_site] + R @ self.tcp_local, R

    def _solve(self, p, a_star, n_perp, seed, roll=None):
        """Square DLS IK for TCP position + pitch + roll. Returns (q5, ok)."""
        env, m, sc = self.env, self.m, self._sc
        q = np.array(seed, float)
        roll = self._roll if roll is None else roll
        cols = env._arm_vadr[:5]
        ok = False
        for _ in range(300):
            tcp, R = self._fk(q)
            a = R @ self.point_local
            e_pos = p - tcp
            e_pitch = float(np.cross(a, a_star) @ n_perp)
            e_roll = roll - q[4]
            if np.linalg.norm(e_pos) < 5e-4 and abs(e_pitch) < 5e-3 and abs(e_roll) < 5e-3:
                ok = True
                break
            jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
            mujoco.mj_jac(m, sc, jacp, jacr, tcp, self.site_body)
            J = np.vstack([jacp[:, cols], (n_perp @ jacr[:, cols])[None, :], np.array([[0, 0, 0, 0, 1.0]])])
            err = np.concatenate([e_pos, [e_pitch], [e_roll]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-5 * np.eye(5), err)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        inside = bool(np.all(q > self.lo + LIMIT_MARGIN) and np.all(q < self.hi - LIMIT_MARGIN))
        return q, ok and inside

    def _line(self, p0, p1, a_star, n_perp, seed, step=0.005):
        n = max(1, int(np.ceil(np.linalg.norm(p1 - p0) / step)))
        qs, q = [], seed
        for i in range(1, n + 1):
            q, ok = self._solve(p0 + (p1 - p0) * i / n, a_star, n_perp, q)
            if not ok:
                return None
            qs.append(q)
        return qs

    def _collides(self, qs, obj_xyz, grip=GRIP_OPEN):
        """True if any arm geom penetrates the object OR the table along the path."""
        d, m = self._sc, self.m
        box_body = self.env._box_body
        for q in qs:
            self._fk(q, grip, with_object=obj_xyz)
            for c in range(d.ncon):
                if d.contact.dist[c] >= -1e-4:
                    continue
                g1, g2 = int(d.contact.geom1[c]), int(d.contact.geom2[c])
                b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
                if box_body in (b1, b2):                           # ANY geom of the object (cylinder, base plate, ...)
                    if (b2 if b1 == box_body else b1) != 0:        # object vs arm (object vs floor is fine)
                        return True
                elif 0 in (b1, b2):                                # arm vs table
                    return True
        return False

    # ------------------------------------------------------------------ planning
    def _plan(self):
        env = self.env
        obj = env.data.xpos[env._box_body].copy()
        radius, height = self._obj_dims()
        bottom = float(obj[2]) - height / 2.0                      # cylinder's underside: > 0 when it stands on a base plate
        bottom = bottom if bottom > 0.002 else 0.0                 # (plain jar / cube: exactly the old behaviour)
        zg = max(0.035, bottom + Z_FRAC * height)
        h0 = obj[:2] - self.pan_xy
        h0 /= np.linalg.norm(h0)
        q_start = env.data.qpos[env._arm_qadr[:5]].copy()
        lateral = max(0.0, radius - HALF_BOX) + CLEAR              # shift so the FIXED jaw clears the object
        cands = PITCH_CANDIDATES if PITCH == "auto" else (float(PITCH),)
        plan, why = None, []
        for pd in cands:
          for roll in ROLL_CANDS:
            self._roll = roll
            tag = f"{pd}/{int(np.degrees(roll)):+d}"
            dl = np.deg2rad(pd)
            h = h0.copy()
            for _ in range(3):                                     # refine heading/jaw axis at the grasp pose
                a_star = np.array([np.cos(dl) * h[0], np.cos(dl) * h[1], -np.sin(dl)])
                n_perp = np.array([-h[1], h[0], 0.0])
                qg, _ok = self._solve(np.array([obj[0], obj[1], zg]), a_star, n_perp, q_start)
                _, R = self._fk(qg)
                ag = (R @ self.point_local)[:2]
                u = R @ self.jaw_local                             # fixed jaw -> moving jaw
                u[2] = 0.0
                u /= (np.linalg.norm(u) + 1e-9)
                if pd < 80:
                    h = ag / (np.linalg.norm(ag) + 1e-9)
            a_star = np.array([np.cos(dl) * h[0], np.cos(dl) * h[1], -np.sin(dl)])
            n_perp = np.array([-h[1], h[0], 0.0])
            p_grasp = np.array([obj[0], obj[1], zg]) - lateral * u
            p_pre = p_grasp - a_star * STANDOFF
            p_lift = p_grasp + np.array([0.0, 0.0, LIFT_DZ])
            found = False
            for high in HIGH_CANDS:
                p_high = p_pre + np.array([0.0, 0.0, high])
                q_high, ok_h = None, False
                pan0 = float(np.arctan2(h0[1], h0[0])) * -1.0          # shoulder_pan axis is -z
                for seed in (q_start, np.array([pan0, 0.0, 0.0, np.pi / 2, roll]),
                             np.array([pan0, -0.6, 1.0, 1.1, roll]), np.array([pan0, 0.6, 0.4, 0.5, roll])):
                    q_high, ok_h = self._solve(p_high, a_star, n_perp, seed)
                    if ok_h:
                        break
                if not ok_h:
                    why.append(f"{tag}:high{high}"); continue
                seg_down = self._line(p_high, p_pre, a_star, n_perp, q_high)
                if seg_down is None:
                    why.append(f"{tag}:descend"); continue
                seg_adv = self._line(p_pre, p_grasp, a_star, n_perp, seg_down[-1])
                if seg_adv is None:
                    why.append(f"{tag}:advance"); break
                seg_lift = self._line(p_grasp, p_lift, a_star, n_perp, seg_adv[-1])
                if seg_lift is None:
                    why.append(f"{tag}:lift"); break
                n_tr = max(2, int(np.ceil(np.abs(q_high - q_start).max() / 0.03)))
                seg_transfer = [q_start + (q_high - q_start) * i / n_tr for i in range(1, n_tr + 1)]
                if self._collides(seg_transfer + seg_down + seg_adv, obj):
                    why.append(f"{tag}:collision"); continue
                plan = dict(transfer=seg_transfer, descend=seg_down, advance=seg_adv, lift=seg_lift)
                self.reachable, self.pitch_deg, self.roll_deg, found = True, pd, int(np.degrees(roll)), True
                break
            if found:
                break
          if plan is not None:
            break
        self.why = ",".join(why)
        if plan is None:
            return None

        def ticks_joint(seg, q_from, w):
            out, q0 = [], q_from
            for q1 in seg:
                n = max(1, int(np.ceil(np.abs(q1 - q0).max() / (w * SPEED * self.dt))))
                out += [q0 + (q1 - q0) * i / n for i in range(1, n + 1)]
                q0 = q1
            return out

        def ticks_line(seg, q_from, v, step=0.005):
            n = max(1, int(np.ceil(step / (v * SPEED * self.dt))))
            out, q0 = [], q_from
            for q1 in seg:
                out += [q0 + (q1 - q0) * i / n for i in range(1, n + 1)]
                q0 = q1
            return out

        return {
            "transfer": ticks_joint(plan["transfer"], q_start, W_JOINT),
            "descend": ticks_line(plan["descend"], plan["transfer"][-1], V_FREE),
            "advance": ticks_line(plan["advance"], plan["descend"][-1], V_SLOW),
            "lift": ticks_line(plan["lift"], plan["advance"][-1], V_SLOW),
        }

    def _both_jaws_touch(self):
        d, box = self.env.data, self.env._box_geom
        hit_g = hit_j = False
        for c in range(d.ncon):
            g1, g2 = int(d.contact.geom1[c]), int(d.contact.geom2[c])
            if box not in (g1, g2):
                continue
            o = g2 if g1 == box else g1
            hit_g |= o in self._grip_geoms
            hit_j |= o in self._jaw_geoms
        return hit_g and hit_j

    # ------------------------------------------------------------------ playback
    def act(self):
        env = self.env
        ctrl = np.zeros(6)
        if self.phase == "plan":
            self.traj = self._plan()
            if self.traj is None:                                  # unreachable: hold still
                self.phase, self.done = "unreachable", True
                self.q_hold = env.data.qpos[env._arm_qadr[:5]].copy()
            else:
                self.phase, self.k = "transfer", 0
        if self.phase == "unreachable":
            ctrl[:5], ctrl[5] = self.q_hold, GRIP_OPEN
            return ctrl
        ph = self.phase
        if ph in ("transfer", "descend", "advance", "lift"):
            seg = self.traj[ph]
            q = seg[min(self.k, len(seg) - 1)]
            self.k += 1
            if self.k >= len(seg):
                self.dwell += 1
                need = 0 if ph == "transfer" else self.dwell_ticks
                if self.dwell > need:
                    self.dwell, self.k = 0, 0
                    self.phase = {"transfer": "descend", "descend": "advance",
                                  "advance": "grasp", "lift": "hold"}[ph]
            self.q_hold = q
        elif ph == "grasp":
            q = self.q_hold
            if self.q_contact is None:
                self.grip_cmd = max(GRIP_MIN, self.grip_cmd - self.grip_rate)
                gq = float(env.data.qpos[env._arm_qadr[5]])
                if self._both_jaws_touch():
                    self.q_contact = gq
                    self.grip_cmd = max(GRIP_MIN, gq - SQUEEZE)
                elif self.grip_cmd <= GRIP_MIN + 1e-6:
                    self.q_contact = GRIP_MIN                      # closed on nothing
            else:
                self.dwell += 1
                if self.dwell > self.dwell_ticks:
                    self.dwell, self.k, self.phase = 0, 0, "lift"
        else:  # hold
            q = self.q_hold
            self.done = True
        ctrl[:5], ctrl[5] = q, self.grip_cmd
        return ctrl
