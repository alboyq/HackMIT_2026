"""Planned pick-and-deliver expert for the YAM.

Same shape as rl/native/scripted_expert_side.py: the whole joint-space trajectory is PLANNED
up front from the object's pose, so what comes out is a list of ABSOLUTE joint targets that do
not depend on tracking error. That is what makes it safe to replay open-loop on the real arm
and usable as imitation labels (no pose-hugging stall trap). Only the gripper close is
reactive.

What is different here is the strategy choice, because the YAM's envelope is not the SO-101's
(measured, see yam_scene docstring):

    r <= 0.50 m   tool straight down          top-down grasp, jaw yaw free (or pinned for boxes)
    r <= 0.65 m   tool tilted 30 deg          fallback for objects further out
    r >= 0.65 m   tool level                  side grasp; the arm cannot fold back any closer

So `strategy_for` picks by radius rather than by object shape, and the object's shape only
decides the grasp HEIGHT and whether the jaw yaw matters. A parallel gripper whose pads open
to 82.8 mm also cannot grasp everything: `check_graspable` is the feasibility gate the agent
layer calls before promising anything.

Env knobs: YAM_APPROACH (m above the grasp point), YAM_LIFT (m), YAM_SQUEEZE (m of pad
overlap), YAM_CLEAR (m of jaw clearance when open), YAM_SPEED (joint rad/s), YAM_STAGING (m
from the mouth).
"""
import os

import mujoco
import numpy as np

from yam_ik import ArmIK, ToolFrame
from yam_scene import HOME_Q

APPROACH = float(os.environ.get("YAM_APPROACH", "0.10"))
LIFT = float(os.environ.get("YAM_LIFT", "0.14"))
# How far PAST contact to command the jaws. This is a position-controlled parallel gripper:
# the object stops the pads and the grip force is kp * (commanded - achieved). Measured on this
# model, width = 0.8 mm + 2 * ctrl and the finger kp is 100, so force per pad ~= 50 N/m * the
# commanded overlap. 8 mm of overlap is 0.4 N and a 150 g apple slips straight out on the lift;
# 60 mm is ~3 N and holds. Real YAM linear grippers force-limit in firmware, which is the
# hardware equivalent of the same trick.
SQUEEZE = float(os.environ.get("YAM_SQUEEZE", "0.060"))
CLEAR = float(os.environ.get("YAM_CLEAR", "0.004"))    # per side, when the jaws are open
SPEED = float(os.environ.get("YAM_SPEED", "1.1"))          # rad/s on free transfers
SLOW = float(os.environ.get("YAM_SLOW", "0.30"))           # rad/s on Cartesian segments: at
                                                           # 1.1 rad/s the descent runs at 0.28 m/s
                                                           # and the servo lags 16 mm CROSS-track,
                                                           # which drops a finger onto the object
CARRY = float(os.environ.get("YAM_CARRY", "0.6"))           # rad/s while carrying: the 0.3 limit is about
                                                           # descent ACCURACY; a held payload tolerates more
STAGING = float(os.environ.get("YAM_STAGING", "0.15"))     # m short of the mouth
FINGER_REACH = float(os.environ.get("YAM_FINGER_REACH", "0.062"))   # m of finger below the gripper housing
# On linear_4310 the TCP (grasp_site) is the midpoint of the fingertips' DISTAL facets, so putting
# it at the object's centre pinches with the very tips and the object slips. Push the TCP this far
# further along the tool axis, which seats the object in the middle of the pad faces instead.
GRASP_DEPTH = float(os.environ.get("YAM_GRASP_DEPTH", "0.0"))
CART_STEP = 0.008                                          # m between IK'd Cartesian samples
DWELL = float(os.environ.get("YAM_DWELL", "0.25"))         # s at ordinary waypoints
PRE_DWELL = float(os.environ.get("YAM_PRE_DWELL", "0.6"))  # s at the pre-grasp: the descent is
                                                           # open-loop, so cross-track error must
                                                           # settle BEFORE it starts
SETTLE = float(os.environ.get("YAM_SETTLE", "0.40"))       # s to let the servo catch up before closing
CLOSE_DWELL = float(os.environ.get("YAM_CLOSE_DWELL", "0.60"))


class Waypoint:
    __slots__ = ("q", "grip", "hold", "tag", "speed")

    def __init__(self, q, grip, hold=0.0, tag="", speed=None):
        self.q = np.asarray(q, float)
        self.grip = float(grip)
        self.hold = float(hold)          # SECONDS to dwell, not sim steps: the position
        self.tag = tag                   # servo lags the command by ~100 ms and the grasp
                                         # must not close while the arm is still catching up
        self.speed = speed               # rad/s for the move INTO this waypoint (None = SPEED)

    def __repr__(self):
        return f"<{self.tag} q={np.round(np.degrees(self.q), 1)} grip={self.grip:.3f}>"


class Plan:
    def __init__(self, waypoints=None, ok=True, why=""):
        self.waypoints = waypoints or []
        self.ok = ok
        self.why = why

    def __bool__(self):
        return self.ok and bool(self.waypoints)

    def __repr__(self):
        return (f"<Plan ok={self.ok} {len(self.waypoints)} wp"
                + (f" why={self.why!r}" if self.why else "") + ">")


class Expert:
    """Plans grasps and deliveries. Holds no episode state beyond the last plan, so the agent
    layer can call it repeatedly and cheaply."""

    def __init__(self, scene):
        self.s = scene
        self.m = scene.model
        self.tool = ToolFrame(self.m)
        self.ik = ArmIK(self.m, self.tool)
        self.home_q = np.array(HOME_Q, float)
        self._sc = mujoco.MjData(self.m)
        arm_bodies = self._subtree("arm")
        self.arm_geoms = {g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] in arm_bodies}
        self.user_geoms = {self._gid(n) for n in ("head", "nose", "torso", "hand")
                           if self._gid(n) >= 0}

    # ---------------------------------------------------------------- helpers
    def _gid(self, n):
        return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, n)

    def _subtree(self, root):
        rid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, root)
        out = set()
        for b in range(self.m.nbody):
            x = b
            while x != 0 and x != rid:
                x = self.m.body_parentid[x]
            if x == rid:
                out.add(b)
        return out

    def _collides(self, q, grip, ignore=()):
        """True if configuration q puts the arm into the user or an object it should not touch."""
        sc = self._sc
        sc.qpos[:] = self.s.data.qpos
        sc.qpos[self.s.arm_qadr] = q
        sc.ctrl[:] = 0
        mujoco.mj_forward(self.m, sc)
        ignore = set(ignore)
        for i in range(sc.ncon):
            c = sc.contact[i]
            a, b = int(c.geom1), int(c.geom2)
            if a in ignore or b in ignore:
                continue
            if (a in self.arm_geoms) != (b in self.arm_geoms):
                other = b if a in self.arm_geoms else a
                if other in self.user_geoms and c.dist < -1e-3:
                    return True
        return False

    def check_graspable(self, name):
        """The gate the agent layer calls before promising anything. Returns (ok, why)."""
        o = self.s.spec(name)
        if o.width + 2 * CLEAR > self.tool.max_width:
            return False, (f"{name} is {o.width * 1000:.0f} mm across; the pads open to "
                           f"{self.tool.max_width * 1000:.0f} mm, so it will not fit")
        p = self.s.object_pos(name)
        r = float(np.linalg.norm(p[:2]))
        if r > 0.75:
            return False, f"{name} is {r:.2f} m out; the arm tops out at 0.75 m"
        return True, ""

    # ---------------------------------------------------------------- strategy
    def strategy_for(self, pos):
        """(name, tool_dir) by radius — the envelope, not the object shape."""
        r = float(np.linalg.norm(pos[:2]))
        az = float(np.arctan2(pos[1], pos[0]))
        out = np.array([np.cos(az), np.sin(az), 0.0])
        if r <= 0.50:
            return "top", np.array([0.0, 0.0, -1.0])
        if r <= 0.65:
            t = np.radians(30)
            return "tilt", np.sin(t) * out + np.cos(t) * np.array([0, 0, -1.0])
        return "side", out

    def grasp_pose(self, name):
        """Where the TCP must be, which way the tool points, and how the jaws should be
        oriented for `name` as it sits right now."""
        o = self.s.spec(name)
        p = self.s.object_pos(name)
        strategy, tool_dir = self.strategy_for(p)
        if strategy == "top":
            # grasp at mid-height, but (a) never so low the pads hit the table and (b) never deeper than
            # the fingers reach: the housing sits ~6 cm above the pads, so a tall can is taken near its top
            z = max(0.022, min(p[2], o.rest_z), o.height - FINGER_REACH)
            tcp = np.array([p[0], p[1], z])
        else:
            tcp = np.array([p[0], p[1], max(0.025, o.rest_z * 0.9)])
        tcp = tcp + tool_dir * GRASP_DEPTH                     # seat the object mid-pad, not at the tips
        tcp[2] = max(tcp[2], 0.012)                            # ...but keep the fingertips off the table
        jaw_dir = None
        if o.kind == "box":
            # square-on to a face; solve_best tries both signs, and 90 deg round is equivalent
            jaw_dir = np.array([0.0, 1.0, 0.0])
        return tcp, tool_dir, jaw_dir, strategy

    # ---------------------------------------------------------------- planning
    def _cart_segment(self, q0, p_from, p_to, tool_dir, grip, tag, jaw_dir=None, speed=None,
                      tool_dir_to=None, min_steps=2):
        """IK a straight TCP line, seeding each solve from the previous one.

        `tool_dir_to` turns the tool from `tool_dir` to it along the way, so a payload picked
        up top-down can be rotated to a level presentation gradually instead of being snapped
        through 90 degrees in one waypoint (which flings it out of a 3 N pinch).
        """
        n = max(min_steps, int(np.ceil(np.linalg.norm(np.asarray(p_to) - np.asarray(p_from)) / CART_STEP)))
        a = np.asarray(tool_dir, float)
        b = a if tool_dir_to is None else np.asarray(tool_dir_to, float)
        wps, q = [], np.array(q0, float)
        for i in range(1, n + 1):
            f = i / n
            p = p_from + (np.asarray(p_to) - np.asarray(p_from)) * f
            d = a * (1 - f) + b * f                       # good enough for < 180 deg turns
            nd = np.linalg.norm(d)
            d = b if nd < 1e-8 else d / nd
            qs, ok, info = self.ik.solve_best(p, d, jaw_dir=jaw_dir, q_init=q)
            if not ok:
                return None, f"{tag}: no IK at {np.round(p, 3)} ({info.get('err_pos', 9) * 1000:.0f} mm off)"
            q = qs
            wps.append(Waypoint(q, grip, tag=tag, speed=speed if speed is not None else SLOW))
        return wps, ""

    def plan_pick(self, name):
        ok, why = self.check_graspable(name)
        if not ok:
            return Plan(ok=False, why=why)
        o = self.s.spec(name)
        tcp, tool_dir, jaw_dir, strategy = self.grasp_pose(name)
        # Approach fully open, always. Opening to just "object + a few mm" leaves no room for
        # the servo's cross-track lag and a finger lands on top of the object instead of beside
        # it — measured: the 50 mm block was approached 16 mm off-centre with a 58 mm opening.
        g_open = float(self.tool.grip_range[1])
        g_shut = self.tool.ctrl_for_width(max(0.0, o.width - SQUEEZE))
        # The pre-grasp standoff only has to leave room for the descent, so when the full
        # APPROACH is out of reach a shorter one is a real grasp, not a compromise. The TCP sits
        # 30 mm further down the tool axis on linear_4310 than on the crank, which pushes the
        # wrist that much higher here and takes the low, close objects out of reach at 100 mm.
        pre = q_pre = None
        first_err = None
        for standoff in (APPROACH, 0.08, 0.06, 0.045):
            cand = tcp - tool_dir * standoff
            q, ok, info = self.ik.solve_best(cand, tool_dir, jaw_dir=jaw_dir, q_init=self.home_q)
            if first_err is None:
                first_err = info.get("err_pos", 9)
            if ok:
                pre, q_pre = cand, q
                break
        if q_pre is None:
            return Plan(ok=False, why=f"pre-grasp unreachable for {name} "
                                      f"({first_err * 1000:.0f} mm off, {strategy})")
        wps = [Waypoint(q_pre, g_open, hold=PRE_DWELL, tag="pre")]
        seg, why = self._cart_segment(q_pre, pre, tcp, tool_dir, g_open, "descend", jaw_dir)
        if seg is None:
            return Plan(ok=False, why=why)
        wps += seg
        q_at = wps[-1].q
        wps.append(Waypoint(q_at, g_open, hold=SETTLE, tag="settle"))
        wps.append(Waypoint(q_at, g_shut, hold=CLOSE_DWELL, tag="close"))
        # Straight up is NOT always reachable: tool-down at r = 0.44 m runs out of wrist by
        # z ~ 0.10 m. So try progressively gentler lifts, each also pulled in toward the base,
        # which is where the payload has to end up anyway.
        seg = None
        r = float(np.linalg.norm(tcp[:2]))
        for dz in (LIFT, 0.10, 0.07, 0.05):
            for r_to in (min(r, 0.33), min(r, 0.40), r):
                lift_to = np.array([tcp[0] * r_to / r, tcp[1] * r_to / r, tcp[2] + dz])
                seg, why = self._cart_segment(q_at, tcp, lift_to, tool_dir, g_shut, "lift", jaw_dir)
                if seg is not None:
                    break
            if seg is not None:
                break
        if seg is None:
            return Plan(ok=False, why=f"cannot lift {name}: {why}")
        wps += seg
        wps[-1].hold = DWELL
        plan = Plan(wps)
        plan.strategy, plan.grip_open, plan.grip_shut = strategy, g_open, g_shut
        return plan

    def plan_present(self, target, approach_dir=None, standoff=STAGING, q_from=None, grip=None):
        """Bring whatever is held to `standoff` metres short of `target`, tool pointing at it.

        The arm stops at the staging point. It does not close the last few centimetres on a
        person's face: that gap is the safety margin, and crossing it is a separate, explicitly
        confirmed action.
        """
        target = np.asarray(target, float)
        q0 = self.home_q if q_from is None else np.asarray(q_from, float)
        if approach_dir is None:
            approach_dir = target - np.array([0, 0, target[2]])     # from the base, horizontally
            approach_dir[2] = 0
            n = np.linalg.norm(approach_dir)
            approach_dir = approach_dir / n if n > 1e-6 else np.array([1.0, 0, 0])
        grip = self.tool.grip_range[0] if grip is None else grip
        stage = target - approach_dir * standoff
        p0, R0 = self.ik.fk(q0, qpos_full=self.s.data.qpos)
        held_dir = R0 @ self.tool.tool_local          # keep the grasp orientation until clear
        # Pin the jaw axis too. With roll left free the 5-DoF solve is allowed to spin the
        # wrist about the tool axis as the tool turns, which throws the payload out of the
        # pinch; solve_best falls back to free roll if a pose cannot be locked.
        held_jaw = R0 @ self.tool.jaw_local
        # up first (payload never dragged across the table), then turn to the presentation
        # direction while translating: both at Cartesian speed.
        # Only a SHORT raise while still tool-down: holding the tool vertical runs out of
        # wrist by z ~ 0.22 m, so the rest of the height is gained during the carry, once the
        # tool has turned level (level at z = 0.38 m is reachable across the whole table).
        via = np.array([p0[0], p0[1], min(max(p0[2], p0[2] + 0.06), 0.20)])
        wps = []
        if via[2] > p0[2] + 1e-3:
            seg, why = self._cart_segment(q0, p0, via, held_dir, grip, "raise", jaw_dir=held_jaw, speed=CARRY)
            if seg is not None:
                wps += seg
                q0 = wps[-1].q
            else:
                via = p0
        else:
            via = p0
        # How far to turn the payload. Rotating a top-down grasp all the way to "tool pointing
        # at the mouth" is what drops the apple: measured, the pad contacts fall away past
        # ~70 deg from vertical and a sphere simply rolls out from between two flat pads.
        # Nothing requires the tool to point AT the mouth — the payload just has to arrive in
        # front of it — and keeping the jaws tilted down also keeps them off the person's face.
        # So take the SMALLEST turn that reaches the staging pose.
        seg = None
        for f in (0.0, 0.25, 0.4, 0.55, 0.7, 1.0):
            d_to = held_dir * (1 - f) + approach_dir * f
            d_to = d_to / np.linalg.norm(d_to)
            q_try, ok_try, _ = self.ik.solve_best(stage, d_to, q_init=q0)
            if not ok_try or self._collides(q_try, grip):
                continue
            seg, why = self._cart_segment(q0, via, stage, held_dir, grip, "carry",
                                          jaw_dir=held_jaw, tool_dir_to=d_to, min_steps=24, speed=CARRY)
            if seg is not None:
                turn = np.degrees(np.arccos(np.clip(held_dir @ d_to, -1, 1)))
                break
        if seg is None:
            return Plan(ok=False, why=f"cannot carry to the staging point: {why}")
        wps += seg
        wps.append(Waypoint(wps[-1].q, grip, hold=SETTLE, tag="stage"))
        if self._collides(wps[-1].q, grip):
            return Plan(ok=False, why="staging pose intersects the user")
        plan = Plan(wps)
        plan.stage_pos = stage
        plan.turn_deg = float(turn)
        return plan

    def plan_home(self, q_from=None, grip=None):
        grip = self.tool.grip_range[1] if grip is None else grip
        return Plan([Waypoint(self.home_q, grip, hold=DWELL, tag="home")])

    # ---------------------------------------------------------------- execution
    def trajectory(self, plan, dt):
        """Expand a plan into a list of (7,) absolute ctrl vectors at the sim's control rate."""
        out = []
        q = self.s.q_arm.copy()
        grip = float(self.s.data.ctrl[6]) if self.m.nu > 6 else 0.0
        for wp in plan.waypoints:
            step = (wp.speed if wp.speed is not None else SPEED) * dt
            while True:
                d = wp.q - q
                n = float(np.abs(d).max())
                if n <= step:
                    q = wp.q.copy()
                    break
                q = q + d * (step / n)
                out.append(np.concatenate([q, [grip]]))
            gd = wp.grip - grip
            gstep = 0.06 * dt
            while abs(gd) > gstep:
                grip += np.sign(gd) * gstep
                gd = wp.grip - grip
                out.append(np.concatenate([q, [grip]]))
            grip = wp.grip
            for _ in range(max(1, int(round(wp.hold / dt)))):
                out.append(np.concatenate([q, [grip]]))
        return out
