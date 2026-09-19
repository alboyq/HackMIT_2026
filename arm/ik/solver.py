"""Damped-least-squares IK for the 6-DoF YAM, plus the frame bookkeeping the grasp planner needs.

The SO-101 expert (rl/native/scripted_expert_side.py) could solve its arm in closed form
because past the pan joint it is a planar 3R chain — 5 joints, 5 constraints. The YAM has a
real 3-DoF wrist, so full 6-DoF pose IK is both possible and simpler to write generically.

Everything is expressed at `grasp_site`, and the two axes the planner cares about are
measured off the model rather than assumed:

  tool_local  unit vector, in the grasp_site frame, pointing from the wrist out past the jaws
  jaw_local   unit vector, in the grasp_site frame, along which the jaws open

so a grasp is specified as "put the TCP here, with the tool axis along d and the jaws opening
along j", and `rot_from_axes` turns that into a target rotation matrix.
"""
import mujoco
import numpy as np


def _unit(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("zero-length axis")
    return np.asarray(v, float) / n


class ToolFrame:
    """Measured geometry of the gripper.

    `grasp_site` is NOT between the pads on this model — at qpos0 it sits 7.4 cm beyond them
    along the tool axis and 4.4 cm above. So the TCP is derived from the pad geoms themselves
    (the `sphere_collision` spheres inside lf_down / rf_down) and everything the planner does
    is expressed at that point.
    """

    def __init__(self, model, site="grasp_site", left="lf_down", right="rf_down"):
        self.model = model
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        if self.site_id < 0:
            raise RuntimeError(f"no site {site}")
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_6")
        self.grip_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
        self.grip_range = model.actuator_ctrlrange[self.grip_act].copy()
        self._jid = model.actuator_trnid[self.grip_act, 0]
        rj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_finger")
        self._qadr_l = model.jnt_qposadr[self._jid]
        self._qadr_r = model.jnt_qposadr[rj]
        lb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, left)
        rb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, right)
        # the pads: the small spheres that actually touch the object
        self._pad_l = [g for g in range(model.ngeom) if model.geom_bodyid[g] == lb
                       and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
        self._pad_r = [g for g in range(model.ngeom) if model.geom_bodyid[g] == rb
                       and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
        if not self._pad_l or not self._pad_r:
            raise RuntimeError("no pad spheres found on the fingers")

        d = mujoco.MjData(model)
        pl, pr = self._pads(d, 0.5 * float(self.grip_range[1]))
        tcp = 0.5 * (pl + pr)
        R = d.site_xmat[self.site_id].reshape(3, 3)
        p = d.site_xpos[self.site_id]
        self.tcp_local = R.T @ (tcp - p)                     # TCP as an offset from grasp_site
        self.tool_local = _unit(R.T @ (tcp - d.xpos[self.body_id]))
        self.jaw_local = _unit(R.T @ (pr - pl))
        self.jaw_local = _unit(self.jaw_local - self.tool_local * (self.jaw_local @ self.tool_local))
        self.opening = {tag: float(np.linalg.norm(np.subtract(*reversed(self._pads(d, c)))))
                        for tag, c in (("closed", self.grip_range[0]), ("open", self.grip_range[1]))}
        self.max_width = self.opening["open"]

    def _pads(self, d, ctrl):
        d.qpos[:] = self.model.qpos0
        d.qpos[self._qadr_l] = ctrl
        d.qpos[self._qadr_r] = -ctrl                         # the mimic equality, applied by hand
        mujoco.mj_forward(self.model, d)
        return (d.geom_xpos[self._pad_l].mean(axis=0).copy(),
                d.geom_xpos[self._pad_r].mean(axis=0).copy())

    def ctrl_for_width(self, width):
        """Gripper command that opens the pads to `width` metres (linear in this mechanism)."""
        lo, hi = self.grip_range
        f = np.clip((width - self.opening["closed"]) /
                    max(1e-6, self.opening["open"] - self.opening["closed"]), 0.0, 1.0)
        return float(lo + f * (hi - lo))

    def rot_from_axes(self, tool_dir, jaw_dir):
        """Target rotation R such that R @ tool_local = tool_dir and R @ jaw_local ~ jaw_dir."""
        t = _unit(tool_dir)
        j = np.asarray(jaw_dir, float)
        j = j - t * (j @ t)
        if np.linalg.norm(j) < 1e-8:                       # degenerate: pick any perpendicular
            j = np.cross(t, [0, 0, 1.0])
            if np.linalg.norm(j) < 1e-8:
                j = np.cross(t, [1.0, 0, 0])
        j = _unit(j)
        # source and target orthonormal triads, R maps one to the other
        A = np.stack([self.tool_local, self.jaw_local, np.cross(self.tool_local, self.jaw_local)], axis=1)
        B = np.stack([t, j, np.cross(t, j)], axis=1)
        return B @ A.T


class ArmIK:
    def __init__(self, model, tool: ToolFrame = None, joints=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")):
        self.model = model
        self.tool = tool or ToolFrame(model)
        self.jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joints]
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.jids])
        self.dofadr = np.array([model.jnt_dofadr[j] for j in self.jids])
        self.lo = np.array([model.jnt_range[j][0] for j in self.jids])
        self.hi = np.array([model.jnt_range[j][1] for j in self.jids])
        self.n = len(self.jids)
        self._d = mujoco.MjData(model)

    def fk(self, q, qpos_full=None):
        """TCP position (between the pads) and gripper rotation for arm configuration q."""
        d = self._d
        d.qpos[:] = self.model.qpos0 if qpos_full is None else qpos_full
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, d)
        R = d.site_xmat[self.tool.site_id].reshape(3, 3).copy()
        p = d.site_xpos[self.tool.site_id] + R @ self.tool.tcp_local
        return p.copy(), R

    def solve(self, target_pos, target_R=None, q_init=None, iters=250, tol_p=5e-4, tol_r=1e-2,
              damping=0.06, margin=0.02, rest=None, rest_w=0.01, qpos_full=None, target_dir=None):
        """Returns (q, ok, err_pos, err_rot). Joint limits are enforced with `margin` rad of
        headroom, which keeps the planner off the stops where the real arm loses authority.

        `target_R` constrains the full orientation (6 constraints). `target_dir` constrains only
        the TOOL AXIS and leaves roll about it free (5 constraints) — which is what most grasps
        actually need, and what this wrist can actually do: measured on this model, pointing the
        tool straight down while also demanding a particular jaw yaw is frequently infeasible
        (joint4/5 are +-90 deg, joint6 +-120 deg), while tool-down with free roll is reachable
        across a 0.0-0.62 m radius band.
        """
        m, d = self.model, self._d
        d.qpos[:] = self.model.qpos0 if qpos_full is None else qpos_full
        q = np.array(q_init if q_init is not None else d.qpos[self.qadr], float)
        lo, hi = self.lo + margin, self.hi - margin
        q = np.clip(q, lo, hi)
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        err_p = err_r = np.inf
        for _ in range(iters):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            R = d.site_xmat[self.tool.site_id].reshape(3, 3)
            p = d.site_xpos[self.tool.site_id] + R @ self.tool.tcp_local
            e = np.zeros(6)
            e[:3] = target_pos - p
            if target_R is not None:
                qc = np.zeros(4); qd = np.zeros(4); dv = np.zeros(3)
                mujoco.mju_mat2Quat(qc, R.flatten())
                mujoco.mju_mat2Quat(qd, np.ascontiguousarray(target_R).flatten())
                mujoco.mju_subQuat(dv, qd, qc)             # local-frame rotation vector
                e[3:] = R @ dv                             # -> world
            elif target_dir is not None:
                a = R @ self.tool.tool_local               # where the tool points now
                c = np.cross(a, target_dir)
                sn, cs = float(np.linalg.norm(c)), float(a @ target_dir)
                if sn > 1e-9:
                    e[3:] = (c / sn) * np.arctan2(sn, cs)
                elif cs < 0:                               # exactly antiparallel: any perpendicular
                    e[3:] = np.pi * _unit(np.cross(a, [0, 0, 1.0] if abs(a[2]) < 0.9 else [1.0, 0, 0]))
            err_p, err_r = float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))
            if err_p < tol_p and ((target_R is None and target_dir is None) or err_r < tol_r):
                return q, True, err_p, err_r
            # Jacobian of the TCP point (offset from the site) attached to link_6
            mujoco.mj_jac(m, d, jacp, jacr, p, self.tool.body_id)
            J = np.vstack([jacp[:, self.dofadr], jacr[:, self.dofadr]])
            if target_R is None and target_dir is None:
                J, e = J[:3], e[:3]
            # Increase damping as the smallest singular value collapses. This keeps
            # near-singular wrist poses from turning millimetre errors into large jumps.
            sigma_min = float(np.linalg.svd(J, compute_uv=False)[-1])
            adaptive = damping + max(0.0, 0.08 - sigma_min) * 2.0
            lam2 = adaptive ** 2
            # Joint-limit weighting: movement toward a nearby stop is progressively
            # suppressed while movement back toward the middle remains available.
            span = np.maximum(hi - lo, 1e-6)
            normalized = 2.0 * (q - 0.5 * (lo + hi)) / span
            weights = 1.0 + 8.0 * np.power(np.abs(normalized), 6)
            Winv = np.diag(1.0 / weights)
            dq = Winv @ J.T @ np.linalg.solve(J @ Winv @ J.T + lam2 * np.eye(J.shape[0]), e)
            if rest is not None:                            # nullspace pull toward a posture
                N = np.eye(self.n) - Winv @ J.T @ np.linalg.solve(
                    J @ Winv @ J.T + lam2 * np.eye(J.shape[0]), J
                )
                dq = dq + N @ (rest_w * (np.asarray(rest) - q))
            s = float(np.linalg.norm(dq))
            if s > 0.25:
                dq *= 0.25 / s
            q = np.clip(q + dq, lo, hi)
        return q, False, err_p, err_r


    def seeds_for(self, target_pos, extra=None):
        """Deterministic seed set: pan the shoulder at the target, then a spread of
        shoulder/elbow postures. Cheap insurance against DLS local minima."""
        pan = float(np.arctan2(target_pos[1], target_pos[0]))
        pan = float(np.clip(pan, self.lo[0] + 0.05, self.hi[0] - 0.05))
        out = [] if extra is None else [np.asarray(extra, float)]
        for j2, j3 in ((1.0, 1.2), (1.4, 1.8), (0.7, 0.8), (1.8, 2.4), (1.2, 2.0)):
            for j5 in (-0.6, 0.0, 0.6):
                out.append(np.array([pan, j2, j3, 0.0, j5, 0.0]))
        return out

    def solve_best(self, target_pos, tool_dir, jaw_dir=None, q_init=None, **kw):
        """Position + tool direction, trying several seeds; then, if `jaw_dir` is given,
        attempt to also pin the roll (a parallel gripper is 180-degree symmetric, so both
        signs are tried). Returns (q, ok, dict).

        The jaw constraint is a PREFERENCE: if no full-orientation solution exists the
        5-DoF one is returned with jaw_locked=False, which is the right answer for spheres
        and upright cylinders and an accepted approximation for boxes.
        """
        best = None
        for seed in self.seeds_for(target_pos, q_init):
            q, ok, ep, er = self.solve(target_pos, q_init=seed, target_dir=tool_dir, **kw)
            if ok:
                best = (q, ep, er)
                break
            if best is None or ep < best[1]:
                best = (q, ep, er)
        if best is None:
            return None, False, {}
        q5, ep, er = best
        ok5 = ep < kw.get("tol_p", 5e-4) * 2 + 5e-4 and er < 0.05
        info = dict(err_pos=ep, err_rot=er, jaw_locked=False,
                    reason="" if ok5 else "no seed converged within position/orientation tolerance")
        if jaw_dir is not None and ok5:
            R5 = self.fk(q5)[1]
            for sgn in (1.0, -1.0):
                Rd = self.tool.rot_from_axes(tool_dir, sgn * np.asarray(jaw_dir, float))
                q6, ok6, ep6, er6 = self.solve(target_pos, Rd, q_init=q5, **kw)
                if ok6:
                    return q6, True, dict(err_pos=ep6, err_rot=er6, jaw_locked=True)
        return q5, ok5, info


def look_at_xyaxes(pos, target, up=(0, 0, 1)):
    """MuJoCo camera xyaxes string. A camera looks along -z of its own frame, so with
    z = normalize(pos - target), x = up x z, y = z x x."""
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    z = _unit(pos - target)
    x = np.cross(np.asarray(up, float), z)
    if np.linalg.norm(x) < 1e-8:
        x = np.cross([1.0, 0, 0], z)
    x = _unit(x)
    y = np.cross(z, x)
    return " ".join(f"{v:.5f}" for v in np.concatenate([x, y]))
