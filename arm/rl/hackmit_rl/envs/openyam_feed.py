"""Pick one of four objects and present it to the seated user.

Differences from `openyam_reach.py`, all of them deliberate:

* **Four objects, one chosen per episode**, with its identity in the observation as a one-hot.
  The policy has to condition on WHICH object it was asked for, which is the task the demo needs.
* **A `present` stage.** Reach/grasp/lift end with the object in the air; this carries it to a
  staging point 15 cm short of the mouth and holds it there. Per HANDOFF.md the arm stops at the
  staging point on purpose -- closing that last gap is a separate, explicitly confirmed action,
  so the reward never asks the policy to cross it.
* **The velocity cap is enforced, not faked.** `openyam_reach.py` clipped `data.qvel` AFTER
  mj_step, which rewrites physics state and makes any velocity assertion circular. Here the cap
  applies to the commanded delta only (the sole thing we control), and a breach is reported in
  `info` and penalised, never edited away.
* **No target teleportation.** The old env jumped the target up to 5.2 cm mid-episode against a
  2 cm success tolerance, so ~30% of episodes were unwinnable through no fault of the policy.
* **A virtual wall around the head**, per the plan's safety section.
* **Objects are re-sized every episode** and the chosen object's width is observed, so the policy
  cannot memorise one geometry. Scale is clamped so the object still fits the pads.
* **Grip force is scored both ways.** Squeezing past `max_grip_force_n` is penalised (a soft
  object would be crushed) and so is letting go while carrying. The band between them is the
  only place the policy is rewarded, which is what "firm but not crushing" has to mean when the
  simulator has no notion of bruising.

Observation (31):  qpos 6 | qvel 6 | grip 1 | target-tcp 3 | mouth-tcp 3 | one-hot 4 | width 1 | prev action 7
Action (7):        six joint deltas + gripper

The layout is fixed across every stage so one stage's checkpoint seeds the next.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from arm.ik.scene import YamScene
from arm.ik.solver import ToolFrame

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
OBJECTS = ("apple", "mug", "marker", "block")
STAGES = ("reach", "grasp", "lift", "present")


class OpenYAMFeedEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, cfg: dict, render_mode: str | None = None):
        super().__init__()
        self.cfg, self.ecfg = cfg, cfg["env"]
        self.stage = self.ecfg.get("stage", "reach")
        if self.stage not in STAGES:
            raise ValueError(f"unknown stage {self.stage!r}; expected one of {STAGES}")
        self.render_mode = render_mode

        self.scene = YamScene(objects=list(OBJECTS))
        self.model, self.data = self.scene.model, self.scene.data
        self.tool = ToolFrame(self.model)

        self.jids = np.asarray([self.model.joint(n).id for n in ARM_JOINTS])
        self.qadr = np.asarray([self.model.jnt_qposadr[j] for j in self.jids])
        self.dadr = np.asarray([self.model.jnt_dofadr[j] for j in self.jids])
        self.lo, self.hi = self.model.jnt_range[self.jids].T.astype(np.float64)
        self.grip_aid = self.model.actuator("gripper").id
        self.grip_range = self.model.actuator_ctrlrange[self.grip_aid].copy()

        self.obj_bid = {n: self.model.body(n).id for n in OBJECTS}
        self.obj_gid = {n: self.model.geom(n).id for n in OBJECTS}
        self.obj_qadr = {n: self.scene.object_qadr(n) for n in OBJECTS}
        self.table_gid = self.model.geom("table").id

        # Nominal geometry, kept so every episode can re-scale from the same baseline.
        self.base_size = {n: self.model.geom_size[self.obj_gid[n]].copy() for n in OBJECTS}
        self.base_mass = {n: float(self.model.body_mass[self.obj_bid[n]]) for n in OBJECTS}
        self.base_rest_z = {n: float(self.scene.spec(n).rest_z) for n in OBJECTS}
        self.base_width = {n: float(self.scene.spec(n).width) for n in OBJECTS}
        self.rest_z = dict(self.base_rest_z)
        self.width = dict(self.base_width)
        self.geom_body = {i: int(self.model.geom_bodyid[i]) for i in range(self.model.ngeom)}
        self.left_pads = set(self.tool._pad_l)
        self.right_pads = set(self.tool._pad_r)
        self.pad_geoms = self.left_pads | self.right_pads
        self._wrench = np.zeros(6)
        root = self.model.body("arm").id
        self.arm_geoms = {i for i in range(self.model.ngeom)
                          if self._descends(int(self.model.geom_bodyid[i]), root)}

        self.home = np.asarray(self.ecfg["home_qpos"], dtype=np.float64)
        self.dt = 1.0 / float(self.ecfg["control_hz"])
        self.n_substeps = int(self.ecfg["physics_substeps"])
        self.model.opt.timestep = self.dt / self.n_substeps

        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(31,), dtype=np.float32)

        self.name = OBJECTS[0]
        self.onehot = np.zeros(len(OBJECTS))
        self.target = np.zeros(3)
        self.mouth = np.zeros(3)
        self.stage_point = np.zeros(3)
        self.filtered = np.zeros(7)
        self.prev_action = np.zeros(7)
        self.steps = self.hold_steps = self.collisions = self.wall_hits = self.crush_steps = 0
        self.self_collisions = self.curl_steps = 0
        self.prev_dist = 0.0
        self.object_start = np.zeros(3)
        self.peak_velocity = 0.0
        self.peak_grip_force = 0.0

    # ------------------------------------------------------------------ helpers
    def _descends(self, body: int, root: int) -> bool:
        while body and body != root:
            body = int(self.model.body_parentid[body])
        return body == root

    def _tcp(self) -> np.ndarray:
        R = self.data.site_xmat[self.tool.site_id].reshape(3, 3)
        return self.data.site_xpos[self.tool.site_id] + R @ self.tool.tcp_local

    def _contacts(self):
        """(touching, pinched, table_hits).

        `pinched` requires BOTH pads on the object, which is what distinguishes holding it from
        batting it: a single pad in contact is a swipe, and an earlier version of this env
        rewarded exactly that.
        """
        touching, table_hits, self_hits = False, 0, 0
        left = right = False
        gid = self.obj_gid[self.name]
        for i in range(self.data.ncon):
            pair = {int(self.data.contact[i].geom1), int(self.data.contact[i].geom2)}
            if gid in pair:
                touching |= bool(pair & self.arm_geoms)
                left |= bool(pair & self.left_pads)
                right |= bool(pair & self.right_pads)
            table_hits += int(self.table_gid in pair and bool(pair & self.arm_geoms))
            g1, g2 = int(self.data.contact[i].geom1), int(self.data.contact[i].geom2)
            if g1 in self.arm_geoms and g2 in self.arm_geoms:
                b1, b2 = self.geom_body[g1], self.geom_body[g2]
                adjacent = (b1 == b2
                            or int(self.model.body_parentid[b1]) == b2
                            or int(self.model.body_parentid[b2]) == b1)
                self_hits += int(not adjacent)
        return touching, (left and right), table_hits, self_hits

    def _grip_force(self) -> float:
        """Largest normal force any gripper pad is putting into the held object, in newtons."""
        gid = self.obj_gid[self.name]
        peak = 0.0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            pair = {int(contact.geom1), int(contact.geom2)}
            if gid in pair and pair & self.pad_geoms:
                mujoco.mj_contactForce(self.model, self.data, i, self._wrench)
                peak = max(peak, abs(float(self._wrench[0])))
        return peak

    def _observation(self) -> np.ndarray:
        grip = 2.0 * (self.data.ctrl[self.grip_aid] - self.grip_range[0]) / np.ptp(self.grip_range) - 1.0
        tcp = self._tcp()
        return np.concatenate((self.data.qpos[self.qadr], self.data.qvel[self.dadr], [grip],
                               self.target - tcp, self.mouth - tcp, self.onehot,
                               [self.width[self.name]], self.prev_action)).astype(np.float32)

    # ------------------------------------------------------------------ episode
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.scene.reset()

        self.name = OBJECTS[int(self.np_random.integers(len(OBJECTS)))]
        self.onehot = np.eye(len(OBJECTS))[OBJECTS.index(self.name)]

        noise = self.np_random.uniform(-0.02, 0.02, 6)
        self.data.qpos[self.qadr] = np.clip(self.home + noise, self.lo + 0.15, self.hi - 0.15)
        self.data.ctrl[:6] = self.data.qpos[self.qadr]
        self.data.ctrl[self.grip_aid] = self.grip_range[1]

        # Re-size every object, then spread them around the top-down band
        # (MEASUREMENTS.md: 0.15-0.50 m). Scale is clamped so the widest axis still fits
        # between the pads with clearance -- an object that cannot be grasped is not a
        # lesson, it is noise.
        lo_s, hi_s = self.ecfg["object_scale_range"]
        clearance = float(self.ecfg["pad_clearance_m"])
        usable = max(1e-3, float(self.tool.max_width) - clearance)
        for obj in OBJECTS:
            scale = float(self.np_random.uniform(lo_s, min(hi_s, usable / self.base_width[obj])))
            self.model.geom_size[self.obj_gid[obj]] = self.base_size[obj] * scale
            self.model.body_mass[self.obj_bid[obj]] = self.base_mass[obj] * scale ** 3
            self.rest_z[obj] = self.base_rest_z[obj] * scale
            self.width[obj] = self.base_width[obj] * scale

        angles = self.np_random.permutation(
            np.linspace(*self.ecfg["object_angle_range_rad"], len(OBJECTS)))
        for obj, angle in zip(OBJECTS, angles):
            radius = self.np_random.uniform(*self.ecfg["object_radius_range_m"])
            adr = self.obj_qadr[obj]
            self.data.qpos[adr:adr + 3] = [radius * np.cos(angle), radius * np.sin(angle),
                                           self.rest_z[obj]]
        mujoco.mj_forward(self.model, self.data)

        self.mouth = self.scene.site("mouth")
        approach = self.mouth - np.array([0.0, 0.0, self.mouth[2]])
        approach = approach / max(1e-9, np.linalg.norm(approach))       # base -> mouth, horizontal
        self.stage_point = self.mouth - approach * float(self.ecfg["staging_m"])

        if self.stage == "present":
            self.target = self.stage_point.copy()
        else:
            self.target = self.scene.object_pos(self.name).copy()

        self.filtered.fill(0)
        self.prev_action.fill(0)
        self.steps = self.hold_steps = self.collisions = self.wall_hits = self.crush_steps = 0
        self.self_collisions = self.curl_steps = 0
        self.peak_velocity = 0.0
        self.peak_grip_force = 0.0
        self.object_start = self.scene.object_pos(self.name).copy()
        self.prev_dist = float(np.linalg.norm(self.target - self._tcp()))
        return self._observation(), {"success": False, "object": self.name}

    def step(self, action: np.ndarray):
        raw = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        alpha = float(self.ecfg["action_lowpass"])
        self.filtered = (1 - alpha) * self.filtered + alpha * raw

        # The ONLY velocity enforcement: cap the commanded joint delta. Physics is never edited.
        cap = float(self.ecfg["max_joint_velocity_rad_s"])
        max_delta = cap * self.dt
        desired = self.data.ctrl[:6] + np.clip(
            float(self.ecfg["action_delta_rad"]) * self.filtered[:6], -max_delta, max_delta)
        self.data.ctrl[:6] = np.clip(desired, self.lo + 0.15, self.hi - 0.15)
        self.data.ctrl[self.grip_aid] = (self.grip_range[0]
                                         + 0.5 * (self.filtered[6] + 1) * np.ptp(self.grip_range))
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self.steps += 1
        measured_velocity = float(np.abs(self.data.qvel[self.dadr]).max())
        self.peak_velocity = max(self.peak_velocity, measured_velocity)

        tcp = self._tcp()
        obj_pos = self.scene.object_pos(self.name)
        self.mouth = self.scene.site("mouth")
        if self.stage == "present":
            self.target = self.stage_point
        else:
            self.target = obj_pos

        distance = float(np.linalg.norm(self.target - tcp))
        touching, pinched, table_hits, self_hits = self._contacts()
        held = touching
        self.collisions += table_hits
        self.self_collisions += self_hits
        grip_force = self._grip_force()
        crush_limit = float(self.ecfg["max_grip_force_n"])
        crushing = grip_force > crush_limit
        self.crush_steps += int(crushing)
        self.peak_grip_force = max(self.peak_grip_force, grip_force)
        closed = self.data.ctrl[self.grip_aid] < 0.45 * self.grip_range[1]
        lifted = float(obj_pos[2]) >= self.rest_z[self.name] + float(self.ecfg["lift_height_m"])
        carrying = pinched and closed and lifted

        # Virtual wall around the head: never reward getting closer than this.
        wall = float(self.ecfg["head_radius_m"])
        head_gap = float(np.linalg.norm(self.mouth - tcp))
        breached = head_gap < wall
        self.wall_hits += int(breached)

        if self.stage == "reach":
            success = distance <= float(self.ecfg["success_distance_m"])
        elif self.stage == "grasp":
            settled = float(np.linalg.norm(obj_pos[:2] - self.object_start[:2])) <= float(
                self.ecfg["grasp_max_displacement_m"])
            self.hold_steps = self.hold_steps + 1 if (pinched and closed and settled) else 0
            success = self.hold_steps >= round(float(self.ecfg["grasp_hold_s"]) / self.dt)
        elif self.stage == "lift":
            self.hold_steps = self.hold_steps + 1 if carrying else 0
            success = self.hold_steps >= round(float(self.ecfg["lift_hold_s"]) / self.dt)
        else:
            near = float(np.linalg.norm(self.stage_point - obj_pos)) <= float(
                self.ecfg["present_tolerance_m"])
            ok = carrying and near and not breached and not crushing
            self.hold_steps = self.hold_steps + 1 if ok else 0
            success = self.hold_steps >= round(float(self.ecfg["present_hold_s"]) / self.dt)

        progress = self.prev_dist - distance
        reward = 10.0 * progress - 0.1 * distance
        reward += float(self.ecfg["grasp_bonus"]) * held
        reward += float(self.ecfg["lift_bonus"]) * max(0.0, float(obj_pos[2]) - self.rest_z[self.name])
        if self.stage == "present":
            reward += float(self.ecfg["carry_bonus"]) * carrying
            if held and not carrying and self.steps > 10:
                reward -= float(self.ecfg["drop_penalty"])      # dropping the food is the failure mode
        reward -= float(self.ecfg["velocity_penalty"]) * np.square(self.data.qvel[self.dadr]).mean()
        reward -= float(self.ecfg["jerk_penalty"]) * np.square(raw - self.prev_action).mean()
        # Curling up: links touching, the wrist tucked back over the base, or joints
        # pinned against their stops. Shaped, not terminal.
        tcp_radius = float(np.linalg.norm(tcp[:2]))
        curled = tcp_radius < float(self.ecfg["min_tcp_radius_m"])
        self.curl_steps += int(curled)
        span = np.maximum(self.hi - self.lo, 1e-6)
        normalized = 2.0 * (self.data.qpos[self.qadr] - 0.5 * (self.lo + self.hi)) / span
        limit_strain = float(np.mean(np.power(np.abs(normalized), 8)))
        reward -= float(self.ecfg["self_collision_penalty"]) * self_hits
        reward -= float(self.ecfg["curl_penalty"]) * curled
        reward -= float(self.ecfg["joint_limit_penalty"]) * limit_strain
        reward -= float(self.ecfg["collision_penalty"]) * table_hits
        displaced = float(np.linalg.norm(obj_pos[:2] - self.object_start[:2]))
        if not carrying and displaced > float(self.ecfg["grasp_max_displacement_m"]):
            reward -= float(self.ecfg["knock_penalty"]) * min(1.0, displaced)
        reward -= float(self.ecfg["wall_penalty"]) * breached
        # Firm but not crushing: penalise force past the limit, and reward the band below it
        # only while actually holding, so the policy cannot earn it by hovering with open jaws.
        if crushing:
            reward -= float(self.ecfg["crush_penalty"]) * (grip_force - crush_limit) / crush_limit
        elif held and grip_force >= float(self.ecfg["min_grip_force_n"]):
            reward += float(self.ecfg["grip_band_bonus"])
        if measured_velocity > cap * 1.5:
            reward -= float(self.ecfg["velocity_breach_penalty"])
        if success:
            reward += float(self.ecfg["success_bonus"])

        self.prev_dist, self.prev_action = distance, raw.copy()
        truncated = self.steps >= int(self.ecfg["episode_steps"])
        info = {"success": bool(success), "is_success": bool(success), "object": self.name,
                "distance": distance, "max_joint_velocity": measured_velocity,
                "peak_joint_velocity": self.peak_velocity, "velocity_cap": cap,
                "collision_count": self.collisions, "wall_hits": self.wall_hits,
                "grip_force_n": grip_force, "peak_grip_force_n": self.peak_grip_force,
                "crush_steps": self.crush_steps, "object_width_m": self.width[self.name],
                "object_height": float(obj_pos[2]), "carrying": bool(carrying), "pinched": bool(pinched),
                "object_displaced_m": displaced, "self_collisions": self.self_collisions,
                "curl_steps": self.curl_steps, "tcp_radius_m": tcp_radius}
        return self._observation(), float(reward), bool(success), bool(truncated), info

    def render(self):
        return self.scene.render("scene_cam", 640)

    def close(self):
        if self.scene._renderer is not None:
            self.scene._renderer.close()
            self.scene._renderer = None
