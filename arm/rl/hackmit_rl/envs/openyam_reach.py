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


class OpenYAMEnv(gym.Env):
    """One low-dimensional interface shared by reach, grasp, and lift curricula."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, cfg: dict, render_mode: str | None = None):
        super().__init__()
        self.cfg, self.ecfg = cfg, cfg["env"]
        self.stage = self.ecfg.get("stage", "reach")
        self.render_mode = render_mode
        self.scene = YamScene(objects=["apple"])
        self.model, self.data = self.scene.model, self.scene.data
        self.tool = ToolFrame(self.model)
        self.jids = np.asarray([self.model.joint(n).id for n in ARM_JOINTS])
        self.qadr = np.asarray([self.model.jnt_qposadr[j] for j in self.jids])
        self.dadr = np.asarray([self.model.jnt_dofadr[j] for j in self.jids])
        self.lo, self.hi = self.model.jnt_range[self.jids].T.astype(np.float64)
        self.grip_aid = self.model.actuator("gripper").id
        self.grip_range = self.model.actuator_ctrlrange[self.grip_aid].copy()
        self.object_bid = self.model.body("apple").id
        self.object_qadr = self.scene.object_qadr("apple")
        self.table_gid = self.model.geom("table").id
        self.object_gid = self.model.geom("apple").id
        root = self.model.body("arm").id
        self.arm_geom_ids = {i for i in range(self.model.ngeom)
                             if self._is_descendant(int(self.model.geom_bodyid[i]), root)}
        self.home = np.asarray(self.ecfg["home_qpos"], dtype=np.float64)
        self.dt = 1.0 / float(self.ecfg["control_hz"])
        self.n_substeps = int(self.ecfg["physics_substeps"])
        self.model.opt.timestep = self.dt / self.n_substeps
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(23,), dtype=np.float32)
        self.target = np.zeros(3)
        self.filtered_action = np.zeros(7)
        self.previous_action = np.zeros(7)
        self.action_queue: list[np.ndarray] = []
        self.steps = self.hold_steps = self.collision_count = 0
        self.previous_distance = 0.0

    def _is_descendant(self, body: int, root: int) -> bool:
        while body and body != root:
            body = int(self.model.body_parentid[body])
        return body == root

    def _tcp(self) -> np.ndarray:
        rotation = self.data.site_xmat[self.tool.site_id].reshape(3, 3)
        return self.data.site_xpos[self.tool.site_id] + rotation @ self.tool.tcp_local

    def _observation(self) -> np.ndarray:
        grip = 2.0 * (self.data.ctrl[self.grip_aid] - self.grip_range[0]) / np.ptp(self.grip_range) - 1.0
        return np.concatenate((self.data.qpos[self.qadr], self.data.qvel[self.dadr], [grip],
                               self.target - self._tcp(), self.previous_action)).astype(np.float32)

    def _contacts(self) -> tuple[bool, int]:
        grasp_contact, table_collisions = False, 0
        for i in range(self.data.ncon):
            pair = {int(self.data.contact[i].geom1), int(self.data.contact[i].geom2)}
            grasp_contact |= self.object_gid in pair and bool(pair & self.arm_geom_ids)
            table_collisions += int(self.table_gid in pair and bool(pair & self.arm_geom_ids))
        return grasp_contact, table_collisions

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.scene.reset()
        noise = self.np_random.uniform(-0.02, 0.02, 6)
        self.data.qpos[self.qadr] = np.clip(self.home + noise, self.lo + 0.15, self.hi - 0.15)
        self.data.ctrl[:6] = self.data.qpos[self.qadr]
        self.data.ctrl[self.grip_aid] = self.grip_range[1]
        self.filtered_action.fill(0); self.previous_action.fill(0); self.action_queue.clear()
        self.steps = self.hold_steps = self.collision_count = 0
        mujoco.mj_forward(self.model, self.data)
        if self.stage == "reach":
            low = np.asarray(self.ecfg["target_offset_low"])
            high = np.asarray(self.ecfg["target_offset_high"])
            self.target = self._tcp() + self.np_random.uniform(low, high)
        else:
            radius = self.np_random.uniform(*self.ecfg["object_radius_range_m"])
            angle = self.np_random.uniform(*self.ecfg["object_angle_range_rad"])
            point = np.array([radius * np.cos(angle), radius * np.sin(angle), 0.033])
            self.data.qpos[self.object_qadr:self.object_qadr + 3] = point
            self.target = point.copy()
            mujoco.mj_forward(self.model, self.data)
        self.previous_distance = float(np.linalg.norm(self.target - self._tcp()))
        return self._observation(), {"success": False, "distance": self.previous_distance}

    def step(self, action: np.ndarray):
        raw = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        alpha = float(self.ecfg["action_lowpass"])
        self.filtered_action = (1 - alpha) * self.filtered_action + alpha * raw
        self.action_queue.append(self.filtered_action.copy())
        delay = int(self.ecfg.get("action_delay_steps", 0))
        applied = self.action_queue.pop(0) if len(self.action_queue) > delay else np.zeros(7)
        max_delta = float(self.ecfg["max_joint_velocity_rad_s"]) * self.dt
        desired = self.data.ctrl[:6] + np.clip(float(self.ecfg["action_delta_rad"]) * applied[:6],
                                               -max_delta, max_delta)
        self.data.ctrl[:6] = np.clip(desired, self.lo + 0.15, self.hi - 0.15)
        self.data.ctrl[self.grip_aid] = self.grip_range[0] + 0.5 * (applied[6] + 1) * np.ptp(self.grip_range)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        velocity_cap = float(self.ecfg["max_joint_velocity_rad_s"])
        self.data.qvel[self.dadr] = np.clip(self.data.qvel[self.dadr], -velocity_cap, velocity_cap)
        self.steps += 1
        if self.np_random.random() < float(self.ecfg.get("target_jump_probability_per_step", 0.0)):
            self.target += self.np_random.uniform(-0.03, 0.03, 3)

        distance = float(np.linalg.norm(self.target - self._tcp()))
        grasp_contact, table_hits = self._contacts()
        self.collision_count += table_hits
        object_z = float(self.data.xpos[self.object_bid, 2])
        lifted = object_z >= 0.033 + float(self.ecfg["lift_height_m"])
        closed = self.data.ctrl[self.grip_aid] < 0.45 * self.grip_range[1]
        self.hold_steps = self.hold_steps + 1 if lifted and grasp_contact and closed else 0
        success = distance <= float(self.ecfg["success_distance_m"])
        if self.stage == "grasp":
            success = grasp_contact and closed
        elif self.stage == "lift":
            success = self.hold_steps >= round(float(self.ecfg["lift_hold_s"]) / self.dt)

        progress = self.previous_distance - distance
        reward = 10.0 * progress - 0.1 * distance
        reward += float(self.ecfg["grasp_bonus"]) * grasp_contact
        reward += float(self.ecfg["lift_bonus"]) * max(0.0, object_z - 0.033)
        reward -= float(self.ecfg["velocity_penalty"]) * np.square(self.data.qvel[self.dadr]).mean()
        reward -= float(self.ecfg["jerk_penalty"]) * np.square(raw - self.previous_action).mean()
        reward -= float(self.ecfg["collision_penalty"]) * table_hits
        if success:
            reward += float(self.ecfg["success_bonus"])
        self.previous_distance, self.previous_action = distance, raw.copy()
        truncated = self.steps >= int(self.ecfg["episode_steps"])
        info = {"success": bool(success), "is_success": bool(success), "distance": distance,
                "max_joint_velocity": float(np.abs(self.data.qvel[self.dadr]).max()),
                "collision_count": self.collision_count, "object_height": object_z}
        return self._observation(), float(reward), bool(success), bool(truncated), info

    def render(self):
        return self.scene.render("scene_cam", 640)

    def close(self):
        if self.scene._renderer is not None:
            self.scene._renderer.close()
            self.scene._renderer = None


OpenYAMReachEnv = OpenYAMEnv
