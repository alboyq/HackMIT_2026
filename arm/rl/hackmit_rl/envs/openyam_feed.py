"""Pick one of three objects and present it to the seated user.

Differences from `openyam_reach.py`, all of them deliberate:

* **Three objects, one chosen per episode**, with its identity in the observation as a one-hot.
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

Observation (33):  qpos 6 | qvel 6 | grip 1 | target-tcp 3 | mouth-tcp 3 | mouth-object 3 | one-hot 3 | width 1 | prev action 7
Action (7):        six joint deltas + gripper

The layout is fixed across every stage so one stage's checkpoint seeds the next.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from arm.ik.scene import YamScene
from arm.ik.solver import ToolFrame

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
OBJECTS = ("apple", "mug", "block")   # the marker is too thin for these pads
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
        # Stage hand-off: a later stage starts from the state the FROZEN earlier stages leave
        # the arm in, not from home. That is the point of splitting the task: grasp should
        # learn to grasp, not to re-learn reaching, and it must cope with the pose reach
        # actually delivers rather than an idealised one.
        self.final_stage = self.stage
        self.handoff = dict(self.ecfg.get("handoff") or {})
        self._priors = None          # loaded lazily, inside the worker process
        self.prior_hook = None       # viewers set this to draw the frozen stages too
        self.handoff_ok = True
        self.handoff_steps = 0
        self.prev_lift = 0.0
        self.stage_object_start = np.zeros(3)
        self.tip_ahead = None        # fingertip distance past the TCP, measured off the model
        self.was_lifted = False
        self.pinch_paid = False
        self._bank = []              # recent hand-off states, see reset()

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
        # Every geom on either finger. The jaws meeting is the gripper working, not a
        # self-collision -- penalising it taught the policy to hold the jaws open.
        finger_roots = [self.model.body(n).id for n in ("lf_down", "rf_down")]
        self.finger_geoms = {i for i in range(self.model.ngeom)
                             if any(self._descends(int(self.model.geom_bodyid[i]), r)
                                    for r in finger_roots)}
        self.left_pads = set(self.tool._pad_l)
        self.right_pads = set(self.tool._pad_r)
        self.pad_geoms = self.left_pads | self.right_pads
        self._wrench = np.zeros(6)
        root = self.model.body("arm").id
        self.arm_geoms = {i for i in range(self.model.ngeom)
                          if self._descends(int(self.model.geom_bodyid[i]), root)}

        # Per-joint headroom. The wrist has a much smaller range than the shoulder, so a
        # flat margin costs it proportionally far more travel.
        self.margin = np.asarray(self.ecfg["joint_margin_rad"], dtype=np.float64)
        self.ctrl_lo = self.lo + self.margin
        self.ctrl_hi = self.hi - self.margin
        self.limit_weights = np.asarray(self.ecfg["joint_limit_weights"], dtype=np.float64)
        self.home = np.asarray(self.ecfg["home_qpos"], dtype=np.float64)
        self.dt = 1.0 / float(self.ecfg["control_hz"])
        self.n_substeps = int(self.ecfg["physics_substeps"])
        self.model.opt.timestep = self.dt / self.n_substeps

        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(33,), dtype=np.float32)

        self.name = OBJECTS[0]
        self.onehot = np.zeros(len(OBJECTS))
        self.target = np.zeros(3)
        self.mouth = np.zeros(3)
        self.stage_point = np.zeros(3)
        self.filtered = np.zeros(7)
        self.prev_action = np.zeros(7)
        self.steps = self.hold_steps = self.collisions = self.wall_hits = self.crush_steps = 0
        self.self_collisions = self.curl_steps = 0
        self.knocked = self.was_pinched = False
        self.phase = 0
        self.settle_calm = 0.0
        self.prev_dist = 0.0
        self.object_start = np.zeros(3)
        self.obj_bias = np.zeros(3)
        self.mouth_bias = np.zeros(3)
        self.width_bias = 0.0
        self.prev_tcp = np.zeros(3)
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
                both_fingers = g1 in self.finger_geoms and g2 in self.finger_geoms
                self_hits += int(not adjacent and not both_fingers)
        return touching, (left and right), table_hits, self_hits

    def _pad_distance_to(self, point) -> float:
        """Distance from `point` to the nearest gripper pad."""
        return float(min(np.linalg.norm(point - self.data.geom_xpos[g]) for g in self.pad_geoms))

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

    def _sample_perception_bias(self):
        """Per-episode calibration error. Fixed for the whole episode, because a miscalibrated
        camera is wrong in the same direction all run -- it does not average out."""
        n = self.cfg.get("perception_noise", {})
        if not n.get("enabled", False):
            self.obj_bias = np.zeros(3)
            self.mouth_bias = np.zeros(3)
            self.width_bias = 0.0
            return
        self.obj_bias = self.np_random.normal(0, float(n["object_bias_m"]), 3)
        self.mouth_bias = self.np_random.normal(0, float(n["mouth_bias_m"]), 3)
        self.width_bias = float(self.np_random.normal(0, float(n["width_rel"])))

    def _perceived(self, true_point, bias, jitter_key):
        """Eye-in-hand error shrinks with range: the same pixel error is fewer millimetres when
        the lens is close, and the table-plane depth assumption is least wrong up close."""
        n = self.cfg.get("perception_noise", {})
        if not n.get("enabled", False):
            return true_point
        rng_m = float(np.linalg.norm(true_point - self._tcp()))
        scale = float(np.clip(rng_m / float(n["reference_range_m"]),
                              float(n["min_scale"]), float(n["max_scale"])))
        return (true_point + bias * scale
                + self.np_random.normal(0, float(n[jitter_key]) * scale, 3))

    def _observation(self) -> np.ndarray:
        grip = 2.0 * (self.data.ctrl[self.grip_aid] - self.grip_range[0]) / np.ptp(self.grip_range) - 1.0
        tcp = self._tcp()
        # What a camera would report, not what the simulator knows.
        obj_seen = self._perceived(self.scene.object_pos(self.name), self.obj_bias, "object_jitter_m")
        mouth_seen = self._perceived(self.mouth, self.mouth_bias, "mouth_jitter_m")
        # The goal is derived from whichever of those the stage is chasing, so its error is
        # consistent with the thing it was measured from.
        if self.stage == "present":
            target_seen = self.target + (mouth_seen - self.mouth)
        else:
            target_seen = self.target + (obj_seen - self.scene.object_pos(self.name))
        width_seen = float(self.width[self.name]) * (1.0 + self.width_bias)
        return np.concatenate((self.data.qpos[self.qadr], self.data.qvel[self.dadr], [grip],
                               target_seen - tcp, mouth_seen - tcp, mouth_seen - obj_seen,
                               self.onehot, [width_seen],
                               self.prev_action)).astype(np.float32)

    # ------------------------------------------------------------------ episode
    def _reset_scene(self) -> None:
        self.scene.reset()

        self.name = OBJECTS[int(self.np_random.integers(len(OBJECTS)))]
        self.onehot = np.eye(len(OBJECTS))[OBJECTS.index(self.name)]

        noise = self.np_random.uniform(-0.02, 0.02, 6)
        self.data.qpos[self.qadr] = np.clip(self.home + noise, self.ctrl_lo, self.ctrl_hi)
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

        # One angular SLOT per object, shuffled, with jitter inside the slot. Structural
        # separation: rejection sampling could not honour the spacing inside a narrow arc and
        # quietly fell back to overlapping positions.
        lo_a, hi_a = self.ecfg["object_angle_range_rad"]
        margin = float(self.ecfg["object_spacing_m"])
        n_obj = len(OBJECTS)
        edges = np.linspace(float(lo_a), float(hi_a), n_obj + 1)
        order = self.np_random.permutation(n_obj)
        placed: list[tuple[str, np.ndarray]] = []
        for obj, slot in zip(OBJECTS, order):
            lo_s, hi_s = edges[slot], edges[slot + 1]
            pad = 0.12 * (hi_s - lo_s)                      # keep off the slot boundaries
            angle = self.np_random.uniform(lo_s + pad, hi_s - pad)
            radius = self.np_random.uniform(*self.ecfg["object_radius_range_m"])
            point = np.array([radius * np.cos(angle), radius * np.sin(angle)])
            # If a slot neighbour still lands too close, push this one out along its own ray.
            for _ in range(12):
                tight = [other for name, other in placed
                         if float(np.linalg.norm(point - other))
                         < 0.5 * (self.width[obj] + self.width[name]) + margin]
                if not tight:
                    break
                radius = min(radius + 0.02, float(self.ecfg["object_radius_range_m"][1]))
                point = np.array([radius * np.cos(angle), radius * np.sin(angle)])
            placed.append((obj, point))
            adr = self.obj_qadr[obj]
            self.data.qpos[adr:adr + 3] = [point[0], point[1], self.rest_z[obj]]
            self.data.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]      # upright, no tilt
            vadr = self.model.jnt_dofadr[self.model.joint(f"{obj}_free").id]
            self.data.qvel[vadr:vadr + 6] = 0.0                          # and not spinning
        mujoco.mj_forward(self.model, self.data)

        # Let everything come to rest        # Let everything come to rest before the episode starts, with the arm held where it is.
        hold = self.data.ctrl.copy()
        for _ in range(int(self.ecfg["settle_steps"])):
            self.data.ctrl[:] = hold
            mujoco.mj_step(self.model, self.data)
        for obj in OBJECTS:
            vadr = self.model.jnt_dofadr[self.model.joint(f"{obj}_free").id]
            self.data.qvel[vadr:vadr + 6] = 0.0
        self.data.qvel[self.dadr] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.mouth = self.scene.site("mouth")
        approach = self.mouth - np.array([0.0, 0.0, self.mouth[2]])
        approach = approach / max(1e-9, np.linalg.norm(approach))       # base -> mouth, horizontal
        self.stage_point = self.mouth - approach * float(self.ecfg["staging_m"])
        # Where the FOOD should end up: just short of the lips.
        self.food_point = self.mouth - approach * float(self.ecfg["food_gap_m"])

        # Hover so the FINGERTIPS clear the object's TOP. Keyed off the centre, a tall mug had
        # the tips arriving ~30 mm below its rim, from the side: 6/9 mug grasps failed shoved.
        if self.tip_ahead is None:
            Rt = self.data.site_xmat[self.tool.site_id].reshape(3, 3)
            axis, tcp = Rt @ self.tool.tool_local, self._tcp()
            self.tip_ahead = 0.008 + max(float((self.data.geom_xpos[g] - tcp) @ axis)
                                         for g in self.pad_geoms)
        self.reach_offset = np.array([0.0, 0.0, self.rest_z[self.name] + self.tip_ahead
                                      + float(self.ecfg["reach_clearance_m"])])
        self.filtered.fill(0)
        self.prev_action.fill(0)
        self.collisions = self.wall_hits = self.crush_steps = 0
        self.self_collisions = self.curl_steps = 0
        self.knocked = self.was_pinched = False
        self.peak_velocity = 0.0
        self.peak_grip_force = 0.0
        self._sample_perception_bias()
        self.object_start = self.scene.object_pos(self.name).copy()

    def _begin_stage(self, stage: str) -> None:
        """Start `stage` from wherever the arm is now. The low-pass state and previous action
        are deliberately NOT cleared: on the real arm one policy hands over to the next
        mid-motion, and the next one sees that."""
        self.stage = stage
        obj_pos = self.scene.object_pos(self.name)
        if stage == "present":
            self.target = self.food_point.copy()
            self.prev_dist = float(np.linalg.norm(self.target - obj_pos))
        else:
            self.target = obj_pos + self.reach_offset if stage == "reach" else obj_pos.copy()
            self.prev_dist = float(np.linalg.norm(self.target - self._tcp()))
        self.steps = self.hold_steps = 0
        self.phase = 0
        self.settle_calm = 0.0
        self.prev_tcp = self._tcp().copy()
        self.was_lifted = False
        self.pinch_paid = False
        # Each stage is judged on what IT did to the object, not on what it inherited.
        self.stage_object_start = obj_pos.copy()
        self.prev_lift = float(np.clip(obj_pos[2] - self.rest_z[self.name], 0.0,
                                       float(self.ecfg["lift_height_m"])))

    def _load_priors(self) -> list:
        """Frozen policies for every stage before this one, with their observation statistics."""
        import pickle

        import torch
        from stable_baselines3 import PPO

        torch.set_num_threads(1)
        priors = []
        for stage in STAGES[:STAGES.index(self.final_stage)]:
            if stage not in self.handoff["runs"]:
                continue                     # e.g. lift, now folded into grasp
            run = Path(self.handoff["runs"][stage])
            weights, stats = run / f"ppo_{stage}_final.zip", run / "vecnormalize.pkl"
            if not weights.exists() or not stats.exists():
                # Loud on purpose. Quietly starting from home instead would train a policy for
                # a situation the deployed pipeline never produces.
                raise FileNotFoundError(
                    f"hand-off for {self.final_stage!r} needs a finished {stage!r} in {run}")
            with open(stats, "rb") as fh:
                vn = pickle.load(fh)
            priors.append((stage, PPO.load(weights, device="cpu").policy, vn.obs_rms.mean.copy(),
                           np.sqrt(vn.obs_rms.var + vn.epsilon), float(vn.clip_obs)))
        return priors

    def _run_priors(self) -> bool:
        """Play the frozen earlier stages, each until its own success test fires."""
        self.handoff_steps = 0
        for stage, policy, mean, std, clip in self._priors:
            self._begin_stage(stage)
            obs = self._observation()
            for _ in range(int(self.handoff.get("max_steps", 150))):
                action, _ = policy.predict(np.clip((obs - mean) / std, -clip, clip)[None],
                                           deterministic=True)
                obs, _, done, truncated, info = self.step(action[0])
                self.handoff_steps += 1
                if self.prior_hook is not None:
                    self.prior_hook(self)
                if done or truncated:
                    break
            if not info["success"]:
                return False
        return True

    def _snapshot(self) -> dict:
        gids = [self.obj_gid[n] for n in OBJECTS]
        bids = [self.obj_bid[n] for n in OBJECTS]
        return dict(qpos=self.data.qpos.copy(), qvel=self.data.qvel.copy(), ctrl=self.data.ctrl.copy(),
                    warm=self.data.qacc_warmstart.copy(), time=float(self.data.time),
                    size=self.model.geom_size[gids].copy(), mass=self.model.body_mass[bids].copy(),
                    rest_z=dict(self.rest_z), width=dict(self.width), name=self.name,
                    onehot=self.onehot.copy(), filtered=self.filtered.copy(),
                    prev_action=self.prev_action.copy(), object_start=self.object_start.copy(),
                    mouth=self.mouth.copy(), stage_point=self.stage_point.copy(),
                    food_point=self.food_point.copy(), reach_offset=self.reach_offset.copy(),
                    steps=self.handoff_steps)

    def _restore(self, snap: dict) -> None:
        gids = [self.obj_gid[n] for n in OBJECTS]
        bids = [self.obj_bid[n] for n in OBJECTS]
        self.model.geom_size[gids] = snap["size"]
        self.model.body_mass[bids] = snap["mass"]
        self.data.qpos[:], self.data.qvel[:], self.data.ctrl[:] = snap["qpos"], snap["qvel"], snap["ctrl"]
        self.data.qacc_warmstart[:] = snap["warm"]
        self.data.time = snap["time"]
        mujoco.mj_forward(self.model, self.data)
        self.rest_z, self.width = dict(snap["rest_z"]), dict(snap["width"])
        self.name, self.onehot = snap["name"], snap["onehot"].copy()
        self.filtered, self.prev_action = snap["filtered"].copy(), snap["prev_action"].copy()
        self.object_start = snap["object_start"].copy()
        self.mouth, self.stage_point = snap["mouth"].copy(), snap["stage_point"].copy()
        self.food_point, self.reach_offset = snap["food_point"].copy(), snap["reach_offset"].copy()
        self.handoff_steps = snap["steps"]
        self.collisions = self.wall_hits = self.crush_steps = 0
        self.self_collisions = self.curl_steps = 0
        self.knocked = self.was_pinched = False
        self.peak_velocity = self.peak_grip_force = 0.0
        self._sample_perception_bias()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        use_priors = bool(self.handoff.get("enabled", False)) and self.final_stage != STAGES[0]
        if use_priors and self._priors is None:
            self._priors = self._load_priors()
        self.handoff_ok = True
        # Replaying the frozen stages costs ~90 ms, and the vectorised envs step in lockstep, so
        # one env resetting stalls the other 19: measured load 3.7 on 20 cores, 1,200 steps/s.
        # Most resets therefore restore a recent hand-off state (with fresh perception error)
        # instead of regenerating one.
        reuse = float(os.environ.get("YAM_HANDOFF_REUSE", self.handoff.get("reuse_prob", 0.0)))
        if use_priors and self._bank and self.np_random.random() < reuse:
            self._restore(self._bank[int(self.np_random.integers(len(self._bank)))])
        else:
            for _ in range(1 + int(self.handoff.get("retries", 4)) if use_priors else 1):
                self._reset_scene()
                self.handoff_ok = self._run_priors() if use_priors else True
                if self.handoff_ok:
                    break
            if use_priors and self.handoff_ok:
                self._bank.append(self._snapshot())
                del self._bank[:-int(self.handoff.get("bank", 24))]
        self._begin_stage(self.final_stage)
        return self._observation(), {"success": False, "object": self.name,
                                     "handoff_ok": self.handoff_ok,
                                     "handoff_steps": self.handoff_steps}

    def step(self, action: np.ndarray):
        raw = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        alpha = float(self.ecfg["action_lowpass"])
        self.filtered = (1 - alpha) * self.filtered + alpha * raw

        # The ONLY velocity enforcement: cap the commanded joint delta. Physics is never edited.
        cap = float(self.ecfg["max_joint_velocity_rad_s"])
        max_delta = cap * self.dt
        desired = self.data.ctrl[:6] + np.clip(
            float(self.ecfg["action_delta_rad"]) * self.filtered[:6], -max_delta, max_delta)
        self.data.ctrl[:6] = np.clip(desired, self.ctrl_lo, self.ctrl_hi)
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
            self.target = self.food_point
        elif self.stage == "reach":
            self.target = obj_pos + self.reach_offset
        else:
            self.target = obj_pos

        tcp_speed = float(np.linalg.norm(tcp - self.prev_tcp) / self.dt)
        self.prev_tcp = tcp.copy()
        if self.stage == "present":
            distance = float(np.linalg.norm(self.target - obj_pos))
        else:
            distance = float(np.linalg.norm(self.target - tcp))
        # Does the food reach the mouth before the jaws do?
        obj_radius = 0.5 * float(self.width[self.name])
        lead_margin = (self._pad_distance_to(self.mouth)
                       - max(0.0, float(np.linalg.norm(self.mouth - obj_pos)) - obj_radius))
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
        lift_h = float(self.ecfg["lift_height_m"])
        height = float(obj_pos[2]) - self.rest_z[self.name]
        stage_drift = float(np.linalg.norm(obj_pos[:2] - self.stage_object_start[:2]))
        joint_speed_now = float(np.abs(self.data.qvel[self.dadr]).max())
        # Gripper pose relative to the object. Straight down the camera axis, jaws squared to
        # the object with the wrist, object SEATED between the claws rather than nipped by the
        # tips. Measured before any of this was asked for: 13-19 deg tilt at hand-off (max 32),
        # jaws 15-23 deg off-square, and a block up to 82 mm corner-to-corner against an 82.8 mm
        # opening -- which only fits squared up. The block was the worst object in every census.
        Rt = self.data.site_xmat[self.tool.site_id].reshape(3, 3)
        tool_axis, jaw_axis = Rt @ self.tool.tool_local, Rt @ self.tool.jaw_local
        tilt_deg = float(np.degrees(np.arccos(np.clip(-tool_axis[2], -1.0, 1.0))))
        if self.name == "block":
            face = self.data.xmat[self.obj_bid[self.name]].reshape(3, 3)[:, 0]
            yaw = float(np.arctan2(jaw_axis[1], jaw_axis[0]) - np.arctan2(face[1], face[0]))
            square_err = float(np.sin(2.0 * yaw) ** 2)                  # 0 squared, 1 at 45 deg
            off_square_deg = abs(((np.degrees(yaw) + 45.0) % 90.0) - 45.0)
        else:
            square_err, off_square_deg = 0.0, 0.0                       # round: any yaw fits
        if self.tip_ahead is None:
            self.tip_ahead = 0.008 + max(float((self.data.geom_xpos[g] - tcp) @ tool_axis)
                                         for g in self.pad_geoms)
        # How far the object's centre is past the fingertips, toward the gripper base, as a
        # fraction of what the table allows for an object this tall.
        insertion = float((obj_pos - (tcp + tool_axis * self.tip_ahead)) @ (-tool_axis))
        seat_frac = float(np.clip(insertion / max(1e-3, self.rest_z[self.name] - 0.005), 0.0, 1.0))
        # Where the object sits between the claws, in the claw's own axes.
        to_obj = obj_pos - tcp
        e_jaw = float(to_obj @ jaw_axis)                       # toward one claw or the other
        e_pad = float(to_obj @ np.cross(tool_axis, jaw_axis))  # across the width of the claws
        lateral = float(np.hypot(e_jaw, e_pad))
        side_clearance = 0.5 * (float(self.tool.max_width) - float(self.width[self.name]))
        centred = (abs(e_jaw) <= max(0.003, side_clearance - 0.002)
                   and abs(e_pad) <= float(self.ecfg["centre_across_m"]))
        oriented = (tilt_deg <= float(self.ecfg["max_tilt_deg"])
                    and off_square_deg <= float(self.ecfg["max_off_square_deg"]))

        # Virtual wall around the head: never reward getting closer than this.
        wall = float(self.ecfg["pad_min_distance_m"])
        head_gap = self._pad_distance_to(self.mouth)
        breached = head_gap < wall
        self.wall_hits += int(breached)

        if self.stage == "reach":
            # Arrive AND settle. Handing grasp a moving wrist lets momentum carry the gripper
            # through the object before the jaws can act.
            joint_speed = float(np.abs(self.data.qvel[self.dadr]).max())
            settled = (distance <= float(self.ecfg["success_distance_m"])
                       and oriented
                       and joint_speed <= float(self.ecfg["reach_settle_qvel"])
                       and tcp_speed <= float(self.ecfg["reach_settle_speed_mps"]))
            self.hold_steps = self.hold_steps + 1 if settled else 0
            success = self.hold_steps >= round(float(self.ecfg["reach_settle_s"]) / self.dt)
            # Measured here, PAID below once `reward` exists. Inside tolerance, being slow
            # pays in proportion, so the policy can descend into the settle gate rather than
            # having to stumble onto it.
            self.settle_calm = (1.0 - min(1.0, joint_speed / float(self.ecfg["reach_settle_qvel"]))
                                if distance <= float(self.ecfg["success_distance_m"]) else 0.0)
        elif self.stage == "grasp":
            # A grab is a PICK-UP: seated between the claws, lifted clear of the table, straight
            # up, and held still. A squeeze that never leaves the table is not a grasp, and the
            # next stage needs a stationary start.
            settled = stage_drift <= float(self.ecfg["pick_max_drift_m"])
            seated = seat_frac >= float(self.ecfg["min_seat_frac"])
            at_height = carrying and height <= lift_h + float(self.ecfg["lift_band_m"])
            steady = (joint_speed_now <= float(self.ecfg["reach_settle_qvel"])
                      and tcp_speed <= float(self.ecfg["reach_settle_speed_mps"]))
            self.hold_steps = self.hold_steps + 1 if (at_height and seated and settled and steady) else 0
            success = self.hold_steps >= round(float(self.ecfg["grasp_hold_s"]) / self.dt)
        elif self.stage == "lift":
            # Straight up, then STOP. Present needs a stationary, known starting pose, and an
            # object dragged sideways on the way up is one that was nearly knocked over.
            at_height = carrying and height <= lift_h + float(self.ecfg["lift_band_m"])
            steady = (joint_speed_now <= float(self.ecfg["reach_settle_qvel"])
                      and tcp_speed <= float(self.ecfg["reach_settle_speed_mps"]))
            straight = stage_drift <= float(self.ecfg["lift_max_drift_m"])
            self.hold_steps = self.hold_steps + 1 if (at_height and steady and straight) else 0
            success = self.hold_steps >= round(float(self.ecfg["lift_hold_s"]) / self.dt)
        else:
            near = distance <= float(self.ecfg["present_tolerance_m"])
            gentle = tcp_speed <= float(self.ecfg["approach_speed_mps"])
            leads = lead_margin > 0.0
            ok = carrying and near and leads and gentle and not breached and not crushing
            self.hold_steps = self.hold_steps + 1 if ok else 0
            success = self.hold_steps >= round(float(self.ecfg["present_hold_s"]) / self.dt)

        # ---------------------------------------------------------------- phases
        # Where the gripper has to be before closing means anything: over the object,
        # at grasp height. Two separate tests, because a single 3D distance also passes
        # when the gripper is beside the object rather than above it.
        grip_fraction = float((self.data.ctrl[self.grip_aid] - self.grip_range[0])
                              / max(1e-9, np.ptp(self.grip_range)))
        over_xy = float(np.linalg.norm(tcp[:2] - obj_pos[:2]))
        height_error = abs(float(tcp[2] - obj_pos[2]))
        # Closing is only worth anything once the claw is PLACED: object centred within the real
        # side clearance, seated deep, claw vertical and squared. The old test (35 mm sideways,
        # 50 mm vertically) was already true hovering at the object's top -- in 9 of 15 failures
        # it was squeezing at seating depth 0.11, with the finger bodies landing on the object.
        in_position = (centred and oriented
                       and seat_frac >= float(self.ecfg["min_seat_frac"]))
        if pinched:
            phase = 2                                   # SECURE
        elif in_position:
            phase = 1                                   # CLOSE
        else:
            phase = 0                                   # APPROACH
        self.phase = phase

        progress = self.prev_dist - distance
        reward = 10.0 * progress - 0.1 * distance
        if self.stage == "grasp" and phase == 2:
            # Once pinched, TCP-to-object distance means nothing -- and for a grasp the table
            # limits, the TCP sits ABOVE the object's centre, so this term paid for pushing DOWN.
            # Measured: after the pinch the TCP went 10-30 mm down, the object squirted sideways
            # to the fail line, while a scripted straight-up pull lifted the same pinch 95-135 mm.
            reward = 0.0
        if self.stage in ("reach", "grasp") and phase < 2:
            # COSTS, fading in with proximity so the transit is free. Never a bonus: a pose that
            # pays per step is a pose worth loitering in.
            near = float(np.clip(1.0 - distance / float(self.ecfg["orient_radius_m"]), 0.0, 1.0))
            reward -= float(self.ecfg["orient_penalty"]) * near * (
                min(1.0, tilt_deg / 30.0) + square_err)
            if self.stage == "grasp":
                # Centre FIRST, then descend. Off-centre, the cost is flat in depth, so there is
                # nothing to gain by going down; centring lowers it, and only then does depth.
                c = min(1.0, lateral / float(self.ecfg["centre_scale_m"]))
                reward -= float(self.ecfg["seat_penalty"]) * near * (c + (1.0 - c) * (1.0 - seat_frac))

        if phase == 0:
            # Approach clean: a COST for closing early or touching, never a per-step reward for
            # being here. Paying to stay in a phase is how the policy learned to park.
            if grip_fraction < float(self.ecfg["open_enough"]):
                reward -= float(self.ecfg["early_close_penalty"])
            if touching:
                reward -= float(self.ecfg["approach_contact_penalty"])
        elif phase == 1:
            # In position: now, and only now, squeezing is what earns.
            reward += float(self.ecfg["close_bonus"]) * (1.0 - grip_fraction)
            reward += float(self.ecfg["in_position_bonus"])
        else:
            if self.stage == "grasp" and seat_frac < float(self.ecfg["min_seat_frac"]):
                # Nipped by the tips. It cannot succeed like this, so it must not earn like
                # this either: census found 243 steps parked in a shallow pinch on the apple.
                reward -= float(self.ecfg["seat_penalty"]) * (1.0 - seat_frac)
            elif self.stage != "grasp":
                reward += float(self.ecfg["pinch_bonus"])
            else:
                # In the grab, a seated pinch pays ONCE, as a milestone, and per step only while
                # HOLDING AT HEIGHT. Paid per step on the table it was rent: +0.10/step net, and
                # with discounting ~70 steps of it beat the eventual fail charge. Measured: after
                # clean pinches (1-6 mm displacement) the arm never rose, best lift 6 mm, and
                # 29/30 episodes ended dragged to exactly the 40 mm gate.
                if not self.pinch_paid:
                    reward += float(self.ecfg["pinch_once_bonus"])
                    self.pinch_paid = True
                if carrying:
                    reward += float(self.ecfg["pinch_bonus"])
            # Height pays as PROGRESS, capped at the lift height, so it telescopes to a fixed
            # total and cannot be farmed. The old `lift_bonus * height` was rent with no ceiling:
            # 0.9/step at 30 cm, 270 an episode against a success bonus of 50, so once pinched the
            # best income was to swing the object as high as the arm goes.
            capped = float(np.clip(height, 0.0, lift_h))
            if self.stage != "grasp" or seat_frac >= float(self.ecfg["min_seat_frac"]):
                reward += float(self.ecfg["lift_progress_gain"]) * (capped - self.prev_lift)
                self.prev_lift = capped
            if self.stage in ("grasp", "lift"):
                # All three are COSTS. Sideways travel, going past the band, and moving once up.
                # The speed cost fades in with height so there is no cliff to stall beneath.
                self.was_lifted |= bool(carrying)
                reward -= float(self.ecfg["drift_penalty"]) * stage_drift
                reward -= float(self.ecfg["overshoot_penalty"]) * max(
                    0.0, height - lift_h - float(self.ecfg["lift_band_m"]))
                reward -= (float(self.ecfg["hover_speed_penalty"]) * (capped / lift_h) ** 2
                           * min(1.0, joint_speed_now / float(self.ecfg["reach_settle_qvel"])))
            if self.stage == "present":
                reward += float(self.ecfg["carry_bonus"]) * carrying
                if carrying and lead_margin > 0.0:
                    reward += float(self.ecfg["lead_bonus"])
                # Slow only once carrying AND close to the person; the rest is free.
                if carrying and float(np.linalg.norm(self.mouth - tcp)) < float(
                        self.ecfg["slow_radius_m"]):
                    excess = tcp_speed - float(self.ecfg["approach_speed_mps"])
                    if excess > 0:
                        reward -= float(self.ecfg["speed_penalty"]) * excess

        # Losing the object after having held it is the failure mode that matters.
        if self.stage in ("lift", "present") and self.was_pinched and not pinched and self.steps > 10:
            reward -= float(self.ecfg["drop_penalty"])
        self.was_pinched = pinched

        # ---------------------------------------------------------------- always-on costs
        if self.stage == "reach" and self.settle_calm:
            reward += float(self.ecfg["settle_bonus"]) * self.settle_calm
        # Time costs. Without it, any per-step bonus makes loitering profitable.
        reward -= float(self.ecfg["time_penalty"])
        reward -= float(self.ecfg["velocity_penalty"]) * np.square(self.data.qvel[self.dadr]).mean()
        reward -= float(self.ecfg["jerk_penalty"]) * np.square(raw - self.prev_action).mean()
        tcp_radius = float(np.linalg.norm(tcp[:2]))
        curled = tcp_radius < float(self.ecfg["min_tcp_radius_m"])
        self.curl_steps += int(curled)
        span = np.maximum(self.hi - self.lo, 1e-6)
        normalized = 2.0 * (self.data.qpos[self.qadr] - 0.5 * (self.lo + self.hi)) / span
        limit_strain = float(np.mean(self.limit_weights * np.power(np.abs(normalized), 8)))
        reward -= float(self.ecfg["self_collision_penalty"]) * self_hits
        reward -= float(self.ecfg["curl_penalty"]) * curled
        reward -= float(self.ecfg["joint_limit_penalty"]) * limit_strain
        reward -= float(self.ecfg["collision_penalty"]) * table_hits
        displaced = float(np.linalg.norm(obj_pos[:2] - self.object_start[:2]))
        # Charged ONCE. Per-step, a single early nudge fined the policy for the rest of the
        # episode, and never touching anything became the better strategy.
        if not carrying and not self.knocked and displaced > float(self.ecfg["knock_free_m"]):
            self.knocked = True
            reward -= float(self.ecfg["knock_penalty"])
        reward -= float(self.ecfg["wall_penalty"]) * breached
        # Firm but not crushing: penalise force past the limit, and reward the band below it
        # only while actually holding, so the policy cannot earn it by hovering with open jaws.
        if crushing:
            reward -= float(self.ecfg["crush_penalty"]) * (grip_force - crush_limit) / crush_limit
        elif (held and grip_force >= float(self.ecfg["min_grip_force_n"])
              and (self.stage != "grasp" or carrying)):          # same rent, same fix
            reward += float(self.ecfg["grip_band_bonus"])
        if measured_velocity > cap * 1.5:
            reward -= float(self.ecfg["velocity_breach_penalty"])
        if success:
            reward += float(self.ecfg["success_bonus"])

        self.prev_dist, self.prev_action = distance, raw.copy()
        # An object off the table is unrecoverable; chasing it is what made the arm fly away.
        lost = (displaced > float(self.ecfg["object_lost_m"])
                or float(obj_pos[2]) < float(self.ecfg["object_lost_z"]))
        if lost:
            reward -= float(self.ecfg["lost_penalty"])
        # An episode that can no longer succeed ends NOW. Left running, its only income is
        # SECURE rent, and collecting that is what taught the policy to wave the object about.
        failed = lost
        if self.stage == "grasp":
            # Before the pinch a 4 cm move is a shove. Once held, some sideways travel on the way
            # up is a lift, not a shove: a scripted vertical pull itself drifted 10-40 mm.
            gate = "pick_abort_drift_m" if self.pinch_paid else "grasp_max_displacement_m"
            failed |= stage_drift > float(self.ecfg[gate])
            failed |= self.was_lifted and not pinched          # picked it up and DROPPED it
        elif self.stage == "lift":
            failed |= stage_drift > float(self.ecfg["lift_abort_drift_m"])
        failed = failed and not success
        if failed:
            # Charge the time it skipped. Without this, flinging the object off the table cost
            # 4 while parking cost 15, and the policy learned to end episodes by throwing.
            reward -= float(self.ecfg["time_penalty"]) * (int(self.ecfg["episode_steps"]) - self.steps)
        truncated = self.steps >= int(self.ecfg["episode_steps"])
        info = {"success": bool(success), "is_success": bool(success), "object": self.name,
                "distance": distance, "max_joint_velocity": measured_velocity,
                "peak_joint_velocity": self.peak_velocity, "velocity_cap": cap,
                "collision_count": self.collisions, "wall_hits": self.wall_hits,
                "grip_force_n": grip_force, "peak_grip_force_n": self.peak_grip_force,
                "crush_steps": self.crush_steps, "object_width_m": self.width[self.name],
                "object_height": float(obj_pos[2]), "carrying": bool(carrying), "pinched": bool(pinched),
                "object_displaced_m": displaced, "self_collisions": self.self_collisions,
                "curl_steps": self.curl_steps, "tcp_radius_m": tcp_radius,
                "grip_fraction": grip_fraction, "lead_margin_m": lead_margin,
                "tcp_speed_mps": tcp_speed, "joint_speed": float(np.abs(self.data.qvel[self.dadr]).max()), "pad_gap_m": head_gap, "phase": phase, "in_position": bool(in_position),
                "knocked": bool(self.knocked), "lost": bool(lost), "failed": bool(failed),
                "lift_m": height, "stage_drift_m": stage_drift, "tilt_deg": tilt_deg,
                "off_square_deg": float(off_square_deg), "seat_frac": seat_frac, "e_jaw_m": e_jaw, "e_pad_m": e_pad,
                "centred": bool(centred), "oriented": bool(oriented)}
        return self._observation(), float(reward), bool(success or failed), bool(truncated), info

    def render(self):
        return self.scene.render("scene_cam", 640)

    def close(self):
        if self.scene._renderer is not None:
            self.scene._renderer.close()
            self.scene._renderer = None
