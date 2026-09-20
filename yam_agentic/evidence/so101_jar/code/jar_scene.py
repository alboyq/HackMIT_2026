"""JarScene — jar side-grasp scene rendered through the real-camera compositing pipeline.

Physics: the object described by rl/jar/object_spec.py (rl/real/object.json; kind "jar" = the legacy
scene_jar_rl.xml, a 5.1 x 10.2 cm jar with lid + label) in SO101LiftEnv.
Pixels: MuJoCo render from the calibrated camera (camera_calib*.json: pos/target/fovy),
segmentation-composited over a background, with per-episode randomization of camera
pose/roll/FOV, lighting, robot + jar colours and the background's brightness/tint/shift.
Backgrounds: photos of the real empty workspace (bg_dir) mixed with a pool of unrelated
images (bg_pool) so the policy keys on the arm and the jar, not on the room:
    p_real = probability that an episode uses a real-scene photo.
obs = {image (IMG,IMG,3) uint8, state qpos(6) float32}; action = absolute joint targets
(rad), the convention LeRobot ACT expects.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/Users/adipu/so101Sim/rl/native")
sys.path.insert(0, "/Users/adipu/so101Sim/rl/real")
import cv2
import mujoco
import numpy as np
import so101_lift

sys.path.insert(0, str(Path(__file__).parent))
import arm_variant  # noqa: E402   (rl/real)
from object_spec import SPEC  # noqa: E402   (rl/real/object.json, or OBJECT_SPEC=...; kind "jar" = the legacy scene)

so101_lift.SCENE = SPEC.scene
from so101_lift import SO101LiftEnv  # noqa: E402
from camera_alignment import lookat_quat  # noqa: E402

JAR_HH = SPEC.rest_z                            # height of the object's body origin when it stands on the table
DEFAULT_ZONE = (0.28, 0.42, -0.6, 0.6)          # r_min, r_max, th_min, th_max: where SideExpert is ~100%
DEFAULT_POOL = "/Users/adipu/so101Sim/rl/jar/bg_pool"


def _square(im, img):
    h, w = im.shape[:2]
    s = min(h, w)
    im = im[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
    return cv2.cvtColor(cv2.resize(im, (img, img), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def load_backgrounds(bg_dir, img):
    bg_dir = Path(bg_dir)
    files = sorted((bg_dir / "full").glob("*.jpg")) or sorted(bg_dir.glob("*.png")) + sorted(bg_dir.glob("*.jpg"))
    out = [_square(im, img) for im in (cv2.imread(str(f)) for f in files) if im is not None]
    if not out:
        raise RuntimeError(f"no background photos in {bg_dir}")
    return out


class JarScene:
    def __init__(self, calib, bg_dir, img=256, ctrl_dt=0.04, zone=DEFAULT_ZONE, dr=True,
                 bg_pool=DEFAULT_POOL, p_real=0.5, cam_jitter=0.04):
        self.env = SO101LiftEnv(ctrl_dt=ctrl_dt, randomize=False)
        arm_variant.apply(self.env.model)               # the real arm has no wrist-camera bracket: hide it + no collisions
        self.m, self.d = self.env.model, self.env.data
        m = self.m
        self.img, self.zone, self.dr, self.dt = img, zone, dr, ctrl_dt
        self.p_real, self.cam_jitter = p_real, cam_jitter
        if cam_jitter == 0.04 and os.environ.get("CAM_JITTER"):          # training-time override (eval passes it explicitly)
            self.cam_jitter = float(os.environ["CAM_JITTER"])
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), img)
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), img)
        self.cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "base_cam")
        m.cam_sensorsize[self.cid] = 0
        c = json.loads(Path(calib).read_text())
        self.cam_pos0, self.cam_tgt0, self.fovy0 = np.array(c["pos"], float), np.array(c["target"], float), float(c["fovy"])
        self.cam_up0 = np.array(c.get("up", [0.0, 0.0, 1.0]), float)
        self.ren = mujoco.Renderer(m, img, img)
        self.seg = mujoco.Renderer(m, img, img)
        self.seg.enable_segmentation_rendering()
        jar_body = self.env._box_body
        self.jar_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == jar_body]
        robot = [g for g in range(m.ngeom) if m.geom_bodyid[g] not in (0, jar_body)]
        self.keep = np.array(sorted(robot + self.jar_geoms))
        self.robot_mats = sorted({int(m.geom_matid[g]) for g in robot if m.geom_matid[g] >= 0})
        self.jar_mat = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, int(m.geom_matid[g])): int(m.geom_matid[g])
                        for g in self.jar_geoms if m.geom_matid[g] >= 0}
        self.mat0 = m.mat_rgba.copy()
        self.emission0 = m.mat_emission.copy()
        self.light_dir0, self.light_diff0 = m.light_dir.copy(), m.light_diffuse.copy()
        self.head0 = (m.vis.headlight.ambient.copy(), m.vis.headlight.diffuse.copy())
        self.bgs = load_backgrounds(bg_dir, img)
        self.pool_files = sorted(Path(bg_pool).glob("*.jpg")) if bg_pool and Path(bg_pool).is_dir() else []
        self._pool_cache = {}
        self.hold_ticks = max(1, int(round(0.4 / ctrl_dt)))
        self._gain0, self._bias0 = m.actuator_gainprm.copy(), m.actuator_biasprm.copy()
        if zone == DEFAULT_ZONE and os.environ.get("JAR_ZONE"):
            self.zone = tuple(float(v) for v in os.environ["JAR_ZONE"].split())
        self.lo, self.hi = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()
        self.rng = np.random.default_rng(0)
        self.bg = self.bgs[0]
        self._blur = False
        self._frames, self._lat, self._shadow = [], 0, None      # set per episode in reset()

    # ------------------------------------------------------------------ episode
    def reset(self, seed):
        rng = self.rng = np.random.default_rng(seed)
        env, m, d = self.env, self.m, self.d
        env.reset()
        r, th = rng.uniform(self.zone[0], self.zone[1]), rng.uniform(self.zone[2], self.zone[3])
        yaw = rng.uniform(-np.pi, np.pi)
        a = env._box_qadr
        d.qpos[a:a + 3] = (r * np.cos(th), r * np.sin(th), JAR_HH)
        d.qpos[a + 3:a + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        d.qvel[:] = 0
        # jar-on-table friction: plastic on a stone counter is ~0.3-0.6 (the MJCF default of 1.0 made
        # the jar topple at the slightest nudge instead of sliding). JAR_FRICTION="lo hi" overrides.
        lo, hi = (float(v) for v in os.environ.get("JAR_FRICTION", "0.35 0.7").split())
        mu = rng.uniform(lo, hi)
        for g in self.jar_geoms:                    # cylinder AND base plate (the plate is what touches the table)
            if m.geom_contype[g] or m.geom_conaffinity[g]:
                m.geom_priority[g] = 1              # MuJoCo takes max(friction) unless one geom has priority
                m.geom_friction[g, 0] = mu
        # servo stiffness: the real STS3215s (P=16) are far softer than the MJCF's kp=998, so the real
        # arm lags and sags. KP_SCALE="lo hi" randomises the position gain per episode.
        klo, khi = (float(v) for v in os.environ.get("KP_SCALE", "1.0 1.0").split())
        ks = rng.uniform(klo, khi)
        m.actuator_gainprm[:, 0] = self._gain0[:, 0] * ks
        m.actuator_biasprm[:, 1] = self._bias0[:, 1] * ks
        d.qpos[env._arm_qadr[:5]] = rng.normal(0, 0.03, 5) if self.dr else 0.0
        d.qpos[env._arm_qadr[5]] = 0.3
        d.ctrl[:] = d.qpos[env._arm_qadr]
        self.jar_xy = np.array([r * np.cos(th), r * np.sin(th)])
        self._held, self.success, self.max_rise = 0, False, 0.0
        # camera latency: the phone delivers frames ~85-100 ms late (2-3 ticks at 25 Hz). OBS_LATENCY="lo hi" ticks.
        llo, lhi = (int(v) for v in os.environ.get("OBS_LATENCY", "0 0").split())
        self._lat = int(rng.integers(llo, lhi + 1)); self._frames = []
        # cast shadow under the jar (the composite otherwise has none): JAR_SHADOW=1
        self._shadow = None
        if os.environ.get("JAR_SHADOW") == "1":
            ang = rng.uniform(0, 2 * np.pi)
            self._shadow = dict(dir=np.array([np.cos(ang), np.sin(ang)]), length=rng.uniform(0.02, 0.08),
                                 opacity=rng.uniform(0.12, 0.45), width=rng.uniform(0.025, 0.04))
        mujoco.mj_forward(m, d)        # settle the new object/arm pose BEFORE the camera is chosen:
        self._visual_dr()              # _frames_task projects d.xpos, which would otherwise be a tick stale
        mujoco.mj_forward(m, d)
        return self.obs()

    def _pool_image(self, rng):
        f = self.pool_files[int(rng.integers(len(self.pool_files)))]
        if f not in self._pool_cache:
            im = cv2.imread(str(f))
            self._pool_cache[f] = im
        im = self._pool_cache[f]
        h, w = im.shape[:2]
        s = int(min(h, w) * rng.uniform(0.5, 1.0))                     # random square crop
        y0, x0 = int(rng.integers(0, h - s + 1)), int(rng.integers(0, w - s + 1))
        crop = cv2.resize(im[y0:y0 + s, x0:x0 + s], (self.img, self.img), interpolation=cv2.INTER_AREA)
        if rng.random() < 0.5:
            crop = crop[:, ::-1]
        return cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_BGR2RGB)

    def _frames_task(self, pos, tgt, up, fovy, margin=8):
        """Would this camera actually SHOW the task? Projects the object and the gripper analytically
        (no mj_forward needed) and requires both inside the image with a margin.

        Wide camera randomisation without this check produces unanswerable training samples: at +-15 cm,
        13 % of episodes put the object outside the frame and 9 % the gripper. The action label is still the
        expert's precise reach, so the only way to fit those samples is to predict from joint state alone -
        i.e. the mean trajectory - which is gradient pressure to IGNORE the camera. Measured 2026-09-18.
        """
        f = tgt - pos
        f = f / (np.linalg.norm(f) + 1e-9)
        r = np.cross(f, up); r /= (np.linalg.norm(r) + 1e-9)
        u = np.cross(r, f)
        fpx = (self.img / 2) / np.tan(np.deg2rad(fovy) / 2)
        for P in (self.d.xpos[self.env._box_body], self.d.site_xpos[self.env._grip_site]):
            dp = np.asarray(P) - pos
            z = dp @ f
            if z <= 0.05:
                return False
            px = self.img / 2 + fpx * (dp @ r) / z
            py = self.img / 2 - fpx * (dp @ u) / z
            if not (margin <= px < self.img - margin and margin <= py < self.img - margin):
                return False
        return True

    def _visual_dr(self):
        rng, m = self.rng, self.m
        m.mat_rgba[:] = self.mat0
        pos, tgt, fovy, up = self.cam_pos0.copy(), self.cam_tgt0.copy(), self.fovy0, self.cam_up0.copy()
        bg = self.bgs[int(rng.integers(len(self.bgs)))]
        if self.dr:
            j = self.cam_jitter
            # rejection-sample the camera so the randomisation stays wide but every sample is ANSWERABLE.
            # CAM_REQUIRE_VISIBLE=0 restores the old unfiltered behaviour (runs before 2026-09-18 used that).
            need = os.environ.get("CAM_REQUIRE_VISIBLE", "1") == "1"
            for attempt in range(16):
                pos = self.cam_pos0 + rng.uniform(-j, j, 3)
                tgt = self.cam_tgt0 + rng.uniform(-j, j, 3)
                fovy = self.fovy0 + rng.normal(0, 2.0)
                up = self.cam_up0 + np.concatenate([rng.normal(0, 0.04, 2), [0.0]])
                if not need or self._frames_task(pos, tgt, up, fovy):
                    break
            else:                                              # nothing framed the task: fall back to nominal
                pos, tgt, fovy, up = self.cam_pos0.copy(), self.cam_tgt0.copy(), self.fovy0, self.cam_up0.copy()
            gain = rng.uniform(0.8, 1.15)
            for mid in self.robot_mats:                                         # robot: keep hue, vary brightness
                m.mat_rgba[mid, :3] = np.clip(self.mat0[mid, :3] * gain * rng.uniform(0.95, 1.05, 3), 0, 1)
            jm = self.jar_mat
            if SPEC.rgb is not None:                                            # known colour: stay near it
                for name in ("jar_body", "jar_label"):                          # (cylinder, base plate)
                    if name in jm:
                        m.mat_rgba[jm[name], :3] = np.clip(np.array(SPEC.rgb) * rng.uniform(0.75, 1.08) * rng.uniform(0.96, 1.04, 3), 0, 1)
                        m.mat_emission[jm[name]] = self.emission0[jm[name]] * rng.uniform(0.3, 1.6)
                if "jar_lid" in jm:                                             # bolt: some grey metal
                    m.mat_rgba[jm["jar_lid"], :3] = rng.uniform(0.35, 0.7)
            else:                                                               # unknown colour: any hue
                hsv = np.uint8([[[rng.integers(0, 180), rng.integers(40, 200), rng.integers(90, 230)]]])
                if "jar_body" in jm:
                    m.mat_rgba[jm["jar_body"], :3] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0] / 255.0
                if "jar_lid" in jm:
                    m.mat_rgba[jm["jar_lid"], :3] = rng.uniform(0.03, 0.25)
                hsv = np.uint8([[[rng.integers(0, 180), rng.integers(120, 255), rng.integers(80, 230)]]])
                if "jar_label" in jm:
                    m.mat_rgba[jm["jar_label"], :3] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0] / 255.0
            m.light_dir[:] = self.light_dir0 + rng.uniform(-0.3, 0.3, self.light_dir0.shape)
            m.light_diffuse[:] = np.clip(self.light_diff0 * rng.uniform(0.6, 1.2), 0, 1)
            m.vis.headlight.ambient[:] = np.clip(self.head0[0] * rng.uniform(0.8, 1.6), 0, 1)
            m.vis.headlight.diffuse[:] = np.clip(self.head0[1] * rng.uniform(0.7, 1.3), 0, 1)
            if self.pool_files and rng.random() > self.p_real:
                bg = self._pool_image(rng)
            b = bg.astype(np.float32) * rng.uniform(0.75, 1.25) * rng.uniform(0.92, 1.08, 3)
            M = np.float32([[1, 0, rng.uniform(-8, 8)], [0, 1, rng.uniform(-8, 8)]])
            bg = cv2.warpAffine(np.clip(b, 0, 255).astype(np.uint8), M, (self.img, self.img), borderMode=cv2.BORDER_REFLECT)
        self.bg = bg
        m.cam_pos[self.cid] = pos
        m.cam_quat[self.cid] = lookat_quat(pos, tgt, up=tuple(up))
        m.cam_fovy[self.cid] = fovy
        self._blur = bool(self.dr and rng.random() < 0.5)

    # ------------------------------------------------------------------ io
    def _project(self, P):
        R = self.d.cam_xmat[self.cid].reshape(3, 3)
        dcam = (np.asarray(P) - self.d.cam_xpos[self.cid]) @ R           # camera frame: x right, y up, looks along -z
        fpx = (self.img / 2) / np.tan(np.deg2rad(self.m.cam_fovy[self.cid]) / 2)
        return np.array([self.img / 2 + fpx * dcam[0] / -dcam[2], self.img / 2 - fpx * dcam[1] / -dcam[2]])

    def _render_now(self):
        self.ren.update_scene(self.d, camera="base_cam")
        rgb = self.ren.render()
        self.seg.update_scene(self.d, camera="base_cam")
        ids = self.seg.render()[:, :, 0]
        keep = np.isin(ids, self.keep)
        out = self.bg.copy()
        if self._shadow is not None:
            jar = self.d.xpos[self.env._box_body]
            fade = float(np.clip(1.0 - (jar[2] - JAR_HH) / 0.08, 0.0, 1.0))
            if fade > 0:
                sh = self._shadow
                a = self._project([jar[0], jar[1], 0.0])
                b = self._project([jar[0] + sh["dir"][0] * sh["length"], jar[1] + sh["dir"][1] * sh["length"], 0.0])
                wpx = int(np.linalg.norm(self._project([jar[0] + sh["width"], jar[1], 0.0]) - a))
                pts = np.array([a, b])
                if 2 <= wpx <= 80 and np.all(np.isfinite(pts)) and np.all(np.abs(pts) < 4 * self.img):
                    mask = np.zeros((self.img, self.img), np.float32)     # (a jar knocked toward the lens projects absurdly: skip)
                    cv2.line(mask, tuple(int(v) for v in a), tuple(int(v) for v in b), 1.0, thickness=2 * wpx)
                    mask = cv2.GaussianBlur(mask, (0, 0), max(1.5, wpx * 0.6))[..., None] * sh["opacity"] * fade
                    out = (out.astype(np.float32) * (1.0 - mask)).astype(np.uint8)
        out[keep] = rgb[keep]
        if self._blur:
            out = cv2.GaussianBlur(out, (3, 3), 0)
        if os.environ.get("OBS_CORRUPT"):                   # whole-image camera effects: see obs_corrupt.py
            from obs_corrupt import from_env
            out = from_env(out)
        return out

    def render(self):
        """One call per control tick. Returns the frame from `latency` ticks ago (camera lag)."""
        self._frames.append(self._render_now())
        if len(self._frames) > self._lat + 1:
            self._frames.pop(0)
        return self._frames[0]

    def state(self):
        return self.d.qpos[self.env._arm_qadr].astype(np.float32)

    def obs(self):
        return {"image": self.render(), "state": self.state()}

    def tilt_deg(self):
        up = self.d.xmat[self.env._box_body].reshape(3, 3)[2, 2]
        return float(np.degrees(np.arccos(np.clip(up, -1, 1))))

    def step(self, ctrl):
        self.d.ctrl[:] = np.clip(ctrl, self.lo, self.hi)
        for _ in range(self.env._n_sub):
            mujoco.mj_step(self.m, self.d)
        rise = float(self.d.xpos[self.env._box_body][2] - JAR_HH)
        self.max_rise = max(self.max_rise, rise)
        self._held = self._held + 1 if (rise > 0.05 and self.tilt_deg() < 45) else 0
        if self._held >= self.hold_ticks:
            self.success = True
        return self.success
