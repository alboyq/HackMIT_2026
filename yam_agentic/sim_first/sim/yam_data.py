"""Randomised data scene for the sim-first YAM policy.

One compiled model, re-dressed every episode (no recompiles):

  objects   2-4 of the four present, anywhere in the top-down grasp band, sizes +-15 %, colours
            either near their nominal or fully random; the rest are parked off-stage
  user      head / torso / hand moved per episode; head size +-10 %
  camera    scene camera anywhere in a wide box on the robot's LEFT (+y), aimed near the table
            centre, FOV 50-75 deg, roll +-5 deg; rejected until it frames every object and the mouth
  dynamics  servo stiffness 0.6-1x, object friction and mass jitter, start pose +-0.08 rad
  images    arm + objects + user rendered; everything else replaced by a background photo; the
            user painted flat grey with a ragged edge; soft shadows under objects; then whole-image
            camera effects (obs_corrupt) sampled per episode, plus random erasing
  prompts   2D only, exactly what the real system can supply:
              object box  [cx, cy, w, h, conf, valid]  captured ONCE at selection time, jittered
              mouth       [mx, my, face_h, valid]      per tick, lagged with the image, held when lost

Observation = scene RGB, wrist RGB, state(17) = 6 joints + gripper(0..1) + box(6) + mouth(4).
Action      = 6 absolute joint targets (rad) + gripper command (0..1), one per control tick.

Env knobs: YAM_HZ (control/record rate, default 10), YAM_SCENE_RES (224), YAM_WRIST_RES (160),
YAM_BG (background dir), YAM_AUG (1 = camera effects on), YAM_LAT ("0 2" ticks of image lag).
"""
import os
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from obs_corrupt import corrupt                                   # noqa: E402
from yam_scene import HOME_Q, LIBRARY, YamScene                   # noqa: E402

HZ = float(os.environ.get("YAM_HZ", "10"))
SCENE_RES = int(os.environ.get("YAM_SCENE_RES", "224"))
WRIST_RES = int(os.environ.get("YAM_WRIST_RES", "160"))
BG_DIR = Path(os.environ.get("YAM_BG", Path(__file__).resolve().parent / "bg_train"))
AUG = os.environ.get("YAM_AUG", "1") == "1"
LAT = tuple(int(v) for v in os.environ.get("YAM_LAT", "0 2").split())
AUG_PROFILE = os.environ.get("YAM_AUG_PROFILE", "heavy")       # "heavy" = yam_v1; "real" for everything after
GREY = 127
STATE_DIM, ACTION_DIM = 17, 7
GRIP_Q_OPEN, GRIP_CTRL_OPEN = 0.0375, 0.041
PARK = np.array([4.0, 0.0, 0.2])


def lookat_quat(pos, target, roll=0.0):
    """MuJoCo camera quaternion: looks down -z, +y up, optionally rolled about the view axis."""
    z = np.asarray(pos, float) - np.asarray(target, float)
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c, s = np.cos(roll), np.sin(roll)
    x, y = c * x + s * y, -s * x + c * y
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.stack([x, y, z], axis=1).flatten())
    return q


class DataScene(YamScene):
    def __init__(self, bg_dir=BG_DIR, aug=AUG):
        super().__init__(objects=list(LIBRARY))
        m = self.model
        gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
        self.names = list(LIBRARY)
        self.obj_geom = {n: gid(n) for n in self.names}
        self.user_geoms = np.array([gid(n) for n in ("head", "nose", "torso", "hand")])
        arm_root = self._bid("arm")
        arm_bodies = set()
        for b in range(m.nbody):
            x = b
            while x != 0 and x != arm_root:
                x = m.body_parentid[x]
            if x == arm_root:
                arm_bodies.add(b)
        self.arm_geoms = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] in arm_bodies])
        self.keep = np.concatenate([self.arm_geoms, np.array(list(self.obj_geom.values())), self.user_geoms])
        self.cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "scene_cam")
        self.cam_body_pos = m.body_pos[m.cam_bodyid[self.cid]].copy()
        self.n_sub = max(1, int(round(1.0 / HZ / m.opt.timestep)))
        self.dt = self.n_sub * m.opt.timestep
        self.aug = aug
        self.bg_files = sorted(str(p) for p in Path(bg_dir).glob("*.jpg")) + sorted(str(p) for p in Path(bg_dir).glob("*.png"))
        if not self.bg_files:
            raise RuntimeError(f"no background images in {bg_dir}")
        # nominal values to randomise around
        self.size0 = {n: m.geom_size[g].copy() for n, g in self.obj_geom.items()}
        self.rgba0 = {n: m.geom_rgba[g].copy() for n, g in self.obj_geom.items()}
        self.mass0 = {n: float(m.body_mass[self._bid(n)]) for n in self.names}
        self.gain0, self.bias0 = m.actuator_gainprm.copy(), m.actuator_biasprm.copy()
        self.head_r0 = float(m.geom_size[gid("head")][0])
        self.user_pos0 = {b: m.body_pos[self._bid(b)].copy() for b in ("user_head", "user_torso", "user_hand")}
        self.light0 = (m.light_dir.copy(), m.light_diffuse.copy(), m.vis.headlight.ambient.copy(), m.vis.headlight.diffuse.copy())
        self._ren = {SCENE_RES: mujoco.Renderer(m, SCENE_RES, SCENE_RES)}
        self._ren.setdefault(WRIST_RES, mujoco.Renderer(m, WRIST_RES, WRIST_RES))
        self._seg = {r: mujoco.Renderer(m, r, r) for r in {SCENE_RES, WRIST_RES}}
        for r in self._seg.values():
            r.enable_segmentation_rendering()

    # ------------------------------------------------------------------ episode randomisation
    def randomize(self, seed):
        rng = self.rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        m.actuator_gainprm[:], m.actuator_biasprm[:] = self.gain0, self.bias0
        k = rng.uniform(0.6, 1.0)                                       # softer servos than the model's
        m.actuator_gainprm[:6, 0] *= k
        m.actuator_biasprm[:6, 1] *= k
        # --- user
        dx, dy, dz = rng.uniform(-0.06, 0.12), rng.uniform(-0.18, 0.18), rng.uniform(-0.08, 0.08)
        for b, p0 in self.user_pos0.items():
            m.body_pos[self._bid(b)] = p0 + np.array([dx, dy, dz if b == "user_head" else 0.0])
        hs = rng.uniform(0.9, 1.1)
        m.geom_size[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "head")][0] = self.head_r0 * hs
        # --- objects: which, where, how big, what colour
        n_present = int(rng.integers(2, 5))
        present = list(rng.choice(self.names, size=n_present, replace=False))
        self.target = str(rng.choice(present))
        mujoco.mj_resetDataKeyframe(m, d, 0)
        placed, specs = [], []
        for n in self.names:
            g, a = self.obj_geom[n], self.object_qadr(n)
            s = min(rng.uniform(0.85, 1.15), 0.072 / LIBRARY[n].width)      # must still fit the 83 mm jaws
            size = self.size0[n] * s
            if n in present:
                for _ in range(200):
                    r, az = rng.uniform(0.25, 0.45), rng.uniform(-0.70, 0.70)
                    p = np.array([r * np.cos(az), r * np.sin(az)])
                    if all(np.linalg.norm(p - q) > 0.10 for q in placed):
                        break
                placed.append(p)
            m.geom_size[g] = size
            m.geom_rbound[g] = float(np.linalg.norm(size)) if LIBRARY[n].kind == "box" else float(max(size[0], np.hypot(size[0], size[1]) if LIBRARY[n].kind == "cylinder" else size[0]))
            m.geom_aabb[g, 3:] = size if LIBRARY[n].kind == "box" else (np.array([size[0], size[0], size[1]]) if LIBRARY[n].kind == "cylinder" else np.full(3, size[0]))
            o = replace(LIBRARY[n], size=tuple(float(v) for v in (size[:1] if LIBRARY[n].kind == "sphere" else size[:2] if LIBRARY[n].kind == "cylinder" else size)))
            specs.append(o)
            d.qpos[a:a + 3] = (p[0], p[1], o.rest_z + 0.001) if n in present else PARK + [0, 0.3 * self.names.index(n), 0]
            d.qpos[a + 3:a + 7] = (1, 0, 0, 0)
            m.geom_rgba[g, :3] = (np.clip(self.rgba0[n][:3] * rng.uniform(0.75, 1.2, 3), 0, 1) if rng.random() < 0.5
                                  else cv2.cvtColor(np.uint8([[[rng.integers(0, 180), rng.integers(60, 255), rng.integers(70, 245)]]]), cv2.COLOR_HSV2RGB)[0, 0] / 255.0)
            m.geom_friction[g, 0] = rng.uniform(0.7, 1.3)
            m.body_mass[self._bid(n)] = self.mass0[n] * rng.uniform(0.7, 1.3)
        self.objects = specs                                               # the expert reads sizes from here
        self.present = present
        d.qpos[self.arm_qadr] = np.array(HOME_Q) + rng.uniform(-0.08, 0.08, 6)
        d.ctrl[:6] = d.qpos[self.arm_qadr]
        d.ctrl[6] = GRIP_CTRL_OPEN
        # --- lighting
        ld, lf, ha, hd = self.light0
        m.light_dir[:] = ld + rng.uniform(-0.35, 0.35, ld.shape)
        m.light_diffuse[:] = np.clip(lf * rng.uniform(0.5, 1.3), 0, 1)
        m.vis.headlight.ambient[:] = np.clip(ha * rng.uniform(0.6, 1.7), 0, 1)
        m.vis.headlight.diffuse[:] = np.clip(hd * rng.uniform(0.6, 1.4), 0, 1)
        mujoco.mj_forward(m, d)
        # --- scene camera: wide box on the robot's left, must frame every present object and the mouth
        must = [self.object_pos(n) for n in present] + [self.site("mouth")]
        for _ in range(300):
            pos = np.array([rng.uniform(-0.35, 0.15), rng.uniform(0.20, 0.65), rng.uniform(0.30, 0.75)])
            tgt = np.array([0.40, 0.0, 0.12]) + rng.uniform(-0.08, 0.08, 3)
            fovy, roll = rng.uniform(50, 75), np.radians(rng.uniform(-5, 5))
            m.cam_pos[self.cid] = pos - self.cam_body_pos
            m.cam_quat[self.cid] = lookat_quat(pos, tgt, roll)
            m.cam_fovy[self.cid] = fovy
            mujoco.mj_forward(m, d)
            uv = np.array([self.project(P) for P in must])
            if np.all((uv > 0.06) & (uv < 0.94)):
                break
        # --- per-episode image pipeline
        self.bg = {"scene": self._background(SCENE_RES), "wrist": self._background(WRIST_RES)}
        self.ops = self._sample_ops() if self.aug else {}
        self.shadow = dict(dir=rng.uniform(-1, 1, 2) * 0.03, op=rng.uniform(0.15, 0.5)) if rng.random() < 0.8 else None
        self.mask_jit = int(rng.integers(-3, 4))
        self.lat = int(rng.integers(LAT[0], LAT[1] + 1))
        self._frames, self._mouth_hist = [], []
        self._mouth_last = np.array([0.5, 0.5, 0.2, 0.0], np.float32)
        self._drop = 0
        self.box = None
        return self.target

    # ------------------------------------------------------------------ image pipeline
    def _background(self, res):
        rng = self.rng
        if rng.random() < 0.15:                                            # plain / gradient tabletop-like fields
            a, b = rng.uniform(40, 235, 3), rng.uniform(40, 235, 3)
            t = np.linspace(0, 1, res, dtype=np.float32)[:, None, None] if rng.random() < 0.5 else np.linspace(0, 1, res, dtype=np.float32)[None, :, None]
            img = (a * (1 - t) + b * t) + rng.normal(0, rng.uniform(0, 6), (res, res, 3))
            return np.clip(np.broadcast_to(img, (res, res, 3)), 0, 255).astype(np.uint8)
        img = cv2.imread(self.bg_files[int(rng.integers(len(self.bg_files)))], cv2.IMREAD_COLOR)[..., ::-1]
        h, w = img.shape[:2]
        s = rng.uniform(0.35, 1.0)                                          # random crop: zoom varies texture scale
        ch, cw = max(8, int(h * s)), max(8, int(min(w, h * s)))
        y0, x0 = int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1))
        img = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (res, res), interpolation=cv2.INTER_AREA)
        if rng.random() < 0.5:
            img = img[:, ::-1]
        img = img.astype(np.float32) * rng.uniform(0.6, 1.3) * rng.uniform(0.9, 1.1, 3)
        return np.clip(img, 0, 255).astype(np.uint8)

    def _sample_ops(self):
        """Whole-image camera effects for this episode. Two profiles (YAM_AUG_PROFILE):

        heavy  the first attempt (dataset yam_v1): blur to sigma 2.6 px on 70 % of episodes. Sized to
               cover the SO-101 blur cliff, but at 224 px that is far beyond any real camera and it
               erases small objects - kept only so yam_v1's evaluations stay comparable.
        real   what a phone or UVC camera actually delivers once downscaled to the policy's input:
               mostly sharp, mild exposure / white-balance drift, light noise, video compression,
               occasional slight defocus or motion blur. The default for everything after yam_v1.
        """
        rng, ops = self.rng, {}
        if AUG_PROFILE == "heavy":
            ops["exposure"], ops["gamma"] = rng.uniform(0.55, 1.6), rng.uniform(0.75, 1.3)
            if rng.random() < 0.7: ops["blur"] = rng.uniform(0.3, 2.6)
            if rng.random() < 0.2: ops["mblur"] = rng.uniform(3, 9)
            if rng.random() < 0.8: ops["noise"] = rng.uniform(0.005, 0.055)
            if rng.random() < 0.3: ops["vignette"] = rng.uniform(0.1, 0.45)
            if rng.random() < 0.7: ops["jpeg"] = rng.uniform(25, 95)
            ops["_wb"] = rng.uniform(0.82, 1.2, 3)
            ops["_erase"] = int(rng.integers(0, 3)) if rng.random() < 0.3 else 0
            return ops
        ops["exposure"], ops["gamma"] = rng.uniform(0.7, 1.4), rng.uniform(0.85, 1.2)
        if rng.random() < 0.35: ops["blur"] = float(np.clip(abs(rng.normal(0, 0.6)), 0.2, 1.6))
        if rng.random() < 0.15: ops["mblur"] = rng.uniform(3, 6)
        if rng.random() < 0.7: ops["noise"] = rng.uniform(0.004, 0.03)
        if rng.random() < 0.2: ops["vignette"] = rng.uniform(0.1, 0.35)
        if rng.random() < 0.6: ops["jpeg"] = rng.uniform(45, 95)
        ops["_wb"] = rng.uniform(0.88, 1.14, 3)
        ops["_erase"] = 1 if rng.random() < 0.15 else 0
        return ops

    def project(self, P):
        """World point -> normalised (u, v) in the scene camera, origin top-left."""
        d, m = self.data, self.model
        R = d.cam_xmat[self.cid].reshape(3, 3)
        c = (np.asarray(P, float) - d.cam_xpos[self.cid]) @ R
        f = 0.5 / np.tan(np.radians(m.cam_fovy[self.cid]) / 2)
        if c[2] > -1e-6:
            return np.array([-1.0, -1.0])
        return np.array([0.5 + f * c[0] / -c[2], 0.5 - f * c[1] / -c[2]])

    def _shot(self, cam, res, bg):
        self._ren[res].update_scene(self.data, camera=cam)
        rgb = self._ren[res].render()
        self._seg[res].update_scene(self.data, camera=cam)
        ids = self._seg[res].render()[:, :, 0]
        out = bg.copy()
        if self.shadow is not None and cam == "scene_cam":                  # soft blobs under objects still on the table
            sh = np.zeros((res, res), np.float32)
            for n in self.present:
                p = self.object_pos(n)
                if p[2] < self.spec(n).rest_z + 0.015:
                    uv = self.project([p[0] + self.shadow["dir"][0], p[1] + self.shadow["dir"][1], 0.0]) * res
                    rad = self.project([p[0] + self.spec(n).width / 2, p[1], 0.0]) * res
                    r = int(max(2, np.linalg.norm(rad - self.project([p[0], p[1], 0.0]) * res) * 1.3))
                    if np.all(np.isfinite(uv)) and 0 <= uv[0] < res and 0 <= uv[1] < res and r < res // 3:
                        cv2.ellipse(sh, (int(uv[0]), int(uv[1])), (r, max(1, r // 2)), 0, 0, 360, 1.0, -1)
            if sh.any():
                sh = cv2.GaussianBlur(sh, (0, 0), 2.5)[..., None] * self.shadow["op"]
                out = (out.astype(np.float32) * (1 - sh)).astype(np.uint8)
        keep = np.isin(ids, self.keep)
        out[keep] = rgb[keep]
        person = np.isin(ids, self.user_geoms).astype(np.uint8)             # the user: flat grey, ragged edge
        if person.any():
            if self.mask_jit:
                k = np.ones((2 * abs(self.mask_jit) + 1,) * 2, np.uint8)
                person = cv2.dilate(person, k) if self.mask_jit > 0 else cv2.erode(person, k)
            out[person.astype(bool) & ~np.isin(ids, self.arm_geoms)] = GREY
        if self.ops:
            ops = {k: v for k, v in self.ops.items() if not k.startswith("_")}
            x = out.astype(np.float32) * self.ops["_wb"]
            out = corrupt(np.clip(x, 0, 255).astype(np.uint8), ops)
            for _ in range(self.ops["_erase"]):
                ew, eh = self.rng.integers(res // 12, res // 4, 2)
                ex, ey = self.rng.integers(0, res - ew), self.rng.integers(0, res - eh)
                out[ey:ey + eh, ex:ex + ew] = self.rng.integers(0, 255, 3)
        return out, ids

    # ------------------------------------------------------------------ prompts
    def capture_box(self):
        """The selection-time detector box for the target, from visible pixels. None if hidden."""
        _, ids = self._shot("scene_cam", SCENE_RES, self.bg["scene"])
        ys, xs = np.nonzero(ids == self.obj_geom[self.target])
        if len(xs) < 10:
            return None
        rng, R = self.rng, float(SCENE_RES)
        x0, x1, y0, y1 = xs.min() / R, (xs.max() + 1) / R, ys.min() / R, (ys.max() + 1) / R
        w, h = (x1 - x0) * rng.uniform(0.85, 1.25), (y1 - y0) * rng.uniform(0.85, 1.25)
        cx, cy = (x0 + x1) / 2 + rng.normal(0, 0.012), (y0 + y1) / 2 + rng.normal(0, 0.012)
        self.box = np.array([cx, cy, w, h, rng.uniform(0.45, 1.0), 1.0], np.float32)
        return self.box

    def _mouth_prompt(self, ids):
        rng, m = self.rng, self.model
        uv = self.project(self.site("mouth"))
        ok = bool(np.all((uv > 0.0) & (uv < 1.0)))
        if ok:
            px = (uv * SCENE_RES).astype(int).clip(0, SCENE_RES - 1)
            ok = ids[px[1], px[0]] not in self.arm_geoms                    # a face detector loses an occluded face
        if self._drop > 0:
            self._drop -= 1
            ok = False
        elif rng.random() < 0.02:
            self._drop = int(rng.integers(1, 5))
        if ok:
            top = self.project(self.site("face") + np.array([0, 0, self.head_r0]))
            bot = self.project(self.site("face") - np.array([0, 0, self.head_r0]))
            fh = abs(bot[1] - top[1]) * rng.uniform(0.92, 1.08)
            self._mouth_last = np.array([uv[0] + rng.normal(0, 0.008), uv[1] + rng.normal(0, 0.008), fh, 1.0], np.float32)
        else:
            self._mouth_last = np.append(self._mouth_last[:3], 0.0).astype(np.float32)
        return self._mouth_last

    # ------------------------------------------------------------------ one control tick of observation
    def observe(self):
        scene, ids = self._shot("scene_cam", SCENE_RES, self.bg["scene"])
        wrist, _ = self._shot("wrist_cam", WRIST_RES, self.bg["wrist"])
        self._frames.append((scene, wrist, self._mouth_prompt(ids)))
        if len(self._frames) > self.lat + 1:
            self._frames.pop(0)
        scene, wrist, mouth = self._frames[0]                               # cameras lag; proprioception does not
        jl = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_finger")]
        grip = float(np.clip(self.data.qpos[jl] / GRIP_Q_OPEN, 0, 1))
        state = np.concatenate([self.q_arm, [grip], self.box, mouth]).astype(np.float32)
        return scene, wrist, state

    @staticmethod
    def to_ctrl(action):
        """Policy action (6 rad + gripper 0..1) -> MuJoCo ctrl."""
        c = np.array(action, float)
        c[6] = np.clip(c[6], 0, 1) * GRIP_CTRL_OPEN
        return c

    def tick(self, action, prev_action):
        """Run one control period, interpolating the command linearly — the same way the real
        30 Hz loop will be fed from a slower policy — so demos and deployment share dynamics."""
        a0, a1 = self.to_ctrl(prev_action), self.to_ctrl(action)
        for i in range(1, self.n_sub + 1):
            self.data.ctrl[:7] = a0 + (a1 - a0) * (i / self.n_sub)
            mujoco.mj_step(self.model, self.data)
