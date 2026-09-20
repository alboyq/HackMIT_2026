"""The YAM agentic-manipulation scene: arm + wrist cam + base-mounted scene cam + table
+ several graspable objects + a seated user with a mouth and a hand.

Built the same way rl/jar/object_spec.py builds the SO-101 scenes: the MJCF is GENERATED
from a spec and written next to the robot model (mesh paths in mujoco_menagerie/i2rt_yam
resolve relative to the file that includes yam.xml), with a content hash in the name so
parallel workers with different specs never overwrite each other.

Two things the SO-101 scenes did not have:

  * a WRIST CAMERA on link_6. It looks down the tool axis from behind the jaws, so its pose
    relative to the gripper is welded — the one view that survives the rig being moved to a
    different room.
  * a USER: head + torso bodies carrying a `mouth` site and a `hand` site. The head has to
    exist as geometry because it is what the planner must not hit; nothing learned ever
    looks at a face (MediaPipe finds the mouth on the real camera frame, and here the site
    position is simply known).

The scene camera is mounted on a post attached to the ROBOT's base, not to the table, so its
extrinsics are a property of the arm rather than of the venue.

Grasp envelope, IK-verified on this model 2026-09-19 (solve_best at 5 azimuths over +-40 deg,
TCP = the point between the pads, not grasp_site):

  tool straight down        r 0.15-0.50 m   <- the top-down grasp zone; put objects here
  tool tilted 30 deg        r 0.15-0.65 m   <- fallback for objects further out
  tool level (side grasp)   r 0.65-0.75 m   <- only works FAR out; the arm cannot fold back
  tool level at z = 0.38 m  r 0.15-0.70 m   <- mouth-height delivery, easy everywhere

An earlier pass reported 0.55 / 0.78 m from random-sample FK, but it used the site z-axis as a
proxy for the tool axis and grasp_site as the TCP; both are wrong on this model (see ToolFrame).
The numbers above are the ones to design against.

Env knobs: YAM_OBJECTS (comma-separated subset), YAM_USER_X/Y/Z (seat position),
YAM_CAM_RES, YAM_WRIST_FOVY, YAM_SCENE_FOVY.
"""
import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

def _find_menagerie() -> Path:
    """Locate mujoco_menagerie/i2rt_yam. YAM_MENAGERIE wins; otherwise walk up from this file
    and try the usual checkout spots, so the same code runs on the Mac and on the GX10."""
    env = os.environ.get("YAM_MENAGERIE")
    if env:
        p = Path(env)
        return p if p.name == "i2rt_yam" else p / "i2rt_yam"
    here = Path(__file__).resolve()
    for base in [*here.parents, Path.cwd()]:
        for cand in (base / "mujoco_menagerie" / "i2rt_yam", base / "i2rt_yam"):
            if (cand / "yam.xml").exists():
                return cand
    raise RuntimeError(
        "cannot find mujoco_menagerie/i2rt_yam. Clone it "
        "(git clone https://github.com/google-deepmind/mujoco_menagerie) and either put it "
        "beside this checkout or set YAM_MENAGERIE=/path/to/mujoco_menagerie")


YAMDIR = _find_menagerie()
# Which gripper the arm carries. "linear_4310" is what is on the bench: rack-and-pinion sliding
# jaws on a DM4310, 95 mm throw, matching I2RT's published spec. The stock menagerie yam.xml
# carries crank_4310 instead (79 mm) — see rl/yam/make_arm_linear4310.py. YAM_ARM=stock reverts.
_VARIANT = os.environ.get("YAM_ARM", "linear_4310")   # the gripper actually on the bench; "stock" = menagerie crank_4310
ARM_XML = YAMDIR / ("yam.xml" if _VARIANT == "stock" else f"_yam_{_VARIANT.replace('_', '')}.xml")
if not ARM_XML.exists() and _VARIANT != "stock":
    raise RuntimeError(f"{ARM_XML} missing - run: python rl/yam/make_arm_linear4310.py")

CAM_RES = int(os.environ.get("YAM_CAM_RES", "256"))
WRIST_FOVY = float(os.environ.get("YAM_WRIST_FOVY", "125"))   # wide pinhole source for the fisheye warp
# wrist camera mount in the link_6 frame (x = jaw axis, +y = the top of the hand, z = along the forearm):
# (y, z) position and the aim point past the TCP, which sits at (0, -0.044, 0.130)
WRIST_POS = tuple(float(v) for v in os.environ.get("YAM_WRIST_POS", "0.068 0.080").split())
WRIST_AIM = tuple(float(v) for v in os.environ.get("YAM_WRIST_AIM", "-0.050 0.240").split())


def _wrist_xyaxes():
    import numpy as _np
    pos, aim = _np.array([0.0, *WRIST_POS]), _np.array([0.0, *WRIST_AIM])
    z = pos - aim; z /= _np.linalg.norm(z)
    x = _np.array([-1.0, 0.0, 0.0])                       # image right = -jaw axis so the view is upright at rest
    y = _np.cross(z, x); y /= _np.linalg.norm(y)
    x = _np.cross(y, z)
    return " ".join(f"{v:.5f}" for v in _np.concatenate([x, y]))


WRIST_XYAXES = _wrist_xyaxes()
SCENE_FOVY = float(os.environ.get("YAM_SCENE_FOVY", "58"))
GRIP_KP = float(os.environ.get("YAM_GRIP_KP", "800"))    # see _patched_arm
GRIP_KV = float(os.environ.get("YAM_GRIP_KV", "30"))

# the user, seated across the table from the arm (metres, robot base frame)
USER_X = float(os.environ.get("YAM_USER_X", "0.62"))
USER_Y = float(os.environ.get("YAM_USER_Y", "0.0"))
USER_Z = float(os.environ.get("YAM_USER_Z", "0.42"))     # mouth-ish height above the table
HEAD_R = 0.095


@dataclass
class ObjectDef:
    """One graspable object. `grasp` is the strategy the planner should prefer; the planner
    still checks reachability and falls back to the other one."""
    name: str
    kind: str                      # sphere | cylinder | box
    size: tuple                    # sphere (r,) | cylinder (r, half_h) | box (hx, hy, hz)
    rgba: tuple
    mass: float
    grasp: str = "top"             # top | side
    pos: tuple = (0.35, 0.0)       # x, y on the table
    prompt: str = ""               # the noun phrase a segmentation model would be given

    @property
    def rest_z(self):
        if self.kind == "sphere":
            return self.size[0]
        if self.kind == "cylinder":
            return self.size[1]
        return self.size[2]

    @property
    def width(self):
        """Widest dimension across the jaw axis — must fit the 95 mm gripper throw."""
        if self.kind == "sphere":
            return 2 * self.size[0]
        if self.kind == "cylinder":
            return 2 * self.size[0]
        return 2 * max(self.size[0], self.size[1])

    @property
    def height(self):
        if self.kind == "sphere":
            return 2 * self.size[0]
        if self.kind == "cylinder":
            return 2 * self.size[1]
        return 2 * self.size[2]


# A deliberately small, distinct set: one of each grasp geometry the planner has to handle.
LIBRARY = {
    "apple":  ObjectDef("apple", "sphere", (0.033,), (0.78, 0.13, 0.11, 1), 0.15,
                        grasp="top", pos=(0.36, 0.12), prompt="red apple"),
    "mug":    ObjectDef("mug", "cylinder", (0.032, 0.048), (0.93, 0.93, 0.95, 1), 0.30,
                        grasp="side", pos=(0.31, -0.16), prompt="white mug"),
    "marker": ObjectDef("marker", "cylinder", (0.009, 0.062), (0.15, 0.35, 0.75, 1), 0.02,
                        grasp="top", pos=(0.44, -0.03), prompt="blue marker pen"),
    "block":  ObjectDef("block", "box", (0.025, 0.025, 0.030), (0.95, 0.72, 0.15, 1), 0.10,
                        grasp="top", pos=(0.40, 0.22), prompt="yellow wooden block"),
    # what is actually on the team's table: slim drinks cans (53-58 mm across, 12-16 cm tall)
    "can":    ObjectDef("can", "cylinder", (0.028, 0.070), (0.80, 0.82, 0.85, 1), 0.28,
                        grasp="top", pos=(0.27, 0.04), prompt="drinks can"),
}
DEFAULT_OBJECTS = ("apple", "mug", "marker", "block", "can")


def _patched_arm() -> Path:
    """yam.xml + a wrist camera on link_6, minus the stock keyframe (which is sized for the
    bare arm and goes stale as soon as the scene adds free bodies)."""
    src = ARM_XML.read_text()
    # Anchor on tcp_site, NOT grasp_site. Both sit in link_6 on the menagerie arm, but the
    # linear_4310 graft moves grasp_site down into the `gripper_mount` body, which carries a
    # 90 deg rotation — anchoring there put the wrist camera in the mount's frame, pointing
    # 124 deg away from the TCP, and the wrist view was nothing but gripper. tcp_site stays in
    # link_6 in both variants, which is the frame WRIST_POS/WRIST_AIM are expressed in.
    anchor = '<site name="tcp_site"'
    if anchor not in src:
        raise RuntimeError(f"{ARM_XML} has no tcp_site — model changed?")
    # Mount measured off the model, not guessed: in link_6 local coords the gripper subtree
    # occupies x [-0.02,0.04] y [-0.049,0.039] z [0.03,0.138] and the TCP sits at
    # (0,-0.044,0.130). The camera goes on the +y side — the back of the hand, perpendicular
    # to both the tool axis and the jaw axis, which is where a real wrist bracket goes —
    # 6 cm back along the tool, aimed just past the TCP. Chosen by mj_ray line-of-sight
    # test over a grid of mounts: this one sees the TCP and points 5 and 12 cm in front of it
    # with no gripper geometry in the way (19 of 84 candidate mounts were clear).
    cam = (f'<camera name="wrist_cam" pos="0 {WRIST_POS[0]:g} {WRIST_POS[1]:g}" xyaxes="{WRIST_XYAXES}" '
           f'fovy="{WRIST_FOVY:g}" resolution="{CAM_RES} {CAM_RES}"/>\n                  ')
    src = src.replace(anchor, cam + anchor, 1)
    # Grip strength. The menagerie model gives the finger position actuator kp=100, which
    # measures out at only ~1.2 N per pad when commanded hard against an object — a 150 g
    # apple slides out of the jaws during a gentle lift. That is a placeholder value, not a
    # spec: the real YAM is rated for a 2 kg payload, which needs ~20 N of grip at mu ~ 1.
    # kp=800 gives ~23 N at the joint and ~10 N per pad through the finger linkage.
    # regex, not an exact string: the ctrlrange differs between gripper variants (0.041 crank,
    # 0.0475 linear_4310) and an exact-match replace silently left kp at 100 = ~1.2 N of grip.
    src, n = re.subn(r'(<position ctrlrange="[^"]*")\s+kp="[\d.]+"\s+kv="[\d.]+"',
                     rf'\1 kp="{GRIP_KP:g}" kv="{GRIP_KV:g}"', src)
    if n != 1:
        raise RuntimeError(f"expected exactly one finger <position> to re-gain, patched {n}")
    k0 = src.find("<keyframe>")
    if k0 != -1:
        k1 = src.index("</keyframe>") + len("</keyframe>")
        src = src[:k0] + src[k1:]
    return _write(src, "yam_wristcam")


def _write(xml: str, stem: str) -> Path:
    out = YAMDIR / f"_{stem}_{hashlib.md5(xml.encode()).hexdigest()[:8]}.xml"
    if not out.exists():                                   # atomic: parallel workers race here
        tmp = out.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(xml)
        tmp.replace(out)
    return out


def _object_xml(o: ObjectDef) -> str:
    rgba = " ".join(f"{v:g}" for v in o.rgba)
    if o.kind == "sphere":
        geom = f'type="sphere" size="{o.size[0]:.5f}"'
    elif o.kind == "cylinder":
        geom = f'type="cylinder" size="{o.size[0]:.5f} {o.size[1]:.5f}"'
    else:
        geom = f'type="box" size="{o.size[0]:.5f} {o.size[1]:.5f} {o.size[2]:.5f}"'
    return (f'    <body name="{o.name}" pos="{o.pos[0]:.4f} {o.pos[1]:.4f} {o.rest_z:.5f}">\n'
            f'      <freejoint name="{o.name}_free"/>\n'
            f'      <geom name="{o.name}" {geom} rgba="{rgba}" mass="{o.mass:.4f}" '
            f'condim="4" friction="1 .01 .001" solref="0.01 1"/>\n'
            f'    </body>\n')


def build_scene_xml(objects=None, user=(USER_X, USER_Y, USER_Z)) -> str:
    objs = [LIBRARY[n] for n in (objects or DEFAULT_OBJECTS)]
    arm = _patched_arm().name
    ux, uy, uz = user
    bodies = "".join(_object_xml(o) for o in objs)
    return f"""<mujoco model="yam_agentic">
  <include file="{arm}"/>
  <statistic center="0.35 0 0.35" extent="1.1"/>
  <visual>
    <global azimuth="150" elevation="-22"/>
    <headlight diffuse="0.5 0.5 0.5" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <quality shadowsize="2048"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.24 0.28 0.33" rgb2="0.08 0.09 0.11" width="512" height="512"/>
    <texture name="tabletex" type="2d" builtin="checker" rgb1="0.72 0.60 0.43" rgb2="0.67 0.55 0.39"
             width="400" height="400"/>
    <material name="table" texture="tabletex" texrepeat="10 10" specular="0.1" shininess="0.1"/>
    <material name="skin" rgba="0.82 0.66 0.55 1"/>
    <material name="shirt" rgba="0.27 0.33 0.45 1"/>
  </asset>
  <worldbody>
    <light name="key" pos="0.3 0.4 1.6" dir="-0.1 -0.2 -1" directional="true" diffuse="0.6 0.6 0.6"/>
    <light name="fill" pos="0.5 -0.6 1.2" dir="-0.2 0.4 -1" directional="true" diffuse="0.3 0.3 0.3"/>
    <geom name="table" type="plane" size="1.6 1.6 0.01" material="table" condim="3" friction="1 .005 .0001"/>

{bodies}
    <!-- seated user: geometry only. Nothing learned looks at a face; the head exists so the
         planner has something to avoid and the virtual wall has something to wrap. -->
    <body name="user_head" pos="{ux:.4f} {uy:.4f} {uz:.4f}">
      <geom name="head" type="sphere" size="{HEAD_R:.4f}" material="skin"/>
      <geom name="nose" type="capsule" fromto="{-HEAD_R * 0.92:.4f} 0 -0.005 {-HEAD_R * 1.12:.4f} 0 -0.020"
            size="0.012" material="skin" contype="0" conaffinity="0"/>
      <site name="mouth" pos="{-HEAD_R * 0.92:.4f} 0 -0.038" size="0.010" rgba="0.8 0.15 0.15 1" group="4"/>
      <site name="face" pos="{-HEAD_R:.4f} 0 0" size="0.008" rgba="0.8 0.5 0.15 1" group="4"/>
    </body>
    <body name="user_torso" pos="{ux + 0.12:.4f} {uy:.4f} {max(0.20, uz - 0.22):.4f}">
      <geom name="torso" type="box" size="0.10 0.17 0.19" material="shirt"/>
    </body>
    <body name="user_hand" pos="{ux - 0.20:.4f} {uy - 0.26:.4f} 0.035">
      <geom name="hand" type="box" size="0.045 0.035 0.018" material="skin"/>
      <site name="hand" pos="0 0 0.030" size="0.010" rgba="0.15 0.55 0.35 1" group="4"/>
    </body>

    <!-- scene camera on a post rigidly attached to the robot's own base plate -->
    <body name="cam_post" pos="-0.06 0.30 0">
      <geom name="post" type="capsule" fromto="0 0 0 0.02 -0.03 0.46" size="0.010" rgba="0.25 0.25 0.27 1"
            contype="0" conaffinity="0"/>
      <camera name="scene_cam" pos="0.02 -0.03 0.47" xyaxes="-0.53330 -0.84593 0 0.50927 -0.32106 0.79847"
              fovy="{SCENE_FOVY:g}" resolution="{CAM_RES} {CAM_RES}"/>
    </body>
  </worldbody>

  <keyframe>
    <key name="home" qpos="{_home_qpos_str(objs, user)}" ctrl="{' '.join(f'{v:g}' for v in HOME_Q)} 0.041"/>
  </keyframe>
</mujoco>
"""


# ready pose: TCP at (0.30, 0, 0.26) with the tool tilted 25 deg off vertical, solved by IK.
# Straight down at that height is NOT reachable (joint4 hits its -90 deg stop); 25 deg is.
HOME_Q = (-0.0279, 1.2776, 1.4870, -1.0699, -0.1920, -0.5637)


def _home_qpos_str(objs, user):
    q = list(HOME_Q) + [0.0374, -0.0374]                           # 6 joints + both fingers open
    for o in objs:
        q += [o.pos[0], o.pos[1], o.rest_z, 1, 0, 0, 0]
    return " ".join(f"{v:g}" for v in q)


class YamScene:
    """Thin wrapper: model/data, named lookups, a renderer per camera."""

    def __init__(self, objects=None, user=(USER_X, USER_Y, USER_Z), res=CAM_RES):
        names = list(objects or os.environ.get("YAM_OBJECTS", "").split(",") or DEFAULT_OBJECTS)
        names = [n for n in names if n] or list(DEFAULT_OBJECTS)
        self.objects = [LIBRARY[n] for n in names]
        self.user = user
        self.path = _write(build_scene_xml(names, user), "yam_scene")
        self.model = mujoco.MjModel.from_xml_path(str(self.path))
        self.data = mujoco.MjData(self.model)
        self.res = res
        self._renderer = None
        self.nu = self.model.nu                                     # 7 = 6 arm + gripper
        self.arm_qadr = [self.model.jnt_qposadr[self._jid(f"joint{i}")] for i in range(1, 7)]
        self.grasp_site = self._sid("grasp_site")
        self.reset()

    # -------------------------------------------------------------- lookups
    def _jid(self, n): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
    def _sid(self, n): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n)
    def _bid(self, n): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)

    def site(self, name):
        return self.data.site_xpos[self._sid(name)].copy()

    def object_pos(self, name):
        return self.data.xpos[self._bid(name)].copy()

    def object_qadr(self, name):
        return self.model.jnt_qposadr[self._jid(f"{name}_free")]

    def spec(self, name):
        return next(o for o in self.objects if o.name == name)

    @property
    def q_arm(self):
        return self.data.qpos[self.arm_qadr].copy()

    # -------------------------------------------------------------- sim
    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        return self.data

    def step(self, ctrl, n=1):
        self.data.ctrl[:len(ctrl)] = ctrl
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def render(self, camera="scene_cam", res=None):
        res = res or self.res
        if self._renderer is None or self._renderer.height != res:
            self._renderer = mujoco.Renderer(self.model, res, res)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def segment(self, camera="scene_cam", res=None):
        """Exact per-geom segmentation — free in sim, and the ground truth we score the real
        segmentation model against."""
        res = res or self.res
        if self._renderer is None or self._renderer.height != res:
            self._renderer = mujoco.Renderer(self.model, res, res)
        self._renderer.enable_segmentation_rendering()
        self._renderer.update_scene(self.data, camera=camera)
        seg = self._renderer.render()[:, :, 0].copy()
        self._renderer.disable_segmentation_rendering()
        return seg


if __name__ == "__main__":
    s = YamScene()
    print(f"scene: {s.path.name}")
    print(f"nq={s.model.nq} nu={s.model.nu} ncam={s.model.ncam} nbody={s.model.nbody}")
    print("objects:", ", ".join(f"{o.name}({o.kind}, w={o.width * 1000:.0f}mm, h={o.height * 1000:.0f}mm)"
                                for o in s.objects))
    print("grasp_site:", np.round(s.data.site_xpos[s.grasp_site], 3))
    print("mouth:", np.round(s.site("mouth"), 3), " hand:", np.round(s.site("hand"), 3))
    for cam in ("scene_cam", "wrist_cam"):
        import imageio.v3 as iio
        out = f"/private/tmp/claude-501/-Users-adipu-so101Sim/0bbaecf6-c03b-4137-b839-fa8a809e7763/scratchpad/yam_{cam}.png"
        iio.imwrite(out, s.render(cam, 384))
        print("wrote", out)
