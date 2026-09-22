"""The graspable object, described ONCE (rl/real/object.json) and used by every tool:
sim scene, side-grasp expert, camera jar-finder, demo generation, real deployment.

  {"kind": "post", "diam": 0.038, "height": 0.062,          upright cylinder (m)
   "base_w": 0.055, "base_l": 0.040, "base_t": 0.004,       base plate under it: w x l (l defaults to w = square), 0 = none
   "bolt_d": 0.00635, "bolt_h": 0.008,                      bolt sticking out of the top (visual only)
   "mass": 0.04, "rgb": [0.93, 0.93, 0.93]}                 rgb given -> colour DR stays near it

kind "jar" = the original spice jar scene (scene_jar_rl.xml), kept so old runs reproduce.
OBJECT_SPEC=/path/to.json overrides the default file. The body and the cylinder geom keep the name
"box" (the lift env and the experts look the object up by that name); the body origin is the
cylinder's centre, so the body rests at z = base_t + height / 2  (= spec.rest_z).

Workspace: rl/real/workspace.json {"wall_y": -0.06} (or WALL_Y=-0.06; WALL_Y=none disables) puts an INVISIBLE
wall (render group 3, never composited) at that y: the partition on the robot's right at the new desk. It is a
world geom, so the side-grasp planner's "arm vs table" collision check rejects plans that would touch it, and
the sim physics stops the arm there too.
"""
import hashlib
import json
import os
from pathlib import Path

MENAGERIE = Path("/Users/adipu/so101Sim/mujoco_menagerie/robotstudio_so101")
DEFAULT_FILE = Path("/Users/adipu/so101Sim/rl/real/object.json")
WORKSPACE_FILE = Path("/Users/adipu/so101Sim/rl/real/workspace.json")
JAR = dict(kind="jar", diam=0.051, height=0.102, base_w=0.0, base_l=None, base_t=0.0, bolt_d=0.0, bolt_h=0.0, mass=0.11, rgb=None)


class ObjectSpec:
    def __init__(self, path=None):
        path = Path(path or os.environ.get("OBJECT_SPEC") or DEFAULT_FILE)
        s = dict(JAR)
        if path.exists():
            s = {**dict(JAR, kind="post"), **json.loads(path.read_text())}
        self.path, self.s = path, s
        self.kind = s["kind"]
        self.radius, self.height = s["diam"] / 2.0, float(s["height"])
        self.base_w, self.base_t = float(s["base_w"]), float(s["base_t"])
        self.base_l = float(s["base_l"]) if s.get("base_l") else self.base_w        # rectangular plate: w (body x) by l (body y)
        self.bolt_d, self.bolt_h = float(s["bolt_d"]), float(s["bolt_h"])
        self.mass, self.rgb = float(s["mass"]), s.get("rgb")
        self.rest_z = self.base_t + self.height / 2.0            # body origin height when standing on the table
        self.top_z = self.base_t + self.height + self.bolt_h
        self.foot_r = max(self.radius, 0.5 * (self.base_w ** 2 + self.base_l ** 2) ** 0.5)   # farthest point of the footprint from the axis
        self.square = abs(self.base_w - self.base_l) < 1e-4                          # silhouette repeats every 90 deg (else 180)
        w = os.environ.get("WALL_Y")
        if w is None and WORKSPACE_FILE.exists():
            w = json.loads(WORKSPACE_FILE.read_text()).get("wall_y")
        self.wall_y = None if w in (None, "", "none") else float(w)

    def describe(self):
        if self.kind == "jar":
            return "spice jar 5.1 x 10.2 cm (legacy scene)"
        return (f"{self.kind}: cylinder {self.radius * 2000:.1f} mm dia x {self.height * 1000:.1f} mm"
                + (f" on a {self.base_w * 1000:.1f} x {self.base_l * 1000:.1f} x {self.base_t * 1000:.1f} mm plate" if self.base_t else "")
                + (f", bolt {self.bolt_h * 1000:.0f} mm" if self.bolt_h else "") + f", {self.mass * 1000:.0f} g"
                + (f"; wall at y={self.wall_y * 100:+.1f} cm" if self.wall_y is not None else ""))

    @property
    def scene(self):
        """Path of the MJCF scene for this object (generated next to the robot model when needed)."""
        if self.kind == "jar" and self.wall_y is None:
            return MENAGERIE / "scene_jar_rl.xml"
        src = (MENAGERIE / "scene_jar_rl.xml").read_text()
        if self.wall_y is not None:                               # 2 cm thick slab whose near face is at y = wall_y
            floor = src.index("\n", src.index('<geom name="floor"')) + 1          # right after the floor geom (floor stays geom 0)
            src = (src[:floor] + f'    <geom name="partition" type="box" size="0.7 0.01 0.4" pos="0.25 {self.wall_y - 0.01:.4f} 0.4" '
                   f'group="3" rgba="0.2 0.4 0.8 0.3"/>\n' + src[floor:])
        if self.kind == "jar":
            return self._write(src)
        a, b = src.index('<body name="box"'), src.index("</body>") + len("</body>")
        rgb = " ".join(f"{v:.3f}" for v in (self.rgb or (0.62, 0.55, 0.45)))
        hh = self.height / 2.0
        # plate and cylinder share the mass by volume; MuJoCo derives the inertia from the geoms
        v_c, v_p = 3.14159 * self.radius ** 2 * self.height, self.base_w * self.base_l * self.base_t
        m_c = self.mass * v_c / (v_c + v_p)
        body = [f'<body name="box" pos="0.32 0 {self.rest_z:.5f}">', "      <freejoint/>",
                f'      <geom type="cylinder" name="box" size="{self.radius:.5f} {hh:.5f}" mass="{m_c:.5f}" condim="4" '
                f'friction="1 .01 .001" material="jar_body" contype="2" conaffinity="1" solref="0.01 1"/>']
        if self.base_t > 0:
            body.append(f'      <geom type="box" name="obj_base" size="{self.base_w / 2:.5f} {self.base_l / 2:.5f} {self.base_t / 2:.5f}" '
                        f'pos="0 0 {-(hh + self.base_t / 2):.5f}" mass="{self.mass - m_c:.5f}" condim="4" friction="1 .01 .001" '
                        f'material="jar_label" contype="2" conaffinity="1" solref="0.01 1"/>')
        if self.bolt_h > 0:
            body.append(f'      <geom type="cylinder" name="obj_bolt" size="{self.bolt_d / 2:.5f} {self.bolt_h / 2:.5f}" '
                        f'pos="0 0 {hh + self.bolt_h / 2:.5f}" contype="0" conaffinity="0" group="1" mass="0" material="jar_lid"/>')
        body.append("    </body>")
        xml = src[:a] + "\n".join(body) + src[b:]
        # a bright (white) print photographs at ~215/255; MuJoCo's lights alone render it ~190, so add some emission
        em = float(self.s.get("emission", 0.18 if (self.rgb and sum(self.rgb) / 3 > 0.8) else 0.0))
        xml = xml.replace('<material name="jar_body" rgba="0.62 0.55 0.45 1"', f'<material name="jar_body" emission="{em:.2f}" rgba="{rgb} 1"')
        xml = xml.replace('<material name="jar_label" rgba="0.15 0.22 0.62 1"', f'<material name="jar_label" emission="{em:.2f}" rgba="{rgb} 1"')
        xml = xml.replace('<material name="jar_lid" rgba="0.08 0.08 0.08 1"', '<material name="jar_lid" rgba="0.55 0.55 0.57 1"')
        return self._write(xml)

    def _write(self, xml):
        # the content hash in the name keeps runs with different specs / walls from overwriting each other's scene
        out = MENAGERIE / f"scene_obj_{self.kind}_{hashlib.md5(xml.encode()).hexdigest()[:6]}_rl.xml"
        if not out.exists():                                      # atomic: parallel workers may race here
            tmp = out.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(xml)
            tmp.replace(out)
        return out


SPEC = ObjectSpec()
