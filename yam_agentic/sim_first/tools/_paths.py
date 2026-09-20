"""Locate the two external model trees these tools need, from anywhere in a clone.

Neither is vendored here: mujoco_menagerie is ~1 GB and i2rt ships its own repo. Set the env
var or clone next to this one; the error message tells you exactly what to do.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = HERE.parent / "sim"


def _search(env, names, marker, clone_hint):
    v = os.environ.get(env)
    if v:
        p = Path(v)
        for n in names:
            if (p / marker).exists():
                return p
            if (p / n / marker).exists():
                return p / n
        raise SystemExit(f"{env}={v} does not contain {marker}")
    roots = [*HERE.parents, Path.cwd(), Path.home()]
    for base in roots:
        for n in names:
            c = base / n
            if (c / marker).exists():
                return c
    raise SystemExit(
        f"cannot find {names[0]}. Set {env}=/path/to/{names[0]}, or clone it:\n  {clone_hint}")


def menagerie_yam():
    """mujoco_menagerie/i2rt_yam — the arm model the sim is built on."""
    return _search("YAM_MENAGERIE", ["mujoco_menagerie/i2rt_yam", "mujoco_menagerie", "i2rt_yam"],
                   "yam.xml", "git clone https://github.com/google-deepmind/mujoco_menagerie")


def i2rt_root():
    """i2rt repo — ships the linear_4310 gripper MJCF and its meshes."""
    return _search("I2RT_ROOT", ["i2rt", "third_party/i2rt"],
                   "i2rt/robot_models/gripper/linear_4310/linear_4310.xml",
                   "git clone https://github.com/i2rt-robotics/i2rt")


def linear_4310_xml():
    return i2rt_root() / "i2rt/robot_models/gripper/linear_4310/linear_4310.xml"


def load_gripper_xml():
    """The standalone linear_4310 MJCF with meshdir made absolute (so it loads from anywhere)."""
    src = linear_4310_xml()
    assets = (src.parent / "assets").resolve()
    return src.read_text().replace('meshdir="assets"', f'meshdir="{assets}"')


def add_sim_to_path():
    sys.path.insert(0, str(SIM))
