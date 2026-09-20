"""Phase-0 gate: the scene loads, the tool frame is right, IK is exact, and the expert can
pick each object and present it at the mouth without hitting the user.

Run:  urlab_bridge/.venv/bin/python -m pytest rl/yam/test_yam_sim.py -q
      urlab_bridge/.venv/bin/python rl/yam/test_yam_sim.py       (same checks, verbose)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco
import numpy as np
import pytest

from yam_expert import Expert
from yam_ik import ArmIK, ToolFrame
from yam_scene import LIBRARY, YamScene

DOWN = np.array([0.0, 0.0, -1.0])


@pytest.fixture(scope="module")
def scene():
    return YamScene()


@pytest.fixture(scope="module")
def expert(scene):
    return Expert(scene)


def test_scene_loads(scene):
    assert scene.model.nu == 7                      # 6 arm joints + gripper
    assert scene.model.ncam == 2                    # wrist + scene
    assert {o.name for o in scene.objects} == set(LIBRARY)


def test_tool_frame_is_between_the_pads(scene):
    tf = ToolFrame(scene.model)
    # the TCP is NOT grasp_site: on this model it sits ~4.4 cm off it
    assert np.linalg.norm(tf.tcp_local) > 0.03
    assert abs(float(np.dot(tf.tool_local, tf.jaw_local))) < 1e-9
    assert 0.070 < tf.max_width < 0.095             # pads open to ~83 mm
    assert tf.opening["closed"] < 0.005


def test_ik_round_trip(scene):
    tf = ToolFrame(scene.model)
    ik = ArmIK(scene.model, tf)
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(12):
        q_true = rng.uniform(ik.lo + 0.25, ik.hi - 0.25)
        p, R = ik.fk(q_true)
        q, ok, ep, er = ik.solve(p, R, q_init=q_true + rng.normal(0, 0.06, 6))
        assert ok, f"round trip failed: {ep * 1000:.1f} mm"
        errs.append(ep)
    assert max(errs) < 1e-3                          # sub-millimetre


def test_measured_envelope_holds(scene):
    """The numbers the plan was designed against — if the model changes, this fails loudly."""
    tf = ToolFrame(scene.model)
    ik = ArmIK(scene.model, tf)
    assert ik.solve_best(np.array([0.45, 0.0, 0.05]), DOWN)[1], "top-down should reach 0.45 m"
    assert not ik.solve_best(np.array([0.62, 0.0, 0.05]), DOWN)[1], "top-down should fail at 0.62 m"
    level = np.array([1.0, 0.0, 0.0])
    assert ik.solve_best(np.array([0.40, 0.0, 0.38]), level)[1], "mouth-height delivery"


def test_every_object_is_graspable(expert):
    for name in LIBRARY:
        ok, why = expert.check_graspable(name)
        assert ok, f"{name}: {why}"


def test_gripper_width_gate_rejects_oversized(scene, expert):
    """The feasibility gate has to actually say no — otherwise the agent layer will promise
    things the hardware cannot do."""
    o = scene.spec("mug")
    wide = type(o)(o.name, o.kind, (0.060, o.size[1]), o.rgba, o.mass)
    assert wide.width > expert.tool.max_width
    orig = scene.objects
    try:
        scene.objects = [wide if x.name == "mug" else x for x in orig]
        ok, why = expert.check_graspable("mug")
        assert not ok and "not fit" in why
    finally:
        scene.objects = orig


@pytest.mark.parametrize("name", list(LIBRARY))
def test_pick_and_lift(name):
    """Full closed-loop physics: plan, execute, and check the object actually came up."""
    s = YamScene()
    e = Expert(s)
    dt = s.model.opt.timestep
    plan = e.plan_pick(name)
    assert plan, f"{name}: {plan.why}"
    z0 = s.object_pos(name)[2]
    for ctrl in e.trajectory(plan, dt):
        s.step(ctrl, 1)
    z1 = s.object_pos(name)[2]
    assert z1 - z0 > 0.05, f"{name} rose only {(z1 - z0) * 1000:.0f} mm (strategy {plan.strategy})"


def test_present_at_mouth_without_touching_the_user():
    s = YamScene()
    e = Expert(s)
    dt = s.model.opt.timestep
    pick = e.plan_pick("apple")
    assert pick, pick.why
    for ctrl in e.trajectory(pick, dt):
        s.step(ctrl, 1)
    held_z = s.object_pos("apple")[2]
    assert held_z > 0.08, "apple was dropped before the delivery"

    mouth = s.site("mouth")
    pres = e.plan_present(mouth, q_from=s.q_arm, grip=float(s.data.ctrl[6]))
    assert pres, pres.why
    for ctrl in e.trajectory(pres, dt):
        s.step(ctrl, 1)

    tcp = e.ik.fk(s.q_arm, qpos_full=s.data.qpos)[0]
    gap = float(np.linalg.norm(tcp - mouth))
    assert 0.10 < gap < 0.22, f"stopped {gap * 100:.0f} cm from the mouth, expected ~15 cm"
    # the delivery only counts if the payload actually arrived: an earlier version of this
    # test passed while the apple was being flung onto the table during the transport
    apple = s.object_pos("apple")
    assert apple[2] > 0.15, f"apple was dropped in transit (ended at {np.round(apple, 3)})"
    assert float(np.linalg.norm(apple - tcp)) < 0.08, "apple is not in the gripper"
    # and nothing ever entered the head
    head = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_GEOM, "head")
    for i in range(s.data.ncon):
        c = s.data.contact[i]
        assert head not in (c.geom1, c.geom2) or c.dist > -1e-3, "the arm touched the head"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "--no-header"]))
