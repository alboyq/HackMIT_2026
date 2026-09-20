"""Reward-shape regression tests.

The same class of bug has bitten three times in one session, each time costing a full retrain:

* APPROACH paid 0.10/step for holding the jaws open, so loitering 300 steps was worth 30 against
  a success_bonus of 5. The policy parked at the object and collected rent.
* CLOSE paid up to 0.25/step while SECURE paid 0.10, so completing the grasp CUT the policy's
  income by 60%. It sat half-open in position for 81% of every episode and never pinched.

Both are arithmetic, not learning problems, and both are checkable without running anything.
Two invariants:

1. **Phase value must increase.** Every phase must pay strictly more per step than the one
   before it, or the policy is paid to stall at the boundary.
2. **No per-step term may rival the terminal bonus.** The best possible loitering return over a
   full episode must be worth less than succeeding, or finishing is irrational.
"""
from pathlib import Path

import pytest

from hackmit_rl.config import load_config

CONFIGS = sorted(Path("arm/rl/configs").glob("feed*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_phase_value_increases(path):
    e = load_config(path)["env"]
    # Best case each phase can earn per step, ignoring shared terms that apply to all of them.
    approach = 0.0                                    # phase 0 only ever charges
    close = float(e["close_bonus"]) + float(e["in_position_bonus"])
    secure = float(e["pinch_bonus"])
    assert close > approach, f"{path.name}: CLOSE must out-pay APPROACH"
    assert secure > close, (
        f"{path.name}: SECURE pays {secure}/step but CLOSE pays {close}/step -- completing the "
        f"grasp would cut the policy's income, so it will stall half-closed in position")


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_finishing_beats_loitering(path):
    e = load_config(path)["env"]
    steps = int(e["episode_steps"])
    # grip_band_bonus stacks on pinch_bonus while holding, so it belongs in the rent.
    best_per_step = max(float(e["close_bonus"]) + float(e["in_position_bonus"]),
                        float(e["pinch_bonus"]) + float(e["grip_band_bonus"]))
    loiter = (best_per_step - float(e["time_penalty"])) * steps
    success = float(e["success_bonus"])
    assert loiter < success, (
        f"{path.name}: loitering a full {steps}-step episode is worth {loiter:.1f} against a "
        f"success bonus of {success:.1f} -- running the clock out is the better strategy")


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_tolerances_exceed_perception_error(path):
    """A tolerance tighter than the perception noise is unwinnable for reasons the robot does
    not have. Bias is 3D, so compare against its expected magnitude, not one axis."""
    cfg = load_config(path)
    noise = cfg.get("perception_noise", {})
    if not noise.get("enabled", False):
        pytest.skip("perception noise disabled")
    typical = float(noise["object_bias_m"]) * 1.6          # ~E|N(0,s)| in 3D
    assert float(cfg["env"]["success_distance_m"]) > typical, (
        f"{path.name}: reach tolerance is below typical perception error {typical*1000:.0f} mm")


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_objects_fit_the_gripper(path):
    """Scaling must leave room to straddle the object, with margin over perception error."""
    e = load_config(path)["env"]
    assert float(e["pad_clearance_m"]) >= 0.015, (
        f"{path.name}: pad clearance {e['pad_clearance_m']*1000:.0f} mm leaves less per side "
        f"than the perception error, so the gripper cannot reliably straddle the object")


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_height_is_never_rent(path):
    """`lift_bonus * height` paid 0.9/step at 30 cm with no ceiling -- 270 an episode against a
    success bonus of 50 -- and the policy swung the object as high as it could. Height may only
    pay as capped progress, whose total is fixed."""
    e = load_config(path)["env"]
    assert "lift_bonus" not in e, f"{path.name}: lift_bonus is per-step rent on height; use progress"
    total = float(e["lift_progress_gain"]) * float(e["lift_height_m"])
    assert total < 0.2 * float(e["success_bonus"]), (
        f"{path.name}: lifting alone is worth {total:.1f}; it must stay small next to finishing")


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_present_rent_cannot_beat_finishing(path):
    e = load_config(path)["env"]
    rent = (float(e["pinch_bonus"]) + float(e["grip_band_bonus"]) + float(e["carry_bonus"])
            + float(e["lead_bonus"]) - float(e["time_penalty"]))
    assert rent * int(e["episode_steps"]) < float(e["success_bonus"])


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_lift_gates_are_ordered(path):
    e = load_config(path)["env"]
    assert float(e["lift_max_drift_m"]) < float(e["lift_abort_drift_m"])
    assert float(e["lift_band_m"]) > 0.02, "the stop band must be wider than the arm can hold"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_milestones_stay_small(path):
    """One-off payments (seated pinch, lift progress) must not rival finishing either, or the
    policy collects them and stops."""
    e = load_config(path)["env"]
    once = float(e["pinch_once_bonus"]) + float(e["lift_progress_gain"]) * float(e["lift_height_m"])
    assert once < 0.2 * float(e["success_bonus"]), f"{path.name}: milestones total {once:.1f}"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_not_picking_up_is_the_worst_outcome(path):
    """Hovering must never be the cheap option. It collapsed into exactly that once attempts were
    penalised and doing nothing was not."""
    e = load_config(path)["env"]
    worst_other = max(float(e["lost_penalty"]), float(e["knock_penalty"]), float(e["drop_penalty"]),
                      float(e["time_penalty"]) * int(e["episode_steps"]))
    assert float(e["no_pickup_penalty"]) > worst_other
    assert 0.4 * float(e["no_pickup_penalty"]) < float(e["success_bonus"])
