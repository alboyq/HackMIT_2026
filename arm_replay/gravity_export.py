"""Fit on the SIM's own model (menagerie i2rt_yam/yam.xml) so the correction is relative to what the ML teammate simulates, and write
gravity_fit.json for replay_pose2.py. Data: passive CAN captures of the arm's own runs 2-5 (nothing was weighed or measured by hand)."""
import json, time
from pathlib import Path
import numpy as np
import gravity_fit as gf
from gravity_fit import *
from gravity_eval_lib import fit_tied, fit_groups

P, rows = build([4, 5, 6]); print("samples:", len(rows), "(runs", sorted({r[0] for r in rows}), ")")
sv = fit(rows, False, False); s0 = fit_groups(rows); s1 = fit(rows, False, True); sc = fit_tied(rows)
gf.PRIOR = np.array([1e-3] * 8 + [0.6, 0.06, 0.06, 0.06]); se = fit(rows, True, False)      # alternative: all extra weight beyond the elbow
for n, s in (("stock model, no correction", sv), ("ONE scale, all joints", sc), ("DM4340 pair + wrist scale", s0), ("per-joint scale", s1), ("extra weight beyond the elbow", se)):
    x = s.x_full; print(f"  {n:30s} cost {s.cost:6.0f}  kappa {np.round(x[24:],3)}  b {np.round(x[12:18],2)}  RMS J2..J4 {np.round(rms_by_joint(x, rows)[1:4], 2)}")
print("  extra-weight alternative: dm=%+.2f kg  mu=%s" % (se.x_full[8], np.round(se.x_full[9:12], 3)))
print("\nheld-out-run test (J3 RMS, N.m):")
for held in (4, 5, 6):   # run 7 was flown AFTER this fit and is not in it: see README
    tr = [r for r in rows if r[0] != held]; te = [r for r in rows if r[0] == held]
    print(f"  fit without run {held}, score on run {held}:  stock {rms_by_joint(fit(tr, False, False).x_full, te)[2]:.2f}   chosen {rms_by_joint(fit_groups(tr).x_full, te)[2]:.2f}")

kk = s0.x_full[24:27]; k = float(kk[0]); b = s0.x_full[12:18]; print('scale J2,J3 / J4:', np.round(kk[[0, 2]], 3), '1-sigma (statistical):', np.round(s0.sd[[12, 13]] if len(s0.sd) > 13 else s0.sd[-2:], 3))
poses = [(r["label"], np.array(r["q_model"])) for r in json.load(open(Path(__file__).resolve().parent.parent / "fk_check" / "poses.json"))["poses"]] + [("HOME", np.array([-0.0279, 1.2776, 1.4870, -1.0699, -0.1920, -0.5637]))]
spread = []
for lab, q in poses:
    t0, A = P.design(q); a = gravity_term(s0.x_full, t0, A)[1:4]; alt = [gravity_term(x.x_full, t0, A)[1:4] for x in (s1, se)]
    spread.append(max(np.abs(np.array(alt) - a).max(axis=1)))
print(f"\nkappa (common) = {k:.3f}; alternatives disagree with it by at most {max(spread):.2f} N.m over the 8 taught poses + HOME")
out = {
  "what": "gravity + friction feed-forward correction for the real OpenYAM, fitted to the torques its own motors delivered (runs 2-6, 2026-09-20); nothing weighed or measured by hand",
  "base_model": "mujoco_menagerie/i2rt_yam/yam.xml with body_gravcomp zeroed (the sim's own model)",
  "convention": "feed-forward torque_j = factor_j * qfrc_bias_j(q) + coulomb_j * tanh(v_cmd_j / 0.02);  torques in the units of the MIT command (what the motor is TOLD to output)",
  "factor": [1.0, round(float(kk[0]), 3), round(float(kk[1]), 3), round(float(kk[2]), 3), 1.0, 1.0], "coulomb_Nm": [round(float(x), 2) for x in b[:4]] + [0.06, 0.06],
  "factor_per_joint_alternative": [1.0] + [round(float(x), 3) for x in s1.x_full[24:27]] + [1.0, 1.0],
  "extra_weight_alternative": {"link_4_extra_mass_kg": round(float(se.x_full[8]), 3), "link_4_mass_moment_kg_m": [round(float(x), 4) for x in se.x_full[9:12]]},
  "max_disagreement_between_structures_at_taught_poses_Nm": round(float(max(spread)), 2),
  "friction_note": "J3 static band is about +-2 N.m (stick-slip: the elbow sat still while torque swung 11.2 -> 7.0). Coulomb value is the fitted half-width.",
  "data_coverage": "runs 4,5 (elbow 0.03..0.18 rad) and run 6 (elbow up to 0.94 rad, shoulder up to 1.21, wrist -0.74..1.6). Elbow bent beyond ~1 rad is still extrapolation.",
  "fit_cost_vs_stock": {"stock_model": round(float(sv.cost)), "chosen": round(float(s0.cost))},
  "fitted": time.strftime("%Y-%m-%d %H:%M"),
}
json.dump(out, open(Path(__file__).resolve().parent / "gravity_fit.json", "w"), indent=2); print("wrote gravity_fit.json:", out["factor"], out["coulomb_Nm"])
