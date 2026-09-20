"""Where is the mass the gravity model is missing?  Offline fit on arm_replay's telemetry (no CAN, no arm).

For a point mass m rigidly attached to body B at local position p, the extra holding torque is
    dtau(q) = m*g*Jp_B(q, p)^T z,     and Jp is affine in p,
so dtau is LINEAR in (m, m*px, m*py, m*pz). Fit those four numbers for each candidate body from the three
static poses in the logs (60% hold, 80% hold, folded return-hold; joints 2-4 -> 9 equations), and see which
body gives a small residual with a believable mass and position.
    python arm/real/fit_gravity.py

RESULT (2026-09-20, 3 static poses): NO body gives a believable fit (best: >1.1 kg, residual 1.15 N.m). The
reason is visible in the raw errors: extended, the ELBOW is short by 4.5 N.m while the SHOULDER is short by only
1.0 - but anything the elbow lifts the shoulder lifts too, so a missing mass cannot do that. The error is LOCAL
TO THE ELBOW JOINT. Two candidates the logs cannot tell apart:
  a) the elbow motor's torque scale is ~1.65x off in the driver (then ONE factor of ~1.65 is right everywhere,
     and the "0.95x folded" reading is the rubber stop still carrying part of the load 0.02 rad above it);
  b) something pose-dependent at the elbow (cable, spring).
ONE more held pose settles it: elbow >= 0.5 rad off its stop with the forearm still folded back. If the elbow
again needs ~1.65x the model there, it is (a): set the J3 factor to 1.65 and the problem is gone.
That is a powered move - a person's job with arm_replay/replay_pose2.py - so it is NOT done here.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "arm_replay"))
from replay_pose2 import Gravity  # noqa: E402

LOGS = sorted((ROOT / "arm_replay" / "logs").glob("*telemetry.json"))
OUT = Path(__file__).resolve().parent / "calibration" / "payload.json"
LAP = np.array([0, 1, 1, 0, 0, 0], float)       # both logged sessions (joint_map_measured.json convention)
JOINTS = (1, 2, 3)                              # shoulder, elbow, wrist pitch: the gravity-loaded ones


def static_poses():
    """Mean of the last 30 samples of every held phase: (tag, raw position, reported torque)."""
    out = []
    for f in LOGS:
        s = json.loads(f.read_text())["series"]
        for ph in ("hold", "return-hold"):
            r = [np.array(x[2]) for x in s if x[1] == ph][-30:]
            if len(r) >= 10:
                out.append((f"{f.name.split('_')[0]}:{ph}", np.mean([x[0] for x in r], 0), np.mean([x[1] for x in r], 0)))
    return out


def main():
    import mujoco
    g = Gravity(LAP); M, D = g.M, g.D
    poses = static_poses()
    print("static poses:", [p[0] for p in poses])
    err = np.array([[tau[j] - g(raw)[j] for j in JOINTS] for _, raw, tau in poses])
    print("measured - model (N.m), rows = poses, cols = J2 J3 J4:\n", np.round(err, 2))
    best = None
    for body in ("link_2", "link_3", "link_4", "link_5", "link_6"):
        bid = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, body)
        if bid < 0:
            continue
        A, b = [], []
        for _, raw, tau in poses:
            G = g(raw)
            R, o = D.xmat[bid].reshape(3, 3), D.xpos[bid]
            cols = []
            for local in (np.zeros(3), np.eye(3)[0], np.eye(3)[1], np.eye(3)[2]):
                jac = np.zeros((3, M.nv)); mujoco.mj_jac(M, D, jac, None, o + R @ local, bid)
                cols.append(9.81 * jac[2, :6])
            base = cols[0]
            for j in JOINTS:                     # unknowns: m, m*px, m*py, m*pz
                A.append([base[j]] + [cols[k][j] - base[j] for k in (1, 2, 3)]); b.append(tau[j] - G[j])
        A, b = np.array(A), np.array(b)
        x, *_ = np.linalg.lstsq(A, b, rcond=1e-3)
        m = float(x[0]); p = x[1:] / m if abs(m) > 1e-6 else np.zeros(3)
        rms = float(np.sqrt(np.mean((b - A @ x) ** 2)))
        ok = 0.05 < m < 1.0 and np.linalg.norm(p) < 0.30
        print(f"  on {body}: m = {m:6.3f} kg at {np.round(100*p, 1)} cm   residual rms {rms:.2f} N.m (was {np.sqrt(np.mean(b**2)):.2f})  {'plausible' if ok else 'not plausible'}")
        if ok and (best is None or rms < best["rms_after"]):
            best = {"body": body, "mass_kg": m, "pos_m": p.tolist(), "rms_after": rms, "rms_before": float(np.sqrt(np.mean(b**2)))}
    if best is None:
        print("no believable missing mass: the error is local to the elbow joint (see the docstring). Nothing written.")
        return
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"fit": best, "poses": [p[0] for p in poses], "validated_on_arm": False}, indent=2))
    print("best:", best, "->", OUT)


if __name__ == "__main__":
    main()
