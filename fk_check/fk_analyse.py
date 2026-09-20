"""Re-run the FK check from the recorded poses and ruler distances (no hardware needed).

    python fk_check/fk_analyse.py [--model PATH/yam.xml] [--tip X,Y,Z] [--json]

What it does
  1. Takes each recorded pose (encoder readings already lap-corrected by fk_capture.py), applies
     the measured joint map (q_model = encoder - offset; every sign is +1) and runs forward
     kinematics on the menagerie arm chain.
  2. Fits the tool tip from the dots that were touched from several arm configurations
     (pivot calibration: dot 2 = poses 2/2b/2c, dot 3 = poses 3/3b). This was needed because the
     corrected linear_4310 gripper model was not available on the GX10. Pass --tip to use a tool
     point from that model instead (millimetres, link_6 frame).
  3. Compares every predicted dot-to-dot distance with the ruler/caliper measurement.

The chain J1-J6 is identical for the stock and the linear_4310 gripper, so only the tool tip
depends on the gripper. Distances between dots cannot see a J1 offset (rotation about the base
axis) or a J6 offset (rotation about the tool axis) - see joint_map_measured.json.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
OFFSET = np.array([-1.4313, 0.0044, 0.0101, -0.0074, 0.0015, 1.3035])     # rad, joint_map_measured.json
GROUPS = {"2": ["2", "2b", "2c"], "3": ["3", "3b"]}                        # captures that touched the same dot
SINGLE = {"1": "1", "4": "4", "5": "5"}                                    # dots touched once


def find_model(arg):
    if arg: return Path(arg)
    env = os.environ.get("YAM_MENAGERIE")
    cands = ([Path(env)] if env else []) + [HERE.parent / "third_party/mujoco_menagerie/i2rt_yam",
                                            Path.home() / "HackMIT_2026/third_party/mujoco_menagerie/i2rt_yam"]
    for c in cands:
        c = c if c.name == "yam.xml" else c / "yam.xml"
        if c.exists(): return c
    sys.exit("cannot find mujoco_menagerie/i2rt_yam/yam.xml - pass --model or set YAM_MENAGERIE")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model"); ap.add_argument("--tip"); ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    M = mujoco.MjModel.from_xml_path(str(find_model(a.model))); D = mujoco.MjData(M)
    l6 = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, "link_6")
    poses = {p["label"]: p for p in json.load(open(HERE / "poses.json"))["poses"]}
    ruler = json.load(open(HERE / "distances.json"))["pairs"]
    F = {}
    for k, p in poses.items():
        D.qpos[:6] = np.array(p["encoder_unwrapped"]) - OFFSET; mujoco.mj_forward(M, D)
        F[k] = (D.xpos[l6].copy() * 1000, D.xmat[l6].reshape(3, 3).copy())
    if a.tip:
        t = np.array([float(x) for x in a.tip.split(",")]); src = "given on the command line"
    else:                                                   # pivot fit: one tool tip, one unknown point per multi-touch dot
        groups = [GROUPS["2"], GROUPS["3"]]; rows, rhs = [], []
        for gi, ids in enumerate(groups):
            for k in ids:
                r = np.zeros((3, 3 + 3 * len(groups))); r[:, :3] = F[k][1]; r[:, 3 + 3 * gi:6 + 3 * gi] = -np.eye(3)
                rows.append(r); rhs.append(-F[k][0])
        A_, b_ = np.vstack(rows), np.concatenate(rhs); sol = np.linalg.lstsq(A_, b_, rcond=None)[0]
        t = sol[:3]; src = "pivot fit from dots 2 and 3 (5 captures)"
        res = (A_ @ sol - b_).reshape(-1, 3)
    tip = {k: F[k][0] + F[k][1] @ t for k in F}
    dot = {n: np.mean([tip[k] for k in ids], axis=0) for n, ids in GROUPS.items()}
    dot.update({n: tip[k] for n, k in SINGLE.items()})
    rows_out, errs = [], []
    for pair, r in ruler.items():
        x, y = pair.split("-"); pred = float(np.linalg.norm(dot[x] - dot[y])); errs.append(pred - r)
        rows_out.append({"pair": pair, "model_mm": round(pred, 1), "ruler_mm": r, "error_mm": round(pred - r, 1)})
    e = np.array(errs); n3 = np.array([v for (p, v) in zip(ruler, errs) if "3" not in p.split("-")])
    z = [tip[k][2] for k in F if k not in ("2",)]
    out = {"tool_tip_link6_mm": [round(float(v), 1) for v in t], "tool_tip_source": src, "distances": rows_out,
           "worst_error_mm": round(float(np.abs(e).max()), 1), "rms_mm": round(float(np.sqrt((e ** 2).mean())), 1),
           "without_dot3": {"n": len(n3), "worst_error_mm": round(float(np.abs(n3).max()), 1), "rms_mm": round(float(np.sqrt((n3 ** 2).mean())), 1)},
           "gate_mm": 10, "pass": bool(np.abs(e).max() < 10)}
    if a.json: print(json.dumps(out, indent=2)); return
    print(f"tool tip (link_6 frame): {out['tool_tip_link6_mm']} mm   [{src}]\n")
    print(f"{'pair':<6s}{'model':>9s}{'ruler':>8s}{'error':>8s}")
    for r in rows_out: print(f"{r['pair']:<6s}{r['model_mm']:9.1f}{r['ruler_mm']:8.0f}{r['error_mm']:+8.1f}")
    print(f"\nworst |error| {out['worst_error_mm']} mm, rms {out['rms_mm']} mm over {len(e)} distances "
          f"(gate {out['gate_mm']} mm): {'PASS' if out['pass'] else 'FAIL'}")
    print(f"without the dot-3 pairs ({out['without_dot3']['n']}): worst {out['without_dot3']['worst_error_mm']} mm, rms {out['without_dot3']['rms_mm']} mm")


if __name__ == "__main__":
    main()
