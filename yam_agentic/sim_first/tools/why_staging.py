"""Which staging sub-condition rejects demo episodes, and by how much.

    N=40 python tools/why_staging.py

Measured 2026-09-20: every staging rejection was the pose-match term alone. The rejected
episodes were good — object held (median 1.6 cm from the TCP), off the table (median 33.8 cm),
delivered to the right standoff (TCP->mouth 14.4-15.7 cm against a 15 cm target). They had
settled at a different point on the standoff sphere, because the arm droops ~2.8 cm under
payload. Filtering on that discarded 30% of successful deliveries and biased the set toward
users the arm happens to track precisely. gen_demos_yam.py now accepts on the task instead:
63% -> 94% kept.
"""
import os, sys
from pathlib import Path
import _paths; _paths.add_sim_to_path()
import numpy as np, mujoco
from yam_data import DataScene
from yam_expert import Expert
from yam_data import GRIP_CTRL_OPEN
HOLD_S, SETTLE_TICKS = 1.0, 3

sc = DataScene(); ex = Expert(sc)
stats = {"held": [], "height": [], "reach": []}
fails = {"hidden": 0, "pick": 0, "lift": 0, "present": 0, "staging": 0, "ok": 0}
N = int(os.environ.get("N", "40"))
for k in range(N):
    seed = 1_000_000 + k * 7
    target = sc.randomize(seed)
    hold = np.concatenate([sc.data.ctrl[:6], [1.0]])
    for _ in range(SETTLE_TICKS): sc.tick(hold, hold)
    if sc.capture_box() is None: fails["hidden"] += 1; continue
    plan = ex.plan_pick(target)
    if not plan: fails["pick"] += 1; continue
    sim_dt = sc.model.opt.timestep
    prev = hold.copy()
    def play(traj):
        global prev
        n = int(np.ceil(len(traj) / sc.n_sub))
        for t in range(n):
            c = np.array(traj[min((t + 1) * sc.n_sub, len(traj)) - 1], float)
            act = np.concatenate([c[:6], [c[6] / GRIP_CTRL_OPEN]]).astype(np.float32)
            sc.tick(act, prev); prev = act
    z0 = sc.object_pos(target)[2]
    play(ex.trajectory(plan, sim_dt))
    if sc.object_pos(target)[2] - z0 < 0.05: fails["lift"] += 1; continue
    pres = ex.plan_present(sc.site("mouth"), q_from=sc.q_arm, grip=float(sc.data.ctrl[6]))
    if not pres: fails["present"] += 1; continue
    play(ex.trajectory(pres, sim_dt))
    play([sc.to_ctrl(prev)] * int(HOLD_S / sim_dt))
    tcp = ex.ik.fk(sc.q_arm, qpos_full=sc.data.qpos)[0]
    obj = sc.object_pos(target)
    d_held  = float(np.linalg.norm(obj - tcp))
    z_obj   = float(obj[2])
    d_reach = float(np.linalg.norm(tcp - pres.stage_pos))
    mouth = sc.site("mouth")
    d_mouth_tcp  = float(np.linalg.norm(mouth - tcp))
    d_mouth_obj  = float(np.linalg.norm(mouth - obj))
    stats.setdefault("mouth_tcp", []).append(d_mouth_tcp)
    stats.setdefault("mouth_obj", []).append(d_mouth_obj)
    stats["held"].append(d_held); stats["height"].append(z_obj); stats["reach"].append(d_reach)
    if d_held < 0.08 and z_obj > 0.12 and d_reach < 0.04: fails["ok"] += 1
    else:
        fails["staging"] += 1
        which = [n for n, c in (("held>8cm", d_held >= 0.08), ("z<12cm", z_obj <= 0.12),
                                ("reach>4cm", d_reach >= 0.04)) if c]
        print(f"  seed {seed} {target:7s} REJECT {','.join(which):28s} "
              f"held {d_held*100:5.1f} cm  reach {d_reach*100:5.1f} cm  "
              f"TCP->mouth {d_mouth_tcp*100:5.1f} cm  obj->mouth {d_mouth_obj*100:5.1f} cm")
print("\n", fails)
for k, v in stats.items():
    if v: print(f"{k:7s} median {np.median(v)*100:6.1f} cm   p90 {np.percentile(v,90)*100:6.1f} cm")
