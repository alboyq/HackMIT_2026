"""Gate A's ceiling: what does the EXPERT score under eval_act_yam.py's own success test?

    python tools/expert_ceiling.py          # N=24 by default

Run this BEFORE reading any policy number. Measured 2026-09-20: 23/23 = 100%, median 2.9 cm
from the staging point (p90 4.6 cm) against eval's 6 cm threshold — so the yardstick is sound
and a 0% policy score is real, not a scoring artifact.
"""
import os, sys
import _paths; _paths.add_sim_to_path()
import numpy as np
from yam_data import DataScene, GRIP_CTRL_OPEN
from yam_expert import Expert, STAGING

sc = DataScene(); ex = Expert(sc)
N = int(os.environ.get("N", "24"))
pose_ok = task_ok = held_n = done = 0
dists = []
for k in range(N):
    seed = 9_000_000 + k                       # eval's seed range
    target = sc.randomize(int(seed))
    hold = np.concatenate([sc.data.ctrl[:6], [1.0]])
    for _ in range(3): sc.tick(hold, hold)
    if sc.capture_box() is None: continue
    plan = ex.plan_pick(target)
    if not plan: continue
    prev = hold.copy()
    def play(traj):
        global prev
        n = int(np.ceil(len(traj) / sc.n_sub))
        for t in range(n):
            c = np.array(traj[min((t + 1) * sc.n_sub, len(traj)) - 1], float)
            act = np.concatenate([c[:6], [c[6] / GRIP_CTRL_OPEN]]).astype(np.float32)
            sc.tick(act, prev); prev = act
    dt = sc.model.opt.timestep
    play(ex.trajectory(plan, dt))
    pres = ex.plan_present(sc.site("mouth"), q_from=sc.q_arm, grip=float(sc.data.ctrl[6]))
    if not pres: continue
    play(ex.trajectory(pres, dt))
    play([sc.to_ctrl(prev)] * int(1.0 / dt))
    done += 1
    tcp = ex.ik.fk(sc.q_arm, qpos_full=sc.data.qpos)[0]
    obj = sc.object_pos(target); mouth = sc.site("mouth")
    d = mouth - np.array([0, 0, mouth[2]]); d[2] = 0; d /= np.linalg.norm(d)
    stage = mouth - d * STAGING
    held = bool(np.linalg.norm(obj - tcp) < 0.08 and obj[2] > 0.12)
    dist = float(np.linalg.norm(tcp - stage))
    dists.append(dist)
    held_n += held
    pose_ok += bool(held and dist < 0.06)
    task_ok += bool(held and 0.12 <= np.linalg.norm(obj - mouth) <= 0.20)
print(f"\nexpert over {done} completed episodes (eval seeds 9,000,000+):")
print(f"  held                                   {held_n}/{done} = {100*held_n/max(done,1):.0f}%")
print(f"  eval's test  (held and dist < 6 cm)    {pose_ok}/{done} = {100*pose_ok/max(done,1):.0f}%   <- Gate A ceiling")
print(f"  task-level   (held, 12-20 cm to mouth) {task_ok}/{done} = {100*task_ok/max(done,1):.0f}%")
print(f"  dist to stage point: median {np.median(dists)*100:.1f} cm, p90 {np.percentile(dists,90)*100:.1f} cm")
