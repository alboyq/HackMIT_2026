"""Generate pick-and-deliver demonstrations for the sim-first YAM policy.

Each episode: randomise the scene, capture the selection-time object box, plan with the tested
expert (yam_expert.py), then EXECUTE the plan the way a slow policy will drive the real arm —
targets at YAM_HZ, linearly interpolated in between — recording (scene RGB, wrist RGB, state,
action) once per tick. Pick -> carry -> stop 15 cm short of the mouth -> hold. Only episodes that
end with the object held at the staging point are kept ("filtered demos").

Usage: gen_demos_yam.py OUT_DIR [episodes] [--workers 14] [--seed0 1000000] [--film 2]
Output: OUT_DIR/shard_XX.npz  (scene u8 NxHxWx3, wrist u8, state f32 Nx17, action f32 Nx7, episode i64)
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOLD_S, SETTLE_TICKS = 1.0, 3


def run_episode(sc, ex, seed, np, mujoco):
    """Returns (frames dict, info) — frames is None if the episode is unusable."""
    from yam_data import GRIP_CTRL_OPEN
    target = sc.randomize(seed)
    hold = np.concatenate([sc.data.ctrl[:6], [1.0]])
    for _ in range(SETTLE_TICKS):
        sc.tick(hold, hold)
    if sc.capture_box() is None:
        return None, "target hidden at selection time"
    plan = ex.plan_pick(target)
    if not plan:
        return None, f"pick plan: {plan.why}"
    sim_dt = sc.model.opt.timestep
    S, W, X, A = [], [], [], []
    prev = hold.copy()

    def play(traj):
        nonlocal prev
        n_ticks = int(np.ceil(len(traj) / sc.n_sub))
        for t in range(n_ticks):
            c = np.array(traj[min((t + 1) * sc.n_sub, len(traj)) - 1], float)
            act = np.concatenate([c[:6], [c[6] / GRIP_CTRL_OPEN]]).astype(np.float32)
            scene, wrist, state = sc.observe()
            S.append(scene); W.append(wrist); X.append(state); A.append(act)
            sc.tick(act, prev)
            prev = act

    z0 = sc.object_pos(target)[2]
    play(ex.trajectory(plan, sim_dt))
    if sc.object_pos(target)[2] - z0 < 0.05:
        return None, "lift failed"
    pres = ex.plan_present(sc.site("mouth"), q_from=sc.q_arm, grip=float(sc.data.ctrl[6]))
    if not pres:
        return None, f"present plan: {pres.why}"
    play(ex.trajectory(pres, sim_dt))
    play([sc.to_ctrl(prev)] * int(HOLD_S / sim_dt))                      # learn to stop and stay
    tcp = ex.ik.fk(sc.q_arm, qpos_full=sc.data.qpos)[0]
    obj = sc.object_pos(target)
    # Accept on the TASK, not on pose-matching. Measured 2026-09-20: a 4 cm match against the
    # planner's chosen stage_pos rejected 30 % of episodes whose object was still held (median
    # 1.6 cm from the TCP), off the table (median 33.8 cm) and at the right standoff
    # (TCP->mouth 14.4-15.7 cm against a 15 cm target). They had simply settled at a different
    # point on the standoff sphere, because the arm droops ~2.8 cm under payload — which is
    # realistic and should not be filtered out. Filtering on it biased the set toward users the
    # arm happens to track precisely.
    d_mouth = float(np.linalg.norm(obj - sc.site("mouth")))
    if np.linalg.norm(obj - tcp) >= 0.08:
        return None, "payload dropped"
    if obj[2] <= 0.12:
        return None, "payload too low"
    if not (0.12 <= d_mouth <= 0.20):
        return None, "payload not at the mouth standoff"
    if np.linalg.norm(tcp - pres.stage_pos) >= 0.10:
        return None, "staging pose wildly off"
    return dict(scene=S, wrist=W, state=X, action=A), target


def filmstrip(fr, path, np, cv2, n=8):
    idx = np.linspace(0, len(fr["scene"]) - 1, n).astype(int)
    rows = []
    for i in idx:
        s = fr["scene"][i].copy(); st = fr["state"][i]; R = s.shape[0]
        cx, cy, w, h = st[7:11] * R
        cv2.rectangle(s, (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2)), (255, 255, 0), 1)
        col = (0, 255, 0) if st[16] > 0.5 else (255, 0, 0)
        cv2.circle(s, (int(st[13] * R), int(st[14] * R)), 4, col, 1)
        w_img = cv2.resize(fr["wrist"][i], (R, R))
        rows.append(np.concatenate([s, w_img], axis=0))
    import imageio.v3 as iio
    iio.imwrite(path, np.concatenate(rows, axis=1))


def worker(a):
    wid, seeds, out, n_film = a
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(HERE))
    import cv2
    import mujoco
    import numpy as np
    cv2.setNumThreads(1)
    from yam_data import DataScene
    from yam_expert import Expert
    sc = DataScene()
    ex = Expert(sc)
    S, W, X, A, E, why = [], [], [], [], [], {}
    kept = 0
    for seed in seeds:
        fr, info = run_episode(sc, ex, int(seed), np, mujoco)
        if fr is None:
            k = info.split(":")[0]
            why[k] = why.get(k, 0) + 1
            continue
        if wid == 0 and kept < n_film:
            filmstrip(fr, str(Path(out) / f"film_{seed}_{info}.png"), np, cv2)
        kept += 1
        S += fr["scene"]; W += fr["wrist"]; X += fr["state"]; A += fr["action"]; E += [seed] * len(fr["state"])
    if kept:
        np.savez(Path(out) / f"shard_{wid:02d}.npz", scene=np.array(S, np.uint8), wrist=np.array(W, np.uint8),
                 state=np.array(X, np.float32), action=np.array(A, np.float32), episode=np.array(E, np.int64))
    return wid, kept, len(seeds), len(E), why


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("episodes", type=int, nargs="?", default=300)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed0", type=int, default=1_000_000)
    ap.add_argument("--film", type=int, default=2)
    args = ap.parse_args()
    if "heldout" in str(os.environ.get("YAM_BG", "")):
        sys.exit("refusing to generate TRAINING demos on a held-out background set")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed0, args.seed0 + args.episodes))
    jobs = [(w, seeds[w::args.workers], str(out), args.film) for w in range(args.workers)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        res = pool.map(worker, jobs)
    kept, tried, frames = sum(r[1] for r in res), sum(r[2] for r in res), sum(r[3] for r in res)
    why = {}
    for r in res:
        for k, v in r[4].items():
            why[k] = why.get(k, 0) + v
    msg = (f"DONE {kept}/{tried} episodes kept ({100 * kept / max(tried, 1):.0f}%), {frames} frames, "
           f"{time.time() - t0:.0f}s, rejects={why}")
    print(msg)
    (out / "meta.txt").write_text(msg + f"\nhz={os.environ.get('YAM_HZ', '10')} seed0={args.seed0}\n")
