"""Closed-loop evaluation of a YAM ACT checkpoint in the randomised data scene.

Held-out seeds (9,000,000+), and by default HELD-OUT BACKGROUNDS (rl/jar/bg_heldout) with the
training-time camera effects on — the deployment-like stack that the SO-101 stress test showed
compounds (84 % -> 51 %), so this is the number to steer by, not a clean in-distribution score.

Success = at the end the selected object is in the gripper, lifted, and the TCP is within 6 cm of
the staging point 15 cm short of the mouth. Also reported: picked (lifted at any time), and the
final TCP-to-staging distance, so partial credit is visible.

Usage: eval_act_yam.py CKPT_DIR [episodes] [--na 10] [--bg DIR] [--workers 14] [--no-aug]
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
HELDOUT = HERE.parent / "jar" / "bg_heldout" / "full"


def worker(a):
    ckpt, seeds, na, bg, aug, n_film, max_ticks = a
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(HERE))
    import cv2
    import numpy as np
    import torch
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    from lerobot.policies.act.modeling_act import ACTPolicy
    from yam_data import DataScene
    from yam_expert import STAGING, Expert
    policy = ACTPolicy.from_pretrained(ckpt)
    policy.to("cpu").eval()
    if na:
        policy.config.n_action_steps = na
    sc = DataScene(bg_dir=bg, aug=aug)
    ex = Expert(sc)                                                    # only for FK and the staging geometry
    out, films = [], []
    for seed in seeds:
        target = sc.randomize(int(seed))
        hold = np.concatenate([sc.data.ctrl[:6], [1.0]]).astype(np.float32)
        for _ in range(3):
            sc.tick(hold, hold)
        if sc.capture_box() is None:
            continue
        policy.reset()
        mouth = sc.site("mouth")
        d = mouth - np.array([0, 0, mouth[2]]); d[2] = 0; d /= np.linalg.norm(d)
        stage = mouth - d * STAGING
        z0, prev, picked, frames = sc.object_pos(target)[2], hold, False, []
        for t in range(max_ticks):
            scene, wrist, state = sc.observe()
            if len(films) < n_film and t % 12 == 0:
                frames.append(np.concatenate([scene, cv2.resize(wrist, scene.shape[:2])], 0))
            obs = {"observation.images.scene": torch.from_numpy(scene.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0,
                   "observation.images.wrist": torch.from_numpy(wrist.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0,
                   "observation.state": torch.from_numpy(state).unsqueeze(0)}
            with torch.no_grad():
                act = policy.select_action(obs)[0].numpy().astype(np.float32)
            sc.tick(act, prev)
            prev = act
            picked = picked or (sc.object_pos(target)[2] - z0 > 0.05)
        tcp = ex.ik.fk(sc.q_arm, qpos_full=sc.data.qpos)[0]
        obj = sc.object_pos(target)
        held = bool(np.linalg.norm(obj - tcp) < 0.08 and obj[2] > 0.12)
        dist = float(np.linalg.norm(tcp - stage))
        out.append(dict(seed=int(seed), target=target, picked=bool(picked), held=held, stage_cm=round(dist * 100, 1),
                        success=bool(held and dist < 0.06), head_gap_cm=round(float(np.linalg.norm(tcp - mouth)) * 100, 1)))
        if len(films) < n_film:
            films.append((out[-1]["success"], frames))
    return out, films


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("episodes", type=int, nargs="?", default=56)
    ap.add_argument("--na", type=int, default=10, help="actions executed per inference (chunk 20 @ 10 Hz; 10 = 1 s)")
    ap.add_argument("--bg", default=str(HELDOUT))
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed0", type=int, default=9_000_000)
    ap.add_argument("--ticks", type=int, default=150)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--film", type=int, default=1)
    args = ap.parse_args()
    seeds = list(range(args.seed0, args.seed0 + args.episodes))
    jobs = [(args.ckpt, seeds[w::args.workers], args.na, args.bg, not args.no_aug, args.film if w == 0 else 0, args.ticks)
            for w in range(args.workers)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        res = pool.map(worker, jobs)
    rows = [r for o, _ in res for r in o]
    n = max(len(rows), 1)
    ok, pk = sum(r["success"] for r in rows), sum(r["picked"] for r in rows)
    tag = f"na{args.na}_{'heldout' if 'heldout' in args.bg else 'trainbg'}{'' if not args.no_aug else '_clean'}"
    by = {}
    for r in rows:
        by.setdefault(r["target"], []).append(r["success"])
    per = " ".join(f"{k}:{sum(v)}/{len(v)}" for k, v in sorted(by.items()))
    print(f"YAM {Path(args.ckpt).name} {tag}: success {ok}/{n} = {100 * ok / n:.0f}%  picked {pk}/{n} = {100 * pk / n:.0f}%  [{per}]  ({time.time() - t0:.0f}s)")
    (Path(args.ckpt) / f"eval_{tag}.json").write_text(json.dumps(rows))
    import imageio.v3 as iio
    import numpy as np
    for o, films in res:
        for i, (succ, frames) in enumerate(films):
            if frames:
                iio.imwrite(Path(args.ckpt) / f"film_{tag}_{'ok' if succ else 'FAIL'}_{i}.png", np.concatenate(frames[:10], 1))
