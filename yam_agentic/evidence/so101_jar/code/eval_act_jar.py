"""Closed-loop evaluation of a jar ACT checkpoint through the compositing pipeline.

Parallel CPU workers (same pattern as rl/native/eval_act_parallel.py). Held-out seeds
(9,000,000+), success = jar lifted >5 cm, upright, held 0.4 s. Writes pdiag JSON next to
the checkpoint and a filmstrip PNG of the first episodes (successes and failures) so the
behaviour can be inspected by eye.

Usage: eval_act_jar.py CKPT_DIR [episodes] [--calib path] [--bg dir] [--na K] [--p-real 0.5]
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path("/Users/adipu/so101Sim")


def worker(a):
    ckpt, seeds, calib, bg, na, p_real, n_film, cam_jit = a
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(ROOT / "rl/jar"))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from jar_scene import JarScene
    from lerobot.policies.act.modeling_act import ACTPolicy
    policy = ACTPolicy.from_pretrained(ckpt)
    policy.to("cpu").eval()
    if na:
        policy.config.n_action_steps = na
    sc = JarScene(calib, bg, p_real=p_real, cam_jitter=cam_jit)
    out, films = [], []
    for seed in seeds:
        sc.reset(seed)
        policy.reset()
        frames, t_succ = [], None
        for t in range(260):
            img, st = sc.render(), sc.state()
            if len(films) < n_film and t % 8 == 0:
                frames.append(img)
            obs = {"observation.images.base": torch.from_numpy(img.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0,
                   "observation.state": torch.from_numpy(st).unsqueeze(0)}
            with torch.no_grad():
                act = policy.select_action(obs)[0].numpy()
            if sc.step(act) and t_succ is None:
                t_succ = t
            if t_succ is not None and t > t_succ + 10:
                break
        out.append({"seed": int(seed), "success": bool(sc.success), "max_rise": round(sc.max_rise, 3),
                    "tilt": round(sc.tilt_deg(), 1), "jar_r": round(float(np.linalg.norm(sc.jar_xy)), 3),
                    "jar_th": round(float(np.arctan2(sc.jar_xy[1], sc.jar_xy[0])), 3), "ticks": t + 1})
        if len(films) < n_film:
            films.append((bool(sc.success), frames))
    return out, films


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("episodes", type=int, nargs="?", default=48)
    ap.add_argument("--calib", default=str(ROOT / "rl/real/camera_calib_side_recommended.json"))
    ap.add_argument("--bg", default=str(ROOT / "rl/real/bg"))
    ap.add_argument("--na", type=int, default=12, help="n_action_steps at execution (chunk 50 @ 25 Hz; 12 = 0.5 s)")
    ap.add_argument("--p-real", type=float, default=0.5)
    ap.add_argument("--cam-jitter", type=float, default=0.04, help="phone placement error range in m (training used 0.04)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=9_000_000)
    args = ap.parse_args()
    import numpy as np
    import cv2
    seeds = list(range(args.seed0, args.seed0 + args.episodes))
    jobs = [(args.ckpt, seeds[w::args.workers], args.calib, args.bg, args.na, args.p_real, 1, args.cam_jitter) for w in range(args.workers)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        res = pool.map(worker, jobs)
    rec = [r for out, _ in res for r in out]
    wins = sum(r["success"] for r in rec)
    tag = f"na{args.na}_preal{args.p_real}" + (f"_cj{args.cam_jitter}" if args.cam_jitter != 0.04 else "")
    if os.environ.get("EVAL_TAG"):                           # keeps stress-test runs from overwriting the plain pdiag
        tag += "_" + os.environ["EVAL_TAG"]
    Path(args.ckpt, f"pdiag_{tag}_{args.episodes}.json").write_text(json.dumps(rec, indent=1))
    rows = []
    for _, films in res:
        for ok, frames in films:
            pick = [frames[i] for i in np.linspace(0, len(frames) - 1, 8).astype(int)]
            strip = np.hstack(pick)
            cv2.rectangle(strip, (0, 0), (150, 22), (0, 0, 0), -1)
            cv2.putText(strip, "SUCCESS" if ok else "FAIL", (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (80, 255, 80) if ok else (255, 80, 80), 2)
            rows.append(strip)
    if rows:
        cv2.imwrite(str(Path(args.ckpt, f"filmstrip_{tag}.png")), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    print(f"ACT jar {Path(args.ckpt).name} {tag}: {wins}/{len(rec)} = {wins/len(rec):.0%} "
          f"(mean max rise {np.mean([r['max_rise'] for r in rec]):.3f} m, {time.time()-t0:.0f}s)")
