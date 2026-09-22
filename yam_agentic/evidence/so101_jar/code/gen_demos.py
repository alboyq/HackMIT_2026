"""Side-grasp expert demos through the compositing pipeline -> npz shards.

Usage: gen_demos.py OUT_DIR [episodes] [--calib path] [--bg dir] [--workers 8]
Each shard: images (N,256,256,3) u8, state (N,6) f32, action (N,6) f32, episode (N,) i64.
Only successful episodes are kept. 25 Hz; action = the expert's absolute joint targets.
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path("/Users/adipu/so101Sim")


def worker(a):
    wid, seeds, calib, bg, out, speed, p_real = a
    os.environ["SIDE_SPEED"] = str(speed)
    os.environ["SIDE_DWELL"] = "0.3"
    os.environ.setdefault("SIDE_ZFRAC", "0.58")
    sys.path.insert(0, str(ROOT / "rl/jar"))
    sys.path.insert(0, str(ROOT / "rl/native"))
    import numpy as np
    from jar_scene import JarScene
    from scripted_expert_side import SideExpert
    sc = JarScene(calib, bg, p_real=p_real)
    ex = SideExpert(sc.env)
    I, S, A, E = [], [], [], []
    kept = tried = 0
    for seed in seeds:
        tried += 1
        sc.reset(seed)
        ex.reset()
        fi, fs, fa, tail = [], [], [], 0
        for t in range(400):
            img, st = sc.render(), sc.state()
            ctrl = ex.act()
            fi.append(img); fs.append(st); fa.append(ctrl.astype(np.float32))
            sc.step(ctrl)
            if ex.done:
                tail += 1
                if tail > 8:
                    break
        if sc.success:
            kept += 1
            I += fi; S += fs; A += fa; E += [seed] * len(fi)
    np.savez(Path(out) / f"shard_{wid:02d}.npz", images=np.array(I, np.uint8), state=np.array(S, np.float32),
             action=np.array(A, np.float32), episode=np.array(E, np.int64))
    return wid, kept, tried, len(E)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("episodes", type=int, nargs="?", default=300)
    ap.add_argument("--calib", default=str(ROOT / "rl/real/camera_calib_side_recommended.json"))
    ap.add_argument("--bg", default=str(ROOT / "rl/real/bg"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--speed", type=float, default=1.5)
    ap.add_argument("--p-real", type=float, default=0.5, help="fraction of episodes on the real-scene photos")
    ap.add_argument("--seed0", type=int, default=6_000_000)
    args = ap.parse_args()
    if "bg_heldout" in str(Path(args.bg).resolve()):          # evaluation-only photos: see rl/jar/bg_heldout/README.md
        sys.exit("refusing to generate TRAINING demos on rl/jar/bg_heldout - that set is held out for evaluation")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = int(args.episodes * 1.08) + args.workers                      # a few spares for failed episodes
    seeds = list(range(args.seed0, args.seed0 + n))
    jobs = [(w, seeds[w::args.workers], args.calib, args.bg, str(out), args.speed, args.p_real)
            for w in range(args.workers)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        res = pool.map(worker, jobs)
    kept, tried, frames = sum(r[1] for r in res), sum(r[2] for r in res), sum(r[3] for r in res)
    (out / "meta.txt").write_text(f"calib={args.calib}\nbg={args.bg}\np_real={args.p_real}\n"
                                  f"episodes={kept}/{tried}\nframes={frames}\nfps=25\n")
    print(f"DONE {kept}/{tried} episodes kept, {frames} frames, {time.time()-t0:.0f}s -> {out}")
