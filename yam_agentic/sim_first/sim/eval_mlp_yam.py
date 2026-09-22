"""Closed-loop evaluation of the prompt-only MLP baseline, in the SAME scene and with the
same success test as eval_act_yam.py. The policy never sees an image: its whole input is
the 13 numbers (6 joints + gripper + object box) that ACT also gets in its state vector.

Usage: eval_mlp_yam.py MLP_PT [episodes] [--na 10] [--bg DIR] [--workers 14]
"""
import argparse, json, multiprocessing as mp, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
HELDOUT = HERE.parent / "jar" / "bg_heldout" / "full"


def worker(a):
    blob_path, seeds, na, bg, aug, max_ticks = a
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(HERE))
    import numpy as np, torch, torch.nn as nn
    torch.set_num_threads(1)
    from yam_data import DataScene
    from yam_expert import STAGING, Expert
    blob = torch.load(blob_path, map_location="cpu", weights_only=False)
    CHUNK, AD = blob["chunk"], blob["adim"]
    net = nn.Sequential(nn.Linear(13, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(),
                        nn.Linear(512, 512), nn.GELU(), nn.Linear(512, CHUNK * AD))
    net.load_state_dict(blob["sd"]); net.eval()
    mu, sd, ymu, ysd = blob["mu"], blob["std"], blob["ymu"], blob["ysd"]
    sc = DataScene(bg_dir=bg, aug=aug)
    ex = Expert(sc)
    out = []
    for seed in seeds:
        target = sc.randomize(int(seed))
        hold = np.concatenate([sc.data.ctrl[:6], [1.0]]).astype(np.float32)
        for _ in range(3):
            sc.tick(hold, hold)
        if sc.capture_box() is None:
            continue
        mouth = sc.site("mouth")
        d = mouth - np.array([0, 0, mouth[2]]); d[2] = 0; d /= np.linalg.norm(d)
        stage = mouth - d * STAGING
        z0, prev, picked = sc.object_pos(target)[2], hold, False
        chunk, k = None, 0
        for t in range(max_ticks):
            _, _, state = sc.observe()
            if chunk is None or k >= na:
                x = torch.tensor(((state[:13] - mu) / sd)[None].astype(np.float32))
                with torch.no_grad():
                    y = net(x)[0].numpy() * ysd + ymu
                chunk, k = y.reshape(CHUNK, AD), 0
            act = chunk[k].astype(np.float32); k += 1
            sc.tick(act, prev)
            prev = act
            picked = picked or (sc.object_pos(target)[2] - z0 > 0.05)
        tcp = ex.ik.fk(sc.q_arm, qpos_full=sc.data.qpos)[0]
        obj = sc.object_pos(target)
        held = bool(np.linalg.norm(obj - tcp) < 0.08 and obj[2] > 0.12)
        dist = float(np.linalg.norm(tcp - stage))
        out.append(dict(seed=int(seed), target=target, picked=bool(picked), held=held,
                        stage_cm=round(dist * 100, 1), success=bool(held and dist < 0.06)))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mlp"); ap.add_argument("episodes", type=int, nargs="?", default=56)
    ap.add_argument("--na", type=int, default=10)
    ap.add_argument("--bg", default=str(HELDOUT))
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed0", type=int, default=9_000_000)
    ap.add_argument("--ticks", type=int, default=150)
    ap.add_argument("--no-aug", action="store_true")
    args = ap.parse_args()
    seeds = list(range(args.seed0, args.seed0 + args.episodes))
    jobs = [(args.mlp, seeds[w::args.workers], args.na, args.bg, not args.no_aug, args.ticks)
            for w in range(args.workers)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        res = pool.map(worker, jobs)
    rows = [r for o in res for r in o]
    n = max(len(rows), 1)
    ok, pk = sum(r["success"] for r in rows), sum(r["picked"] for r in rows)
    by = {}
    for r in rows:
        by.setdefault(r["target"], []).append(r["success"])
    per = " ".join(f"{k}:{sum(v)}/{len(v)}" for k, v in sorted(by.items()))
    med = sorted(r["stage_cm"] for r in rows)[n // 2]
    print(f"MLP na{args.na}: success {ok}/{n} = {100*ok/n:.0f}%  picked {pk}/{n} = {100*pk/n:.0f}%  "
          f"[{per}]  median stage dist {med} cm  ({time.time()-t0:.0f}s)")
    Path(args.mlp).with_suffix(".eval.json").write_text(json.dumps(rows))
