"""The baselines every fit number has to be compared against.

    python tools/baselines.py <demo_dir> [--mlp]

The project's original baseline was "predict the dataset's mean action" (~14 deg). That is the
wrong yardstick for a policy emitting ABSOLUTE joint targets to a slow arm at 10 Hz: the honest
floor is what a controller could trivially fall back on. Measured 2026-09-20 on yam_v6:

    mean action ......... 14.05 deg     <- the old baseline
    hold still ..........  1.73 deg     <- predict the current joint angles
    constant velocity ...  1.20 deg     <- extrapolate the last step
    ACT step_12000 ......  2.70 deg     <- worse than freezing the arm

--mlp additionally trains a 3x512 MLP on the 13-number prompt (6 joints + gripper + object box)
with NO images at all, held out by episode. It reaches 0.62 deg in ~17 s. A policy with two
cameras that cannot beat that has not converged.
"""
import argparse
import glob
import numpy as np

DEG = 180.0 / np.pi


def load(demo_dir, limit=None):
    S, A, E, off = [], [], [], 0
    for p in sorted(glob.glob(f"{demo_dir}/shard_*.npz"))[:limit]:
        z = np.load(p)
        S.append(z["state"]); A.append(z["action"]); E.append(z["episode"] + off)
        off = int(E[-1].max()) + 1
    if not S:
        raise SystemExit(f"no shard_*.npz under {demo_dir}")
    return np.concatenate(S), np.concatenate(A), np.concatenate(E)


def baselines(S, A, E):
    prev = np.vstack([S[:1, :6], S[:-1, :6]])
    same = np.r_[False, E[1:] == E[:-1]]
    cv = np.where(same[:, None], 2 * S[:, :6] - prev, S[:, :6])
    return {
        "predict the mean action (old baseline)": np.abs(A[:, :6] - A[:, :6].mean(0)).mean() * DEG,
        "hold still (predict current joints)": np.abs(A[:, :6] - S[:, :6]).mean() * DEG,
        "constant-velocity extrapolation": np.abs(A[:, :6] - cv).mean() * DEG,
    }


def prompt_only_mlp(S, A, E, steps=12000):
    """13 numbers in, next action out. No cameras. Held out BY EPISODE."""
    import torch, torch.nn as nn
    X, Y = S[:, :13].astype(np.float32), A[:, :6].astype(np.float32)
    eps = np.unique(E)
    rng = np.random.default_rng(0); rng.shuffle(eps)
    te = set(eps[:max(1, len(eps) // 10)].tolist())
    m = np.array([e in te for e in E])
    mu, sd = X[~m].mean(0), X[~m].std(0) + 1e-6
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    Xtr = torch.tensor((X[~m] - mu) / sd).to(dev); Ytr = torch.tensor(Y[~m]).to(dev)
    Xte = torch.tensor((X[m] - mu) / sd).to(dev); Yte = torch.tensor(Y[m]).to(dev)
    net = nn.Sequential(nn.Linear(13, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(),
                        nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 6)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        b = torch.randint(0, len(Xtr), (512,), device=dev)
        loss = (net(Xtr[b]) - Ytr[b]).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    with torch.no_grad():
        return (net(Xte) - Yte).abs().mean().item() * DEG, len(te)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("demos")
    ap.add_argument("--shards", type=int, default=None, help="limit shards (they are large)")
    ap.add_argument("--mlp", action="store_true", help="also train the prompt-only MLP")
    a = ap.parse_args()
    S, A, E = load(a.demos, a.shards)
    print(f"{len(S)} frames / {len(np.unique(E))} episodes from {a.demos}\n")
    print("first-action MAE, all frames:")
    for k, v in baselines(S, A, E).items():
        print(f"  {k:42s} {v:6.2f} deg")
    if a.mlp:
        import time
        t0 = time.time()
        mae, n_te = prompt_only_mlp(S, A, E)
        print(f"\n  prompt-only MLP, NO images, {n_te} held-out episodes")
        print(f"  {'(0.5 M params, ' + f'{time.time()-t0:.0f}' + ' s)':42s} {mae:6.2f} deg")
