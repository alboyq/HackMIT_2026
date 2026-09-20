"""The prompt-only baseline policy: state(13) -> action chunk. No cameras, ~30 s to train.

    python train_mlp_yam.py DEMO_DIR OUT_DIR [chunk]

Exists to keep everything else honest. It sees only the 13 numbers ACT already gets in its state
vector (6 joints + gripper + the object box) and reaches 0.62-0.75 deg first-action MAE held out
by episode -- 3.6x better than the ACT checkpoint that had two cameras as well. Any visuomotor
policy that cannot beat this has not converged.

It is NOT a solution: closed-loop it still scores ~2 % (see eval_mlp_yam.py and
../VLA_VERDICT.md section 3), which is the point -- open-loop MAE does not predict success.
"""
import glob
import sys
import numpy as np
import torch
import torch.nn as nn

DEMOS, OUT = sys.argv[1], sys.argv[2]
CHUNK = int(sys.argv[3]) if len(sys.argv) > 3 else 20
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 20_000
DEG = 180.0 / np.pi

S, A, E, off = [], [], [], 0
for p in sorted(glob.glob(f"{DEMOS}/shard_*.npz")):
    z = np.load(p)
    S.append(z["state"]); A.append(z["action"]); E.append(z["episode"] + off)
    off = int(E[-1].max()) + 1
S, A, E = np.concatenate(S), np.concatenate(A), np.concatenate(E)

# chunk targets padded with the episode's last action - the same convention as train_act_yam.py
st = np.flatnonzero(np.diff(E, prepend=E[0] - 1)); en = np.append(st[1:], len(E))
ep_end = np.empty(len(E), np.int64)
for s0, e0 in zip(st, en):
    ep_end[s0:e0] = e0
idx = np.arange(len(E))[:, None] + np.arange(CHUNK)[None, :]
Y = A[np.minimum(idx, ep_end[:, None] - 1)].reshape(len(E), -1).astype(np.float32)
X = S[:, :13].astype(np.float32)

eps = np.unique(E)
rng = np.random.default_rng(0); rng.shuffle(eps)
te = set(eps[:max(1, len(eps) // 10)].tolist())
m = np.array([e in te for e in E])
mu, sd = X[~m].mean(0), X[~m].std(0) + 1e-6
ymu, ysd = Y[~m].mean(0), Y[~m].std(0) + 1e-6
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
Xtr = torch.tensor((X[~m] - mu) / sd).to(dev); Ytr = torch.tensor((Y[~m] - ymu) / ysd).to(dev)
Xte = torch.tensor((X[m] - mu) / sd).to(dev); Yte = torch.tensor(Y[m]).to(dev)
print(f"train {len(Xtr)} / heldout {len(Xte)} frames ({len(te)} episodes), chunk {CHUNK}, on {dev}", flush=True)

net = nn.Sequential(nn.Linear(13, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(),
                    nn.Linear(512, 512), nn.GELU(), nn.Linear(512, Y.shape[1])).to(dev)
opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
ym, ys = torch.tensor(ymu).to(dev), torch.tensor(ysd).to(dev)
for i in range(STEPS):
    b = torch.randint(0, len(Xtr), (512,), device=dev)
    loss = (net(Xtr[b]) - Ytr[b]).abs().mean()
    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if (i + 1) % 5000 == 0:
        with torch.no_grad():
            p = net(Xte) * ys + ym
            print(f"  step {i+1:6d}  heldout first-action {(p[:, :6] - Yte[:, :6]).abs().mean().item() * DEG:5.2f} deg"
                  f"   whole-chunk {(p - Yte).abs().mean().item() * DEG:5.2f} deg", flush=True)
torch.save({"sd": net.state_dict(), "mu": mu, "std": sd, "ymu": ymu, "ysd": ysd,
            "chunk": CHUNK, "adim": A.shape[1]}, f"{OUT}/mlp.pt")
print("saved", f"{OUT}/mlp.pt")
