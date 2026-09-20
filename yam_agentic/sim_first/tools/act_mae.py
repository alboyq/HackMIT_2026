"""A checkpoint's first-action MAE, against the baselines that matter, on the SAME frames.

    python tools/act_mae.py <ckpt_dir> <demo_dir> [n_frames]

Measured 2026-09-20, yam_v9/act/step_12000 on 320 frames of its own training set:
    ACT 2.70 deg   vs   hold still 1.95 deg   vs   mean action 12.80 deg
i.e. the policy is worse than freezing the arm. See ../VLA_VERDICT.md section 1.
"""
import glob
import sys
import numpy as np
import torch
from lerobot.policies.act.modeling_act import ACTPolicy

CK, DEMOS = sys.argv[1], sys.argv[2]
NF = int(sys.argv[3]) if len(sys.argv) > 3 else 320
DEG = 180.0 / np.pi
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
pol = ACTPolicy.from_pretrained(CK).to(dev).eval()

sc, wr, st, ac = [], [], [], []
for f in sorted(glob.glob(f"{DEMOS}/shard_*.npz"))[:2]:
    z = np.load(f)
    sc.append(z["scene"]); wr.append(z["wrist"]); st.append(z["state"]); ac.append(z["action"])
sc, wr = np.concatenate(sc), np.concatenate(wr)
st, ac = np.concatenate(st), np.concatenate(ac)
idx = np.random.default_rng(0).choice(len(st), min(NF, len(st)), replace=False)

errs = []
with torch.no_grad():
    for s in range(0, len(idx), 32):
        b = idx[s:s + 32]
        batch = {"observation.images.scene": torch.from_numpy(sc[b]).permute(0, 3, 1, 2).float().div(255).to(dev),
                 "observation.images.wrist": torch.from_numpy(wr[b]).permute(0, 3, 1, 2).float().div(255).to(dev),
                 "observation.state": torch.from_numpy(st[b]).to(dev)}
        batch["observation.images"] = [batch[k] for k in pol.config.image_features]
        pred = pol.model(batch)[0][:, 0, :6].float().cpu().numpy()   # first action of the chunk
        errs.append(np.abs(pred - ac[b][:, :6]).mean(1) * DEG)
errs = np.concatenate(errs)
print(f"\nfirst-action MAE on {len(idx)} frames of {DEMOS}")
print(f"  {CK.rstrip('/').split('/')[-1]:38s} {errs.mean():6.2f} deg   <- the policy")
print(f"  {'hold still (predict current joints)':38s} {np.abs(ac[idx][:, :6] - st[idx][:, :6]).mean() * DEG:6.2f} deg")
print(f"  {'predict the mean action':38s} {np.abs(ac[idx][:, :6] - ac[:, :6].mean(0)).mean() * DEG:6.2f} deg")
