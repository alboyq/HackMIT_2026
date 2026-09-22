"""Which inputs is a trained policy actually using?

    python tools/attribution.py <ckpt_dir> <demo_dir>

Shuffles each input across the batch so it carries no information about the frame, and measures
how far the prediction moves. A shuffle that changes nothing is an input the policy ignores.

Measured 2026-09-20, yam_v9/act/step_12000 (baseline error 2.59 deg):
    joints 10.54 | scene image 7.28 | wrist image 3.79 | box-in-state 0.90
So ACT is NOT blind to the cameras -- it simply has not converged. ../VLA_VERDICT.md section 5.
"""
import glob
import sys
import numpy as np
import torch
from lerobot.policies.act.modeling_act import ACTPolicy

CK, DEMOS = sys.argv[1], sys.argv[2]
NF = int(sys.argv[3]) if len(sys.argv) > 3 else 256
DEG = 180.0 / np.pi
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
pol = ACTPolicy.from_pretrained(CK).to(dev).eval()
z = np.load(sorted(glob.glob(f"{DEMOS}/shard_*.npz"))[0])
rng = np.random.default_rng(0)
idx = rng.choice(len(z["state"]), NF, replace=False)
SC, WR, ST, AC = z["scene"][idx], z["wrist"][idx], z["state"][idx], z["action"][idx]
perm = rng.permutation(NF)


def run(scene, wrist, state):
    out = []
    with torch.no_grad():
        for s in range(0, NF, 32):
            b = {"observation.images.scene": torch.from_numpy(scene[s:s+32]).permute(0, 3, 1, 2).float().div(255).to(dev),
                 "observation.images.wrist": torch.from_numpy(wrist[s:s+32]).permute(0, 3, 1, 2).float().div(255).to(dev),
                 "observation.state": torch.from_numpy(state[s:s+32]).to(dev)}
            b["observation.images"] = [b[k] for k in pol.config.image_features]
            out.append(pol.model(b)[0][:, 0, :6].float().cpu().numpy())
    return np.concatenate(out)


base = run(SC, WR, ST)
print(f"baseline first-action MAE vs truth: {np.abs(base - AC[:, :6]).mean() * DEG:.2f} deg\n")
print("input shuffled across the batch   ->  how far the prediction moves")
for name, args in {
    "joints (state 0-6)": (SC, WR, np.concatenate([ST[perm][:, :7], ST[:, 7:]], 1)),
    "scene camera image": (SC[perm], WR, ST),
    "wrist camera image": (SC, WR[perm], ST),
    "both camera images": (SC[perm], WR[perm], ST),
    "box (state 7-12)": (SC, WR, np.concatenate([ST[:, :7], ST[perm][:, 7:13], ST[:, 13:]], 1)),
    "whole state vector": (SC, WR, ST[perm]),
}.items():
    print(f"  {name:32s} {np.abs(run(*args) - base).mean() * DEG:7.2f} deg")
