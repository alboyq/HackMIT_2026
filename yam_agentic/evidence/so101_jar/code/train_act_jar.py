"""RAM-resident LeRobot ACT training on the jar side-grasp demos (npz shards).

Same conventions as rl/native/train_act2.py (the 72% cube champion): images float [0,1]
CHW, raw radians for state/action, no normalisation, AdamW with ACT's default lrs, batch 8,
non-finite guards. Single camera, 25 Hz data, chunk 50 (= 2 s).

Usage: train_act_jar.py DEMO_DIR OUT_DIR [steps]
"""
import copy
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

DEMOS, OUT = Path(sys.argv[1]), Path(sys.argv[2])
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 15_000
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", 3_000))
BATCH = int(os.environ.get("BATCH", 8))
CHUNK = int(os.environ.get("CHUNK_SIZE", 50))
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
IMG_KEY = "observation.images.base"

shards = sorted(glob.glob(str(DEMOS / "shard_*.npz")))[:int(os.environ.get("MAX_DEMO_SHARDS", 99))]
dag = [f for d in os.environ.get("DAGGER_DIRS", "").split(":") if d for f in sorted(glob.glob(str(Path(d) / "dagger_*.npz")))]
files = [(f, False) for f in shards] + [(f, True) for f in dag]
counts = [len(np.load(f)["episode"]) for f, _ in files]
N = int(sum(counts))
first = np.load(files[0][0])["images"]
IM = torch.empty((N, 3, first.shape[1], first.shape[2]), dtype=torch.uint8)     # preallocate: no 2x RAM peak
S = torch.empty((N, 6)); CH = torch.empty((N, CHUNK, 6)); PD = torch.zeros((N, CHUNK), dtype=torch.bool)
del first
o = 0
for (f, is_dag), c in zip(files, counts):
    z = np.load(f)
    IM[o:o + c] = torch.from_numpy(z["images"]).permute(0, 3, 1, 2)
    S[o:o + c] = torch.from_numpy(z["state"])
    if is_dag:                                   # DAgger shards carry explicit expert chunks per visited state
        CH[o:o + c] = torch.from_numpy(z["chunks"]); PD[o:o + c] = torch.from_numpy(z["pad"])
    else:                                        # demo shards: chunk = the episode's next CHUNK actions, padded with the last
        A, ep = z["action"], z["episode"]
        st = np.flatnonzero(np.diff(ep, prepend=ep[0] - 1)); en = np.append(st[1:], c)
        ep_end = np.empty(c, np.int64)
        for s0, e0 in zip(st, en):
            ep_end[s0:e0] = e0
        idx = np.arange(c)[:, None] + np.arange(CHUNK)[None, :]
        PD[o:o + c] = torch.from_numpy(idx >= ep_end[:, None])
        CH[o:o + c] = torch.from_numpy(A[np.minimum(idx, ep_end[:, None] - 1)])
    o += c
    del z
print(f"loaded {N} frames ({sum(c for (f, d), c in zip(files, counts) if d)} from DAgger), images {tuple(IM.shape)} ({IM.numel()/1e9:.1f} GB)", flush=True)

h, w = IM.shape[2], IM.shape[3]
cfg = ACTConfig(
    input_features={IMG_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, h, w)),
                    "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,))},
    output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
    chunk_size=CHUNK, n_action_steps=CHUNK, device=DEVICE,
    # ACT_DILATION=1: ResNet stride 32 -> 16, i.e. a 16x16 instead of 8x8 feature grid (finer localisation, slower)
    replace_final_stride_with_dilation=os.environ.get("ACT_DILATION") == "1")
policy = ACTPolicy(cfg).to(DEVICE)
policy.train()


def make_batch(idx):
    return {IMG_KEY: IM[idx].to(DEVICE).float() / 255.0, "observation.state": S[idx].to(DEVICE),
            "action": CH[idx].to(DEVICE), "action_is_pad": PD[idx].to(DEVICE)}


params = [{"params": [p for n, p in policy.named_parameters() if "backbone" not in n and p.requires_grad],
           "lr": cfg.optimizer_lr},
          {"params": [p for n, p in policy.named_parameters() if "backbone" in n and p.requires_grad],
           "lr": cfg.optimizer_lr_backbone}]
opt = torch.optim.AdamW(params, weight_decay=cfg.optimizer_weight_decay)
OUT.mkdir(parents=True, exist_ok=True)
log = open(OUT / "train_progress.txt", "a", buffering=1)
good = copy.deepcopy(policy.state_dict())
gen = torch.Generator().manual_seed(0)
run_loss, skipped, step, t0 = 0.0, 0, 0, time.time()
while step < STEPS:
    batch = make_batch(torch.randint(0, N, (BATCH,), generator=gen))
    loss, _ = policy.forward(batch)
    if not torch.isfinite(loss):
        skipped += 1
        if skipped % 10 == 0:
            policy.load_state_dict(good)
        continue
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
    opt.step()
    run_loss += float(loss.detach())
    step += 1
    if step % 250 == 0:
        if all(torch.isfinite(p).all() for p in policy.parameters()):
            good = copy.deepcopy(policy.state_dict())
        else:
            policy.load_state_dict(good)
        msg = f"step {step}/{STEPS} loss {run_loss/250:.4f} ({step/(time.time()-t0):.1f} it/s, skipped {skipped})"
        print(msg, flush=True); log.write(msg + "\n")
        run_loss = 0.0
    if step % CKPT_EVERY == 0 or step == STEPS:
        policy.save_pretrained(OUT / f"step_{step}")
        log.write(f"ckpt step_{step}\n")
log.write("done\n")
print("done")
