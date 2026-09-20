"""RAM-resident LeRobot ACT training for the sim-first YAM policy.

Two cameras (scene + wrist), state(17) = joints + gripper + object box + mouth prompt, action(7).
Same conventions as rl/jar/train_act_jar.py: images u8 in RAM -> float [0,1] per batch, raw radians,
no normalisation, AdamW with ACT's default lrs, non-finite guards. Chunk 20 at 10 Hz = 2 s.

Runs on CUDA (GX10), MPS (Mac) or CPU. On CUDA it uses bf16 autocast and channels_last.

Usage: train_act_yam.py DEMO_DIR OUT_DIR [steps]      env: BATCH CHUNK_SIZE CKPT_EVERY MAX_SHARDS
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
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 20_000
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", 2_500))
BATCH = int(os.environ.get("BATCH", 24))
CHUNK = int(os.environ.get("CHUNK_SIZE", 20))
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
K_SCENE, K_WRIST = "observation.images.scene", "observation.images.wrist"

files = sorted(glob.glob(str(DEMOS / "shard_*.npz")))[:int(os.environ.get("MAX_SHARDS", 999))]
counts = [len(np.load(f)["episode"]) for f in files]
N = int(sum(counts))
z0 = np.load(files[0])
hs, hw = z0["scene"].shape[1], z0["wrist"].shape[1]
SD, AD = z0["state"].shape[1], z0["action"].shape[1]
del z0
SC = torch.empty((N, 3, hs, hs), dtype=torch.uint8)                       # preallocate: no 2x RAM peak
WR = torch.empty((N, 3, hw, hw), dtype=torch.uint8)
S = torch.empty((N, SD)); CH = torch.empty((N, CHUNK, AD)); PD = torch.zeros((N, CHUNK), dtype=torch.bool)
o = 0
for f, c in zip(files, counts):
    z = np.load(f)
    SC[o:o + c] = torch.from_numpy(z["scene"]).permute(0, 3, 1, 2)
    WR[o:o + c] = torch.from_numpy(z["wrist"]).permute(0, 3, 1, 2)
    S[o:o + c] = torch.from_numpy(z["state"])
    A, ep = z["action"], z["episode"]
    st = np.flatnonzero(np.diff(ep, prepend=ep[0] - 1)); en = np.append(st[1:], c)
    ep_end = np.empty(c, np.int64)
    for s0, e0 in zip(st, en):
        ep_end[s0:e0] = e0
    idx = np.arange(c)[:, None] + np.arange(CHUNK)[None, :]
    PD[o:o + c] = torch.from_numpy(idx >= ep_end[:, None])                # chunk = next CHUNK actions, padded with the last
    CH[o:o + c] = torch.from_numpy(A[np.minimum(idx, ep_end[:, None] - 1)])
    o += c
    del z
print(f"loaded {N} frames from {len(files)} shards: scene {tuple(SC.shape)} wrist {tuple(WR.shape)} "
      f"({(SC.numel() + WR.numel()) / 1e9:.1f} GB) on {DEVICE}", flush=True)

cfg = ACTConfig(
    input_features={K_SCENE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, hs, hs)),
                    K_WRIST: PolicyFeature(type=FeatureType.VISUAL, shape=(3, hw, hw)),
                    "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(SD,))},
    output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(AD,))},
    chunk_size=CHUNK, n_action_steps=CHUNK, device=DEVICE)
policy = ACTPolicy(cfg).to(DEVICE)
policy.train()
CUDA = DEVICE == "cuda"
if CUDA:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if SC.numel() + WR.numel() < 8e9:                                     # pinning tens of GB on a unified-memory box is asking for trouble
        SC, WR = SC.pin_memory(), WR.pin_memory()


def make_batch(idx):
    nb = CUDA
    return {K_SCENE: SC[idx].to(DEVICE, non_blocking=nb).float() / 255.0,
            K_WRIST: WR[idx].to(DEVICE, non_blocking=nb).float() / 255.0,
            "observation.state": S[idx].to(DEVICE, non_blocking=nb),
            "action": CH[idx].to(DEVICE, non_blocking=nb), "action_is_pad": PD[idx].to(DEVICE, non_blocking=nb)}


params = [{"params": [p for n, p in policy.named_parameters() if "backbone" not in n and p.requires_grad], "lr": cfg.optimizer_lr},
          {"params": [p for n, p in policy.named_parameters() if "backbone" in n and p.requires_grad], "lr": cfg.optimizer_lr_backbone}]
opt = torch.optim.AdamW(params, weight_decay=cfg.optimizer_weight_decay)
OUT.mkdir(parents=True, exist_ok=True)
log = open(OUT / "train_progress.txt", "a", buffering=1)
good = copy.deepcopy(policy.state_dict())
gen = torch.Generator().manual_seed(int(os.environ.get("SEED", 0)))
run_loss, skipped, step, t0, resumed_elapsed = 0.0, 0, 0, time.time(), 0.0
# Resume: RESUME=0 disables. Picks up the newest complete checkpoint (weights + optimizer + RNG),
# so pulling the plug costs at most CKPT_EVERY steps.
if os.environ.get("RESUME", "1") == "1" and (OUT / "LATEST").exists():
    last = int((OUT / "LATEST").read_text().strip())
    ck = OUT / f"step_{last}"
    if (ck / "model.safetensors").exists():
        policy = ACTPolicy.from_pretrained(ck).to(DEVICE)
        policy.train()
        params = [{"params": [p for n, p in policy.named_parameters() if "backbone" not in n and p.requires_grad], "lr": cfg.optimizer_lr},
                  {"params": [p for n, p in policy.named_parameters() if "backbone" in n and p.requires_grad], "lr": cfg.optimizer_lr_backbone}]
        opt = torch.optim.AdamW(params, weight_decay=cfg.optimizer_weight_decay)
        step = last
        if (ck / "trainer.pt").exists():                       # full resume: optimiser + RNG too
            blob = torch.load(ck / "trainer.pt", map_location=DEVICE, weights_only=False)
            opt.load_state_dict(blob["opt"])
            gen.set_state(blob["gen"].cpu() if hasattr(blob["gen"], "cpu") else blob["gen"])
            step, resumed_elapsed = blob["step"], blob.get("elapsed", 0.0)
        else:                                                  # weights-only (checkpoint from an older run):
            print("no trainer.pt - resuming weights only, optimiser moments rebuild in a few hundred steps", flush=True)
        good = copy.deepcopy(policy.state_dict())
        print(f"RESUMED from {ck.name} ({resumed_elapsed / 60:.0f} min already spent)", flush=True)
        log.write(f"resumed from step_{step}\n")
while step < STEPS:
    batch = make_batch(torch.randint(0, N, (BATCH,), generator=gen))
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=CUDA):
        loss, _ = policy.forward(batch)
    if not torch.isfinite(loss):
        skipped += 1
        if skipped % 10 == 0:
            policy.load_state_dict(good)
        continue
    opt.zero_grad(set_to_none=True)
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
        msg = f"step {step}/{STEPS} loss {run_loss / 250:.4f} ({step / (time.time() - t0):.2f} it/s, {BATCH * step / (time.time() - t0):.0f} samples/s, skipped {skipped})"
        print(msg, flush=True); log.write(msg + "\n")
        run_loss = 0.0
    if step % CKPT_EVERY == 0 or step == STEPS:
        ck = OUT / f"step_{step}"
        policy.save_pretrained(ck)
        torch.save({"step": step, "opt": opt.state_dict(), "gen": gen.get_state(),
                    "elapsed": time.time() - t0 + resumed_elapsed}, ck / "trainer.pt")
        (OUT / "LATEST").write_text(str(step))                 # unplug-safe: where to resume from
        log.write(f"ckpt step_{step}\n")
log.write("done\n")
print("done")
