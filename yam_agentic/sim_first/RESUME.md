# Resume / move checklist

Everything here survives an unplug. Nothing needs a network connection to restart, and no stage
redoes work it has already done. If the rig moves, follow this.

## Before you unplug

Nothing. Both machines checkpoint continuously:

| Thing | Where it lives | Lost on unplug |
|---|---|---|
| Demonstration datasets (npz shards) | `<run>/demos` wherever you generated them (GX10 `~/yam_simfirst/runs/`, or a Mac checkout). Too large for the repo. | nothing — written once, reread on restart |
| Training | checkpoint + optimiser state + RNG every 1000 steps, plus an `act/LATEST` pointer | at most 1000 steps (~2 min on the GX10) |
| Closed-loop eval results | appended to `runs/<run>/results.txt` as each finishes | the one in flight |
| Code | this repo | nothing, once pushed |
| Background photo pools | `bg_train/`, `bg_heldout/full/` beside the sim. Not in the repo (800+ photos). | re-fetchable: [`bg_heldout/fetch.sh`](../evidence/so101_jar/bg_heldout/fetch.sh), and `bg_train` is the same trick with seeds `yamtrain1..800` |

If you have a spare minute before pulling power, `touch` nothing — just pull it. The trainer
writes checkpoints atomically enough that a half-written one simply fails the `model.safetensors`
existence check and the previous one is used.

## After you plug back in

**GX10** (does generation, training and eval on its own):

```bash
ssh gx10-lan
cd ~/yam_simfirst
bash /tmp/relaunch_gx10.sh     # or, equivalently:
#   setsid nohup ./chain_g1.sh yam_g1 2500 16000 > chain_g1.log 2>&1 < /dev/null &
tail -f runs/yam_g1/train.log  # expect "RESUMED from step_N"
```

`chain_g1.sh` skips generation when `runs/<run>/demos/meta.txt` exists, resumes training from
`act/LATEST`, and skips evals already recorded in `results.txt`. Running it twice is safe.

**Mac** (sim development, and a second training pipeline if wanted):

```bash
cd yam_agentic/sim_first/sim
BATCH=8 CKPT_EVERY=4000 ./run_chain_yam.sh <run> 20000    # same resume behaviour
```

**Check nothing is doubled up** — two trainers on one GPU halves throughput:

```bash
pgrep -af "python train_act_yam" | wc -l     # want 1 per machine
```

Note: do not `pkill -f chain_g1.sh` from an ssh one-liner — the pattern matches the ssh
session's own command line and kills the shell before it can relaunch. Kill by PID, or run the
kill from a script file (that is what `/tmp/relaunch_gx10.sh` is for).

## If the venue changes

The policy is trained on randomised backgrounds, camera poses and camera effects, so a new room
needs no recalibration of the *sim*. What does need redoing:

1. Re-capture real reference frames (they set the augmentation ranges and get composited in):
   [`real_ref/`](real_ref/) in this repo (scene + wrist), `real_wrist/` on the GX10 — see `capture` notes in HANDOFF.
2. Re-check the scene camera actually frames the table and the user's face.
3. Nothing else. No marker board, no intrinsics, no camera-to-robot transform: the policy takes
   2D prompts only.

## Current state

**Read [NEXT_AGENT.md](NEXT_AGENT.md) first** — it has the open bug, what is ruled out, and the
order to work in.

### As of 19 Sep, late

- GX10 `runs/yam_g1`: 2,104 demos / 197k frames, ACT training, ~253 samples/s, evals every 2000
  steps against held-out backgrounds. This dataset predates the wrist-camera remount and the
  gripper correction below, so treat its numbers as a pipeline check, not a final result.
- `yam_v1` (1,017 demos, Mac, not in repo) — superseded: too-heavy blur AND the wrong gripper.
- **Open:** the gripper. The menagerie model ships `crank_4310` (79 mm throw); the real arm has
  rack-and-pinion sliding jaws on a DM4310 = `linear_4310` (95 mm throw, matching I2RT's spec).
  Meshes are vendored at `third_party/i2rt/i2rt/robot_models/gripper/`. Until that swap lands,
  the sim's jaws are ~16 mm narrower than the real ones and the object-size cap is too strict.
