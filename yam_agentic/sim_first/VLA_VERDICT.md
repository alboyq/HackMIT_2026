# Should we switch to a VLA? — measured answer

**2026-09-20.** The question that prompted this: *ACT went nowhere; would a VLA or a
prompt-driven engine do better, and is it true that VLAs benefit less from domain
randomisation than ACT does?*

Short answer, in order of how much it should change what you do next:

1. **The ACT runs never tested ACT.** The policy is *worse than not moving the arm*, and the
   "5–7× better than baseline" claim in [`NEXT_AGENT.md`](NEXT_AGENT.md) came from comparing it
   to a baseline that no position-controlled arm would ever use. §1.
2. **The task is not hard to fit.** A 3-layer MLP on 13 numbers, with **no cameras at all**,
   trained for **17 seconds**, beats the ACT checkpoint 4× on open-loop accuracy. §2.
3. **But that MLP also scores 2 % closed-loop.** So open-loop MAE — the metric every decision in
   this project has been steered by — does not predict success. §3.
4. **The real failure is compounding error**, measured tick by tick: teacher-forced error
   0.05–0.3°, closed-loop divergence 9.5°. It reproduces on a model 100× smaller than ACT, so
   **it would reproduce on a VLA too.** §4.
5. Your recollection about domain randomisation is **roughly right but backwards in emphasis**,
   and in any case it is about a failure mode this project does not have yet. §6.

**Recommendation: do not fine-tune a VLA yet.** Not because VLAs are wrong for this — §7 argues
one is a reasonable second bet — but because the current pipeline cannot tell a good policy from
a bad one, and a VLA costs 20–70 GPU-hours on the GX10 to find that out. Fix the yardstick first
(§8, ~2 hours), and the VLA question answers itself.

---

## 1. The ACT policy is worse than doing nothing

The fit metric this project has steered by is first-action MAE in degrees, compared against a
baseline that predicts the **mean action over the whole dataset** (14–20°). That baseline is
meaningless here. The policy emits **absolute joint targets** at 10 Hz to a slow-moving arm, so
the honest baselines are the ones a controller could actually fall back on:

| predictor | first-action MAE | measured on |
|---|---|---|
| predict the dataset's mean action — *the repo's baseline* | **14.05°** | 165 k frames, `yam_v6` |
| **ACT `yam_v9/act/step_12000`** | **2.70°** | 320 frames of its **own training set** |
| **predict the current joint angles** (hold still) | **1.95°** | the same 320 frames |
| predict the current joint angles | 1.73° | 165 k frames, `yam_v6` |
| **constant-velocity extrapolation** | **1.20°** | 165 k frames, `yam_v6` |

ACT is **39 % worse than freezing the arm** and **2.4× worse than a two-line extrapolator**, on
data it was trained on. Every "×0.75 per doubling, no plateau" and "the ~3° floor is invariant"
conclusion was measured against the 14° straw man, which is why the floor looked like a floor:
it was never a learning curve, it was a model that had not yet reached zero.

Reproduce: `tools/baselines.py <demos>` and `tools/act_mae.py <ckpt> <demos>`.

## 2. Thirteen numbers and seventeen seconds beat it

Strip every camera away. Feed a 3×512 MLP only the 13 numbers ACT already gets in its state
vector — 6 joint angles, gripper, and the 6-number object box — and regress the next action:

| policy | inputs | train time | first-action MAE |
|---|---|---|---|
| ACT (51 M params) | 2 cameras + state(17) | ~16 min GPU, 12 k steps | 2.70–2.94° |
| **MLP (0.5 M params)** | **state(13), no images** | **17 s CPU/MPS** | **0.62°** held-out |

Held out **by episode**, so it is a generalisation number, not memorisation. Chunked to 20
actions on `yam_v9`'s data it gives 0.75° on the first action.

A model with strictly *less* information cannot beat one with more unless the bigger one has not
converged. It has not: at batch 8 × 12 000 steps ACT has drawn 96 000 samples against a 171 548-frame
dataset — **0.56 epochs**. The MLP saw 37. The original handoff's live hypothesis ("simply
optimiser steps") was correct and was abandoned one experiment too early.

Two contributing deviations from the published recipe, both worth fixing regardless:

- **No input or target normalisation.** `train_act_yam.py` feeds raw radians, raw 0–1 box
  coordinates and `[0,1]` images straight in. LeRobot 0.6 moved normalisation out of the policy
  and into a processor pipeline that this script never builds, so `ACTConfig`'s declared
  `MEAN_STD` normalisation is **silently never applied**. ACT's default `optimizer_lr = 1e-5` is
  tuned for unit-variance targets.
- **No ImageNet normalisation** on images going into an ImageNet-pretrained ResNet-18. Note that
  LeRobot's X-VLA pipeline lists `XVLAImageNetNormalizeProcessorStep` as *required*.

## 3. Open-loop accuracy does not predict success

The 0.62° MLP was run through **the same evaluator, the same held-out seeds and the same
held-out backgrounds** as every ACT checkpoint:

| policy | open-loop MAE | closed-loop success | picked |
|---|---|---|---|
| ACT `yam_v9` step_12000 | 2.70° | 0/56 | 0/56 |
| MLP, `na=10` | 0.75° | **1/56 (2 %)** | 8/56 (14 %) |
| MLP, `na=1` (re-infer every tick) | 0.75° | 0/56 | 0/56 |
| MLP, `na=1`, **training seeds + training backgrounds** | 0.75° | 0/55 | 0/55 |

A 3.6× better open-loop policy buys 2 % instead of 0 %. And it fails on the *training*
distribution, which rules out generalisation, backgrounds and camera realism as the cause —
exactly as the earlier ACT investigation found, for the same reason.

**Stop using MAE as the steering signal.** §8 says what to use instead.

## 4. What actually goes wrong, tick by tick

`tools/divergence.py` runs one seed twice — once replaying the expert, once with the policy in
the loop at `na=1` — and compares them:

| tick | closed-loop gap vs expert | error if fed the **expert's** state (teacher forced) |
|---|---|---|
| 0 | 0.00° | 2.15° |
| 1 | 0.57° | 1.34° |
| 4 | 2.61° | 0.90° |
| 8 | 4.89° | 0.27° |
| 28 | **9.18°** | 0.10° |
| 44 | 9.36° | **0.05°** |

The policy knows exactly what to do **from states the expert visits** — 0.05° is essentially
perfect — and cannot stay on them. This is textbook covariate shift: every demonstration is a
clean expert rollout, so a policy that drifts 1° off the path has never seen anything like where
it now is, and nothing pulls it back.

Watching the rollout in world coordinates shows the shape of the failure:

| tick | off-path | TCP → object | gripper cmd |
|---|---|---|---|
| 10 | 1.99° | 7.7 cm | open |
| 30 | 4.80° | **5.3 cm** | open |
| 50 | 12.28° | 7.2 cm | open |
| 60 | 14.44° | 8.0 cm | **closing** |
| 80 | 42.56° | 103.0 cm | closed on nothing |
| 140 | 108.31° | 45.1 cm | closed on nothing |

It **approaches correctly to about 5 cm, stalls, closes on empty air, and then flies apart.**
The last 5 cm is where it dies, and after the empty grasp the state is so far out of
distribution that the rollout is garbage — which is also why the filmstrips looked like the
policy had gone blind.

Two structural contributors, both fixable:

- **The prompt is open-loop by construction.** `self.box` is captured **once**, at selection
  time, and never updated. Within an episode the policy's entire non-proprioceptive input is a
  constant. Nothing in the state vector says "you are 5 cm short"; only the wrist camera can,
  and it is doing some work but not enough (§5).
- **Joint targets are never clamped.** `DataScene.to_ctrl` clips the gripper to `[0,1]` and
  passes the 6 joint targets through untouched. The 108°-off-path rollouts are the policy
  commanding poses nothing rejects. On the real arm the limiter catches this; in sim it turns
  every post-failure tick into noise that pollutes the eval.

## 5. ACT is not ignoring the cameras

Worth ruling out before blaming the encoder. Shuffle each input across a batch and measure how
far the prediction moves (`tools/attribution.py`):

| input shuffled | prediction moves |
|---|---|
| joints (state 0–6) | 10.54° |
| **scene camera image** | **7.28°** |
| **wrist camera image** | **3.79°** |
| box (state 7–12) | 0.90° |

Both cameras carry real weight — the scene image more than the box, which makes sense since it
contains the box. So "ACT can't see" is not the problem; "ACT hasn't converged and nothing
teaches it to recover" is.

## 6. On "VLAs don't benefit from domain randomisation as much as ACT"

Roughly right, for a real mechanistic reason, and it matters less here than it sounds.

**The mechanism.** ACT's ResNet-18 is 100 % trainable, so randomised pixels reach the
convolutional filters and the filters adapt. A VLA's vision tower is frozen or near-frozen —
SmolVLA is 450 M parameters with 22.2 % trainable — so randomised pixels have to survive a
frozen SigLIP encoder and be decoded downstream. [SEVO](https://arxiv.org/abs/2605.11114)
measures this on SO-101 arms (with, then without, its randomised collection protocol):

| | training env | novel env |
|---|---|---|
| ACT, randomised collection | **95 %** | **85 %** |
| SmolVLA, randomised collection | 83 % | 75 % |
| ACT, no randomisation | 75 % | 30 % |
| SmolVLA, no randomisation | 70 % | 35 % |

Randomisation buys ACT **+55 points** out of domain and SmolVLA **+40**, and ACT ends ahead on
both axes. SEVO's explanation is architectural, not a gradient measurement — treat it as a
mechanism, not a proof.

**The other half, which inverts the headline.**
[Colosseum V2](https://arxiv.org/abs/2605.27759) benchmarks ACT against π0.5 across 28 tasks:
π0.5 degrades far less under visual perturbation (7.7 % vs 28.6 % single-arm) — but from a much
lower base (340 vs 540 cumulative). *After* degrading, ACT is still ahead in absolute terms.
And it flips for language: ACT loses 0.3 % under instruction rephrasing, π0.5 loses 19–24 %.

So the honest statement is: **domain randomisation is what ACT uses instead of pretraining.**
The VLA arrives robust and plateaus lower; ACT starts brittle and can be trained past it. Neither
gives you precision.

**Why it does not decide anything today.** All of the above is about *unseen* visual conditions.
This policy fails on *training* seeds with *training* backgrounds (§3). Robustness is a problem
this project has not reached yet.

One randomisation finding *is* immediately actionable.
[Grounding Sim-to-Real Generalization](https://arxiv.org/abs/2603.22876) trains OpenVLA-OFT
sim-only and deploys zero-shot (0–64 % real, by task), and reports that **spatial randomisation
— camera pose, table height — matters substantially more than appearance**, and that
**frame-wise randomisation beats episode-wise by 3–13 points**. `yam_data.py` samples camera
effects **per episode**. That is a one-line change worth taking whenever training resumes.

## 7. The VLA landscape, September 2026, filtered by what this project can run

`lerobot 0.6.1` is **already installed** in `urlab_bridge/.venv` and already ships
`act, diffusion, pi0, pi05, pi0_fast, smolvla, groot, molmoact2, xvla, eo1, evo1, wall_x,
vla_jepa, fastwam, lingbot_va, multi_task_dit, rtc, vqbet, tdmpc`. Nothing needs to be built;
the question is only which one is worth the GPU-hours.

| model | size | new embodiment? | why / why not here |
|---|---|---|---|
| **X-VLA** | 0.9 B | **yes, by design** — `action_mode=auto` reads the action dim from the dataset; a new robot is a new set of soft prompts | Closest fit. Pretrained on 290 k episodes / 7 platforms; 93 % LIBERO; the docs' own example fine-tunes a bimanual SO-101 in **3 000 steps**. The one model here that treats "novel single-arm YAM" as the normal case. |
| SmolVLA | 450 M | yes, LeRobot-native | Cheapest real VLA. 20 k steps ≈ **4 h on an A100**. Lower ceiling than ACT in-domain (§6). |
| π0.5 | 3 B | yes | Strong; the Colosseum V2 reference point. 94.7 ms/action on DGX Spark with TensorRT FP8 — fast enough at 20 Hz with chunking. |
| MolmoAct 2 | — | **no** | Out of the box only SO-100, **bimanual** YAM and Franka; Ai2 say a different platform "requires additional training". 180 ms/action is an **H100** number. Still the most on-target published system; still not a single-arm YAM. |
| GR00T N1.7 | 3 B | yes | NVIDIA's whole pitch is synthetic post-training data. 63.6 ms mean latency on DGX Spark, the fastest measured. |
| π*0.6 / RECAP | — | — | Not open. But conceptually the right answer to §4: it learns **from its own failures** via advantage-conditioned RL, which is precisely the corrective data this dataset lacks. |

**The compute reality, and it is the binding constraint.** The GX10's GB10 has ~273 GB/s of
memory bandwidth against an A100's ~2 TB/s, and a measured fine-tuning benchmark puts it at
**~5× slower wall-clock** than an A100 40 GB. So:

| job | A100 | **GX10 (measured 5× rule)** |
|---|---|---|
| SmolVLA, 20 k steps | ~4 h | **~20 h** |
| OpenVLA-OFT LoRA, one task | 10–15 h | **50–75 h** |
| X-VLA, 3 k steps, 0.9 B | ~1–2 h | **~5–10 h** |

Note the M4 Max has **~546 GB/s** — roughly **2× the GX10's bandwidth**. For a bandwidth-bound
fine-tune the laptop may genuinely be the faster machine, MPS kernel maturity permitting. That is
worth ten minutes of measurement before committing a day of GX10 time to the wrong box.

**And the cheapest VLA-family option involves no fine-tuning at all.** What §4 shows is that the
*planner* is fine — `tools/expert_ceiling.py` scores the expert 23/23 — and the learned
controller is what fails. A "prompt-driven engine" that uses a VLM for **perception and
planning** and leaves control to the planner skips the entire failure mode:

- Gemini Robotics-ER 2 (shipped 30 Jul 2026) or Molmo 2-ER (63.8 avg over 13 embodied-reasoning
  benchmarks, ahead of GPT-5 and Gemini 2.5 Pro) emit 2D points and boxes zero-shot.
- Those are exactly the `{label,bbox,image_width,image_height,confidence}` the team's UDP
  contract on `:8765` already carries.
- Table-plane back-projection + the existing IK planner does the rest.

This is [`GAME_PLAN.md`](../GAME_PLAN.md) §7, the "fallback". On today's evidence it is not the
fallback — it is the only route with a measured 100 % component in it.

## 8. The tests, in the order that kills the most uncertainty per hour

Each one is written to produce a number that changes what you do next. Times are wall-clock on
the M4 Max unless stated.

### T0 — Fix the yardstick. ~30 min. Do this before anything else.
Nothing below is interpretable until the evaluator can rank two policies.
1. Add the **hold-still** and **constant-velocity** baselines to every fit report
   (`tools/baselines.py` already computes them). A policy above 1.73° is not a policy.
2. **Clamp** policy joint targets to the model's joint ranges and to a max per-tick delta in
   `DataScene.to_ctrl`. Mirrors the real limiter and stops post-failure ticks polluting evals.
3. Report **closed-loop success**, not MAE, as the steering signal. `eval_mlp_yam.py` runs
   56 episodes in 26 s on 14 workers — there is no excuse for steering by a proxy.

**Gate:** the expert scores ~100 % and the hold-still policy scores 0 % under the same harness.

### T1 — Give ACT the recipe it was supposed to have. ~2 h. Highest value per hour.
The cheapest explanation of §1–§2 is that ACT never converged.
- Normalise state, action and images (mean/std from the dataset; ImageNet stats for the images).
- Raise the LR off `1e-5` now that targets are unit-variance, or train to **≥ 20 epochs**
  (≈ 430 k samples at batch 8, not 96 k).
- Keep the chunk and rate as they are; §1 already showed they are not the problem.

**Gate: does ACT beat 1.73°?** If no, the implementation is broken and no VLA will fix it. If
yes, you finally have a working ACT and the closed-loop question (T2) is the real one.

### T2 — Corrective data. ~1 h per iteration. This is the one that decides the project.
§4 says the dataset has no recovery behaviour. `YAM_DART_DEG` is now implemented in
`gen_demos_yam.py`: it executes a **disturbed** action while labelling the plan's own
**absolute** joint target — which stays correct from wherever the disturbance puts the arm, so
[DART](https://arxiv.org/abs/1703.09327) costs no replanning here. DART reports **+62 %** over
behaviour cloning on grasping in clutter.

**Already run, and it did not work yet:** σ = 1.5°, 256 episodes → open-loop MAE 2.91°
(4× *worse*, as DART predicts) and closed-loop still 2 %. Two reasons not to conclude anything
from that: DART prescribes matching the noise to the *policy's own* error, which §4 measures at
**~9.5°**, so 1.5° is ~6× too small; and 256 episodes is 7× less data than the baseline run.
Sweep σ ∈ {3°, 6°, 9°} at ~700 episodes. Episode yield falls (64 % at σ=1.5°) — that is expected
and is why DART iterates.

**Gate: does closed-loop success clear 20 %?** If yes, the sim-first plan is alive and ACT is the
cheap way to finish it. If no after the sweep, the open-loop single-shot formulation is the
problem, not the model — go to T3.

### T3 — Close the loop on the last 5 cm. ~2 h.
§4's rollout stalls at 5 cm because nothing in its input says "you are short".
- **Re-capture the box every tick** instead of once per episode. The real detector publishes at
  ≥ 5 Hz on `:8765`, so a live box is *more* faithful to deployment, not less.
- Add the **TCP-relative** object box, or crop the wrist image around the predicted grasp — give
  the policy an error signal rather than an absolute target.
- Switch the action space to **Cartesian deltas** (handoff idea #1). `ToolFrame` gives the TCP
  and `ArmIK` inverts it, so both ends already exist.

### T4 — Only now, the VLA. ~5–10 h GX10, or ~1–2 h on rented A100/H100.
Run it **only if T1–T3 leave a gap a pretrained visual prior could plausibly close** — i.e. ACT
converges, recovers, and still cannot do the last 5 cm.
- **X-VLA first**, `--policy.path=lerobot/xvla-base --policy.action_mode=auto`, ~3 k steps. It is
  the only model in the list that treats a novel single-arm embodiment as routine, and it is the
  cheapest real test of the hypothesis "a pretrained encoder sees the last 5 cm that a
  from-scratch ResNet-18 cannot".
- Benchmark one training step on the **M4 Max vs the GX10** before picking the machine.
- Pick `domain_id` from a single-arm entry (Bridge 0, RT1 1, libero 3, widowx-air 4).

**Gate:** X-VLA beats the fixed ACT from T1 on closed-loop success. If it does not, stop — the
evidence in §6–§7 says it will not get there on sim-only data on an unseen arm.

### T5 — The route with no training in it. Independent of all the above; run in parallel.
Gemini Robotics-ER 2 / Molmo 2-ER → 2D points → table-plane → the existing IK planner. No
fine-tuning, no sim2real gap on the action side, and the only component in this project with a
measured 100 %. Insurance for the demo whatever T1–T4 do.

## 9. What is now in the repo

| file | what it answers |
|---|---|
| `tools/baselines.py` | hold-still / constant-velocity / mean-action baselines, and the prompt-only MLP |
| `tools/act_mae.py` | a checkpoint's first-action MAE against those baselines, same frames |
| `tools/attribution.py` | which inputs a trained policy actually uses |
| `tools/divergence.py` | teacher-forced vs closed-loop error, tick by tick |
| `tools/prompt_manifold.py` | label ambiguity, local Lipschitz, intrinsic dimension, data-scaling |
| `sim/train_mlp_yam.py`, `sim/eval_mlp_yam.py` | the prompt-only baseline policy, end to end |
| `sim/gen_demos_yam.py` | `YAM_DART_DEG` noise injection |

### A result worth keeping even though it changed nothing

`tools/prompt_manifold.py` asked whether the 2.9° floor was *irreducible* — whether the expert
labels two indistinguishable prompts differently. **They do not:** extrapolating 1-NN label
disagreement to zero prompt distance gives **−0.29° ≈ 0**. The mapping is a clean function.

It also measures the local Lipschitz constant (**3.88° per sd-unit**) and the intrinsic dimension
of the prompt manifold (**6.4**), which together price the "just collect orders more data" option
in the old handoff:

| target | data multiple | episodes | generation time | storage |
|---|---|---|---|---|
| 2.0° | 54× | 84 k | ~12 h | ~1.8 TB |
| **1.5°** | **339×** | **530 k** | **~80 h** | **~12 TB** |

That option is dead, and it was dead for a reason unrelated to the bug in §1. Worth knowing
before anyone proposes it again.

## Sources

[SEVO](https://arxiv.org/abs/2605.11114) ·
[Colosseum V2](https://arxiv.org/abs/2605.27759) ·
[Grounding Sim-to-Real Generalization](https://arxiv.org/abs/2603.22876) ·
[DART](https://arxiv.org/abs/1703.09327) ·
[X-VLA](https://arxiv.org/abs/2510.10274) · [X-VLA in LeRobot](https://huggingface.co/docs/lerobot/xvla) ·
[LeRobot v0.6.0](https://huggingface.co/blog/lerobot-release-v060) ·
[OpenVLA-OFT](https://openvla-oft.github.io/) ·
[MolmoAct 2](https://allenai.org/blog/molmoact2) ·
[π*0.6 / RECAP](https://www.pi.website/download/pistar06.pdf) ·
[DGX Spark fine-tuning throughput](https://kaitchup.substack.com/p/dgx-spark-use-it-for-fine-tuning)
