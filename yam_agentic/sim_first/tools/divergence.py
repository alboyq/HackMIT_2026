"""Where does a closed-loop rollout part company with the expert it copied?

    python tools/divergence.py <ckpt_dir | mlp.pt> [seed]

Runs one seed twice -- once replaying the expert, once with the policy in the loop, re-inferring
every tick -- and prints, per tick: the joint gap to the expert, and what the policy WOULD have
said if fed the expert's own state (teacher forced). Set the same YAM_* scene env the run trained
under, or the comparison is against a distribution the policy never saw.

Measured 2026-09-20 (prompt-only MLP, 0.75 deg open loop):
    tick  1: gap 0.57 deg, teacher-forced 1.34      tick 28: gap 9.18 deg, teacher-forced 0.10
The policy is near-perfect on states the expert visits and cannot stay on them: compounding
error / covariate shift, not a capacity or a perception problem. ../VLA_VERDICT.md section 4.
"""
import os
import sys
import _paths; _paths.add_sim_to_path()
import numpy as np
import torch
from yam_data import DataScene, GRIP_CTRL_OPEN
from yam_expert import Expert

CK = sys.argv[1]
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
BG = os.environ.get("YAM_BG")
DEG = 180.0 / np.pi


def load_policy(path):
    """Returns predict(state13, scene, wrist) -> first action, for either policy type."""
    if path.endswith(".pt"):
        import torch.nn as nn
        b = torch.load(path, map_location="cpu", weights_only=False)
        net = nn.Sequential(nn.Linear(13, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(),
                            nn.Linear(512, 512), nn.GELU(), nn.Linear(512, b["chunk"] * b["adim"]))
        net.load_state_dict(b["sd"]); net.eval()

        def predict(s13, scene, wrist):
            x = torch.tensor(((s13 - b["mu"]) / b["std"])[None].astype(np.float32))
            with torch.no_grad():
                y = net(x)[0].numpy() * b["ysd"] + b["ymu"]
            return y.reshape(b["chunk"], b["adim"])[0].astype(np.float32)
        return predict

    from lerobot.policies.act.modeling_act import ACTPolicy
    pol = ACTPolicy.from_pretrained(path).to("cpu").eval()

    def predict(s13, scene, wrist, _full=[None]):
        batch = {"observation.images.scene": torch.from_numpy(scene.copy()).permute(2, 0, 1).float().div(255)[None],
                 "observation.images.wrist": torch.from_numpy(wrist.copy()).permute(2, 0, 1).float().div(255)[None],
                 "observation.state": torch.from_numpy(_full[0]).unsqueeze(0)}
        batch["observation.images"] = [batch[k] for k in pol.config.image_features]
        with torch.no_grad():
            return pol.model(batch)[0][0, 0].numpy().astype(np.float32)
    predict.needs_full_state = True
    return predict


def expert_roll(sc, ex, seed):
    target = sc.randomize(seed)
    hold = np.concatenate([sc.data.ctrl[:6], [1.0]])
    for _ in range(3):
        sc.tick(hold, hold)
    assert sc.capture_box() is not None, "target hidden at selection time; pick another seed"
    plan = ex.plan_pick(target)
    assert plan, f"pick plan failed: {plan.why}"
    dt = sc.model.opt.timestep
    Q, A, prev = [], [], hold.copy()

    def play(traj):
        nonlocal prev
        for t in range(int(np.ceil(len(traj) / sc.n_sub))):
            c = np.array(traj[min((t + 1) * sc.n_sub, len(traj)) - 1], float)
            act = np.concatenate([c[:6], [c[6] / GRIP_CTRL_OPEN]]).astype(np.float32)
            _, _, st = sc.observe()
            Q.append(st[:7].copy()); A.append(act.copy())
            sc.tick(act, prev); prev = act

    play(ex.trajectory(plan, dt))
    pres = ex.plan_present(sc.site("mouth"), q_from=sc.q_arm, grip=float(sc.data.ctrl[6]))
    if pres:
        play(ex.trajectory(pres, dt))
    return np.array(Q), np.array(A), target


predict = load_policy(CK)
sc = DataScene(bg_dir=BG, aug=True)
Qe, Ae, target = expert_roll(sc, Expert(sc), SEED)
print(f"seed {SEED} target={target}: expert episode {len(Qe)} ticks\n")

sc2 = DataScene(bg_dir=BG, aug=True)
sc2.randomize(SEED)
hold = np.concatenate([sc2.data.ctrl[:6], [1.0]]).astype(np.float32)
for _ in range(3):
    sc2.tick(hold, hold)
sc2.capture_box()
ex2 = Expert(sc2)
prev = hold.copy()
print(f"{'tick':>5} {'gap vs expert':>14} {'teacher-forced':>15} {'TCP->obj cm':>12} {'off-path':>9}")
for t in range(min(len(Qe), 90)):
    scene, wrist, st = sc2.observe()
    act = predict(st[:13], scene, wrist)
    gap = np.abs(st[:6] - Qe[t][:6]).mean() * DEG
    forced = predict(np.concatenate([Qe[t], sc2.box]), scene, wrist)
    tf = np.abs(forced[:6] - Ae[t][:6]).mean() * DEG
    tcp = ex2.ik.fk(sc2.q_arm, qpos_full=sc2.data.qpos)[0]
    off = np.abs(Qe[:, :6] - st[:6]).mean(1).min() * DEG
    if t < 6 or t % 6 == 0:
        print(f"{t:>5} {gap:>14.2f} {tf:>15.2f} "
              f"{np.linalg.norm(tcp - sc2.object_pos(target)) * 100:>12.1f} {off:>9.2f}")
    sc2.tick(act, prev); prev = act
