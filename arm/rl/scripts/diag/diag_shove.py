"""WHAT shoves the object? State of the claw at first contact and at the moment the 4 cm gate breaks."""
import glob, os, re, sys
from collections import Counter
import numpy as np
os.environ.setdefault("MUJOCO_GL", "egl")
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from hackmit_rl.config import load_config
from hackmit_rl.envs import OpenYAMFeedEnv
RUN, N = sys.argv[1], int(sys.argv[2])
cks = sorted(glob.glob(f"{RUN}/checkpoints/ppo_grasp_*_steps.zip"), key=lambda p: int(re.search(r"_(\d+)_steps", p).group(1)))
cks = [c for c in cks if os.path.getsize(c) > 0]
CK = cks[-1]; step = re.search(r"_(\d+)_steps", CK).group(1)
VN = glob.glob(f"{RUN}/checkpoints/*vecnormalize_{step}_steps.pkl")[0]
cfg = load_config("arm/rl/configs/feed.yaml"); cfg["env"]["stage"] = "grasp"
raw = DummyVecEnv([lambda: OpenYAMFeedEnv(cfg)]); e = raw.envs[0]
env = VecNormalize.load(VN, raw); env.training, env.norm_reward = False, False
m = PPO.load(CK, device="cpu"); d = e.data
print("ckpt", CK)

def claw_state():
    R = d.site_xmat[e.tool.site_id].reshape(3, 3)
    t, j = R @ e.tool.tool_local, R @ e.tool.jaw_local
    pdl = np.cross(t, j)
    v = e.scene.object_pos(e.name) - e._tcp()
    gid = e.obj_gid[e.name]; parts = set()
    for i in range(d.ncon):
        c = d.contact[i]; pair = {int(c.geom1), int(c.geom2)}
        if gid in pair:
            for g in pair - {gid}:
                if g in e.pad_geoms: parts.add("pad")
                elif g in e.finger_geoms: parts.add("finger-body")
                elif g in e.arm_geoms: parts.add("wrist/arm")
    open_w = float(np.linalg.norm(d.geom_xpos[sorted(e.left_pads)].mean(0) - d.geom_xpos[sorted(e.right_pads)].mean(0)))
    return dict(e_jaw=1000*float(v @ j), e_pad=1000*float(v @ pdl), parts=parts, open=1000*open_w,
                clear=1000*(open_w - e.width[e.name]) / 2)

rows, parts_at_first = [], Counter()
for ep in range(N):
    obs = env.reset(); name = e.name; first = None; at_gate = None; info = {}
    for i in range(300):
        a, _ = m.predict(obs, deterministic=True)
        pre = claw_state()
        obs, r, done, infos = env.step(a); info = infos[0]
        if done[0]:
            break
        cs = claw_state()
        if first is None and cs["parts"]:
            first = dict(cs, t=i, seat=info["seat_frac"], grip=info["grip_fraction"], tilt=info["tilt_deg"], sq=info["off_square_deg"], phase=info["phase"])
    rows.append((name, bool(info.get("success")), bool(info.get("failed")), first, 1000*e.width[name]))
ok = [r for r in rows if r[1]]; bad = [r for r in rows if not r[1]]
print(f"success {len(ok)}/{N}")
def summarise(tag, rs):
    rs = [r for r in rs if r[3]]
    if not rs: return
    f = [r[3] for r in rs]
    print(f"\n{tag} (n={len(rs)}) -- claw state at FIRST contact with the object:")
    print(f"  off-centre along jaw axis : mean |{np.mean([abs(x['e_jaw']) for x in f]):.1f}| mm   across the claw: mean |{np.mean([abs(x['e_pad']) for x in f]):.1f}| mm")
    print(f"  side clearance available  : mean {np.mean([x['clear'] for x in f]):.1f} mm per side   (jaw opening {np.mean([x['open'] for x in f]):.0f} mm, grip cmd {np.mean([x['grip'] for x in f]):.2f})")
    print(f"  seated depth at contact   : mean {np.mean([x['seat'] for x in f]):.2f}   tilt {np.mean([x['tilt'] for x in f]):.1f} deg   off-square {np.mean([x['sq'] for x in f]):.1f} deg   phase {Counter(x['phase'] for x in f)}")
    print(f"  what touched first        : {Counter('+'.join(sorted(x['parts'])) for x in f)}")
    print(f"  off-centre > clearance?   : {sum(abs(x['e_jaw']) > x['clear'] for x in f)}/{len(f)}")
summarise("SUCCESSES", ok); summarise("FAILURES", bad)
print("\nfailures by object:", Counter(r[0] for r in bad), " of ", Counter(r[0] for r in rows))
