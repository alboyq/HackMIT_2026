"""Re-run one seed and record CLEAN video (no text) of chosen episodes, for the demo matrices.
Episodes are reproducible from (SEED, index), so failures found by the data-only search can be filmed.
usage: SEED=<s> record_tiles.py <out_dir> <ep,ep,ep>"""
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("YAM_CAM_RES", "320")
OUT = sys.argv[1]
WANT = sorted({int(x) for x in sys.argv[2].split(",")})
sys.argv = [sys.argv[0]]

import cv2  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hybrid_feed as hf  # noqa: E402
import watch_feed as wf  # noqa: E402

os.makedirs(OUT, exist_ok=True)
SEED = os.environ.get("SEED", "0")
st = {"writer": None, "phases": [], "ep": None}
PHASE = {"reach": "POSITION", "pinch": "PINCH", "lift": "LIFT", "retreat": "PULL BACK", "approach": "APPROACH", "hold": "HOLD"}


def write(e, phase):
    if st["writer"] is None:
        st["writer"] = cv2.VideoWriter(os.path.join(OUT, f"s{SEED}_ep{st['ep']:04d}.mp4"),
                                       cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480))
    st["writer"].write(wf.frame(e, []))
    st["phases"].append(PHASE.get(phase, phase))


def hook(e, phase, size, o, info, ctx):
    ep = ctx["episode"] - 1                      # 0-based, same index the data files use
    nxt = ep + 1
    # the frozen reach of the NEXT episode runs inside reset(): film it only if that episode is wanted
    e.prior_hook = (lambda en: (st.__setitem__("ep", nxt), write(en, "reach"))) if (phase == "done" and nxt in WANT) else \
        (e.prior_hook if phase != "done" else None)
    if ep not in WANT:
        return
    if phase == "done":
        if st["writer"] is not None:
            st["writer"].release()
        meta = {k: (float(v) if hasattr(v, "dtype") else v) for k, v in o.items()}
        meta.update(seed=int(SEED), episode=ep, phases=st["phases"], head_scale=float(e.head_scale))
        json.dump(meta, open(os.path.join(OUT, f"s{SEED}_ep{ep:04d}.json"), "w"))
        print(f"[tile] s{SEED} ep{ep}: {o['result']}  {len(st['phases'])} frames", flush=True)
        st["writer"], st["phases"] = None, []
        return
    st["ep"] = ep
    write(e, phase)


hf.STEP_HOOK = hook
hf.main(n_episodes=max(WANT) + 1, report=False)
