"""Capture one FK-check pose. READ-ONLY: motors stay disabled (zero MIT frames,
enable() is never called). Stores the raw encoder reading, the 2*pi lap each joint
is on (worked out from the hard stops measured by hand on 2026-09-19), and the
resulting model-frame joint angles. It does NOT compute a tool position - that is
done later with the corrected linear_4310 model.

    python fk_capture.py <label>          e.g.  python fk_capture.py 1
"""
import json, os, sys, time
import numpy as np
from openyam import CanBus, OpenYAMArm

OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "poses.json")
TAU  = 2 * np.pi
STOPS = {1:(-4.0309,1.6047), 2:(0.0044,3.6952), 3:(0.0101,3.9569),
         4:(-1.6726,1.6577), 5:(-1.5761,1.5791), 6:(-0.7822,3.3892)}
OFF  = {1:-1.4313, 2:0.0044, 3:0.0101, 4:-0.0074, 5:0.0015, 6:1.3035}
MARGIN, EPS = 0.15, 1e-9

label = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%H%M%S")

with CanBus() as bus:
    arm = OpenYAMArm(bus=bus)
    Q = np.array([[s.position if s else np.nan for s in arm.read_states()] for _ in range(12)])
enc  = np.nanmedian(Q, axis=0)[:6]
span = (np.nanmax(Q, axis=0) - np.nanmin(Q, axis=0))[:6]
if np.isnan(enc).any():
    print("!! a motor gave no reading - not saved"); sys.exit(1)
if span.max() > 0.005:
    print(f"!! arm still moving (drift {span.max():.4f} rad, limit 0.005) - hold it still and retry"); sys.exit(1)

laps, unwrapped, q_model, problems = [], [], [], []
for j in range(1, 7):
    lo, hi = STOPS[j]
    fits = [k for k in range(-3, 4) if lo-MARGIN-EPS <= enc[j-1] + TAU*k <= hi+MARGIN+EPS]
    if len(fits) != 1:
        problems.append(f"J{j}: {len(fits)} laps fit (reading {enc[j-1]:+.4f}, stops [{lo:+.3f},{hi:+.3f}])")
        laps.append(None); unwrapped.append(None); q_model.append(None); continue
    k = fits[0]; eu = enc[j-1] + TAU*k
    laps.append(k); unwrapped.append(float(eu)); q_model.append(float(eu - OFF[j]))   # every sign is +1
if problems:
    print("!! REFUSED - not saved:"); [print("   ", p) for p in problems]; sys.exit(1)

rec = {"label": label, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
       "drift_rad": float(span.max()), "encoder_raw": enc.tolist(), "lap": laps, "encoder_unwrapped": unwrapped, "q_model": q_model}
data = json.load(open(OUT)) if os.path.exists(OUT) else {"poses": []}
data["poses"] = [p for p in data["poses"] if p["label"] != label] + [rec]
json.dump(data, open(OUT, "w"), indent=2)

print(f"[pose {label}] saved  ({len(data['poses'])} pose(s) on file)")
print("  raw      :", np.round(enc, 4))
print(f"  drift    : {span.max():.4f} rad over the read (limit 0.005)")
print("  lap      :", laps)
print("  q_model  :", np.round(q_model, 4))
