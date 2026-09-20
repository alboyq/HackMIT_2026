"""Aggregate the recorded feed episodes: rebuild the calibration from EVERY food-to-face touch, and write
a one-row-per-episode CSV. Usage: aggregate_feed_data.py <record_dir> [--write-calibration]"""
import csv
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

D = sys.argv[1]
WRITE = "--write-calibration" in sys.argv
FISHEYE_H_FOV = 150.0 * 1280.0 / np.hypot(1280.0, 720.0)
metas = []
for f in sorted(glob.glob(os.path.join(D, "*.json"))):
    with open(f) as fh:
        ep = json.load(fh)
    m = ep["meta"]
    m["n_steps"] = len(ep["steps"])
    m["file"] = os.path.basename(f)
    metas.append(m)
print(f"{len(metas)} recorded episodes in {D}")
print("outcomes:", dict(Counter(m["result"].split(" (")[0][:48] for m in metas)))

cols = ["file", "seed", "episode", "obj", "object_width_m", "head_scale", "calibrate", "sensors_only", "result", "picked",
        "dropped", "drop_phase", "arm_hit", "food_touch", "touch_size", "touch_speed", "stop_size", "near_vmax", "vmax",
        "amax", "ahead", "side", "n_steps"]
with open(os.path.join(D, "episodes.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(metas)

t = [m for m in metas if m.get("calibrate") and m.get("food_touch") and m.get("touch_size")]
if t:
    s = np.array([m["touch_size"] for m in t])
    hs = np.array([m["head_scale"] for m in t])
    print(f"\nFOOD-TO-FACE TOUCHES: {len(s)}")
    print(f"  face width at touch: median {np.median(s):.1f} deg, mean {s.mean():.1f}, sd {s.std():.1f}, "
          f"p05 {np.percentile(s, 5):.1f}, p10 {np.percentile(s, 10):.1f}, min {s.min():.1f}, max {s.max():.1f}")
    print(f"  = {100*np.median(s)/FISHEYE_H_FOV:.0f}% of image width on the 150-deg lens "
          f"(p05-max: {100*np.percentile(s, 5)/FISHEYE_H_FOV:.0f}-{100*s.max()/FISHEYE_H_FOV:.0f}%)")
    print(f"  contact closing speed: max {max(m['touch_speed'] for m in t):.3f} m/s")
    for name in ("apple", "mug", "block"):
        v = [m["touch_size"] for m in t if m["obj"] == name]
        if v:
            print(f"  {name:6s} n={len(v):3d}  median {np.median(v):.1f} deg")
    k = np.polyfit(hs, s, 1)
    print(f"  head size effect: {k[0]:+.1f} deg per 100% head scale -> {k[0]*0.08:+.1f} deg across the +-8% range")
    if WRITE:
        out = "arm/rl/configs/feed_calibration.json"
        json.dump({"contact_size_deg_median": float(np.median(s)), "contact_size_deg_mean": float(s.mean()),
                   "contact_size_deg_sd": float(s.std()), "contact_size_deg_p05": float(np.percentile(s, 5)),
                   "contact_size_deg_p10": float(np.percentile(s, 10)), "contact_size_deg_min": float(s.min()),
                   "contact_size_deg_max": float(s.max()), "n_touches": int(len(s)),
                   "pct_of_image_width_150deg_fisheye": float(100 * np.median(s) / FISHEYE_H_FOV),
                   "assumed_face_width_m": 0.145, "source": os.path.basename(os.path.normpath(D)),
                   "note": "angular face width when the FOOD first touched the face. Sim, contact detection on, "
                           "decisions from wrist camera + motor feedback only, head size +-8%, ten skin tones."},
                  open(out, "w"), indent=2)
        print("  calibration written to", out)
