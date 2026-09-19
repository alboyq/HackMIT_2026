"""Measure the YAM's IK-verified grasp envelope. This is the provenance of every reach number
in MEASUREMENTS.md — rerun it if the model, the gripper or the tool frame ever changes.

    python tools/measure_envelope.py

Why it exists: a first pass estimated the envelope by sampling 300k random joint configurations
and looking at where `grasp_site` ended up and which way its z-axis pointed. Both proxies are
wrong on this model (grasp_site is 4.4 cm from the real TCP, and its z-axis is 19 deg off the
tool axis), and the resulting numbers — 0.55 m top-down, 0.78 m side — were too optimistic.
Running the actual solver at a grid of poses is slower and correct.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim"))

import numpy as np

from yam_ik import ArmIK, ToolFrame
from yam_scene import YamScene

RADII = np.arange(0.15, 0.81, 0.05)
AZIMUTHS = np.radians([-40, -20, 0, 20, 40])


def band(ik, z, mode, tilt=0.0):
    hits = {}
    for r in RADII:
        n_ok = 0
        for a in AZIMUTHS:
            p = np.array([r * np.cos(a), r * np.sin(a), z])
            if mode == "top":                       # tool mostly down, `tilt` off vertical
                d = np.array([np.sin(tilt) * np.cos(a), np.sin(tilt) * np.sin(a), -np.cos(tilt)])
            else:                                   # tool mostly level, pointing outward
                d = np.array([np.cos(a) * np.cos(tilt), np.sin(a) * np.cos(tilt), -np.sin(tilt)])
            n_ok += bool(ik.solve_best(p, d)[1])
        hits[round(float(r), 2)] = n_ok
    return hits


def main():
    s = YamScene()
    ik = ArmIK(s.model, ToolFrame(s.model))
    cases = (
        ("tool straight down, z=4cm", 0.04, "top", 0.0),
        ("tool straight down, z=10cm", 0.10, "top", 0.0),
        ("tool tilted 30deg, z=6cm", 0.06, "top", np.radians(30)),
        ("tool level (side), z=6cm", 0.06, "side", 0.0),
        ("tool level, z=38cm (mouth)", 0.38, "side", 0.0),
    )
    t0 = time.time()
    print(f"{'case':32s} {'any azimuth':>18s} {'all 5 azimuths':>18s}")
    for tag, z, mode, tilt in cases:
        g = band(ik, z, mode, tilt)
        any_r = [r for r, c in g.items() if c > 0]
        all_r = [r for r, c in g.items() if c == len(AZIMUTHS)]
        fmt = lambda v: f"{min(v):.2f}-{max(v):.2f} m" if v else "none"
        print(f"{tag:32s} {fmt(any_r):>18s} {fmt(all_r):>18s}")
    print(f"\n({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
