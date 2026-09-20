"""Turn a segmentation mask into the metric numbers a grasp needs.

Deliberately knows nothing about where the mask came from -- SAM 3.1, SAM 2.1, a colour
threshold. It takes a boolean array and the RGB-D frame and returns a profile. That keeps
mask production and metric extraction on opposite sides of one boolean array.

Why a profile rather than a single point. Different questions want different statistics of
the same masked point cloud:

  * a TOP-DOWN grasp contacts the object's top surface, not its centroid. Aiming at the
    centroid drives the gripper half an object-height too deep.
  * jaw opening wants the object's minor horizontal extent, not its depth spread.
  * a bimodal depth histogram means the mask leaked onto the background, which is worth
    knowing before acting on any of the other numbers.

Resolution, measured on this rig. The LiDAR grid is 256x192, so at 0.9 m one depth pixel is
~4.9 mm and a 15 mm marker is ~3 px across -- too coarse for a believable extent. The RGB frame
is 7.5x finer (0.74 mm/px at 1 m). So when an RGB-resolution mask is supplied, EXTENT is taken
from it at full resolution while RANGE comes from the depth grid, which only needs a handful of
good returns. Depth for scale, RGB for geometry.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CONF_HIGH = 2
CONF_MEDIUM = 1
# A small object is only a few depth pixels across (a 15 mm marker at 0.9 m is ~3 px). Depth
# supplies RANGE, which needs a handful of returns; an RGB mask supplies EXTENT. A higher floor
# would reject exactly the case rgb_mask exists to serve.
MIN_DEPTH_POINTS = 5


@dataclass(frozen=True)
class ObjectProfile:
    """Metric description of one masked object, in the camera frame (z forward, metres)."""
    n_points: int
    valid_fraction: float
    centroid: np.ndarray        # robust centre of the visible surface
    top: np.ndarray             # nearest-surface point: aim a top-down grasp here
    extent: np.ndarray          # oriented box side lengths, descending
    axes: np.ndarray            # 3x3, rows are the oriented box axes (major first)
    jaw_width: float            # smallest horizontal extent -> how far to open the jaws
    depth_p10: float
    depth_p50: float
    depth_p90: float
    bimodal: bool               # True => the mask probably leaked onto the background
    confidence: float           # 0..1, feeds Target3D.confidence
    extent_source: str          # "rgb" (fine) or "depth" (coarse)

    def describe(self) -> str:
        e = self.extent * 1000
        return (f"{self.n_points} px, centre {np.round(self.centroid, 3)} m, "
                f"top {np.round(self.top, 3)} m, "
                f"box {e[0]:.0f}x{e[1]:.0f}x{e[2]:.0f} mm ({self.extent_source}), "
                f"jaw {self.jaw_width * 1000:.0f} mm, conf {self.confidence:.2f}"
                + (" [BIMODAL: mask may have leaked]" if self.bimodal else ""))


def backproject(rows, cols, depths, K) -> np.ndarray:
    """Pixel coordinates + depths -> (N, 3) camera-frame points, in metres."""
    x = (np.asarray(cols, float) - K["cx"]) / K["fx"] * depths
    y = (np.asarray(rows, float) - K["cy"]) / K["fy"] * depths
    return np.stack([x, y, np.asarray(depths, float)], axis=1)


def masked_points(depth, conf, mask, K, min_conf=CONF_HIGH) -> tuple[np.ndarray, float]:
    """Camera-frame points under `mask`, keeping only confident returns.

    Returns the points and the fraction of masked pixels that survived -- a low fraction is
    itself the signal that the surface is dark, shiny or transparent.
    """
    mask = np.asarray(mask, bool)
    if mask.shape != depth.shape:
        raise ValueError(f"mask {mask.shape} does not match depth {depth.shape}; "
                         "downsample an RGB-resolution mask with mask_to_depth_grid first")
    good = mask & (depth > 0) & np.isfinite(depth)
    if conf is not None:
        good &= conf >= min_conf
    rows, cols = np.nonzero(good)
    frac = float(good.sum()) / max(int(mask.sum()), 1)
    return backproject(rows, cols, depth[good], K), frac


def mask_to_depth_grid(mask, depth_shape) -> np.ndarray:
    """Nearest-neighbour downsample an RGB-resolution mask onto the depth grid."""
    mask = np.asarray(mask, bool)
    dh, dw = depth_shape
    rows = (np.arange(dh) * mask.shape[0] // dh).clip(0, mask.shape[0] - 1)
    cols = (np.arange(dw) * mask.shape[1] // dw).clip(0, mask.shape[1] - 1)
    return mask[np.ix_(rows, cols)]


def _oriented_box(pts):
    """PCA oriented box. Returns (axes rows major-first, side lengths)."""
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    proj = (pts - c) @ vt.T
    return vt, np.ptp(proj, axis=0)


def _depth_split(d, gap_m=0.04):
    """Find the empty band separating two surfaces, if there is one.

    Returns the depth to cut at, or None when the points look like one surface. A mask that
    leaks does so onto the table BEHIND the object, so the near cluster is the object. That
    assumes nothing occludes it -- a mask leaking onto the gripper in FRONT would break the
    assumption, which is why bimodality is reported as well as acted on.
    """
    if len(d) < 30:
        return None
    x = np.sort(d)
    if x[-1] - x[0] < gap_m:
        return None
    lo = max(1, len(x) // 20)          # sparse tails always leave large gaps; ignore them
    core = x[lo: len(x) - lo]
    if len(core) < 3:
        return None
    gaps = np.diff(core)
    i = int(np.argmax(gaps))
    return float((core[i] + core[i + 1]) / 2) if gaps[i] > gap_m else None


def profile(depth, conf, mask, K_depth, *, rgb_mask=None, K_rgb=None,
            min_conf=CONF_HIGH) -> ObjectProfile | None:
    """Build an ObjectProfile from a depth-grid mask, optionally refining extent from RGB.

    `mask` must be on the depth grid. Pass `rgb_mask` and `K_rgb` to take the extent from the
    full-resolution colour frame, which is 7.5x finer and the only way a small object gets a
    believable size.
    """
    pts, frac = masked_points(depth, conf, mask, K_depth, min_conf)
    if len(pts) < MIN_DEPTH_POINTS:
        return None

    # Two surfaces => keep the near one. Trimming a percentile is not enough: a leak can be the
    # MAJORITY of the mask (a bare-table patch is easily larger than the object), and then any
    # percentile-based centre follows the leak rather than the object.
    cut = _depth_split(pts[:, 2])
    bimodal = cut is not None
    if bimodal:
        near = pts[:, 2] <= cut
        if near.sum() >= MIN_DEPTH_POINTS:
            pts = pts[near]

    d = pts[:, 2]
    p10, p50, p90 = (float(np.percentile(d, q)) for q in (10, 50, 90))
    centroid = pts[d <= np.percentile(d, 90)].mean(0)

    # The top surface is the nearest decile, not the single nearest pixel (which is noise).
    top = pts[d <= p10].mean(0) if (d <= p10).any() else centroid

    source = "depth"
    if rgb_mask is not None and K_rgb is not None:
        rows, cols = np.nonzero(np.asarray(rgb_mask, bool))
        if len(rows) >= 10:
            rgb_pts = backproject(rows, cols, np.full(len(rows), p50), K_rgb)
            axes, extent = _oriented_box(rgb_pts)
            extent[-1] = max(float(p90 - p10), float(extent[-1]))   # thickness from depth
            source = "rgb"
        else:
            axes, extent = _oriented_box(pts)
    else:
        axes, extent = _oriented_box(pts)

    order = np.argsort(extent)[::-1]
    axes, extent = axes[order], extent[order]

    confidence = float(np.clip(frac, 0, 1) * min(1.0, len(pts) / 80.0) * (0.4 if bimodal else 1.0))

    return ObjectProfile(
        n_points=len(pts), valid_fraction=frac,
        centroid=centroid, top=top, extent=extent, axes=axes,
        jaw_width=float(min(extent[0], extent[1])),
        depth_p10=p10, depth_p50=p50, depth_p90=p90,
        bimodal=bimodal, confidence=confidence, extent_source=source)
