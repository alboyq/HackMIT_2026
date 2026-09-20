from __future__ import annotations

import numpy as np
import pytest

from arm.depth_profile import (backproject, mask_to_depth_grid, masked_points,
                               profile)

# The rig's measured depth intrinsics (256x192 LiDAR at ~0.9 m).
K_DEPTH = {"fx": 180.6, "fy": 180.6, "cx": 128.8, "cy": 96.0}
K_RGB = {"fx": 1355.4, "fy": 1355.4, "cx": 960.0, "cy": 720.0}
SHAPE = (192, 256)


def scene(obj_depth=0.90, table_depth=1.00, radius_px=8, centre=(96, 128)):
    """A disc floating in front of a flat table, plus a full-confidence map."""
    depth = np.full(SHAPE, table_depth, np.float32)
    conf = np.full(SHAPE, 2, np.uint8)
    ys, xs = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    disc = (ys - centre[0]) ** 2 + (xs - centre[1]) ** 2 <= radius_px ** 2
    depth[disc] = obj_depth
    return depth, conf, disc


def test_centroid_and_top_sit_at_the_object_not_the_table():
    depth, conf, disc = scene()
    p = profile(depth, conf, disc, K_DEPTH)
    assert p is not None
    assert p.centroid[2] == pytest.approx(0.90, abs=1e-3)
    assert p.top[2] == pytest.approx(0.90, abs=1e-3)
    assert p.valid_fraction == pytest.approx(1.0)


def test_top_is_nearer_than_the_centroid_on_a_curved_surface():
    """The whole point of the top statistic: a dome's near face, not its middle."""
    depth, conf, disc = scene()
    ys, xs = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    r = np.sqrt((ys - 96.0) ** 2 + (xs - 128.0) ** 2)
    dome = disc & (r <= 8)
    depth[dome] = 0.90 + 0.02 * (r[dome] / 8.0) ** 2   # curves away from the camera
    p = profile(depth, conf, dome, K_DEPTH)
    assert p.top[2] < p.centroid[2]
    # A grasp aimed at the centroid would be this far into the object:
    assert (p.centroid[2] - p.top[2]) > 0.003


def test_low_confidence_returns_are_dropped_and_reported():
    depth, conf, disc = scene()
    ys, xs = np.nonzero(disc)
    conf[ys[: len(ys) // 2], xs[: len(xs) // 2]] = 0      # half the object unreliable
    p = profile(depth, conf, disc, K_DEPTH)
    assert p.valid_fraction < 0.75
    assert p.confidence < 1.0


def test_leaked_mask_is_flagged_bimodal_and_penalised():
    """A mask covering both object and table has two depth populations."""
    depth, conf, disc = scene(obj_depth=0.90, table_depth=1.00)
    leaked = disc.copy()
    ys, xs = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    leaked |= (np.abs(ys - 96) < 9) & (np.abs(xs - 150) < 9)   # a patch of bare table
    p = profile(depth, conf, leaked, K_DEPTH)
    assert p.bimodal
    clean = profile(depth, conf, disc, K_DEPTH)
    assert p.confidence < clean.confidence
    # The centroid must still favour the object, not average the two surfaces.
    assert p.centroid[2] < 0.95


def test_extent_from_rgb_beats_the_depth_grid_on_a_small_object():
    """A 15 mm marker is ~3 depth px across; the coarse grid cannot size it."""
    size_m, Z = 0.015, 0.90
    depth = np.full(SHAPE, Z, np.float32)
    conf = np.full(SHAPE, 2, np.uint8)
    half_d = size_m / 2 * K_DEPTH["fx"] / Z
    ys, xs = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    dmask = (np.abs(ys - 96) <= half_d) & (np.abs(xs - 128) <= half_d)

    half_r = size_m / 2 * K_RGB["fx"] / Z
    ry, rx = np.mgrid[0:1440, 0:1920]
    rmask = (np.abs(ry - 720) <= half_r) & (np.abs(rx - 960) <= half_r)

    coarse = profile(depth, conf, dmask, K_DEPTH)
    fine = profile(depth, conf, dmask, K_DEPTH, rgb_mask=rmask, K_rgb=K_RGB)
    assert coarse.extent_source == "depth" and fine.extent_source == "rgb"
    err = lambda p: abs(p.extent[0] - size_m)
    assert err(fine) < err(coarse)
    assert fine.extent[0] == pytest.approx(size_m, abs=0.004)


def test_jaw_width_is_the_minor_horizontal_extent():
    """An elongated object must be gripped across its narrow axis."""
    depth = np.full(SHAPE, 0.90, np.float32)
    conf = np.full(SHAPE, 2, np.uint8)
    ys, xs = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    bar = (np.abs(ys - 96) <= 3) & (np.abs(xs - 128) <= 18)   # long in x, thin in y
    p = profile(depth, conf, bar, K_DEPTH)
    assert p.jaw_width < p.extent[0]
    assert p.jaw_width == pytest.approx(2 * 3 * 0.90 / K_DEPTH["fx"], abs=0.004)


def test_rejects_a_mask_that_does_not_match_the_depth_grid():
    depth, conf, _ = scene()
    with pytest.raises(ValueError, match="does not match"):
        masked_points(depth, conf, np.ones((1440, 1920), bool), K_DEPTH)


def test_mask_downsample_preserves_a_centred_blob():
    rgb = np.zeros((1440, 1920), bool)
    rgb[600:840, 800:1120] = True
    small = mask_to_depth_grid(rgb, SHAPE)
    assert small.shape == SHAPE and small.any()
    ys, xs = np.nonzero(small)
    assert abs(ys.mean() - 96) < 6 and abs(xs.mean() - 128) < 6


def test_too_few_points_returns_none():
    depth, conf, _ = scene()
    tiny = np.zeros(SHAPE, bool); tiny[96, 128] = True
    assert profile(depth, conf, tiny, K_DEPTH) is None


def test_backprojection_round_trips_through_the_intrinsics():
    pts = backproject([96, 50], [128, 200], np.array([0.9, 1.2]), K_DEPTH)
    u = pts[:, 0] / pts[:, 2] * K_DEPTH["fx"] + K_DEPTH["cx"]
    v = pts[:, 1] / pts[:, 2] * K_DEPTH["fy"] + K_DEPTH["cy"]
    assert np.allclose(u, [128, 200]) and np.allclose(v, [96, 50])
