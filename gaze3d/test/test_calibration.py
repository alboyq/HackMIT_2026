"""Synthetic end-to-end check of the geometric mapping + calibrator.

A virtual head sits in front of a virtual screen; for each target we compute the true
gaze direction from the face centre, corrupt it with a person-specific kappa offset in
the head frame plus noise, push it through the *same* normalisation used at runtime,
and check that the calibrator recovers a mapping that predicts held-out targets.
Run: .venv/bin/python -m pytest test/test_calibration.py -q   (or python test/test_calibration.py)
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.spatial.transform import Rotation
from gaze3d.normalize import normalize, XGAZE_FACE
from gaze3d.calibration import Calibrator, ScreenGeometry, geometric_pog, Params, screen_point_mm as _spm

rng = np.random.default_rng(0)
GEOM = ScreenGeometry(1512, 982, 0.168, (0.0, 18.0))
K = np.array([[1440.0, 0, 960], [0, 1440.0, 540], [0, 0, 1]])
IMG = np.zeros((1080, 1920, 3), np.uint8)


def screen_point_mm(u, v, omega, t):
    return _spm(u, v, Params(omega=np.asarray(omega, float), t=np.asarray(t, float)), GEOM)


def make_samples(targets, poses, kappa, omega, t, noise_deg=1.0, per=8):
    """poses: list of (head_rotvec, face_centre_mm). Returns sample dicts."""
    Rk = Rotation.from_euler("yx", [kappa[1], kappa[0]]).as_matrix()
    out = []
    for pi, (u, v) in enumerate(targets):
        P = screen_point_mm(u, v, omega, t)
        for _ in range(per):
            rv, o = poses[rng.integers(len(poses))]
            Rh = Rotation.from_rotvec(rv).as_matrix()
            g_vis = P - o; g_vis /= np.linalg.norm(g_vis)
            # the network sees the "optical" axis: undo the kappa correction
            g_opt = Rh @ Rk.T @ Rh.T @ g_vis
            # noise
            n = Rotation.from_rotvec(rng.normal(0, math.radians(noise_deg), 3)).as_matrix()
            g_opt = n @ g_opt
            _, Rn, _ = normalize(IMG, K, Rh, o, XGAZE_FACE)
            gn = Rn @ g_opt
            out.append(dict(o=o.tolist(), Rh=Rh.tolist(), Rn=Rn.tolist(), gn=gn.tolist(),
                            feat=[0, 0, o[0] / 100, o[1] / 100, o[2] / 100, rv[0], rv[1], rv[2], 0, 0, 0, 0, 0.3, 0.3],
                            target=[u, v], point=pi))
    return out


def grid(fr):
    return [(fx * GEOM.width_px, fy * GEOM.height_px) for fy in fr for fx in fr]


def test_prior_geometry_is_sane():
    """With no personal offset and the prior screen pose, a ray from a frontal head to a
    target must land on that target (checks Rn / de-normalisation / plane maths)."""
    p = Params(t=GEOM.t_prior.copy())
    o = np.array([0.0, 120.0, 550.0])
    Rh = np.eye(3)
    for (u, v) in grid([0.1, 0.5, 0.9]):
        P = screen_point_mm(u, v, np.zeros(3), GEOM.t_prior)
        g = P - o; g /= np.linalg.norm(g)
        _, Rn, _ = normalize(IMG, K, Rh, o, XGAZE_FACE)
        uv, ok = geometric_pog(o[None], Rh[None], Rn[None], (Rn @ g)[None], p, GEOM)
        assert ok[0] and np.linalg.norm(uv[0] - [u, v]) < 1e-6, (u, v, uv)


def test_recovers_kappa_and_generalises_across_head_pose():
    kappa = np.radians([3.0, -4.0])              # person-specific offset, head frame
    omega = np.radians([2.0, -1.0, 0.5])         # screen slightly tilted vs camera
    t = GEOM.t_prior + np.array([6.0, -4.0, 8.0])
    calib_poses = [(np.radians([p, y, r]), np.array([x, 110.0, z])) for p, y, r, x, z in
                   [(0, 0, 0, 0, 550), (5, 15, 0, 40, 520), (-8, -12, 3, -30, 600), (10, 0, -2, 0, 480), (0, 20, 0, 60, 560)]]
    samples = make_samples(grid([0.1, 0.5, 0.9]), calib_poses, kappa, omega, t)
    cal = Calibrator(GEOM)
    rep = cal.fit(samples)
    print("chosen", rep["chosen"], "cv px", round(rep["cv_err_px"], 1), "fit px", round(rep["fit_err_px"], 1))
    print("kappa est deg", np.degrees(rep["params"]["kappa"]), "true", np.degrees(kappa))
    # held-out targets, at NEW head poses well outside the calibration set
    test_poses = [(np.radians([-12, 25, 5]), np.array([90.0, 140.0, 650.0])), (np.radians([12, -22, -4]), np.array([-80.0, 90.0, 450.0]))]
    tests = make_samples(grid([0.25, 0.75]), test_poses, kappa, omega, t, noise_deg=0.0, per=1)
    errs = []
    for s in tests:
        uv, ok = cal.predict(np.array(s["o"]), np.array(s["Rh"]), np.array(s["Rn"]), np.array(s["gn"]), np.array(s["feat"]))
        assert ok
        errs.append(np.linalg.norm(uv - s["target"]))
    e = np.mean(errs)
    print("held-out error at novel poses: %.1f px (%.2f deg at 55 cm)" % (e, math.degrees(math.atan(e * GEOM.pitch_mm / 550))))
    assert e < 40, e            # < ~0.7 deg
    # both kappa parameterisations are admissible; the magnitude must match the injected 5 deg
    assert abs(np.degrees(np.hypot(*rep["params"]["kappa"])) - 5.0) < 1.2


def test_static_only_calibration_still_fits():
    kappa = np.radians([2.0, 2.0])
    samples = make_samples(grid([0.1, 0.5, 0.9]), [(np.zeros(3), np.array([0.0, 110.0, 550.0]))], kappa, np.zeros(3), GEOM.t_prior)
    rep = Calibrator(GEOM).fit(samples)
    print("static-only: chosen", rep["chosen"], "cv px", round(rep["cv_err_px"], 1))
    # 1 deg noise ~ 57 px per sample; the held-out median of 8 samples should sit well inside that
    assert rep["cv_err_px"] < 60, rep["cv_err_px"]


if __name__ == "__main__":
    test_prior_geometry_is_sane(); print("prior geometry ok")
    test_recovers_kappa_and_generalises_across_head_pose()
    test_static_only_calibration_still_fits()
    print("all ok")
