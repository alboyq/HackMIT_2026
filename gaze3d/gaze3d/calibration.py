"""Person-specific calibration: joint fit of a geometric screen model, then a
regularised residual regressor, with model selection by leave-one-target-out CV.

A frame sample carries everything needed to re-evaluate the mapping under different
parameters, so the fit can re-run the whole chain cheaply:
    o     gaze origin (face centre) in camera frame, mm      (3,)
    Rh    head rotation, model->camera                         (3,3)
    Rn    normalisation rotation, camera->normalised frame     (3,3)
    gn    ensemble gaze in the normalised frame, unit          (3,)
    feat  extra features for the residual regressor            (F,)

Geometric chain (Zhang et al. ETRA'18 de-normalisation + kappa + ray/plane):
    g_cam = Rn.T @ gn
    g_vis = Rh @ R_kappa @ Rh.T @ g_cam         (person-specific offset fixed in the head frame)
    PoG   = o + lambda * g_vis, lambda = n.(t_s - o) / n.g_vis
    (u,v) = (-l_x, l_y) / pitch_mm  with l = R_s.T (PoG - t_s)
Screen axes in the camera frame: u runs along -x (mirrored, both face the user), v along +y.
"""
from __future__ import annotations
import json, math
from dataclasses import dataclass, field, asdict
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .normalize import vec_to_pitchyaw, pitchyaw_to_vec


@dataclass
class ScreenGeometry:
    width_px: float             # CSS px
    height_px: float
    pitch_mm: float             # mm per CSS px
    cam_offset_px: tuple = (0.0, 18.0)   # camera centre relative to top-centre of pixel area (+y down)

    @property
    def t_prior(self) -> np.ndarray:
        """Camera-frame position of the screen's top-left pixel. The screen and the camera
        both face the user, so screen-right is camera -x: the top-left corner sits at +x."""
        cx, cy = self.cam_offset_px
        return np.array([(self.width_px / 2 - cx) * self.pitch_mm, -cy * self.pitch_mm, 0.0])


FEATURE_NAMES = [
    "u_geo", "v_geo",                       # geometric prediction (normalised by screen size)
    "ox", "oy", "oz",                      # face position (dm)
    "h_pitch", "h_yaw", "h_roll",          # head pose (rad)
    "geo_r_p", "geo_r_y", "geo_l_p", "geo_l_y",   # iris-geometry gaze, per eye, minus ensemble angles
    "open_r", "open_l",                    # eyelid aperture
]


@dataclass
class Params:
    kappa: np.ndarray = field(default_factory=lambda: np.zeros(2))      # (dpitch, dyaw) rad, head frame
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))      # screen rotation vector
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))          # screen origin, mm (set from prior)
    log_depth: float = 0.0                                              # depth scale on o_z
    kappa_mode: str = "head"                                            # "head" | "additive"

    def vec(self):
        return np.concatenate([self.kappa, self.omega, self.t, [self.log_depth]])

    @staticmethod
    def from_vec(v, mode):
        return Params(kappa=v[0:2].copy(), omega=v[2:5].copy(), t=v[5:8].copy(), log_depth=float(v[8]), kappa_mode=mode)

    def to_json(self):
        d = asdict(self)
        for k in ("kappa", "omega", "t"):
            d[k] = [float(x) for x in d[k]]
        return d

    @staticmethod
    def from_json(d):
        return Params(kappa=np.array(d["kappa"]), omega=np.array(d["omega"]), t=np.array(d["t"]),
                      log_depth=float(d["log_depth"]), kappa_mode=d.get("kappa_mode", "head"))


def _kappa_rot(kappa):
    dp, dy = kappa
    return Rotation.from_euler("yx", [dy, dp]).as_matrix()


def apply_kappa(g_cam, Rh, Rn, kappa, mode):
    if mode == "head":
        Rk = _kappa_rot(kappa)
        return (Rh @ Rk @ Rh.T @ g_cam.T).T if g_cam.ndim == 2 else Rh @ Rk @ Rh.T @ g_cam
    # additive offset in normalised pitch/yaw, then de-normalise
    gn = (Rn @ g_cam.T).T if g_cam.ndim == 2 else Rn @ g_cam
    py = vec_to_pitchyaw(gn) + kappa
    gn2 = pitchyaw_to_vec(py)
    return (Rn.T @ gn2.T).T if g_cam.ndim == 2 else (Rn.T @ gn2.T).ravel()


def geometric_pog(o, Rh, Rn, gn, params: Params, geom: ScreenGeometry):
    """Vectorised over samples. Returns (N,2) screen px and (N,) validity mask."""
    o = np.atleast_2d(o).astype(np.float64).copy()
    o[:, 2] *= math.exp(params.log_depth)
    N = o.shape[0]
    g_cam = np.einsum("nji,nj->ni", Rn, gn)        # Rn.T @ gn per sample
    if params.kappa_mode == "head":
        Rk = _kappa_rot(params.kappa)
        g_vis = np.einsum("nij,njk,nlk,nl->ni", Rh, np.broadcast_to(Rk, (N, 3, 3)), Rh, g_cam)
    else:
        py = vec_to_pitchyaw(gn) + params.kappa
        g_vis = np.einsum("nji,nj->ni", Rn, pitchyaw_to_vec(py))
    Rs = Rotation.from_rotvec(params.omega).as_matrix()
    n = Rs[:, 2]
    denom = g_vis @ n
    ok = np.abs(denom) > 1e-6
    lam = ((params.t - o) @ n) / np.where(ok, denom, 1.0)
    P = o + lam[:, None] * g_vis
    local = (P - params.t) @ Rs
    uv = np.stack([-local[:, 0], local[:, 1]], 1) / geom.pitch_mm
    ok &= lam > 0
    return uv, ok


def screen_point_mm(u, v, params: Params, geom: ScreenGeometry) -> np.ndarray:
    """Inverse of the (u,v) extraction above: screen pixel -> camera-frame mm."""
    Rs = Rotation.from_rotvec(params.omega).as_matrix()
    return params.t + Rs @ np.array([-u * geom.pitch_mm, v * geom.pitch_mm, 0.0])


# --- residual regressors -----------------------------------------------------------

class RidgeResidual:
    """Ridge on standardised features -> (du, dv) in px. alpha in standardised units."""
    def __init__(self, alpha=10.0, cols=None):
        self.alpha, self.cols = alpha, cols
        self.mu = self.sd = self.W = self.b = None

    def _X(self, F):
        F = F if self.cols is None else F[:, self.cols]
        return (F - self.mu) / self.sd

    def fit(self, F, resid, w=None):
        Fc = F if self.cols is None else F[:, self.cols]
        self.mu, self.sd = Fc.mean(0), Fc.std(0) + 1e-6
        X = self._X(F)
        w = np.ones(len(X)) if w is None else w
        Xw = X * w[:, None]
        A = X.T @ Xw + self.alpha * np.eye(X.shape[1])
        self.b = (resid * w[:, None]).sum(0) / w.sum()
        self.W = np.linalg.solve(A, Xw.T @ (resid - self.b))
        return self

    def predict(self, F):
        return self._X(np.atleast_2d(F)) @ self.W + self.b

    def to_json(self):
        return dict(kind="ridge", alpha=self.alpha, cols=self.cols, mu=self.mu.tolist(), sd=self.sd.tolist(),
                    W=self.W.tolist(), b=self.b.tolist())

    @staticmethod
    def from_json(d):
        r = RidgeResidual(d["alpha"], d["cols"])
        r.mu, r.sd, r.W, r.b = (np.array(d[k]) for k in ("mu", "sd", "W", "b"))
        return r


def poly_features(uv_norm, order):
    u, v = uv_norm[:, 0], uv_norm[:, 1]
    cols = [u, v]
    if order >= 2:
        cols += [u * u, u * v, v * v]
    if order >= 3:
        cols += [u ** 3, u * u * v, u * v * v, v ** 3]
    return np.stack(cols, 1)


# --- calibrator -------------------------------------------------------------------

@dataclass
class Candidate:
    name: str
    params: Params
    residual: object | None = None      # RidgeResidual over FEATURE_NAMES or poly features
    residual_kind: str = "none"         # none | ridge | poly2 | poly3
    cv_err_px: float = float("nan")
    fit_err_px: float = float("nan")


class Calibrator:
    def __init__(self, geom: ScreenGeometry):
        self.geom = geom
        self.best: Candidate | None = None
        self.report: dict = {}
        self.prior_sigma = dict(kappa=np.radians(15), omega=np.radians(4), t=15.0, log_depth=0.25)

    # -- packing ------------------------------------------------------------------
    @staticmethod
    def pack(samples):
        o = np.array([s["o"] for s in samples])
        Rh = np.array([s["Rh"] for s in samples])
        Rn = np.array([s["Rn"] for s in samples])
        gn = np.array([s["gn"] for s in samples])
        F = np.array([s["feat"] for s in samples])
        tgt = np.array([s["target"] for s in samples], dtype=np.float64)
        grp = np.array([s["point"] for s in samples])
        return o, Rh, Rn, gn, F, tgt, grp

    def _features(self, F, uv):
        """Augment stored per-frame features with the geometric prediction."""
        Fx = F.copy()
        Fx[:, 0] = uv[:, 0] / self.geom.width_px
        Fx[:, 1] = uv[:, 1] / self.geom.height_px
        return Fx

    # -- geometric fit --------------------------------------------------------------
    def fit_geometric(self, o, Rh, Rn, gn, tgt, mode, w=None, x0=None):
        geom = self.geom
        p0 = Params(kappa_mode=mode)
        p0.t = geom.t_prior.copy()
        x0 = p0.vec() if x0 is None else x0
        tp = geom.t_prior
        ps = self.prior_sigma
        w = np.ones(len(o)) if w is None else w
        sw = np.sqrt(w)
        px_sigma = 40.0     # px, sets scale between data and priors

        def resid(x):
            p = Params.from_vec(x, mode)
            uv, ok = geometric_pog(o, Rh, Rn, gn, p, geom)
            d = (uv - tgt) / px_sigma
            d[~ok] = 25.0
            data = (d * sw[:, None]).ravel()
            prior = np.concatenate([
                p.kappa / ps["kappa"], p.omega / ps["omega"], (p.t - tp) / ps["t"], [p.log_depth / ps["log_depth"]]])
            return np.concatenate([data, prior])

        lo = np.concatenate([[-0.5, -0.5], [-0.2] * 3, tp - 60.0, [-0.5]])
        hi = np.concatenate([[0.5, 0.5], [0.2] * 3, tp + 60.0, [0.5]])
        x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
        r = least_squares(resid, x0, bounds=(lo, hi), loss="soft_l1", f_scale=1.0, max_nfev=400, xtol=1e-8, ftol=1e-8)
        return Params.from_vec(r.x, mode)

    # -- full pipeline ----------------------------------------------------------------
    def _make_candidates(self, o, Rh, Rn, gn, F, tgt, w):
        cands = []
        for mode in ("head", "additive"):
            p = self.fit_geometric(o, Rh, Rn, gn, tgt, mode, w)
            uv, ok = geometric_pog(o, Rh, Rn, gn, p, self.geom)
            resid = tgt - uv
            cands.append(Candidate(f"geo-{mode}", p, None, "none"))
            Fx = self._features(F, uv)
            for alpha in (3.0, 30.0, 300.0):
                rr = RidgeResidual(alpha).fit(Fx[ok], resid[ok], w[ok])
                cands.append(Candidate(f"geo-{mode}+ridge{alpha:g}", p, rr, "ridge"))
            uvn = np.stack([uv[:, 0] / self.geom.width_px, uv[:, 1] / self.geom.height_px], 1)
            for order, alpha in ((2, 1.0), (3, 3.0)):
                pr = RidgeResidual(alpha).fit(poly_features(uvn, order)[ok], resid[ok], w[ok])
                cands.append(Candidate(f"geo-{mode}+poly{order}", p, pr, f"poly{order}"))
        return cands

    def predict_candidate(self, c: Candidate, o, Rh, Rn, gn, F):
        uv, ok = geometric_pog(o, Rh, Rn, gn, c.params, self.geom)
        if c.residual is not None:
            if c.residual_kind == "ridge":
                uv = uv + c.residual.predict(self._features(F, uv))
            else:
                order = int(c.residual_kind[-1])
                uvn = np.stack([uv[:, 0] / self.geom.width_px, uv[:, 1] / self.geom.height_px], 1)
                uv = uv + c.residual.predict(poly_features(uvn, order))
        return uv, ok

    def fit(self, samples, weights=None):
        """samples: list of dicts (see module doc). Chooses the candidate pipeline with
        the lowest leave-one-target-out error; returns a report dict."""
        d = np.array([np.linalg.norm(s["o"]) for s in samples])
        samples = [s for s, di in zip(samples, d) if 200.0 <= di <= 1200.0]
        o, Rh, Rn, gn, F, tgt, grp = self.pack(samples)
        w = np.ones(len(o)) if weights is None else np.asarray(weights, dtype=np.float64)
        points = np.unique(grp)
        names = None
        cv = {}
        if len(points) >= 4:
            for pt in points:
                tr, te = grp != pt, grp == pt
                cands = self._make_candidates(o[tr], Rh[tr], Rn[tr], gn[tr], F[tr], tgt[tr], w[tr])
                names = names or [c.name for c in cands]
                for c in cands:
                    uv, ok = self.predict_candidate(c, o[te], Rh[te], Rn[te], gn[te], F[te])
                    # error of the *median* prediction for the held-out target: what a fixation would see
                    med = np.median(uv[ok], axis=0) if ok.any() else np.array([np.nan, np.nan])
                    e = float(np.linalg.norm(med - tgt[te][0]))
                    cv.setdefault(c.name, []).append(e if np.isfinite(e) else 1e4)
        full = self._make_candidates(o, Rh, Rn, gn, F, tgt, w)
        for c in full:
            uv, ok = self.predict_candidate(c, o, Rh, Rn, gn, F)
            c.fit_err_px = float(np.sqrt(((uv[ok] - tgt[ok]) ** 2).sum(1).mean())) if ok.any() else np.nan
            c.cv_err_px = float(np.mean(cv[c.name])) if c.name in cv else c.fit_err_px
        full.sort(key=lambda c: c.cv_err_px)
        self.best = full[0]
        self.report = dict(
            n_samples=int(len(o)), n_points=int(len(points)),
            chosen=self.best.name, cv_err_px=self.best.cv_err_px, fit_err_px=self.best.fit_err_px,
            candidates=[dict(name=c.name, cv_err_px=c.cv_err_px, fit_err_px=c.fit_err_px) for c in full],
            params=self.best.params.to_json(), t_prior=[float(x) for x in self.geom.t_prior],
            per_point_cv={int(p): float(np.mean([cv[self.best.name][i] for i in range(len(points)) if points[i] == p]))
                          for p in points} if cv else {},
        )
        return self.report

    def predict(self, o, Rh, Rn, gn, feat):
        if self.best is None:
            p = Params(t=self.geom.t_prior.copy())
            uv, ok = geometric_pog(o[None], Rh[None], Rn[None], gn[None], p, self.geom)
            return uv[0], bool(ok[0])
        uv, ok = self.predict_candidate(self.best, o[None], Rh[None], Rn[None], gn[None], np.asarray(feat)[None])
        return uv[0], bool(ok[0])

    # -- persistence ------------------------------------------------------------------
    def to_json(self):
        if self.best is None:
            return None
        c = self.best
        return dict(name=c.name, params=c.params.to_json(), residual_kind=c.residual_kind,
                    residual=c.residual.to_json() if c.residual else None,
                    cv_err_px=c.cv_err_px, fit_err_px=c.fit_err_px, geom=asdict(self.geom), report=self.report)

    def load_json(self, d):
        self.geom = ScreenGeometry(**d["geom"])
        self.best = Candidate(d["name"], Params.from_json(d["params"]),
                              RidgeResidual.from_json(d["residual"]) if d["residual"] else None,
                              d["residual_kind"], d["cv_err_px"], d["fit_err_px"])
        self.report = d.get("report", {})
