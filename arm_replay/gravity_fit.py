"""Fit the real arm's gravity + friction to the torques the motors delivered (passive CAN captures of runs 2-5).

Physics, not a lookup table: the vendor MJCF (YAM + linear_4310) is the prior; the unknowns are the extra mass and mass-moment
(m, m*c_x, m*c_y, m*c_z in the body frame) of links 2, 3 and 4 (link 4 stands for everything beyond J4: wrist, gripper, camera,
cables), plus Coulomb + viscous friction for every joint. Gravity torque is exactly linear in those, so this is a ridge regression
with an honest posterior covariance: it tells us how well each pose is constrained by the data we have."""
import json, sys, numpy as np, mujoco
from scipy.optimize import least_squares
from gravity_data import samples, RUNS, TAU, LAP, OFFSET

def menagerie_xml():
    import os
    from pathlib import Path
    for c in ([Path(os.environ["YAM_MENAGERIE"])] if os.environ.get("YAM_MENAGERIE") else []) + [Path.home() / "HackMIT_2026/third_party/mujoco_menagerie/i2rt_yam"]:
        c = c if c.name == "yam.xml" else c / "yam.xml"
        if c.exists(): return str(c)
    raise FileNotFoundError("mujoco_menagerie/i2rt_yam/yam.xml not found; set YAM_MENAGERIE")


XML = menagerie_xml()                                  # the fit is relative to the SIM's own stock model
BODIES = ["link_2", "link_3", "link_4"]; SIG = 0.25; V0 = 0.02; V_STATIC = 0.006
PRIOR = np.array([0.25, 0.03, 0.03, 0.03] * 3)             # 1-sigma: extra mass (kg), mass-moment components (kg.m)

class Phys:
    def __init__(self):
        self.M = mujoco.MjModel.from_xml_path(XML); self.M.body_gravcomp[:] = 0.0; self.D = mujoco.MjData(self.M)
        self.ids = [mujoco.mj_name2id(self.M, mujoco.mjtObj.mjOBJ_BODY, b) for b in BODIES]
        self.m0 = self.M.body_mass.copy(); self.c0 = self.M.body_ipos.copy()
    def _set(self, theta):
        self.M.body_mass[:] = self.m0; self.M.body_ipos[:] = self.c0
        for k, b in enumerate(self.ids):
            dm, mu = theta[4 * k], theta[4 * k + 1:4 * k + 4]
            m = self.m0[b] + dm; self.M.body_mass[b] = m; self.M.body_ipos[b] = (self.m0[b] * self.c0[b] + mu) / m
    def tau(self, q, theta=None):
        self._set(np.zeros(12) if theta is None else theta)
        self.D.qpos[:] = 0.0; self.D.qpos[:6] = q; self.D.qvel[:] = 0.0; mujoco.mj_forward(self.M, self.D); return self.D.qfrc_bias[:6].copy()
    def design(self, q):
        """tau0 (6) and A (6 x 12): gravity torque is linear in theta (exact)."""
        t0 = self.tau(q); A = np.zeros((6, 12))
        for p in range(12):
            e = np.zeros(12); e[p] = 1.0; A[:, p] = self.tau(q, e) - t0
        return t0, A

def on_a_stop(q, tol=0.03):
    """Samples where a joint rests on its mechanical stop measure the stop's reaction, not gravity: leave them out."""
    return q[1] < tol or q[2] < tol or q[3] > 1.665 - 0.06 or q[3] < -1.665 + 0.06

def build(runs, step=8, static_step=40):
    P = Phys(); rows = []
    for r in runs:
        d = samples(r); g = np.where(d["good"])[0]
        for n, i in enumerate(g):
            if on_a_stop(d["q"][i]): continue
            moving = (np.abs(d["v"][i, 1:4]) > V_STATIC).any()
            if n % (step if moving else static_step): continue
            t0, A = P.design(d["q"][i]); rows.append((r, d["t"][i], d["q"][i], d["v"][i], d["tau"][i], t0, A))
    return P, rows

# parameter vector: theta(12) | b(6) | c(6) | kappa(3: torque scale for J2..J4 - what the motor really delivers per commanded N.m)
N_TH, N_B, N_C, N_K = 12, 6, 6, 3
def split(x): return x[:12], x[12:18], x[18:24], x[24:27]

def gravity_term(x, t0, A):
    th, b, c, k = split(x); g = t0 + A @ th; out = np.zeros(6); out[1:4] = k * g[1:4]; return out

def predict_static(x, t0, A, v_dir=None):
    return gravity_term(x, t0, A)

def residuals(xf, rows, mask, x_full):
    x = x_full.copy(); x[mask] = xf; th, b, c, k = split(x); out = []
    for r, t, q, v, tau, t0, A in rows:
        grav = gravity_term(x, t0, A)
        for j in range(6):
            if abs(v[j]) > V_STATIC: out.append((tau[j] - grav[j] - b[j] * np.tanh(v[j] / V0) - c[j] * v[j]) / SIG)
            else:                                    # standing still: friction may be anywhere in [-b, +b]
                e = tau[j] - grav[j]; out.append(np.sign(e) * max(0.0, abs(e) - b[j]) / SIG)
    out.extend(th / PRIOR); return np.array(out)

X0 = np.concatenate([np.zeros(12), 0.3 * np.ones(6), 0.05 * np.ones(6), np.ones(3)])
def fit(rows, free_theta=True, free_kappa=True, x_init=None):
    x_full = (X0 if x_init is None else x_init).copy(); mask = np.ones(27, bool)
    if not free_theta: mask[:12] = False
    if not free_kappa: mask[24:] = False
    lo = np.full(27, -np.inf); hi = np.full(27, np.inf); lo[12:24] = 0.0; lo[24:] = 0.3; hi[24:] = 3.0
    scale = np.concatenate([PRIOR, 0.3 * np.ones(12), 0.3 * np.ones(3)])
    s = least_squares(residuals, x_full[mask], args=(rows, mask, x_full), bounds=(lo[mask], hi[mask]), x_scale=scale[mask])
    x = x_full.copy(); x[mask] = s.x; s.x_full = x; return s

def rms_by_joint(x, rows, moving_only=False):
    th, b, c, k = split(x); err = {j: [] for j in range(6)}
    for r, t, q, v, tau, t0, A in rows:
        grav = gravity_term(x, t0, A)
        for j in range(6):
            if abs(v[j]) > V_STATIC: err[j].append(tau[j] - grav[j] - b[j] * np.tanh(v[j] / V0) - c[j] * v[j])
            elif not moving_only: e = tau[j] - grav[j]; err[j].append(np.sign(e) * max(0.0, abs(e) - b[j]))
    return np.array([np.sqrt(np.mean(np.square(err[j]))) if err[j] else np.nan for j in range(6)])

def show(name, s, rows):
    x = s.x_full; th, b, c, k = split(x)
    print(f"\n== {name}   cost {s.cost:.0f}"); print("   kappa J2..J4:", np.round(k, 3), "  friction b:", np.round(b, 2), " c:", np.round(c, 2))
    if np.any(th): print("   theta:", " ".join(f"{n}: dm={th[4*i]:+.2f} mu={np.round(th[4*i+1:4*i+4],3)}" for i, n in enumerate(BODIES)))
    print("   RMS N.m per joint (J1..J6) all:", np.round(rms_by_joint(x, rows), 2), " moving only:", np.round(rms_by_joint(x, rows, True), 2))

if __name__ == "__main__":
    P, rows = build([2, 3, 4, 5]); print("samples:", len(rows))
    show("vendor model as is (kappa=1, friction fitted)", fit(rows, False, False), rows)
    show("M1: torque scale only (3 gravity numbers)", fit(rows, False, True), rows)
    show("M2: mass lumps only", fit(rows, True, False), rows)
    show("M3: both", fit(rows, True, True), rows)
