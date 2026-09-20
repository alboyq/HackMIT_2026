import numpy as np, json
from gravity_fit import *

def fit_tied(rows_):
    """ONE torque scale for J2..J4 (the whole arm is X% heavier than the model, or the motors deliver X% less), plus friction."""
    def res(z):
        x = X0.copy(); x[12:24] = z[:12]; x[24:27] = z[12]; return residuals(np.array([]), rows_, np.zeros(27, bool), x)
    z0 = np.concatenate([X0[12:24], [1.0]]); lo = np.concatenate([np.zeros(12), [0.3]]); hi = np.concatenate([np.full(12, np.inf), [3.0]])
    s = least_squares(res, z0, bounds=(lo, hi), x_scale=np.concatenate([0.3 * np.ones(12), [0.3]]))
    x = X0.copy(); x[12:24] = s.x[:12]; x[24:27] = s.x[12]; s.x_full = x; s.J = s.jac; return s



def fit_groups(rows_):
    """Same torque scale for the two DM4340 joints (J2, J3), its own for the DM4310 wrist (J4): matches the hardware (two motor types)."""
    def res(z):
        x = X0.copy(); x[12:24] = z[:12]; x[24:26] = z[12]; x[26] = z[13]; return residuals(np.array([]), rows_, np.zeros(27, bool), x)
    z0 = np.concatenate([X0[12:24], [1.0, 1.0]]); lo = np.concatenate([np.zeros(12), [0.3, 0.3]]); hi = np.concatenate([np.full(12, np.inf), [3.0, 3.0]])
    s = least_squares(res, z0, bounds=(lo, hi), x_scale=np.concatenate([0.3 * np.ones(12), [0.3, 0.3]]))
    x = X0.copy(); x[12:24] = s.x[:12]; x[24:26] = s.x[12]; x[26] = s.x[13]; s.x_full = x
    J = s.jac; cov = np.linalg.inv(J.T @ J); s.sd = np.sqrt(np.diag(cov))[12:]
    return s
