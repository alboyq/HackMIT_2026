"""Turn the passive CAN captures (capture_run*.txt) into per-tick samples: joint positions, velocities and the torque the motors
actually delivered. Only ticks where all six motors are enabled, running with kp>0 (i.e. NOT a limp/zero-gain frame), and the arm is
not accelerating hard - so torque = gravity + friction (+ negligible inertia)."""
import bisect
from pathlib import Path
import numpy as np
from canview import load
LAP = np.array([0, 1, 1, 0, 0, 0]); TAU = 2 * np.pi
OFFSET = np.array([-1.4313, 0.0044, 0.0101, -0.0074, 0.0015, 1.3035])
HERE = Path(__file__).resolve().parent
RUNS = {r: HERE / "logs" / f"capture_run{r}.txt.gz" for r in (4, 5, 6, 7)}      # runs 1-3 (old position-only code) are not needed by the fit
LAPS = {6: np.zeros(6), 7: np.zeros(6)}      # runs 4-5: first power session (J2,J3 one lap lower, LAP below); runs 6-7: after a power cycle, all laps 0

def samples(run, acc_max=1.5, t_skip=0.4):
    c, f, _ = load(RUNS[run])
    ref = f[3]; T = np.array([x[0] for x in ref]); n = len(T)
    P = np.full((n, 6), np.nan); V = np.full((n, 6), np.nan); TQ = np.full((n, 6), np.nan); KP = np.zeros((n, 6)); ST = np.zeros((n, 6), bool)
    for m in range(1, 7):
        fm = f[m]; t = np.array([x[0] for x in fm]); pos = np.array([x[2] for x in fm]); vel = np.array([x[3] for x in fm]); tq = np.array([x[4] for x in fm])
        ok = np.array([x[1] == "normal" for x in fm])
        idx = np.clip(np.searchsorted(t, T), 0, len(t) - 1)
        P[:, m - 1], V[:, m - 1], TQ[:, m - 1], ST[:, m - 1] = pos[idx], vel[idx], tq[idx], ok[idx]
        mit = [x for x in c[m] if x[1] == "mit"]; tm = np.array([x[0] for x in mit]); kp = np.array([x[4] for x in mit])
        k = np.clip(np.searchsorted(tm, T) - 1, 0, len(tm) - 1); KP[:, m - 1] = kp[k]
    # acceleration from smoothed velocity (window ~0.1 s)
    w = 20; kern = np.ones(w) / w; A = np.zeros_like(V)
    for j in range(6):
        vs = np.convolve(V[:, j], kern, mode="same"); A[:, j] = np.gradient(vs, T)
    good = ST.all(axis=1) & (KP > 0).all(axis=1) & (np.abs(A[:, 1:4]) < acc_max).all(axis=1)
    # drop the first t_skip after all motors become enabled
    if good.any(): good &= T > T[np.argmax(ST.all(axis=1))] + t_skip
    Q = P + TAU * LAPS.get(run, LAP) - OFFSET          # model joint angles
    return dict(t=T - T[0], raw=P, q=Q, v=V, tau=TQ, acc=A, good=good, run=np.full(n, run))

if __name__ == "__main__":
    for r in RUNS:
        d = samples(r); g = d["good"]; q = d["q"][g]; print(f"run {r}: {g.sum()}/{len(g)} usable ticks; model q2 {q[:,1].min():.2f}..{q[:,1].max():.2f}  q3 {q[:,2].min():.2f}..{q[:,2].max():.2f}  q4 {q[:,3].min():.2f}..{q[:,3].max():.2f}")
        # stationary clusters
        st = g & (np.abs(d["v"][:, 1:4]) < 0.006).all(axis=1)
        idx = np.where(st)[0][::200]
        for i in idx: print(f"     t={d['t'][i]:5.1f}  q2..q4={np.round(d['q'][i,1:4],3)}  tau={np.round(d['tau'][i,:4],2)}")
