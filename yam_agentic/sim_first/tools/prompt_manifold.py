"""Is the 2.9 deg fit floor irreducible, and would more data ever close it?

    python tools/prompt_manifold.py <demo_dir> [--shards N]

Three questions, all answered from the demo shards alone (states and actions; no images, no GPU):

1. LABEL AMBIGUITY. If two frames from different episodes show the policy the same prompt but
   the expert labelled them differently, no model can fit both. Extrapolating 1-NN label
   disagreement to zero prompt distance measures that irreducible floor.
   Measured 2026-09-20 on yam_v6: intercept -0.29 deg ~= 0. There is NO ambiguity; the mapping
   is a clean function, so the 2.9 deg floor was never a Bayes limit.

2. LOCAL LIPSCHITZ. How fast the correct action changes with the prompt: 3.88 deg per sd-unit.

3. INTRINSIC DIMENSION, from how 1-NN distance shrinks with dataset size (~N^(-1/d)): 6.4.
   Together these price the "just collect orders more data" option:
       2.0 deg ->  54x the data (~84 k episodes, ~12 h, ~1.8 TB)
       1.5 deg -> 339x the data (~530 k episodes, ~80 h, ~12 TB)
   i.e. dead. See ../VLA_VERDICT.md section 9.
"""
import argparse
import glob
import numpy as np

DEG = 180.0 / np.pi


def load(demo_dir, limit):
    S, A, E, off = [], [], [], 0
    for p in sorted(glob.glob(f"{demo_dir}/shard_*.npz"))[:limit]:
        z = np.load(p)
        S.append(z["state"]); A.append(z["action"]); E.append(z["episode"] + off)
        off = int(E[-1].max()) + 1
    if not S:
        raise SystemExit(f"no shard_*.npz under {demo_dir}")
    return np.concatenate(S), np.concatenate(A), np.concatenate(E)


def nn_against(Fn, E, A, Q, pool):
    """1-NN distance and action disagreement, restricted to CROSS-EPISODE neighbours."""
    P, PE = Fn[pool], E[pool]
    d1, e1 = [], []
    for s in range(0, len(Q), 250):
        q = Q[s:s + 250]
        d = ((Fn[q][:, None, :] - P[None, :, :]) ** 2).sum(-1)
        d[E[q][:, None] == PE[None, :]] = np.inf
        j = np.argmin(d, axis=1)
        d1.append(np.sqrt(d[np.arange(len(q)), j]))
        e1.append(np.abs(A[pool][j][:, :6] - A[q][:, :6]).mean(1) * DEG)
    return np.concatenate(d1), np.concatenate(e1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("demos")
    ap.add_argument("--shards", type=int, default=10)
    ap.add_argument("--queries", type=int, default=3000)
    a = ap.parse_args()
    S, A, E = load(a.demos, a.shards)
    F = S[:, :13].astype(np.float32)
    sd = F.std(0); sd[sd < 1e-6] = 1.0
    Fn = F / sd
    rng = np.random.default_rng(0)
    Q = rng.choice(len(Fn), min(a.queries, len(Fn)), replace=False)
    print(f"{len(S)} frames / {len(np.unique(E))} episodes\n")

    d1, e1 = nn_against(Fn, E, A, Q, np.arange(len(Fn)))
    print("1. label disagreement vs prompt distance (1-NN, cross-episode)")
    qs = np.quantile(d1, np.linspace(0, 1, 11))
    xs, ys = [], []
    for i in range(10):
        m = (d1 >= qs[i]) & (d1 <= qs[i + 1])
        if m.sum() < 10:
            continue
        xs.append(d1[m].mean()); ys.append(e1[m].mean())
        print(f"   decile {i+1:2d}: distance {xs[-1]:6.3f} sd-units -> action MAE {ys[-1]:6.2f} deg")
    xs, ys = np.array(xs), np.array(ys)
    k = min(5, len(xs))
    lip, icept = np.polyfit(xs[:k], ys[:k], 1)
    print(f"\n   fit over the {k} closest deciles: MAE(d) = {icept:.2f} + {lip:.2f} * d")
    print(f"   >>> irreducible error at d=0: {icept:.2f} deg   (~0 = no label ambiguity)")
    print(f"   >>> local Lipschitz constant: {lip:.2f} deg per sd-unit\n")

    print("2. how 1-NN distance shrinks with dataset size")
    eps = np.unique(E)
    rows = []
    for frac in (0.125, 0.25, 0.5, 1.0):
        keep = rng.choice(eps, max(2, int(len(eps) * frac)), replace=False)
        pool = np.where(np.isin(E, keep))[0]
        m = nn_against(Fn, E, A, Q, pool)[0].mean()
        rows.append((len(keep), m))
        print(f"   {len(keep):6d} episodes -> mean 1-NN distance {m:.4f} sd-units")
    n = np.array([r[0] for r in rows], float)
    dd = np.array([r[1] for r in rows])
    dim = -1.0 / np.polyfit(np.log(n), np.log(dd), 1)[0]
    print(f"\n   >>> intrinsic dimension ~= {dim:.1f}\n")

    print("3. what it would cost to reach a target by data alone")
    cur_n, cur_d = rows[-1]
    for target in (2.0, 1.5):
        factor = (cur_d / (target / lip)) ** dim
        print(f"   {target} deg -> {factor:,.0f}x the data = {cur_n * factor:,.0f} episodes")
