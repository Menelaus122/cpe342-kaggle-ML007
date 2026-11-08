"""Task 3: is the +0.00011 of H3-all real, or optimizer noise? Solve the L1 blend EXACTLY.

Non-negative L1 (least absolute deviation) blending is a linear program:
    min sum(u + v)  s.t.  P w + u - v = y,  w, u, v >= 0.
HiGHS finds the global optimum, so two pools can be compared without Powell's path dependence.
Same nested protocol as the notebook: weights from 4 folds, applied to the 5th; zero gate; best global scale.
"""
import time
import warnings

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

warnings.filterwarnings("ignore")
import common  # noqa: E402
from common import OUT  # noqa: E402
from src import task3  # noqa: E402
from src.metrics import nmae  # noqa: E402
from src.modeling import make_folds  # noqa: E402

tr, _ = task3.load()
y = tr[task3.TARGET].values.astype(float)
folds = make_folds(y, kind="regression")
v8 = np.load(OUT / "t3_v8.npz")
P0, gate = v8["P_oof"], v8["gate_oof"]
new = np.load(OUT / "t3_round8_members.npz")
P1 = np.column_stack([P0, new["oof"]])
scales = np.linspace(0.85, 1.15, 61)


def lad_weights(P, t):
    n, k = P.shape
    s = P.max(0) + 1e-9                         # column scaling for numerical stability
    A = sp.hstack([sp.csr_matrix(P / s), sp.eye(n), -sp.eye(n)], format="csr")
    c = np.r_[np.zeros(k), np.ones(2 * n)]
    res = linprog(c, A_eq=A, b_eq=t, bounds=(0, None), method="highs")
    assert res.status == 0, res.message
    return res.x[:k] / s


def nested(P):
    bo, W = np.zeros(len(y)), []
    for a, b in folds:
        w = lad_weights(P[a], y[a])
        W.append(w)
        bo[b] = P[b] @ w
    sc = [nmae(y, bo * s * gate) for s in scales]
    i = int(np.argmax(sc))
    return sc[i], bo * scales[i] * gate, np.array(W)


rng = np.random.default_rng(0)


def boot(a, b, B=1000):
    e1, e0 = np.abs(y - a), np.abs(y - b)
    d = []
    for _ in range(B):
        w = rng.poisson(1.0, len(y)).astype(float)
        sy = w @ y
        d.append(1 / (1 + (w @ e1) / sy) - 1 / (1 + (w @ e0) / sy))
    return np.percentile(d, [2.5, 97.5])


t0 = time.time()
s0, f0, W0 = nested(P0)
print(f"B0 pool (26 members), exact L1 blend: {s0:.5f}   (Powell, as in v8: 0.79403)   [{time.time() - t0:.0f}s]", flush=True)
s1, f1, W1 = nested(P1)
lo, hi = boot(f1, f0)
print(f"H3-all pool (34 members), exact L1 blend: {s1:.5f}   (Powell: 0.79415); delta {s1 - s0:+.5f}, 95% CI [{lo:+.5f}, {hi:+.5f}]")
names = list(v8["names"]) + list(new["names"])
print("mean fold weight of the new members:", {n: round(float(w), 3) for n, w in zip(names[26:], W1.mean(0)[26:]) if w > 1e-3})
np.savez(OUT / "t3_exact_blend.npz", f0=f0, f1=f1, W0=W0, W1=W1)
