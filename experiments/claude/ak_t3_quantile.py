"""Task 3 round 5: metric-aligned read-out with nonparametric conditional quantiles.

For L1 loss the optimal prediction is the median of y | x. With P(y > 0 | x) = p, that median is 0 if p <= 0.5,
otherwise the q = 1 - 0.5 / p quantile of the positive part. v8 reads this quantile off a lognormal with one
global sigma (mixture-median). Here the quantiles of r = log1p(y) - log1p(historical_spending) are fitted directly
(LightGBM quantile objective at 12 levels, positives only), so a row-specific spread and skew are used.
Diagnostics first: shape and predictability of the stage-2 residual.
Same folds, features (identity fill + imputed segment) and tuned params as v8.
"""
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

warnings.filterwarnings("ignore")
import common  # noqa: E402
from common import OUT, t3_blend_score  # noqa: E402
from src import task3  # noqa: E402
from src.metrics import nmae  # noqa: E402
from src.modeling import make_folds, fit_predict  # noqa: E402
from t3_fill import fill  # noqa: E402
from t3_segment import impute_segment  # noqa: E402

tr, te = task3.load()
y = tr[task3.TARGET].values.astype(float)
z, pos = np.log1p(y), y > 0
folds = make_folds(y, kind="regression")
trf, tef = fill(tr)[0], fill(te)[0]
X, Xt, _ = impute_segment(task3.add_features(trf), task3.add_features(tef))
b, bt = np.log1p(trf.historical_spending.values), np.log1p(tef.historical_spending.values)
okb, okbt = np.isfinite(b), np.isfinite(bt)
TP = json.loads((OUT / "t3_tuned_params.json").read_text())
half = lambda p: {**p, "learning_rate": p["learning_rate"] / 2}
S1 = half(TP["t3_stage1"])
R2 = {"objective": "regression_l1", **half(TP["t3_ratio_stage2"])}
seg = X.segment.values

# ---------------------------------------------------------------- stage 1 and the L1 ratio model (as v8)
p_oof, p_test = np.zeros(len(y)), np.zeros(len(Xt))
r_l1 = np.full(len(y), np.nan)
for a, v in folds:
    _, (pv, pt), _ = fit_predict("lgb", "binary", S1, X.iloc[a], pos[a].astype(int), X.iloc[v], pos[v].astype(int), [X.iloc[v], Xt])
    p_oof[v], p_test = pv, p_test + pt / len(folds)
    ap, vp = a[pos[a] & okb[a]], v[pos[v] & okb[v]]
    _, (rv,), _ = fit_predict("lgb", "regression", R2, X.iloc[ap], z[ap] - b[ap], X.iloc[vp], z[vp] - b[vp], [X.iloc[vp]])
    r_l1[vp] = rv
m = pos & okb
e = (z - b)[m] - r_l1[m]
print(f"stage-2 residual on positives: MAE {np.mean(np.abs(e)):.4f}, sd {e.std():.4f}, MAE/sd {np.mean(np.abs(e)) / e.std():.3f} "
      f"(normal 0.798, Laplace 0.707), skew {skew(e):.2f}, excess kurtosis {kurtosis(e):.2f}")
print("residual MAE by segment:", {int(s): round(float(np.mean(np.abs(e[seg[m] == s]))), 4) for s in (0, 1, 2, 3)})
# is the spread predictable? LightGBM on |e| (2 folds)
Xm, ae = X[m].reset_index(drop=True), np.abs(e)
idx = np.random.default_rng(0).permutation(len(ae))
h1, h2 = idx[: len(idx) // 2], idx[len(idx) // 2:]
_, (pa,), _ = fit_predict("lgb", "regression", {"objective": "regression", "learning_rate": 0.05}, Xm.iloc[h1], ae[h1], Xm.iloc[h2], ae[h2], [Xm.iloc[h2]])
r2 = 1 - np.mean((ae[h2] - pa) ** 2) / np.var(ae[h2])
print(f"spread predictability: R^2 of |residual| from features = {r2:.3f} (0 = constant spread)", flush=True)

# ---------------------------------------------------------------- quantile models of r on positives
A = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55])
QP = {k: v for k, v in R2.items() if k != "objective"}
QP["learning_rate"] = 0.03
Qo, Qt = np.full((len(y), len(A)), np.nan), np.zeros((len(Xt), len(A)))
for j, al in enumerate(A):
    prm = {**QP, "objective": "quantile", "alpha": float(al), "metric": "quantile"}
    for a, v in folds:
        ap, vp, vb = a[pos[a] & okb[a]], v[pos[v] & okb[v]], v[okb[v]]
        _, (rv, rt), _ = fit_predict("lgb", "regression", prm, X.iloc[ap], z[ap] - b[ap], X.iloc[vp], z[vp] - b[vp], [X.iloc[vb], Xt])
        Qo[vb, j] = rv
        Qt[:, j] += rt / len(folds)
    cover = np.mean(((z - b)[m] <= Qo[m, j]))
    print(f"  alpha {al:.2f}: empirical coverage on held-out positives {cover:.3f}", flush=True)
Qo, Qt = np.sort(Qo, axis=1), np.sort(Qt, axis=1)          # remove quantile crossing
np.savez(OUT / "t3_quantile_members.npz", Qo=Qo, Qt=Qt, A=A, p_oof=p_oof, p_test=p_test)


def readout(p, Q, base, shift):
    pa = np.clip(p + shift, 1e-6, 1 - 1e-6)
    q = np.clip(1 - 0.5 / pa, A[0], A[-1])
    rq = np.array([np.interp(qq, A, row) for qq, row in zip(q, Q)])
    out = np.where(pa > 0.5, np.expm1(base + rq), 0.0)
    return np.clip(np.nan_to_num(out), 0, None)


best = None
for sh in np.linspace(-0.08, 0.08, 17):
    s = nmae(y, readout(p_oof, Qo, np.where(okb, b, 0), sh))
    if best is None or s > best[0]:
        best = (s, sh)
print(f"\nquantile read-out alone: {best[0]:.5f} (p shift {best[1]:+.3f}); v8 best single member 0.79376", flush=True)
q_oof = readout(p_oof, Qo, np.where(okb, b, 0), best[1])
q_test = readout(p_test, Qt, np.where(okbt, bt, 0), best[1])

v8 = np.load(OUT / "t3_v8.npz")
P, gate = v8["P_oof"], v8["gate_oof"]
s8, _ = t3_blend_score(y, folds, P, gate)
s9, _ = t3_blend_score(y, folds, np.column_stack([P, q_oof]), gate)
print(f"blend: v8 pool {s8:.5f} -> + quantile read-out {s9:.5f} ({s9 - s8:+.5f})")
np.savez(OUT / "t3_quantile_readout.npz", oof=q_oof, test=q_test, shift=best[1])
