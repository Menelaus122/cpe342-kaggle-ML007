"""Task 3: do the many zero predictions cost score? (user question, 2026-11-08)

The metric 1 / (1 + sum|y - yhat| / sum y) only sees the total absolute error. Moving one row's prediction from 0 up to
a small c changes the error by +c if the row really spent 0 and by -c if it spent more than c. So for a group of rows,
0 is the best prediction whenever more than half of them spend 0, i.e. whenever P(y = 0 | x) > 0.5 (the median rule).
Checks on the v8 out-of-fold predictions (same folds, same blend recipe as the notebook):
  1. zero shares: labels vs OOF predictions vs test predictions, by segment
  2. where the error comes from (true zero / positive x predicted zero / positive)
  3. calibration of the zero decision by the stage-1 probability P(y > 0)
  4. counterfactuals: no gate, zeros -> alpha * blend, zeros -> constant, gate threshold sweep, mean read-out
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import common  # noqa: E402
from common import OUT  # noqa: E402
from src import task3  # noqa: E402
from src.metrics import nmae  # noqa: E402
from src.modeling import make_folds  # noqa: E402

pd.set_option("display.width", 200)
tr3, te3 = task3.load()
y = tr3[task3.TARGET].values.astype(float)
folds = make_folds(y, kind="regression")
v8 = np.load(OUT / "t3_v8.npz")
P, names, gate = v8["P_oof"], list(v8["names"]), v8["gate_oof"]
qm = np.load(OUT / "t3_quantile_members.npz")
p = qm["p_oof"]                                                   # v8 stage 1, P(y > 0), out-of-fold

# ------------------------------------------------------------------ nested L1 blend (cached; Powell is slow)
cache = OUT / "t3_v8_blend_oof.npz"
if cache.exists():
    blend = np.load(cache)["blend"]
else:
    blend, W = np.zeros(len(y)), []
    for a, b in folds:
        w = task3.l1_blend_weights(y[a], [P[a, j] for j in range(P.shape[1])])
        W.append(w)
        blend[b] = P[b] @ w
    np.savez(cache, blend=blend, W=np.array(W))
scales = np.linspace(0.85, 1.15, 61)


def best_scale(pred_unscaled):
    sc = [nmae(y, pred_unscaled * s) for s in scales]
    i = int(np.argmax(sc))
    return sc[i], scales[i]


s_v8, k0 = best_scale(blend * gate)
f = blend * k0 * gate
print(f"v8 OOF score {s_v8:.5f} (stored {float(v8['score']):.5f}); blend scale {k0:.3f}")
lo, hi = p[gate == 1].min(), p[gate == 0].max()
print(f"gate = stage-1 P(y>0) above {hi:.4f}..{lo:.4f} (consistent: {hi < lo})")
thr = (lo + hi) / 2

# ------------------------------------------------------------------ 1. zero shares
seg = tr3.segment.fillna(-1).astype(int).values
seg_t = te3.segment.fillna(-1).astype(int).values
sub = pd.read_csv("submission_v8_try.csv")
ft = sub.task3.values
names_seg = {-1: "missing", 0: "0", 1: "1", 2: "2", 3: "3 (whales)"}
rows = []
for s in [0, 1, 2, 3, -1]:
    m, mt = seg == s, seg_t == s
    rows.append([names_seg[s], m.sum(), (y[m] == 0).mean(), (f[m] == 0).mean(), mt.sum(), (ft[mt] == 0).mean(),
                 y[m].sum() / y.sum()])
rows.append(["all", len(y), (y == 0).mean(), (f == 0).mean(), len(ft), (ft == 0).mean(), 1.0])
print("\n1. Share of zeros: training labels vs our predictions")
print(pd.DataFrame(rows, columns=["segment", "train rows", "label = 0", "OOF pred = 0", "test rows", "test pred = 0",
                                  "share of all spending"]).round(3).to_string(index=False))

# ------------------------------------------------------------------ 2. where the error comes from
err = np.abs(y - f)
cells = {"true 0, pred 0": (y == 0) & (f == 0), "true >0, pred 0": (y > 0) & (f == 0),
         "true 0, pred >0": (y == 0) & (f > 0), "true >0, pred >0": (y > 0) & (f > 0)}
rows = []
for k, m in cells.items():
    rows.append([k, m.sum(), m.mean(), err[m].sum() / err.sum(), np.mean(y[m]), np.median(y[m]) if m.any() else 0])
print("\n2. Where the absolute error comes from (OOF)")
print(pd.DataFrame(rows, columns=["cell", "rows", "share of rows", "share of total error", "mean actual",
                                  "median actual"]).round(3).to_string(index=False))
z = f == 0
print(f"rows predicted 0: {z.sum():,}; of these {np.mean(y[z] == 0):.3f} really spent 0 "
      f"(0 is the best constant for a group when this is above 0.5)")

# ------------------------------------------------------------------ 3. calibration of the zero decision
bins = [0, .1, .2, .3, .4, .45, .5, .55, .6, .7, .8, .9, 1.0001]
b = pd.cut(p, bins, right=False)
d = pd.DataFrame({"P(y>0) bin": b, "y": y, "zero": y == 0, "pred0": f == 0, "err": err})
g = d.groupby("P(y>0) bin", observed=True).agg(rows=("y", "size"), actual_zero_share=("zero", "mean"),
                                               pred_zero_share=("pred0", "mean"), mean_actual=("y", "mean"),
                                               error_share=("err", "sum"))
g["error_share"] /= err.sum()
g["net rows that favour 0 (zeros - spenders)"] = d.groupby("P(y>0) bin", observed=True).apply(
    lambda q: int(q.zero.sum() - (~q.zero).sum()))
print(f"\n3. Zero decision vs the stage-1 probability (gate threshold on P(y>0) = {thr:.3f})")
print(g.round(3).to_string())

# ------------------------------------------------------------------ 4. counterfactuals
print("\n4. What if we predicted fewer zeros? (OOF score; v8 = %.5f)" % s_v8)
res = []
s, k = best_scale(blend)
res.append(["no zero gate (blend only; members still give some zeros)", (blend == 0).mean(), s])
pos_any = blend * k0
for a_ in [0.05, 0.1, 0.25, 0.5, 1.0]:
    pred = np.where(gate == 1, f, a_ * pos_any)
    res.append([f"gated rows get {a_:.2f} x blend instead of 0", (pred == 0).mean(), nmae(y, pred)])
for c in [1, 10, 50, 200]:
    pred = np.where(f == 0, c, f)
    res.append([f"every zero replaced by {c} THB", 0.0, nmae(y, pred)])
for t in [0.30, 0.35, 0.40, 0.45, thr, 0.55, 0.60, 0.65, 0.70]:
    gg = (p > t).astype(float)
    s, k = best_scale(blend * gg)
    res.append([f"gate threshold P(y>0) > {t:.3f}" + ("  (v8)" if t == thr else ""), (blend * gg == 0).mean(), s])
for s_ in [0, 1, 2, 3]:
    pred = np.where((seg == s_) & (f == 0), pos_any, f)
    res.append([f"remove the zeros of segment {s_} only", (pred == 0).mean(), nmae(y, pred)])
for nm_ in ["ratio[hist] cat mixture-median", "ratio[hist] cat hard gate", "ratio[hist] cat E[y] + smearing"]:
    col = P[:, names.index(nm_)]
    res.append([f"single member: {nm_}", (col == 0).mean(), nmae(y, col)])
pred = np.where(y == 0, 0, pos_any)
res.append(["oracle: zero exactly on the rows that really spent 0 (not achievable)", (pred == 0).mean(), nmae(y, pred)])
R = pd.DataFrame(res, columns=["variant", "share predicted 0", "OOF score"])
R["delta vs v8"] = R["OOF score"] - s_v8
print(R.round(5).to_string(index=False))
