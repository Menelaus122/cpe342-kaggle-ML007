import time
from common import *
from src.modeling import fit_predict
X3, y3, X3t, folds3, m3, g_oof, g_te = t3_pool()
tr, te = task3.load()
P = np.column_stack([v[0] for v in m3.values()])
s3, pred = t3_blend_score(y3, folds3, P, g_oof)
seg = tr.segment.fillna(-1).values
w = seg == 3
print(f"baseline score {s3:.5f}; whale abs err {np.abs(y3[w] - pred[w]).sum():,.0f}")
def whale_cv(params, target="log"):
    oof = np.full(len(y3), np.nan)
    for a, b in folds3:
        a, b = a[w[a]], b[w[b]]
        yt = np.log1p(y3) if target == "log" else y3
        _, (pv,), _ = fit_predict("lgb", "regression", params, X3.iloc[a], yt[a], X3.iloc[b], yt[b], [X3.iloc[b]])
        oof[b] = np.expm1(pv) if target == "log" else pv
    return np.clip(oof, 0, None)
for name, params, tgt in [
    ("whale-only L1 log", dict(objective="regression_l1", learning_rate=0.02, num_leaves=15, min_child_samples=50, colsample_bytree=0.8), "log"),
    ("whale-only L2 log linear_tree", dict(objective="regression", learning_rate=0.02, num_leaves=7, min_child_samples=200, linear_tree=True, linear_lambda=1.0, colsample_bytree=0.8), "log"),
    ("whale-only L1 raw", dict(objective="regression_l1", learning_rate=0.02, num_leaves=15, min_child_samples=50, colsample_bytree=0.8), "raw"),
]:
    t0 = time.time()
    o = whale_cv(params, tgt)
    for mix in [1.0, 0.5]:
        new = pred.copy(); new[w] = mix * o[w] + (1 - mix) * pred[w]
        print(f"{name:32s} mix {mix:.1f}: whale abs err {np.abs(y3[w] - new[w]).sum():,.0f}  total score {nmae(y3, new):.5f} [{time.time()-t0:.0f}s]")
