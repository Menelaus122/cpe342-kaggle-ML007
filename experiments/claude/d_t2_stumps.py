"""Task 2: is the signal additive? LightGBM stumps (a GAM) vs deep trees on the same features."""
import time
from common import *
from src.modeling import run_cv

tr, te, X, y, Xt, folds = load_t2()
ll = lambda p: -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))
for n in ["t2_base_raw", "t2_base_fe", "t2_base_fe_te", "t2_lgb"]:
    d = load_preds(n); print(f"{n:15s} argmax {macro_f1(y, d['oof'].argmax(1)):.4f} logloss {ll(d['oof']):.4f}")
raw2 = [c for c in X.columns if c in tr.columns]
# Whale within-class pairs
num = [c for c in tr.columns if pd.api.types.is_numeric_dtype(tr[c]) and c != "segment"]
C3 = tr.loc[y == 3, num].corr(method="spearman")
print(C3.where(np.triu(np.ones(C3.shape), 1).astype(bool)).stack().abs().sort_values(ascending=False).head(8).round(3).to_string())

for name, feats, params in [
    ("stumps raw", raw2, dict(num_leaves=2, learning_rate=0.1, min_child_samples=50, colsample_bytree=1.0, subsample=0.8)),
    ("depth2 raw", raw2, dict(num_leaves=3, max_depth=2, learning_rate=0.1, min_child_samples=50, colsample_bytree=1.0)),
    ("stumps FE", list(X.columns), dict(num_leaves=2, learning_rate=0.1, min_child_samples=50, colsample_bytree=1.0, subsample=0.8)),
]:
    t0 = time.time()
    r = run_cv("lgb", "multiclass", params, X[feats], y, Xt[feats], folds, n_classes=4, name=name, verbose=False)
    w, s = optimize_class_weights(y, r["oof"])
    print(f"{name:12s} argmax {macro_f1(y, r['oof'].argmax(1)):.4f} +w {s:.4f} logloss {ll(r['oof']):.4f} iters {np.mean(r['best_iters']):.0f} [{time.time()-t0:.0f}s]")
    np.savez_compressed(OUT / f"t2_{name.replace(' ', '_')}.npz", oof=r["oof"], test=r["test"])
