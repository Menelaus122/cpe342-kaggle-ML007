from common import *
from sklearn.metrics import roc_auc_score, average_precision_score
tr, te, X, y, Xt, folds, n_unl = load_t1()
print(f"train labelled {len(y)}, unlabelled rows dropped {n_unl}, test {len(te)}, prevalence {y.mean():.4f}")
ms = {n: load_preds(n) for n in T1_MODELS}
for n, d in ms.items():
    t, s = optimize_threshold(y, d["oof"], 2.0)
    print(f"{n:8s} AUC {roc_auc_score(y, d['oof']):.4f} AP {average_precision_score(y, d['oof']):.4f} best F2 {s:.4f} @ {t:.3f}")
s, thr, so, st = t1_score([d["oof"] for d in ms.values()], [d["test"] for d in ms.values()], y, folds)
print(f"stack: AUC {roc_auc_score(y, so):.4f} best F2 {s:.4f} @ {thr:.3f}; F2 @ 0.190 {f2(y, (so >= 0.190).astype(int)):.4f}")
# confident errors (label-noise check)
for lo, hi in [(0, .02), (.02, .05), (.05, .2), (.2, .5), (.5, .8), (.8, .95), (.95, 1.01)]:
    m = (so >= lo) & (so < hi)
    print(f"  p in [{lo:.2f},{hi:.2f}) rows {m.sum():6d} cheater rate {y[m].mean():.4f}")
pred = (so >= 0.190).astype(int)
tp, fp, fn = ((pred == 1) & (y == 1)).sum(), ((pred == 1) & (y == 0)).sum(), ((pred == 0) & (y == 1)).sum()
print(f"@0.190: TP {tp} FP {fp} FN {fn} precision {tp / (tp + fp):.4f} recall {tp / (tp + fn):.4f}")
raw = [c for c in tr.columns if c not in ("player_id", "is_cheater")]
for k in (0, 1):
    C = tr.loc[y == k, raw].corr(method="spearman").values
    off = np.abs(C[np.triu_indices_from(C, 1)])
    print(f"class {k}: within-class |rho| mean {np.nanmean(off):.4f} max {np.nanmax(off):.4f}")
C = tr[raw].corr(method="spearman").values; off = np.abs(C[np.triu_indices_from(C, 1)])
print(f"pooled |rho| mean {np.nanmean(off):.4f} max {np.nanmax(off):.4f}")
mr = pd.DataFrame({c: [tr.loc[y == k, c].isna().mean() for k in (0, 1)] for c in raw}).T
print("missing-rate spread between classes, max:", (mr[1] - mr[0]).abs().max().round(4))
# univariate AUC (signal per feature)
ua = {c: roc_auc_score(y[tr[c].notna()], tr.loc[tr[c].notna(), c]) for c in raw}
print(pd.Series(ua).sub(0.5).abs().add(0.5).sort_values(ascending=False).round(3).to_string())
