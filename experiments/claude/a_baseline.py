"""Reproduce the notebook's Task 2 / Task 3 OOF scores and do the error analysis."""
from common import *
from sklearn.metrics import confusion_matrix, f1_score, classification_report

# ---------------- Task 2
tr2, te2, X2, y2, X2t, folds2 = load_t2()
s2, so2, st2, w2 = t2_baseline(y2, folds2)
pred = apply_class_weights(so2, w2)
print(f"T2 baseline macro-F1 {s2:.4f}  weights {np.round(w2,3)}")
print("class counts", np.bincount(y2), "test pred dist", np.bincount(apply_class_weights(st2, w2), minlength=4) / len(st2))
print(confusion_matrix(y2, pred))
print(classification_report(y2, pred, digits=4))
for n in T2_MODELS:
    d = load_preds(n); print(n, round(macro_f1(y2, d["oof"].argmax(1)), 4))
# how confident are errors?
pmax = so2.max(1)
for lo, hi in [(0, .4), (.4, .5), (.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01)]:
    m = (pmax >= lo) & (pmax < hi)
    print(f"  pmax [{lo:.1f},{hi:.1f}) share {m.mean():.3f} acc {(pred[m] == y2[m]).mean():.3f}")
print("missing per row train", tr2.isna().sum(1).mean(), "test", te2.isna().sum(1).mean())

# ---------------- Task 3
X3, y3, X3t, folds3, m3, g_oof, g_te = t3_pool()
P = np.column_stack([v[0] for v in m3.values()])
s3, pred3 = t3_blend_score(y3, folds3, P, g_oof)
print(f"\nT3 baseline NMAE score {s3:.5f}")
for k, v in m3.items():
    print(f"  {k:25s} {nmae(y3, v[0]):.5f}")
err = np.abs(y3 - pred3)
tot = err.sum()
print(f"sum y {y3.sum():,.0f}  sum|err| {tot:,.0f}  ratio {tot / y3.sum():.4f}")
print("zero share", (y3 == 0).mean(), "pred zero share", (pred3 == 0).mean())
z, pz = y3 == 0, pred3 == 0
for nm, m in [("y=0,pred=0", z & pz), ("y=0,pred>0", z & ~pz), ("y>0,pred=0", ~z & pz), ("y>0,pred>0", ~z & ~pz)]:
    print(f"  {nm:12s} rows {m.mean():.3f}  error share {err[m].sum() / tot:.3f}")
q = pd.qcut(y3[~z], 10, labels=False)
yy, ee = y3[~z], err[~z]
print("error share by actual decile (spenders):")
for d in range(10):
    mm = q == d
    print(f"  d{d} y in [{yy[mm].min():,.0f},{yy[mm].max():,.0f}] err share {ee[mm].sum() / tot:.3f}  mean |err|/y {np.mean(ee[mm] / yy[mm]):.3f}")
print(pd.Series(y3).describe(percentiles=[.5, .75, .9, .99, .999]).round(1).to_string())
