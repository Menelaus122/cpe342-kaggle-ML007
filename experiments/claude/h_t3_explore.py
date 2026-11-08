from common import *
from sklearn.metrics import roc_auc_score
X3, y3, X3t, folds3, m3, g_oof, g_te = t3_pool()
tr, te = task3.load()
P = np.column_stack([v[0] for v in m3.values()])
s3, pred = t3_blend_score(y3, folds3, P, g_oof)
h = load_preds("t3_hurdle")
err = np.abs(y3 - pred); tot = err.sum()
seg = tr.segment.fillna(-1).values
print(f"score {s3:.5f}")
for s in [-1, 0, 1, 2, 3]:
    m = seg == s
    print(f"seg {s:2d}: rows {m.mean():.3f} sum y share {y3[m].sum() / y3.sum():.3f} err share {err[m].sum() / tot:.3f} "
          f"zero rate {(y3[m] == 0).mean():.3f} pred0 rate {(pred[m] == 0).mean():.3f} |err|/y {err[m].sum() / max(y3[m].sum(), 1):.3f}")
w = seg == 3
print("whales: AUC of P(y>0)", round(roc_auc_score(y3[w] > 0, h["p_oof"][w]), 4), " p_oof quantiles", np.quantile(h["p_oof"][w], [.01, .05, .5]).round(3))
zw = w & (y3 == 0)
print(f"whale zeros: {zw.sum()} rows, error share {err[zw].sum() / tot:.3f}")
pw = w & (y3 > 0)
r = np.log1p(y3[pw]) - np.log1p(np.maximum(pred[pw], 0))
print(f"whale spenders: log residual median {np.median(r):+.3f} MAD {np.median(np.abs(r - np.median(r))):.3f}; err share {err[pw].sum() / tot:.3f}")
# segment-missing rows: how many are whales-like (by y)?
mm = seg == -1
print("seg-missing rows: y quantiles", np.quantile(y3[mm], [.5, .75, .85, .9, .95]).round(0))
# relationships among whale spenders
d = tr.loc[pw].copy(); d["ly"] = np.log1p(d.spending_30d)
cors = d.drop(columns=["player_id"]).corr(method="spearman", numeric_only=True)["ly"].drop(["ly", "spending_30d"]).sort_values(key=abs, ascending=False)
print("whale spenders: spearman with log y\n", cors.head(12).round(3).to_string())
# non-whale
d = tr.loc[(seg >= 0) & (seg < 3) & (y3 > 0)].copy(); d["ly"] = np.log1p(d.spending_30d)
cors = d.drop(columns=["player_id"]).corr(method="spearman", numeric_only=True)["ly"].drop(["ly", "spending_30d"]).sort_values(key=abs, ascending=False)
print("non-whale spenders: spearman with log y\n", cors.head(8).round(3).to_string())
