from common import *
from sklearn.metrics import roc_auc_score
tr, te, X, y, Xt, folds, n_unl = load_t1()
ms = [load_preds(n) for n in T1_MODELS]
s, thr, so, st = t1_score([d["oof"] for d in ms], [d["test"] for d in ms], y, folds)
raw = [c for c in tr.columns if c not in ("player_id", "is_cheater")]
nm = tr[raw].isna().sum(1).values
print("NaN per row: train", nm.mean().round(2), "test", te[raw].isna().sum(1).mean().round(2))
print(pd.Series(nm).clip(upper=7).value_counts(normalize=True).sort_index().round(3).to_dict(), "(train)")
print(pd.Series(te[raw].isna().sum(1)).clip(upper=7).value_counts(normalize=True).sort_index().round(3).to_dict(), "(test)")
for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 99)]:
    m = (nm >= lo) & (nm < hi)
    t, sc = optimize_threshold(y[m], so[m], 2.0)
    print(f"NaN {lo}-{hi - 1}: rows {m.sum():6d} AUC {roc_auc_score(y[m], so[m]):.4f} best F2 {sc:.4f} @ {t:.3f}  F2@0.190 {f2(y[m], (so[m] >= .19).astype(int)):.4f}")
full = pd.read_csv(ROOT / "01_binary_classification" / "train.csv")
unl = full[full.is_cheater.isna()]
print("unlabelled rows:", len(unl), "NaN/row", unl[raw].isna().sum(1).mean().round(2))
