from common import *
from sklearn.metrics import roc_auc_score
tr, te, X, y, Xt, folds, n_unl = load_t1()
ms = {n: load_preds(n) for n in T1_MODELS}
s, thr, so, st = t1_score([d["oof"] for d in ms.values()], [d["test"] for d in ms.values()], y, folds)
raw = [c for c in tr.columns if c not in ("player_id", "is_cheater")]
C = tr.loc[y == 0, raw].corr(method="spearman")
print(C.where(np.triu(np.ones(C.shape), 1).astype(bool)).stack().abs().sort_values(ascending=False).head(6).round(3).to_string())
mid = (so >= 0.05) & (so < 0.8)
print(f"middle rows {mid.sum()} cheater rate {y[mid].mean():.3f} stack AUC within middle {roc_auc_score(y[mid], so[mid]):.4f}")
grp = np.where(y == 0, "legit", np.where(so >= 0.95, "cheat-obvious", np.where(so < 0.8, "cheat-hidden", "cheat-mid")))
legit_mid = (y == 0) & mid
med = pd.DataFrame({g: tr.loc[grp == g, raw].median() for g in ["legit", "cheat-hidden", "cheat-obvious"]})
med["legit-middle"] = tr.loc[legit_mid, raw].median()
med["cheat-hidden"] = tr.loc[(grp == "cheat-hidden") & mid, raw].median()
print(med[["legit", "legit-middle", "cheat-hidden", "cheat-obvious"]].round(3).to_string())
print("group sizes", pd.Series(grp).value_counts().to_dict())
# reports_received distribution by class
print(pd.crosstab(tr.reports_received.clip(upper=10), y, normalize="columns").round(3).T.to_string())
