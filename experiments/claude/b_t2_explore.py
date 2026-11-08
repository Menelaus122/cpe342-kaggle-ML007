from common import *
tr, te, X, y, Xt, folds = load_t2()
raw = [c for c in tr.columns if c not in ("player_id", "segment")]
num = [c for c in raw if pd.api.types.is_numeric_dtype(tr[c])]
cat = [c for c in raw if not pd.api.types.is_numeric_dtype(tr[c])]
print("n num", len(num), "n cat", len(cat))
print("missing rate per col (mean, min, max):", tr[raw].isna().mean().describe()[["mean", "min", "max"]].round(3).to_dict())
# within-class correlation (conditional independence check)
for k in range(4):
    C = tr.loc[y == k, num].corr(method="spearman").values
    off = C[np.triu_indices_from(C, 1)]
    print(f"class {k}: within-class |spearman| mean {np.nanmean(np.abs(off)):.3f} max {np.nanmax(np.abs(off)):.3f}")
C = tr[num].corr(method="spearman").values; off = C[np.triu_indices_from(C, 1)]
print(f"pooled: |spearman| mean {np.nanmean(np.abs(off)):.3f} max {np.nanmax(np.abs(off)):.3f}")
# top within-class correlated pairs in class 0
C0 = tr.loc[y == 0, num].corr(method="spearman")
pairs = C0.where(np.triu(np.ones(C0.shape), 1).astype(bool)).stack().abs().sort_values(ascending=False)
print(pairs.head(12).round(3).to_string())
# per-class medians / unique counts
desc = pd.DataFrame({c: [tr[c].nunique()] + [tr.loc[y == k, c].median() for k in range(4)] for c in num},
                    index=["nuniq", "Casual", "Grinder", "Social", "Whale"]).T
print(desc.round(2).to_string())
for c in cat:
    print(c, pd.crosstab(tr[c].fillna("NA"), y, normalize="index").round(2).to_dict("index"))
