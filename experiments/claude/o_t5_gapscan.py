"""Task 5: run the same label-free gap detector on the derived temporal features (period-4 change etc.)."""
from common import *
from src import task5
t5 = task5.load()
X5, dev_cols = task5.build_features(t5)
part = pd.read_csv(ROOT / "outputs" / "submission_parts" / "task5.csv")
assert (part.iloc[:, 0].values == t5[part.columns[0]].values).all() if part.columns[0] in t5 else True
flag5 = part["task5"].values.astype(bool)
gc_table, gc_masks = task5.gap_clusters(t5, X5)
gap5 = gc_masks.any(axis=1).values
print(f"rows {len(t5)}; current union flags {flag5.sum()}; raw-column families {gap5.sum()}")
Xd = X5.drop(columns=[c for c in X5.columns if c in task5.STATIC_COLS or c in ("n_flags", "n_missing")]).copy()
Xd[task5.ID_COL] = 0
tab, masks = task5.gap_clusters(Xd)
tab["new_vs_union"] = [int((masks[g].values & ~flag5).sum()) for g in tab.group]
tab["new_vs_families"] = [int((masks[g].values & ~gap5).sum()) for g in tab.group]
pd.set_option("display.width", 200)
print(tab[["group", "size", "expected_in_gap", "new_vs_families", "new_vs_union"]].round(1).to_string())
new = masks.any(axis=1).values & ~flag5
print(f"derived-feature groups: {masks.any(axis=1).sum()} accounts, of which NOT in current union: {new.sum()}")
np.save(OUT / "t5_derived_new.npy", new)
