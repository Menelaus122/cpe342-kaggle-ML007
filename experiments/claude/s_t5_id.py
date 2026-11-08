"""Task 5: our six families sit at the top of the id range. Is that block exactly the families?

Looks at the 259 unflagged accounts that sit inside the id range of the flagged ones (23,819..25,888).
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import common  # noqa: F401,E402
from src import task5  # noqa: E402

t5 = task5.load()
X5, dev5 = task5.build_features(t5)
gc_table, gc_masks = task5.gap_clusters(t5, X5)
fams = task5.merge_families(gc_masks)
idn = t5.player_id.str[1:].astype(int).values
n = len(idn)
fam_id = np.full(n, -1)
fam_names = []
for j, f in enumerate(sorted(fams, key=lambda f: -gc_masks[f].any(axis=1).sum())):
    m = gc_masks[f].any(axis=1).values
    fam_id[m & (fam_id == -1)] = j
    fam_names.append(f[0])
flag = fam_id >= 0
print(f"families: {flag.sum()} accounts; names {fam_names}")

# id layout of each family
o = np.argsort(idn)
print("\nid range per family:")
for j, nm in enumerate(fam_names):
    ids = np.sort(idn[fam_id == j])
    print(f"  {j} {nm:28s} n={len(ids):4d} ids {ids.min()}..{ids.max()} (span {ids.max() - ids.min() + 1})")

lo = idn[flag].min()
blk = idn >= lo
print(f"\nid block {lo}..{n - 1}: {blk.sum()} accounts = {blk.mean():.3%} of {n}; flagged inside {(flag & blk).sum()}, "
      f"unflagged inside {(~flag & blk).sum()}; flagged outside {(flag & ~blk).sum()}")
# where are the unflagged-in-block accounts along the id axis?
seq = pd.Series(np.where(flag[o], fam_id[o], -1), index=idn[o]).loc[lo:]
runs = (seq != seq.shift()).cumsum()
rl = seq.groupby(runs).agg(["first", "size"])
rl["start"] = seq.index.to_series().groupby(runs.values).min().values
print("\nruns along the id axis inside the block (value = family index, -1 = not flagged), runs >= 20 long:")
print(rl[rl["size"] >= 20].rename(columns={"first": "family"}).to_string(index=False))
print(f"number of runs: {len(rl)}; unflagged runs: {(rl['first'] == -1).sum()}, longest unflagged run {rl.loc[rl['first'] == -1, 'size'].max()}")

# how do the 259 look? compare with normal accounts (below the block) and with the families on key columns
miss = ~flag & blk
norm = ~blk
cols = [c for c in X5.columns if pd.api.types.is_numeric_dtype(X5[c])]
z = {}
for c in cols:
    a, b = X5.loc[norm, c], X5.loc[miss, c]
    s = a.std()
    if s and np.isfinite(s):
        z[c] = (b.median() - a.median()) / s
zz = pd.Series(z).dropna().sort_values(key=np.abs, ascending=False)
print("\nthe in-block unflagged accounts vs normal accounts: largest median shifts (in normal-sd units)")
print(zz.head(15).round(2).to_string())
from scipy import stats
ks = {c: stats.ks_2samp(X5.loc[norm, c].dropna(), X5.loc[miss, c].dropna()).pvalue for c in cols
      if X5.loc[miss, c].notna().sum() > 20}
ks = pd.Series(ks).sort_values()
print(f"\nKS tests in-block-unflagged vs normal: {(ks < 0.001 / len(ks)).sum()} of {len(ks)} columns differ "
      f"at Bonferroni 0.001; smallest p:")
print(ks.head(10).to_string())
pd.DataFrame({"player_id": t5.player_id, "id": idn, "family": fam_id, "in_block": blk}).to_csv(
    common.OUT / "t5_id_families.csv", index=False)
