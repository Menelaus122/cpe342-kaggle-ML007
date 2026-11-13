"""Round 23, Task 5: if we flagged 10% or 12% of accounts instead of 8.30%, whom would we add, and does anything say they are anomalies?

    python da_r23_t5_extend.py

Label-free (test features only). Starting from the v15 flags (2,148 = six gap families + the dependence pocket), three ways to extend:
  R1  ensemble: rank-mean of the six detectors (LOF excluded as in the pipeline; rank fusion, so Mahalanobis does not dominate)
  R2  pocket: the next accounts by the pocket detector's own score (min rank percentile over the pocket features). Its permutation null
      gives, for every cut-off, the expected number of normal accounts, so est. anomaly share of the added accounts =
      (est_TP(k* + added) - est_TP(k*)) / added
  R3  family edges: for each gap-separated group, the unflagged accounts closest to its gap on the bulk side
F3 rule (FINDINGS round 21): adding accounts raises F3 only if more than about F3 / 10 (~0.1) of them are anomalies.
For each extension: overlap between rankings, binary-flag rates, the column that makes each added account stand out, and whether added
accounts carry a family's signature (extreme, top 1% of normal accounts, in another column of the same family).
"""
import numpy as np
import pandas as pd

import common  # noqa: F401  (chdir to the repo root)
from src import task5  # noqa: E402

t5 = task5.load()
n = len(t5)
X5, _ = task5.build_features(t5)
cols5 = task5.detector_columns(X5)
Z5 = task5.detector_matrix(X5, cols5, "robust")
gc_table, gc_masks = task5.gap_clusters(t5, X5)
gap5 = gc_masks.any(axis=1).values
fams = task5.merge_families(gc_masks)
final = pd.read_csv("outputs/submission_parts/task5.csv").task5.values.astype(bool)
rest = np.where(~final)[0]
ADD = {"10%": int(0.10 * n) - int(final.sum()), "12%": int(0.12 * n) - int(final.sum())}
print(f"{n} accounts; v15 flags {final.sum()} ({final.mean():.2%}); to reach 10% add {ADD['10%']}, 12% add {ADD['12%']}")

# ---------------------------------------------------------------- R1 ensemble (rank mean)
DET = {k: f for k, f in task5.DETECTORS.items() if k != "LOF"}
R = pd.DataFrame({k: pd.Series(f(Z5, device="cpu") if k == "AutoEncoder" else f(Z5)).rank(pct=True).values for k, f in DET.items()})
ens = R.mean(axis=1).values
r1 = rest[np.argsort(-ens[rest])]

# ---------------------------------------------------------------- R2 pocket continuation, with its permutation-null estimate
scan5, noise5 = task5.dependence_pockets(X5, ~gap5)
feats = sorted(set(scan5[scan5.z > noise5].a) | set(scan5[scan5.z > noise5].b))
pocket, info = task5.pocket_detector(X5, gap5, feats)                       # pipeline settings: k* as in v15
assert (pocket == (final & ~gap5)).all(), "pocket differs from the v15 file"
_, info_long = task5.pocket_detector(X5, gap5, feats, max_k=3000 + max(ADD.values()))   # only to read est_TP further out
U = X5.loc[~gap5, feats].rank(pct=True).to_numpy()
with np.errstate(all="ignore"):
    S = np.nanmin(U, axis=1) + 1e-9 * np.nansum(U, axis=1)
S = np.where(np.isnan(S), -np.inf, S)
idx_nogap = np.where(~gap5)[0]
order = idx_nogap[np.argsort(-S)]
r2 = order[info["k"]:]
curve = info_long["curve"].set_index("k")
n_known, g_hat = int(gap5.sum()), info["group_size"]
exp_f = lambda k: 10 * (n_known + curve.loc[k, "est_tp"]) / (9 * (n_known + g_hat) + n_known + k)   # the pipeline's expected F3
print(f"\npocket features {feats}; cut-off k* = {info['k']} (est. members {info['est_tp']:.0f}, est. group size {g_hat:.0f}); "
      f"expected F3 at k* {exp_f(info['k']):.4f}")
for nm, a in ADD.items():
    k2 = info["k"] + a
    gain = curve.loc[k2, "est_tp"] - curve.loc[info["k"], "est_tp"]
    print(f"  pocket extended by {a} ({nm}): est. extra members {gain:.0f} -> est. anomaly share of the added {gain / a:.3f}; "
          f"expected F3 {exp_f(info['k']):.4f} -> {exp_f(k2):.4f}")

# ---------------------------------------------------------------- R3 family edges
edge_score = np.full(n, np.inf)
for _, g in gc_table.iterrows():
    x = (X5[g.column] if g.column in X5 else t5[g.column]).values.astype(float)
    bulk_spread = np.nanstd(x[~gap5])
    d = ((g.cut - x) if g.side == "high" else (x - g.cut)) / bulk_spread     # >= 0 on the bulk side, small = near the gap
    d = np.where(np.isnan(d) | (d < 0) | final, np.inf, d)
    edge_score = np.minimum(edge_score, d)
r3 = np.argsort(edge_score)
r3 = r3[np.isfinite(edge_score[r3])]

# family signature: for each family, its columns; an account "carries the signature" if it is beyond the top 1% of NORMAL accounts
# (or bottom 1%, following the group's side) in at least one of the family's columns
sig_cols = []
for fam in fams:
    for gname in fam:
        g = gc_table.set_index("group").loc[gname]
        x = (X5[g.column] if g.column in X5 else t5[g.column]).values.astype(float)
        q = np.nanquantile(x[~final], 0.99 if g.side == "high" else 0.01)
        sig_cols.append((g.column, (x > q) if g.side == "high" else (x < q)))
SIG = np.column_stack([m for _, m in sig_cols])

# ---------------------------------------------------------------- report
A = np.abs(Z5)
normal_ref = rest[np.argsort(-ens[rest])][3000:]                    # clearly unremarkable accounts, for reference rates


def describe(idx):
    top = pd.Series(np.array(cols5)[A[idx].argmax(1)]).value_counts(normalize=True).head(4).round(2).to_dict()
    flags = {c[:12]: round(float(np.nanmean(t5[c].values[idx])), 3) for c in task5.FLAG_COLS}
    return (f"median max|z| {np.median(A[idx].max(1)):.2f} | family signature in >=1 column {SIG[idx].any(1).mean():.3f}, in >=2 "
            f"{(SIG[idx].sum(1) >= 2).mean():.3f} | flags {flags} | stands out in {top}")


print(f"\nreference, unremarkable unflagged accounts: {describe(normal_ref)}")
print(f"reference, v15 flags:                       {describe(np.where(final)[0])}")
for nm, a in ADD.items():
    sets = {"R1 ensemble": r1[:a], "R2 pocket next": r2[:a], "R3 family edges": r3[:a]}
    print(f"\n== extend to {nm} (+{a} accounts)")
    for k, s in sets.items():
        print(f"  {k:16s}: {describe(s)}")
    ks = list(sets)
    for i in range(3):
        for j in range(i + 1, 3):
            print(f"  overlap {ks[i]} & {ks[j]}: {len(set(sets[ks[i]]) & set(sets[ks[j]]))}")
