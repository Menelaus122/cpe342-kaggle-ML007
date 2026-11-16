"""Round 30 (Jackpot loop), Task 5: small groups isolated in a mutual k-NN graph (label-free; Task 5 has only test.csv, by design).

    python fa_r30_t5_knn_graph.py

The 1-D gap scan and the 2-D pocket scan look at one or two columns at a time. A group that is unusual only jointly in many columns
would show as a small connected component of a mutual k-nearest-neighbour graph (an edge only if each account is among the other's
k = 15 nearest), cut off from the main body. Space: the pipeline's detector matrix (task5.detector_columns / detector_matrix,
robust-scaled, clipped at 50). Null: the same graph on column-shuffled data (every marginal kept, every joint structure destroyed).
Report: component sizes; components of 20-150 accounts, their overlap with the 2,148 flags, and their most deviating columns;
the same counts on the null. A new group is only worth considering if it lies outside the flags and has no counterpart in the null.
"""
import time

import numpy as np
import pandas as pd
import psutil
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

import common  # noqa: F401  (chdir to the repo root)
from src import task5  # noqa: E402

t0 = time.time()
df = task5.load()
X, _ = task5.build_features(df)
cols = task5.detector_columns(X)
Z = task5.detector_matrix(X, cols, "robust").astype(np.float32)
_, gm = task5.gap_clusters(df, X)
gap = gm.any(axis=1).values
scan, noise = task5.dependence_pockets(X, ~gap)
pocket, _ = task5.pocket_detector(X, gap, sorted(set(scan[scan.z > noise].a) | set(scan[scan.z > noise].b)))
flag = gap | pocket
print(f"{Z.shape[0]} accounts x {Z.shape[1]} detector columns; flags {flag.sum()} (gap {gap.sum()}, pocket {pocket.sum()}) [{time.time() - t0:.0f}s]", flush=True)


def mutual_components(M, k=15):
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", n_jobs=4).fit(M)
    _, idx = nn.kneighbors(M)
    n = len(M)
    rows = np.repeat(np.arange(n), k)
    A = csr_matrix((np.ones(n * k, np.int8), (rows, idx[:, 1:].ravel())), shape=(n, n))
    mutual = A.multiply(A.T)
    ncomp, lab = connected_components(mutual, directed=False)
    return lab, np.bincount(lab)


lab, sizes = mutual_components(Z)
print(f"real data: {len(sizes)} components; largest {sizes.max()}; size distribution: "
      + ", ".join(f"{a}-{b}: {np.sum((sizes >= a) & (sizes <= b))}" for a, b in [(1, 1), (2, 4), (5, 19), (20, 150), (151, 10 ** 6)])
      + f" [{time.time() - t0:.0f}s, peak {psutil.Process().memory_info().peak_wset / 2**20:.0f} MB]", flush=True)
mid = [c for c in np.where((sizes >= 20) & (sizes <= 150))[0]]
for c in sorted(mid, key=lambda c: -sizes[c]):
    m = lab == c
    dev = pd.Series(np.abs(np.median(Z[m], 0)), index=cols).sort_values(ascending=False)
    print(f"  component {c}: {m.sum()} accounts; flagged {flag[m].mean():.0%} (gap {gap[m].mean():.0%}, pocket {pocket[m].mean():.0%}); "
          f"largest |median robust z|: " + ", ".join(f"{k_} {v:.1f}" for k_, v in dev.head(4).items()))
rng = np.random.default_rng(0)
null_counts = []
for r in range(3):
    Zs = Z.copy()
    for j in range(Zs.shape[1]):
        Zs[:, j] = Zs[rng.permutation(len(Zs)), j]
    _, s_null = mutual_components(Zs)
    null_counts.append(int(np.sum((s_null >= 20) & (s_null <= 150))))
    print(f"null {r}: {len(s_null)} components, largest {s_null.max()}, 20-150: {null_counts[-1]}, 5-19: {np.sum((s_null >= 5) & (s_null <= 19))} "
          f"[{time.time() - t0:.0f}s]", flush=True)
new = [c for c in mid if flag[lab == c].mean() < 0.5]
print(f"\ncomponents of 20-150 accounts: real {len(mid)}, null {null_counts}; of the real ones mostly outside the flags: {len(new)} "
      f"({sum(int((lab == c).sum()) for c in new)} accounts)")
