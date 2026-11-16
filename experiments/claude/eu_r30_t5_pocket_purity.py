"""Round 30 (Jackpot loop), Task 5: purity of the dependence pocket's weakest flags (label-free; Task 5 has only test.csv).

    python eu_r30_t5_pocket_purity.py

The pocket detector ranks the accounts outside the gap families by min(rank of accuracy_std_level, rank of kda_std_level) and flags
the top k* = 339, k* maximising the expected F3 under a column-shuffle permutation null (est_TP(k) = k - mean null count above the
k-th score). Questions: (1) how pure is the weakest 5-10% of the 339 (marginal precision of the last 20 / 30 / 50 accounts)? (2) how
does the expected F3 change if the last 20-30 are dropped? (3) how stable is k* across permutation seeds (5 seeds, n_perm 300)?
"""
import numpy as np

import common  # noqa: F401  (chdir to the repo root)
from src import task5  # noqa: E402

df = task5.load()
X, _ = task5.build_features(df)
_, gm = task5.gap_clusters(df, X)
gap = gm.any(axis=1).values
feats = ["accuracy_std_level", "kda_std_level"]
res = []
for seed in range(5):
    flags, info = task5.pocket_detector(X, gap, feats, n_perm=300, seed=seed)
    res.append((seed, info))
    c = info["curve"].set_index("k")
    k = info["k"]
    line = (f"seed {seed}: k* {k}, group size {info['group_size']:.0f}, est. precision of the flags {info['precision']:.3f}, expected F3 "
            f"{info['expected_f']:.4f}")
    for drop in (20, 30, 50):
        kk = k - drop
        marg = (c.est_tp[k] - c.est_tp[kk]) / drop
        line += f" | drop last {drop}: F3 {c.expected_f[kk]:.4f} ({c.expected_f[kk] - info['expected_f']:+.4f}), their marginal precision {marg:.2f}"
    print(line, flush=True)
ks = [r[1]["k"] for r in res]
print(f"\nk* over 5 permutation seeds: {ks} (mean {np.mean(ks):.0f}, SD {np.std(ks):.1f})")
c = res[0][1]["curve"].set_index("k")
k = res[0][1]["k"]
print("expected F3 around k* (seed 0): " + "  ".join(f"k={kk}: {c.expected_f[kk]:.4f}" for kk in (k - 60, k - 40, k - 20, k, k + 20, k + 40, k + 60) if kk in c.index))
print(f"break-even marginal precision for F3 at this operating point is about F / (1 + beta^2) = {res[0][1]['expected_f'] / 10:.3f}: "
      "an added account raises the expected F3 if its probability of being anomalous exceeds that")
