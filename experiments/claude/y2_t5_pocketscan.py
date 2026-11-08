"""Task 5: blind "dependence pocket" scan.

Normal accounts draw their features (nearly) independently, so on a 20x20 grid of quantile bins every cell of a
feature pair should hold n/400 accounts. An injected group that is high on two features at once leaves a
pocket of excess accounts. For all pairs of features (six gap families removed; pairs built from the same base
feature skipped) the largest cell z-score = (observed - expected) / sqrt(expected) is recorded.
The id is used only afterwards, to grade what the top pockets contain.
"""
import itertools
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import common  # noqa: E402
from src import task5  # noqa: E402

G = 20
t5 = task5.load()
X5, _ = task5.build_features(t5)
_, gc_masks = task5.gap_clusters(t5, X5)
fam = gc_masks.any(axis=1).values
idn = t5.player_id.str[1:].astype(int).values
target = ((~fam) & (idn >= 23819))[~fam]          # the 259 among the remaining accounts (grading only)

cols = [c for c in X5.columns if pd.api.types.is_numeric_dtype(X5[c]) and X5[c].nunique() > 50]
R = X5.loc[~fam, cols]
Q = np.full(R.shape, -1, dtype=int)
for j, c in enumerate(cols):
    v = R[c]
    ok = v.notna().values
    Q[ok, j] = np.minimum((v[ok].rank(pct=True, method="first").values * G).astype(int), G - 1)


def base(c):
    c = re.sub(r"_(level|cv|burst|maxjump|dev4|selfz4)$", "", c)
    return "geo" if c.startswith("geo") else re.sub(r"_(mean|std)$", "", c)


rows = []
for i, j in itertools.combinations(range(len(cols)), 2):
    if base(cols[i]) == base(cols[j]):
        continue
    ok = (Q[:, i] >= 0) & (Q[:, j] >= 0)
    cnt = np.bincount(Q[ok, i] * G + Q[ok, j], minlength=G * G).reshape(G, G).astype(float)
    exp = np.outer(cnt.sum(1), cnt.sum(0)) / cnt.sum()
    z = (cnt - exp) / np.sqrt(np.maximum(exp, 1))
    k = int(np.argmax(z))
    rows.append((z.flat[k], cols[i], cols[j], k // G, k % G, cnt.flat[k], exp.flat[k]))
res = pd.DataFrame(rows, columns=["z", "a", "b", "bin_a", "bin_b", "obs", "exp"]).sort_values("z", ascending=False)
res.to_csv(common.OUT / "t5_pocketscan.csv", index=False)
print(f"scanned {len(res):,} pairs of {len(cols)} features; expected max z under independence ~ 4-5")
print("\ntop 15 pockets (grading: how many accounts in the pocket cell are among the 259):")
for _, r in res.head(15).iterrows():
    i, j = cols.index(r.a), cols.index(r.b)
    m = (Q[:, i] == r.bin_a) & (Q[:, j] == r.bin_b)
    print(f"  z={r.z:6.1f}  {r.a} [bin {r.bin_a}] x {r.b} [bin {r.bin_b}]: {int(r.obs)} accounts vs {r.exp:.0f} expected; "
          f"{int((m & target).sum())} of them in the 259")
print("\nz-score distribution over all pairs:", res.z.quantile([0.5, 0.9, 0.99, 0.999]).round(1).to_dict())
