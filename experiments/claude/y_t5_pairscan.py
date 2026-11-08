"""Task 5: blind 2-D cluster scan. Can a feature-only method find the 259 accounts we miss?

The six gap families are removed first. For every pair of the 179 features a 2-component Gaussian mixture is
fitted (robust-scaled, clipped, 6,000-account subsample). A pair scores high when the smaller component is a
compact, well-separated group of 0.3-4% of accounts: score = distance between the two means in units of the
pooled covariance. Heavy tails give a wide second component and a small score. The id is used only
afterwards, to check what the top pairs found.
"""
import itertools
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")
import common  # noqa: E402
from src import task5  # noqa: E402

t5 = task5.load()
X5, _ = task5.build_features(t5)
gc_table, gc_masks = task5.gap_clusters(t5, X5)
fam = gc_masks.any(axis=1).values
idn = t5.player_id.str[1:].astype(int).values
target = (~fam) & (idn >= 23819)            # the 259, used ONLY for the evaluation below

cols = [c for c in X5.columns if pd.api.types.is_numeric_dtype(X5[c]) and X5[c].nunique() > 10]
R = X5.loc[~fam, cols]
med, iqr = R.median(), (R.quantile(0.75) - R.quantile(0.25)).replace(0, np.nan)
Z = ((X5[cols] - med) / iqr).clip(-10, 10)
Z = Z.loc[:, iqr.notna()]
cols = list(Z.columns)
rest = np.where(~fam)[0]
rng = np.random.default_rng(0)
sub = rng.choice(rest, 6000, replace=False)


def score_pair(A):
    A = A[np.isfinite(A).all(1)]
    g = GaussianMixture(2, covariance_type="full", random_state=0, max_iter=200).fit(A)
    s = int(np.argmin(g.weights_))
    w = g.weights_[s]
    if not 0.003 <= w <= 0.04:
        return w, 0.0, g
    S = (g.covariances_[0] + g.covariances_[1]) / 2
    d = g.means_[0] - g.means_[1]
    return w, float(np.sqrt(d @ np.linalg.solve(S, d))), g


t0 = time.time()
rows = []
Zs = Z.iloc[sub].to_numpy()
for i, j in itertools.combinations(range(len(cols)), 2):
    w, sc, _ = score_pair(Zs[:, [i, j]])
    rows.append((sc, w, cols[i], cols[j]))
print(f"scanned {len(rows):,} pairs of {len(cols)} features in {time.time() - t0:.0f}s")
res = pd.DataFrame(rows, columns=["score", "small_weight", "a", "b"]).sort_values("score", ascending=False)
res.to_csv(common.OUT / "t5_pairscan.csv", index=False)

print("\ntop 15 pairs, refitted on all non-family accounts; 'caught' = members of the small component that are"
      " among the 259 (id check, evaluation only):")
Zall = Z.iloc[rest].to_numpy()
for _, r in res.head(15).iterrows():
    A = Zall[:, [cols.index(r.a), cols.index(r.b)]]
    ok = np.isfinite(A).all(1)
    g = GaussianMixture(2, covariance_type="full", random_state=0, max_iter=300).fit(A[ok])
    s = int(np.argmin(g.weights_))
    post = np.zeros(len(rest)); post[ok] = g.predict_proba(A[ok])[:, s]
    members = post > 0.5
    tgt = target[rest]
    print(f"  {r.score:5.2f}  w={r.small_weight:.3%}  {r.a} x {r.b}: small component {members.sum():4d} accounts, "
          f"{int((members & tgt).sum()):3d} of 259 caught, precision {(members & tgt).sum() / max(members.sum(), 1):.2f}")
