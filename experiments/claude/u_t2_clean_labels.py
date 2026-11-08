"""Task 2 without the id: can a features-only model get near the id's 0.968?

1. Does any feature encode the row order (correlate with id *inside* a block)? That would be a hidden leak.
2. Train LightGBM on the clean segments (id block, train rows only) instead of the noisy labels.
   At test time the model sees features only. Evaluate OOF against the real (noisy) labels.

Usage: python u_t2_clean_labels.py <path-to-original-train.csv>
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats

import common  # noqa: F401
from common import P2_lgb
from src import task2
from src.metrics import macro_f1, optimize_class_weights
from src.modeling import make_folds, run_cv

tr = pd.read_csv(sys.argv[1])
te = pd.read_csv("02_multiclass_classification/test.csv")
X, y, Xt = task2.build(tr, te)
folds = make_folds(y)
idn = tr.player_id.str[1:].astype(int).values
blk = np.searchsorted([52000, 84500, 110500], idn, side="right")
K = 4

# ------------------------------------------------------------------ 1. hidden order inside features?
print("=== 1. does any feature follow the id inside a block? ===")
raw = tr.drop(columns=["player_id", "segment"])
num = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
worst = []
for k in range(K):
    m = blk == k
    for c in num:
        r = stats.spearmanr(idn[m], raw.loc[m, c], nan_policy="omit")[0]
        worst.append((abs(r), k, c, r))
worst.sort(reverse=True)
print(f"largest within-block |Spearman(id, feature)| over {len(num)} columns x 4 blocks "
      f"(noise level ~{2 / np.sqrt((blk == 3).sum()):.3f}):")
for a, k, c, r in worst[:6]:
    print(f"  block {k} {c:32s} {r:+.4f}")
cat = [c for c in raw.columns if c not in num]
pmin = min(stats.chi2_contingency(pd.crosstab(pd.qcut(idn[blk == k], 10, labels=False),
                                              raw.loc[blk == k, c].fillna("NA")))[1] for k in range(K) for c in cat)
print(f"categorical columns vs id decile inside blocks: smallest p = {pmin:.3f} over {len(cat) * K} tests")

# ------------------------------------------------------------------ 2. train on clean labels
print("\n=== 2. features-only LightGBM: noisy vs clean training labels (5-fold OOF, same folds) ===")
res = {}
for name, target in [("trained on given labels", y), ("trained on clean segments", blk)]:
    r = run_cv("lgb", "multiclass", dict(P2_lgb), X, target, Xt, folds, n_classes=K, verbose=False)
    w, s = optimize_class_weights(y, r["oof"])
    pred = np.argmax(r["oof"] * w, axis=1)
    res[name] = s
    print(f"  {name:26s}: macro-F1 vs given labels {s:.4f} | accuracy vs given labels {(pred == y).mean():.4f}, "
          f"vs clean segments {(pred == blk).mean():.4f} | macro-F1 vs clean {macro_f1(blk, np.argmax(r['oof'], 1)):.4f}")
print(f"\nfor reference: id block -> macro-F1 vs given labels {macro_f1(y, blk):.4f} "
      f"(the 3% flipped labels cap even a perfect model at about this)")
