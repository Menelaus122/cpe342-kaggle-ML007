"""Task 2 player_id leak, deeper look: exact blocks, what the off-block rows are, and what id + features gives.

Usage: python q_t2_id_deep.py <path-to-original-train.csv>   (original row order: the cached OOF preds are
aligned to it by position).
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
import common  # noqa: E402  (sets cwd/sys.path)
from common import stack_multi, T2_MODELS  # noqa: E402
from src import task2  # noqa: E402
from src.metrics import macro_f1, optimize_class_weights  # noqa: E402
from src.modeling import make_folds  # noqa: E402
from src.utils import load_preds  # noqa: E402

tr = pd.read_csv(sys.argv[1])
te = pd.read_csv("02_multiclass_classification/test.csv")
X, y, Xt = task2.build(tr, te)
folds = make_folds(y)
K = 4
idn = tr.player_id.str[1:].astype(int).values
te_idn = te.player_id.str[1:].astype(int).values

# sanity: the cached OOF really are aligned with this row order
base = [load_preds(n) for n in T2_MODELS]
so, st = stack_multi([d["oof"] for d in base], [d["test"] for d in base], y, folds)
w, s_base = optimize_class_weights(y, so)
print(f"baseline features-only stack (reproduces notebook): macro-F1 {s_base:.4f}")

# ------------------------------------------------------------------ 1. exact block boundaries
# best monotone step function 0|1|2|3 over sorted ids (exhaustive over boundaries via cumulative counts)
o = np.argsort(idn); ys, ids = y[o], idn[o]
C = np.vstack([np.r_[0, np.cumsum(ys == k)] for k in range(K)])  # C[k, i] = #class k among first i rows
n = len(ys)
best = (-1, None)
# b1 < b2 < b3 positions; search coarse-to-fine to keep it cheap
def agree(b1, b2, b3):
    return C[0, b1] + (C[1, b2] - C[1, b1]) + (C[2, b3] - C[2, b2]) + (C[3, n] - C[3, b3])
grid = np.arange(0, n + 1, 200)
for b1 in grid:
    for b2 in grid[grid >= b1]:
        g3 = grid[grid >= b2]
        a = C[0, b1] + C[1, b2] - C[1, b1] - C[2, b2] + C[2, g3] + C[3, n] - C[3, g3]
        j = int(np.argmax(a))
        if a[j] > best[0]:
            best = (a[j], (b1, b2, int(g3[j])))
b = list(best[1])
for _ in range(3):  # refine each boundary at full resolution
    for i in range(3):
        lo = b[i - 1] if i else 0; hi = b[i + 1] if i < 2 else n
        cand = np.arange(max(lo, b[i] - 400), min(hi, b[i] + 400) + 1)
        sc = [agree(*(b[:i] + [c] + b[i + 1:])) for c in cand]
        b[i] = int(cand[int(np.argmax(sc))])
cut_ids = [int((ids[p - 1] + ids[p]) / 2) for p in b]
print(f"\nblock boundaries (ids): 0 | {cut_ids[0]} | {cut_ids[1]} | {cut_ids[2]} | {ids.max()}")
print(f"  (prior x 130,000 would put them at {np.cumsum(np.bincount(y) / len(y))[:3] * 130000})")
blk = np.searchsorted(cut_ids, idn, side="right")          # block class of every train row
blk_te = np.searchsorted(cut_ids, te_idn, side="right")
print(f"rows whose label = block class: {(blk == y).mean():.4f}; block-only macro-F1 {macro_f1(y, blk):.4f}")
print("label x block (rows = block):\n", pd.crosstab(blk, y).to_string())
print("test rows per block:", np.bincount(blk_te), " train rows per block:", np.bincount(blk))

# are the off-block rows spread evenly or clustered inside a block?
off = blk != y
for k in range(K):
    m = blk == k
    q = pd.qcut(idn[m], 5, labels=False)
    print(f"  block {k}: off-block share by id quintile {np.round(pd.Series(off[m]).groupby(q).mean().values, 3)}")
gaps = np.diff(np.sort(idn[off]))
print(f"off-block rows: {off.sum():,}; their id gaps look random (mean {gaps.mean():.1f}, sd {gaps.std():.1f}; "
      f"geometric would have sd ~ mean)")

# ------------------------------------------------------------------ 2. what are the off-block rows?
pred = np.argmax(so * w, axis=1)
print("\nfeatures-only model on in-block vs off-block rows:")
for name, m in [("in-block", ~off), ("off-block", off)]:
    print(f"  {name:9s} ({m.sum():6,}): model agrees with label {(pred[m] == y[m]).mean():.3f}, "
          f"with block class {(pred[m] == blk[m]).mean():.3f}; "
          f"mean P(label) {so[m, :][np.arange(m.sum()), y[m]].mean():.3f}, "
          f"mean P(block) {so[m, :][np.arange(m.sum()), blk[m]].mean():.3f}")
# per-feature view: do off-block rows labelled k look like class k or like their block?
key = ["total_spending_thb", "avg_monthly_spending", "play_frequency", "total_playtime_hours", "vip_tier",
       "competitive_rank", "friend_count"]
key = [c for c in key if c in X.columns]
rows = []
for k in range(K):
    for j in range(K):
        m = (y == k) & (blk == j)
        if m.sum() >= 30:
            rows.append({"label": k, "block": j, "n": int(m.sum()), **{c: X.loc[m, c].median() for c in key}})
print("\nmedian features by (label, block): off-block rows follow their LABEL if they match the diagonal rows "
      "of the same label\n", pd.DataFrame(rows).round(1).to_string(index=False))

# ------------------------------------------------------------------ 3. id + features
print("\nOOF macro-F1 when the block is added to the features-only stack:")
for eps in (0.003, 0.01, 0.03):
    # P(label | block) from the training folds, as a log-feature next to the model log-probs
    oofs = [d["oof"] for d in base]
    pb = np.zeros((len(y), K))
    for a, b_ in folds:
        T = pd.crosstab(blk[a], y[a]).reindex(index=range(K), columns=range(K), fill_value=0).values + eps
        pb[b_] = (T / T.sum(1, keepdims=True))[blk[b_]]
    T = pd.crosstab(blk, y).reindex(index=range(K), columns=range(K), fill_value=0).values + eps
    pb_te = (T / T.sum(1, keepdims=True))[blk_te]
    so2, _ = stack_multi(oofs + [pb], [d["test"] for d in base] + [pb_te], y, folds)
    w2, s2 = optimize_class_weights(y, so2)
    print(f"  stack(features + P(label|block), eps={eps}): {s2:.4f}")
    break
# does the feature model help *inside* the off-block group? (could we spot them?)
p_off = so[np.arange(len(y)), blk]           # model prob of the block class
print(f"AUC of (1 - P_model(block class)) for finding off-block rows: {roc_auc_score(off, 1 - p_off):.4f}")
