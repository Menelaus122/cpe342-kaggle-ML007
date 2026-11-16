"""Round 30, Task 1: is there signal left after the v20 stacker? Residual boosting from the stacker's log-odds (training rows only).

    python eh2_r30_t1_residual.py

eh_r30_t1_forensics.py found the misses and false alarms spread over every sub-population (missing cells, account age, skill
quantiles) with no pocket of high error, and 64% of the misses within 0.5 log-odds of the cut-off. The direct test of "something the
level 2 cannot see" is to boost from its output: a LightGBM with init_score = the v20 level-2 log-odds (out of fold) and the raw
31 columns + missing-cell count as inputs, few shallow trees and a small learning rate (fixed before the run: 150 trees, lr 0.02,
15 leaves, 300 rows per leaf, 3 seeds averaged). It can only change predictions where the raw features explain the stacker's
residual. Nested: on each outer fold the booster is fitted on the 4 other folds; the F2 cut-off comes from inner 4-fold OOF
predictions of the same booster on those 4 folds; labels on the 5th. Reference = the v20 score with its cut-off chosen on the
4 training folds. Acceptance (user, 2026-11-16): pooled delta > 0, >= 3/5 folds better, no fold below -0.0005.
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import common  # noqa: F401  (chdir to the repo root)
from src import task1  # noqa: E402
from src.config import DATA_DIRS  # noqa: E402
from src.metrics import f2, optimize_threshold  # noqa: E402
from src.modeling import make_folds  # noqa: E402
from src.utils import load_preds  # noqa: E402

t0 = time.time()
tr = pd.read_csv(DATA_DIRS[1] / "train.csv")
tr = tr.dropna(subset=[task1.TARGET]).reset_index(drop=True)
y = tr[task1.TARGET].astype(int).values
folds = make_folds(y)
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
KEYS = ["lgb", "xgb", "cat", "et", "lr", "rf24", "lgb24", "xgb24", "cat24"]
S = np.column_stack([logit(load_preds(f"t1_{k}")["oof"]) for k in KEYS])
lr_oof = np.zeros(len(y))
for a, b in folds:
    lr_oof[b] = LogisticRegression(C=1.0, max_iter=2000).fit(S[a], y[a]).predict_proba(S[b])[:, 1]
L = (logit(lr_oof) + 2 * load_preds("t1_l2_trees")["oof"]) / 3          # v20 level-2 log-odds, out of fold
raw = [c for c in tr.columns if c not in ("player_id", "id", task1.TARGET)]
X = tr[raw].copy()
X["n_missing"] = X.isna().sum(1)
P = dict(n_estimators=150, learning_rate=0.02, num_leaves=15, min_child_samples=300, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.8, n_jobs=2, verbose=-1)


def boost(a, b_list):
    """Mean over 3 seeds of the boosted log-odds, fitted on rows a, returned for each index array in b_list."""
    outs = [np.zeros(len(b)) for b in b_list]
    for s in (0, 1, 2):
        m = lgb.LGBMClassifier(**P, random_state=s).fit(X.iloc[a], y[a], init_score=L[a])
        for k, b in enumerate(b_list):
            outs[k] += (L[b] + m.predict(X.iloc[b], raw_score=True)) / 3
    return outs


fo = np.zeros(len(y), int)
for i, (_, b) in enumerate(folds):
    fo[b] = i
ref, cand, per, thr_r, thr_c = np.zeros(len(y), int), np.zeros(len(y), int), [], [], []
for i, (a, b) in enumerate(folds):
    t_ref = optimize_threshold(y[a], expit(L[a]), 2.0)[0]
    ref[b] = expit(L[b]) >= t_ref
    inner = np.zeros(len(a))
    for ia, ib in StratifiedKFold(4, shuffle=True, random_state=i).split(a, y[a]):
        inner[ib] = boost(a[ia], [a[ib]])[0]
    t_c = optimize_threshold(y[a], expit(inner), 2.0)[0]
    cand[b] = expit(boost(a, [b])[0]) >= t_c
    per.append(f2(y[b], cand[b]) - f2(y[b], ref[b]))
    thr_r.append(t_ref)
    thr_c.append(t_c)
    print(f"fold {i}: reference F2 {f2(y[b], ref[b]):.4f} (cut-off {t_ref:.3f}), residual-boosted {f2(y[b], cand[b]):.4f} (cut-off {t_c:.3f}) "
          f"delta {per[-1]:+.4f} [{time.time() - t0:.0f}s]", flush=True)
d = f2(y, cand) - f2(y, ref)
rng = np.random.default_rng(0)
bs = []
for _ in range(1000):
    i = rng.integers(0, len(y), len(y))
    bs.append(f2(y[i], cand[i]) - f2(y[i], ref[i]))
print(f"\nnested F2: reference {f2(y, ref):.4f}, residual boosting {f2(y, cand):.4f}; pooled delta {d:+.4f} "
      f"[{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}]; folds +{sum(p > 0 for p in per)}/5, worst fold {min(per):+.4f}; "
      f"labels changed {np.mean(cand != ref):.2%}")
print(f"[{time.time() - t0:.0f}s]")
