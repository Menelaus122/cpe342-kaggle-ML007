"""Round 33, Task 1 (added from the literature): a cost-sensitive level 2 (training rows only, CPU).

    python fq_r33_t1_costsens.py

The F-beta-optimal classifier is a cut on the posterior, which is the same as cost-sensitive classification with a cost ratio fixed by
the optimum (Koyejo et al., NeurIPS 2014; Parambath, Usunier & Grandvalet, NeurIPS 2014): flagging p >= t is the 0.5 rule of a model
fitted with class weights (1 - t) : t. The v22 level 2 minimises the unweighted log-loss, so its trees and its logistic fit spend
their capacity evenly over the probability range; with positive weight w = (1 - t) / t (t = 0.162 -> about 5) the weighted log-loss
is steepest around p = t, where the F2 decision is made.
Candidates (level 2 = LogReg + LightGBM x3 + CatBoost x3 as in v22, every meta-model fitted with positive weight w):
  A  w = 5;     B  w = 3;     C  mean of the v22 level-2 log-odds and A's (each standardised on the training folds).
The cut-off for a fold is the exact F2 optimum of the other four folds' out-of-fold scores, as for the base.
Nested on the pipeline folds and stacker split seeds 1, 2, 3. Acceptance (user, 2026-11-16): pooled delta > 0, >= 3/5 folds on each
split, no fold below -0.0005.
"""
import lightgbm as lgb
import numpy as np
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

import r33_t1_common as c
from r33_t1_common import L, SPLITS, f2, fold_of, labels_exact, labels_from, log, logit, y


def level2_w(fl, w):
    """v22 level 2 with positive-class weight w, out of fold on `fl`. Returns the mean meta log-odds."""
    z = np.zeros(len(y))
    sw = np.where(y == 1, float(w), 1.0)
    for a, b in fl:
        z_lr = logit(LogisticRegression(C=1.0, max_iter=2000).fit(L[a], y[a], sample_weight=sw[a]).predict_proba(L[b])[:, 1])
        z_lgb = np.mean([logit(lgb.LGBMClassifier(**c.LGBP, random_state=s).fit(L[a], y[a], sample_weight=sw[a]).predict_proba(L[b])[:, 1])
                         for s in (0, 1, 2)], 0)
        z_cat = np.mean([logit(CatBoostClassifier(depth=4, iterations=400, learning_rate=0.05, random_seed=s, verbose=0, thread_count=c.NT,
                                                  allow_writing_files=False).fit(L[a], y[a], sample_weight=sw[a]).predict_proba(L[b])[:, 1])
                         for s in (0, 1, 2)], 0)
        z[b] = (z_lr + z_lgb + z_cat) / 3
    return z


base_p = c.base_probs()
base_labels = [labels_from(p, fold_of(fl)) for p, (_, fl) in zip(base_p, SPLITS)]
log("base level 2 ready: " + ", ".join(f"{f2(y, b):.5f}" for b in base_labels))
names = ["A positive weight 5", "B positive weight 3", "C mean of v22 level 2 and A (standardised)"]
cand = {n: [] for n in names}
for k, (label, fl) in enumerate(SPLITS):
    fo = fold_of(fl)
    zA, zB = level2_w(fl, 5.0), level2_w(fl, 3.0)
    zb = logit(base_p[k])
    zC = np.zeros(len(y))
    for a, b in fl:                                                   # scale from the training folds' out-of-fold scores
        zC[b] = (zb[b] - zb[a].mean()) / zb[a].std() + (zA[b] - zA[a].mean()) / zA[a].std()
    for n, z in zip(names, (zA, zB, zC)):
        cand[n].append(labels_exact(z, fo))
    log(f"{label}: base {f2(y, base_labels[k]):.5f}; " + "; ".join(f"{n.split(' ')[0]} {f2(y, cand[n][-1]):.5f}" for n in names)
        + f"; base under the exact cut-off search {f2(y, labels_exact(base_p[k], fo)):.5f}")
print()
for n in names:
    c.report(n, cand[n], base_labels)
log("done")
