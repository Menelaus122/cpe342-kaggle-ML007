"""Round 26, Task 1: the level-2 combination that passed (lr + lgb + cat, dm_r26_t1_l2stack.py) fitted as the pipeline would.

    python do_r26_t1_l2_final.py

Label-free sanity check before a pipeline change: tree meta-models are fitted on out-of-fold logits but applied to test logits that
are averages (fold models + full refit), which are a little smoother. If that mattered, the share of test labels the new stacker
changes, or its flag rate, would differ from what it does out of fold. No test labels are involved; nothing is tuned here.
"""
import warnings

import lightgbm as lgb
import numpy as np
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

import common  # noqa: F401  (chdir to the repo root)
from common import load_t1  # noqa: E402
from src.metrics import f2, optimize_threshold  # noqa: E402
from src.utils import load_preds  # noqa: E402

warnings.filterwarnings("ignore")
_, _, _, y, _, folds, _ = load_t1()
y = np.asarray(y)
NAMES = ["t1_lgb", "t1_xgb", "t1_cat", "t1_et", "t1_lr", "t1_rf24", "t1_lgb24", "t1_xgb24", "t1_cat24"]
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sig = lambda z: 1 / (1 + np.exp(-z))
S = np.column_stack([logit(load_preds(k)["oof"]) for k in NAMES])
St = np.column_stack([logit(0.5 * load_preds(k)["test"] + 0.5 * load_preds(k + "_full")["test"]) for k in NAMES])
LGBP = dict(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            n_jobs=8, verbose=-1)
make = {"lr": lambda: [LogisticRegression(C=1.0, max_iter=2000)],
        "lgb": lambda: [lgb.LGBMClassifier(**LGBP, random_state=s) for s in (0, 1, 2)],
        "cat": lambda: [CatBoostClassifier(depth=4, iterations=400, learning_rate=0.05, random_seed=s, verbose=0, thread_count=8) for s in (0, 1, 2)]}
oof = {k: np.zeros(len(y)) for k in make}
for a, b in folds:
    for k, mk in make.items():
        oof[k][b] = np.mean([logit(m.fit(S[a], y[a]).predict_proba(S[b])[:, 1]) for m in mk()], 0)
tst = {k: np.mean([logit(m.fit(S, y).predict_proba(St)[:, 1]) for m in mk()], 0) for k, mk in make.items()}
print("column means of the member logits: OOF", np.round(S.mean(0), 3), "| test", np.round(St.mean(0), 3))
print("column SDs  of the member logits: OOF", np.round(S.std(0), 3), "| test", np.round(St.std(0), 3))
res = {}
for name, parts in [("logistic stacker (pipeline)", ["lr"]), ("lr + lgb + cat", ["lr", "lgb", "cat"])]:
    po, pt = sig(np.mean([oof[k] for k in parts], 0)), sig(np.mean([tst[k] for k in parts], 0))
    thr, s = optimize_threshold(y, po, 2.0)
    res[name] = (po >= thr, pt >= thr)
    print(f"{name:28s} OOF F2 {s:.4f} at cut {thr:.3f} | flagged: OOF {np.mean(po >= thr):.4f}, test {np.mean(pt >= thr):.4f}")
a, b = res["logistic stacker (pipeline)"], res["lr + lgb + cat"]
print(f"labels changed by the new stacker: OOF {np.mean(a[0] != b[0]):.3%} | test {np.mean(a[1] != b[1]):.3%} "
      f"(test: {int((~a[1] & b[1]).sum())} newly flagged, {int((a[1] & ~b[1]).sum())} released, of {len(a[1])})")
