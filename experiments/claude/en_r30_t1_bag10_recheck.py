"""Round 30, Task 1: the round 14 10-seed bag under the v20 level 2, re-scored with the 2026-11-16 acceptance rule (training only).

    python en_r30_t1_bag10_recheck.py

Round 27 (dv_r27_t1_bag_l2.py) found the 10-seed bag at +0.0003 mean over four stacker splits, folds better 3, 3, 4, 4 of 5, and
discarded it under the old rule (>= 4/5 on every split and the test-like gate). The new rule (user, 2026-11-16): pooled delta > 0,
>= 3/5 folds, no fold below -0.0005. Two folds printed as -0.0005 at four decimals, so the same nested evaluation is repeated here with
five decimals; the test-like part (built from the test's missing-cell rates) is dropped and test.csv is not read.
Members: outputs/r14_bag/t1_<member>_<seed>.npz (oof); LogReg is deterministic (pipeline cache). Level 2 = LogReg + LightGBM +
CatBoost (3 seeds each) on the nine log-odds, fitted on 4 folds, cut-off from those folds' OOF scores, labels on the 5th.
"""
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import common  # noqa: F401  (chdir to the repo root)
from common import OUT  # noqa: E402
from src import task1  # noqa: E402
from src.config import DATA_DIRS  # noqa: E402
from src.metrics import f2, optimize_threshold  # noqa: E402
from src.modeling import make_folds  # noqa: E402
from src.utils import load_preds  # noqa: E402

warnings.filterwarnings("ignore")
t0 = time.time()
tr = pd.read_csv(DATA_DIRS[1] / "train.csv", usecols=[task1.TARGET])
y = tr[task1.TARGET].dropna().astype(int).values
folds = make_folds(y)
SEEDS = (42, 100, 2026, 777, 999, 1234, 5555, 8888, 31415, 27182)
KEYS = ["lgb", "xgb", "cat", "et", "lr", "rf24", "lgb24", "xgb24", "cat24"]
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
pipe = [logit(load_preds(f"t1_{k}")["oof"]) for k in KEYS]
bag = [logit(load_preds("t1_lr")["oof"]) if k == "lr" else
       np.mean([logit(np.load(OUT / "r14_bag" / f"t1_{k}_{s}.npz")["oof"]) for s in SEEDS], axis=0) for k in KEYS]
LGBP = dict(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            n_jobs=2, verbose=-1)
MAKE = [lambda: [LogisticRegression(C=1.0, max_iter=2000)],
        lambda: [lgb.LGBMClassifier(**LGBP, random_state=s) for s in (0, 1, 2)],
        lambda: [CatBoostClassifier(depth=4, iterations=400, learning_rate=0.05, random_seed=s, verbose=0, thread_count=2,
                                    allow_writing_files=False) for s in (0, 1, 2)]]


def level2(cols, fl):
    S = np.column_stack(cols)
    p = np.zeros(len(y))
    for a, b in fl:
        p[b] = expit(np.mean([np.mean([logit(m.fit(S[a], y[a]).predict_proba(S[b])[:, 1]) for m in mk()], 0) for mk in MAKE], 0))
    fo = np.zeros(len(y), int)
    for i, (_, b) in enumerate(fl):
        fo[b] = i
    lab = np.zeros(len(y), int)
    for i in range(5):
        t = optimize_threshold(y[fo != i], p[fo != i], 2.0)[0]
        lab[fo == i] = p[fo == i] >= t
    return lab, fo


splits = [("pipeline folds", folds)] + [(f"split seed {s}", list(StratifiedKFold(5, shuffle=True, random_state=s).split(pipe[0], y))) for s in (1, 2, 3)]
rows = []
for label, fl in splits:
    base, fo = level2(pipe, fl)
    cand, _ = level2(bag, fl)
    pf = [f2(y[fo == i], cand[fo == i]) - f2(y[fo == i], base[fo == i]) for i in range(5)]
    d = f2(y, cand) - f2(y, base)
    rows.append((d, sum(x > 0 for x in pf), min(pf)))
    print(f"{label}: pipeline {f2(y, base):.5f}, 10-seed bag {f2(y, cand):.5f}, delta {d:+.5f}; folds +{sum(x > 0 for x in pf)}/5 "
          f"({' '.join(f'{x:+.5f}' for x in pf)}); labels changed {np.mean(cand != base):.3%} [{time.time() - t0:.0f}s]", flush=True)
ok = np.mean([r[0] for r in rows]) > 0 and all(r[1] >= 3 and r[2] >= -0.0005 for r in rows)
print(f"\npooled mean delta {np.mean([r[0] for r in rows]):+.5f}; folds better {' '.join(str(r[1]) + '/5' for r in rows)}; "
      f"worst fold {' '.join(f'{r[2]:+.5f}' for r in rows)} -> {'PASS' if ok else 'fail'} under the 2026-11-16 rule")
