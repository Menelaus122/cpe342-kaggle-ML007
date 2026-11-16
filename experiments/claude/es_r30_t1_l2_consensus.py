"""Round 30 (Jackpot loop), Task 1: logit-consensus meta-features for the tree meta-models of the level 2 (training rows only, CPU).

    python es_r30_t1_l2_consensus.py

Base = v22: nine members (eight 10-seed bags + LogReg), level 2 = LogReg + LightGBM + CatBoost (3 seeds each) on the nine log-odds,
mean of the three meta log-odds, cut-off from the 4 training folds' OOF level-2 scores. Candidate: the two tree meta-models also see
  max - min of the nine log-odds, their SD, mean(24-column members) - mean(49-feature members), and the row's missing-cell count
(the logistic meta-model is unchanged). Nested on the pipeline folds and three other stacker splits; five decimals.
Acceptance (user, 2026-11-16): pooled delta > 0, >= 3/5 folds on each split, no fold below -0.0005.
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
from src import task1  # noqa: E402
from src.config import DATA_DIRS  # noqa: E402
from src.metrics import f2, optimize_threshold  # noqa: E402
from src.modeling import make_folds  # noqa: E402
from src.utils import load_preds  # noqa: E402

warnings.filterwarnings("ignore")
t0 = time.time()
tr = pd.read_csv(DATA_DIRS[1] / "train.csv")
tr = tr.dropna(subset=[task1.TARGET]).reset_index(drop=True)
y = tr[task1.TARGET].astype(int).values
nan_n = tr[[c for c in tr.columns if c not in ("player_id", "id", task1.TARGET)]].isna().sum(1).values.astype(float)
folds = make_folds(y)
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
KEYS = ["lgb", "xgb", "cat", "et", "lr", "rf24", "lgb24", "xgb24", "cat24"]
L = np.column_stack([logit(load_preds("t1_lr")["oof"]) if k == "lr" else logit(load_preds(f"t1_bag10_{k}")["oof"]) for k in KEYS])
i24, i49 = [KEYS.index(k) for k in ("rf24", "lgb24", "xgb24", "cat24")], [KEYS.index(k) for k in ("lgb", "xgb", "cat", "et", "lr")]
EXTRA = np.column_stack([L.max(1) - L.min(1), L.std(1), L[:, i24].mean(1) - L[:, i49].mean(1), nan_n])
LGBP = dict(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            n_jobs=4, verbose=-1)


def level2(extra, fl):
    S_lr, S_tr = L, (L if extra is None else np.column_stack([L, extra]))
    p = np.zeros(len(y))
    for a, b in fl:
        z_lr = logit(LogisticRegression(C=1.0, max_iter=2000).fit(S_lr[a], y[a]).predict_proba(S_lr[b])[:, 1])
        z_lgb = np.mean([logit(lgb.LGBMClassifier(**LGBP, random_state=s).fit(S_tr[a], y[a]).predict_proba(S_tr[b])[:, 1]) for s in (0, 1, 2)], 0)
        z_cat = np.mean([logit(CatBoostClassifier(depth=4, iterations=400, learning_rate=0.05, random_seed=s, verbose=0, thread_count=4,
                                                  allow_writing_files=False).fit(S_tr[a], y[a]).predict_proba(S_tr[b])[:, 1]) for s in (0, 1, 2)], 0)
        p[b] = expit((z_lr + z_lgb + z_cat) / 3)
    fo = np.zeros(len(y), int)
    for i, (_, b) in enumerate(fl):
        fo[b] = i
    lab = np.zeros(len(y), int)
    for i in range(5):
        t = optimize_threshold(y[fo != i], p[fo != i], 2.0)[0]
        lab[fo == i] = p[fo == i] >= t
    return lab, fo


splits = [("pipeline folds", folds)] + [(f"split seed {s}", list(StratifiedKFold(5, shuffle=True, random_state=s).split(L, y))) for s in (1, 2, 3)]
rows = []
for label, fl in splits:
    base, fo = level2(None, fl)
    cand, _ = level2(EXTRA, fl)
    pf = [f2(y[fo == i], cand[fo == i]) - f2(y[fo == i], base[fo == i]) for i in range(5)]
    d = f2(y, cand) - f2(y, base)
    rows.append((d, sum(x > 0 for x in pf), min(pf)))
    print(f"{label}: v22 level 2 {f2(y, base):.5f}, + consensus features {f2(y, cand):.5f}, delta {d:+.5f}; folds +{sum(x > 0 for x in pf)}/5 "
          f"({' '.join(f'{x:+.5f}' for x in pf)}); labels changed {np.mean(cand != base):.3%} [{time.time() - t0:.0f}s]", flush=True)
ok = np.mean([r[0] for r in rows]) > 0 and all(r[1] >= 3 and r[2] >= -0.0005 for r in rows)
print(f"\npooled mean delta {np.mean([r[0] for r in rows]):+.5f}; folds better {' '.join(str(r[1]) + '/5' for r in rows)}; "
      f"worst fold {' '.join(f'{r[2]:+.5f}' for r in rows)} -> {'PASS' if ok else 'fail'} under the 2026-11-16 rule")
