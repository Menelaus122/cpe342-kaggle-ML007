"""Round 33, Task 1: shared pieces (training rows only, CPU, few threads so the GPU trainer is not slowed).

Base = v22: nine members (eight 10-seed bags + LogReg, out of fold on the pipeline folds), level 2 = LogReg + LightGBM + CatBoost
(3 seeds each) on the nine log-odds, mean of the three meta log-odds, cut-off for a fold from the other four folds' out-of-fold
level-2 scores (as es_r30_t1_l2_consensus.py). Splits: the pipeline folds + stacker split seeds 1, 2, 3. The base level-2
probabilities of every split are cached in outputs/r33/t1_l2_base.npz.
Acceptance (user, 2026-11-16): pooled delta > 0, >= 3/5 folds on each split, no fold below -0.0005.
"""
import os
import time
import warnings
from pathlib import Path

NT = int(os.environ.get("R33_THREADS", 2))
os.environ.setdefault("OMP_NUM_THREADS", str(NT))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
from catboost import CatBoostClassifier  # noqa: E402
from scipy.special import expit  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

import common  # noqa: F401,E402  (chdir to the repo root)
from src import task1  # noqa: E402
from src.config import DATA_DIRS  # noqa: E402
from src.metrics import f2, optimize_threshold  # noqa: E402
from src.modeling import make_folds  # noqa: E402
from src.utils import load_preds  # noqa: E402

warnings.filterwarnings("ignore")
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
t0 = time.time()
OUT = Path("experiments/claude/outputs/r33")
OUT.mkdir(parents=True, exist_ok=True)
tr = pd.read_csv(DATA_DIRS[1] / "train.csv")
tr = tr.dropna(subset=[task1.TARGET]).reset_index(drop=True)
y = tr[task1.TARGET].astype(int).values
RAW = [c for c in tr.columns if c not in ("player_id", "id", task1.TARGET)]
nan_n = tr[RAW].isna().sum(1).values.astype(float)
folds = make_folds(y)
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
KEYS = ["lgb", "xgb", "cat", "et", "lr", "rf24", "lgb24", "xgb24", "cat24"]
L = np.column_stack([logit(load_preds("t1_lr")["oof"]) if k == "lr" else logit(load_preds(f"t1_bag10_{k}")["oof"]) for k in KEYS])
LGBP = dict(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            n_jobs=NT, verbose=-1)
SPLITS = [("pipeline folds", folds)] + [(f"split seed {s}", list(StratifiedKFold(5, shuffle=True, random_state=s).split(L, y))) for s in (1, 2, 3)]


def log(m):
    print(f"{m} [{time.time() - t0:.0f}s]", flush=True)


def fold_of(fl):
    fo = np.zeros(len(y), int)
    for i, (_, b) in enumerate(fl):
        fo[b] = i
    return fo


def level2(fl, S_lr=None, S_tr=None, extra_by_fold=None):
    """v22 level 2, out of fold on the split `fl`. S_lr / S_tr: inputs of the logistic / the tree meta-models (default: the nine
    log-odds). extra_by_fold: a list of 5 matrices (one per outer fold, rows = all training rows) appended to the tree inputs of
    that fold only (features that must be recomputed inside each fold). Returns the level-2 probabilities."""
    S_lr = L if S_lr is None else S_lr
    S_tr = L if S_tr is None else S_tr
    p = np.zeros(len(y))
    for i, (a, b) in enumerate(fl):
        T = S_tr if extra_by_fold is None else np.column_stack([S_tr, extra_by_fold[i]])
        z_lr = logit(LogisticRegression(C=1.0, max_iter=2000).fit(S_lr[a], y[a]).predict_proba(S_lr[b])[:, 1])
        z_lgb = np.mean([logit(lgb.LGBMClassifier(**LGBP, random_state=s).fit(T[a], y[a]).predict_proba(T[b])[:, 1]) for s in (0, 1, 2)], 0)
        z_cat = np.mean([logit(CatBoostClassifier(depth=4, iterations=400, learning_rate=0.05, random_seed=s, verbose=0, thread_count=NT,
                                                  allow_writing_files=False).fit(T[a], y[a]).predict_proba(T[b])[:, 1]) for s in (0, 1, 2)], 0)
        p[b] = expit((z_lr + z_lgb + z_cat) / 3)
    return p


def labels_from(p, fo):
    """The pipeline's cut-off rule: for each fold the F2 cut-off of the other four folds' out-of-fold scores."""
    lab = np.zeros(len(y), int)
    for i in range(5):
        t = optimize_threshold(y[fo != i], p[fo != i], 2.0)[0]
        lab[fo == i] = p[fo == i] >= t
    return lab


def best_cut(yy, s):
    """Exact F2-optimal cut-off of an arbitrary score (flag s >= cut): scan the sorted scores."""
    o = np.argsort(-s)
    tp = np.cumsum(yy[o])
    k = np.arange(1, len(s) + 1)
    f = 5 * tp / (4 * yy.sum() + k)
    last = np.r_[s[o][1:] != s[o][:-1], True]                     # only cuts between distinct scores
    j = int(np.argmax(np.where(last, f, -1)))
    return s[o][j]


def labels_exact(s, fo):
    lab = np.zeros(len(y), int)
    for i in range(5):
        lab[fo == i] = s[fo == i] >= best_cut(y[fo != i], s[fo != i])
    return lab


def base_probs():
    """Base level-2 probabilities of the four splits (cached)."""
    f = OUT / "t1_l2_base.npz"
    if f.exists():
        d = np.load(f)
        return [d[f"p{k}"] for k in range(len(SPLITS))]
    ps = []
    for label, fl in SPLITS:
        ps.append(level2(fl))
        log(f"base level 2, {label}: F2 {f2(y, labels_from(ps[-1], fold_of(fl))):.5f}")
    np.savez(f, **{f"p{k}": p for k, p in enumerate(ps)})
    return ps


def report(name, cand_labels, base_labels):
    """cand_labels / base_labels: one label vector per split. Prints per-split deltas and the verdict; returns the rows."""
    rows = []
    for (label, fl), cand, base in zip(SPLITS, cand_labels, base_labels):
        fo = fold_of(fl)
        pf = [f2(y[fo == i], cand[fo == i]) - f2(y[fo == i], base[fo == i]) for i in range(5)]
        d = f2(y, cand) - f2(y, base)
        rows.append((d, sum(x > 0 for x in pf), min(pf)))
        print(f"  {name} | {label}: base {f2(y, base):.5f}, candidate {f2(y, cand):.5f}, delta {d:+.5f}; folds +{sum(x > 0 for x in pf)}/5 "
              f"({' '.join(f'{x:+.5f}' for x in pf)}); labels changed {np.mean(cand != base):.3%}", flush=True)
    ok = np.mean([r[0] for r in rows]) > 0 and all(r[1] >= 3 and r[2] >= -0.0005 for r in rows)
    log(f"=> {name}: pooled mean delta {np.mean([r[0] for r in rows]):+.5f}; folds better {' '.join(str(r[1]) + '/5' for r in rows)}; "
        f"worst fold {' '.join(f'{r[2]:+.5f}' for r in rows)} -> {'PASS' if ok else 'fail'}")
    return rows
