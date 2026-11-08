"""Shared harness for the Task 2 / Task 3 improvement experiments.

Rebuilds exactly the notebook's features, folds and baseline scores from the cached predictions
in outputs/preds/, so every experiment is compared against the same OOF numbers.
"""
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src import task2, task3
from src.metrics import macro_f1, nmae, optimize_class_weights, apply_class_weights
from src.modeling import make_folds
from src.utils import load_preds, load_params
from src.config import SEED

OUT = ROOT / "experiments" / "claude" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

T2_MODELS = ["t2_lgb", "t2_xgb", "t2_cat", "t2_lr"]


def tuned(name, fallback):
    """Same rule as the notebook (json params, else fallback; learning rate halved)."""
    p = load_params(name)
    p = dict(fallback) if p is None else dict(p)
    if "learning_rate" in p:
        p["learning_rate"] = p["learning_rate"] / 2
    return p


P2_lgb = tuned("task2_lgb", {"learning_rate": 0.05})
P3_lgb = tuned("task3_lgb", {"learning_rate": 0.05})


def load_t2():
    tr, te = task2.load()
    X, y, Xt = task2.build(tr, te)
    return tr, te, X, y, Xt, make_folds(y)


def load_t3():
    tr, te = task3.load()
    X, y, Xt = task3.build(tr, te)
    return tr, te, X, y, Xt, make_folds(y, kind="regression")


def stack_multi(oofs, tests, y, folds, C_=0.5):
    """The notebook's multinomial LogReg stacker on log-probabilities (OOF + full fit)."""
    S = np.column_stack([np.log(np.clip(o, 1e-6, 1)) for o in oofs])
    St = np.column_stack([np.log(np.clip(t, 1e-6, 1)) for t in tests])
    oof = np.zeros((len(y), oofs[0].shape[1]))
    for a, b in folds:
        oof[b] = LogisticRegression(C=C_, max_iter=3000).fit(S[a], y[a]).predict_proba(S[b])
    final = LogisticRegression(C=C_, max_iter=3000).fit(S, y)
    return oof, final.predict_proba(St)


def t2_score(oofs, tests, y, folds):
    """Stack -> class multipliers -> macro-F1 (the notebook's Task 2 score)."""
    so, st = stack_multi(oofs, tests, y, folds)
    w, s = optimize_class_weights(y, so)
    return s, so, st, w


def t2_baseline(y, folds):
    m = [load_preds(n) for n in T2_MODELS]
    return t2_score([d["oof"] for d in m], [d["test"] for d in m], y, folds)


# ---------------------------------------------------------------------------- Task 3
def t3_pool():
    """Rebuild the notebook's level-1 pool (OOF, test) for Task 3 from the caches."""
    from src.task3 import tune_mixture, mixture_median, tune_hard_hurdle, hard_hurdle, expected_value
    _, _, X, y, Xt, folds = load_t3()
    m = {}
    for name, key, log in [("LGB-L1 raw THB", "t3_lgb_raw", False), ("XGB-L1 raw THB", "t3_xgb_raw", False),
                           ("Cat-MAE raw THB", "t3_cat_raw", False), ("LGB-L1 log1p", "t3_lgb_log", True),
                           ("LGB-Tweedie (mean)", "t3_tweedie", False)]:
        d = load_preds(key)
        o, t = (np.expm1(d["oof"]), np.expm1(d["test"])) if log else (d["oof"], d["test"])
        m[name] = (np.clip(o, 0, None), np.clip(t, 0, None))
    h = load_preds("t3_hurdle")
    sigma0, smear = float(h["sigma"]), float(h["smear"])
    tm = tune_mixture(y, h["p_oof"], h["mu_oof"], sigma0)
    m["Hurdle mixture-median"] = (mixture_median(np.clip(h["p_oof"] + tm["shift"], 0, 1), h["mu_oof"], tm["sigma"]),
                                  mixture_median(np.clip(h["p_test"] + tm["shift"], 0, 1), h["mu_test"], tm["sigma"]))
    th = tune_hard_hurdle(y, h["p_oof"], h["mu_oof"])
    m["Hurdle hard gate"] = (hard_hurdle(h["p_oof"], h["mu_oof"], th["tau"]), hard_hurdle(h["p_test"], h["mu_test"], th["tau"]))
    m["Hurdle E[y] + smearing"] = (expected_value(h["p_oof"], h["mu_oof"], smear), expected_value(h["p_test"], h["mu_test"], smear))
    gate_oof = (np.clip(h["p_oof"] + tm["shift"], 0, 1) > 0.5).astype(float)
    gate_test = (np.clip(h["p_test"] + tm["shift"], 0, 1) > 0.5).astype(float)
    return X, y, Xt, folds, m, gate_oof, gate_test


def t3_blend_score(y, folds, P, gate):
    """OOF L1 blend -> zero gate -> best global scale (the notebook's Task 3 score)."""
    blend = np.zeros(len(y))
    for a, b in folds:
        w = task3.l1_blend_weights(y[a], [P[a, j] for j in range(P.shape[1])])
        blend[b] = P[b] @ w
    scales = np.linspace(0.85, 1.15, 61)
    sc = [nmae(y, blend * s * gate) for s in scales]
    i = int(np.argmax(sc))
    return sc[i], blend * scales[i] * gate


# ---------------------------------------------------------------------------- Task 1
from src import task1
from src.metrics import f2, optimize_threshold

T1_MODELS = ["t1_lgb", "t1_xgb", "t1_cat", "t1_et", "t1_lr"]


def load_t1():
    tr, te, n_unl = task1.load()
    X, y, Xt = task1.build(tr, te)
    return tr, te, X, y, Xt, make_folds(y), n_unl


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def stack_binary(oofs, tests, y, folds, C_=1.0):
    S = np.column_stack([_logit(o) for o in oofs]); St = np.column_stack([_logit(t) for t in tests])
    oof = np.zeros(len(y))
    for a, b in folds:
        oof[b] = LogisticRegression(C=C_, max_iter=2000).fit(S[a], y[a]).predict_proba(S[b])[:, 1]
    return oof, LogisticRegression(C=C_, max_iter=2000).fit(S, y).predict_proba(St)[:, 1]


def t1_score(oofs, tests, y, folds):
    """Stack -> best positive-class F2 over thresholds (OOF)."""
    so, st = stack_binary(oofs, tests, y, folds)
    thr, s = optimize_threshold(y, so, 2.0)
    return s, thr, so, st
