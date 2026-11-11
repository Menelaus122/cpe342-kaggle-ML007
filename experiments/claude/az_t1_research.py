"""Task 1 round 9 (2026-11-08): methods from the literature, strict validation (nested tau, paired bootstrap).

Baseline B0 = notebook stack (5 members), nested F2 0.8364.
Q1 loss alignment: focal loss (Lin et al. 2017) LightGBM, gamma = 1 and 2, as a 6th stack member. Focal loss down-weights
   easy rows (the obvious cheaters and obvious clean players) and spends capacity on the ambiguous middle.
Q2 ranking objective: F2 at high recall depends only on how well cheaters are RANKED above clean players around the cut-off.
   XGBoost with a pairwise ranking objective (rank:pairwise, one query = all rows) optimises that ordering directly
   (AUC-type loss), as a 6th stack member.
Q3 decision rule: the decision-theoretic plug-in rule for F-measures (Lipton, Elkan & Narayanaswamy 2014; Nan et al. 2012):
   with calibrated probabilities, flag the top-k rows that maximise the EXPECTED F2 of the fold being predicted,
   k* = argmax 5*sum_{i<=k} p_(i) / (4*sum_i p_i + k), using only that fold's predicted probabilities (no labels).
   Evaluation only: applied to the test set it would set the cut-off from test predictions (transductive), which is a
   question for the TAs, so the test value is printed for information and not used.
"""
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
import common  # noqa: E402
from common import load_t1, T1_MODELS, OUT, _logit, tuned  # noqa: E402
from src.utils import load_preds  # noqa: E402
from src.metrics import optimize_threshold, fbeta  # noqa: E402
from src.modeling import LGB_BASE  # noqa: E402

T0 = time.time()
log = lambda *a: print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
rng = np.random.default_rng(0)
tr, te, X, y, Xt, folds, _ = load_t1()
base = {n: load_preds(n) for n in T1_MODELS}
f2 = lambda yt, yp: fbeta(yt, yp, 2, "binary")


def stack(oofs, tests=None):
    S = np.column_stack([_logit(o) for o in oofs])
    o = np.zeros(len(y))
    for a, b in folds:
        o[b] = LogisticRegression(max_iter=3000).fit(S[a], y[a]).predict_proba(S[b])[:, 1]
    if tests is None:
        return o
    St = np.column_stack([_logit(t) for t in tests])
    return o, LogisticRegression(max_iter=3000).fit(S, y).predict_proba(St)[:, 1]


def nested(p):
    pred = np.zeros(len(y), int)
    for a, b in folds:
        t, _ = optimize_threshold(y[a], p[a], 2.0, average="binary")
        pred[b] = p[b] >= t
    return pred


def boot(pn, po, B=1000):
    yb = y.astype(bool)
    M = np.column_stack([yb & (pn == 1), ~yb & (pn == 1), yb & (pn == 0), yb & (po == 1), ~yb & (po == 1), yb & (po == 0)]).astype(float)
    d = np.empty(B)
    for i in range(B):
        c = rng.poisson(1.0, len(y)) @ M
        d[i] = 5 * c[0] / (5 * c[0] + 4 * c[2] + c[1]) - 5 * c[3] / (5 * c[3] + 4 * c[5] + c[4])
    return np.percentile(d, [2.5, 97.5])


rows = []


def report(name, desc, p, pred, ref):
    s, s0 = f2(y, pred), f2(y, ref)
    lo, hi = boot(pred, ref)
    rows.append([name, desc, roc_auc_score(y, p), s, s - s0, f"[{lo:+.4f}, {hi:+.4f}]", "KEEP" if lo > 0 else "discard"])
    log(f"{name}: AUC {rows[-1][2]:.4f}, nested F2 {s:.4f} (delta {s - s0:+.4f}, CI [{lo:+.4f}, {hi:+.4f}])")


p0 = stack([base[n]["oof"] for n in T1_MODELS])
pred0 = nested(p0)
log(f"B0: AUC {roc_auc_score(y, p0):.4f}, nested F2 {f2(y, pred0):.4f}")

# ------------------------------------------------------------------ Q1: focal-loss LightGBM
P1 = tuned("task1_lgb", {"learning_rate": 0.05})
P1 = {k: v for k, v in P1.items() if k not in ("scale_pos_weight", "objective")}
P1["learning_rate"] = 0.03


def focal(zr, yy, g, al=0.5):
    p = np.clip(1 / (1 + np.exp(-zr)), 1e-9, 1 - 1e-9)
    return -(al * yy * (1 - p) ** g * np.log(p) + (1 - al) * (1 - yy) * p ** g * np.log(1 - p))


def focal_obj(g):
    def obj(preds, data):
        yy, e = data.get_label(), 1e-4
        l0, lp, lm = focal(preds, yy, g), focal(preds + e, yy, g), focal(preds - e, yy, g)
        return (lp - lm) / (2 * e), np.maximum((lp - 2 * l0 + lm) / e ** 2, 1e-6)
    return obj


def logloss_eval(preds, data):
    p = np.clip(1 / (1 + np.exp(-preds)), 1e-9, 1 - 1e-9)
    yy = data.get_label()
    return "logloss", float(-np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p))), False


init = np.log(y.mean() / (1 - y.mean()))
lgb_params = {k: v for k, v in LGB_BASE.items() if k not in ("n_estimators",)}
for g in (1.0, 2.0):
    oof = np.zeros(len(y))
    for a, b in folds:
        dtr = lgb.Dataset(X.iloc[a], y[a], init_score=np.full(len(a), init))
        dva = lgb.Dataset(X.iloc[b], y[b], init_score=np.full(len(b), init), reference=dtr)
        prm = {**lgb_params, **P1, "objective": focal_obj(g), "metric": "None"}
        bst = lgb.train(prm, dtr, 10000, valid_sets=[dva], feval=logloss_eval,
                        callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[b] = 1 / (1 + np.exp(-(bst.predict(X.iloc[b], num_iteration=bst.best_iteration, raw_score=True) + init)))
    log(f"focal gamma={g:g}: member AUC {roc_auc_score(y, oof):.4f} (notebook LightGBM {roc_auc_score(y, base['t1_lgb']['oof']):.4f})")
    ps = stack([base[n]["oof"] for n in T1_MODELS] + [oof])
    report(f"Q1 (gamma={g:g})", f"+ focal-loss LightGBM, gamma {g:g}, as 6th member", ps, nested(ps), pred0)

# ------------------------------------------------------------------ Q2: pairwise ranking XGBoost
oof = np.zeros(len(y))
for a, b in folds:
    dtr = xgb.DMatrix(X.iloc[a], label=y[a])
    dtr.set_group([len(a)])
    dva = xgb.DMatrix(X.iloc[b], label=y[b])
    dva.set_group([len(b)])
    prm = {"objective": "rank:pairwise", "eta": 0.05, "max_depth": 6, "min_child_weight": 10, "subsample": 0.8,
           "colsample_bytree": 0.6, "eval_metric": "auc", "tree_method": "hist", "lambdarank_pair_method": "mean",
           "lambdarank_num_pair_per_sample": 8, "seed": 42}
    bst = xgb.train(prm, dtr, 5000, evals=[(dva, "va")], early_stopping_rounds=200, verbose_eval=False)
    s = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
    oof[b] = 1 / (1 + np.exp(-(s - s.mean()) / s.std()))       # monotone squashing; the stacker re-scales it
log(f"pairwise ranker: member AUC {roc_auc_score(y, oof):.4f}")
ps = stack([base[n]["oof"] for n in T1_MODELS] + [oof])
report("Q2", "+ pairwise-ranking XGBoost (rank:pairwise) as 6th member", ps, nested(ps), pred0)


# ------------------------------------------------------------------ Q3: decision-theoretic expected-F2 rule
def dta(p):
    o = np.argsort(-p)
    cum = np.cumsum(p[o])
    k = np.arange(1, len(p) + 1)
    kstar = int(np.argmax(5 * cum / (4 * p.sum() + k))) + 1
    pred = np.zeros(len(p), int)
    pred[o[:kstar]] = 1
    return pred, p[o[kstar - 1]]


pred3 = np.zeros(len(y), int)
taus = []
for a, b in folds:
    pb, tb = dta(p0[b])
    pred3[b] = pb
    taus.append(tb)
report("Q3", "expected-F2 plug-in rule per fold (no labels used)", p0, pred3, pred0)
log(f"  implied cut-offs per fold: {np.round(taus, 3)} (nested threshold rule: ~0.16)")
try:
    full = {n: load_preds(n + "_full") for n in T1_MODELS}
    tests = [0.5 * base[n]["test"] + 0.5 * full[n]["test"] for n in T1_MODELS]
    _, pt = stack([base[n]["oof"] for n in T1_MODELS], tests)
    _, tt = dta(pt)
    log(f"  for information only (not applied): on the test predictions the rule would cut at {tt:.3f}, flagging "
        f"{np.mean(pt >= tt):.1%} (the file uses 0.163, flagging {np.mean(pt >= 0.163):.1%})")
except Exception as e:  # noqa: BLE001
    log(f"  test-side check skipped: {e}")

R = pd.DataFrame(rows, columns=["Experiment", "Change", "OOF AUC", "OOF F2 (nested)", "delta vs B0", "95% CI", "Decision"])
print("\n" + R.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
log("done")
