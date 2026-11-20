"""Round 35, Task 1: a level-1 LightGBM trained with an asymmetric focal loss (training rows only, CPU).

    python fy_r35_t1_focal.py

Hypothesis (user brief of 2026-11-16 ~22:25): every Task 1 change so far worked on the level-2 logits of members that all minimise
the log-loss; a level-1 member with a different loss may add diversity where the F2 decision is made.
Loss: asymmetric focal loss, gamma = 2.0, alpha = 0.75 on the positive (cheater) class:
    L = -alpha (1 - p)^gamma log p          (y = 1)        L = -(1 - alpha) p^gamma log(1 - p)     (y = 0),   p = sigmoid(score)
as a LightGBM custom objective (analytic gradient and Hessian; the Hessian is floored at 5% of the weighted log-loss Hessian,
because the focal loss is not convex in the score; min_gain_to_split, tuned on the log-loss scale, is set to 0 for the focal
member).
Features: the 49 pipeline features + the 15 pairwise ratios of the top-6 skill columns, the six chosen INSIDE each fold (gain
importance of a 200-tree LightGBM fitted on the training fold, among the mechanical / game-sense columns and reaction time).
Members (pipeline's tuned LightGBM parameters at the tuned learning rate, early stopping on the validation fold's AUC as in the
pipeline, 3 seeds, mean of the raw scores):
  C  control: log-loss, 49 features + ratios (separates the ratio features from the loss);
  F  focal loss, 49 features + ratios.
CatBoost is left out: its custom objectives run in Python per row and take hours on the CPU.
Level 2 = v22 (r33_t1_common.level2) with C / F as a tenth member, nested on the pipeline folds and stacker split seeds 1, 2, 3.
Acceptance (user, 2026-11-16): pooled delta > 0, >= 3/5 folds on each split, no fold below -0.0005.
"""
import itertools
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

import r33_t1_common as c
from r33_t1_common import L, SPLITS, f2, fold_of, labels_from, log, y
from src import task1

OUT = c.OUT.parent / "r35"
OUT.mkdir(parents=True, exist_ok=True)
GAMMA, ALPHA = 2.0, 0.75
raw = c.tr.drop(columns=[col for col in ("player_id", "id", task1.TARGET) if col in c.tr.columns])
X49 = task1.add_features(c.tr, ref=raw)
SKILL = task1.MECH_COLS + task1.SENSE_COLS + ["reaction_time_ms"]


def focal_parts(p, w):
    """Gradient and Hessian (w.r.t. the score) of -w (1 - p)^gamma log p, p = sigmoid(score): the y = 1 branch."""
    p = np.clip(p, 1e-7, 1 - 1e-7)
    q = 1 - p
    lp = np.log(p)
    g = w * (GAMMA * p * q ** GAMMA * lp - q ** (GAMMA + 1))
    h = w * p * q * (GAMMA * q ** GAMMA * lp - GAMMA ** 2 * p * q ** (GAMMA - 1) * lp + GAMMA * q ** GAMMA + (GAMMA + 1) * q ** GAMMA)
    return g, h


def focal_obj(score, data):
    yy = data.get_label()
    p = expit(score)
    g1, h1 = focal_parts(p, ALPHA)                     # positives
    g0, h0 = focal_parts(1 - p, 1 - ALPHA)             # negatives: the same form in 1 - p, gradient sign flipped
    g = np.where(yy == 1, g1, -g0)
    h = np.where(yy == 1, h1, h0)
    floor = 0.05 * np.where(yy == 1, ALPHA, 1 - ALPHA) * p * (1 - p)
    return g, np.maximum(h, floor) + 1e-9


# numerical check of the derivatives
_z = np.linspace(-4, 4, 9)
for _w, _f in ((ALPHA, lambda z: -ALPHA * (1 - expit(z)) ** GAMMA * np.log(expit(z))),):
    _g, _h = focal_parts(expit(_z), _w)
    _gn = (_f(_z + 1e-5) - _f(_z - 1e-5)) / 2e-5
    _hn = (_f(_z + 1e-4) - 2 * _f(_z) + _f(_z - 1e-4)) / 1e-8
    assert np.allclose(_g, _gn, atol=1e-6) and np.allclose(_h, _hn, atol=1e-4), (_g - _gn, _h - _hn)

P = {k: v for k, v in json.load(open("outputs/params/task1_lgb.json")).items() if k != "scale_pos_weight"}
BASE = dict(P, verbose=-1, num_threads=c.NT, metric="auc", bagging_freq=1, feature_fraction=P["colsample_bytree"], bagging_fraction=P["subsample"],
            min_data_in_leaf=P["min_child_samples"], lambda_l1=P["reg_alpha"], lambda_l2=P["reg_lambda"], min_gain_to_split=P["min_split_gain"])
for k in ("colsample_bytree", "subsample", "min_child_samples", "reg_alpha", "reg_lambda", "min_split_gain"):
    BASE.pop(k)


def with_ratios(a, b):
    """49 features + the 15 ratios of the six skill columns chosen on the training fold a."""
    sel = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, n_jobs=c.NT, verbose=-1, random_state=0).fit(raw.iloc[a], y[a])
    imp = pd.Series(sel.booster_.feature_importance("gain"), index=raw.columns)[SKILL].sort_values(ascending=False)
    top = list(imp.index[:6])
    R = pd.DataFrame({f"ratio_{i}_{j}": raw[u].values / (raw[v].values + 1e-6) for (i, u), (j, v) in itertools.combinations(enumerate(top), 2)},
                     index=X49.index)
    X = pd.concat([X49, R], axis=1)
    return X.iloc[a], X.iloc[b], top


f_m = OUT / "t1_focal_members.npz"
if f_m.exists():
    d = np.load(f_m)
    ZC, ZF = d["zc"], d["zf"]
else:
    zc, zf = np.zeros((3, len(y))), np.zeros((3, len(y)))
    for i, (a, b) in enumerate(c.folds):
        Xa, Xb, top = with_ratios(a, b)
        its = []
        for s in (0, 1, 2):
            for kind, store in (("logloss", zc), ("focal", zf)):
                prm = dict(BASE, seed=s, bagging_seed=s, feature_fraction_seed=s, objective="binary" if kind == "logloss" else focal_obj)
                if kind == "focal":
                    prm["min_gain_to_split"] = 0.0          # tuned for the log-loss scale; focal gradients are several times smaller
                dtr, dva = lgb.Dataset(Xa, y[a]), lgb.Dataset(Xb, y[b])
                m = lgb.train(prm, dtr, num_boost_round=6000, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
                store[s, b] = m.predict(Xb, num_iteration=m.best_iteration, raw_score=True)
                its.append(m.best_iteration)
        log(f"fold {i}: top-6 skill columns {top}; best iterations (log-loss, focal per seed) {its}; AUC log-loss "
            f"{roc_auc_score(y[b], zc[:, b].mean(0)):.5f}, focal {roc_auc_score(y[b], zf[:, b].mean(0)):.5f}")
    ZC, ZF = zc.mean(0), zf.mean(0)
    np.savez(f_m, zc=ZC, zf=ZF)
cut_f2 = lambda z: f2(y, (z >= c.best_cut(y, z)).astype(int))
log(f"members alone (3 seeds, out of fold): C log-loss AUC {roc_auc_score(y, ZC):.5f}, F2 at its best cut {cut_f2(ZC):.5f}; F focal AUC "
    f"{roc_auc_score(y, ZF):.5f}, F2 {cut_f2(ZF):.5f}; correlation with the pipeline's LightGBM member: C {np.corrcoef(ZC, L[:, 0])[0, 1]:.4f}, "
    f"F {np.corrcoef(ZF, L[:, 0])[0, 1]:.4f}; Spearman C vs F {pd.Series(ZC).corr(pd.Series(ZF), method='spearman'):.4f}")
base_p = c.base_probs()
base_labels = [labels_from(p, fold_of(fl)) for p, (_, fl) in zip(base_p, SPLITS)]
for name, Z in (("C control: log-loss LightGBM on 49 features + ratios as a tenth member", ZC),
                ("F focal-loss LightGBM on 49 features + ratios as a tenth member", ZF)):
    S10 = np.column_stack([L, Z])
    cand = []
    for k, (label, fl) in enumerate(SPLITS):
        cand.append(labels_from(c.level2(fl, S_lr=S10, S_tr=S10), fold_of(fl)))
        log(f"{label}: {name.split(' ')[0]} {f2(y, cand[-1]):.5f} (base {f2(y, base_labels[k]):.5f})")
    c.report(name, cand, base_labels)
log("done")
