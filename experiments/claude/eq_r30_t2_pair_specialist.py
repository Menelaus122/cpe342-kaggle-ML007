"""Round 30, Task 2: a pairwise specialist for the most confused pair, Casual <-> Grinder (training rows only, CPU, 2 threads).

    python eq_r30_t2_pair_specialist.py

Pairwise coupling (Hastie & Tibshirani 1998): a binary model trained only on Casual and Grinder rows re-splits the probability mass
the v20 blend (0.4 stacker + 0.6 KDE NB) gives to those two classes: P'_C = (P_C + P_G)(1 - q), P'_G = (P_C + P_G) q, with q the
specialist's P(Grinder | Casual or Grinder); other classes unchanged; then the class multipliers. The specialist (LightGBM, 300 trees,
lr 0.03, 31 leaves, 200 rows per leaf; fixed before the run) sees the raw features plus the blend's four probabilities.
Nested on the pipeline folds: on the 4 training folds the specialist's own OOF q comes from an inner 4-fold CV (so the multipliers
are chosen on honest probabilities), a fit on all 4 folds gives q for the 5th. Reference = the v20 procedure on the same folds
(KDE weight w from {0.3..0.9} and multipliers chosen on the 4 folds). Acceptance (user, 2026-11-16): pooled delta > 0, >= 3/5
folds, no fold below -0.0005.
"""
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
import common  # noqa: F401,E402
from src import task2  # noqa: E402
from src.config import DATA_DIRS  # noqa: E402
from src.modeling import make_folds  # noqa: E402

t0 = time.time()
tr = pd.read_csv(DATA_DIRS[2] / "train.csv")
y = tr[task2.TARGET].values
folds = make_folds(y)
K, C, G = 4, 0, 1


def mf1(yy, pred):
    cm = np.bincount(yy * K + pred, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(float)
    den = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    return float(np.mean(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)))


def opt_mult(yy, P):
    loss = lambda v: -mf1(yy, np.argmax(P * np.exp(np.r_[0.0, v])[None, :], 1))
    res = minimize(loss, np.zeros(K - 1), method="Powell", options={"xtol": 1e-3, "ftol": 1e-6, "maxiter": 3000})
    v = res.x if res.fun < loss(np.zeros(K - 1)) else np.zeros(K - 1)
    w = np.exp(np.r_[0.0, v])
    return w / w.sum() * K, -min(res.fun, loss(np.zeros(K - 1)))


kde = np.load("outputs/preds/t2_kde_nb.npz")["oof"]
S = np.column_stack([np.log(np.clip(np.load(f"outputs/preds/t2_{k}.npz")["oof"], 1e-6, 1)) for k in ("lgb", "xgb", "cat", "lr")])
Ps = np.zeros((len(y), K))
for a, b in folds:
    Ps[b] = LogisticRegression(C=0.5, max_iter=3000).fit(S[a], y[a]).predict_proba(S[b])
feat = [c for c in tr.columns if c not in ("player_id", "id", task2.TARGET)]
X = tr[feat].copy()
for c in X.columns:
    if not pd.api.types.is_numeric_dtype(X[c]):
        X[c] = X[c].astype("category")
PRM = dict(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, n_jobs=2, verbose=-1)
WS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def couple(P, q):
    out = P.copy()
    m = P[:, C] + P[:, G]
    out[:, C], out[:, G] = m * (1 - q), m * q
    return out


def spec_fit_predict(rows_fit, rows_pred, Pb):
    pair = rows_fit[np.isin(y[rows_fit], (C, G))]
    Xf = X.iloc[pair].assign(**{f"pb{k}": Pb[pair, k] for k in range(K)})
    m = lgb.LGBMClassifier(**PRM).fit(Xf, (y[pair] == G).astype(int))
    return m.predict_proba(X.iloc[rows_pred].assign(**{f"pb{k}": Pb[rows_pred, k] for k in range(K)}))[:, 1]


ref, cand, per, ws = np.zeros(len(y), int), np.zeros(len(y), int), [], []
for i, (a, b) in enumerate(folds):
    sc = {w: opt_mult(y[a], ((1 - w) * Ps + w * kde)[a]) for w in WS}
    w = max(WS, key=lambda w_: sc[w_][1])
    ws.append(w)
    Pb = (1 - w) * Ps + w * kde
    ref[b] = np.argmax(Pb[b] * sc[w][0][None, :], 1)
    q_in = np.zeros(len(y))
    for ia, ib in StratifiedKFold(4, shuffle=True, random_state=i).split(a, y[a]):
        q_in[a[ib]] = spec_fit_predict(a[ia], a[ib], Pb)
    q_in[b] = spec_fit_predict(a, b, Pb)
    Pc = couple(Pb, q_in)
    mc, _ = opt_mult(y[a], Pc[a])
    cand[b] = np.argmax(Pc[b] * mc[None, :], 1)
    per.append(mf1(y[b], cand[b]) - mf1(y[b], ref[b]))
    print(f"fold {i}: w {w}, reference {mf1(y[b], ref[b]):.4f}, coupled {mf1(y[b], cand[b]):.4f}, delta {per[-1]:+.4f} [{time.time() - t0:.0f}s]", flush=True)
d = mf1(y, cand) - mf1(y, ref)
ok = d > 0 and sum(p > 0 for p in per) >= 3 and min(per) >= -0.0005
cg = np.isin(y, (C, G))
print(f"\nnested macro-F1: reference {mf1(y, ref):.4f}, pairwise-coupled {mf1(y, cand):.4f}; delta {d:+.4f}; folds +{sum(p > 0 for p in per)}/5, "
      f"worst {min(per):+.4f} -> {'PASS' if ok else 'fail'}; labels changed {np.mean(cand != ref):.2%}; "
      f"Casual/Grinder accuracy among true C/G rows {np.mean(ref[cg] == y[cg]):.4f} -> {np.mean(cand[cg] == y[cg]):.4f} [{time.time() - t0:.0f}s]")
