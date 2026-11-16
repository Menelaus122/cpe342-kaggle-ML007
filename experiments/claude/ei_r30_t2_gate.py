"""Round 30, Task 2: when should the blend trust the stacker and when the KDE naive Bayes? (training rows only, OOF caches, CPU)

    python ei_r30_t2_gate.py

v20: P = 0.4 * stacker + 0.6 * KDE_NB (weight chosen on OOF rows with the class multipliers), label = argmax(P * m).
Part A (forensics, pipeline folds): confusion of the blend; rows where the stacker and the KDE NB disagree; who is right there, by
the number of missing cells, the two models' entropies and the number of values sitting on a clip bound; can "the stacker is right"
be predicted from those row descriptors (5-fold CV AUC, rows where exactly one of the two is right)?
Part B (method): input-aware gating (a mixture of two experts with a logistic gate; Jacobs et al. 1991, as used for model
averaging): w(x) = sigmoid(b0 + b . z(x)) is the KDE weight of row x, z = standardised [missing cells, stacker entropy, KDE entropy,
clip-bound count, stacker max prob, KDE max prob]. b is fitted by L2-regularised maximum likelihood of the blended probability on the
training folds (lambda fixed before the run: 1e-3 per row), then the class multipliers on the same folds; labels on the held-out
fold. Reference = the v20 procedure (constant w chosen from {0.3, ..., 0.9} with multipliers on the training folds).
Four stacker splits (pipeline folds, split seeds 1, 2, 3), as in rounds 27-28.
Acceptance (user, 2026-11-16): pooled mean delta > 0, better in >= 3/5 folds, no fold below -0.0005 (on each split; reported).
"""
import time
import warnings

import numpy as np
import pandas as pd
import psutil
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
import common  # noqa: F401,E402
from src import task2  # noqa: E402
from src.config import DATA_DIRS  # noqa: E402
from src.modeling import make_folds  # noqa: E402

t0 = time.time()
tr = pd.read_csv(DATA_DIRS[2] / "train.csv")                    # training rows only
y = tr[task2.TARGET].values
folds = make_folds(y)
K = 4
NAMES = ["Casual", "Grinder", "Social", "Whale"]


def mf1(yy, pred):
    cm = np.bincount(yy * K + pred, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(float)
    den = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    return float(np.mean(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)))


def opt_mult(yy, P, n_restarts=1, seed=0):
    rng = np.random.default_rng(seed)
    loss = lambda v: -mf1(yy, np.argmax(P * np.exp(np.r_[0.0, v])[None, :], 1))
    best_v, best = np.zeros(K - 1), loss(np.zeros(K - 1))
    for r in range(n_restarts):
        v0 = np.zeros(K - 1) if r == 0 else rng.normal(0, 0.3, K - 1)
        res = minimize(loss, v0, method="Powell", options={"xtol": 1e-3, "ftol": 1e-6, "maxiter": 3000})
        if res.fun < best:
            best, best_v = res.fun, res.x
    w = np.exp(np.r_[0.0, best_v])
    return w / w.sum() * K, -best


kde = np.load("outputs/preds/t2_kde_nb.npz")["oof"]
M = [np.load(f"outputs/preds/t2_{k}.npz")["oof"] for k in ("lgb", "xgb", "cat", "lr")]
S = np.column_stack([np.log(np.clip(m, 1e-6, 1)) for m in M])
ent = lambda P: -(P * np.log(np.clip(P, 1e-12, 1))).sum(1)
feat_cols = [c for c in tr.columns if c not in ("player_id", "id", task2.TARGET)]
num = [c for c in feat_cols if pd.api.types.is_numeric_dtype(tr[c])]
nan_n = tr[feat_cols].isna().sum(1).values.astype(float)
clip_n = np.zeros(len(y))
for c in num:
    v = tr[c]
    for b in (v.min(), v.max()):
        at = (v == b).values
        if at.mean() >= 0.002:
            clip_n += at
print(f"{len(y)} training rows; missing cells per row mean {nan_n.mean():.2f}; values on a clip bound per row mean {clip_n.mean():.2f} [{time.time() - t0:.0f}s]")

# ------------------------------------------------------------------ Part A: forensics on the pipeline folds
Ps = np.zeros((len(y), K))
for a, b in folds:
    Ps[b] = LogisticRegression(C=0.5, max_iter=3000).fit(S[a], y[a]).predict_proba(S[b])
Pb = 0.4 * Ps + 0.6 * kde
m_all, _ = opt_mult(y, Pb, n_restarts=3)
lab_b, lab_s, lab_k = np.argmax(Pb * m_all, 1), Ps.argmax(1), kde.argmax(1)
print(f"\nA. pipeline folds: macro-F1 stacker {mf1(y, lab_s):.4f}, KDE NB {mf1(y, lab_k):.4f}, blend + multipliers {mf1(y, lab_b):.4f} (in-sample multipliers)")
cm = pd.DataFrame(np.bincount(y * K + lab_b, minlength=K * K).reshape(K, K), index=[f"true {n}" for n in NAMES], columns=NAMES)
print(cm.to_string())
dis = lab_s != lab_k
both = (lab_s == y) & (lab_k == y)
print(f"stacker and KDE NB disagree on {dis.mean():.2%} of rows; there stacker right {np.mean(lab_s[dis] == y[dis]):.3f}, "
      f"KDE right {np.mean(lab_k[dis] == y[dis]):.3f}, blend right {np.mean(lab_b[dis] == y[dis]):.3f}, neither {np.mean((lab_s[dis] != y[dis]) & (lab_k[dis] != y[dis])):.3f}")
pairs = pd.Series([f"{NAMES[a]}/{NAMES[b]}" for a, b in zip(lab_s[dis], lab_k[dis])]).value_counts().head(6)
print("most common disagreements (stacker / KDE): " + ", ".join(f"{k} {v}" for k, v in pairs.items()))
H_s, H_k = ent(Ps), ent(kde)
desc = {"missing cells": nan_n, "values on a clip bound": clip_n, "stacker entropy": H_s, "KDE entropy": H_k,
        "entropy gap (stacker - KDE)": H_s - H_k}
one = dis & ((lab_s == y) ^ (lab_k == y))
print(f"\nrows where exactly one of the two is right: {one.sum()} ({one.mean():.2%}); share where the stacker is the right one, by quintile:")
for name, v in desc.items():
    qs = np.unique(np.quantile(v[one], [0.2, 0.4, 0.6, 0.8]))
    bins = np.digitize(v[one], qs)
    print(f"  {name:28s} " + "  ".join(f"q{k + 1}: {np.mean(lab_s[one][bins == k] == y[one][bins == k]):.3f} (n {np.sum(bins == k)})"
                                         for k in range(len(qs) + 1)))
Z = np.column_stack([nan_n, clip_n, H_s, H_k, Ps.max(1), kde.max(1)])
Z = (Z - Z.mean(0)) / Z.std(0)
t_ = (lab_s[one] == y[one]).astype(int)
o = np.zeros(one.sum())
for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(Z[one], t_):
    o[b] = LogisticRegression(max_iter=2000).fit(Z[one][a], t_[a]).predict_proba(Z[one][b])[:, 1]
print(f"'stacker is the right one' predictable from the six row descriptors: AUC {roc_auc_score(t_, o):.3f} (0.5 = no signal) [{time.time() - t0:.0f}s]")


# ------------------------------------------------------------------ Part B: gated blend, nested on four stacker splits
def fit_gate(Ps_, Pk_, y_, Z_, lam):
    n = len(y_)
    ps, pk = Ps_[np.arange(n), y_], Pk_[np.arange(n), y_]

    def f(beta):
        w = 1 / (1 + np.exp(-(beta[0] + Z_ @ beta[1:])))
        p = np.clip((1 - w) * ps + w * pk, 1e-12, 1)
        g = (pk - ps) / p * w * (1 - w)                    # d log p / d logit(w)
        val = -np.log(p).sum() + lam * n * (beta[1:] ** 2).sum()
        grad = -np.r_[g.sum(), Z_.T @ g] + np.r_[0.0, 2 * lam * n * beta[1:]]
        return val, grad
    b0 = np.r_[np.log(0.6 / 0.4), np.zeros(Z_.shape[1])]
    return minimize(f, b0, jac=True, method="L-BFGS-B").x


WS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
LAM = 1e-3
splits = [("pipeline folds", folds)] + [(f"split seed {s}", list(StratifiedKFold(5, shuffle=True, random_state=s).split(S, y))) for s in (1, 2, 3)]
summary = []
for label, fl in splits:
    Ps = np.zeros((len(y), K))
    for a, b in fl:
        Ps[b] = LogisticRegression(C=0.5, max_iter=3000).fit(S[a], y[a]).predict_proba(S[b])
    Zs = np.column_stack([nan_n, clip_n, ent(Ps), ent(kde), Ps.max(1), kde.max(1)])
    ref, gat = np.zeros(len(y), int), np.zeros(len(y), int)
    per, chosen, betas = [], [], []
    for i, (a, b) in enumerate(fl):
        scores = {w: opt_mult(y[a], ((1 - w) * Ps + w * kde)[a]) for w in WS}
        w_best = max(WS, key=lambda w: scores[w][1])
        ref[b] = np.argmax(((1 - w_best) * Ps + w_best * kde)[b] * scores[w_best][0][None, :], 1)
        mu, sd = Zs[a].mean(0), Zs[a].std(0)
        beta = fit_gate(Ps[a], kde[a], y[a], (Zs[a] - mu) / sd, LAM)
        wrow = 1 / (1 + np.exp(-(beta[0] + ((Zs - mu) / sd) @ beta[1:])))
        Pg = (1 - wrow[:, None]) * Ps + wrow[:, None] * kde
        mg, _ = opt_mult(y[a], Pg[a])
        gat[b] = np.argmax(Pg[b] * mg[None, :], 1)
        per.append(mf1(y[b], gat[b]) - mf1(y[b], ref[b]))
        chosen.append(w_best)
        betas.append(beta)
    d = mf1(y, gat) - mf1(y, ref)
    summary.append((d, sum(p > 0 for p in per), min(per)))
    bm = np.mean(betas, 0)
    print(f"\nB. {label}: reference (constant w nested, picks {chosen}) {mf1(y, ref):.4f}; gated {mf1(y, gat):.4f}; delta {d:+.4f}  "
          f"folds +{sum(p > 0 for p in per)}/5 ({' '.join(f'{p:+.4f}' for p in per)}); labels changed {np.mean(gat != ref):.2%}")
    print(f"   mean gate coefficients (standardised): intercept {bm[0]:+.2f}; missing {bm[1]:+.2f}, clip {bm[2]:+.2f}, H_stacker {bm[3]:+.2f}, "
          f"H_KDE {bm[4]:+.2f}, maxp_stacker {bm[5]:+.2f}, maxp_KDE {bm[6]:+.2f} [{time.time() - t0:.0f}s]", flush=True)
print(f"\npooled over the four splits: mean delta {np.mean([s[0] for s in summary]):+.4f}; folds better per split "
      f"{' '.join(str(s[1]) + '/5' for s in summary)}; worst fold per split {' '.join(f'{s[2]:+.4f}' for s in summary)}")
print(f"peak working set {psutil.Process().memory_info().peak_wset / 2**20:.0f} MB; {time.time() - t0:.0f}s")
