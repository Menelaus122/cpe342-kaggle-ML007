"""Round 25, Task 2: a full 4 x 4 decision matrix instead of per-class multipliers (macro-F1 plug-in rule).

    python dd_r25_t2_matrix.py

Pipeline: stack (LogReg on log-probabilities of LGB/XGB/Cat/LR) -> y = argmax_c w_c p_c with w chosen on OOF (nested: 0.7564).
Challenger: y = argmax_c (M p)_c with M = diag(w) + Delta (off-diagonal terms let a row's Casual probability count towards Social,
etc.). Fitted by coordinate ascent on macro-F1 - lambda * ||M - diag(w)||_F^2, started from the pipeline multipliers.
Everything tuned is nested: chosen on 4 folds' OOF, applied to the 5th. lambda is reported per value and also chosen by an inner
4-fold split inside each training part (so the reported number for "lambda chosen" carries no selection bias).
Acceptance (round 25 portfolio rule): mean nested gain > 0 and better in >= 3/5 folds; the 95% CI is reported, not used as a gate.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

import common  # noqa: F401  (chdir to the repo root)
from src.metrics import optimize_class_weights  # noqa: E402
from src.modeling import make_folds  # noqa: E402
from src import task2  # noqa: E402

train, _ = task2.load()
y = train[task2.TARGET].values
folds = make_folds(y)
M4 = {k: np.load(f"outputs/preds/t2_{k}.npz") for k in ("lgb", "xgb", "cat", "lr")}
S = np.column_stack([np.log(np.clip(m["oof"], 1e-6, 1)) for m in M4.values()])
P = np.zeros((len(y), 4))
for tr, va in folds:
    P[va] = LogisticRegression(C=0.5, max_iter=3000).fit(S[tr], y[tr]).predict_proba(S[va])
fold_of = np.zeros(len(y), int)
for f, (_, va) in enumerate(folds):
    fold_of[va] = f


def mf1(yt, yp):
    cm = np.bincount(yt * 4 + yp, minlength=16).reshape(4, 4)
    tp = np.diag(cm)
    return float(np.mean(2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1)))


def fit_matrix(Pt, yt, w, lam, passes=6):
    M0 = np.diag(w / w.mean())
    M = M0.copy()
    obj = lambda M: mf1(yt, (Pt @ M.T).argmax(1)) - lam * np.sum((M - M0) ** 2)
    best = obj(M)
    steps = [0.2, 0.1, 0.05, 0.02, 0.01]
    for _ in range(passes):
        improved = False
        for i in range(4):
            for j in range(4):
                for s in steps:
                    for sign in (1, -1):
                        M2 = M.copy()
                        M2[i, j] += sign * s
                        v = obj(M2)
                        if v > best + 1e-7:
                            M, best, improved = M2, v, True
        if not improved:
            break
    return M


def fast_weights(Pt, yt):
    """Per-class multipliers by coordinate ascent on macro-F1 (inner folds only; the outer baseline uses the pipeline's optimiser)."""
    w = np.ones(4)
    best = mf1(yt, (Pt * w).argmax(1))
    for _ in range(6):
        improved = False
        for c in range(1, 4):
            for s in (0.2, 0.1, 0.05, 0.02, 0.01):
                for sign in (1, -1):
                    w2 = w.copy()
                    w2[c] *= np.exp(sign * s)
                    v = mf1(yt, (Pt * w2).argmax(1))
                    if v > best + 1e-7:
                        w, best, improved = w2, v, True
        if not improved:
            break
    return w


LAMS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]
base = np.zeros(len(y), int)
pred = {lam: np.zeros(len(y), int) for lam in LAMS}
pred_sel = np.zeros(len(y), int)
chosen = []
for f, (tr, va) in enumerate(folds):
    w = optimize_class_weights(y[tr], P[tr])[0]
    base[va] = (P[va] * w).argmax(1)
    for lam in LAMS:
        M = fit_matrix(P[tr], y[tr], w, lam)
        pred[lam][va] = (P[va] @ M.T).argmax(1)
    # lambda chosen by an inner 4-fold split of the training part
    inner = make_folds(y[tr], n_folds=4, seed=100 + f)
    sc = {}
    wis = [fast_weights(P[tr][a], y[tr][a]) for a, _ in inner]
    for lam in LAMS:
        ip = np.zeros(len(tr), int)
        for (a, b), wi in zip(inner, wis):
            ip[b] = (P[tr][b] @ fit_matrix(P[tr][a], y[tr][a], wi, lam).T).argmax(1)
        sc[lam] = mf1(y[tr], ip)
    lam_f = max(sc, key=sc.get)
    chosen.append(lam_f)
    pred_sel[va] = pred[lam_f][va]
    print(f"fold {f}: inner scores {{{', '.join(f'{k}: {v:.4f}' for k, v in sc.items())}}} -> lambda {lam_f}", flush=True)

rng = np.random.default_rng(0)
boot = [rng.integers(0, len(y), len(y)) for _ in range(500)]


def report(name, p):
    d = mf1(y, p) - mf1(y, base)
    pf = [mf1(y[fold_of == f], p[fold_of == f]) - mf1(y[fold_of == f], base[fold_of == f]) for f in range(5)]
    bs = [mf1(y[i], p[i]) - mf1(y[i], base[i]) for i in boot]
    print(f"  {name:28s} nested {mf1(y, p):.4f}  delta {d:+.4f} [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}]  "
          f"folds +{sum(x > 0 for x in pf)}/5  ({' '.join(f'{x:+.4f}' for x in pf)})  rows changed {np.mean(p != base):.3%}")


print(f"\npipeline (stack + nested multipliers): {mf1(y, base):.4f}")
for lam in LAMS:
    report(f"4x4 matrix, lambda {lam}", pred[lam])
report(f"4x4, lambda chosen {chosen}", pred_sel)
