"""Round 28, Task 2: temperature of the KDE naive Bayes before it is blended with the stacker (OOF caches only, CPU, one thread).

    python ed_r28_t2_kde_temperature.py

v19 / v20: P = (1 - w) * stacker + w * KDE_NB, w = 0.6 chosen on the OOF rows with the class multipliers, then argmax(P * m).
The KDE NB multiplies 38 per-feature likelihoods as if independent, so its probabilities are pushed towards 0 and 1. Tempered:
    q_T = KDE_NB ** (1 / T), renormalised,  T in {1, 1.5, 2, 2.5, 3, 4}
then the same blend and multipliers. Inputs: outputs/preds/t2_{lgb,xgb,cat,lr}.npz (stacker members) and t2_kde_nb.npz (the
pipeline's KDE NB, out of fold on the pipeline folds). Level 2 repeated on the pipeline folds and split seeds 1, 2, 3 (round 27).
Nested everywhere: on each split, T, w (KDE weight in {0.3, 0.4, ..., 0.9}; nested choices in round 27 were 0.5-0.7) and the multipliers are chosen on 4
folds by their macro-F1 with multipliers and applied to the 5th. w runs to 0.9 here (a flatter KDE may need more weight).
    reference      T = 1, w nested                 (the v19 procedure, nested)
    T fixed        T = 1.5 ... 4, w nested
    T nested       T and w chosen together
Acceptance: mean gain > 0 over the splits and >= 4/5 folds.
The multiplier search is src.metrics.optimize_class_weights (Powell, seed 0) with a bincount macro-F1 (identical values, about 20x
faster than sklearn's f1_score) and one start instead of five (five took 14 s per search, 3 hours for the grid); reference and
candidates use the same search.
"""
import os
import time
import warnings

import numpy as np
import pandas as pd
import psutil
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
import common  # noqa: F401,E402
from src import task2  # noqa: E402
from src.config import DATA_DIRS  # noqa: E402
from src.modeling import make_folds  # noqa: E402

t0 = time.time()
y = pd.read_csv(DATA_DIRS[2] / "train.csv", usecols=[task2.TARGET])[task2.TARGET].values      # the target column only
folds = make_folds(y)
K = 4


def mf1(yy, pred):
    cm = np.bincount(yy * K + pred, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(float)
    den = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    return float(np.mean(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)))


def opt_mult(yy, P, n_restarts=5, seed=0):
    """src.metrics.optimize_class_weights with the fast macro-F1."""
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
TS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
WS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def temper(P, T):
    L = np.log(np.clip(P, 1e-300, 1)) / T
    Q = np.exp(L - L.max(1, keepdims=True))
    return Q / Q.sum(1, keepdims=True)


Q = {T: temper(kde, T) for T in TS}
nll = lambda P: -np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)))
print(f"{len(y)} rows. Calibration of the KDE NB out of fold (descriptive): log-loss / mean top probability / accuracy")
for T in TS:
    print(f"  T = {T}: {nll(Q[T]):.4f} / {Q[T].max(1).mean():.4f} / {np.mean(Q[T].argmax(1) == y):.4f}", flush=True)
splits = [("pipeline folds", folds)] + [(f"split seed {s}", list(StratifiedKFold(5, shuffle=True, random_state=s).split(S, y))) for s in (1, 2, 3)]
def boots(n, B):
    """The same B bootstrap draws for every candidate, generated one at a time (no B x n index matrix in memory)."""
    r = np.random.default_rng(0)
    for _ in range(B):
        yield r.integers(0, n, n).astype(np.int32)


summary = {}
for label, fl in splits:
    Ps = np.zeros((len(y), K))
    for tr, va in fl:
        Ps[va] = LogisticRegression(C=0.5, max_iter=3000).fit(S[tr], y[tr]).predict_proba(S[va])
    if label == "pipeline folds":
        print(f"stacker out of fold: log-loss {nll(Ps):.4f}, mean top probability {Ps.max(1).mean():.4f}, accuracy {np.mean(Ps.argmax(1) == y):.4f}")
    fo = np.zeros(len(y), int)
    for i, (_, va) in enumerate(fl):
        fo[va] = i
    # every (T, w) on every fold: multipliers and score on the 4 training folds, labels on the 5th
    lab = {(T, w): np.zeros(len(y), np.int8) for T in TS for w in WS}
    trs = {(T, w): np.zeros(5) for T in TS for w in WS}
    for i, (tr, va) in enumerate(fl):
        for T in TS:
            for w in WS:
                if w == 0.0 and T != 1.0:                       # w = 0 does not depend on T
                    lab[T, w][va], trs[T, w][i] = lab[1.0, 0.0][va], trs[1.0, 0.0][i]
                    continue
                P = (1 - w) * Ps + w * Q[T]
                m, s = opt_mult(y[tr], P[tr], n_restarts=1)
                lab[T, w][va] = np.argmax(P[va] * m[None, :], 1)
                trs[T, w][i] = s
        print(f"  {label}: fold {i} done [{time.time() - t0:.0f}s]", flush=True)

    def pick(keys):
        out, ch = np.zeros(len(y), np.int8), []
        for i in range(5):
            best = max(keys, key=lambda k: trs[k][i] + 1e-12 * (k == keys[0]))
            ch.append(best)
            out[fo == i] = lab[best][fo == i]
        return out, ch

    base, ch_b = pick([(1.0, w) for w in WS])
    cands = {f"T = {T}, w nested": pick([(T, w) for w in WS]) for T in TS[1:]}
    cands["T and w nested"] = pick([(T, w) for T in TS for w in WS])
    print(f"\n{label}: reference (T = 1, w nested) macro-F1 {mf1(y, base):.4f}, w chosen {[w for _, w in ch_b]}")
    for k, (p, ch) in cands.items():
        d = mf1(y, p) - mf1(y, base)
        pf = [mf1(y[fo == i], p[fo == i]) - mf1(y[fo == i], base[fo == i]) for i in range(5)]
        bs = [mf1(y[b], p[b]) - mf1(y[b], base[b]) for b in boots(len(y), 500)]
        summary.setdefault(k, []).append((d, sum(x > 0 for x in pf)))
        print(f"  {k:18s} {mf1(y, p):.4f}  delta {d:+.4f} [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}]  folds +{sum(x > 0 for x in pf)}/5 "
              f"({' '.join(f'{x:+.4f}' for x in pf)})  labels changed {np.mean(p != base):.2%}  chosen (T, w) {ch}", flush=True)
    print("  in-sample macro-F1 with multipliers on the 4 training folds, mean over folds, by T (rows) and w (columns):")
    for T in TS:
        print(f"    T {T}: " + "  ".join(f"{w}: {trs[T, w].mean():.4f}" for w in WS))
print("\nmean delta over the 4 splits (folds better per split):")
for k, v in summary.items():
    print(f"  {k:18s} {np.mean([d for d, _ in v]):+.4f}  ({' '.join(str(n) + '/5' for _, n in v)})")
print(f"peak working set {psutil.Process().memory_info().peak_wset / 2**20:.0f} MB; {time.time() - t0:.0f}s")
