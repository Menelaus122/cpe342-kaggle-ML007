"""Round 33, Task 2 (added from the literature): the KDE naive Bayes in a pooled normal-score space (training rows only, CPU).

    python fp_r33_t2_kde_zspace.py

The pipeline's KDE naive Bayes (src/task2.kde_nb_predict) estimates each class-conditional density with a Gaussian kernel on the raw
scale, one Silverman bandwidth per class and feature. Two known weaknesses of a fixed-bandwidth kernel: boundary bias on bounded
features (kernel mass leaks past the limits, so the density is under-estimated next to them) and over-smoothed modes / under-smoothed
tails on skewed features (round 21: several marginals are gamma-like). Transformation kernel density estimation (Wand, Marron &
Ruppert, JASA 1991; probit transformation on bounded support, Geenens, JASA 2014): map the feature through a monotone transform to a
scale without boundaries and estimate the density there. Here T = normal score of the pooled interior training values of the fold
(mid-rank ECDF -> probit), so the pooled distribution is N(0, 1). T is the same for every class, so its Jacobian cancels in the class
posterior: the naive Bayes only needs the class densities of z = T(x). The clip-bound point masses, the low-cardinality features,
the priors and the missing-value rule stay exactly as in the pipeline.
Candidates (each replaces the KDE naive Bayes in the v20 blend; KDE weight and class multipliers chosen on the 4 training folds):
  z-space KDE, Silverman bandwidth in z;   z-space KDE, 0.7 x Silverman;
  geometric mean of the raw-scale and the z-space posteriors (0.5 / 0.5 in the log).
Splits: pipeline folds + stacker split seeds 1, 2, 3 (the KDE stays out of fold on the pipeline folds).
Acceptance (user, 2026-11-16): pooled delta > 0, >= 3/5 folds better on each split, no fold below -0.0005.
"""
import os
import time
import warnings
from pathlib import Path

NT = int(os.environ.get("R33_THREADS", 2))
os.environ.setdefault("OMP_NUM_THREADS", str(NT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

warnings.filterwarnings("ignore")
import common  # noqa: F401,E402
from src import task2  # noqa: E402
from src.config import DATA_DIRS  # noqa: E402
from src.modeling import make_folds  # noqa: E402

psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
t0 = time.time()
log = lambda m: print(f"{m} [{time.time() - t0:.0f}s]", flush=True)
OUT = Path("experiments/claude/outputs/r33")
OUT.mkdir(parents=True, exist_ok=True)
tr = pd.read_csv(DATA_DIRS[2] / "train.csv")
y = tr[task2.TARGET].values
folds = make_folds(y)
K = 4
drop = [task2.ID_COL, "id", task2.TARGET, *task2.NOISE_COLS, *task2.NOMINAL_COLS]
R = tr.drop(columns=[c for c in drop if c in tr.columns]).copy()
for c, m in task2.ORDINAL_MAPS.items():
    R[c] = R[c].map(m).astype(float)
cat = [c for c in R.columns if R[c].nunique() <= 12]
cont = [c for c in R.columns if c not in cat]
XR, CR = R[cont].values.astype(float), R[cat].values.astype(float)
softmax = lambda Z: (lambda E: E / E.sum(1, keepdims=True))(np.exp(Z - Z.max(1, keepdims=True)))
lg = lambda P: np.log(np.clip(P, 1e-300, 1))
GRID = np.linspace(-5.5, 5.5, 2048)
STEP = GRID[1] - GRID[0]


def zspace_scores(tr_idx, ap_idx, hmults):
    """Class log-scores (prior + sum of feature log-likelihoods) of rows ap_idx for each bandwidth multiplier, fitted on tr_idx."""
    yt = y[tr_idx]
    base = np.tile(np.log(np.bincount(yt, minlength=K) / len(yt)), (len(ap_idx), 1))
    for j in range(len(cat)):                                              # low-cardinality features: as the pipeline
        vals = np.unique(CR[tr_idx, j][~np.isnan(CR[tr_idx, j])])
        x = CR[ap_idx, j]
        idx = np.minimum(np.searchsorted(vals, x), len(vals) - 1)
        ok = vals[idx] == x
        for k in range(K):
            v = CR[tr_idx, j][yt == k]
            cnt = np.array([(v == u).sum() for u in vals]) + 1.0
            base[ok, k] += np.log(cnt / cnt.sum())[idx[ok]]
    out = {h: base.copy() for h in hmults}
    for jj in range(len(cont)):
        col = XR[tr_idx, jj]
        obs = col[~np.isnan(col)]
        lo, hi = obs.min(), obs.max()
        lo_m = lo if np.mean(obs == lo) >= 0.002 else -np.inf
        hi_m = hi if np.mean(obs == hi) >= 0.002 else np.inf
        pooled = np.sort(obs[(obs != lo_m) & (obs != hi_m)])
        n = len(pooled)
        T = lambda v: norm.ppf(np.clip((np.searchsorted(pooled, v, side="left") + np.searchsorted(pooled, v, side="right")) / 2 / n,
                                       0.5 / n, 1 - 0.5 / n))
        x = XR[ap_idx, jj]
        m = ~np.isnan(x)
        zx = T(x[m])
        for k in range(K):
            v = col[yt == k]
            v = v[~np.isnan(v)]
            a, b = v == lo_m, v == hi_m
            pa, pb = (a.sum() + 0.5) / (len(v) + 1), (b.sum() + 0.5) / (len(v) + 1)
            z = T(v[~a & ~b])
            sd = min(z.std(), np.subtract(*np.percentile(z, [75, 25])) / 1.34) or z.std()
            H = np.histogram(z, bins=np.append(GRID - STEP / 2, GRID[-1] + STEP / 2))[0].astype(float)
            for hm in hmults:
                h = hm * 0.9 * sd * len(z) ** -0.2
                kk = np.arange(-int(4 * h / STEP) - 1, int(4 * h / STEP) + 2) * STEP
                dens = np.convolve(H, np.exp(-0.5 * (kk / h) ** 2), mode="same")
                dens = np.maximum(dens, dens.max() * 1e-6)
                dens /= dens.sum() * STEP
                l = np.log(1 - pa - pb) + np.log(np.interp(zx, GRID, dens))
                l = np.where(x[m] == lo_m, np.log(pa), l)
                out[hm][m, k] += np.where(x[m] == hi_m, np.log(pb), l)
    return out


HM = (1.0, 0.7)
f_z = OUT / "t2_kde_zspace_oof.npz"
if f_z.exists():
    d = np.load(f_z)
    Z = {h: d[f"h{h}"] for h in HM}
else:
    Z = {h: np.zeros((len(y), K)) for h in HM}
    for i, (a, b) in enumerate(folds):
        o = zspace_scores(a, b, HM)
        for h in HM:
            Z[h][b] = softmax(o[h])
        log(f"z-space KDE fold {i} done")
    np.savez(f_z, **{f"h{h}": Z[h] for h in HM})
KDE = np.load("outputs/preds/t2_kde_nb.npz")["oof"]
ll = lambda P: -np.mean(lg(P)[np.arange(len(y)), y])


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


CANDS = {"z-space KDE (Silverman)": Z[1.0], "z-space KDE (0.7 x Silverman)": Z[0.7],
         "geometric mean raw + z-space": softmax(0.5 * lg(KDE) + 0.5 * lg(Z[1.0]))}
print(f"alone (argmax): raw-scale KDE NB {mf1(y, KDE.argmax(1)):.4f} (log-loss {ll(KDE):.4f}); "
      + "; ".join(f"{n} {mf1(y, Q.argmax(1)):.4f} ({ll(Q):.4f})" for n, Q in CANDS.items())
      + f"; argmax agreement raw vs z-space {np.mean(KDE.argmax(1) == Z[1.0].argmax(1)):.4f}", flush=True)
S = np.column_stack([np.log(np.clip(np.load(f"outputs/preds/t2_{k}.npz")["oof"], 1e-6, 1)) for k in ("lgb", "xgb", "cat", "lr")])
WS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
splits = [("pipeline folds", folds)] + [(f"split seed {s}", list(StratifiedKFold(5, shuffle=True, random_state=s).split(S, y))) for s in (1, 2, 3)]
summary = {n: [] for n in CANDS}
for label, fl in splits:
    Ps = np.zeros((len(y), K))
    for a, b in fl:
        Ps[b] = LogisticRegression(C=0.5, max_iter=3000).fit(S[a], y[a]).predict_proba(S[b])
    labs = {}
    for name, Q in [("reference", KDE)] + list(CANDS.items()):
        lab = np.zeros(len(y), int)
        for a, b in fl:
            sc = {w_: opt_mult(y[a], ((1 - w_) * Ps + w_ * Q)[a]) for w_ in WS}
            wb_ = max(WS, key=lambda w_: sc[w_][1])
            lab[b] = np.argmax(((1 - wb_) * Ps + wb_ * Q)[b] * sc[wb_][0][None, :], 1)
        labs[name] = lab
    for name in CANDS:
        per = [mf1(y[b], labs[name][b]) - mf1(y[b], labs["reference"][b]) for _, b in fl]
        d = mf1(y, labs[name]) - mf1(y, labs["reference"])
        summary[name].append((d, sum(p > 0 for p in per), min(per)))
        print(f"{label}: {name:30s} {mf1(y, labs[name]):.4f} vs {mf1(y, labs['reference']):.4f}  delta {d:+.4f}  folds +{sum(p > 0 for p in per)}/5 "
              f"({' '.join(f'{p:+.4f}' for p in per)})", flush=True)
    log(f"{label} done")
for name, v in summary.items():
    ok = np.mean([d for d, _, _ in v]) > 0 and all(n >= 3 and m >= -0.0005 for _, n, m in v)
    print(f"{name}: pooled mean delta {np.mean([d for d, _, _ in v]):+.4f}; folds better {' '.join(str(n) + '/5' for _, n, _ in v)}; "
          f"worst fold {' '.join(f'{m:+.4f}' for _, n, m in v)} -> {'PASS' if ok else 'fail'}")
log("done")
