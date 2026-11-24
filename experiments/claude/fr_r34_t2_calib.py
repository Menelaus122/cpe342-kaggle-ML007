"""Round 34, Task 2: fewer calibration parameters, winsorised naive-Bayes evidence, smoother multiplier search (training rows only, CPU).

    python fr_r34_t2_calib.py

Round 33 (fm_r33): Dirichlet calibration of the KDE naive Bayes gained on average (+0.0002 to +0.0010) but dropped single folds by
up to 0.0019: 20 parameters per model, and the class multipliers are a Powell search on the raw macro-F1 (a step function).
Reference = the v20 blend nested as in fm_r33 / ev_r30: stacker (multinomial LogReg on four members' log-probabilities, out of fold
on the split) averaged with the out-of-fold KDE naive Bayes, weight w from (0.3 ... 0.9) and class multipliers chosen on the four
training folds.
  1a  temperature-only log-linear blend  p ~ p_stacker^(alpha/T1) * p_kde^((1-alpha)/T2):
      (i)  the two exponents a1 = alpha/T1, a2 = (1-alpha)/T2 fitted by the log-loss on the training folds (2 parameters), then the
           class multipliers;  (ii) with free class multipliers only the ratio a2 / a1 moves the argmax, so: p ~ p_s^(1-l) p_k^l with
           l from the reference grid, chosen with the multipliers on the macro-F1 like the reference weight.
  1b  winsorised evidence in the naive Bayes: per feature j and class c the one-vs-rest log-likelihood ratio
      log p(x_j | c) / p(x_j | not c) is clipped to [-M, M] before the sum (one extreme cell cannot decide the class);
      score_c = logit(prior_c) + sum_j clip(LLR_jc), probabilities = normalised sigmoid(score_c). M = infinity is the control
      (one-vs-rest form without clipping); M from (2, 3, 4) nested (chosen with w and the multipliers on the training folds).
      The per-feature densities are the pipeline's (src/task2.kde_nb_predict), out of fold on the pipeline folds.
  1c  the multiplier search on a smooth objective instead of Powell on the raw macro-F1 (reference probabilities unchanged):
      (A) coordinate ascent on the macro-F1 averaged over 40 bootstrap resamples of the training folds (fixed resamples);
      (B) L-BFGS on a soft macro-F1 (soft assignment softmax((log p + v) / 0.1), fixed in advance).
      The blend weight is chosen on the same smooth objective.
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
import scipy.sparse as sp  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import log_expit, logsumexp  # noqa: E402
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
OUT = Path("experiments/claude/outputs/r34")
OUT.mkdir(parents=True, exist_ok=True)
tr = pd.read_csv(DATA_DIRS[2] / "train.csv")
y = tr[task2.TARGET].values
folds = make_folds(y)
K = 4
softmax = lambda Z: np.exp(Z - logsumexp(Z, axis=1, keepdims=True))
lg = lambda P: np.log(np.clip(P, 1e-300, 1))


# ----------------------------------------------------------------------------- per-feature class log-likelihoods (pipeline's KDE NB)
def feature_loglik(fit, app, n_grid=2048):
    """log p(x_j | class) of rows `app` for every feature, fitted on rows `fit` as src/task2.kde_nb_predict does (0 for a missing
    or unseen cell). Returns (rows x features x K array, log priors)."""
    drop = [task2.ID_COL, "id", task2.TARGET, *task2.NOISE_COLS, *task2.NOMINAL_COLS]
    R = tr.drop(columns=[c for c in drop if c in tr.columns]).copy()
    for c, m in task2.ORDINAL_MAPS.items():
        R[c] = R[c].map(m).astype(float)
    cat = [c for c in R.columns if R[c].nunique() <= 12]
    cont = [c for c in R.columns if c not in cat]
    XR, CR = R[cont].values.astype(float), R[cat].values.astype(float)
    yt = y[fit]
    out = np.zeros((len(app), len(cat) + len(cont), K), np.float32)
    for j in range(len(cat)):
        vals = np.unique(CR[fit, j][~np.isnan(CR[fit, j])])
        x = CR[app, j]
        idx = np.minimum(np.searchsorted(vals, x), len(vals) - 1)
        ok = vals[idx] == x
        for k in range(K):
            v = CR[fit, j][yt == k]
            cnt = np.array([(v == u).sum() for u in vals]) + 1.0
            out[ok, j, k] = np.log(cnt / cnt.sum())[idx[ok]]
    for jj in range(len(cont)):
        col = XR[fit, jj]
        obs = col[~np.isnan(col)]
        lo, hi = obs.min(), obs.max()
        lo_m = lo if np.mean(obs == lo) >= 0.002 else -np.inf
        hi_m = hi if np.mean(obs == hi) >= 0.002 else np.inf
        grid = np.linspace(lo, hi, n_grid)
        step = grid[1] - grid[0]
        x = XR[app, jj]
        m = ~np.isnan(x)
        for k in range(K):
            v = col[yt == k]
            v = v[~np.isnan(v)]
            a, b = v == lo_m, v == hi_m
            pa, pb = (a.sum() + 0.5) / (len(v) + 1), (b.sum() + 0.5) / (len(v) + 1)
            inner = v[~a & ~b]
            sd = min(inner.std(), np.subtract(*np.percentile(inner, [75, 25])) / 1.34) or inner.std()
            h = 0.9 * sd * len(inner) ** -0.2
            H = np.histogram(inner, bins=np.append(grid - step / 2, grid[-1] + step / 2))[0].astype(float)
            kk = np.arange(-int(4 * h / step) - 1, int(4 * h / step) + 2) * step
            dens = np.convolve(H, np.exp(-0.5 * (kk / h) ** 2), mode="same")
            dens = np.maximum(dens, dens.max() * 1e-6)
            dens /= dens.sum() * step
            l = np.log(1 - pa - pb) + np.log(np.interp(x[m], grid, dens))
            l = np.where(x[m] == lo_m, np.log(pa), l)
            out[m, len(cat) + jj, k] = np.where(x[m] == hi_m, np.log(pb), l)
    return out, np.log(np.bincount(yt, minlength=K) / len(yt))


f_ll = OUT / "t2_kde_feature_loglik.npz"
if f_ll.exists():
    d = np.load(f_ll)
    LL, LPRI = d["ll"], d["lprior"]
else:
    LL, LPRI = None, np.zeros((len(y), K))
    for i, (a, b) in enumerate(folds):
        o, lp = feature_loglik(a, b)
        if LL is None:
            LL = np.zeros((len(y), o.shape[1], K), np.float32)
        LL[b], LPRI[b] = o, lp
        log(f"per-feature log-likelihoods fold {i} done ({o.shape[1]} features)")
    np.savez(f_ll, ll=LL, lprior=LPRI)
KDE = np.load("outputs/preds/t2_kde_nb.npz")["oof"]
REB = softmax(LPRI + LL.sum(1, dtype=np.float64))
print(f"rebuilt naive Bayes vs the pipeline's out-of-fold file: max |dp| {np.abs(REB - KDE).max():.2e}, argmax agreement "
      f"{np.mean(REB.argmax(1) == KDE.argmax(1)):.5f}", flush=True)


def winsor_nb(M):
    """Normalised one-vs-rest naive Bayes with every per-feature log-likelihood ratio clipped to [-M, M]."""
    pri = np.exp(LPRI)                                                       # rows x K
    s = np.zeros((len(y), K))
    clipped = 0.0
    for c in range(K):
        oth = [k for k in range(K) if k != c]
        w = np.log(pri[:, oth] / (1 - pri[:, [c]]))                          # rows x 3
        rest = logsumexp(LL[:, :, oth].astype(np.float64) + w[:, None, :], axis=2)
        r = LL[:, :, c] - rest
        clipped += float(np.mean(np.abs(r) > M)) / K
        s[:, c] = (LPRI[:, c] - np.log1p(-pri[:, c])) + np.clip(r, -M, M).sum(1)
    return softmax(log_expit(s)), clipped


def mf1(yy, pred):
    cm = np.bincount(yy * K + pred, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(float)
    den = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    return float(np.mean(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)))


def opt_mult(yy, P):
    """Reference: Powell on the raw macro-F1 of argmax(P * multipliers). Returns (log-multipliers with v_0 = 0, score)."""
    L = lg(P)
    loss = lambda v: -mf1(yy, np.argmax(L + np.r_[0.0, v][None, :], 1))
    res = minimize(loss, np.zeros(K - 1), method="Powell", options={"xtol": 1e-3, "ftol": 1e-6, "maxiter": 3000})
    v = res.x if res.fun < loss(np.zeros(K - 1)) else np.zeros(K - 1)
    return np.r_[0.0, v], -min(res.fun, loss(np.zeros(K - 1)))


def boot_weights(n, seed, B=40):
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(np.float32)        # B x n resampling counts


def opt_mult_boot(yy, P, W):
    """Coordinate ascent on the macro-F1 averaged over the bootstrap resamples W (B x n counts)."""
    L = lg(P)
    n = len(yy)
    rows = np.arange(n)

    def score(v):
        pred = np.argmax(L + v[None, :], 1)
        Mx = sp.csr_matrix((np.ones(n, np.float32), (rows, yy * K + pred)), shape=(n, K * K))
        cm = np.asarray((Mx.T @ W.T).T, float).reshape(-1, K, K)                       # B x K x K
        tp = np.einsum("bii->bi", cm)
        den = cm.sum(1) + cm.sum(2)
        return float(np.mean(np.where(den > 0, 2 * tp / np.maximum(den, 1e-9), 0.0)))

    v = np.zeros(K)
    best = score(v)
    for width, stepsz in ((1.0, 0.05), (0.05, 0.01), (0.05, 0.01)):
        for c in range(1, K):
            for cand in v[c] + np.arange(-width, width + 1e-9, stepsz):
                vv = v.copy()
                vv[c] = cand
                sc = score(vv)
                if sc > best + 1e-9:
                    best, v = sc, vv
    return v, best


def opt_mult_soft(yy, P, temp=0.1):
    """L-BFGS on the soft macro-F1 (soft assignment softmax((log p + v) / temp))."""
    L = lg(P)
    Y = np.eye(K)[yy]
    ny = Y.sum(0)

    def loss(v3):
        q = softmax((L + np.r_[0.0, v3][None, :]) / temp)
        return -float(np.mean(2 * (Y * q).sum(0) / (q.sum(0) + ny)))
    res = minimize(loss, np.zeros(K - 1), method="L-BFGS-B")
    return np.r_[0.0, res.x], -res.fun


def loglin_fit(yy, Ls, Lk):
    """Two exponents (a1, a2) >= 0 minimising the log-loss of softmax(a1 log p_s + a2 log p_k)."""
    idx = np.arange(len(yy))

    def nll(a):
        Z = a[0] * Ls + a[1] * Lk
        return -float(np.mean((Z - logsumexp(Z, axis=1, keepdims=True))[idx, yy]))
    return minimize(nll, np.array([0.5, 0.5]), method="L-BFGS-B", bounds=[(0, 5), (0, 5)]).x


S = np.column_stack([np.log(np.clip(np.load(f"outputs/preds/t2_{k}.npz")["oof"], 1e-6, 1)) for k in ("lgb", "xgb", "cat", "lr")])
WS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MS = (2.0, 3.0, 4.0)
QWC = {M: winsor_nb(M) for M in MS}
QW = {M: v[0] for M, v in QWC.items()}
QINF = winsor_nb(np.inf)[0]
print(f"naive Bayes alone (argmax macro-F1): pipeline {mf1(y, KDE.argmax(1)):.4f}; one-vs-rest, no clipping {mf1(y, QINF.argmax(1)):.4f}; "
      + "; ".join(f"M = {M:g}: {mf1(y, QW[M].argmax(1)):.4f} (ratios clipped {QWC[M][1]:.2%})" for M in MS), flush=True)
splits = [("pipeline folds", folds)] + [(f"split seed {s}", list(StratifiedKFold(5, shuffle=True, random_state=s).split(S, y))) for s in (1, 2, 3)]
NAMES = ["1a(i) log-linear, exponents by log-loss", "1a(ii) log-linear, weight by macro-F1", "1b one-vs-rest NB, no clipping (control)",
         "1b winsorised NB, M nested", "1b winsorised NB, M = 3", "1c(A) bootstrap-averaged coordinate ascent", "1c(B) soft macro-F1 L-BFGS"]
summary = {n: [] for n in NAMES}
for si, (label, fl) in enumerate(splits):
    Ps = np.zeros((len(y), K))
    for a, b in fl:
        Ps[b] = LogisticRegression(C=0.5, max_iter=3000).fit(S[a], y[a]).predict_proba(S[b])
    Ls, Lk = lg(Ps), lg(KDE)
    labs = {n: np.zeros(len(y), int) for n in ["reference"] + NAMES}
    notes = {n: [] for n in NAMES}
    for fi, (a, b) in enumerate(fl):
        lin = lambda w_, Q: (1 - w_) * Ps + w_ * Q
        apply = lambda P, v: np.argmax(lg(P[b]) + v[None, :], 1)
        # reference and the smooth solvers on the same probabilities
        W = boot_weights(len(a), 1000 * si + fi)
        ref, bo, so = {}, {}, {}
        for w_ in WS:
            P = lin(w_, KDE)
            ref[w_] = opt_mult(y[a], P[a])
            bo[w_] = opt_mult_boot(y[a], P[a], W)
            so[w_] = opt_mult_soft(y[a], P[a])
        for name, sc in (("reference", ref), (NAMES[5], bo), (NAMES[6], so)):
            wb = max(WS, key=lambda w_: sc[w_][1])
            labs[name][b] = apply(lin(wb, KDE), sc[wb][0])
            if name != "reference":
                notes[name].append(wb)
        # 1a(i): exponents by the log-loss
        a1, a2 = loglin_fit(y[a], Ls[a], Lk[a])
        P = softmax(a1 * Ls + a2 * Lk)
        labs[NAMES[0]][b] = apply(P, opt_mult(y[a], P[a])[0])
        notes[NAMES[0]].append(f"{a1:.2f}/{a2:.2f}")
        # 1a(ii): geometric weight on the macro-F1
        sc = {}
        for w_ in WS:
            P = softmax((1 - w_) * Ls + w_ * Lk)
            sc[w_] = (opt_mult(y[a], P[a]), P)
        wb = max(WS, key=lambda w_: sc[w_][0][1])
        labs[NAMES[1]][b] = apply(sc[wb][1], sc[wb][0][0])
        notes[NAMES[1]].append(wb)
        # 1b: one-vs-rest control, M nested, M = 3
        sc = {(M, w_): opt_mult(y[a], lin(w_, Q)[a]) for M, Q in list(QW.items()) + [(np.inf, QINF)] for w_ in WS}
        for name, cand in ((NAMES[2], [np.inf]), (NAMES[3], MS), (NAMES[4], [3.0])):
            Mb, wb = max(((M, w_) for M in cand for w_ in WS), key=lambda k_: sc[k_][1])
            labs[name][b] = apply(lin(wb, QINF if Mb == np.inf else QW[Mb]), sc[(Mb, wb)][0])
            notes[name].append(f"{Mb:g}/{wb}")
        log(f"{label} fold {fi} done")
    for name in NAMES:
        per = [mf1(y[b], labs[name][b]) - mf1(y[b], labs["reference"][b]) for _, b in fl]
        d = mf1(y, labs[name]) - mf1(y, labs["reference"])
        summary[name].append((d, sum(p > 0 for p in per), min(per)))
        print(f"{label}: {name:45s} {mf1(y, labs[name]):.4f} vs {mf1(y, labs['reference']):.4f}  delta {d:+.4f}  folds +{sum(p > 0 for p in per)}/5 "
              f"({' '.join(f'{p:+.4f}' for p in per)})  chosen {notes[name]}", flush=True)
    np.savez(OUT / f"t2_calib_labels_split{si}.npz", **{f"c{i}": v for i, v in enumerate(labs.values())})
print()
for name, v in summary.items():
    ok = np.mean([d for d, _, _ in v]) > 0 and all(n >= 3 and m >= -0.0005 for _, n, m in v)
    print(f"{name}: pooled mean delta {np.mean([d for d, _, _ in v]):+.4f}; folds better {' '.join(str(n) + '/5' for _, n, _ in v)}; "
          f"worst fold {' '.join(f'{m:+.4f}' for _, n, m in v)} -> {'PASS' if ok else 'fail'}")
log("done")
