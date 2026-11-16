"""Round 30 (Jackpot loop), Task 2: per-feature weights for the KDE naive Bayes, and a joint 2-D density for one dependent pair.

    python ev_r30_t2_kde_variants.py

The pipeline's KDE NB (src/task2.kde_nb_predict) adds one log-density per feature with weight 1. Here its per-feature, per-class
log-likelihoods are rebuilt out of fold on the pipeline folds (the same code path; the plain sum reproduces the cached OOF), then:
  weighted NB   log P(c | x) = log pi_c + sum_j w_j l_j(x, c), w_j in [0, 2] fitted by maximum likelihood (L-BFGS-B, small L2 pull
                towards 1) on the training folds of each split, applied to the held-out fold ("attribute-weighted naive Bayes",
                Zaidi et al., JMLR 2013)
  joint pair    l_a + l_b for (avg_monthly_spending, spending_frequency) replaced by a class-conditional 2-D kernel density (Whale
                has a latent intensity factor loading on both, round 19); rows with either value missing or on a clip bound keep
                the two marginals
Each candidate replaces the KDE NB in the v20 blend (KDE weight w in {0.3..0.9} and class multipliers chosen on the 4 training folds).
Splits: pipeline folds + stacker split seeds 1, 2, 3 (the KDE components stay out of fold on the pipeline folds, as in rounds 27-28).
Acceptance (user, 2026-11-16): pooled delta > 0, >= 3/5 folds better on each split, no fold below -0.0005.
"""
import time
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
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
K = 4
drop = [task2.ID_COL, "id", task2.TARGET, *task2.NOISE_COLS, *task2.NOMINAL_COLS]
R = tr.drop(columns=[c for c in drop if c in tr.columns]).copy()
for c, m in task2.ORDINAL_MAPS.items():
    R[c] = R[c].map(m).astype(float)
cat = [c for c in R.columns if R[c].nunique() <= 12]
cont = [c for c in R.columns if c not in cat]
XR, CR = R[cont].values.astype(float), R[cat].values.astype(float)
PAIR = [cont.index("avg_monthly_spending"), cont.index("spending_frequency")]
J = len(cat) + len(cont)


def components(tr_idx, ap_idx, n_grid=2048):
    """Per-feature class log-likelihoods (n_ap, J, 4) and log priors, fitted on tr_idx (as task2.kde_nb_predict), plus the joint
    2-D class log-density of the pair (n_ap, 4) and a mask of the rows where it applies."""
    yt = y[tr_idx]
    out = np.zeros((len(ap_idx), J, K))
    for j in range(len(cat)):
        vals = np.unique(CR[tr_idx, j][~np.isnan(CR[tr_idx, j])])
        x = CR[ap_idx, j]
        idx = np.minimum(np.searchsorted(vals, x), len(vals) - 1)
        ok = vals[idx] == x
        for k in range(K):
            v = CR[tr_idx, j][yt == k]
            cnt = np.array([(v == u).sum() for u in vals]) + 1.0
            out[ok, j, k] = np.log(cnt / cnt.sum())[idx[ok]]
    bounds = {}
    for jj in range(len(cont)):
        j = len(cat) + jj
        col = XR[tr_idx, jj]
        obs = col[~np.isnan(col)]
        lo, hi = obs.min(), obs.max()
        lo_m = lo if np.mean(obs == lo) >= 0.002 else -np.inf
        hi_m = hi if np.mean(obs == hi) >= 0.002 else np.inf
        bounds[jj] = (lo, hi, lo_m, hi_m)
        grid = np.linspace(lo, hi, n_grid)
        step = grid[1] - grid[0]
        x = XR[ap_idx, jj]
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
            out[m, j, k] = np.where(x[m] == hi_m, np.log(pb), l)
    # joint 2-D density of the pair on the interior (both observed, neither on a clip bound)
    ja, jb = PAIR
    (loa, hia, lma, hma), (lob, hib, lmb, hmb) = bounds[ja], bounds[jb]
    inside = lambda A, B: (~np.isnan(A) & ~np.isnan(B) & (A != lma) & (A != hma) & (B != lmb) & (B != hmb))
    A, B = XR[ap_idx, ja], XR[ap_idx, jb]
    use = inside(A, B)
    joint = np.zeros((len(ap_idx), K))
    G = 256
    ea, eb = np.linspace(loa, hia, G + 1), np.linspace(lob, hib, G + 1)
    for k in range(K):
        At, Bt = XR[tr_idx, ja][yt == k], XR[tr_idx, jb][yt == k]
        allv = ~np.isnan(At) & ~np.isnan(Bt)
        inn = inside(At, Bt)
        p_in = (inn.sum() + 0.5) / (allv.sum() + 1)                     # P(both interior | class, both observed)
        H, _, _ = np.histogram2d(At[inn], Bt[inn], bins=[ea, eb])
        n = inn.sum()
        sa = 0.9 * min(At[inn].std(), np.subtract(*np.percentile(At[inn], [75, 25])) / 1.34) * n ** (-1 / 6)
        sb = 0.9 * min(Bt[inn].std(), np.subtract(*np.percentile(Bt[inn], [75, 25])) / 1.34) * n ** (-1 / 6)
        D = gaussian_filter(H, sigma=(sa / (ea[1] - ea[0]), sb / (eb[1] - eb[0])), mode="constant")
        D = np.maximum(D, D.max() * 1e-6)
        D /= D.sum() * (ea[1] - ea[0]) * (eb[1] - eb[0])
        ia = np.clip(np.searchsorted(ea, A[use]) - 1, 0, G - 1)
        ib = np.clip(np.searchsorted(eb, B[use]) - 1, 0, G - 1)
        # the marginal terms already contain log(1 - pa - pb) for each column; the joint term carries log P(both interior)
        joint[use, k] = np.log(p_in) + np.log(D[ia, ib])
    prior = np.log(np.bincount(yt, minlength=K) / len(yt))
    return out, prior, joint, use


Lc = np.zeros((len(y), J, K))
JT = np.zeros((len(y), K))
USE = np.zeros(len(y), bool)
PRIOR = np.zeros((len(y), K))
for a, b in folds:
    o, pr, jt, u = components(a, b)
    Lc[b], JT[b], USE[b], PRIOR[b] = o, jt, u, pr
    print(f"  components of a fold done [{time.time() - t0:.0f}s]", flush=True)
softmax = lambda Z: (lambda E: E / E.sum(1, keepdims=True))(np.exp(Z - Z.max(1, keepdims=True)))
kde_plain = softmax(PRIOR + Lc.sum(1))
cached = np.load("outputs/preds/t2_kde_nb.npz")["oof"]
print(f"plain sum vs the pipeline's cached KDE OOF: max |diff| {np.abs(kde_plain - cached).max():.2e} [{time.time() - t0:.0f}s]")
pair_sum = Lc[:, len(cat) + PAIR[0], :] + Lc[:, len(cat) + PAIR[1], :]
kde_joint = softmax(PRIOR + Lc.sum(1) - np.where(USE[:, None], pair_sum, 0) + np.where(USE[:, None], JT, 0))
print(f"joint pair used on {USE.mean():.1%} of rows; log-loss plain {-np.mean(np.log(kde_plain[np.arange(len(y)), y])):.4f}, "
      f"joint {-np.mean(np.log(kde_joint[np.arange(len(y)), y])):.4f}")


def fit_weights(idx):
    Li, Pi, yi = Lc[idx], PRIOR[idx], y[idx]
    n = len(idx)

    def f(w):
        Z = Pi + np.einsum("njk,j->nk", Li, w)
        Z = Z - Z.max(1, keepdims=True)
        lse = np.log(np.exp(Z).sum(1))
        P = np.exp(Z - lse[:, None])
        nll = -(Z[np.arange(n), yi] - lse).sum() + 0.5 * n * 1e-3 * ((w - 1) ** 2).sum()
        G = (P[:, None, :] * Li).sum(2) - Li[np.arange(n), :, yi]
        return nll, G.sum(0) + n * 1e-3 * (w - 1)
    return minimize(f, np.ones(J), jac=True, method="L-BFGS-B", bounds=[(0, 2)] * J).x


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


S = np.column_stack([np.log(np.clip(np.load(f"outputs/preds/t2_{k}.npz")["oof"], 1e-6, 1)) for k in ("lgb", "xgb", "cat", "lr")])
WS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
splits = [("pipeline folds", folds)] + [(f"split seed {s}", list(StratifiedKFold(5, shuffle=True, random_state=s).split(S, y))) for s in (1, 2, 3)]
summary = {"weighted NB": [], "joint pair": []}
for label, fl in splits:
    Ps = np.zeros((len(y), K))
    for a, b in fl:
        Ps[b] = LogisticRegression(C=0.5, max_iter=3000).fit(S[a], y[a]).predict_proba(S[b])
    kde_w = np.zeros((len(y), K))
    wts = []
    for a, b in fl:
        w = fit_weights(a)
        wts.append(w)
        kde_w[b] = softmax(PRIOR[b] + np.einsum("njk,j->nk", Lc[b], w))
    labs = {}
    for name, Q in [("reference", kde_plain), ("weighted NB", kde_w), ("joint pair", kde_joint)]:
        lab = np.zeros(len(y), int)
        for a, b in fl:
            sc = {w_: opt_mult(y[a], ((1 - w_) * Ps + w_ * Q)[a]) for w_ in WS}
            wb_ = max(WS, key=lambda w_: sc[w_][1])
            lab[b] = np.argmax(((1 - wb_) * Ps + wb_ * Q)[b] * sc[wb_][0][None, :], 1)
        labs[name] = lab
    for name in ("weighted NB", "joint pair"):
        per = [mf1(y[b], labs[name][b]) - mf1(y[b], labs["reference"][b]) for _, b in fl]
        d = mf1(y, labs[name]) - mf1(y, labs["reference"])
        summary[name].append((d, sum(p > 0 for p in per), min(per)))
        print(f"{label}: {name:11s} {mf1(y, labs[name]):.4f} vs {mf1(y, labs['reference']):.4f}  delta {d:+.4f}  folds +{sum(p > 0 for p in per)}/5 "
              f"({' '.join(f'{p:+.4f}' for p in per)}) [{time.time() - t0:.0f}s]", flush=True)
    mw = np.mean(wts, 0)
    names = cat + cont
    order = np.argsort(mw)
    print(f"   mean feature weights: lowest " + ", ".join(f"{names[i]} {mw[i]:.2f}" for i in order[:5]) + "; highest "
          + ", ".join(f"{names[i]} {mw[i]:.2f}" for i in order[-5:]), flush=True)
for name, v in summary.items():
    ok = np.mean([d for d, _, _ in v]) > 0 and all(n >= 3 and m >= -0.0005 for _, n, m in v)
    print(f"\n{name}: pooled mean delta {np.mean([d for d, _, _ in v]):+.4f}; folds better {' '.join(str(n) + '/5' for _, n, _ in v)}; "
          f"worst fold {' '.join(f'{m:+.4f}' for _, _, m in v)} -> {'PASS' if ok else 'fail'}")
