"""Round 21, Task 2: generative classifiers built on the data-generating process found in round 19, alone and as a stack member.

    python cs_r21_t2_generative.py

Round 19: each class draws every feature independently from a (clipped) normal-like distribution, except Whale, which has one
continuous intensity factor; counts are integers >= 0 with a point mass at 0; bounded columns pile up at their clip bounds.
Round 18's naive Bayes used 24-bin per-class histograms (non-parametric, so it already saw the bound masses): 0.7549, and +0.0001
nested as a stack member. This script asks whether a smoother or dependence-aware generative model gets closer to the Bayes rule.

Models (all fitted on the training part of each pipeline fold; test = fit on all training rows; NaN is MCAR and simply skipped):
  hist24     round 18's histogram naive Bayes (reference)
  tobit      censored-normal naive Bayes: per class and column, MLE of (mu, sigma) with the clip-bound masses as censored tails
  kde        smooth marginals: Gaussian KDE (binned, Silverman bandwidth) for the interior + point masses at the clip bounds
  kde_cop_w  kde + a Gaussian copula for Whale only (within-Whale correlation of normal scores, i.e. the intensity factor)
  kde_cop    kde + a Gaussian copula for every class
Columns with <= 12 distinct values are categorical (Laplace-smoothed frequencies) in every model. Noise columns are dropped as in the
pipeline (task2.NOISE_COLS); nominal columns are kept out (round 18: no information).

Scored: OOF macro-F1 (argmax, nested multipliers); then the pipeline stack (LogReg on log-probabilities of LGB/XGB/Cat/LR, C = 0.5)
with each model added as a fifth member, multipliers chosen nested (4 folds -> 5th). Also: multipliers with an L2 pull towards 1.
Pass rule (fixed in advance): nested stack + multipliers gain with paired-bootstrap 95% CI above 0 and better in >= 4/5 folds.
Writes outputs/r21_t2/<model>.npz (oof, test) for a possible pipeline member.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtri
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
import common  # noqa: F401,E402
from src import task2  # noqa: E402
from src.metrics import apply_class_weights, optimize_class_weights  # noqa: E402
from src.modeling import make_folds  # noqa: E402

OUT = Path("experiments/claude/outputs/r21_t2")
OUT.mkdir(parents=True, exist_ok=True)
NAMES = ["Casual", "Grinder", "Social", "Whale"]
WHALE = NAMES.index("Whale")
train, test = task2.load()
y = train[task2.TARGET].values
folds = make_folds(y)
drop = ["player_id", task2.TARGET, *task2.NOISE_COLS, *task2.NOMINAL_COLS]
R = train.drop(columns=[c for c in drop if c in train.columns])
T = test.drop(columns=[c for c in drop if c in test.columns])
for c, m in task2.ORDINAL_MAPS.items():
    R[c], T[c] = R[c].map(m).astype(float), T[c].map(m).astype(float)
CAT = [c for c in R.columns if R[c].nunique() <= 12]
CONT = [c for c in R.columns if c not in CAT]
XR, XT = R[CONT].values.astype(float), T[CONT].values.astype(float)
# clip bounds: the column's training min / max when at least 0.2% of the values sit exactly on it (label-free)
LO = np.array([np.nanmin(XR[:, j]) if np.nanmean(XR[:, j] == np.nanmin(XR[:, j])) >= 0.002 else -np.inf for j in range(len(CONT))])
HI = np.array([np.nanmax(XR[:, j]) if np.nanmean(XR[:, j] == np.nanmax(XR[:, j])) >= 0.002 else np.inf for j in range(len(CONT))])
print(f"{len(y)} rows; {len(CONT)} continuous/count columns ({np.isfinite(LO).sum()} with a lower clip mass, {np.isfinite(HI).sum()} upper); "
      f"{len(CAT)} categorical: {CAT}")


# ---------------------------------------------------------------- marginals: each returns (logpdf(x), cdf(x)) fitted on v
def fit_tobit(v, lo, hi):
    a, b = v == lo, v == hi
    inner = v[~a & ~b]
    def nll(t):
        mu, s = t[0], np.exp(t[1])
        z = (inner - mu) / s
        out = -(-0.5 * z ** 2 - np.log(s)).sum()
        if a.any():
            out -= a.sum() * log_ndtr((lo - mu) / s)
        if b.any():
            out -= b.sum() * log_ndtr((mu - hi) / s)
        return out
    t = minimize(nll, [inner.mean(), np.log(inner.std() + 1e-9)], method="L-BFGS-B").x
    mu, s = t[0], np.exp(t[1])
    def logpdf(x):
        l = -0.5 * ((x - mu) / s) ** 2 - np.log(s) - 0.5 * np.log(2 * np.pi)
        l = np.where(x == lo, log_ndtr((lo - mu) / s), l)
        return np.where(x == hi, log_ndtr((mu - hi) / s), l)
    cdf = lambda x: np.exp(log_ndtr((x - mu) / s))
    return logpdf, cdf


def fit_kde(v, lo, hi, grid):
    a, b = v == lo, v == hi
    n = len(v)
    pa, pb = (a.sum() + 0.5) / (n + 1), (b.sum() + 0.5) / (n + 1)
    inner = v[~a & ~b]
    sd = min(inner.std(), np.subtract(*np.percentile(inner, [75, 25])) / 1.34) or inner.std()
    h = 0.9 * sd * len(inner) ** -0.2
    step = grid[1] - grid[0]
    H = np.histogram(inner, bins=np.append(grid - step / 2, grid[-1] + step / 2))[0].astype(float)
    k = np.arange(-int(4 * h / step) - 1, int(4 * h / step) + 2) * step
    dens = np.convolve(H, np.exp(-0.5 * (k / h) ** 2), mode="same")
    dens = np.maximum(dens, dens.max() * 1e-6)
    dens /= dens.sum() * step
    cum = np.cumsum(dens) * step
    def logpdf(x):
        l = np.log(1 - pa - pb) + np.log(np.interp(x, grid, dens))
        l = np.where(x == lo, np.log(pa), l)
        return np.where(x == hi, np.log(pb), l)
    def cdf(x):
        c = pa + (1 - pa - pb) * np.interp(x, grid, cum)
        c = np.where(x == lo, pa / 2, c)
        return np.where(x == hi, 1 - pb / 2, c)
    return logpdf, cdf


GRIDS = [np.linspace(np.nanmin(np.r_[XR[:, j], XT[:, j]]), np.nanmax(np.r_[XR[:, j], XT[:, j]]), 2048) for j in range(len(CONT))]


def copula_loglik(Z, Rm):
    """log Gaussian-copula density of normal scores Z (n, p; NaN = missing, marginalised out) under correlation Rm."""
    out = np.zeros(len(Z))
    obs = ~np.isnan(Z)
    keys = np.packbits(obs, axis=1)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    for g in np.unique(inv):
        rows = np.where(inv == g)[0]
        o = obs[rows[0]]
        if o.sum() < 2:
            continue
        S = Rm[np.ix_(o, o)]
        Si = np.linalg.inv(S) - np.eye(o.sum())
        z = Z[np.ix_(rows, np.where(o)[0])]
        out[rows] = -0.5 * np.linalg.slogdet(S)[1] - 0.5 * np.einsum("ij,jk,ik->i", z, Si, z)
    return out


def fit_predict(tr, Xs, cats, kind):
    """Class log-probabilities (unnormalised) for each matrix in Xs / cats, fitted on rows tr."""
    yt = y[tr]
    outs = [np.tile(np.log(np.bincount(yt, minlength=4) / len(yt)), (len(X), 1)) for X in Xs]
    # categorical columns
    for c in CAT:
        vals = np.sort(R[c].dropna().unique())
        for k in range(4):
            cnt = np.array([(R[c].values[tr][yt == k] == u).sum() for u in vals]) + 1.0
            lp = dict(zip(vals, np.log(cnt / cnt.sum())))
            for o, C_ in zip(outs, cats):
                o[:, k] += C_[c].map(lp).fillna(0).values
    # continuous / count columns
    for k in range(4):
        rows = tr[yt == k]
        Zs = [np.full(X.shape, np.nan) for X in Xs]
        for j in range(len(CONT)):
            v = XR[rows, j]
            v = v[~np.isnan(v)]
            lp, cdf = fit_tobit(v, LO[j], HI[j]) if kind == "tobit" else fit_kde(v, LO[j], HI[j], GRIDS[j])
            for o, X, Z in zip(outs, Xs, Zs):
                x = X[:, j]
                m = ~np.isnan(x)
                o[m, k] += lp(x[m])
                Z[m, j] = ndtri(np.clip(cdf(x[m]), 1e-6, 1 - 1e-6))
        if kind == "kde_cop" or (kind == "kde_cop_w" and k == WHALE):
            Ztr = np.full((len(rows), len(CONT)), np.nan)
            for j in range(len(CONT)):
                v = XR[rows, j]
                m = ~np.isnan(v)
                _, cdf = fit_kde(v[m], LO[j], HI[j], GRIDS[j])
                Ztr[m, j] = ndtri(np.clip(cdf(v[m]), 1e-6, 1 - 1e-6))
            Rm = pd.DataFrame(Ztr).corr().fillna(0).values
            Rm = 0.95 * Rm + 0.05 * np.eye(len(CONT))
            for o, Z in zip(outs, Zs):
                o[:, k] += copula_loglik(Z, Rm)
    return outs


def softmax(L):
    P = np.exp(L - L.max(1, keepdims=True))
    return P / P.sum(1, keepdims=True)


def hist24(tr, va_rows):
    """Round 18's histogram NB on the same columns (reference)."""
    L = np.tile(np.log(np.bincount(y[tr], minlength=4) / len(tr)), (len(va_rows), 1))
    for c in R.columns:
        vv = R[c].values.astype(float)
        if c in CAT:
            u = {v: i for i, v in enumerate(np.sort(R[c].dropna().unique()))}
            b = R[c].map(u).fillna(len(u)).astype(int).values
            nb = len(u) + 1
        else:
            edges = np.unique(np.nanquantile(vv[tr], np.linspace(0, 1, 25)[1:-1]))
            b = np.where(np.isnan(vv), len(edges) + 1, np.searchsorted(edges, vv))
            nb = len(edges) + 2
        H = np.zeros((4, nb))
        np.add.at(H, (y[tr], b[tr]), 1)
        L += np.log((H + 1) / (H.sum(1, keepdims=True) + nb))[:, b[va_rows]].T
    return softmax(L)


mf1 = lambda p: f1_score(y, p.argmax(1) if p.ndim == 2 else p, average="macro")


def nested_mult(P, lam=0.0):
    out = np.zeros(len(y), int)
    for tr, va in folds:
        if lam == 0:
            w = optimize_class_weights(y[tr], P[tr])[0]
        else:
            f = lambda u: -(f1_score(y[tr], apply_class_weights(P[tr], np.exp(u)), average="macro") - lam * np.sum(u ** 2))
            w = np.exp(minimize(f, np.zeros(4), method="Powell", options={"xtol": 1e-3, "ftol": 1e-6}).x)
        out[va] = apply_class_weights(P[va], w)
    return out


GEN = {}
for kind in ["hist24", "tobit", "kde", "kde_cop_w", "kde_cop"]:
    f = OUT / f"{kind}.npz"
    if f.exists():
        z = np.load(f)
        GEN[kind] = {"oof": z["oof"], "test": z["test"]}
        continue
    oof = np.zeros((len(y), 4))
    for tr, va in folds:
        if kind == "hist24":
            oof[va] = hist24(tr, va)
        else:
            oof[va] = softmax(fit_predict(tr, [XR[va]], [R.iloc[va]], kind)[0])
    tst = None if kind == "hist24" else softmax(fit_predict(np.arange(len(y)), [XT], [T], kind)[0])
    GEN[kind] = {"oof": oof, "test": tst if tst is not None else np.full((len(T), 4), np.nan)}
    np.savez(f, **GEN[kind])
    print(f"  {kind} fitted", flush=True)

print("\ngenerative models alone (OOF macro-F1):")
for k, g in GEN.items():
    ll = -np.mean(np.log(np.clip(g["oof"][np.arange(len(y)), y], 1e-12, 1)))
    print(f"  {k:10s} argmax {mf1(g['oof']):.4f} | nested multipliers {mf1(nested_mult(g['oof'])):.4f} | log-loss {ll:.4f}")

M = {k: np.load(f"outputs/preds/t2_{k}.npz") for k in ("lgb", "xgb", "cat", "lr")}


def stack(members):
    S = np.column_stack([np.log(np.clip(m["oof"], 1e-6, 1)) for m in members])
    out = np.zeros((len(y), 4))
    for tr, va in folds:
        out[va] = LogisticRegression(C=0.5, max_iter=3000).fit(S[tr], y[tr]).predict_proba(S[va])
    return out


base = stack(list(M.values()))
base_n = nested_mult(base)
fold_of = np.zeros(len(y), int)
for f, (_, va) in enumerate(folds):
    fold_of[va] = f
rng = np.random.default_rng(0)
boot = [rng.integers(0, len(y), len(y)) for _ in range(1000)]
ll_base = -np.mean(np.log(base[np.arange(len(y)), y]))
print(f"\npipeline stack: argmax {mf1(base):.4f} | nested multipliers {mf1(base_n):.4f} | log-loss {ll_base:.5f}")
for lam in [0.01, 0.03, 0.1]:
    print(f"  multipliers with L2 pull to 1, lambda {lam}: nested {mf1(nested_mult(base, lam)):.4f}")
print("stack + one generative member (nested multipliers), delta vs pipeline stack:")
for k, g in GEN.items():
    st = stack(list(M.values()) + [g])
    n = nested_mult(st)
    d = [f1_score(y[i], n[i], average="macro") - f1_score(y[i], base_n[i], average="macro") for i in boot]
    pf = [f1_score(y[fold_of == f], n[fold_of == f], average="macro") - f1_score(y[fold_of == f], base_n[fold_of == f], average="macro")
          for f in range(5)]
    ll = -np.mean(np.log(st[np.arange(len(y)), y]))
    print(f"  + {k:10s} argmax {mf1(st):.4f} nested {mf1(n):.4f} ({mf1(n) - mf1(base_n):+.4f} [{np.quantile(d, .025):+.4f}, "
          f"{np.quantile(d, .975):+.4f}], {sum(x > 0 for x in pf)}/5) | log-loss {ll:.5f} ({ll - ll_base:+.5f})")
