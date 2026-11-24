"""Round 33, Task 2: spline GAM member, Dirichlet calibration before the blend, Whale latent-factor term (training rows only, CPU).

    python fm_r33_t2_gam_dirichlet.py

Reference = the v20 blend, nested as in ev_r30_t2_kde_variants.py: multinomial LogReg stacker on the log-probabilities of
LightGBM / XGBoost / CatBoost / LogReg (out of fold on the pipeline folds), blended with the KDE naive Bayes; the KDE weight
w in {0.3..0.9} and the class multipliers are chosen on the 4 training folds of each split.
  item 1  spline GAM (multinomial logistic regression on a cubic B-spline basis of every continuous feature, 10 quantile knots
          fitted on the training fold, linear extrapolation; one-hot levels for the low-cardinality features; missing-value and
          clip-bound indicators): smooth shape functions, additive in the log-odds like the naive Bayes but fitted discriminatively.
          "+ pairs": tensor-product splines (5 knots, quadratic) for the 6 continuous pairs with the largest mean within-class
          |Spearman| on the training fold. Out of fold on the pipeline folds. Used (a) as a fifth stack member, (b) as a third
          term of the blend, weight g in {0, .1, .2, .3} chosen with w on the training folds.
  item 2  Dirichlet calibration (Kull et al., NeurIPS 2019: multinomial logistic regression on the log-probabilities, fitted on
          the training folds) of the KDE naive Bayes alone, and of both models followed by a log-linear (geometric) blend.
  item 3  Whale latent factor: first principal component of the standardised (vip_tier, avg_monthly_spending, spending_frequency,
          rare_items_count), fitted on the training fold; its class-conditional kernel density is added to the naive Bayes
          log-likelihood with weight lambda = 0.5 or 1 (the four columns' own terms stay).
Splits: pipeline folds + stacker split seeds 1, 2, 3 (members stay out of fold on the pipeline folds).
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
from scipy.stats import spearmanr  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import SplineTransformer  # noqa: E402

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
lg = lambda P: np.log(np.clip(P, 1e-9, 1))
WHALE = int(np.argmax([np.nanmean(tr.loc[y == k, "avg_monthly_spending"]) for k in range(K)]))


# ------------------------------------------------------------------ item 1: spline GAM, out of fold on the pipeline folds
def gam_design(a, rows, pairs):
    """Design matrix for `rows`, every transform fitted on the training rows a."""
    cols = []
    for j in range(len(cat)):
        v = CR[rows, j]
        for u in np.unique(CR[a, j][~np.isnan(CR[a, j])])[1:]:
            cols.append((v == u).astype(float))
        cols.append(np.isnan(v).astype(float))
    sp = {}
    for j in range(len(cont)):
        col = XR[a, j]
        obs = col[~np.isnan(col)]
        lo, hi = obs.min(), obs.max()
        x = XR[rows, j]
        nanm = np.isnan(x)
        for bound in (lo, hi):
            if np.mean(obs == bound) >= 0.002:
                cols.append((x == bound).astype(float))
        st = SplineTransformer(n_knots=10, degree=3, knots="quantile", extrapolation="linear", include_bias=False).fit(obs[:, None])
        B = st.transform(np.where(nanm, np.median(obs), x)[:, None])
        B[nanm] = 0.0
        cols.append(B)
        cols.append(nanm.astype(float))
        sp[j] = (np.median(obs), SplineTransformer(n_knots=5, degree=2, knots="quantile", extrapolation="constant", include_bias=False).fit(obs[:, None]))
    for ja, jb in pairs:
        Ba = sp[ja][1].transform(np.where(np.isnan(XR[rows, ja]), sp[ja][0], XR[rows, ja])[:, None])
        Bb = sp[jb][1].transform(np.where(np.isnan(XR[rows, jb]), sp[jb][0], XR[rows, jb])[:, None])
        T = (Ba[:, :, None] * Bb[:, None, :]).reshape(len(rows), -1)
        T[np.isnan(XR[rows, ja]) | np.isnan(XR[rows, jb])] = 0.0
        cols.append(T)
    return np.column_stack(cols)


def top_pairs(a, n=6):
    """Continuous pairs with the largest mean within-class |Spearman| on the training rows."""
    acc = np.zeros((len(cont), len(cont)))
    for k in range(K):
        Xk = pd.DataFrame(XR[a][y[a] == k])
        acc += np.abs(Xk.corr(method="spearman").values) / K
    iu = np.triu_indices(len(cont), 1)
    order = np.argsort(-acc[iu])[:n]
    return [(int(iu[0][i]), int(iu[1][i])) for i in order], acc[iu][order]


gam_file = OUT / "t2_gam_oof.npz"
if gam_file.exists():
    d = np.load(gam_file)
    GAM, GAMP = d["gam"], d["gam_pairs"]
else:
    GAM, GAMP = np.zeros((len(y), K)), np.zeros((len(y), K))
    for i, (a, b) in enumerate(folds):
        pairs, vals = top_pairs(a)
        for name, store, pp in (("main effects", GAM, []), ("+ pairs", GAMP, pairs)):
            A, B = gam_design(a, a, pp), gam_design(a, b, pp)
            sd = A.std(0) + 1e-9
            m = LogisticRegression(C=1.0, max_iter=400).fit(A / sd, y[a])
            store[b] = m.predict_proba(B / sd)
            log(f"GAM {name} fold {i}: {A.shape[1]} columns, log-loss {-np.mean(lg(store[b])[np.arange(len(b)), y[b]]):.4f}"
                + (f"; pairs {[(cont[p], cont[q]) for p, q in pairs]} |rho| {np.round(vals, 3)}" if pp else ""))
    np.savez(gam_file, gam=GAM, gam_pairs=GAMP)


# ------------------------------------------------------------------ item 3: Whale latent factor, out of fold on the pipeline folds
WCOLS = ["vip_tier", "avg_monthly_spending", "spending_frequency", "rare_items_count"]
W = R[WCOLS].values.astype(float)


def kde_logpdf(v, x, n_grid=1024):
    lo, hi = np.min(v) - 1e-9, np.max(v) + 1e-9
    grid = np.linspace(lo, hi, n_grid)
    step = grid[1] - grid[0]
    sd = min(v.std(), np.subtract(*np.percentile(v, [75, 25])) / 1.34) or v.std()
    h = 0.9 * sd * len(v) ** -0.2
    H = np.histogram(v, bins=np.append(grid - step / 2, grid[-1] + step / 2))[0].astype(float)
    kk = np.arange(-int(4 * h / step) - 1, int(4 * h / step) + 2) * step
    dens = np.convolve(H, np.exp(-0.5 * (kk / h) ** 2), mode="same")
    dens = np.maximum(dens, dens.max() * 1e-6)
    dens /= dens.sum() * step
    return np.log(np.interp(np.clip(x, lo, hi), grid, dens))


FAC = np.zeros((len(y), K))
for a, b in folds:
    mu, sd = np.nanmean(W[a], 0), np.nanstd(W[a], 0)
    Za, Zb = np.nan_to_num((W[a] - mu) / sd), np.nan_to_num((W[b] - mu) / sd)
    _, _, Vt = np.linalg.svd(Za - Za.mean(0), full_matrices=False)
    v = Vt[0] * np.sign((Za @ Vt[0])[y[a] == WHALE].mean() - (Za @ Vt[0]).mean())
    fa, fb = Za @ v, Zb @ v
    for k in range(K):
        FAC[b, k] = kde_logpdf(fa[y[a] == k], fb)
log(f"Whale factor (class {WHALE}) densities done")

# ------------------------------------------------------------------ nested blend
KDE = np.load("outputs/preds/t2_kde_nb.npz")["oof"]
S = np.column_stack([lg(np.clip(np.load(f"outputs/preds/t2_{k}.npz")["oof"], 1e-6, 1)) for k in ("lgb", "xgb", "cat", "lr")])
WS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
GS = (0.0, 0.1, 0.2, 0.3)


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


def pick(a, b, options):
    """options: list of full-length probability matrices; the one with the best multiplier-tuned macro-F1 on rows a labels rows b."""
    sc = [opt_mult(y[a], P[a]) for P in options]
    j = int(np.argmax([s[1] for s in sc]))
    return np.argmax(options[j][b] * sc[j][0][None, :], 1), j


def stack(fl, Sx):
    P = np.zeros((len(y), K))
    for a, b in fl:
        P[b] = LogisticRegression(C=0.5, max_iter=3000).fit(Sx[a], y[a]).predict_proba(Sx[b])
    return P




splits = [("pipeline folds", folds)] + [(f"split seed {s}", list(StratifiedKFold(5, shuffle=True, random_state=s).split(S, y))) for s in (1, 2, 3)]
CANDS = ["1a GAM as a 5th stack member", "1a GAM + pairs as a 5th stack member", "1b GAM as a third blend term", "1b GAM + pairs as a third blend term",
         "2a KDE Dirichlet-calibrated", "2b both calibrated, log-linear blend", "3 Whale factor, lambda 0.5", "3 Whale factor, lambda 1"]
summary = {n: [] for n in CANDS}
for label, fl in splits:
    Ps = stack(fl, S)
    Ps_g, Ps_gp = stack(fl, np.column_stack([S, lg(GAM)])), stack(fl, np.column_stack([S, lg(GAMP)]))
    labs = {n: np.zeros(len(y), int) for n in ["reference"] + CANDS}
    chosen_g = {"1b GAM as a third blend term": [], "1b GAM + pairs as a third blend term": []}
    for a, b in fl:
        kde_cal = np.zeros_like(KDE)
        ps_cal = np.zeros_like(Ps)
        for src, dst in ((KDE, kde_cal), (Ps, ps_cal)):
            m = LogisticRegression(C=1.0, max_iter=2000).fit(lg(src[a]), y[a])
            dst[a], dst[b] = m.predict_proba(lg(src[a])), m.predict_proba(lg(src[b]))
        labs["reference"][b], _ = pick(a, b, [(1 - w) * Ps + w * KDE for w in WS])
        labs[CANDS[0]][b], _ = pick(a, b, [(1 - w) * Ps_g + w * KDE for w in WS])
        labs[CANDS[1]][b], _ = pick(a, b, [(1 - w) * Ps_gp + w * KDE for w in WS])
        for n, G in ((CANDS[2], GAM), (CANDS[3], GAMP)):
            combos = [(w, g) for w in (0.4, 0.5, 0.6, 0.7) for g in GS]
            labs[n][b], j = pick(a, b, [(1 - w - g) * Ps + w * KDE + g * G for w, g in combos])
            chosen_g[n].append(combos[j][1])
        labs[CANDS[4]][b], _ = pick(a, b, [(1 - w) * Ps + w * kde_cal for w in WS])
        labs[CANDS[5]][b], _ = pick(a, b, [softmax((1 - w) * lg(ps_cal) + w * lg(kde_cal)) for w in WS])
        for n, lam in ((CANDS[6], 0.5), (CANDS[7], 1.0)):
            Q = softmax(lg(KDE) + lam * FAC)
            labs[n][b], _ = pick(a, b, [(1 - w) * Ps + w * Q for w in WS])
    ref = labs["reference"]
    for n in CANDS:
        per = [mf1(y[b], labs[n][b]) - mf1(y[b], ref[b]) for _, b in fl]
        d = mf1(y, labs[n]) - mf1(y, ref)
        summary[n].append((d, sum(p > 0 for p in per), min(per)))
        print(f"{label}: {n:40s} {mf1(y, labs[n]):.4f} vs {mf1(y, ref):.4f}  delta {d:+.4f}  folds +{sum(p > 0 for p in per)}/5 "
              f"({' '.join(f'{p:+.4f}' for p in per)})" + (f"  g chosen {chosen_g[n]}" if n in chosen_g else ""), flush=True)
    log(f"{label} done")
alone = lambda P: mf1(y, np.argmax(P, 1))
print(f"\nalone (argmax, no multipliers): GAM {alone(GAM):.4f}, GAM + pairs {alone(GAMP):.4f}, KDE NB {alone(KDE):.4f}; log-loss GAM "
      f"{-np.mean(lg(GAM)[np.arange(len(y)), y]):.4f}, GAM + pairs {-np.mean(lg(GAMP)[np.arange(len(y)), y]):.4f}, KDE NB "
      f"{-np.mean(lg(KDE)[np.arange(len(y)), y]):.4f}")
for n, v in summary.items():
    ok = np.mean([d for d, _, _ in v]) > 0 and all(k >= 3 and m >= -0.0005 for _, k, m in v)
    print(f"{n}: pooled mean delta {np.mean([d for d, _, _ in v]):+.4f}; folds better {' '.join(str(k) + '/5' for _, k, _ in v)}; "
          f"worst fold {' '.join(f'{m:+.4f}' for _, _, m in v)} -> {'PASS' if ok else 'fail'}")
log("done")
