"""Round 35, Task 2: a pair-focused naive-Bayes tie-breaker on top of the soft macro-F1 rule (training rows only, CPU).

    python fx_r35_t2_pair_tiebreak.py

Hypothesis (user brief of 2026-11-16 ~22:25): rows whose two best classes are almost tied after the v26 rule (blend of stacker and
KDE naive Bayes, weight and class multipliers by the soft macro-F1 search) are decided by 34 columns, most of which carry nothing
about THAT pair of classes; a likelihood ratio on the 8-10 columns that separate the pair best is a cleaner tie-breaker.
Base = the v26 rule nested (weight and multipliers from the four training folds). Tie rows: p1 - p2 < 0.06 for the two largest of
p ~ w_c * P_c (multiplied and renormalised). For a tie between classes (a, b):
    T = (v_a - v_b) + log(prior_a / prior_b) + sum_{j in J_ab} [log p(x_j | a) - log p(x_j | b)],   decide a if T > 0, else b,
with v the log-multipliers, the per-feature densities of the pipeline's naive Bayes (out of fold on the pipeline folds;
outputs/r34/t2_kde_feature_loglik.npz), and J_ab the n columns with the largest symmetrised divergence
mean_{y=a}(LLR_j) - mean_{y=b}(LLR_j), measured on the training folds' rows of the two classes.
Variants: n = 8; n = 10; all 34 columns (control: the tie decided by the whole naive Bayes); n = 10 applied only to the class
pairs where it raises the accuracy of the training folds' tie rows (a nested gate per pair).
Splits: the pipeline folds + stacker split seeds 1, 2, 3 (the brief), then seeds 4-9 as an unselected check.
Rules: the fold rule (pooled > 0, >= 3/5 folds, no fold below -0.0005) and the user's Task 2 rule of 2026-11-16 (positive mean and
row-bootstrap 95% CI above 0 on the fresh splits).
"""
import os
import time
import warnings

NT = int(os.environ.get("R33_THREADS", 2))
os.environ.setdefault("OMP_NUM_THREADS", str(NT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import logsumexp  # noqa: E402
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
y = pd.read_csv(DATA_DIRS[2] / "train.csv")[task2.TARGET].values
K = 4
softmax = lambda Z: np.exp(Z - logsumexp(Z, axis=1, keepdims=True))
lg = lambda P: np.log(np.clip(P, 1e-300, 1))
d = np.load("experiments/claude/outputs/r34/t2_kde_feature_loglik.npz")
LL, LPRI = d["ll"].astype(np.float64), d["lprior"]
KDE = np.load("outputs/preds/t2_kde_nb.npz")["oof"]
S = np.column_stack([np.log(np.clip(np.load(f"outputs/preds/t2_{k}.npz")["oof"], 1e-6, 1)) for k in ("lgb", "xgb", "cat", "lr")])
WS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
GAP = 0.06
NF = LL.shape[1]


def mf1(yy, pred, w=None):
    cm = np.bincount(yy * K + pred, weights=w, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(float)
    den = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.where(den > 0, 2 * tp / np.maximum(den, 1e-9), 0.0)))


def opt_mult_soft(yy, P, temp=0.1):
    L = lg(P)
    Y = np.eye(K)[yy]
    ny = Y.sum(0)

    def loss(v3):
        q = softmax((L + np.r_[0.0, v3][None, :]) / temp)
        return -float(np.mean(2 * (Y * q).sum(0) / (q.sum(0) + ny)))
    res = minimize(loss, np.zeros(K - 1), method="L-BFGS-B")
    return np.r_[0.0, res.x], -res.fun


def ties(P, v):
    """(rows almost tied, best class, second class, base label) under the rule argmax(log P + v)."""
    q = softmax(lg(P) + v[None, :])
    o = np.argsort(-q, axis=1)
    c1, c2 = o[:, 0], o[:, 1]
    r = np.arange(len(q))
    return (q[r, c1] - q[r, c2]) < GAP, c1, c2


def tiebreak(rows, c1, c2, v, cols_of):
    """Label of each tie row: the pair's likelihood ratio on its selected columns."""
    out = c1.copy()
    for a_ in range(K):
        for b_ in range(K):
            m = (c1 == a_) & (c2 == b_)
            if a_ == b_ or not m.any():
                continue
            J = cols_of[(min(a_, b_), max(a_, b_))]
            if J is None:
                continue
            r = rows[m]
            T = (v[a_] - v[b_]) + (LPRI[r, a_] - LPRI[r, b_]) + (LL[r][:, J, a_] - LL[r][:, J, b_]).sum(1)
            out[m] = np.where(T > 0, a_, b_)
    return out


NAMES = ["n = 8 columns", "n = 10 columns", "all 34 columns (control)", "n = 10, pairs gated on the training folds"]
splits = [("pipeline folds", make_folds(y))] + [(f"split seed {s}", list(StratifiedKFold(5, shuffle=True, random_state=s).split(S, y)))
                                                for s in (1, 2, 3, 4, 5, 6, 7, 8, 9)]
rows_sum = {n: [] for n in NAMES}
labs_all = []
for si, (label, fl) in enumerate(splits):
    Ps = np.zeros((len(y), K))
    for a, b in fl:
        Ps[b] = LogisticRegression(C=0.5, max_iter=3000).fit(S[a], y[a]).predict_proba(S[b])
    labs = {n: np.zeros(len(y), int) for n in ["base"] + NAMES}
    stats = []
    for a, b in fl:
        sc = {w_: opt_mult_soft(y[a], ((1 - w_) * Ps + w_ * KDE)[a]) for w_ in WS}
        wb = max(WS, key=lambda w_: sc[w_][1])
        v = sc[wb][0]
        P = (1 - wb) * Ps + wb * KDE
        base_b = np.argmax(lg(P[b]) + v[None, :], 1)
        tb, c1b, c2b = ties(P[b], v)
        ta, c1a, c2a = ties(P[a], v)
        # columns per class pair from the training folds
        div = {}
        LLa, ya = LL[a], y[a]
        for a_ in range(K):
            for b_ in range(a_ + 1, K):
                dab = LLa[:, :, a_] - LLa[:, :, b_]
                llr_a, llr_b = dab[ya == a_].mean(0), dab[ya == b_].mean(0)
                div[(a_, b_)] = np.argsort(-(llr_a - llr_b))
        sel = {n: {k_: o[:n] for k_, o in div.items()} for n in (8, 10, NF)}
        # nested gate: keep a pair only if the n = 10 tie-break is more accurate than the base on the training folds' tie rows
        ra = a[ta]
        tr_new = tiebreak(ra, c1a[ta], c2a[ta], v, sel[10])
        gate = {}
        for k_ in div:
            m = ((c1a[ta] == k_[0]) & (c2a[ta] == k_[1])) | ((c1a[ta] == k_[1]) & (c2a[ta] == k_[0]))
            gate[k_] = sel[10][k_] if m.sum() >= 50 and np.mean(tr_new[m] == y[ra][m]) > np.mean(c1a[ta][m] == y[ra][m]) else None
        rb = b[tb]
        labs["base"][b] = base_b
        for n, cols in zip(NAMES, (sel[8], sel[10], sel[NF], gate)):
            lab = base_b.copy()
            lab[tb] = tiebreak(rb, c1b[tb], c2b[tb], v, cols)
            labs[n][b] = lab
        stats.append((tb.mean(), np.mean(base_b[tb] == y[b][tb]), np.mean(labs[NAMES[1]][b][tb] == y[b][tb]),
                      np.mean(labs[NAMES[2]][b][tb] == y[b][tb]), sum(g is not None for g in gate.values())))
    st = np.mean(stats, 0)
    log(f"{label}: base (v26 rule) {mf1(y, labs['base']):.4f}; tie rows {st[0]:.1%}, accuracy on them: base {st[1]:.3f}, n = 10 {st[2]:.3f}, "
        f"all columns {st[3]:.3f}; pairs kept by the gate {st[4]:.1f} of 6")
    for n in NAMES:
        per = [mf1(y[b], labs[n][b]) - mf1(y[b], labs["base"][b]) for _, b in fl]
        dl = mf1(y, labs[n]) - mf1(y, labs["base"])
        rows_sum[n].append((dl, sum(p > 0 for p in per), min(per)))
        print(f"   {n:42s} delta {dl:+.4f}  folds +{sum(p > 0 for p in per)}/5 ({' '.join(f'{p:+.4f}' for p in per)})  labels changed "
              f"{np.mean(labs[n] != labs['base']):.2%}", flush=True)
    labs_all.append(labs)
rng = np.random.default_rng(0)
W = [rng.multinomial(len(y), np.full(len(y), 1 / len(y))).astype(float) for _ in range(300)]
print()
for n in NAMES:
    for part, sl in (("brief's four splits", slice(0, 4)), ("fresh splits 4-9", slice(4, 10))):
        v_ = rows_sum[n][sl]
        ok = np.mean([r[0] for r in v_]) > 0 and all(r[1] >= 3 and r[2] >= -0.0005 for r in v_)
        bd = np.array([np.mean([mf1(y, L_[n], w) - mf1(y, L_["base"], w) for L_ in labs_all[sl]]) for w in W])
        print(f"{n} | {part}: mean delta {np.mean([r[0] for r in v_]):+.5f}; folds better {' '.join(str(r[1]) + '/5' for r in v_)}; worst fold "
              f"{' '.join(f'{r[2]:+.4f}' for r in v_)} -> fold rule {'PASS' if ok else 'fail'}; row-bootstrap 95% CI "
              f"[{np.quantile(bd, .025):+.5f}, {np.quantile(bd, .975):+.5f}] -> Task 2 rule {'PASS' if np.quantile(bd, .025) > 0 else 'fail'}")
log("done")
