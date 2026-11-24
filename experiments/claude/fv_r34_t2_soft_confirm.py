"""Round 34, Task 2: the soft macro-F1 multiplier search on fresh stacker splits (training rows only, CPU).

    python fv_r34_t2_soft_confirm.py

fr_r34_t2_calib.py: of seven candidates only 1c(B) (class multipliers and blend weight chosen on a soft macro-F1, L-BFGS; soft
assignment softmax((log p + v) / 0.1)) had a positive mean on all four splits (+0.0005 +0.0006 +0.0007 +0.0003; row-bootstrap 95%
CI of the mean [+0.00005, +0.00105]), but it failed the fold rule on split seed 3 (2/5 folds, worst -0.0016). It was the best of 15
Task 2 candidates of rounds 33-34 on those splits, so its mean is selected. Here the same candidate, nothing changed, on six
stacker splits that no Task 2 experiment has used (seeds 4-9): an unselected estimate.
Reference = the v20 blend nested (Powell on the raw macro-F1), exactly as in fr_r34. Also printed: the weight / multipliers each
search picks (in fr_r34 the soft search picked the weight 0.6 in 15 of 20 folds).
Acceptance (user, 2026-11-16): pooled delta > 0, >= 3/5 folds better on each split, no fold below -0.0005.
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

psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
t0 = time.time()
log = lambda m: print(f"{m} [{time.time() - t0:.0f}s]", flush=True)
y = pd.read_csv(DATA_DIRS[2] / "train.csv")[task2.TARGET].values
K = 4
softmax = lambda Z: np.exp(Z - logsumexp(Z, axis=1, keepdims=True))
lg = lambda P: np.log(np.clip(P, 1e-300, 1))
KDE = np.load("outputs/preds/t2_kde_nb.npz")["oof"]
S = np.column_stack([np.log(np.clip(np.load(f"outputs/preds/t2_{k}.npz")["oof"], 1e-6, 1)) for k in ("lgb", "xgb", "cat", "lr")])
WS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def mf1(yy, pred, w=None):
    cm = np.bincount(yy * K + pred, weights=w, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(float)
    den = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.where(den > 0, 2 * tp / np.maximum(den, 1e-9), 0.0)))


def opt_mult(yy, P):
    L = lg(P)
    loss = lambda v: -mf1(yy, np.argmax(L + np.r_[0.0, v][None, :], 1))
    res = minimize(loss, np.zeros(K - 1), method="Powell", options={"xtol": 1e-3, "ftol": 1e-6, "maxiter": 3000})
    v = res.x if res.fun < loss(np.zeros(K - 1)) else np.zeros(K - 1)
    return np.r_[0.0, v], -min(res.fun, loss(np.zeros(K - 1)))


def opt_mult_soft(yy, P, temp=0.1):
    L = lg(P)
    Y = np.eye(K)[yy]
    ny = Y.sum(0)

    def loss(v3):
        q = softmax((L + np.r_[0.0, v3][None, :]) / temp)
        return -float(np.mean(2 * (Y * q).sum(0) / (q.sum(0) + ny)))
    res = minimize(loss, np.zeros(K - 1), method="L-BFGS-B")
    return np.r_[0.0, res.x], -res.fun


SEEDS = [int(s) for s in os.environ.get("R34_SEEDS", "4,5,6,7,8,9").split(",")]
rows, labs_all = [], []
for seed in SEEDS:
    fl = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(S, y))
    Ps = np.zeros((len(y), K))
    for a, b in fl:
        Ps[b] = LogisticRegression(C=0.5, max_iter=3000).fit(S[a], y[a]).predict_proba(S[b])
    ref, soft = np.zeros(len(y), int), np.zeros(len(y), int)
    picks = []
    for a, b in fl:
        r = {w_: opt_mult(y[a], ((1 - w_) * Ps + w_ * KDE)[a]) for w_ in WS}
        s = {w_: opt_mult_soft(y[a], ((1 - w_) * Ps + w_ * KDE)[a]) for w_ in WS}
        wr, ws = max(WS, key=lambda w_: r[w_][1]), max(WS, key=lambda w_: s[w_][1])
        ref[b] = np.argmax(lg(((1 - wr) * Ps + wr * KDE)[b]) + r[wr][0][None, :], 1)
        soft[b] = np.argmax(lg(((1 - ws) * Ps + ws * KDE)[b]) + s[ws][0][None, :], 1)
        picks.append(f"{wr}->{ws} ({' '.join(f'{x:+.2f}' for x in r[wr][0][1:])} -> {' '.join(f'{x:+.2f}' for x in s[ws][0][1:])})")
    per = [mf1(y[b], soft[b]) - mf1(y[b], ref[b]) for _, b in fl]
    d = mf1(y, soft) - mf1(y, ref)
    rows.append((d, sum(p > 0 for p in per), min(per)))
    labs_all.append((ref, soft))
    log(f"split seed {seed}: soft {mf1(y, soft):.4f} vs reference {mf1(y, ref):.4f}  delta {d:+.4f}  folds +{sum(p > 0 for p in per)}/5 "
        f"({' '.join(f'{p:+.4f}' for p in per)})\n   weight and log-multipliers, reference -> soft: " + "; ".join(picks))
ok = np.mean([r[0] for r in rows]) > 0 and all(r[1] >= 3 and r[2] >= -0.0005 for r in rows)
rng = np.random.default_rng(0)
W = [rng.multinomial(len(y), np.full(len(y), 1 / len(y))).astype(float) for _ in range(400)]
bd = np.array([np.mean([mf1(y, s_, w) - mf1(y, r_, w) for r_, s_ in labs_all]) for w in W])
print(f"\nsoft macro-F1 search on {len(SEEDS)} fresh splits: mean delta {np.mean([r[0] for r in rows]):+.5f}; splits with delta > 0: "
      f"{sum(r[0] > 0 for r in rows)}/{len(rows)}; folds better {' '.join(str(r[1]) + '/5' for r in rows)}; worst fold "
      f"{' '.join(f'{r[2]:+.4f}' for r in rows)} -> {'PASS' if ok else 'fail'}\n"
      f"row-bootstrap of the mean delta: 95% CI [{np.quantile(bd, .025):+.5f}, {np.quantile(bd, .975):+.5f}], P(delta > 0) {np.mean(bd > 0):.3f}")
log("done")
