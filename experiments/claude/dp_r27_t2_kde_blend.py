"""Round 27, Task 2: the pipeline stack blended with the KDE naive Bayes of round 21, re-tested under the round 25 rule.

    python dp_r27_t2_kde_blend.py

Round 21: KDE naive Bayes (smooth per-class marginals + point masses at the clip bounds; features independent given the class, as
round 19 / 23 found) scores 0.7572 alone; as a fifth LogReg-stack member -0.0002, but 0.5 stack + 0.5 KDE NB gave +0.0011 (5/5)
and the rule-strict version +0.0008 [-0.0001, +0.0017], 4/5. It was dropped under the CI gate. Here, strictly:
  * the density grid of every column spans the values of the rows the model is fitted on (the fold's training rows; all training
    rows for the test prediction). No test value enters the fit. Values outside the grid get the edge density.
  * the KDE NB is a level-1 member with out-of-fold predictions on the pipeline folds (like the four pipeline members).
  * level 2 is repeated on the pipeline folds and three other splits: LogReg stacker (C = 0.5) on the four members, then
    P = w * stack + (1 - w) * KDE NB, then class multipliers. Multipliers nested (4 folds -> 5th) everywhere. w = 0.5 fixed, and w
    chosen nested from {0.3, 0.4, ..., 1.0} by the macro-F1 with multipliers on the 4 training folds. Also the geometric blend
    (mean of log-probabilities, w = 0.5) and the KDE NB as a fifth stack member, for reference.
Acceptance (round 25 / 27 brief): mean nested gain > 0 and better in >= 4/5 folds.
Writes outputs/r27/t2_kde_strict.npz (oof, test).
"""
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
import common  # noqa: F401,E402
from src import task2  # noqa: E402
from src.metrics import apply_class_weights, optimize_class_weights  # noqa: E402
from src.modeling import make_folds  # noqa: E402

OUT = Path("experiments/claude/outputs/r27")
OUT.mkdir(parents=True, exist_ok=True)
train, test = task2.load()
y = train[task2.TARGET].values
folds = make_folds(y)
drop = ["player_id", "id", task2.TARGET, *task2.NOISE_COLS, *task2.NOMINAL_COLS]
R = train.drop(columns=[c for c in drop if c in train.columns])
T = test.drop(columns=[c for c in drop if c in test.columns])
for c, m in task2.ORDINAL_MAPS.items():
    R[c], T[c] = R[c].map(m).astype(float), T[c].map(m).astype(float)
CAT = [c for c in R.columns if R[c].nunique() <= 12]
CONT = [c for c in R.columns if c not in CAT]
XR, XT = R[CONT].values.astype(float), T[CONT].values.astype(float)
CR, CT = R[CAT].values.astype(float), T[CAT].values.astype(float)
NG = 2048


def kde_nb(tr, Xs, Cs):
    """Class probabilities for each (continuous, categorical) matrix pair, fitted on training rows tr only."""
    yt = y[tr]
    outs = [np.tile(np.log(np.bincount(yt, minlength=4) / len(yt)), (len(X), 1)) for X in Xs]
    for j in range(len(CAT)):                                              # categorical: Laplace-smoothed frequencies
        vals = np.unique(CR[tr, j][~np.isnan(CR[tr, j])])
        for k in range(4):
            v = CR[tr, j][yt == k]
            cnt = np.array([(v == u).sum() for u in vals]) + 1.0
            lp = np.log(cnt / cnt.sum())
            for o, C_ in zip(outs, Cs):
                idx = np.minimum(np.searchsorted(vals, C_[:, j]), len(vals) - 1)
                ok = vals[idx] == C_[:, j]                                 # NaN and unseen values contribute nothing
                o[ok, k] += lp[idx[ok]]
    for j in range(len(CONT)):
        col = XR[tr, j]
        obs = col[~np.isnan(col)]
        lo, hi = obs.min(), obs.max()
        lo_m = lo if np.mean(obs == lo) >= 0.002 else -np.inf              # clip bounds: a point mass of >= 0.2% on the min / max
        hi_m = hi if np.mean(obs == hi) >= 0.002 else np.inf
        grid = np.linspace(lo, hi, NG)
        step = grid[1] - grid[0]
        for k in range(4):
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
            for o, X in zip(outs, Xs):
                x = X[:, j]
                m = ~np.isnan(x)
                l = np.log(1 - pa - pb) + np.log(np.interp(x[m], grid, dens))
                l = np.where(x[m] == lo_m, np.log(pa), l)
                o[m, k] += np.where(x[m] == hi_m, np.log(pb), l)
    res = []
    for o in outs:
        P = np.exp(o - o.max(1, keepdims=True))
        res.append(P / P.sum(1, keepdims=True))
    return res


f = OUT / "t2_kde_strict.npz"
if f.exists():
    z = np.load(f)
    kde_oof, kde_test = z["oof"], z["test"]
else:
    kde_oof = np.zeros((len(y), 4))
    for tr, va in folds:
        kde_oof[va] = kde_nb(tr, [XR[va]], [CR[va]])[0]
    kde_test = kde_nb(np.arange(len(y)), [XT], [CT])[0]
    np.savez(f, oof=kde_oof, test=kde_test)
mf1 = lambda yy, p: f1_score(yy, p, average="macro")
r21 = Path("experiments/claude/outputs/r21_t2/kde.npz")
print(f"{len(y)} rows; {len(CONT)} continuous / count columns, {len(CAT)} categorical")
msg = f"KDE naive Bayes (strict grid) alone: argmax {mf1(y, kde_oof.argmax(1)):.4f}"
if r21.exists():
    z21 = np.load(r21)
    msg += (f" | agreement with round 21's version (train + test grid range): OOF {np.mean(z21['oof'].argmax(1) == kde_oof.argmax(1)):.4%}, "
            f"test {np.mean(z21['test'].argmax(1) == kde_test.argmax(1)):.4%}")
print(msg, flush=True)
M = [np.load(f"outputs/preds/t2_{k}.npz")["oof"] for k in ("lgb", "xgb", "cat", "lr")]
LOGS = [np.log(np.clip(m, 1e-6, 1)) for m in M]
WS = np.round(np.arange(0.3, 1.01, 0.1), 2)


def stack(members, fl):
    S = np.column_stack(members)
    out = np.zeros((len(y), 4))
    for tr, va in fl:
        out[va] = LogisticRegression(C=0.5, max_iter=3000).fit(S[tr], y[tr]).predict_proba(S[va])
    return out


def nested_mult(P, fl):
    out = np.zeros(len(y), int)
    for tr, va in fl:
        out[va] = apply_class_weights(P[va], optimize_class_weights(y[tr], P[tr])[0])
    return out


def nested_w(Ps, fl):
    out, chosen = np.zeros(len(y), int), []
    for tr, va in fl:
        best = (-1, None, None)
        for w in WS:
            P = w * Ps + (1 - w) * kde_oof
            mult, s = optimize_class_weights(y[tr], P[tr])
            if s > best[0] + 1e-9:
                best = (s, w, mult)
        _, w, mult = best
        chosen.append(float(w))
        out[va] = apply_class_weights((w * Ps + (1 - w) * kde_oof)[va], mult)
    return out, chosen


rng = np.random.default_rng(0)
boot = [rng.integers(0, len(y), len(y)) for _ in range(1000)]
splits = [("pipeline folds", folds)] + [(f"split seed {s}", list(StratifiedKFold(5, shuffle=True, random_state=s).split(M[0], y))) for s in (1, 2, 3)]
summary = {}
for label, fl in splits:
    fo = np.zeros(len(y), int)
    for i, (_, va) in enumerate(fl):
        fo[va] = i
    Ps = stack(LOGS, fl)
    base = nested_mult(Ps, fl)
    geo = np.exp(0.5 * np.log(np.clip(Ps, 1e-9, 1)) + 0.5 * np.log(np.clip(kde_oof, 1e-9, 1)))
    wn, chosen = nested_w(Ps, fl)
    cands = {"0.5 stack + 0.5 KDE NB": nested_mult(0.5 * Ps + 0.5 * kde_oof, fl),
             "0.7 stack + 0.3 KDE NB": nested_mult(0.7 * Ps + 0.3 * kde_oof, fl),
             "w chosen nested": wn,
             "geometric 0.5 / 0.5": nested_mult(geo / geo.sum(1, keepdims=True), fl),
             "KDE NB as 5th stack member": nested_mult(stack(LOGS + [np.log(np.clip(kde_oof, 1e-6, 1))], fl), fl)}
    print(f"\n{label}: stack + nested multipliers {mf1(y, base):.4f} (nested w per fold: {chosen})")
    for k, p in cands.items():
        d = mf1(y, p) - mf1(y, base)
        pf = [mf1(y[fo == i], p[fo == i]) - mf1(y[fo == i], base[fo == i]) for i in range(5)]
        bs = [mf1(y[i], p[i]) - mf1(y[i], base[i]) for i in boot]
        summary.setdefault(k, []).append((d, sum(x > 0 for x in pf)))
        print(f"  {k:28s} {mf1(y, p):.4f}  delta {d:+.4f} [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}]  folds +{sum(x > 0 for x in pf)}/5 "
              f"({' '.join(f'{x:+.4f}' for x in pf)})  labels changed {np.mean(p != base):.2%}", flush=True)
print("\nmean delta over the 4 splits (folds better, per split):")
for k, v in summary.items():
    print(f"  {k:28s} {np.mean([d for d, _ in v]):+.4f}  ({' '.join(str(n) + '/5' for _, n in v)})")
