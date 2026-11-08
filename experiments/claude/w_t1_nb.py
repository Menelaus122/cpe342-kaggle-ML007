"""Task 1: histogram Naive Bayes and latent-class Naive Bayes (missing values marginalised exactly), stacked.

A missing feature simply drops out of the likelihood, which is the exact treatment when values are missing
completely at random. Tree models instead send NaN down one learned branch.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
import common  # noqa: E402
from common import load_t1, T1_MODELS, t1_score, stack_binary, OUT  # noqa: E402
from src.metrics import optimize_threshold  # noqa: E402
from src.utils import load_preds  # noqa: E402

NB = 32


def binize(train_col, cols_all):
    """Quantile bin edges from training rows; returns bin index (-1 = NaN) for each array in cols_all."""
    v = train_col[np.isfinite(train_col)]
    edges = np.unique(np.quantile(v, np.linspace(0, 1, NB + 1)[1:-1]))
    return [np.where(np.isfinite(c), np.searchsorted(edges, c, side="right"), -1) for c in cols_all], len(edges) + 1


def encode(Xa, Xb_list):
    """Bin every column with edges from Xa. Returns (Ba, [Bb...], n_bins per column)."""
    Ba, Bbs, nb = [], [[] for _ in Xb_list], []
    for j in range(Xa.shape[1]):
        (a, *bs), k = binize(Xa[:, j], [Xa[:, j]] + [Xb[:, j] for Xb in Xb_list])
        Ba.append(a); nb.append(k)
        for i, b in enumerate(bs):
            Bbs[i].append(b)
    return np.column_stack(Ba), [np.column_stack(b) for b in Bbs], nb


def loglik_table(B, w, nb, alpha=1.0):
    """Per-feature log P(bin | component) from (soft) weights w, Laplace-smoothed; NaN rows ignored."""
    tabs = []
    for j, k in enumerate(nb):
        ok = B[:, j] >= 0
        cnt = np.bincount(B[ok, j], weights=w[ok], minlength=k) + alpha
        tabs.append(np.log(cnt / cnt.sum()))
    return tabs


def row_loglik(B, tabs):
    out = np.zeros(len(B))
    for j, t in enumerate(tabs):
        ok = B[:, j] >= 0
        out[ok] += t[B[ok, j]]
    return out


def fit_lcnb(B, nb, C, n_iter=40, seed=0):
    """Latent-class NB within one class: C components, EM on binned features."""
    rng = np.random.default_rng(seed)
    R = rng.dirichlet(np.ones(C), size=len(B))
    for _ in range(n_iter):
        pi = R.mean(0)
        tabs = [loglik_table(B, R[:, c], nb) for c in range(C)]
        L = np.column_stack([np.log(pi[c]) + row_loglik(B, tabs[c]) for c in range(C)])
        L -= L.max(1, keepdims=True)
        R = np.exp(L); R /= R.sum(1, keepdims=True)
    return pi, tabs


def mix_loglik(B, pi, tabs):
    L = np.column_stack([np.log(pi[c]) + row_loglik(B, tabs[c]) for c in range(len(pi))])
    mx = L.max(1)
    return mx + np.log(np.exp(L - mx[:, None]).sum(1))


def nb_cv(Xr, y, Xt, folds, C=(1, 1)):
    """C = (components for class 0, components for class 1). Returns OOF and test log-odds."""
    oof, test = np.zeros(len(y)), np.zeros(len(Xt))
    for a, b in folds:
        Ba, (Bb, Bt), nb = encode(Xr[a], [Xr[b], Xt])
        ya = y[a]
        prior = np.log(ya.mean() / (1 - ya.mean()))
        ll = []
        for k in (0, 1):
            Bk = Ba[ya == k]
            if C[k] == 1:
                tabs = loglik_table(Bk, np.ones(len(Bk)), nb)
                ll.append((row_loglik(Bb, tabs), row_loglik(Bt, tabs)))
            else:
                pi, tabs = fit_lcnb(Bk, nb, C[k])
                ll.append((mix_loglik(Bb, pi, tabs), mix_loglik(Bt, pi, tabs)))
        oof[b] = prior + ll[1][0] - ll[0][0]
        test += (prior + ll[1][1] - ll[0][1]) / len(folds)
    return oof, test


if __name__ == "__main__":
    tr, te, X, y, Xt, folds, _ = load_t1()
    raw_cols = [c for c in tr.columns if c not in ("player_id", "is_cheater")]
    Xr, Xtr = tr[raw_cols].to_numpy(float), te[raw_cols].to_numpy(float)
    nan_cnt = np.isnan(Xr).sum(1)
    base = [load_preds(n) for n in T1_MODELS]
    s0, thr0, so0, st0 = t1_score([d["oof"] for d in base], [d["test"] for d in base], y, folds)
    print(f"baseline stack: AUC {roc_auc_score(y, so0):.4f}, F2 {s0:.4f} (tau {thr0:.3f})")
    lgb = base[0]["oof"]
    strata = [(0, 0), (1, 2), (3, 3), (4, 5), (6, 99)]

    def by_nan(p):
        return " ".join(f"{roc_auc_score(y[(nan_cnt >= lo) & (nan_cnt <= hi)], p[(nan_cnt >= lo) & (nan_cnt <= hi)]):.4f}"
                        for lo, hi in strata)

    print(f"AUC by #NaN {strata}:\n  LightGBM         {by_nan(lgb)}\n  stack            {by_nan(so0)}")
    res = {}
    for C in [(1, 1), (2, 2), (3, 3), (2, 4)]:
        o, t = nb_cv(Xr, y, Xtr, folds, C)
        res[C] = (o, t)
        np.savez(OUT / f"t1_nb_{C[0]}{C[1]}.npz", oof=o, test=t)
        thr, f = optimize_threshold(y, 1 / (1 + np.exp(-o)), 2.0)
        print(f"NB components {C}: AUC {roc_auc_score(y, o):.4f}, F2 alone {f:.4f}  | by #NaN {by_nan(o)}")
    for C, (o, t) in res.items():
        p, pt = 1 / (1 + np.exp(-o)), 1 / (1 + np.exp(-t))
        s, thr, so, st = t1_score([d["oof"] for d in base] + [p], [d["test"] for d in base] + [pt], y, folds)
        print(f"stack + NB{C}: AUC {roc_auc_score(y, so):.4f}, F2 {s:.4f} (tau {thr:.3f}), delta {s - s0:+.4f}")
