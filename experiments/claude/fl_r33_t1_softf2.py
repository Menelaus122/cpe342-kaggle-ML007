"""Round 33, Task 1 item 1: a level-2 combiner trained on a differentiable F2 instead of the log-loss (training rows only, CPU).

    python fl_r33_t1_softf2.py

The v22 level 2 (LogReg + LightGBM + CatBoost on the nine member log-odds) minimises the log-loss, which weights every row equally,
and the F2 cut-off (about 0.16) is chosen afterwards. Candidates trained directly on the soft F2
    F2_soft = 5 sum(y p) / (4 sum(y) + sum(p)),  p = sigmoid(score)
(the exact F2 with the 0/1 decisions replaced by p; round 12 tried a soft-F2 term only inside a level-1 TabM, never at level 2):
  A  linear combiner of the nine log-odds, the 31 missing-cell indicators and the missing count (standardised on the training folds),
     started from the logistic solution shifted to its F2 cut-off, L-BFGS on 1 - F2_soft + 1e-4 |w|^2;
  B  A plus a small residual MLP (16 tanh units, output layer started at zero, Adam 300 full-batch steps, weight decay 1e-3,
     mean of 3 seeds);
  C  the v22 level 2 and A averaged (each score standardised on the training folds);
  D  A on the nine log-odds only (no missing-cell inputs).
Each candidate's cut-off for a fold is the exact F2 optimum of the other four folds' out-of-fold scores (same rule as the base);
A is also read at its built-in cut (score > 0). Nested on the pipeline folds and stacker split seeds 1, 2, 3.
Acceptance (user, 2026-11-16): pooled delta > 0, >= 3/5 folds on each split, no fold below -0.0005.
"""
import numpy as np
import torch

import r33_t1_common as c
from r33_t1_common import L, SPLITS, f2, fold_of, labels_exact, labels_from, log, y

torch.set_num_threads(c.NT)
M_IND = c.tr[c.RAW].isna().values.astype(float)
X_ALL = np.column_stack([L, M_IND, c.nan_n])
yt_all = torch.tensor(y, dtype=torch.float64)


def soft_f2_loss(s, yt):
    p = torch.sigmoid(s)
    return 1 - 5 * (yt * p).sum() / (4 * yt.sum() + p.sum())


def fit_linear(X, a):
    """Soft-F2 linear combiner fitted on rows a. Returns (mean, sd, w, b)."""
    from sklearn.linear_model import LogisticRegression
    mu, sd = X[a].mean(0), X[a].std(0) + 1e-9
    Z = (X[a] - mu) / sd
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(Z, y[a])
    s0 = Z @ lr.coef_[0] + lr.intercept_[0]
    cut = c.best_cut(y[a], s0)
    w = torch.tensor(lr.coef_[0] * 2.0, dtype=torch.float64, requires_grad=True)          # sharpened start
    b = torch.tensor((lr.intercept_[0] - cut) * 2.0, dtype=torch.float64, requires_grad=True)
    Zt, yt = torch.tensor(Z), yt_all[a]
    opt = torch.optim.LBFGS([w, b], max_iter=200, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = soft_f2_loss(Zt @ w + b, yt) + 1e-4 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return mu, sd, w.detach().numpy(), float(b)


def fit_mlp(X, a, lin, seed):
    mu, sd, w, b = lin
    Zt, yt = torch.tensor((X[a] - mu) / sd), yt_all[a]
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)).double()
    torch.nn.init.zeros_(net[2].weight)
    torch.nn.init.zeros_(net[2].bias)
    wl = torch.tensor(w, requires_grad=True)
    bl = torch.tensor(b, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([{"params": net.parameters(), "weight_decay": 1e-3}, {"params": [wl, bl], "weight_decay": 0.0}], lr=3e-3)
    for _ in range(300):
        opt.zero_grad()
        loss = soft_f2_loss(Zt @ wl + bl + net(Zt).squeeze(1), yt) + 1e-4 * (wl ** 2).sum()
        loss.backward()
        opt.step()
    return lambda Xn: ((Xn - mu) / sd) @ wl.detach().numpy() + float(bl) + net(torch.tensor((Xn - mu) / sd)).squeeze(1).detach().numpy()


base_p = c.base_probs()
log("base level 2 ready")
names = ["A linear soft-F2 (cut from the other folds)", "A linear soft-F2 (built-in cut, score > 0)", "B linear + residual MLP soft-F2",
         "C mean of v22 level 2 and A (standardised)", "D linear soft-F2, nine log-odds only"]
cand = {n: [] for n in names}
base_labels = []
for k, (label, fl) in enumerate(SPLITS):
    fo = fold_of(fl)
    sA, sB, sC, sD = (np.zeros(len(y)) for _ in range(4))
    zb = c.logit(base_p[k])
    for a, b in fl:
        lin = fit_linear(X_ALL, a)
        mu, sd, w, bb = lin
        sA[b] = ((X_ALL[b] - mu) / sd) @ w + bb
        sa_tr = ((X_ALL[a] - mu) / sd) @ w + bb
        fns = [fit_mlp(X_ALL, a, lin, s) for s in (0, 1, 2)]
        sB[b] = np.mean([f(X_ALL[b]) for f in fns], 0)
        sC[b] = (zb[b] - zb[a].mean()) / zb[a].std() + (sA[b] - sa_tr.mean()) / sa_tr.std()
        mu9, sd9, w9, b9 = fit_linear(L, a)
        sD[b] = ((L[b] - mu9) / sd9) @ w9 + b9
    base_labels.append(labels_from(base_p[k], fo))
    cand[names[0]].append(labels_exact(sA, fo))
    cand[names[1]].append((sA > 0).astype(int))
    cand[names[2]].append(labels_exact(sB, fo))
    cand[names[3]].append(labels_exact(sC, fo))
    cand[names[4]].append(labels_exact(sD, fo))
    log(f"{label}: base {f2(y, base_labels[-1]):.5f}; " + "; ".join(f"{n.split(' ')[0]} {f2(y, cand[n][-1]):.5f}" for n in names)
        + f"; base under the exact cut-off search {f2(y, labels_exact(base_p[k], fo)):.5f}")
print()
for n in names:
    c.report(n, cand[n], base_labels)
log("done")
