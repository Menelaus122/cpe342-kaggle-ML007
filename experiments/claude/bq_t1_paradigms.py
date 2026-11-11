"""Round 12, Task 1: the "paradigm shift" brief, only the parts not tested in earlier rounds (FINDINGS.md has the rest:
MLP z_t1_mlp, kNN E1, focal Q1, pairwise ranking Q2, expected-F2 rule Q3, isotonic A5, row order / cross-task checks).

Baseline B = the current pipeline stack: our 5 members + last year's 4 (2-seed averages), nested LR stacker + nested cut-off
(bo_t1_seedavg.py V2, F2 0.8383). Every new member is fitted on the 5 pipeline folds, scored on the held-out fold and on the
test-like copy of it (test missingness), exactly like the members in bn_t1_combined.py.
  N1 spline LR      SplineTransformer (quantile knots) + missing indicators + standardisation + logistic regression
  N2 Mahalanobis    LightGBM on our 49 features + in-fold class-conditional Mahalanobis distances (Ledoit-Wolf covariances)
  N3 DAE            LightGBM on our 49 features + 128-d bottleneck of a swap-noise denoising autoencoder fitted on the
                    TRAINING-FOLD rows only (no test rows: the rules forbid using the test set for training-related purposes)
  N4 TabM-BCE       TabM-style parameter-efficient MLP ensemble (BatchEnsemble, k = 16; Gorishniy et al., ICLR 2025)
  N5 TabM-softF2    the same with BCE + (1 - soft-F2) loss
  S1 rank average   members' rank-normalised scores averaged (no stacker), nested cut-off
  S2 hill climbing  Caruana et al. (2004) forward selection with replacement on F2, fitted on the training folds only
Acceptance rule unchanged: CI > 0, >= 4/5 folds better, fold std <= 1.1x, test-like delta >= 0.
"""
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.covariance import LedoitWolf
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, SplineTransformer, StandardScaler

warnings.filterwarnings("ignore")
import common  # noqa: F401,E402
from common import OUT  # noqa: E402

exec(open(Path(__file__).with_name("bn_t1_combined.py"), encoding="utf-8").read().split("member_f2 = lambda")[0])
from src.modeling import fit_predict  # noqa: E402
from src.metrics import optimize_threshold  # noqa: E402

P = {"lgb": tuned("task1_lgb", {"learning_rate": 0.05, "scale_pos_weight": spw})}  # the pipeline's LightGBM member settings
T0 = time.time()
log = lambda *a: print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
Z = np.load(OUT / "bn_t1_members.npz")
get = lambda k: (Z[f"{k}_oof"], Z[f"{k}_tl"])
avg = lambda k: tuple((Z[f"{k}_42_{s}"] + Z[f"{k}_7_{s}"]) / 2 for s in ("oof", "tl"))
M = {k: get(k) for k in ("lgb", "xgb", "cat", "et", "lr")}
for k in ("rf", "lgb", "xgb", "cat"):
    M[f"their_{k}"] = avg(f"their_{k}")
BASE = ["lgb", "xgb", "cat", "et", "lr", "their_rf", "their_lgb", "their_xgb", "their_cat"]
ref = stack_pred([M[k][0] for k in BASE])
ref_tl = stack_pred([M[k][0] for k in BASE], [M[k][1] for k in BASE])
fr = per_fold(ref)
log(f"B (pipeline stack, 9 members): F2 {f2(y, ref):.4f}, folds {np.round(fr, 4)}, std {fr.std():.4f}; test-like {wf2(ref_tl, w_tl):.4f}")
Xv, Xtlv = X.values.astype(float), X_tl.values.astype(float)
nan_cols = np.where(np.isnan(Xv).any(0))[0]
rows = []


def judge(name, p, p_tl):
    fo = per_fold(p)
    lo, hi = boot(p, ref)
    tlo, thi = boot(p_tl, ref_tl, w_tl)
    dtl = wf2(p_tl, w_tl) - wf2(ref_tl, w_tl)
    better = int((fo > fr).sum())
    keep = lo > 0 and better >= 4 and fo.std() <= fr.std() * 1.1 and dtl >= 0
    rows.append([name, f2(y, p), f2(y, p) - f2(y, ref), f"[{lo:+.4f}, {hi:+.4f}]", f"{better}/5", fo.std(), f"{dtl:+.4f}",
                 f"[{tlo:+.4f}, {thi:+.4f}]", "KEEP" if keep else "discard"])
    log(f"{name}: F2 {rows[-1][1]:.4f} ({rows[-1][2]:+.4f}) CI {rows[-1][3]}, {better}/5, std {fo.std():.4f}; test-like {dtl:+.4f} CI {rows[-1][7]}"
        f" -> {rows[-1][-1]}")


def add_member(name, o, otl):
    M[name] = (o, otl)
    log(f"  member {name}: F2 alone {f2(y, tau_pred(o)):.4f}")
    judge(f"+ {name}", stack_pred([M[k][0] for k in BASE + [name]]), stack_pred([M[k][0] for k in BASE + [name]], [M[k][1] for k in BASE + [name]]))


def prep(a, b_list):
    """Median fill + missing flags + quantile-normal, fitted on the training-fold rows a only."""
    med_a = np.nanmedian(a, 0)
    fill = lambda A: np.where(np.isnan(A), med_a, A)
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000, subsample=200_000, random_state=0).fit(fill(a))
    f = lambda A: np.hstack([qt.transform(fill(A)), np.isnan(A[:, nan_cols]).astype(float)]).astype(np.float32)
    return f(a), [f(B) for B in b_list]


# ------------------------------------------------ N1 spline logistic regression
spl = lambda: make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                            ColumnTransformer([("s", SplineTransformer(n_knots=6, degree=3, knots="quantile"), list(range(X.shape[1])))],
                                              remainder="passthrough"),
                            StandardScaler(), LogisticRegression(C=0.05, max_iter=4000))
o, otl = cv_sk(spl, Xv, Xtlv)
add_member("N1 spline LR", o, otl)

# ------------------------------------------------ N2 class-conditional Mahalanobis distances
o, otl = np.zeros(len(y)), np.zeros(len(y))
for a, b in folds:
    za, (zb, ztl) = prep(Xv[a], [Xv[b], Xtlv[b]])
    stats = []
    for c in (0, 1):
        lw = LedoitWolf().fit(za[y[a] == c])
        stats.append((lw.location_, lw.precision_))
    dist = lambda Zm: np.column_stack([np.sqrt(np.einsum("ij,jk,ik->i", Zm - mu, P_, Zm - mu)) for mu, P_ in stats])
    feat = lambda Zm, Xm: pd.DataFrame(np.hstack([Xm, dist(Zm), (dist(Zm)[:, :1] - dist(Zm)[:, 1:])]),
                                       columns=list(X.columns) + ["maha0", "maha1", "maha_diff"])
    _, (pv, pt), _ = fit_predict("lgb", "binary", P["lgb"], feat(za, Xv[a]), y[a], feat(zb, Xv[b]), y[b], [feat(zb, Xv[b]), feat(ztl, Xtlv[b])])
    o[b], otl[b] = pv, pt
add_member("N2 LightGBM + Mahalanobis", o, otl)


# ------------------------------------------------ N3 denoising autoencoder features (training-fold rows only)
class DAE(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.enc = nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Linear(512, 128), nn.ReLU())
        s.dec = nn.Sequential(nn.Linear(128, 512), nn.ReLU(), nn.Linear(512, d))

    def forward(s, x):
        return s.dec(s.enc(x))


def fit_dae(za, epochs=40, swap=0.15, seed=0):
    torch.manual_seed(seed)
    net = DAE(za.shape[1]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-5)
    Xa = torch.tensor(za, device=DEV)
    n = len(Xa)
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, 512):
            xb = Xa[perm[i:i + 512]]
            mask = torch.rand_like(xb) < swap  # swap noise: replace a cell with the same column of a random training row
            donor = Xa[torch.randint(0, n, (len(xb),), device=DEV)]
            loss = ((net(torch.where(mask, donor, xb)) - xb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return lambda Zm: net.enc(torch.tensor(Zm, device=DEV)).detach().cpu().numpy()


o, otl = np.zeros(len(y)), np.zeros(len(y))
for a, b in folds:
    za, (zb, ztl) = prep(Xv[a], [Xv[b], Xtlv[b]])
    enc = fit_dae(za)
    feat = lambda Zm, Xm: pd.DataFrame(np.hstack([Xm, enc(Zm)]), columns=list(X.columns) + [f"dae{i}" for i in range(128)])
    _, (pv, pt), _ = fit_predict("lgb", "binary", P["lgb"], feat(za, Xv[a]), y[a], feat(zb, Xv[b]), y[b], [feat(zb, Xv[b]), feat(ztl, Xtlv[b])])
    o[b], otl[b] = pv, pt
    log("  DAE fold done")
add_member("N3 LightGBM + DAE bottleneck", o, otl)


# ------------------------------------------------ N4 / N5 TabM-style MLP ensemble
class BELinear(nn.Module):
    def __init__(s, d_in, d_out, k, first):
        super().__init__()
        s.W = nn.Linear(d_in, d_out, bias=False)
        r = torch.ones(k, d_in)
        if first:
            r = torch.where(torch.rand(k, d_in) < 0.5, -1.0, 1.0)
        s.r = nn.Parameter(r)
        s.s = nn.Parameter(torch.ones(k, d_out))
        s.b = nn.Parameter(torch.zeros(k, d_out))

    def forward(s, x):  # x: (B, k, d_in)
        return s.W(x * s.r) * s.s + s.b


class TabM(nn.Module):
    def __init__(s, d, k=16, width=384, depth=3, drop=0.15):
        super().__init__()
        s.k = k
        layers, d_in = [], d
        for i in range(depth):
            layers += [BELinear(d_in, width, k, i == 0), nn.ReLU(), nn.Dropout(drop)]
            d_in = width
        s.body = nn.Sequential(*layers)
        s.head = BELinear(width, 1, k, False)

    def forward(s, x):
        return s.head(s.body(x.unsqueeze(1).expand(-1, s.k, -1))).squeeze(-1)  # (B, k) logits


def fit_tabm(za, ya, zb_list, soft_f2=False, epochs=30, seed=0):
    torch.manual_seed(seed)
    rs = np.random.default_rng(seed)
    inner = rs.random(len(za)) < 0.1  # early stopping on 10% of the TRAINING-fold rows
    Xt_, yt_ = torch.tensor(za[~inner], device=DEV), torch.tensor(ya[~inner], dtype=torch.float32, device=DEV)
    Xi_, yi_ = torch.tensor(za[inner], device=DEV), torch.tensor(ya[inner], dtype=torch.float32, device=DEV)
    net = TabM(za.shape[1]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=3e-4)
    steps = epochs * int(np.ceil(len(Xt_) / 1024))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps, pct_start=0.1)
    bce = nn.BCEWithLogitsLoss()
    best, best_state = 1e9, None
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt_), device=DEV)
        for i in range(0, len(Xt_), 1024):
            idx = perm[i:i + 1024]
            lg = net(Xt_[idx]); yb = yt_[idx].unsqueeze(1).expand_as(lg)
            loss = bce(lg, yb)
            if soft_f2:
                p = torch.sigmoid(lg)
                tp = (p * yb).sum(0); fp = (p * (1 - yb)).sum(0); fn = ((1 - p) * yb).sum(0)
                loss = loss + (1 - 5 * tp / (5 * tp + 4 * fn + fp + 1e-6)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        net.eval()
        with torch.no_grad():
            vl = bce(net(Xi_).mean(1), yi_).item()
        if vl < best:
            best, best_state = vl, {k_: v.clone() for k_, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        return [torch.sigmoid(net(torch.tensor(Zm, device=DEV))).mean(1).cpu().numpy() for Zm in zb_list]


for name, sf in (("N4 TabM (BCE)", False), ("N5 TabM (BCE + soft-F2)", True)):
    o, otl = np.zeros(len(y)), np.zeros(len(y))
    for a, b in folds:
        za, (zb, ztl) = prep(Xv[a], [Xv[b], Xtlv[b]])
        o[b], otl[b] = fit_tabm(za, y[a], [zb, ztl], soft_f2=sf)
    add_member(name, o, otl)

# ------------------------------------------------ S1 / S2 combination rules on the baseline members (no new member)
srt = {k: np.sort(M[k][0]) for k in BASE}
rk = lambda k, v: np.searchsorted(srt[k], v) / len(srt[k])  # empirical CDF of the member's own OOF scores (label-free)
r_oof = np.mean([rk(k, M[k][0]) for k in BASE], 0)
r_tl = np.mean([rk(k, M[k][1]) for k in BASE], 0)
judge("S1 rank average (no stacker)", tau_pred(r_oof), tau_pred(r_oof, r_tl))


def hill(Lo, a, iters=40):
    """Caruana forward selection with replacement on the training-fold rows a; objective = F2 at the best cut-off."""
    score = lambda s: optimize_threshold(y[a], s[a], 2.0, average="binary")[1]
    counts = np.zeros(Lo.shape[1])
    j0 = int(np.argmax([score(Lo[:, j]) for j in range(Lo.shape[1])]))
    counts[j0] = 1
    cur = score(Lo[:, j0])
    for _ in range(iters):
        cand = [score((Lo @ (counts + np.eye(len(counts))[j])) / (counts.sum() + 1)) for j in range(Lo.shape[1])]
        j = int(np.argmax(cand))
        if cand[j] <= cur:
            break
        counts[j] += 1
        cur = cand[j]
    return counts / counts.sum()


Lo = np.column_stack([_logit(M[k][0]) for k in BASE])
Ltl = np.column_stack([_logit(M[k][1]) for k in BASE])
p, p_tl = np.zeros(len(y), int), np.zeros(len(y), int)
for a, b in folds:
    w = hill(Lo, a)
    s = Lo @ w
    t = optimize_threshold(y[a], s[a], 2.0, average="binary")[0]
    p[b], p_tl[b] = s[b] >= t, (Ltl @ w)[b] >= t
    log(f"  hill-climb weights: {dict(zip(BASE, np.round(w, 2)))}")
judge("S2 hill climbing (Caruana), nested", p, p_tl)

print("\n" + pd.DataFrame(rows, columns=["Variant", "OOF F2", "delta", "95% CI", "folds better", "fold std", "test-like delta", "test-like CI",
                                         "Decision"]).to_string(index=False), flush=True)
np.savez(OUT / "bq_t1_members.npz", **{f"{k}_oof": v[0] for k, v in M.items() if k.startswith("N")},
         **{f"{k}_tl": v[1] for k, v in M.items() if k.startswith("N")})
log("done")
