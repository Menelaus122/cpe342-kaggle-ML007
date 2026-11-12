"""Round 16, Task 1: the round-12 TabM (BCE) and spline-LogReg members again, now with test predictions.

    python ce_r16_t1_members.py

Same code as bq_t1_paradigms.py (N1, N4): preprocessing fitted on training-fold rows only, TabM early-stops on 10% of the
training-fold rows. Per member: 5-fold OOF, test-like OOF (rows with the test's missingness), fold-average test predictions and a
Train+Val refit on all labelled rows (the pipeline's final test input is 0.5 x fold average + 0.5 x refit).
Writes outputs/r16_t1_members.npz.
"""
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, SplineTransformer, StandardScaler

warnings.filterwarnings("ignore")
import common  # noqa: F401,E402
from common import OUT  # noqa: E402
from by_r14_bag import setup_t1  # noqa: E402

T0 = time.time()
log = lambda *a: print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
X, y, Xt, X_tl, *_rest = setup_t1()
folds = _rest[-3]
Xv, Xtv, Xtlv = X.values.astype(float), Xt.values.astype(float), X_tl.values.astype(float)
nan_cols = np.where(np.isnan(Xv).any(0))[0]


def prep(a, b_list):
    """Median fill + missing flags + quantile-normal, fitted on the rows a only."""
    med_a = np.nanmedian(a, 0)
    fill = lambda A: np.where(np.isnan(A), med_a, A)
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000, subsample=200_000, random_state=0).fit(fill(a))
    f = lambda A: np.hstack([qt.transform(fill(A)), np.isnan(A[:, nan_cols]).astype(float)]).astype(np.float32)
    return f(a), [f(B) for B in b_list]


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

    def forward(s, x):
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
        return s.head(s.body(x.unsqueeze(1).expand(-1, s.k, -1))).squeeze(-1)


def fit_tabm(za, ya, zb_list, epochs=30, seed=0):
    torch.manual_seed(seed)
    rs = np.random.default_rng(seed)
    inner = rs.random(len(za)) < 0.1
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
            lg = net(Xt_[idx])
            loss = bce(lg, yt_[idx].unsqueeze(1).expand_as(lg))
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


spl = lambda: make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                            ColumnTransformer([("s", SplineTransformer(n_knots=6, degree=3, knots="quantile"), list(range(X.shape[1])))], remainder="passthrough"),
                            StandardScaler(), LogisticRegression(C=0.05, max_iter=4000))

out = {}
for name in ("spline", "tabm"):
    oof, tl, test = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(Xtv))
    for a, b in folds:
        if name == "spline":
            m = spl().fit(Xv[a], y[a])
            oof[b], tl[b], t = m.predict_proba(Xv[b])[:, 1], m.predict_proba(Xtlv[b])[:, 1], m.predict_proba(Xtv)[:, 1]
        else:
            za, (zb, ztl, zt) = prep(Xv[a], [Xv[b], Xtlv[b], Xtv])
            oof[b], tl[b], t = fit_tabm(za, y[a], [zb, ztl, zt])
        test += t / len(folds)
    if name == "spline":
        full = spl().fit(Xv, y).predict_proba(Xtv)[:, 1]
    else:
        za, (zt,) = prep(Xv, [Xtv])
        full = fit_tabm(za, y, [zt])[0]
    out.update({f"{name}_oof": oof, f"{name}_tl": tl, f"{name}_test": test, f"{name}_full": full})
    old = np.load(OUT / "bq_t1_members.npz")["N1 spline LR_oof" if name == "spline" else "N4 TabM (BCE)_oof"]
    log(f"{name}: done; corr with the round-12 OOF {np.corrcoef(oof, old)[0, 1]:.5f}, max |diff| {np.abs(oof - old).max():.4g}; "
        f"corr(fold-average test, refit test) {np.corrcoef(test, full)[0, 1]:.4f}")
np.savez(OUT / "r16_t1_members.npz", **out)
log("saved outputs/r16_t1_members.npz")
