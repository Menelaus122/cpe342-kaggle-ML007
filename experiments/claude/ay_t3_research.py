"""Task 3 round 9 (2026-11-08): methods from the literature, strict 5-fold OOF against the merged pipeline.

Baseline B0 = the pipeline's Task 3 pool (outputs/preds/t3r_*, 26 members) with the exact LP L1 blend = 0.79423.
R1 customer-base analysis (Fader, Hardie & Lee 2005, BG/NBD + Gamma-Gamma): fit the two "buy till you die" models on
   TRAIN features (frequency = total_transactions, recency = days_since_last_purchase, age = account_age_days,
   monetary = avg_transaction_value; no target), use P(alive), expected purchases in the next 30 days, expected value per
   purchase and their product as features of a new hurdle (stage 1 + ratio stage 2) -> 3 read-outs added to the pool.
R2 zero-inflated lognormal network (Wang, Liu & Miao 2019, "A Deep Probabilistic Model for Customer Lifetime Value
   Prediction"): one MLP outputs P(y>0), mu and a per-row sigma; ZILN loss; mu = log1p(hist) + network output (ratio
   parametrisation); read out the mixture MEDIAN (the metric is L1). Added to the pool.
R3 post-processing: L1-optimal multiplier per prediction decile (weighted median of y / yhat), nested.
R4 blending: ensemble selection with replacement (Caruana et al. 2004), nested, instead of the exact LP weights.
Significance: paired Poisson bootstrap (1,000) of the score difference; keep only if the 95% interval is above 0.
"""
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
from scipy.special import gammaln
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import QuantileTransformer

warnings.filterwarnings("ignore")
import common  # noqa: E402
from common import OUT  # noqa: E402
from src import task3  # noqa: E402
from src.metrics import nmae  # noqa: E402
from src.modeling import make_folds, fit_predict, LGB_BASE  # noqa: E402
from src.utils import load_preds, load_params  # noqa: E402

T0 = time.time()
log = lambda *a: print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
rng = np.random.default_rng(0)
tr, te = task3.load()
X, y, Xt, _ = task3.build_repaired(tr, te)
folds = make_folds(y, kind="regression")
z, pos = np.log1p(y), y > 0
clip0 = lambda v: np.clip(v, 0, None)

# ------------------------------------------------------------------ B0: the pipeline pool, from its caches
m3 = {}
for name, key, logt in [("LGB-L1 raw THB", "t3r_lgb_raw", False), ("XGB-L1 raw THB", "t3r_xgb_raw", False),
                        ("Cat-MAE raw THB", "t3r_cat_raw", False), ("LGB-L1 log1p", "t3r_lgb_log", True),
                        ("LGB-Tweedie (mean)", "t3r_tweedie", False)]:
    r = load_preds(key)
    m3[name] = (clip0(np.expm1(r["oof"])), clip0(np.expm1(r["test"]))) if logt else (clip0(r["oof"]), clip0(r["test"]))
h = load_preds("t3r_hurdle")
TAGS = ["ratio[hist] LGB tuned", "ratio[prev] LGB tuned", "ratio[hist] LGB", "ratio[prev] LGB", "ratio[hist] XGB", "ratio[hist] Cat"]
slug3 = lambda tag: "t3r_" + tag.replace("[", "_").replace("] ", "_").replace(" ", "_").lower()
mu3 = {"Hurdle": (h["mu_oof"], h["mu_test"])}
for tag in TAGS:
    rr = load_preds(slug3(tag))
    mu3[tag] = (rr["mu_oof"], rr["mu_test"])
RO = {}
for tag, (mu, mu_t) in mu3.items():
    RO[tag] = task3.readouts(y, h["p_oof"], mu)
    for k in ("mixture-median", "hard gate", "E[y] + smearing"):
        m3[f"{tag} {k}"] = (task3.apply_readout(k, h["p_oof"], mu, RO[tag]), task3.apply_readout(k, h["p_test"], mu_t, RO[tag]))
names = list(m3)
P = np.column_stack([m3[k][0] for k in names])
gate = (np.clip(h["p_oof"] + RO["Hurdle"]["shift"], 0, 1) > 0.5).astype(float)
scales = np.linspace(0.85, 1.15, 61)


def finish(bo):
    E = np.abs(y[None, :] - scales[:, None] * (bo * gate)[None, :]).sum(1)
    return bo * scales[int(E.argmin())] * gate


def lp_nested(Pm):
    bo = np.zeros(len(y))
    for a, b in folds:
        bo[b] = Pm[b] @ task3.l1_blend_weights_exact(y[a], [Pm[a, j] for j in range(Pm.shape[1])])
    return finish(bo)


def boot(a, b, B=1000):
    e1, e0 = np.abs(y - a), np.abs(y - b)
    d = np.empty(B)
    for i in range(B):
        w = rng.poisson(1.0, len(y)).astype(float)
        sy = w @ y
        d[i] = 1 / (1 + (w @ e1) / sy) - 1 / (1 + (w @ e0) / sy)
    return np.percentile(d, [2.5, 97.5])


rows = []


def report(name, desc, pred, ref, ref_name="B0"):
    s, s0 = nmae(y, pred), nmae(y, ref)
    lo, hi = boot(pred, ref)
    rows.append([name, desc, s, ref_name, s - s0, f"[{lo:+.5f}, {hi:+.5f}]", "KEEP" if lo > 0 else "discard"])
    log(f"{name}: {s:.5f} (delta vs {ref_name} {s - s0:+.5f}, CI [{lo:+.5f}, {hi:+.5f}])")


f0 = lp_nested(P)
log(f"B0 pipeline pool, exact LP blend: {nmae(y, f0):.5f} (notebook: 0.79423)")

# ------------------------------------------------------------------ R3: decile multipliers (nested)
def decile_calibrate(f, nb=10):
    out = f.copy()
    for a, v in folds:
        pa = f[a] > 0
        edges = np.quantile(f[a][pa], np.linspace(0, 1, nb + 1))[1:-1]
        ba, bv = np.digitize(f[a], edges), np.digitize(f[v], edges)
        for j in range(nb):
            ma, mv = pa & (ba == j), (f[v] > 0) & (bv == j)
            if ma.sum() < 100 or not mv.any():
                continue
            r_, w_ = y[a][ma] / f[a][ma], f[a][ma]
            o = np.argsort(r_)
            cw = np.cumsum(w_[o])
            c = r_[o][np.searchsorted(cw, cw[-1] / 2)]        # L1-optimal multiplier = weighted median of y / f
            out[v[mv]] = f[v][mv] * c
    return out


report("R3", "L1-optimal multiplier per prediction decile (nested)", decile_calibrate(f0), f0)

# ------------------------------------------------------------------ R4: ensemble selection with replacement (nested)
def caruana(Pa, ya, n_iter=80, init=3):
    k = Pa.shape[1]
    sel = list(np.argsort([np.abs(ya - Pa[:, j]).sum() for j in range(k)])[:init])
    cur = Pa[:, sel].sum(1)
    for _ in range(n_iter):
        n = len(sel) + 1
        j = int(np.abs(ya[:, None] - (cur[:, None] + Pa) / n).sum(0).argmin())
        sel.append(j)
        cur = cur + Pa[:, j]
    return np.bincount(sel, minlength=k) / len(sel)


bo = np.zeros(len(y))
for a, b in folds:
    bo[b] = P[b] @ caruana(P[a], y[a])
report("R4", "ensemble selection with replacement (Caruana 2004) instead of LP weights", finish(bo), f0)

# ------------------------------------------------------------------ R1: BG/NBD + Gamma-Gamma features
def btyd_inputs(df):
    x = np.maximum(df.total_transactions.values - 1, 0)          # repeat purchases
    T = df.account_age_days.values / 7.0                         # weeks
    rec = df.days_since_last_purchase.values / 7.0
    tx = np.where(x > 0, np.clip(T - rec, 0, T), 0.0)
    return x, tx, T


def bgnbd_terms(prm, x, tx, T):
    r, al, a, b = np.exp(prm)
    A = np.where(x > 0, (r + x) * (np.log(al + T) - np.log(al + tx)) + np.log(a) - np.log(np.maximum(b + x - 1, 1e-9)), -np.inf)
    ll = (gammaln(r + x) - gammaln(r) + r * np.log(al) + gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
          - (r + x) * np.log(al + T) + np.where(x > 0, np.logaddexp(0, A), 0.0))
    return ll, A, (r, al, a, b)


xa, txa, Ta = btyd_inputs(X)
ok = np.isfinite(xa) & np.isfinite(txa) & np.isfinite(Ta) & (Ta > 0)
res = minimize(lambda p_: -bgnbd_terms(p_, xa[ok], txa[ok], Ta[ok])[0].mean(), np.log([1.0, 10.0, 1.0, 5.0]), method="Nelder-Mead",
               options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-9})
log(f"BG/NBD fitted on {ok.sum():,} train rows: r, alpha, a, b = {np.round(np.exp(res.x), 4)}")


def gg_fit(df):
    x, m = df.total_transactions.values, df.avg_transaction_value.values
    k = np.isfinite(x) & np.isfinite(m) & (x >= 1) & (m > 0)
    x, m = x[k], m[k]
    f = lambda q_: -(gammaln(np.exp(q_[0]) * x + np.exp(q_[1])) - gammaln(np.exp(q_[0]) * x) - gammaln(np.exp(q_[1]))
                     + np.exp(q_[1]) * q_[2] + (np.exp(q_[0]) * x - 1) * np.log(m) + np.exp(q_[0]) * x * np.log(x)
                     - (np.exp(q_[0]) * x + np.exp(q_[1])) * np.log(x * m + np.exp(q_[2]))).mean()
    return np.exp(minimize(f, [0.0, 1.0, 3.0], method="Nelder-Mead", options={"maxiter": 4000}).x)


p_g, q_g, v_g = gg_fit(X)
log(f"Gamma-Gamma fitted: p, q, v = {p_g:.4f}, {q_g:.4f}, {v_g:.2f}")


def btyd_features(df):
    x, tx, T = btyd_inputs(df)
    _, A, (r, al, a, b) = bgnbd_terms(res.x, x, tx, T)
    p_alive = np.where(x > 0, np.exp(-np.logaddexp(0, A)), 1.0)
    e30 = p_alive * (r + x) / (al + T) * (30 / 7)
    xm, m = df.total_transactions.values, df.avg_transaction_value.values
    gg = np.where((xm >= 1) & (m > 0), p_g * (v_g + xm * m) / (p_g * xm + q_g - 1), np.nan)
    out = pd.DataFrame({"bt_p_alive": p_alive, "bt_exp_purch30": e30, "gg_exp_value": gg}, index=df.index)
    out["bt_clv30"] = out.bt_exp_purch30 * out.gg_exp_value
    out["bt_log_clv30"] = np.log1p(out.bt_clv30)
    bad = ~(np.isfinite(x) & np.isfinite(tx) & np.isfinite(T) & (T > 0))
    out.loc[bad, ["bt_p_alive", "bt_exp_purch30", "bt_clv30", "bt_log_clv30"]] = np.nan
    return out


Fb, Fbt = btyd_features(X), btyd_features(Xt)
print("BTYD feature summary (train):\n", Fb.describe().T[["mean", "50%", "min", "max"]].round(3).to_string(), flush=True)
print("Spearman with y>0:", Fb.corrwith(pd.Series(pos.astype(float), index=Fb.index), method="spearman").round(3).to_dict(), flush=True)
Xb, Xbt = pd.concat([X, Fb], axis=1), pd.concat([Xt, Fbt], axis=1)

half = lambda p_: {**p_, "learning_rate": p_["learning_rate"] / 2}
S1 = half(load_params("task3_stage1"))
R2 = {"objective": "regression_l1", **half(load_params("task3_ratio_stage2"))}
p1 = np.zeros(len(y))
for a, v in folds:
    _, (pv,), _ = fit_predict("lgb", "binary", S1, Xb.iloc[a], pos[a].astype(int), Xb.iloc[v], pos[v].astype(int), [Xb.iloc[v]])
    p1[v] = pv
log(f"R1 stage 1: AUC {roc_auc_score(pos, p1):.4f} vs pipeline {roc_auc_score(pos, h['p_oof']):.4f}; "
    f"log-loss {log_loss(pos, np.clip(p1, 1e-6, 1 - 1e-6)):.5f} vs {log_loss(pos, np.clip(h['p_oof'], 1e-6, 1 - 1e-6)):.5f}")
bh = np.log1p(X.historical_spending.values)
okb = np.isfinite(bh)
mu1 = h["mu_oof"].copy()
for a, v in folds:
    ap, vp, vb = a[pos[a] & okb[a]], v[pos[v] & okb[v]], v[okb[v]]
    _, (rv,), _ = fit_predict("lgb", "regression", R2, Xb.iloc[ap], z[ap] - bh[ap], Xb.iloc[vp], z[vp] - bh[vp], [Xb.iloc[vb]])
    mu1[vb] = bh[vb] + rv
log(f"R1 ratio stage 2 MAE on spenders {np.mean(np.abs(z[pos] - mu1[pos])):.4f} (pipeline ratio[hist] LGB tuned "
    f"{np.mean(np.abs(z[pos] - mu3['ratio[hist] LGB tuned'][0][pos])):.4f})")
ro1 = task3.readouts(y, p1, mu1)
new1 = {f"BTYD hurdle {k}": task3.apply_readout(k, p1, mu1, ro1) for k in ("mixture-median", "hard gate", "E[y] + smearing")}
for k, v in new1.items():
    log(f"  {k:36s} {nmae(y, v):.5f}")
fR1 = lp_nested(np.column_stack([P] + list(new1.values())))
report("R1", "+ hurdle with BG/NBD + Gamma-Gamma features (3 read-outs)", fR1, f0)

# ------------------------------------------------------------------ R2: ZILN network
dev = "cuda" if torch.cuda.is_available() else "cpu"


class ZILN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, 256), nn.SiLU(), nn.Dropout(0.1), nn.Linear(256, 128), nn.SiLU(), nn.Dropout(0.1))
        self.head = nn.Linear(128, 3)

    def forward(self, x, b):
        o = self.head(self.body(x))
        return o[:, 0], b + o[:, 1], F.softplus(o[:, 2]) + 1e-3


def ziln_loss(logit, mu, sig, pz, zz):
    bce = F.binary_cross_entropy_with_logits(logit, pz)
    m = pz > 0
    nll = ((zz[m] - mu[m]) / sig[m]) ** 2 / 2 + torch.log(sig[m])
    return bce + nll.sum() / len(pz)


def ziln_fold(Xa, ya, Xv, ba, bv, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(ya)
    perm = np.random.permutation(n)
    iv, itr = perm[: n // 10], perm[n // 10:]
    T_ = lambda arr: torch.tensor(arr, dtype=torch.float32, device=dev)
    Xa_t, ba_t, Xv_t, bv_t = T_(Xa), T_(ba), T_(Xv), T_(bv)
    pz, zz = T_((ya > 0).astype(float)), T_(np.log1p(ya))
    net = ZILN(Xa.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    best, best_state, bad = np.inf, None, 0
    for ep in range(80):
        net.train()
        idx = torch.tensor(np.random.permutation(itr), device=dev)
        for s in range(0, len(idx), 2048):
            j = idx[s:s + 2048]
            loss = ziln_loss(*net(Xa_t[j], ba_t[j]), pz[j], zz[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            jv = torch.tensor(iv, device=dev)
            vl = float(ziln_loss(*net(Xa_t[jv], ba_t[jv]), pz[jv], zz[jv]))
        if vl < best - 1e-5:
            best, best_state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 8:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        lo_, mu_, sg_ = net(Xv_t, bv_t)
    return torch.sigmoid(lo_).cpu().numpy(), mu_.cpu().numpy(), sg_.cpu().numpy()


cols = list(X.columns)
bfill = np.where(okb, bh, np.nanmedian(bh))
pz_, mz_, sz_ = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
for k, (a, v) in enumerate(folds):
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=200_000, random_state=0)
    A_ = qt.fit_transform(X.iloc[a][cols].values)
    V_ = qt.transform(X.iloc[v][cols].values)
    nan_a, nan_v = np.isnan(A_), np.isnan(V_)
    A_, V_ = np.nan_to_num(A_), np.nan_to_num(V_)
    A_, V_ = np.hstack([A_, nan_a.astype(float)]), np.hstack([V_, nan_v.astype(float)])
    acc = [ziln_fold(A_, y[a], V_, bfill[a], bfill[v], seed) for seed in (0, 1, 2)]
    pz_[v] = np.mean([q[0] for q in acc], 0)
    mz_[v] = np.mean([q[1] for q in acc], 0)
    sz_[v] = np.mean([q[2] for q in acc], 0)
    log(f"  ZILN fold {k} done")
log(f"R2 ZILN: AUC P(y>0) {roc_auc_score(pos, pz_):.4f}; MAE log1p on spenders {np.mean(np.abs(z[pos] - mz_[pos])):.4f}; "
    f"median sigma {np.median(sz_):.3f}")


def tune_vec(p_, mu_, sg_):
    f_ = lambda v_: -nmae(y, task3.mixture_median(np.clip(p_ + v_[1], 0, 1), mu_, abs(v_[0]) * sg_))
    r_ = minimize(f_, [1.0, 0.0], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-7})
    return abs(r_.x[0]), r_.x[1]


sc_, sh_ = tune_vec(pz_, mz_, sz_)
ziln_med = task3.mixture_median(np.clip(pz_ + sh_, 0, 1), mz_, sc_ * sz_)
ziln_mean = pz_ * np.clip(np.exp(mz_ + sz_ ** 2 / 2) - 1, 0, None)
log(f"  ZILN mixture median (per-row sigma x {sc_:.3f}, shift {sh_:+.3f}): {nmae(y, ziln_med):.5f}; ZILN mean (as in the paper): {nmae(y, ziln_mean):.5f}")
fR2 = lp_nested(np.column_stack([P, ziln_med, ziln_mean]))
report("R2", "+ ZILN network (median and mean read-outs)", fR2, f0)
fR12 = lp_nested(np.column_stack([P] + list(new1.values()) + [ziln_med, ziln_mean]))
report("R1+R2", "+ BTYD hurdle and ZILN members", fR12, f0)
np.savez(OUT / "t3_round9_members.npz", btyd=np.column_stack(list(new1.values())), ziln=np.column_stack([ziln_med, ziln_mean]),
         p1=p1, mu1=mu1, pz=pz_, mz=mz_, sz=sz_, f0=f0)

R = pd.DataFrame(rows, columns=["Experiment", "Change", "OOF score", "reference", "delta", "95% CI", "Decision"])
print("\n" + R.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
log("done")
