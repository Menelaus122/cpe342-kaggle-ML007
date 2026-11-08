"""Task 3 round 8 (2026-11-08): three hypotheses around the zero decision and the metric, strict 5-fold OOF.

Baseline B0 = v8 (26-member pool, nested L1 blend, zero gate on P(y>0), one global scale) = 0.79403.
H1 post-processing : one gate threshold and one scale per segment, chosen on the other 4 folds (nested).
H2 loss alignment  : a zero gate learned directly from the metric. For every row the gain of predicting the blend u
                     instead of 0 is g = y - |y - u|. A LightGBM classifier on (features, P(y>0), blend) learns
                     sign(g) with weight |g|, in-fold (trained on 3 folds, early-stopped on a 4th, applied to the 5th).
H3 loss alignment / diversity: the ratio models are fitted in log space, where every row counts the same, but the
                     metric counts THB, where a whale counts 1000x a small spender. Variants:
                       H3a stage 2 weighted by the spending scale (1 + historical_spending, and its square root);
                       H3b a one-stage L1 model of log1p(y) - log1p(hist) on ALL rows (its median gives the zeros by
                           itself), unweighted and weighted.
                     New members go into the same pool and the same nested blend.
Significance: paired Poisson bootstrap over rows (1,000 resamples) of the score difference; keep only if the 95%
interval is above 0. No test rows are used for fitting anything.
"""
import json
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import common  # noqa: E402
from common import OUT  # noqa: E402
from src import task3  # noqa: E402
from src.metrics import nmae  # noqa: E402
from src.modeling import make_folds, LGB_BASE  # noqa: E402
from t3_fill import fill  # noqa: E402
from t3_segment import impute_segment  # noqa: E402

T0 = time.time()
rng = np.random.default_rng(0)
log = lambda *a: print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)

# ------------------------------------------------------------------ data exactly as v8
tr, te = task3.load()
y = tr[task3.TARGET].values.astype(float)
z, pos = np.log1p(y), y > 0
folds = make_folds(y, kind="regression")
trf, tef = fill(tr)[0], fill(te)[0]
X, Xt = task3.add_features(trf), task3.add_features(tef)
X, Xt, acc = impute_segment(X, Xt)
segv = X.segment.values.astype(int)
log(f"features ready, segment imputer accuracy {acc:.4f}")

v8 = np.load(OUT / "t3_v8.npz")
P, Pt, names, gate, gate_t = v8["P_oof"], v8["P_test"], list(v8["names"]), v8["gate_oof"], v8["gate_test"]
qm = np.load(OUT / "t3_quantile_members.npz")
p, p_t = qm["p_oof"], qm["p_test"]
blend0 = np.load(OUT / "t3_v8_blend_oof.npz")["blend"]          # written by at_t3_zeros_check.py
scales = np.linspace(0.85, 1.15, 61)
thr = (p[gate == 1].min() + p[gate == 0].max()) / 2


def best_scale(pred_unscaled, idx=None):
    idx = np.arange(len(y)) if idx is None else idx
    E = np.abs(y[idx][None, :] - scales[:, None] * pred_unscaled[idx][None, :]).sum(1)
    return scales[int(E.argmin())]


k0 = best_scale(blend0 * gate)
f0 = blend0 * k0 * gate
S0 = nmae(y, f0)
log(f"B0 (v8) OOF {S0:.5f}; gate threshold {thr:.4f}; scale {k0:.3f}")


def boot(pred_new, pred_old, B=1000):
    e1, e0 = np.abs(y - pred_new), np.abs(y - pred_old)
    d = np.empty(B)
    for i in range(B):
        w = rng.poisson(1.0, len(y)).astype(float)
        sy = w @ y
        d[i] = 1 / (1 + (w @ e1) / sy) - 1 / (1 + (w @ e0) / sy)
    return np.percentile(d, [2.5, 97.5])


rows = []


def report(name, desc, pred, ref=None, ref_name="B0 (v8)"):
    ref = f0 if ref is None else ref
    s, s_ref = nmae(y, pred), nmae(y, ref)
    lo, hi = boot(pred, ref)
    rows.append([name, desc, s, ref_name, s - s_ref, f"[{lo:+.5f}, {hi:+.5f}]", "KEEP" if lo > 0 else "discard"])
    log(f"{name}: OOF {s:.5f} (delta vs {ref_name} {s - s_ref:+.5f}, 95% CI [{lo:+.5f}, {hi:+.5f}]); zeros {np.mean(pred == 0):.3f}")


# ------------------------------------------------------------------ H1: per-segment threshold / scale, nested
T_GRID = np.round(np.arange(0.30, 0.7001, 0.01), 3)


def fit_tk(idx, fix_t=None, fix_k=None):
    best, arg = np.inf, None
    for t in ([fix_t] if fix_t is not None else T_GRID):
        g = blend0[idx] * (p[idx] > t)
        ks = scales if fix_k is None else np.array([fix_k])
        E = np.abs(y[idx][None, :] - ks[:, None] * g[None, :]).sum(1)
        j = int(E.argmin())
        if E[j] < best:
            best, arg = E[j], (t, ks[j])
    return arg


def h1(per_segment, fix_t=False, fix_k=False):
    pred = np.zeros(len(y))
    for a, b in folds:
        groups = [0, 1, 2, 3] if per_segment else [None]
        kg = fit_tk(a, fix_t=thr)[1] if fix_k else None       # global scale for the "threshold only" variant
        for g_ in groups:
            aa = a if g_ is None else a[segv[a] == g_]
            bb = b if g_ is None else b[segv[b] == g_]
            t, k = fit_tk(aa, fix_t=thr if fix_t else None, fix_k=kg)
            pred[bb] = blend0[bb] * k * (p[bb] > t)
    return pred


f_nested = h1(False)
log(f"B0 with threshold + scale chosen on the other folds (nested reference): {nmae(y, f_nested):.5f}")
report("H1a", "one gate threshold + scale per segment (nested)", h1(True), ref=f_nested, ref_name="B0 nested")
report("H1b", "one scale per segment, v8 threshold (nested)", h1(True, fix_t=True), ref=f_nested, ref_name="B0 nested")
report("H1c", "one threshold per segment, global scale (nested)", h1(True, fix_k=True), ref=f_nested, ref_name="B0 nested")

# ------------------------------------------------------------------ H2: loss-aligned zero gate, in-fold
u = blend0 * k0
g_gain = y - np.abs(y - u)                    # > 0: predicting u beats predicting 0
lab, wt = (g_gain > 0).astype(int), np.abs(g_gain)
F = X.copy()
F["p_pos"], F["log_u"] = p, np.log1p(u)
GP = {**LGB_BASE, "objective": "binary", "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 200,
      "colsample_bytree": 0.8}
q = np.zeros(len(y))
for k, (a, b) in enumerate(folds):
    va = folds[(k + 1) % 5][1]
    trn = np.setdiff1d(a, va)
    trn, va = trn[wt[trn] > 0], va[wt[va] > 0]
    m = lgb.LGBMClassifier(**GP).fit(F.iloc[trn], lab[trn], sample_weight=wt[trn], eval_set=[(F.iloc[va], lab[va])],
                                     eval_sample_weight=[wt[va]], callbacks=[lgb.early_stopping(200, verbose=False)])
    q[b] = m.predict_proba(F.iloc[b])[:, 1]
log(f"H2 gate classifier trained (in-fold); agreement with v8 gate {np.mean((q > 0.5) == (gate == 1)):.4f}")
report("H2a", "metric-trained zero gate (predict 0 where it says 0 beats the blend)", u * (q > 0.5))
pred = np.zeros(len(y))
for a, b in folds:
    t = T_GRID[int(np.argmin([np.abs(y[a] - u[a] * (q[a] > t)).sum() for t in T_GRID]))]
    pred[b] = u[b] * (q[b] > t)
report("H2b", "H2a with its cut-off chosen on the other folds", pred)
report("H2c", "zero only where both gates say 0 (v8 gate OR metric gate keeps the blend)",
       u * np.maximum(gate, (q > 0.5)))

# ------------------------------------------------------------------ H3: metric-weighted ratio models
TP = json.loads((OUT / "t3_tuned_params.json").read_text())
R2 = {**LGB_BASE, "objective": "regression_l1", **TP["t3_ratio_stage2"]}
R2["learning_rate"] /= 2
h = trf.historical_spending.values
b_log = np.log1p(h)
okb = np.isfinite(b_log)
bt_log = np.log1p(tef.historical_spending.values)
okbt = np.isfinite(bt_log)
log(f"rows without a usable historical_spending after the identity fill: train {int((~okb).sum())}, test {int((~okbt).sum())}")
ih = np.load(OUT / "t3_identity_hurdle.npz")                 # plain hurdle mu, fallback for those rows only
W_SCALE = {"1": np.ones(len(y)), "hist": 1 + np.nan_to_num(h), "sqrt_hist": np.sqrt(1 + np.nan_to_num(h))}


def lgb_w(Xa, ya, wa, Xv, yv, wv, Xp):
    m = lgb.LGBMRegressor(**R2).fit(Xa, ya, sample_weight=wa, eval_set=[(Xv, yv)], eval_sample_weight=[wv],
                                    callbacks=[lgb.early_stopping(200, verbose=False)])
    return [m.predict(x) for x in Xp]


def stage2_ratio(wname):
    w = W_SCALE[wname]
    mu, mu_t = ih["mu_oof"].copy(), np.zeros(len(Xt))
    for a, v in folds:
        ap, vp, vb = a[pos[a] & okb[a]], v[pos[v] & okb[v]], v[okb[v]]
        rv, rt = lgb_w(X.iloc[ap], z[ap] - b_log[ap], w[ap], X.iloc[vp], z[vp] - b_log[vp], w[vp], [X.iloc[vb], Xt])
        mu[vb] = b_log[vb] + rv
        mu_t += rt / len(folds)
    mu_test = ih["mu_test"].copy()
    mu_test[okbt] = bt_log[okbt] + mu_t[okbt]
    log(f"  stage 2 (w = {wname}) MAE on positives {np.mean(np.abs(z[pos] - mu[pos])):.4f}; "
        f"weighted by hist {np.average(np.abs(z[pos] - mu[pos]), weights=W_SCALE['hist'][pos]):.4f}")
    resid = z[pos] - mu[pos]
    sigma0 = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    smear = float(np.mean(np.exp(resid)))
    tm = task3.tune_mixture(y, p, mu, sigma0)
    th = task3.tune_hard_hurdle(y, p, mu)
    sh = lambda pp: np.clip(pp + tm["shift"], 0, 1)
    out = {f"ratio[hist] w={wname} mixture-median": (task3.mixture_median(sh(p), mu, tm["sigma"]),
                                                     task3.mixture_median(sh(p_t), mu_test, tm["sigma"])),
           f"ratio[hist] w={wname} hard gate": (task3.hard_hurdle(p, mu, th["tau"]), task3.hard_hurdle(p_t, mu_test, th["tau"])),
           f"ratio[hist] w={wname} E[y] + smearing": (task3.expected_value(p, mu, smear), task3.expected_value(p_t, mu_test, smear))}
    for k_, (o, _) in out.items():
        log(f"  {k_:44s} OOF {nmae(y, o):.5f}")
    return out


def one_stage(wname):
    w = W_SCALE[wname]
    fb_o, fb_t = P[:, names.index("LGB-L1 log1p")], Pt[:, names.index("LGB-L1 log1p")]
    o, t_ = fb_o.copy(), np.zeros(len(Xt))
    for a, v in folds:
        aa, vv = a[okb[a]], v[okb[v]]
        rv, rt = lgb_w(X.iloc[aa], z[aa] - b_log[aa], w[aa], X.iloc[vv], z[vv] - b_log[vv], w[vv], [X.iloc[vv], Xt])
        o[vv] = np.clip(np.expm1(b_log[vv] + rv), 0, None)
        t_ += rt / len(folds)
    tt = fb_t.copy()
    tt[okbt] = np.clip(np.expm1(bt_log[okbt] + t_[okbt]), 0, None)
    k_ = f"one-stage ratio L1 w={wname}"
    log(f"  {k_:44s} OOF {nmae(y, o):.5f}; zeros {np.mean(o < 0.5):.3f}")
    return {k_: (o, tt)}


new = {}
for wn in ("hist", "sqrt_hist"):
    new.update(stage2_ratio(wn))
for wn in ("1", "hist"):
    new.update(one_stage(wn))
np.savez(OUT / "t3_round8_members.npz", names=np.array(list(new)), oof=np.column_stack([v[0] for v in new.values()]),
         test=np.column_stack([v[1] for v in new.values()]), q_gate_oof=q)


def pool_blend(extra):
    Pe = np.column_stack([P] + [new[k_][0] for k_ in extra])
    bo = np.zeros(len(y))
    for a, b in folds:
        w = task3.l1_blend_weights(y[a], [Pe[a, j] for j in range(Pe.shape[1])])
        bo[b] = Pe[b] @ w
    k = best_scale(bo * gate)
    return bo * k * gate


for tag, extra in [("H3a", [k_ for k_ in new if k_.startswith("ratio[hist] w=hist ")]),
                   ("H3a-sqrt", [k_ for k_ in new if "w=sqrt_hist" in k_]),
                   ("H3b", [k_ for k_ in new if k_.startswith("one-stage")]),
                   ("H3-all", list(new))]:
    desc = {"H3a": "+ stage 2 weighted by 1 + hist (3 read-outs)", "H3a-sqrt": "+ stage 2 weighted by sqrt(1 + hist)",
            "H3b": "+ one-stage ratio L1 on all rows (unweighted, weighted)", "H3-all": "+ all 8 new members"}[tag]
    pred = pool_blend(extra)
    np.save(OUT / f"t3_round8_blend_{tag}.npy", pred)
    report(tag, desc, pred)

R = pd.DataFrame(rows, columns=["Experiment", "Change", "OOF score", "reference", "delta", "95% CI of delta", "Decision"])
print("\n" + R.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
R.to_csv(OUT / "t3_round8_results.csv", index=False)
log("done")
