"""Round 13, item 4: Task 3 scale / zero-cutoff calibration and the whale audit.

B0 = the pipeline: 26-member pool, exact L1 blend (weights fitted on the other four folds), median zero gate, one global scale chosen on the OOF.
Variants, every choice made on the training folds only and applied to the held-out fold:
  A1  nested scale alpha in [0.90, 1.15]
  A2  nested alpha and a zero cut-off tau (predictions below tau become 0)
  A3  sum-matching alpha = sum(y) / sum(prediction)
  W1  add a raw-scale LightGBM Huber member to the pool (delta = 90th percentile of the absolute deviation from the training median)
  W2  raw-scale L1 LightGBM trained only on top spenders (historical_spending in the training folds' top 10%), blended on those rows, weight nested
Acceptance: CI > 0, >= 4/5 folds, fold std not above 1.1x.
"""
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import common  # noqa: E402,F401
from common import P3_lgb  # noqa: E402
from src import task3  # noqa: E402
from src.metrics import nmae  # noqa: E402
from src.modeling import make_folds, fit_predict  # noqa: E402
from src.utils import load_preds  # noqa: E402

T0 = time.time()
log = lambda *a: print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
rng = np.random.default_rng(0)
tr, te = task3.load()
X, y, Xt, _ = task3.build_repaired(tr, te)
folds = make_folds(y, kind="regression")
clip0 = lambda v: np.clip(v, 0, None)
slug3 = lambda tag: "t3r_" + tag.replace("[", "_").replace("] ", "_").replace(" ", "_").lower()
SPEC = ["ratio[hist] LGB tuned", "ratio[prev] LGB tuned", "ratio[hist] LGB", "ratio[prev] LGB", "ratio[hist] XGB", "ratio[hist] Cat"]
plain = {}
for name, key, logt in [("LGB-L1 raw THB", "t3r_lgb_raw", False), ("XGB-L1 raw THB", "t3r_xgb_raw", False), ("Cat-MAE raw THB", "t3r_cat_raw", False),
                        ("LGB-L1 log1p", "t3r_lgb_log", True), ("LGB-Tweedie (mean)", "t3r_tweedie", False)]:
    r = load_preds(key)
    plain[name] = clip0(np.expm1(r["oof"])) if logt else clip0(r["oof"])
h = load_preds("t3r_hurdle")
p = h["p_oof"]
mus = {"Hurdle": h["mu_oof"], **{t: load_preds(slug3(t))["mu_oof"] for t in SPEC}}
m = dict(plain)
ro = {}
for tag, mu_ in mus.items():
    ro[tag] = task3.readouts(y, p, mu_)
    for k in ("mixture-median", "hard gate", "E[y] + smearing"):
        m[f"{tag} {k}"] = task3.apply_readout(k, p, mu_, ro[tag])
P = np.column_stack(list(m.values()))
gate = (np.clip(p + ro["Hurdle"]["shift"], 0, 1) > 0.5).astype(float)
log(f"pool {P.shape[1]} members, zero gate keeps {gate.mean():.3f} of rows")


def fold_blend(P_):
    """Nested weights; returns the OOF blend and the blend restricted to each fold's own training rows."""
    bo, inner = np.zeros(len(y)), {}
    for k, (a, b) in enumerate(folds):
        w = task3.l1_blend_weights_exact(y[a], [P_[a, j] for j in range(P_.shape[1])])
        bo[b] = P_[b] @ w
        inner[k] = P_[a] @ w
    return bo, inner


bo, inner = fold_blend(P)
log("nested blend done")
SC = np.linspace(0.85, 1.15, 61)
alpha_full = SC[int(np.argmax([nmae(y, bo * s * gate) for s in SC]))]
pred0 = bo * alpha_full * gate
s0 = nmae(y, pred0)
pf = lambda v: np.array([nmae(y[b], v[b]) for _, b in folds])
g0 = pf(pred0)
log(f"B0 pipeline: NMAE {s0:.5f}, scale {alpha_full:.3f}, folds {np.round(g0, 4)}, std {g0.std():.5f}")
print(f"before scale and gate: sum(pred)/sum(y) = {bo.sum() / y.sum():.4f}; final sum(pred)/sum(y) = {pred0.sum() / y.sum():.4f}")


def boot(pn, po, B=300):
    d = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        A, T = np.abs(y[i] - pn[i]).sum(), y[i].sum()
        Ao = np.abs(y[i] - po[i]).sum()
        d.append(1 / (1 + A / T) - 1 / (1 + Ao / T))
    return np.percentile(d, [2.5, 97.5])


rows = []


def judge(name, v):
    s, g = nmae(y, v), pf(v)
    lo, hi = boot(v, pred0)
    better = int((g > g0).sum())
    keep = lo > 0 and better >= 4 and g.std() <= g0.std() * 1.1
    rows.append([name, s, s - s0, f"[{lo:+.5f}, {hi:+.5f}]", f"{better}/5", g.std(), "KEEP" if keep else "discard"])
    log(f"{name}: {s:.5f} ({s - s0:+.5f}) CI {rows[-1][3]}, {better}/5, std {g.std():.5f} -> {rows[-1][-1]}")


ALPHAS = np.linspace(0.90, 1.15, 51)
TAUS = np.array([0, 25, 50, 100, 150, 200, 300, 400, 600])

v1, v2, v3 = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
chosen = []
for k, (a, b) in enumerate(folds):
    sa = inner[k] * gate[a]
    sb = bo[b] * gate[b]
    al = ALPHAS[int(np.argmax([nmae(y[a], sa * s) for s in ALPHAS]))]
    v1[b] = sb * al
    best, ba, bt = -1, 1.0, 0
    for s in ALPHAS[::2]:
        for t in TAUS:
            q = sa * s
            q = np.where(q < t, 0, q)
            c = nmae(y[a], q)
            if c > best:
                best, ba, bt = c, s, t
    q = sb * ba
    v2[b] = np.where(q < bt, 0, q)
    am = y[a].sum() / sa.sum()
    v3[b] = sb * am
    chosen.append([round(float(al), 3), round(float(ba), 3), int(bt), round(float(am), 3)])
print("chosen per fold [A1 alpha, A2 alpha, A2 tau, A3 sum-matching alpha]:", chosen, flush=True)
judge("A1 nested scale", v1)
judge("A2 nested scale + zero cut-off", v2)
judge("A3 sum-matching scale", v3)

print("\n=== whale audit (B0 out-of-fold) ===")
err = np.abs(y - pred0)
tot = err.sum()
for q in (0.001, 0.01, 0.05):
    t = np.quantile(err, 1 - q)
    mk = err >= t
    print(f"top {q:.1%} largest absolute errors: {mk.sum()} rows, {err[mk].sum() / tot:.1%} of total error; "
          f"of that error, under-prediction {err[mk & (y > pred0)].sum() / err[mk].sum():.1%}, actual zero (all over-prediction) {err[mk & (y == 0)].sum() / err[mk].sum():.1%}")
mk = err >= np.quantile(err, 0.99)
ht = X["historical_spending"].values if "historical_spending" in X else np.nan
print(f"prediction range: max pred {pred0.max():,.0f}, max y {y.max():,.0f}; p99.9 pred {np.quantile(pred0, .999):,.0f}, p99.9 y {np.quantile(y, .999):,.0f}; rows predicted at the clip 0: {(pred0 == 0).mean():.3%}")
top = y >= np.quantile(y, 0.99)
print(f"top 1% spenders by actual y ({top.sum()} rows, {y[top].sum() / y.sum():.1%} of all spending): sum(pred)/sum(y) {pred0[top].sum() / y[top].sum():.3f}, median pred/y {np.median(pred0[top] / y[top]):.3f}, "
      f"share of error {err[top].sum() / tot:.1%}")
print("error of those rows is mostly a conditional-median effect: they are top spenders only after seeing y, so the same comparison on the top 1% of PREDICTIONS is the fair calibration test:")
tp = pred0 >= np.quantile(pred0, 0.99)
print(f"  top 1% by prediction ({tp.sum()} rows): sum(y)/sum(pred) {y[tp].sum() / pred0[tp].sum():.3f}, median y/pred {np.median(y[tp] / np.clip(pred0[tp], 1, None)):.3f}, share y = 0: {(y[tp] == 0).mean():.3f}")
cal = pd.DataFrame({"y": y, "p": pred0})
cal = cal[cal.p > 0]
cal["bin"] = pd.qcut(cal.p, 10, labels=False, duplicates="drop")
g = cal.groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"), median_y=("y", "median"), mean_y=("y", "mean"), share_y0=("y", lambda v: (v == 0).mean()))
g["share of y above pred"] = cal.assign(ab=cal.y > cal.p).groupby("bin").ab.mean()
print("\ncalibration by decile of positive predictions (a calibrated median has about 50% of y above pred)\n", g.round(3).to_string())

log("training raw-scale Huber member")
hub_o, wh_o = np.zeros(len(y)), np.full(len(y), np.nan)
for k, (a, b) in enumerate(folds):
    med = np.median(y[a])
    delta = float(np.quantile(np.abs(y[a] - med), 0.9))
    _, (pv,), _ = fit_predict("lgb", "regression", {**P3_lgb, "objective": "huber", "alpha": delta, "metric": "l1"}, X.iloc[a], y[a], X.iloc[b], y[b], [X.iloc[b]])
    hub_o[b] = np.clip(pv, 0, None)
    hist = X["historical_spending"].values
    cut = np.nanquantile(hist[a], 0.90)
    aw, bw = a[hist[a] >= cut], b[hist[b] >= cut]
    _, (pw,), _ = fit_predict("lgb", "regression", P3_lgb, X.iloc[aw], y[aw], X.iloc[bw], y[bw], [X.iloc[bw]])
    wh_o[bw] = np.clip(pw, 0, None)
    log(f"  fold {k}: Huber delta {delta:,.0f}; whale rows train {len(aw)}, val {len(bw)}")
print(f"Huber member alone: NMAE {nmae(y, hub_o):.5f}; cached LGB-L1 raw alone: {nmae(y, plain['LGB-L1 raw THB']):.5f}", flush=True)

P1 = np.column_stack([P, hub_o])
bo1, _ = fold_blend(P1)
judge("W1 + raw-scale Huber member in the blend", bo1 * alpha_full * gate)

has = ~np.isnan(wh_o)
LAM = [0, .25, .5, .75, 1]
vw = pred0.copy()
ch = []
for k, (a, b) in enumerate(folds):
    aw = a[has[a]]
    bw = b[has[b]]
    sc = [nmae(y[aw], ((1 - l) * bo[aw] + l * wh_o[aw]) * alpha_full * gate[aw]) for l in LAM]
    l = LAM[int(np.argmax(sc))]
    ch.append(l)
    vw[bw] = ((1 - l) * bo[bw] + l * wh_o[bw]) * alpha_full * gate[bw]
print("whale-model weights per fold:", ch, "(note: chosen on rows whose whale-model predictions are out-of-fold)", flush=True)
judge("W2 top-spender raw L1 model, nested weight", vw)

pd.set_option("display.width", 250)
print("\n" + pd.DataFrame(rows, columns=["Variant", "NMAE", "delta", "95% CI", "folds better", "fold std", "Decision"]).to_string(index=False))
log("done")
