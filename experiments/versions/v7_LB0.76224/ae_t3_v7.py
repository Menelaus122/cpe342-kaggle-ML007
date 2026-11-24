"""Task 3 candidate v7: identity fill + imputed segment + ratio-target hurdles, full notebook recipe.

Features: t3_fill.fill (exact row identities) -> task3.add_features -> t3_segment.impute_segment (if IMPUTE).
Pool (5-fold OOF + test, same folds and parameters as the notebook):
  LGB-L1 raw, XGB-L1 raw, Cat-MAE, LGB-L1 log1p, LGB-Tweedie, hurdle (log1p target) x 3 read-outs,
  NEW: ratio hurdles, stage 2 on log1p(y) - log1p(base) for base = historical_spending and prev_month_spending,
  x 3 read-outs each (mixture-median, hard gate, E[y] + smearing).
Blend: OOF L1 weights, median zero gate, scale (notebook section 7). Final test input: every member also refitted
on 100% of the labelled rows and averaged 50/50 with the 5-fold models (notebook section 7.6).
Writes submission_v7_try.csv = versions/v6_LB0.76202 with only Task 3 replaced.
Usage: python ae_t3_v7.py [--no-impute]
"""
import hashlib
import sys
import warnings

import numpy as np
import optuna
import pandas as pd

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
import common  # noqa: E402
from common import P3_lgb, OUT  # noqa: E402
from src import task3  # noqa: E402
from src import config as C  # noqa: E402
from src.metrics import nmae  # noqa: E402
from src.modeling import make_folds, run_cv, fit_predict, refit_full  # noqa: E402
from t3_fill import fill  # noqa: E402
from t3_segment import impute_segment  # noqa: E402

IMPUTE = "--no-impute" not in sys.argv
FULL_W = 0.5
blend_full = lambda a, b: (1 - FULL_W) * np.asarray(a) + FULL_W * np.asarray(b)


def tuned_study(name, fallback):
    p = None
    try:
        st = optuna.load_study(study_name=name, storage=f"sqlite:///{(C.OUT / 'optuna.db').as_posix()}")
        if len(st.trials):
            p = st.best_params
    except KeyError:
        pass
    p = dict(fallback) if p is None else dict(p)
    if "learning_rate" in p:
        p["learning_rate"] /= 2
    return p


tr, te = task3.load()
y = tr[task3.TARGET].values.astype(float)
z, pos = np.log1p(y), y > 0
folds = make_folds(y, kind="regression")
trf, tef = fill(tr)[0], fill(te)[0]
X, Xt = task3.add_features(trf), task3.add_features(tef)
if IMPUTE:
    X, Xt, acc = impute_segment(X, Xt)
    print(f"segment imputed: hold-out accuracy {acc:.4f}; {int(X.segment_imputed.sum())} train / "
          f"{int(Xt.segment_imputed.sum())} test rows", flush=True)

P3_xgb = tuned_study("task3_xgb", {"learning_rate": 0.05, "max_depth": 7})
TW = {"learning_rate": 0.05, "objective": "tweedie", "tweedie_variance_power": 1.5, "metric": "l1"}
specs = {"LGB-L1 raw THB": ("lgb", P3_lgb, False), "XGB-L1 raw THB": ("xgb", P3_xgb, False),
         "Cat-MAE raw THB": ("cat", {"learning_rate": 0.08, "depth": 8}, False),
         "LGB-L1 log1p": ("lgb", P3_lgb, True), "LGB-Tweedie (mean)": ("lgb", TW, False)}
oof, fold_test, full_test = {}, {}, {}
for k, (kind, prm, log) in specs.items():
    tgt = z if log else y
    r = run_cv(kind, "regression", prm, X, tgt, Xt, folds, verbose=False)
    f = refit_full(kind, "regression", prm, X, tgt, Xt, folds, name=k)["test"]
    back = (lambda v: np.clip(np.expm1(v), 0, None)) if log else (lambda v: np.clip(v, 0, None))
    oof[k], fold_test[k], full_test[k] = back(r["oof"]), back(r["test"]), back(f)
    print(f"  {k:22s} OOF {nmae(y, oof[k]):.5f}", flush=True)

# ---- hurdles: stage 1 shared, stage 2 on log1p(y) (as before) and on the ratio to a base column
h = task3.hurdle_cv("lgb", X, y, Xt, folds, clf_params={"learning_rate": 0.05}, reg_params={"learning_rate": 0.05},
                    verbose=False)
hf = task3.hurdle_full("lgb", X, y, Xt, folds, {"learning_rate": 0.05}, {"learning_rate": 0.05})
p_oof, p_test = h["p_oof"], blend_full(h["p_test"], hf["p_test"])
mus = {"hurdle": (h["mu_oof"], blend_full(h["mu_test"], hf["mu_test"]))}
REG = {"objective": "regression_l1", "learning_rate": 0.05}
for bname in ("historical_spending", "prev_month_spending"):
    b, bt = np.log1p(trf[bname].values), np.log1p(tef[bname].values)
    okb, okbt = np.isfinite(b), np.isfinite(bt)
    mu, acc_t = h["mu_oof"].copy(), np.zeros(len(Xt))
    for a, v in folds:
        ap, vp, vb = a[pos[a] & okb[a]], v[pos[v] & okb[v]], v[okb[v]]
        _, (rv, rt), _ = fit_predict("lgb", "regression", REG, X.iloc[ap], z[ap] - b[ap], X.iloc[vp], z[vp] - b[vp],
                                     [X.iloc[vb], Xt])
        mu[vb] = b[vb] + rv
        acc_t += rt / len(folds)
    # full refit on all positive rows with a known base; rounds from fold 0 (as task3.hurdle_full)
    idx = np.where(pos & okb)[0]
    remap = {o: n for n, o in enumerate(idx)}
    f0 = [(np.array([remap[i] for i in folds[0][0] if i in remap]), np.array([remap[i] for i in folds[0][1] if i in remap]))]
    full_r = refit_full("lgb", "regression", REG, X.iloc[idx], z[idx] - b[idx], Xt, f0, name=f"ratio[{bname}]")["test"]
    mu_t = mus["hurdle"][1].copy()
    mu_t[okbt] = bt[okbt] + blend_full(acc_t, full_r)[okbt]
    mus[f"ratio[{bname[:4]}]"] = (mu, mu_t)

gate_shift = None
for tag, (mu, mu_t) in mus.items():
    resid = z[pos] - mu[pos]
    sigma0 = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    smear = float(np.mean(np.exp(resid)))
    tm = task3.tune_mixture(y, p_oof, mu, sigma0)
    th = task3.tune_hard_hurdle(y, p_oof, mu)
    sh = lambda p: np.clip(p + tm["shift"], 0, 1)
    name = "Hurdle" if tag == "hurdle" else tag
    for k, o, t in [(f"{name} mixture-median", task3.mixture_median(sh(p_oof), mu, tm["sigma"]),
                     task3.mixture_median(sh(p_test), mu_t, tm["sigma"])),
                    (f"{name} hard gate", task3.hard_hurdle(p_oof, mu, th["tau"]), task3.hard_hurdle(p_test, mu_t, th["tau"])),
                    (f"{name} E[y] + smearing", task3.expected_value(p_oof, mu, smear), task3.expected_value(p_test, mu_t, smear))]:
        oof[k], fold_test[k], full_test[k] = o, t, t          # hurdle test inputs are already 5-fold/full blended
        print(f"  {k:32s} OOF {nmae(y, o):.5f}", flush=True)
    if tag == "hurdle":
        gate_shift = tm["shift"]                               # the notebook's zero gate comes from the plain hurdle

names = list(oof)
P = np.column_stack([oof[k] for k in names])
Pt = np.column_stack([blend_full(fold_test[k], full_test[k]) for k in names])
gate_oof = (np.clip(p_oof + gate_shift, 0, 1) > 0.5).astype(float)
gate_test = (np.clip(p_test + gate_shift, 0, 1) > 0.5).astype(float)


def blend_score(cols):
    j = [names.index(c) for c in cols]
    bo = np.zeros(len(y))
    for a, v in folds:
        w = task3.l1_blend_weights(y[a], [P[a, i] for i in j])
        bo[v] = P[v][:, j] @ w
    W = task3.l1_blend_weights(y, [P[:, i] for i in j])
    scales = np.linspace(0.85, 1.15, 61)
    s = float(scales[int(np.argmax([nmae(y, bo * sc * gate_oof) for sc in scales]))])
    return nmae(y, bo * s * gate_oof), W, s, j


old = [k for k in names if not k.startswith("ratio")]
s_old, *_ = blend_score(old)
s_new, W, SCALE, j = blend_score(names)
print(f"\nblend OOF: v6-style pool on these features {s_old:.5f} -> + ratio hurdles {s_new:.5f} "
      f"(v6 was 0.79097, v3 0.78917)")
print("weights:", {names[i]: round(float(w), 3) for i, w in zip(j, W) if w > 1e-3})
pred = np.clip(Pt[:, j] @ W * SCALE, 0, None) * gate_test
np.savez(OUT / f"t3_v7{'' if IMPUTE else '_noimpute'}.npz", P_oof=P, P_test=Pt, names=np.array(names),
         gate_oof=gate_oof, gate_test=gate_test, pred_test=pred, score=s_new)

base = pd.read_csv(C.ROOT / "versions/v6_LB0.76202/submission.csv")
sub = base.copy()
sub["task3"] = np.round(pred, 2)
assert sub.shape == (C.N_TEST, 6) and sub.isna().sum().sum() == 0
assert all(str(sub[c].dtype).startswith("int") for c in ["task1", "task2", "task4", "task5"])
assert (sub.task3 >= 0).all() and np.isfinite(sub.task3).all()
for c in ["id", "task1", "task2", "task4", "task5"]:
    assert sub[c].equals(base[c]), c
o = base.task3.values
print(f"Task 3 vs v6: corr(log1p) {np.corrcoef(np.log1p(o), np.log1p(sub.task3))[0, 1]:.4f}; zero share "
      f"{np.mean(o == 0):.3f} -> {np.mean(sub.task3 == 0):.3f}; mean {o.mean():.0f} -> {sub.task3.mean():.0f}")
out = C.ROOT / ("submission_v7_try.csv" if IMPUTE else "submission_v7_noimpute.csv")
sub.to_csv(out, index=False)
print(f"saved {out} | all checks passed | sha256 {hashlib.sha256(out.read_bytes()).hexdigest()[:16]}")
