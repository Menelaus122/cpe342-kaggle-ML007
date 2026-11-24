"""Task 3 candidate (round 2): identity-filled features + whale-only model, built the way the notebook builds
its final Task 3 prediction, then dropped into a copy of the user's 0.76 file (versions/v5_LB0.76).

Final test prediction = notebook recipe on the filled features:
  * 8-member pool (x_t3_identity.py + x2_t3_identity_all.py): OOF L1 blend weights, zero gate, scale (all out-of-fold);
  * every GBDT member and both hurdle stages also refitted on 100% of the labelled rows (Train+Val, rounds from
    fold 0 x 1.15) and averaged 50/50 with the 5-fold models, exactly like notebook section 7.6;
  * whale-only model (segment 3): 5-fold models + a full-data refit, mixed 50/50 with the blend on whale rows.
Nothing here uses the test labels or player_id; only Task 3 changes relative to the 0.76 file.
"""
import hashlib
import warnings

import numpy as np
import pandas as pd
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
import common  # noqa: E402
from common import P3_lgb, OUT  # noqa: E402
from src import task3  # noqa: E402
from src import config as C  # noqa: E402
from src.metrics import nmae  # noqa: E402
from src.modeling import make_folds, fit_predict, refit_full  # noqa: E402
from t3_fill import fill  # noqa: E402


def tuned_study(name, fallback):
    """Same as x2_t3_identity_all.py: best Optuna params, learning rate halved for the final fit."""
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

FULL_W, WHALE_MIX = 0.5, 0.5
blend_full = lambda a, b: (1 - FULL_W) * np.asarray(a) + FULL_W * np.asarray(b)

tr, te = task3.load()
y = tr[task3.TARGET].values.astype(float)
folds = make_folds(y, kind="regression")
X, Xt = task3.add_features(fill(tr)[0]), task3.add_features(fill(te)[0])

# ---------------------------------------------------------------- pool (OOF + 5-fold test) and hurdle
z = np.load(OUT / "t3_identity_pool_all.npz", allow_pickle=False)
names = list(z["names"])
P, Pt = z["P_oof"], z["P_test"]
h = dict(np.load(OUT / "t3_identity_hurdle.npz", allow_pickle=False))
sigma0, smear = float(h["sigma"]), float(h["smear"])
tm = task3.tune_mixture(y, h["p_oof"], h["mu_oof"], sigma0)
th = task3.tune_hard_hurdle(y, h["p_oof"], h["mu_oof"])
gate_oof = (np.clip(h["p_oof"] + tm["shift"], 0, 1) > 0.5).astype(float)
assert np.array_equal(gate_oof, z["gate_oof"])

blend_oof = np.zeros(len(y))
for a, b in folds:
    w = task3.l1_blend_weights(y[a], [P[a, j] for j in range(P.shape[1])])
    blend_oof[b] = P[b] @ w
W3 = task3.l1_blend_weights(y, [P[:, j] for j in range(P.shape[1])])
scales = np.linspace(0.85, 1.15, 61)
SCALE3 = float(scales[int(np.argmax([nmae(y, blend_oof * s * gate_oof) for s in scales]))])
pred_oof = blend_oof * SCALE3 * gate_oof
print(f"blend OOF {nmae(y, pred_oof):.5f} (scale {SCALE3:.3f}); weights "
      f"{ {k: round(float(v), 3) for k, v in zip(names, W3) if v > 1e-3} }", flush=True)

# ---------------------------------------------------------------- Train+Val refit of every member (notebook 7.6)
P3_xgb = tuned_study("task3_xgb", {"learning_rate": 0.05, "max_depth": 7})
tw = {"learning_rate": 0.05, "objective": "tweedie", "tweedie_variance_power": 1.5, "metric": "l1"}
full = {
    "LGB-L1 raw THB": refit_full("lgb", "regression", P3_lgb, X, y, Xt, folds, name="T3 lgb raw")["test"],
    "XGB-L1 raw THB": refit_full("xgb", "regression", P3_xgb, X, y, Xt, folds, name="T3 xgb raw")["test"],
    "Cat-MAE raw THB": refit_full("cat", "regression", {"learning_rate": 0.08, "depth": 8}, X, y, Xt, folds, name="T3 cat")["test"],
    "LGB-L1 log1p": np.expm1(refit_full("lgb", "regression", P3_lgb, X, np.log1p(y), Xt, folds, name="T3 lgb log")["test"]),
    "LGB-Tweedie (mean)": refit_full("lgb", "regression", tw, X, y, Xt, folds, name="T3 tweedie")["test"],
}
full = {k: np.clip(v, 0, None) for k, v in full.items()}
hf = task3.hurdle_full("lgb", X, y, Xt, folds, {"learning_rate": 0.05}, {"learning_rate": 0.05})
p_fin, mu_fin = blend_full(h["p_test"], hf["p_test"]), blend_full(h["mu_test"], hf["mu_test"])
cols = []
for j, k in enumerate(names):
    if k == "Hurdle mixture-median":
        cols.append(task3.mixture_median(np.clip(p_fin + tm["shift"], 0, 1), mu_fin, tm["sigma"]))
    elif k == "Hurdle hard gate":
        cols.append(task3.hard_hurdle(p_fin, mu_fin, th["tau"]))
    elif k == "Hurdle E[y] + smearing":
        cols.append(task3.expected_value(p_fin, mu_fin, smear))
    else:
        cols.append(blend_full(Pt[:, j], full[k]))
gate_test = (np.clip(p_fin + tm["shift"], 0, 1) > 0.5).astype(float)
pred3 = np.clip(np.column_stack(cols) @ W3 * SCALE3, 0, None) * gate_test

# ---------------------------------------------------------------- whale-only model (segment 3)
WP = dict(objective="regression_l1", learning_rate=0.02, num_leaves=15, min_child_samples=50, colsample_bytree=0.8)
w_tr = tr.segment.fillna(-1).values == 3
w_te = te.segment.fillna(-1).values == 3
oof_w, test_w = np.full(len(y), np.nan), np.zeros(int(w_te.sum()))
for a, b in folds:
    a, b = a[w_tr[a]], b[w_tr[b]]
    _, (pv, pt), _ = fit_predict("lgb", "regression", WP, X.iloc[a], np.log1p(y[a]), X.iloc[b], np.log1p(y[b]),
                                 [X.iloc[b], Xt[w_te]])
    oof_w[b] = np.clip(np.expm1(pv), 0, None)
    test_w += np.clip(np.expm1(pt), 0, None) / len(folds)
idx = np.where(w_tr)[0]
remap = {old: new for new, old in enumerate(idx)}
f0 = [(np.array([remap[i] for i in folds[0][0] if w_tr[i]]), np.array([remap[i] for i in folds[0][1] if w_tr[i]]))]
full_w = np.clip(np.expm1(refit_full("lgb", "regression", WP, X.iloc[idx], np.log1p(y[idx]), Xt[w_te], f0,
                                     name="T3 whale")["test"]), 0, None)
pred_oof_final = pred_oof.copy()
pred_oof_final[w_tr] = WHALE_MIX * oof_w[w_tr] + (1 - WHALE_MIX) * pred_oof[w_tr]
pred3_final = pred3.copy()
pred3_final[w_te] = WHALE_MIX * blend_full(test_w, full_w) + (1 - WHALE_MIX) * pred3[w_te]
print(f"\nOOF score: blend {nmae(y, pred_oof):.5f} -> + whale model {nmae(y, pred_oof_final):.5f} "
      f"(v3 pipeline 0.78917)")

# ---------------------------------------------------------------- build the file from the 0.76 file
base = pd.read_csv(C.ROOT / "versions/v5_LB0.76/submission.csv")
assert len(base) == len(pred3_final) == C.N_TEST
sub = base.copy()
sub["task3"] = np.round(pred3_final, 2)
assert sub.shape == (C.N_TEST, 6) and sub.isna().sum().sum() == 0
assert all(str(sub[c].dtype).startswith("int") for c in ["task1", "task2", "task4", "task5"])
assert (sub.task3 >= 0).all() and np.isfinite(sub.task3).all()
for c in ["id", "task1", "task2", "task4", "task5"]:
    assert sub[c].equals(base[c]), c
old = base.task3.values
print(f"Task 3 vs the 0.76 file: corr(log1p) {np.corrcoef(np.log1p(old), np.log1p(sub.task3))[0, 1]:.4f}; "
      f"zero share {np.mean(old == 0):.3f} -> {np.mean(sub.task3 == 0):.3f}; "
      f"median {np.median(old):.0f} -> {np.median(sub.task3):.0f}; mean {old.mean():.0f} -> {sub.task3.mean():.0f}")
np.save(OUT / "t3_candidate_test.npy", pred3_final)
out = C.ROOT / "submission_v6_try.csv"
sub.to_csv(out, index=False)
print(f"saved {out} | {sub.shape} | all checks passed | sha256 {hashlib.sha256(out.read_bytes()).hexdigest()[:16]}")
