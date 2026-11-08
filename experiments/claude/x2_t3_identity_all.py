"""Task 3, step 2: retrain the remaining blend members (XGB-L1, Cat-MAE, Tweedie) on the identity-filled
features, then add the whale-only model (j_t3_whale_model.py) on top. Same folds and parameters as the notebook.
"""
import warnings

import numpy as np
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
import common  # noqa: E402
from common import t3_blend_score, OUT  # noqa: E402
from src import task3  # noqa: E402
from src import config as C  # noqa: E402
from src.metrics import nmae  # noqa: E402
from src.modeling import make_folds, run_cv, fit_predict  # noqa: E402
from t3_fill import fill  # noqa: E402


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
folds = make_folds(y, kind="regression")
X, Xt = task3.add_features(fill(tr)[0]), task3.add_features(fill(te)[0])
z = np.load(OUT / "t3_identity_pool.npz", allow_pickle=False)
names = list(z["names"])
P, Pt = z["P_oof"].copy(), z["P_test"].copy()
gate, gate_t = z["gate_oof"], z["gate_test"]
s1, _ = t3_blend_score(y, folds, P, gate)
print(f"step 1 blend (LightGBM members refitted): {s1:.5f}")

P3_xgb = tuned_study("task3_xgb", {"learning_rate": 0.05, "max_depth": 7})
jobs = {"XGB-L1 raw THB": lambda: run_cv("xgb", "regression", P3_xgb, X, y, Xt, folds, verbose=False),
        "Cat-MAE raw THB": lambda: run_cv("cat", "regression", {"learning_rate": 0.08, "depth": 8}, X, y, Xt, folds, verbose=False),
        "LGB-Tweedie (mean)": lambda: run_cv("lgb", "regression", {"learning_rate": 0.05, "objective": "tweedie",
                                                                   "tweedie_variance_power": 1.5, "metric": "l1"},
                                             X, y, Xt, folds, verbose=False)}
for k, fn in jobs.items():
    r = fn()
    j = names.index(k)
    old = nmae(y, P[:, j])
    P[:, j], Pt[:, j] = np.clip(r["oof"], 0, None), np.clip(r["test"], 0, None)
    print(f"  {k:22s} {old:.5f} -> {nmae(y, P[:, j]):.5f}", flush=True)
s2, pred = t3_blend_score(y, folds, P, gate)
print(f"step 2 blend (all members on filled features): {s2:.5f}")
np.savez(OUT / "t3_identity_pool_all.npz", P_oof=P, P_test=Pt, gate_oof=gate, gate_test=gate_t, names=np.array(names))

# whale-only model on filled features, 50/50 with the blend on whale rows (the earlier best recipe)
seg = tr.segment.fillna(-1).values
w = seg == 3
params = dict(objective="regression_l1", learning_rate=0.02, num_leaves=15, min_child_samples=50, colsample_bytree=0.8)
oof = np.full(len(y), np.nan)
for a, b in folds:
    a, b = a[w[a]], b[w[b]]
    _, (pv,), _ = fit_predict("lgb", "regression", params, X.iloc[a], np.log1p(y[a]), X.iloc[b], np.log1p(y[b]), [X.iloc[b]])
    oof[b] = np.clip(np.expm1(pv), 0, None)
for mix in (0.3, 0.5, 0.7):
    new = pred.copy()
    new[w] = mix * oof[w] + (1 - mix) * pred[w]
    print(f"  + whale-only model, mix {mix:.1f} on whale rows: {nmae(y, new):.5f}")
