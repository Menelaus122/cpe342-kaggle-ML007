"""Round 11, Task 1: confirm the RandomForest member (T1-R in bk_lastyear_methods.py: +0.0011, CI [+0.0004, +0.0017], 5/5 folds).

Checks before merging:
  1. seed stability: RF seeds 42 / 7 / 2026, each added on its own
  2. RF with native NaN handling (sklearn >= 1.4, like our ExtraTrees member) instead of median imputation
  3. RF replacing ExtraTrees instead of being added
  4. FULL test-like evaluation: every member (LightGBM / XGBoost / CatBoost / ExtraTrees / LogReg, notebook settings, + RF) refitted on
     the 5 folds and also scored on the test-like version of the validation rows (extra masking where test misses more cells,
     reweighting where it misses fewer); the stack's meta-model and cut-off are learned on the plain OOF of the training folds.
"""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

warnings.filterwarnings("ignore")
import common  # noqa: E402
from common import load_t1, T1_MODELS, _logit, tuned  # noqa: E402
from src import task1  # noqa: E402
from src.config import SEED  # noqa: E402
from src.metrics import optimize_threshold, fbeta  # noqa: E402
from src.modeling import fit_predict  # noqa: E402
from src.utils import load_preds  # noqa: E402

T0 = time.time()
log = lambda *a: print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
rng = np.random.default_rng(0)
tr, te, X, y, Xt, folds, _ = load_t1()
raw = [c for c in tr.columns if c not in ("player_id", "is_cheater")]
f2 = lambda yt, yp: fbeta(yt, yp, 2, "binary")
fold_id = np.zeros(len(y), int)
for k, (a, b) in enumerate(folds):
    fold_id[b] = k
base = {n: load_preds(n)["oof"] for n in T1_MODELS}
spw = (y == 0).sum() / (y == 1).sum()

q_tr, q_te = tr[raw].isna().mean(), te[raw].isna().mean()
more = [c for c in raw if q_te[c] > q_tr[c]]
less = [c for c in raw if q_te[c] < q_tr[c]]
mrng = np.random.default_rng(123)
raw_tl = tr[raw].copy()
for c in more:
    obs = raw_tl[c].notna().values
    raw_tl.loc[obs & (mrng.random(len(raw_tl)) < (q_te[c] - q_tr[c]) / (1 - q_tr[c])), c] = np.nan
X_tl = task1.add_features(raw_tl, ref=tr[raw])[X.columns]
w_tl = np.ones(len(y))
for c in less:
    m = tr[c].isna().values
    w_tl *= np.where(m, q_te[c] / q_tr[c], (1 - q_te[c]) / (1 - q_tr[c]))


def wf2(pred, w):
    tp = w @ ((y == 1) & (pred == 1)); fp = w @ ((y == 0) & (pred == 1)); fn = w @ ((y == 1) & (pred == 0))
    return 5 * tp / (5 * tp + 4 * fn + fp)


def stack_pred(members, members_eval=None):
    """Nested: meta-model + cut-off from the plain OOF of the other folds; applied to members_eval (default = plain) of fold k."""
    S = np.column_stack([_logit(m) for m in members])
    Se = S if members_eval is None else np.column_stack([_logit(m) for m in members_eval])
    pred = np.zeros(len(y), int)
    for a, b in folds:
        lr = LogisticRegression(max_iter=3000).fit(S[a], y[a])
        t = optimize_threshold(y[a], lr.predict_proba(S[a])[:, 1], 2.0, average="binary")[0]
        pred[b] = lr.predict_proba(Se[b])[:, 1] >= t
    return pred


def nested_tau_stack(members):
    """Same as the notebook evaluation: meta OOF first, then cut-off from the other folds' stacked OOF."""
    S = np.column_stack([_logit(m) for m in members])
    o = np.zeros(len(y))
    for a, b in folds:
        o[b] = LogisticRegression(max_iter=3000).fit(S[a], y[a]).predict_proba(S[b])[:, 1]
    pred = np.zeros(len(y), int)
    for a, b in folds:
        pred[b] = o[b] >= optimize_threshold(y[a], o[a], 2.0, average="binary")[0]
    return pred


per_fold = lambda pred: np.array([f2(y[fold_id == k], pred[fold_id == k]) for k in range(5)])


def boot(pn, po, w=None, B=1000):
    w = np.ones(len(y)) if w is None else w
    d = []
    for _ in range(B):
        pw = w * rng.poisson(1.0, len(y))
        d.append(wf2(pn, pw) - wf2(po, pw))
    return np.percentile(d, [2.5, 97.5])


def report(name, pred, ref, rows):
    fo, fr = per_fold(pred), per_fold(ref)
    lo, hi = boot(pred, ref)
    rows.append([name, f2(y, pred), f2(y, pred) - f2(y, ref), f"[{lo:+.4f}, {hi:+.4f}]", f"{int((fo > fr).sum())}/5", fo.std()])
    log(f"{name}: F2 {rows[-1][1]:.4f} ({rows[-1][2]:+.4f}), CI {rows[-1][3]}, folds better {rows[-1][4]}, std {fo.std():.4f}")


def rf_cv(seed, impute=True, Xe=None):
    Xa = X.fillna(X.median()) if impute else X
    Xb = None if Xe is None else (Xe.fillna(X.median()) if impute else Xe)
    o, oe = np.zeros(len(y)), np.zeros(len(y))
    for a, b in folds:
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, max_features=0.4, n_jobs=-1, random_state=seed).fit(Xa.iloc[a], y[a])
        o[b] = m.predict_proba(Xa.iloc[b])[:, 1]
        if Xb is not None:
            oe[b] = m.predict_proba(Xb.iloc[b])[:, 1]
    return o, oe


rows = []
ref = nested_tau_stack([base[n] for n in T1_MODELS])
log(f"B0 notebook stack: F2 {f2(y, ref):.4f}")
rf = {}
for s in (42, 7, 2026):
    rf[s], _ = rf_cv(s)
    report(f"+RF (median-imputed), seed {s}", nested_tau_stack([base[n] for n in T1_MODELS] + [rf[s]]), ref, rows)
rf_nan, _ = rf_cv(42, impute=False)
report("+RF (native NaN), seed 42", nested_tau_stack([base[n] for n in T1_MODELS] + [rf_nan]), ref, rows)
report("RF replaces ExtraTrees, seed 42", nested_tau_stack([base[n] for n in T1_MODELS if n != "t1_et"] + [rf[42]]), ref, rows)
print("\n" + pd.DataFrame(rows, columns=["Variant", "OOF F2", "delta", "95% CI", "folds better", "fold std"]).to_string(index=False), flush=True)

# ---------------------------------------------------------------- full test-like evaluation (all members refitted)
P = {"lgb": tuned("task1_lgb", {"learning_rate": 0.05, "scale_pos_weight": spw}),
     "xgb": {"learning_rate": 0.03, "max_depth": 6, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.6, "scale_pos_weight": spw},
     "cat": tuned("task1_cat", {"learning_rate": 0.06, "depth": 6, "scale_pos_weight": spw})}
plain, tl = {}, {}
for k, p in P.items():
    o, ot = np.zeros(len(y)), np.zeros(len(y))
    for a, b in folds:
        _, (pv, pt), _ = fit_predict(k, "binary", p, X.iloc[a], y[a], X.iloc[b], y[b], [X.iloc[b], X_tl.iloc[b]])
        o[b], ot[b] = pv, pt
    plain[k], tl[k] = o, ot
    log(f"refit {k}: plain member F2 {f2(y, (o >= optimize_threshold(y, o, 2.0, average='binary')[0]).astype(int)):.4f}")
SK = {"et": lambda: ExtraTreesClassifier(n_estimators=400, min_samples_leaf=4, max_features=0.4, class_weight="balanced", n_jobs=-1, random_state=SEED),
      "lr": lambda: make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                  QuantileTransformer(output_distribution="normal", random_state=SEED),
                                  LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced"))}
for k, mk in SK.items():
    o, ot = np.zeros(len(y)), np.zeros(len(y))
    for a, b in folds:
        m = mk().fit(X.values[a], y[a])
        o[b], ot[b] = m.predict_proba(X.values[b])[:, 1], m.predict_proba(X_tl.values[b])[:, 1]
    plain[k], tl[k] = o, ot
    log(f"refit {k} done")
plain["rf"], tl["rf"] = rf_cv(42, Xe=X_tl)
five = ["lgb", "xgb", "cat", "et", "lr"]
for name, keys in (("+RF", five + ["rf"]), ("RF replaces ExtraTrees", ["lgb", "xgb", "cat", "lr", "rf"])):
    r_plain, n_plain = stack_pred([plain[k] for k in five]), stack_pred([plain[k] for k in keys])
    r_tl = stack_pred([plain[k] for k in five], [tl[k] for k in five])
    n_tl = stack_pred([plain[k] for k in keys], [tl[k] for k in keys])
    lo, hi = boot(n_tl, r_tl, w_tl)
    log(f"{name} (refitted members): plain F2 {f2(y, r_plain):.4f} -> {f2(y, n_plain):.4f} ({f2(y, n_plain) - f2(y, r_plain):+.4f}); "
        f"TEST-LIKE F2 {wf2(r_tl, w_tl):.4f} -> {wf2(n_tl, w_tl):.4f} ({wf2(n_tl, w_tl) - wf2(r_tl, w_tl):+.4f}, CI [{lo:+.4f}, {hi:+.4f}])")
np.savez(common.OUT / "bm_t1_rf.npz", rf42=rf[42], rf7=rf[7], rf2026=rf[2026], rf_nan=rf_nan)
log("done")
