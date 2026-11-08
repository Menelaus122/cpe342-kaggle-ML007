"""Task 1 round 8 (2026-11-08): three new hypotheses, strict validation (nested tau, paired bootstrap).

Error analysis (this round): the stack loses most where a key column is missing. AUC with / without the value:
account_age_days 0.897 / 0.853 (14.5% missing in train), friend_network_size 0.901 / 0.852 (22.3%),
reports_received 0.901 / 0.850 (22.3%), crosshair_placement 0.895 / 0.859 (12.8%). account_age_days is strongly
predictable from level_progression_speed and friend_network_size (|Spearman| 0.81 / 0.63).

E1 feature engineering : model-based imputation of every column whose value is predictable from the others
                         (imputers = LightGBM regressors fitted on TRAIN features only, no labels; R^2 >= 0.15 in
                         2-fold CV), then the notebook LightGBM. Compared with the same LightGBM on the raw data
                         (control, same folds/params/early stopping), each swapped into the 5-member stack.
E2 specialisation      : a LightGBM trained only on the ambiguous middle (stack p between 0.02 and 0.95, in-fold),
                         combined with the stack by a logistic stacker on the middle rows (nested).
E3 post-processing     : multiple imputation at prediction time: 5 stochastic draws per missing value (imputer
                         prediction + resampled out-of-fold residual), probabilities averaged (E1 model).
All decisions: nested tau (chosen on the other 4 folds), paired Poisson bootstrap of the F2 difference (1,000).
"""
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
import common  # noqa: E402
from common import load_t1, T1_MODELS, OUT, _logit, tuned  # noqa: E402
from src import task1  # noqa: E402
from src.utils import load_preds  # noqa: E402
from src.metrics import optimize_threshold, fbeta  # noqa: E402
from src.modeling import fit_predict, LGB_BASE  # noqa: E402

T0 = time.time()
log = lambda *a: print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
rng = np.random.default_rng(0)
tr, te, X, y, Xt, folds, _ = load_t1()
raw = [c for c in tr.columns if c not in ("player_id", "is_cheater")]
spw = (y == 0).sum() / (y == 1).sum()
P1_lgb = tuned("task1_lgb", {"learning_rate": 0.05, "scale_pos_weight": spw})
base = {n: load_preds(n)["oof"] for n in T1_MODELS}
f2 = lambda yt, yp: fbeta(yt, yp, 2, "binary")
key_present = tr[["account_age_days", "friend_network_size", "reports_received", "crosshair_placement"]].notna().all(1).values


def stack(oofs):
    S = np.column_stack([_logit(o) for o in oofs])
    o = np.zeros(len(y))
    for a, b in folds:
        o[b] = LogisticRegression(max_iter=3000).fit(S[a], y[a]).predict_proba(S[b])[:, 1]
    return o


def nested(p):
    pred = np.zeros(len(y), int)
    for a, b in folds:
        t, _ = optimize_threshold(y[a], p[a], 2.0, average="binary")
        pred[b] = p[b] >= t
    return pred


def boot(pn, po, B=1000):
    yb = y.astype(bool)
    M = np.column_stack([yb & (pn == 1), ~yb & (pn == 1), yb & (pn == 0),
                         yb & (po == 1), ~yb & (po == 1), yb & (po == 0)]).astype(float)
    d = np.empty(B)
    for i in range(B):
        c = rng.poisson(1.0, len(y)) @ M
        d[i] = 5 * c[0] / (5 * c[0] + 4 * c[2] + c[1]) - 5 * c[3] / (5 * c[3] + 4 * c[5] + c[4])
    return np.percentile(d, [2.5, 97.5])


rows = []


def report(name, desc, p, pred, ref_pred, ref_name):
    s, s0 = f2(y, pred), f2(y, ref_pred)
    lo, hi = boot(pred, ref_pred)
    kp = key_present
    rows.append([name, desc, roc_auc_score(y, p), s, ref_name, s - s0, f"[{lo:+.4f}, {hi:+.4f}]",
                 f2(y[kp], pred[kp]) - f2(y[kp], ref_pred[kp]), "KEEP" if lo > 0 else "discard"])
    log(f"{name}: AUC {rows[-1][2]:.4f}, nested F2 {s:.4f} (delta vs {ref_name} {s - s0:+.4f}, CI [{lo:+.4f}, {hi:+.4f}])")


p0 = stack(list(base.values()))
pred0 = nested(p0)
log(f"B0 notebook stack: AUC {roc_auc_score(y, p0):.4f}, nested F2 {f2(y, pred0):.4f}")


def lgb_cv(Xa, Xta=None):
    """Notebook-style LightGBM (tuned params, early stopping on the validation fold), OOF + per-fold models."""
    oof, models = np.zeros(len(y)), []
    for a, b in folds:
        m, (pv,), _ = fit_predict("lgb", "binary", P1_lgb, Xa.iloc[a], y[a], Xa.iloc[b], y[b], [Xa.iloc[b]])
        oof[b] = pv
        models.append(m)
    return oof, models


# ------------------------------------------------------------------ control: same LightGBM on raw features
oof_ctrl, _ = lgb_cv(X)
p_ctrl = stack([oof_ctrl if n == "t1_lgb" else base[n] for n in T1_MODELS])
pred_ctrl = nested(p_ctrl)
log(f"control (LightGBM refit on raw data, swapped in): AUC {roc_auc_score(y, p_ctrl):.4f}, nested F2 {f2(y, pred_ctrl):.4f}")

# ------------------------------------------------------------------ E1: imputers on train features only
IMP = dict(LGB_BASE, n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50)
imputed = tr[raw].copy()
resid, imp_models = {}, {}
for c in raw:
    known = tr[c].notna().values
    others = [o for o in raw if o != c]
    idx = np.where(known)[0]
    half = rng.permutation(idx)
    A, B_ = half[: len(half) // 2], half[len(half) // 2:]
    oo = np.zeros(len(tr))
    for fit_i, pred_i in ((A, B_), (B_, A)):
        m = lgb.LGBMRegressor(**IMP).fit(tr.iloc[fit_i][others], tr.iloc[fit_i][c])
        oo[pred_i] = m.predict(tr.iloc[pred_i][others])
    r2 = 1 - np.var(tr[c].values[idx] - oo[idx]) / np.var(tr[c].values[idx])
    if r2 >= 0.15:
        m = lgb.LGBMRegressor(**IMP).fit(tr.iloc[idx][others], tr.iloc[idx][c])
        imp_models[c] = (m, others)
        resid[c] = tr[c].values[idx] - oo[idx]
        miss = ~known
        imputed.loc[miss, c] = m.predict(tr.loc[miss, others])
        log(f"  impute {c:26s} R^2 {r2:.3f} ({int(miss.sum()):,} train values filled)")
Xi = task1.add_features(imputed.assign(is_cheater=y), ref=tr[raw])
Xi["n_missing"] = X["n_missing"].values                     # keep the true missing count
for c in imp_models:
    Xi[f"{c}_was_missing"] = tr[c].isna().astype(int).values
oof_imp, models_imp = lgb_cv(Xi)
p1 = stack([oof_imp if n == "t1_lgb" else base[n] for n in T1_MODELS])
report("E1a", "imputed predictable columns, LightGBM swapped in", p1, nested(p1), pred_ctrl, "control")
p1b = stack(list(base.values()) + [oof_imp])
report("E1b", "imputed LightGBM as a 6th stack member", p1b, nested(p1b), pred0, "B0")

# ------------------------------------------------------------------ E3: multiple imputation at prediction time
K = 5
oof_mi = np.zeros(len(y))
for (a, b), m in zip(folds, models_imp):
    acc = np.zeros(len(b))
    for k in range(K):
        draw = tr.iloc[b][raw].copy()
        for c, (im, others) in imp_models.items():
            miss = draw[c].isna().values
            if miss.any():
                draw.loc[miss, c] = im.predict(draw.loc[miss, others]) + rng.choice(resid[c], miss.sum())
        Xd = task1.add_features(draw.assign(is_cheater=y[b]), ref=tr[raw])
        Xd["n_missing"] = X["n_missing"].values[b]
        for c in imp_models:
            Xd[f"{c}_was_missing"] = tr[c].isna().astype(int).values[b]
        acc += m.predict_proba(Xd[Xi.columns])[:, 1] / K
    oof_mi[b] = acc
p3 = stack([oof_mi if n == "t1_lgb" else base[n] for n in T1_MODELS])
report("E3", "E1 model, 5 stochastic imputations averaged at prediction", p3, nested(p3), pred_ctrl, "control")

# ------------------------------------------------------------------ E2: middle specialist (in-fold)
mid = (p0 > 0.02) & (p0 < 0.95)
log(f"middle rows: {mid.sum():,} ({mid.mean():.3f}), cheater share {y[mid].mean():.3f}")
spec = np.full(len(y), np.nan)
for k, (a, b) in enumerate(folds):
    va = folds[(k + 1) % 5][1]
    trn = np.setdiff1d(a, va)
    trn, va, bb = trn[mid[trn]], va[mid[va]], b[mid[b]]
    _, (pv,), _ = fit_predict("lgb", "binary", P1_lgb, X.iloc[trn], y[trn], X.iloc[va], y[va], [X.iloc[bb]])
    spec[bb] = pv
log(f"specialist AUC inside the middle {roc_auc_score(y[mid], spec[mid]):.4f} vs stack {roc_auc_score(y[mid], p0[mid]):.4f}")
p2 = p0.copy()
for a, b in folds:
    aa, bb = a[mid[a]], b[mid[b]]
    Sa = np.column_stack([_logit(p0[aa]), _logit(spec[aa])])
    p2[bb] = LogisticRegression(max_iter=2000).fit(Sa, y[aa]).predict_proba(
        np.column_stack([_logit(p0[bb]), _logit(spec[bb])]))[:, 1]
report("E2", "middle-region specialist LightGBM, re-stacked on the middle rows", p2, nested(p2), pred0, "B0")

R = pd.DataFrame(rows, columns=["Experiment", "Change", "OOF AUC", "OOF F2 (nested tau)", "reference", "delta F2",
                                "95% CI", "delta F2 on rows with the key columns present", "Decision"])
print("\n" + R.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
R.to_csv(OUT / "t1_round8_results.csv", index=False)
np.savez(OUT / "t1_round8_members.npz", ctrl=oof_ctrl, imp=oof_imp, mi=oof_mi, spec=spec)
log("done")
