"""Round 11: methods of last year's 1st-place team (github.com/Saltywan/CPE342-KAGGLE, CPE342 1/2025, public repo) on our folds.

Their Task 1: median imputation, RandomForest / LightGBM / XGBoost / CatBoost (each Optuna-tuned), LightGBM meta-model on the
members' OOF probabilities, cut-off searched on macro F2. Their Task 3: median imputation, LightGBM-MAE / XGBoost (squared error) /
CatBoost-MAE, LightGBM-MAE meta-model. Our own Round 5 already tested a LightGBM-L1 meta-model for Task 3 (0.7914-0.7921 < 0.7941).
Ideas implemented here from the description only (no code copied); every number is on OUR 5 pipeline folds, nested where it matters:
  T1-M1/M2  LightGBM meta-model instead of our logistic-regression stacker (meta for fold k trained on the other folds' OOF rows)
  T1-R      RandomForest added as a 6th stack member
  T1-REF    their whole Task 1 recipe as a reference (fixed sensible parameters instead of their Optuna runs)
  T3-REF    their whole Task 3 recipe as a reference
Acceptance rule as in Round 10/11: >= 4/5 folds better, fold std not > 1.1x, paired-bootstrap 95% CI > 0, (T1) test-like not worse.
"""
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
import common  # noqa: E402
from common import load_t1, T1_MODELS, OUT, _logit  # noqa: E402
from src import task1, task3  # noqa: E402
from src.metrics import optimize_threshold, fbeta, nmae  # noqa: E402
from src.modeling import fit_predict, make_folds  # noqa: E402
from src.utils import load_preds  # noqa: E402

T0 = time.time()
log = lambda *a: print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
rng = np.random.default_rng(0)

# ============================================================================ Task 1
tr, te, X, y, Xt, folds, _ = load_t1()
raw = [c for c in tr.columns if c not in ("player_id", "is_cheater")]
f2 = lambda yt, yp: fbeta(yt, yp, 2, "binary")
fold_id = np.zeros(len(y), int)
for k, (a, b) in enumerate(folds):
    fold_id[b] = k
base = {n: load_preds(n)["oof"] for n in T1_MODELS}

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


def nested_tau(p):
    pred = np.zeros(len(y), int)
    for a, b in folds:
        pred[b] = p[b] >= optimize_threshold(y[a], p[a], 2.0, average="binary")[0]
    return pred


def lr_stack(members):
    S = np.column_stack([_logit(m) for m in members])
    o = np.zeros(len(y))
    for a, b in folds:
        o[b] = LogisticRegression(max_iter=3000).fit(S[a], y[a]).predict_proba(S[b])[:, 1]
    return o


def lgb_meta(members, params):
    S = np.column_stack(members)
    o = np.zeros(len(y))
    for a, b in folds:
        o[b] = lgb.LGBMClassifier(objective="binary", verbose=-1, n_jobs=-1, random_state=42, **params).fit(S[a], y[a]).predict_proba(S[b])[:, 1]
    return o


per_fold = lambda pred: np.array([f2(y[fold_id == k], pred[fold_id == k]) for k in range(5)])


def boot(pn, po, B=1000):
    d = []
    for _ in range(B):
        pw = rng.poisson(1.0, len(y)).astype(float)
        d.append(wf2(pn, pw) - wf2(po, pw))
    return np.percentile(d, [2.5, 97.5])


pred0 = nested_tau(lr_stack([base[n] for n in T1_MODELS]))
f0 = per_fold(pred0)
log(f"T1 B0 (our stack): F2 {f2(y, pred0):.4f}, folds {np.round(f0, 4)}, std {f0.std():.4f}")
rows = []


def add(name, desc, pred, tl="-"):
    fo = per_fold(pred)
    lo, hi = boot(pred, pred0)
    better = int((fo > f0).sum())
    keep = lo > 0 and better >= 4 and fo.std() <= f0.std() * 1.1 and (tl == "-" or float(tl) >= 0)
    rows.append([name, desc, f2(y, pred), f2(y, pred) - f2(y, pred0), f"[{lo:+.4f}, {hi:+.4f}]", f"{better}/5", fo.std(), tl,
                 "KEEP" if keep else ("reference" if name.endswith("REF") else "discard")])
    log(f"{name}: F2 {rows[-1][2]:.4f} ({rows[-1][3]:+.4f}), CI {rows[-1][4]}, folds better {better}/5, std {fo.std():.4f}, test-like {tl}")


META = {"T1-M1": dict(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=100, subsample=0.8, subsample_freq=1),
        "T1-M2": dict(n_estimators=500, learning_rate=0.02, num_leaves=4, min_child_samples=200, subsample=0.8, subsample_freq=1)}
for name, p in META.items():
    add(name, f"LightGBM meta-model ({p['num_leaves']} leaves, {p['n_estimators']} trees) instead of logistic regression",
        nested_tau(lgb_meta([base[n] for n in T1_MODELS], p)))

med = tr[raw].median()
Xm = X.fillna(X.median())
Xm_tl = X_tl.fillna(X.median())
rf_oof, rf_tl = np.zeros(len(y)), np.zeros(len(y))
for a, b in folds:
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, max_features=0.4, n_jobs=-1, random_state=42).fit(Xm.iloc[a], y[a])
    rf_oof[b] = rf.predict_proba(Xm.iloc[b])[:, 1]
    rf_tl[b] = rf.predict_proba(Xm_tl.iloc[b])[:, 1]
log(f"RandomForest member: F2 {f2(y, nested_tau(rf_oof)):.4f} (ExtraTrees member {f2(y, nested_tau(base['t1_et'])):.4f})")
et_tl = None
add("T1-R", "RandomForest (median-imputed) added as a 6th member", nested_tau(lr_stack([base[n] for n in T1_MODELS] + [rf_oof])),
    f"{wf2(nested_tau(rf_tl), w_tl) - wf2(nested_tau(rf_oof), w_tl):+.4f}")

# their whole recipe (24 raw columns they used, median imputation, 4 members, LightGBM meta)
cols24 = ["kill_death_ratio", "headshot_percentage", "win_rate", "accuracy_score", "kill_consistency", "reaction_time_ms", "account_age_days",
          "level", "level_progression_speed", "friend_network_size", "reports_received", "night_play_ratio", "weapon_switch_speed",
          "aiming_smoothness", "spray_control_score", "game_sense_score", "communication_rate", "team_play_score", "buy_decision_score",
          "first_blood_rate", "survival_time_avg", "damage_per_round", "utility_usage_rate", "crosshair_placement"]
R = tr[cols24].fillna(med[cols24]).reset_index(drop=True)
mem = {k: np.zeros(len(y)) for k in ("rf", "lgb", "xgb", "cat")}
P = {"lgb": {"learning_rate": 0.05, "num_leaves": 31}, "xgb": {"learning_rate": 0.05, "max_depth": 6}, "cat": {"learning_rate": 0.08, "depth": 6}}
for a, b in folds:
    mem["rf"][b] = RandomForestClassifier(n_estimators=300, max_depth=10, max_features=0.7, n_jobs=-1, random_state=42) \
        .fit(R.iloc[a], y[a]).predict_proba(R.iloc[b])[:, 1]
    for k in ("lgb", "xgb", "cat"):
        _, (pv,), _ = fit_predict(k, "binary", P[k], R.iloc[a], y[a], R.iloc[b], y[b], [R.iloc[b]])
        mem[k][b] = pv
add("T1-REF", "last year's recipe: 24 columns, median imputation, RF/LGB/XGB/Cat + LightGBM meta",
    nested_tau(lgb_meta(list(mem.values()), META["T1-M1"])))
log("their members alone: " + ", ".join(f"{k} {f2(y, nested_tau(v)):.4f}" for k, v in mem.items()))

print("\n" + pd.DataFrame(rows, columns=["Exp", "Change", "OOF F2", "delta", "95% CI", "folds better", "fold std", "test-like member delta",
                                         "Decision"]).to_string(index=False), flush=True)

# ============================================================================ Task 3 reference
tr3, te3 = task3.load()
y3 = tr3[task3.TARGET].values.astype(float)
folds3 = make_folds(y3, kind="regression")
c3 = [c for c in tr3.columns if c not in ("player_id", task3.TARGET, "tournament_participation")]
R3 = tr3[c3].fillna(tr3[c3].median()).reset_index(drop=True)
m3 = {k: np.zeros(len(y3)) for k in ("lgb", "xgb", "cat")}
for a, b in folds3:
    _, (pv,), _ = fit_predict("lgb", "regression", {"learning_rate": 0.05, "num_leaves": 63}, R3.iloc[a], y3[a], R3.iloc[b], y3[b], [R3.iloc[b]])
    m3["lgb"][b] = pv
    _, (pv,), _ = fit_predict("xgb", "regression", {"learning_rate": 0.05, "max_depth": 8, "objective": "reg:squarederror"},
                              R3.iloc[a], y3[a], R3.iloc[b], y3[b], [R3.iloc[b]])
    m3["xgb"][b] = pv
    _, (pv,), _ = fit_predict("cat", "regression", {"learning_rate": 0.08, "depth": 8}, R3.iloc[a], y3[a], R3.iloc[b], y3[b], [R3.iloc[b]])
    m3["cat"][b] = pv
S3 = np.column_stack(list(m3.values()))
meta3 = np.zeros(len(y3))
for a, b in folds3:
    meta3[b] = lgb.LGBMRegressor(objective="mae", n_estimators=500, learning_rate=0.03, num_leaves=16, verbose=-1, random_state=42) \
        .fit(S3[a], y3[a]).predict(S3[b])
log("T3 last year's members alone: " + ", ".join(f"{k} {nmae(y3, np.clip(v, 0, None)):.5f}" for k, v in m3.items()))
log(f"T3-REF last year's recipe (median imputation, LGB-MAE / XGB-squared / Cat-MAE + LightGBM-MAE meta): {nmae(y3, np.clip(meta3, 0, None)):.5f}"
    f" vs our pipeline 0.79423")
log("done")
