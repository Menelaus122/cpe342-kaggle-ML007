"""Round 11, Task 1: combine our stack with members built the way last year's 1st-place team built theirs (bk_lastyear_methods.py).

Their recipe on our folds: 0.8376 vs our 0.8364; their members alone (24 raw columns, median imputation) beat ours: CatBoost 0.8370,
LightGBM 0.8355, XGBoost 0.8351. Which part drives it, and does it add to our stack?
  D1/D2/D3  ablation on LightGBM: 24 columns + median / 24 columns + native NaN / all 31 raw columns + median (no engineered features)
  V2 our 5 members + their 4    V3 our 5 + RF + their 4    V5 their 4 with our logistic-regression stacker
Every member is refitted on the 5 pipeline folds and also scored on the test-like validation rows, so every variant gets the full
test-like check; seeds 42 / 7 for the members that are added. Acceptance rule unchanged.
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
from common import load_t1, _logit, tuned  # noqa: E402
from src import task1  # noqa: E402
from src.config import SEED  # noqa: E402
from src.metrics import optimize_threshold, fbeta  # noqa: E402
from src.modeling import fit_predict  # noqa: E402

T0 = time.time()
log = lambda *a: print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
rng = np.random.default_rng(0)
tr, te, X, y, Xt, folds, _ = load_t1()
raw = [c for c in tr.columns if c not in ("player_id", "is_cheater")]
f2 = lambda yt, yp: fbeta(yt, yp, 2, "binary")
fold_id = np.zeros(len(y), int)
for k, (a, b) in enumerate(folds):
    fold_id[b] = k
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
raw_pl = tr[raw].reset_index(drop=True)
raw_tl = raw_tl.reset_index(drop=True)
med = raw_pl.median()
C24 = ["kill_death_ratio", "headshot_percentage", "win_rate", "accuracy_score", "kill_consistency", "reaction_time_ms", "account_age_days",
       "level", "level_progression_speed", "friend_network_size", "reports_received", "night_play_ratio", "weapon_switch_speed",
       "aiming_smoothness", "spray_control_score", "game_sense_score", "communication_rate", "team_play_score", "buy_decision_score",
       "first_blood_rate", "survival_time_avg", "damage_per_round", "utility_usage_rate", "crosshair_placement"]
log("columns they left out:", [c for c in raw if c not in C24])


def wf2(pred, w):
    tp = w @ ((y == 1) & (pred == 1)); fp = w @ ((y == 0) & (pred == 1)); fn = w @ ((y == 1) & (pred == 0))
    return 5 * tp / (5 * tp + 4 * fn + fp)


def tau_pred(o, oe=None):
    oe = o if oe is None else oe
    pred = np.zeros(len(y), int)
    for a, b in folds:
        pred[b] = oe[b] >= optimize_threshold(y[a], o[a], 2.0, average="binary")[0]
    return pred


def stack_pred(members, members_eval=None):
    """Nested: meta-model + cut-off learned on the plain OOF of the other folds, applied to members_eval of fold k."""
    S = np.column_stack([_logit(m) for m in members])
    Se = S if members_eval is None else np.column_stack([_logit(m) for m in members_eval])
    o = np.zeros(len(y))
    for a, b in folds:
        o[b] = LogisticRegression(max_iter=3000).fit(S[a], y[a]).predict_proba(S[b])[:, 1]
    pred = np.zeros(len(y), int)
    for a, b in folds:
        lr = LogisticRegression(max_iter=3000).fit(S[a], y[a])
        t = optimize_threshold(y[a], o[a], 2.0, average="binary")[0]
        pred[b] = lr.predict_proba(Se[b])[:, 1] >= t
    return pred


per_fold = lambda pred: np.array([f2(y[fold_id == k], pred[fold_id == k]) for k in range(5)])


def boot(pn, po, w=None, B=1000):
    w = np.ones(len(y)) if w is None else w
    d = []
    for _ in range(B):
        pw = w * rng.poisson(1.0, len(y))
        d.append(wf2(pn, pw) - wf2(po, pw))
    return np.percentile(d, [2.5, 97.5])


def cv_gbdt(kind, params, A, Ae):
    o, oe = np.zeros(len(y)), np.zeros(len(y))
    for a, b in folds:
        _, (pv, pt), _ = fit_predict(kind, "binary", params, A.iloc[a], y[a], A.iloc[b], y[b], [A.iloc[b], Ae.iloc[b]])
        o[b], oe[b] = pv, pt
    return o, oe


def cv_sk(make, A, Ae):
    o, oe = np.zeros(len(y)), np.zeros(len(y))
    for a, b in folds:
        m = make().fit(A[a], y[a])
        o[b], oe[b] = m.predict_proba(A[b])[:, 1], m.predict_proba(Ae[b])[:, 1]
    return o, oe


member_f2 = lambda o: f2(y, tau_pred(o))

# ------------------------------------------------ ablation: what makes their LightGBM stronger?
L = {"learning_rate": 0.05, "num_leaves": 31}
abl = {"D0 our features (49, native NaN), same LightGBM settings": (X, X_tl),
       "D1 24 columns, median imputation": (raw_pl[C24].fillna(med[C24]), raw_tl[C24].fillna(med[C24])),
       "D2 24 columns, native NaN": (raw_pl[C24], raw_tl[C24]),
       "D3 all 31 raw columns, median imputation": (raw_pl.fillna(med), raw_tl.fillna(med)),
       "D4 all 31 raw columns, native NaN": (raw_pl, raw_tl)}
for name, (A, Ae) in abl.items():
    o, oe = cv_gbdt("lgb", L, A, Ae)
    log(f"{name}: member F2 {member_f2(o):.4f}, test-like member F2 {wf2(tau_pred(o, oe), w_tl):.4f}")

# ------------------------------------------------ members: ours (notebook settings) + RF + theirs (24 columns, median)
P = {"lgb": tuned("task1_lgb", {"learning_rate": 0.05, "scale_pos_weight": spw}),
     "xgb": {"learning_rate": 0.03, "max_depth": 6, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.6, "scale_pos_weight": spw},
     "cat": tuned("task1_cat", {"learning_rate": 0.06, "depth": 6, "scale_pos_weight": spw})}
M = {}
for k, p in P.items():
    M[k] = cv_gbdt(k, p, X, X_tl)
M["et"] = cv_sk(lambda: ExtraTreesClassifier(n_estimators=400, min_samples_leaf=4, max_features=0.4, class_weight="balanced", n_jobs=-1,
                                             random_state=SEED), X.values, X_tl.values)
M["lr"] = cv_sk(lambda: make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                      QuantileTransformer(output_distribution="normal", random_state=SEED),
                                      LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced")), X.values, X_tl.values)
Xm, Xm_tl = X.fillna(X.median()), X_tl.fillna(X.median())
R24, R24_tl = raw_pl[C24].fillna(med[C24]), raw_tl[C24].fillna(med[C24])
for s in (42, 7):
    M[f"rf_{s}"] = cv_sk(lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3, max_features=0.4, n_jobs=-1, random_state=s),
                         Xm.values, Xm_tl.values)
    M[f"their_rf_{s}"] = cv_sk(lambda: RandomForestClassifier(n_estimators=300, max_depth=10, max_features=0.7, n_jobs=-1, random_state=s),
                               R24.values, R24_tl.values)
    for k, p in {"lgb": {"learning_rate": 0.05, "num_leaves": 31}, "xgb": {"learning_rate": 0.05, "max_depth": 6},
                 "cat": {"learning_rate": 0.08, "depth": 6}}.items():
        M[f"their_{k}_{s}"] = cv_gbdt(k, {**p, ("random_seed" if k == "cat" else "random_state"): s}, R24, R24_tl)
    log(f"seed {s}: members done")
for k, (o, oe) in M.items():
    log(f"  member {k}: F2 {member_f2(o):.4f}, test-like {wf2(tau_pred(o, oe), w_tl):.4f}")

ours = ["lgb", "xgb", "cat", "et", "lr"]
ref = stack_pred([M[k][0] for k in ours])
ref_tl = stack_pred([M[k][0] for k in ours], [M[k][1] for k in ours])
fr = per_fold(ref)
log(f"B0 (refitted members, our stack): F2 {f2(y, ref):.4f}, folds {np.round(fr, 4)}; test-like {wf2(ref_tl, w_tl):.4f}")
rows = []
for s in (42, 7):
    th = [f"their_{k}_{s}" for k in ("rf", "lgb", "xgb", "cat")]
    for name, keys in ((f"V1 +RF (seed {s})", ours + [f"rf_{s}"]), (f"V2 +their 4 (seed {s})", ours + th),
                       (f"V3 +RF +their 4 (seed {s})", ours + [f"rf_{s}"] + th), (f"V5 their 4 alone, LR stacker (seed {s})", th)):
        p = stack_pred([M[k][0] for k in keys])
        p_tl = stack_pred([M[k][0] for k in keys], [M[k][1] for k in keys])
        fo = per_fold(p)
        lo, hi = boot(p, ref)
        tlo, thi = boot(p_tl, ref_tl, w_tl)
        better = int((fo > fr).sum())
        dtl = wf2(p_tl, w_tl) - wf2(ref_tl, w_tl)
        keep = lo > 0 and better >= 4 and fo.std() <= fr.std() * 1.1 and dtl >= 0
        rows.append([name, f2(y, p), f2(y, p) - f2(y, ref), f"[{lo:+.4f}, {hi:+.4f}]", f"{better}/5", fo.std(), f"{dtl:+.4f}",
                     f"[{tlo:+.4f}, {thi:+.4f}]", "KEEP" if keep else "discard"])
        log(f"{name}: F2 {rows[-1][1]:.4f} ({rows[-1][2]:+.4f}) CI {rows[-1][3]}, {better}/5, std {fo.std():.4f}; test-like {dtl:+.4f} CI {rows[-1][7]}")
print("\n" + pd.DataFrame(rows, columns=["Variant", "OOF F2", "delta", "95% CI", "folds better", "fold std", "test-like delta", "test-like CI",
                                         "Decision"]).to_string(index=False), flush=True)
np.savez(common.OUT / "bn_t1_members.npz", **{f"{k}_oof": v[0] for k, v in M.items()}, **{f"{k}_tl": v[1] for k, v in M.items()})
log("done")
