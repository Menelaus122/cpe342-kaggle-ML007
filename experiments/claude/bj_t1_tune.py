"""Round 11, Task 1: tune the boosted-tree members of the stack, then test them with the Round 10 acceptance rule.

Before: LightGBM was tuned (40 Optuna trials, 2026-11-03); XGBoost and CatBoost used fixed settings.
Phase A (tuning): Optuna TPE (multivariate, seed 42) on the search spaces of scripts/tune.py, objective = positive-class F2
  at its best cut-off on a 3-fold split (as scripts/tune.py). Trial 0 of every study = the current settings, so a study
  can only match or beat them on its own folds. Studies: experiments/claude/outputs/optuna_round11.db.
Phase B (evaluation, the 5 pipeline folds): each tuned member is refitted with the notebook rule (learning rate halved) and
  swapped into the stack in place of the same model refitted with its current settings (control). Accepted only if
  better on >= 4/5 folds, fold std not > 1.1x, paired-bootstrap 95% CI > 0 and not worse under test-like missingness.
  Caveat: the hyperparameters were chosen on the same training rows (other folds), so a small optimism remains;
  that is why the rule asks for a CI above 0 rather than any gain.
Nothing is written to outputs/params; accepted settings would be merged into the pipeline by hand.
"""
import importlib.util
import json
import sys
import time
import warnings

import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
import common  # noqa: E402
from common import load_t1, T1_MODELS, OUT, _logit, tuned  # noqa: E402
from src import task1  # noqa: E402
from src.config import ROOT  # noqa: E402
from src.metrics import optimize_threshold, fbeta  # noqa: E402
from src.modeling import fit_predict, make_folds, run_cv  # noqa: E402
from src.utils import load_preds  # noqa: E402

spec = importlib.util.spec_from_file_location("tune", ROOT / "scripts" / "tune.py")
tune = importlib.util.module_from_spec(spec); spec.loader.exec_module(tune)
optuna.logging.set_verbosity(optuna.logging.WARNING)

T0 = time.time()
log = lambda *a: print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
rng = np.random.default_rng(0)
TRIALS = {"xgb": 30, "cat": 30, "lgb": 25}
STORAGE = f"sqlite:///{(OUT / 'optuna_round11.db').as_posix()}"

tr, te, X, y, Xt, folds, _ = load_t1()
raw = [c for c in tr.columns if c not in ("player_id", "is_cheater")]
spw = (y == 0).sum() / (y == 1).sum()
f2 = lambda yt, yp: fbeta(yt, yp, 2, "binary")

# current settings exactly as the notebook builds them (json learning rates are halved there; XGBoost is fixed)
CURRENT = {"lgb": tuned("task1_lgb", {"learning_rate": 0.05, "scale_pos_weight": spw}),
           "cat": tuned("task1_cat", {"learning_rate": 0.06, "depth": 6, "scale_pos_weight": spw}),
           "xgb": {"learning_rate": 0.03, "max_depth": 6, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.6,
                   "scale_pos_weight": spw}}
NAME = {"lgb": "t1_lgb", "xgb": "t1_xgb", "cat": "t1_cat"}

# ============================================================== Phase A: tuning on a 3-fold split
folds3 = make_folds(y, n_folds=3)
X_small = Xt.iloc[:10]
best = {}
for kind, n_trials in TRIALS.items():
    def objective(trial, kind=kind):
        params = tune.space(trial, kind, "binary")
        r = run_cv(kind, "binary", params, X, y, X_small, folds3, early_stopping=100, verbose=False)
        return optimize_threshold(y, r["oof"], 2.0, average="binary")[1]

    study = optuna.create_study(study_name=f"t1_{kind}_posF2", storage=STORAGE, direction="maximize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=42, multivariate=True))
    if not study.trials:   # trial 0 = current settings, mapped into the search space (learning rate un-halved for lgb/cat)
        cur = dict(CURRENT[kind])
        if kind in ("lgb", "cat"):
            cur["learning_rate"] *= 2
        # search-space keys the current settings leave at the library/base defaults (src/modeling.py XGB_BASE / CAT_BASE)
        defaults = {"xgb": {"reg_alpha": 1e-3, "reg_lambda": 1.0, "gamma": 0.0},
                    "cat": {"l2_leaf_reg": 3.0, "random_strength": 1.0, "bagging_temperature": 1.0}}.get(kind, {})
        cur = {**defaults, **cur}
        study.enqueue_trial({k: v for k, v in cur.items() if k not in ("n_jobs", "thread_count")}, skip_if_exists=True)
    left = n_trials - len([t for t in study.trials if t.state.name == "COMPLETE"])
    if left > 0:
        study.optimize(objective, n_trials=left,
                       callbacks=[lambda s, t: log(f"  {s.study_name} trial {t.number}: {t.value:.5f} (best {s.best_value:.5f})")])
    first = study.trials[0].value
    log(f"{kind}: current settings {first:.5f} -> best {study.best_value:.5f} (+{study.best_value - first:.5f}) on the tuning folds")
    p = dict(study.best_params)
    p["learning_rate"] = p["learning_rate"] / 2          # notebook rule for tuned parameters
    best[kind] = p
json.dump(best, open(OUT / "bj_t1_tuned_params.json", "w"), indent=1)

# ============================================================== Phase B: acceptance test on the 5 pipeline folds
base = {n: load_preds(n)["oof"] for n in T1_MODELS}
fold_id = np.zeros(len(y), int)
for k, (a, b) in enumerate(folds):
    fold_id[b] = k

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


def stack(members):
    S = np.column_stack([_logit(members[n]) for n in T1_MODELS])
    o = np.zeros(len(y))
    for a, b in folds:
        o[b] = LogisticRegression(max_iter=3000).fit(S[a], y[a]).predict_proba(S[b])[:, 1]
    return o


per_fold = lambda pred: np.array([f2(y[fold_id == k], pred[fold_id == k]) for k in range(5)])


def boot(pn, po, B=1000):
    d = []
    for _ in range(B):
        pw = rng.poisson(1.0, len(y)).astype(float)
        d.append(wf2(pn, pw) - wf2(po, pw))
    return np.percentile(d, [2.5, 97.5])


def cv(kind, params):
    oof, oof_tl = np.zeros(len(y)), np.zeros(len(y))
    for a, b in folds:
        _, (pv, pt), _ = fit_predict(kind, "binary", params, X.iloc[a], y[a], X.iloc[b], y[b], [X.iloc[b], X_tl.iloc[b]])
        oof[b], oof_tl[b] = pv, pt
    return oof, oof_tl


member_tl = lambda o_tl: wf2(nested_tau(o_tl), w_tl)
ctrl, tuned_m, rows = {}, {}, []
for kind in ("lgb", "xgb", "cat"):
    ctrl[kind] = cv(kind, CURRENT[kind])
    tuned_m[kind] = cv(kind, best[kind])
    log(f"{kind}: member F2 current {f2(y, nested_tau(ctrl[kind][0])):.4f} -> tuned {f2(y, nested_tau(tuned_m[kind][0])):.4f}")

ref_members = {**base, **{NAME[k]: ctrl[k][0] for k in ctrl}}
p_ref = stack(ref_members)
pred_ref = nested_tau(p_ref)
f_ref = per_fold(pred_ref)
log(f"control stack (members refitted with current settings): F2 {f2(y, pred_ref):.4f}, folds {np.round(f_ref, 4)}, std {f_ref.std():.4f}"
    f" | notebook stack {f2(y, nested_tau(stack(base))):.4f}")

variants = {"tuned XGBoost": ["xgb"], "tuned CatBoost": ["cat"], "re-tuned LightGBM": ["lgb"], "all three tuned": ["lgb", "xgb", "cat"]}
for name, kinds in variants.items():
    mem = dict(ref_members)
    for k in kinds:
        mem[NAME[k]] = tuned_m[k][0]
    pred = nested_tau(stack(mem))
    fo = per_fold(pred)
    lo, hi = boot(pred, pred_ref)
    better = int((fo > f_ref).sum())
    tl = np.mean([member_tl(tuned_m[k][1]) - member_tl(ctrl[k][1]) for k in kinds])
    keep = lo > 0 and better >= 4 and fo.std() <= f_ref.std() * 1.1 and tl >= 0
    rows.append([name, f2(y, pred), f2(y, pred) - f2(y, pred_ref), f"[{lo:+.4f}, {hi:+.4f}]", f"{better}/5", fo.std(), f"{tl:+.4f}",
                 "KEEP" if keep else "discard"])
    log(f"{name}: F2 {rows[-1][1]:.4f} ({rows[-1][2]:+.4f}), CI {rows[-1][3]}, folds better {better}/5, std {fo.std():.4f}, test-like member delta {tl:+.4f}")

import pandas as pd  # noqa: E402
print("\n" + pd.DataFrame(rows, columns=["Variant", "OOF F2", "delta", "95% CI", "folds better", "fold std", "test-like member delta",
                                         "Decision"]).to_string(index=False))
np.savez(OUT / "bj_t1_tuned_oof.npz", **{f"{k}_ctrl": ctrl[k][0] for k in ctrl}, **{f"{k}_tuned": tuned_m[k][0] for k in tuned_m})
log("done")
