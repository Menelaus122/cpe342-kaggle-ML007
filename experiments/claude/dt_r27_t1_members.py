"""Round 27, Task 1: extra level-1 members under the v18 level-2 stacker (LogReg + LightGBM + CatBoost on the member log-odds).

    python dt_r27_t1_members.py

Earlier rounds, scored under the logistic stacker and the CI gate:
  round 6   members without the 18 no-signal columns: member AUC up in 14 of 15 fold x model cases; stack +0.0004 / +0.0005 (discard)
  round 11  XGBoost with tuned settings: member F2 +0.0019, stack +0.0005 [-0.0002, +0.0012], 3/5 folds (discard)
Their out-of-fold predictions are still on disk and were made on the pipeline folds (outputs/t1_round6b_members.npz: LightGBM 3-seed
bag, XGBoost, CatBoost on the reduced columns; outputs/bj_t1_tuned_oof.npz: xgb_tuned). The question now: do they add anything on
top of the v18 stacker? Member sets (9 pipeline members +):
  A  XGBoost tuned
  B  LightGBM reduced
  C  XGBoost tuned + LightGBM reduced
  D  XGBoost tuned + LightGBM / XGBoost / CatBoost reduced
Level 2 exactly as in the pipeline (dm_r26_t1_l2stack.py: lr + lgb + cat, 3 seeds, mean of log-odds); meta-models and cut-off
nested; pipeline folds + three other stacker splits. Acceptance: mean nested gain > 0 and >= 4/5 folds on the splits.
"""
import warnings

import lightgbm as lgb
import numpy as np
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import common  # noqa: F401  (chdir to the repo root)
from common import load_t1  # noqa: E402
from src.metrics import f2, optimize_threshold  # noqa: E402
from src.utils import load_preds  # noqa: E402

warnings.filterwarnings("ignore")
_, _, _, y, _, folds, _ = load_t1()
y = np.asarray(y)
NAMES = ["t1_lgb", "t1_xgb", "t1_cat", "t1_et", "t1_lr", "t1_rf24", "t1_lgb24", "t1_xgb24", "t1_cat24"]
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
base_cols = [logit(load_preds(k)["oof"]) for k in NAMES]
r6 = np.load("experiments/claude/outputs/t1_round6b_members.npz")
r11 = np.load("experiments/claude/outputs/bj_t1_tuned_oof.npz")
xt, lr_, xr, cr = logit(r11["xgb_tuned"]), logit(r6["lgb"]), logit(r6["xgb"]), logit(r6["cat"])
SETS = {"9 members (v18)": base_cols, "A + XGB tuned": base_cols + [xt], "B + LGB reduced": base_cols + [lr_],
        "C + XGB tuned + LGB reduced": base_cols + [xt, lr_], "D + XGB tuned + 3 reduced": base_cols + [xt, lr_, xr, cr]}
LGBP = dict(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            n_jobs=8, verbose=-1)
MAKE = [lambda: [LogisticRegression(C=1.0, max_iter=2000)],
        lambda: [lgb.LGBMClassifier(**LGBP, random_state=s) for s in (0, 1, 2)],
        lambda: [CatBoostClassifier(depth=4, iterations=400, learning_rate=0.05, random_seed=s, verbose=0, thread_count=8) for s in (0, 1, 2)]]


def level2(cols, fl):
    S = np.column_stack(cols)
    out = np.zeros(len(y))
    for a, b in fl:
        out[b] = np.mean([np.mean([logit(m.fit(S[a], y[a]).predict_proba(S[b])[:, 1]) for m in mk()], 0) for mk in MAKE], 0)
    return 1 / (1 + np.exp(-out))


def nested_cut(p, fo):
    o = np.zeros(len(y), int)
    for i in range(5):
        tr = fo != i
        o[~tr] = p[~tr] >= optimize_threshold(y[tr], p[tr], 2.0)[0]
    return o


rng = np.random.default_rng(0)
boot = [rng.integers(0, len(y), len(y)) for _ in range(1000)]
splits = [("pipeline folds", folds)] + [(f"split seed {s}", list(StratifiedKFold(5, shuffle=True, random_state=s).split(base_cols[0], y))) for s in (1, 2, 3)]
summary = {}
for label, fl in splits:
    fo = np.zeros(len(y), int)
    for i, (_, b) in enumerate(fl):
        fo[b] = i
    lab = {k: nested_cut(level2(c, fl), fo) for k, c in SETS.items()}
    base = lab["9 members (v18)"]
    print(f"\n{label}: v18 stacker F2 {f2(y, base):.4f}", flush=True)
    for k in list(SETS)[1:]:
        d = f2(y, lab[k]) - f2(y, base)
        pf = [f2(y[fo == i], lab[k][fo == i]) - f2(y[fo == i], base[fo == i]) for i in range(5)]
        bs = [f2(y[i], lab[k][i]) - f2(y[i], base[i]) for i in boot]
        summary.setdefault(k, []).append((d, sum(x > 0 for x in pf)))
        print(f"  {k:30s} F2 {f2(y, lab[k]):.4f}  delta {d:+.4f} [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}]  folds +{sum(x > 0 for x in pf)}/5 "
              f"({' '.join(f'{x:+.4f}' for x in pf)})  labels changed {np.mean(lab[k] != base):.3%}", flush=True)
print("\nmean delta over the 4 splits (folds better, per split):")
for k, v in summary.items():
    print(f"  {k:30s} {np.mean([d for d, _ in v]):+.4f}  ({' '.join(str(n) + '/5' for _, n in v)})")
