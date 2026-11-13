"""Round 26, Task 1: a multi-architecture level-2 stacker (non-linear meta-models on the nine members' out-of-fold logits).

    python dm_r26_t1_l2stack.py

Pipeline: logistic regression on the 9 member logits (stack_binary), then one F2 cut-off. Earlier (5 members, old data) a LightGBM
meta-model gave +0.0005 [-0.0002, +0.0012], 4/5 folds, and was discarded by the CI gate; the round 25 rule (mean nested gain > 0 on
>= 3/5 folds) asks for it again on the current 9 members.
Meta-models, all on the same 9 logits, hyper-parameters fixed before the run (nothing is tuned on the scores below):
  lr     the pipeline's logistic regression
  lgb    LightGBM, 300 trees, learning rate 0.03, 8 leaves, min_child_samples 200, row / column subsample 0.8 (3 seeds averaged)
  dart   the same as DART (drop_rate 0.1)
  cat    CatBoost depth 4, 400 iterations, learning rate 0.05 (3 seeds)
  et     ExtraTrees, 400 trees, min_samples_leaf 50
  mlp    MLP (32, 16), early stopping, standardised logits (3 seeds)
Level-2 outputs are averaged in logit space: lr+lgb, lr+lgb+cat, all six. Everything nested: the meta-model is fitted on 4 folds'
OOF and predicts the 5th; the cut-off is chosen on 4 folds and applied to the 5th. Repeated on 3 other stacker splits (members fixed).
"""
import warnings

import lightgbm as lgb
import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common  # noqa: F401  (chdir to the repo root)
from common import load_t1  # noqa: E402
from src.metrics import f2, optimize_threshold  # noqa: E402
from src.utils import load_preds  # noqa: E402

warnings.filterwarnings("ignore")
_, _, _, y, _, folds, _ = load_t1()
y = np.asarray(y)
NAMES = ["t1_lgb", "t1_xgb", "t1_cat", "t1_et", "t1_lr", "t1_rf24", "t1_lgb24", "t1_xgb24", "t1_cat24"]
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
S = np.column_stack([logit(load_preds(k)["oof"]) for k in NAMES])
LGBP = dict(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            n_jobs=8, verbose=-1)
SEEDS = (0, 1, 2)
META = {
    "lr": lambda: [LogisticRegression(C=1.0, max_iter=2000)],
    "lgb": lambda: [lgb.LGBMClassifier(**LGBP, random_state=s) for s in SEEDS],
    "dart": lambda: [lgb.LGBMClassifier(**LGBP, boosting_type="dart", drop_rate=0.1, random_state=s) for s in SEEDS],
    "cat": lambda: [CatBoostClassifier(depth=4, iterations=400, learning_rate=0.05, random_seed=s, verbose=0, thread_count=8) for s in SEEDS],
    "et": lambda: [ExtraTreesClassifier(400, min_samples_leaf=50, n_jobs=8, random_state=0)],
    "mlp": lambda: [make_pipeline(StandardScaler(), MLPClassifier((32, 16), early_stopping=True, max_iter=200, random_state=s)) for s in SEEDS],
}
COMBOS = {"lr+lgb": ["lr", "lgb"], "lr+lgb+cat": ["lr", "lgb", "cat"], "all six": list(META)}


def meta_oof(fl):
    out = {k: np.zeros(len(y)) for k in META}
    for a, b in fl:
        for k, make in META.items():
            out[k][b] = np.mean([logit(m.fit(S[a], y[a]).predict_proba(S[b])[:, 1]) for m in make()], 0)
    for k, parts in COMBOS.items():
        out[k] = np.mean([out[p] for p in parts], 0)
    return {k: 1 / (1 + np.exp(-v)) for k, v in out.items()}


def nested_cut(p, fold_of):
    o = np.zeros(len(y), int)
    for i in range(5):
        tr = fold_of != i
        o[~tr] = p[~tr] >= optimize_threshold(y[tr], p[tr], 2.0)[0]
    return o


def fold_index(fl):
    fo = np.zeros(len(y), int)
    for i, (_, b) in enumerate(fl):
        fo[b] = i
    return fo


rng = np.random.default_rng(0)
boot = [rng.integers(0, len(y), len(y)) for _ in range(1000)]
print(f"{len(y)} rows, 9 members; F2 with a nested cut-off")
splits = [("pipeline folds", folds)] + [(f"split seed {s}", list(StratifiedKFold(5, shuffle=True, random_state=s).split(S, y))) for s in (1, 2, 3)]
summary = {}
for label, fl in splits:
    fo = fold_index(fl)
    P = meta_oof(fl)
    lab = {k: nested_cut(p, fo) for k, p in P.items()}
    base = lab["lr"]
    print(f"\n{label}: logistic stacker F2 {f2(y, base):.4f}")
    for k in [k for k in P if k != "lr"]:
        d = f2(y, lab[k]) - f2(y, base)
        pf = [f2(y[fo == i], lab[k][fo == i]) - f2(y[fo == i], base[fo == i]) for i in range(5)]
        bs = [f2(y[i], lab[k][i]) - f2(y[i], base[i]) for i in boot]
        summary.setdefault(k, []).append((d, sum(x > 0 for x in pf)))
        print(f"  {k:11s} F2 {f2(y, lab[k]):.4f}  delta {d:+.4f} [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}]  folds +{sum(x > 0 for x in pf)}/5 "
              f"({' '.join(f'{x:+.4f}' for x in pf)})  labels changed {np.mean(lab[k] != base):.3%}", flush=True)
print("\nmean delta over the 4 splits (folds better, per split):")
for k, v in summary.items():
    print(f"  {k:11s} {np.mean([d for d, _ in v]):+.4f}  ({' '.join(str(n) + '/5' for _, n in v)})")
