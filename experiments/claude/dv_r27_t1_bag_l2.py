"""Round 27 (overnight queue 1), Task 1: the 10-seed bag of round 14 under the v18 level-2 stacker (LogReg + LightGBM + CatBoost).

    python dv_r27_t1_bag_l2.py

Round 14 bagged eight of the nine members over 10 seeds (outputs/r14_bag/t1_<member>_<seed>.npz: oof, test, tl) and scored the bag
under the logistic stacker: +0.00039 [-0.00010, +0.00089], 3/5 folds, test-like -0.00038 -> discarded. The tree meta-models of v18
split on the member log-odds, so smoother inputs could matter more now. Member sets under the v18 level 2 (meta-models and cut-off
nested; pipeline folds + three other stacker splits):
  pipeline   the cached members (1 seed for the 49-feature members, 2 seeds for the 24-column members)
  bag 3 / 5 / 10   log-odds mean over the first 3 / 5 / 10 seeds (LogReg is deterministic)
Test-like check (round 12): the same validation rows with test-like missingness ("tl" predictions) and rows re-weighted to the
test's missing-cell rates; level 2 fitted on the OOF log-odds, applied to the tl log-odds, cut-off from the OOF.
Acceptance: mean nested gain > 0 and >= 4/5 folds on the splits, and test-like gain >= 0.
"""
import warnings

import lightgbm as lgb
import numpy as np
from catboost import CatBoostClassifier
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import common  # noqa: F401  (chdir to the repo root)
from common import OUT, load_t1  # noqa: E402
from src.metrics import f2, optimize_threshold  # noqa: E402
from src.utils import load_preds  # noqa: E402

warnings.filterwarnings("ignore")
tr, te, _, y, _, folds, _ = load_t1()
y = np.asarray(y)
SEEDS = (42, 100, 2026, 777, 999, 1234, 5555, 8888, 31415, 27182)
KEYS = ["lgb", "xgb", "cat", "et", "lr", "rf24", "lgb24", "xgb24", "cat24"]
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
DIR = OUT / "r14_bag"
Z = np.load(OUT / "bn_t1_members.npz")
avg2 = lambda k, s: (Z[f"{k}_42_{s}"] + Z[f"{k}_7_{s}"]) / 2
pipe_oof = [logit(load_preds(f"t1_{k}")["oof"]) for k in KEYS]
z_oof = {**{k: Z[f"{k}_oof"] for k in ("lgb", "xgb", "cat", "et", "lr")}, **{k: avg2(f"their_{k[:-2]}", "oof") for k in ("rf24", "lgb24", "xgb24", "cat24")}}
z_tl = {**{k: Z[f"{k}_tl"] for k in ("lgb", "xgb", "cat", "et", "lr")}, **{k: avg2(f"their_{k[:-2]}", "tl") for k in ("rf24", "lgb24", "xgb24", "cat24")}}


def bag(n, field):
    out = {k: np.mean([logit(np.load(DIR / f"t1_{k}_{s}.npz")[field]) for s in SEEDS[:n]], axis=0) for k in KEYS if k != "lr"}
    out["lr"] = logit(load_preds("t1_lr")["oof"] if field == "oof" else Z["lr_tl"])
    return [out[k] for k in KEYS]


# test-like row weights (round 12): re-weight rows to the test's missing-cell rate in every column where test has fewer missing cells
raw = [c for c in tr.columns if c not in ("player_id", "id", "is_cheater")]
lab = tr["is_cheater"].notna().values
q_tr, q_te = tr.loc[lab, raw].isna().mean(), te[raw].isna().mean()
w_tl = np.ones(len(y))
for c in [c for c in raw if q_te[c] < q_tr[c]]:
    m = tr.loc[lab, c].isna().values
    w_tl *= np.where(m, q_te[c] / q_tr[c], (1 - q_te[c]) / (1 - q_tr[c]))


def wf2(pred, w, idx=None):
    idx = slice(None) if idx is None else idx
    yy, pp, ww = y[idx], pred[idx], w[idx]
    tp, fp, fn = ww @ ((yy == 1) & (pp == 1)), ww @ ((yy == 0) & (pp == 1)), ww @ ((yy == 1) & (pp == 0))
    return 5 * tp / (5 * tp + 4 * fn + fp)


LGBP = dict(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            n_jobs=6, verbose=-1)
MAKE = [lambda: [LogisticRegression(C=1.0, max_iter=2000)],
        lambda: [lgb.LGBMClassifier(**LGBP, random_state=s) for s in (0, 1, 2)],
        lambda: [CatBoostClassifier(depth=4, iterations=400, learning_rate=0.05, random_seed=s, verbose=0, thread_count=6) for s in (0, 1, 2)]]


def level2(cols, fl, cols_eval=None):
    """Nested labels: level 2 on 4 folds -> 5th, cut-off from the 4 folds' own out-of-fold level-2 scores. With cols_eval the
    fitted level 2 is also applied to those (test-like) inputs of the 5th fold."""
    S = np.column_stack(cols)
    Se = None if cols_eval is None else np.column_stack(cols_eval)
    p, pe = np.zeros(len(y)), np.zeros(len(y))
    for a, b in fl:
        zs, zes = [], []
        for mk in MAKE:
            ms = [m.fit(S[a], y[a]) for m in mk()]
            zs.append(np.mean([logit(m.predict_proba(S[b])[:, 1]) for m in ms], 0))
            if Se is not None:
                zes.append(np.mean([logit(m.predict_proba(Se[b])[:, 1]) for m in ms], 0))
        p[b] = expit(np.mean(zs, 0))
        if Se is not None:
            pe[b] = expit(np.mean(zes, 0))
    fo = np.zeros(len(y), int)
    for i, (_, b) in enumerate(fl):
        fo[b] = i
    lab_, lab_e = np.zeros(len(y), int), np.zeros(len(y), int)
    for i in range(5):
        t = optimize_threshold(y[fo != i], p[fo != i], 2.0)[0]
        lab_[fo == i] = p[fo == i] >= t
        lab_e[fo == i] = pe[fo == i] >= t
    return lab_, lab_e, fo


rng = np.random.default_rng(0)
boot = [rng.integers(0, len(y), len(y)) for _ in range(1000)]
splits = [("pipeline folds", folds)] + [(f"split seed {s}", list(StratifiedKFold(5, shuffle=True, random_state=s).split(pipe_oof[0], y))) for s in (1, 2, 3)]
SETS = {"bag 3": 3, "bag 5": 5, "bag 10": 10}
summary = {}
for label, fl in splits:
    base, _, fo = level2(pipe_oof, fl)
    _, base_tl, _ = level2([logit(z_oof[k]) for k in KEYS], fl, [logit(z_tl[k]) for k in KEYS])
    print(f"\n{label}: pipeline members under the v18 level 2: F2 {f2(y, base):.4f}; test-like {wf2(base_tl, w_tl):.4f}", flush=True)
    for name, n in SETS.items():
        p, p_tl, _ = level2(bag(n, "oof"), fl, bag(n, "tl"))
        d = f2(y, p) - f2(y, base)
        pf = [f2(y[fo == i], p[fo == i]) - f2(y[fo == i], base[fo == i]) for i in range(5)]
        bs = [f2(y[i], p[i]) - f2(y[i], base[i]) for i in boot]
        dtl = wf2(p_tl, w_tl) - wf2(base_tl, w_tl)
        bt = [wf2(p_tl, w_tl, i) - wf2(base_tl, w_tl, i) for i in boot]
        summary.setdefault(name, []).append((d, sum(x > 0 for x in pf), dtl))
        print(f"  {name:7s} F2 {f2(y, p):.4f}  delta {d:+.4f} [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}]  folds +{sum(x > 0 for x in pf)}/5 "
              f"({' '.join(f'{x:+.4f}' for x in pf)})  test-like {dtl:+.4f} [{np.quantile(bt, .025):+.4f}, {np.quantile(bt, .975):+.4f}]  labels changed {np.mean(p != base):.3%}", flush=True)
print("\nmean over the 4 splits: delta | folds better per split | test-like delta")
for k, v in summary.items():
    print(f"  {k:7s} {np.mean([d for d, _, _ in v]):+.4f} | {' '.join(str(n) + '/5' for _, n, _ in v)} | {np.mean([t for _, _, t in v]):+.4f}")
