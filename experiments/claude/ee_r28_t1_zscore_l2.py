"""Round 28, Task 1: standardise the three level-2 log-odds (LogReg, LightGBM, CatBoost) before averaging them (CPU, 2 threads).

    python ee_r28_t1_zscore_l2.py

v18-v20 level 2: mean of the raw log-odds of the logistic stacker, the LightGBM meta-model (3 seeds) and the CatBoost meta-model
(3 seeds), all fitted on the nine cached member log-odds, then one F2 cut-off. The three log-odds have different spreads, so the
plain mean weights them by their SD. Here each is standardised first, z_k = (logit_k - mean_k) / sd_k, and the mean of the z's is
mapped back to the scale of the plain mean (a monotone map, so only the relative weights change).
The pipeline caches only the mean of the two tree meta-models, so the three meta-models are re-fitted on the cached member log-odds
(outputs/preds/t1_*.npz; the same settings as scripts/nb_part2.py, 2 threads instead of 8). Nothing else is re-trained.
Nested as in round 26 / 27: on each split (pipeline folds, split seeds 1, 2, 3) the meta-models are fitted on 4 folds and predict
the 5th; the means / SDs and the cut-off come from the 4 other folds' out-of-fold level-2 scores. Test-like check (round 12): the
round-12 members' OOF log-odds fit the level 2, which is applied to their test-like-missingness ("tl") log-odds, with rows
re-weighted to the test's missing-cell rates; the cut-off and the standardisation come from the OOF scores.
Acceptance: mean gain > 0 and >= 4/5 folds on the splits, and test-like gain >= 0.
"""
import time
import warnings

import lightgbm as lgb
import numpy as np
import psutil
from catboost import CatBoostClassifier
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import common  # noqa: F401  (chdir to the repo root)
from common import OUT  # noqa: E402
from src import task1  # noqa: E402
from src.metrics import f2, optimize_threshold  # noqa: E402
from src.modeling import make_folds  # noqa: E402
from src.utils import load_preds  # noqa: E402

warnings.filterwarnings("ignore")
t0 = time.time()
tr, te, _ = task1.load()                       # raw frames only (no feature building): y, folds, missingness
y = tr[task1.TARGET].values
folds = make_folds(y)
KEYS = ["lgb", "xgb", "cat", "et", "lr", "rf24", "lgb24", "xgb24", "cat24"]
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
pipe_oof = [logit(load_preds(f"t1_{k}")["oof"]) for k in KEYS]
Z = np.load(OUT / "bn_t1_members.npz")
avg2 = lambda k, s: (Z[f"{k}_42_{s}"] + Z[f"{k}_7_{s}"]) / 2
z_oof = {**{k: Z[f"{k}_oof"] for k in ("lgb", "xgb", "cat", "et", "lr")}, **{k: avg2(f"their_{k[:-2]}", "oof") for k in ("rf24", "lgb24", "xgb24", "cat24")}}
z_tl = {**{k: Z[f"{k}_tl"] for k in ("lgb", "xgb", "cat", "et", "lr")}, **{k: avg2(f"their_{k[:-2]}", "tl") for k in ("rf24", "lgb24", "xgb24", "cat24")}}
raw = [c for c in tr.columns if c not in ("player_id", "id", "is_cheater")]
lab = tr["is_cheater"].notna().values
q_tr, q_te = tr.loc[lab, raw].isna().mean(), te[raw].isna().mean()
w_tl = np.ones(len(y))
for c in [c for c in raw if q_te[c] < q_tr[c]]:
    m = tr.loc[lab, c].isna().values
    w_tl *= np.where(m, q_te[c] / q_tr[c], (1 - q_te[c]) / (1 - q_tr[c]))
del tr, te


def wf2(pred, w, idx=None):
    idx = slice(None) if idx is None else idx
    yy, pp, ww = y[idx], pred[idx], w[idx]
    tp, fp, fn = ww @ ((yy == 1) & (pp == 1)), ww @ ((yy == 0) & (pp == 1)), ww @ ((yy == 1) & (pp == 0))
    return 5 * tp / (5 * tp + 4 * fn + fp)


LGBP = dict(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            n_jobs=2, verbose=-1)
MAKE = {"LogReg": lambda: [LogisticRegression(C=1.0, max_iter=2000)],
        "LightGBM": lambda: [lgb.LGBMClassifier(**LGBP, random_state=s) for s in (0, 1, 2)],
        "CatBoost": lambda: [CatBoostClassifier(depth=4, iterations=400, learning_rate=0.05, random_seed=s, verbose=0, thread_count=2,
                                                allow_writing_files=False) for s in (0, 1, 2)]}


def meta_logits(cols, fl, cols_eval=None):
    """Out-of-fold log-odds of each meta-model (seed-mean), shape (3, n); with cols_eval also the fitted models applied to those."""
    S = np.column_stack(cols)
    Se = None if cols_eval is None else np.column_stack(cols_eval)
    L, Le = np.zeros((3, len(y))), np.zeros((3, len(y)))
    for a, b in fl:
        for j, mk in enumerate(MAKE.values()):
            ms = [m.fit(S[a], y[a]) for m in mk()]
            L[j, b] = np.mean([logit(m.predict_proba(S[b])[:, 1]) for m in ms], 0)
            if Se is not None:
                Le[j, b] = np.mean([logit(m.predict_proba(Se[b])[:, 1]) for m in ms], 0)
    return L, Le


def labels(L, fo, Le=None, z=False):
    """Nested labels of the plain mean (z=False) or the standardised mean (z=True) of the three log-odds."""
    out, out_e = np.zeros(len(y), int), np.zeros(len(y), int)
    for i in range(5):
        t_, v_ = fo != i, fo == i
        if z:
            mu, sd = L[:, t_].mean(1, keepdims=True), L[:, t_].std(1, keepdims=True)
            plain = L[:, t_].mean(0)
            comb = lambda A: ((A - mu) / sd).mean(0) * plain.std() + plain.mean()
        else:
            comb = lambda A: A.mean(0)
        c_tr = comb(L[:, t_])
        thr = optimize_threshold(y[t_], expit(c_tr), 2.0)[0]
        out[v_] = expit(comb(L[:, v_])) >= thr
        if Le is not None:
            out_e[v_] = expit(comb(Le[:, v_])) >= thr
    return out, out_e


def boots(n, B):
    """The same B bootstrap draws for every candidate, generated one at a time (no B x n index matrix in memory)."""
    r = np.random.default_rng(0)
    for _ in range(B):
        yield r.integers(0, n, n).astype(np.int32)


splits = [("pipeline folds", folds)] + [(f"split seed {s}", list(StratifiedKFold(5, shuffle=True, random_state=s).split(pipe_oof[0], y))) for s in (1, 2, 3)]
summary = []
print(f"{len(y)} rows; nine cached members; level 2 = LogReg + LightGBM + CatBoost; F2 with a nested cut-off", flush=True)
for label, fl in splits:
    fo = np.zeros(len(y), int)
    for i, (_, b) in enumerate(fl):
        fo[b] = i
    L, _ = meta_logits(pipe_oof, fl)
    Lz, Lz_tl = meta_logits([logit(z_oof[k]) for k in KEYS], fl, [logit(z_tl[k]) for k in KEYS])
    base, _ = labels(L, fo)
    cand, _ = labels(L, fo, z=True)
    _, base_tl = labels(Lz, fo, Lz_tl)
    _, cand_tl = labels(Lz, fo, Lz_tl, z=True)
    sd = L.std(1)
    w_eff = (1 / sd) / (1 / sd).sum()
    d = f2(y, cand) - f2(y, base)
    pf = [f2(y[fo == i], cand[fo == i]) - f2(y[fo == i], base[fo == i]) for i in range(5)]
    bs = [f2(y[b], cand[b]) - f2(y[b], base[b]) for b in boots(len(y), 1000)]
    dtl = wf2(cand_tl, w_tl) - wf2(base_tl, w_tl)
    bt = [wf2(cand_tl, w_tl, b) - wf2(base_tl, w_tl, b) for b in boots(len(y), 1000)]
    summary.append((d, sum(x > 0 for x in pf), dtl))
    print(f"\n{label}: SD of the OOF log-odds LogReg / LightGBM / CatBoost {sd[0]:.3f} / {sd[1]:.3f} / {sd[2]:.3f} "
          f"(standardising moves the weights from 1/3 each to {w_eff[0]:.3f} / {w_eff[1]:.3f} / {w_eff[2]:.3f}); "
          f"correlations {np.corrcoef(L)[0, 1]:.4f} / {np.corrcoef(L)[0, 2]:.4f} / {np.corrcoef(L)[1, 2]:.4f}")
    print(f"  plain mean F2 {f2(y, base):.4f}; standardised mean F2 {f2(y, cand):.4f}  delta {d:+.4f} [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}]  "
          f"folds +{sum(x > 0 for x in pf)}/5 ({' '.join(f'{x:+.4f}' for x in pf)})  labels changed {np.mean(cand != base):.3%}")
    print(f"  test-like: plain {wf2(base_tl, w_tl):.4f}, standardised {wf2(cand_tl, w_tl):.4f}  delta {dtl:+.4f} [{np.quantile(bt, .025):+.4f}, {np.quantile(bt, .975):+.4f}]"
          f"  [{time.time() - t0:.0f}s, peak working set {psutil.Process().memory_info().peak_wset / 2**20:.0f} MB]", flush=True)
print(f"\nmean over the 4 splits: delta {np.mean([s[0] for s in summary]):+.4f} | folds better per split {' '.join(str(s[1]) + '/5' for s in summary)} | "
      f"test-like delta {np.mean([s[2] for s in summary]):+.4f}")
print(f"peak working set {psutil.Process().memory_info().peak_wset / 2**20:.0f} MB; {time.time() - t0:.0f}s")
