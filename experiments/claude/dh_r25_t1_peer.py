"""Round 25, Task 1: peer-group residual features (actual minus the value expected for a legitimate player with the same profile).

    python dh_r25_t1_peer.py

For each performance column t (headshot_percentage, kill_death_ratio, win_rate, accuracy_score, damage_per_round, reaction_time_ms):
a LightGBM regressor predicts t from the other raw columns, fitted ONLY on legitimate players (is_cheater = 0) of the training part
of each pipeline fold. Legitimate training rows get inner 5-fold out-of-fold expectations; cheater training rows, validation rows and
test rows get the mean of the five inner models (so no row is predicted by a model that saw it). New features: residual = actual -
expected, and residual / MAD of the legitimate residuals. Everything is fitted inside the fold (run_cv fold_fn): validation labels
and test rows never touch the regressors.
Member: the pipeline's tuned LightGBM on the 49 features + 12 residual features. Scored alone and in the stack (9 pipeline members
from the caches; nested logistic stacker and nested cut-off), as a 10th member and as a replacement for LightGBM.
Acceptance (round 25): mean nested gain > 0 and >= 3/5 folds; the CI is reported, not used as a gate.
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import common  # noqa: F401  (chdir to the repo root)
from common import load_t1, stack_binary, tuned  # noqa: E402
from src import task1  # noqa: E402
from src.metrics import f2, optimize_threshold  # noqa: E402
from src.modeling import run_cv  # noqa: E402
from src.utils import load_preds  # noqa: E402

OUT = Path("experiments/claude/outputs/r25")
OUT.mkdir(parents=True, exist_ok=True)
tr_df, te_df, X, y, Xt, folds, _ = load_t1()
RAW = [c for c in tr_df.columns if c not in ("player_id", "id", task1.TARGET) and c in X.columns]
TARGETS = ["headshot_percentage", "kill_death_ratio", "win_rate", "accuracy_score", "damage_per_round", "reaction_time_ms"]
spw = float((y == 0).sum() / (y == 1).sum())
P1_lgb = tuned("task1_lgb", {"learning_rate": 0.05, "scale_pos_weight": spw})
REG = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.7, objective="regression_l1", random_state=0, n_jobs=8, verbose=-1)
print(f"{len(y)} rows, cheaters {y.mean():.3f}; {len(RAW)} raw columns; targets {TARGETS}", flush=True)


def peer_fold_fn(X_tr, y_tr, X_others):
    X_tr = X_tr.copy()
    X_others = [x.copy() for x in X_others]
    legit = np.where(np.asarray(y_tr) == 0)[0]
    for t in TARGETS:
        inp = [c for c in RAW if c != t]
        ok = legit[X_tr[t].iloc[legit].notna().values]
        exp_tr = np.zeros(len(X_tr))
        exp_oth = [np.zeros(len(x)) for x in X_others]
        oof_mask = np.zeros(len(X_tr), bool)
        for a, b in KFold(5, shuffle=True, random_state=0).split(ok):
            m = lgb.LGBMRegressor(**REG).fit(X_tr[inp].iloc[ok[a]], X_tr[t].iloc[ok[a]])
            exp_tr[ok[b]] = m.predict(X_tr[inp].iloc[ok[b]])
            oof_mask[ok[b]] = True
            rest = ~np.isin(np.arange(len(X_tr)), ok)
            exp_tr[rest] += m.predict(X_tr[inp].iloc[rest]) / 5
            for e, x in zip(exp_oth, X_others):
                e += m.predict(x[inp]) / 5
        res_tr = X_tr[t].values - exp_tr
        scale = 1.4826 * np.nanmedian(np.abs(res_tr[oof_mask] - np.nanmedian(res_tr[oof_mask]))) + 1e-9
        X_tr[f"{t}_peer_res"], X_tr[f"{t}_peer_z"] = res_tr, res_tr / scale
        for e, x in zip(exp_oth, X_others):
            r = x[t].values - e
            x[f"{t}_peer_res"], x[f"{t}_peer_z"] = r, r / scale
    return X_tr, X_others


f = OUT / "t1_lgb_peer.npz"
if f.exists():
    z = np.load(f)
    new = {"oof": z["oof"], "test": z["test"]}
else:
    r = run_cv("lgb", "binary", P1_lgb, X, y, Xt, folds, fold_fn=peer_fold_fn, name="T1 lgb + peer residuals")
    new = {"oof": r["oof"], "test": r["test"]}
    np.savez(f, **new)
    print("\ntop features:", r["importance"].head(15).round(0).to_dict(), flush=True)

NAMES = ["t1_lgb", "t1_xgb", "t1_cat", "t1_et", "t1_lr", "t1_rf24", "t1_lgb24", "t1_xgb24", "t1_cat24"]
M = {k: load_preds(k) for k in NAMES}
fold_of = np.zeros(len(y), int)
for i, (_, va) in enumerate(folds):
    fold_of[va] = i


def nested_cut(p):
    out = np.zeros(len(y), int)
    for i in range(5):
        tr = fold_of != i
        thr, _ = optimize_threshold(y[tr], p[tr], 2.0)
        out[~tr] = (p[~tr] >= thr).astype(int)
    return out


def stack_pred(members):
    so, _ = stack_binary([m["oof"] for m in members], [m["test"] for m in members], y, folds)
    return so, nested_cut(so)


f2b = lambda yt, yp: f2(yt, yp)
print(f"alone (best cut): pipeline LightGBM {optimize_threshold(y, M['t1_lgb']['oof'], 2.0)[1]:.4f} | + peer residuals {optimize_threshold(y, new['oof'], 2.0)[1]:.4f} "
      f"| corr of the two OOFs {np.corrcoef(M['t1_lgb']['oof'], new['oof'])[0, 1]:.4f}")
base_so, base = stack_pred(list(M.values()))
rng = np.random.default_rng(0)
boot = [rng.integers(0, len(y), len(y)) for _ in range(1000)]
print(f"pipeline stack (9 members), nested cut: F2 {f2b(y, base):.4f}")
for name, members in [("10 members (+ peer LGB)", list(M.values()) + [new]),
                      ("9 members (peer LGB replaces LGB)", [new] + [M[k] for k in NAMES[1:]])]:
    so, p = stack_pred(members)
    d = f2b(y, p) - f2b(y, base)
    pf = [f2b(y[fold_of == i], p[fold_of == i]) - f2b(y[fold_of == i], base[fold_of == i]) for i in range(5)]
    bs = [f2b(y[i], p[i]) - f2b(y[i], base[i]) for i in boot]
    print(f"  {name:36s} F2 {f2b(y, p):.4f}  delta {d:+.4f} [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}]  folds +{sum(x > 0 for x in pf)}/5 "
          f"({' '.join(f'{x:+.4f}' for x in pf)})  labels changed {np.mean(p != base):.3%}")
