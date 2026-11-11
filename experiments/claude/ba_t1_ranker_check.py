"""Task 1: is the pairwise-ranking member (az_t1_research.py, Q2: +0.0009 F2, CI [+0.0002, +0.0015]) real and test-relevant?

1. Seed stability: the same XGBoost rank:pairwise member with seeds 42 (as in Q2) and 7.
2. Test relevance: the stack gain reweighted to the test set's missing-value rates (missingness is random and label-free;
   nothing is fitted on test rows), and on rows that have all four key columns (78% of test rows, 45% of train rows).
3. If it holds: test predictions (5-fold average + Train+Val refit, as the notebook does) are saved for integration.
Saves outputs/t1_ranker.npz.
"""
import time
import warnings

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
import common  # noqa: E402
from common import load_t1, T1_MODELS, OUT, _logit  # noqa: E402
from src.utils import load_preds  # noqa: E402
from src.metrics import optimize_threshold, fbeta  # noqa: E402

T0 = time.time()
log = lambda *a: print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
rng = np.random.default_rng(0)
tr, te, X, y, Xt, folds, _ = load_t1()
base = {n: load_preds(n) for n in T1_MODELS}
f2 = lambda yt, yp: fbeta(yt, yp, 2, "binary")
PRM = {"objective": "rank:pairwise", "eta": 0.05, "max_depth": 6, "min_child_weight": 10, "subsample": 0.8,
       "colsample_bytree": 0.6, "eval_metric": "auc", "tree_method": "hist", "lambdarank_pair_method": "mean",
       "lambdarank_num_pair_per_sample": 8}
dte = xgb.DMatrix(Xt)


def ranker_cv(seed, with_test):
    cache = OUT / f"t1_ranker_seed{seed}.npz"
    if cache.exists():
        c = np.load(cache)
        log(f"  seed {seed}: loaded from {cache.name}")
        return c["oof"], c["test"], list(c["iters"])
    oof, test, iters = np.zeros(len(y)), np.zeros(len(Xt)), []
    for a, b in folds:
        dtr = xgb.DMatrix(X.iloc[a], label=y[a])
        dtr.set_group([len(a)])
        dva = xgb.DMatrix(X.iloc[b], label=y[b])
        dva.set_group([len(b)])
        bst = xgb.train({**PRM, "seed": seed}, dtr, 5000, evals=[(dva, "va")], early_stopping_rounds=200, verbose_eval=False)
        rng_ = (0, bst.best_iteration + 1)
        s = bst.predict(dva, iteration_range=rng_)
        mu, sd = s.mean(), s.std()
        oof[b] = 1 / (1 + np.exp(-(s - mu) / sd))
        if with_test:
            test += 1 / (1 + np.exp(-(bst.predict(dte, iteration_range=rng_) - mu) / sd)) / len(folds)
        iters.append(bst.best_iteration + 1)
    log(f"  seed {seed}: member AUC {roc_auc_score(y, oof):.4f}, best rounds {iters}")
    np.savez(cache, oof=oof, test=test, iters=np.array(iters))
    return oof, test, iters


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


def wf2(pred, w):
    tp = w @ ((y == 1) & (pred == 1)); fp = w @ ((y == 0) & (pred == 1)); fn = w @ ((y == 1) & (pred == 0))
    return 5 * tp / (5 * tp + 4 * fn + fp)


def boot(pn, po, w, B=1000):
    d = []
    for _ in range(B):
        pw = w * rng.poisson(1.0, len(y))
        d.append(wf2(pn, pw) - wf2(po, pw))
    return np.percentile(d, [2.5, 97.5])


imp = ["reaction_time_ms", "account_age_days", "level", "level_progression_speed", "friend_network_size", "reports_received",
       "aiming_smoothness", "spray_control_score", "crosshair_placement", "headshot_percentage"]
w_test = np.ones(len(y))
for c in imp:
    qtr, qte = tr[c].isna().mean(), te[c].isna().mean()
    w_test *= np.where(tr[c].isna().values, qte / qtr, (1 - qte) / (1 - qtr))
key = tr[["account_age_days", "friend_network_size", "reports_received", "crosshair_placement"]].notna().all(1).values
ones = np.ones(len(y))

pred0 = nested(stack([base[n]["oof"] for n in T1_MODELS]))
KEEP = False


def evaluate(name, extra):
    global KEEP
    pn = nested(stack([base[n]["oof"] for n in T1_MODELS] + extra))
    d_all, ci_all = wf2(pn, ones) - wf2(pred0, ones), boot(pn, pred0, ones)
    d_tst, ci_tst = wf2(pn, w_test) - wf2(pred0, w_test), boot(pn, pred0, w_test)
    d_key = wf2(pn, key.astype(float)) - wf2(pred0, key.astype(float))
    log(f"{name:20s}: OOF {d_all:+.4f} CI [{ci_all[0]:+.4f}, {ci_all[1]:+.4f}] | test-weighted {d_tst:+.4f} "
        f"CI [{ci_tst[0]:+.4f}, {ci_tst[1]:+.4f}] | rows with key columns {d_key:+.4f}")
    KEEP = KEEP or ci_tst[0] > 0


o7, _, _ = ranker_cv(7, False)
evaluate("seed 7", [o7])                       # the missing piece: reported as soon as it is ready
o42, t42, it42 = ranker_cv(42, True)
evaluate("seed 42", [o42])
evaluate("both seeds averaged", [(o42 + o7) / 2])

# Train+Val refit for the test input (rounds from fold 0 x 1.15, as modeling.refit_full) -- only worth it if the gain holds
if not KEEP:
    log("no variant passes the test-weighted check -> no refit; Task 1 stays unchanged")
    raise SystemExit(0)
n_iter = max(50, int(round(it42[0] * 1.15)))
dall = xgb.DMatrix(X, label=y)
dall.set_group([len(y)])
bst = xgb.train({**PRM, "seed": 42}, dall, n_iter, verbose_eval=False)
s_all = bst.predict(xgb.DMatrix(X))
s_te = bst.predict(dte)
t_full = 1 / (1 + np.exp(-(s_te - s_all.mean()) / s_all.std()))
np.savez(OUT / "t1_ranker.npz", oof42=o42, oof7=o7, test_fold42=t42, test_full42=t_full, iters42=np.array(it42), n_iter_full=n_iter)
log(f"saved outputs/t1_ranker.npz (full refit {n_iter} rounds)")
