"""Round 27 (overnight queue 3), Task 2: seed bag of the three GBDT members under the v19 level 2 (stacker + KDE naive Bayes).

    BAG_DIR=r27_bag python by_r14_bag.py cv t2      # writes outputs/r27_bag/t2_<lgb|xgb|cat>_<seed>.npz (oof, test)
    python dw_r27_t2_bag_eval.py

The members are the pipeline's (features, hyper-parameters, folds unchanged); only the model seed varies. Bag = mean of the seeds'
probabilities per member; LogReg and the KDE naive Bayes are deterministic. Level 2 as in v19: LogReg stacker (C = 0.5) on the four
members' log-probabilities, 0.4 stacker + 0.6 KDE naive Bayes (the weight v19 chose on the pipeline members), class multipliers.
The stacker is out-of-fold and the multipliers are nested (4 folds -> 5th), on the pipeline folds and three other stacker splits.
Bags of the first 3 / 5 / 10 seeds that are complete on disk for all three members.
Acceptance: mean nested gain > 0 and >= 4/5 folds on the splits.
"""
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
import common  # noqa: F401,E402
from common import OUT  # noqa: E402
from src import task2  # noqa: E402
from src.metrics import apply_class_weights, optimize_class_weights  # noqa: E402
from src.modeling import make_folds  # noqa: E402

SEEDS = (42, 100, 2026, 777, 999, 1234, 5555, 8888, 31415, 27182)
DIR = OUT / "r27_bag"
KDE_W = 0.6
train, _ = task2.load()
y = train[task2.TARGET].values
folds = make_folds(y)
pipe = {k: np.load(f"outputs/preds/t2_{k}.npz")["oof"] for k in ("lgb", "xgb", "cat", "lr")}
kde = np.load("outputs/preds/t2_kde_nb.npz")["oof"]
done = [s for s in SEEDS if all((DIR / f"t2_{k}_{s}.npz").exists() for k in ("lgb", "xgb", "cat"))]
print(f"complete seeds: {done}")
mf1 = lambda yy, p: f1_score(yy, p, average="macro")
for k in ("lgb", "xgb", "cat"):
    one = [mf1(y, np.load(DIR / f"t2_{k}_{s}.npz")["oof"].argmax(1)) for s in done]
    print(f"  {k}: single-seed argmax macro-F1 {np.mean(one):.4f} +- {np.std(one):.4f}; pipeline cache {mf1(y, pipe[k].argmax(1)):.4f}; "
          f"{len(done)}-seed bag {mf1(y, np.mean([np.load(DIR / f't2_{k}_{s}.npz')['oof'] for s in done], 0).argmax(1)):.4f}")


def members(n):
    out = [np.mean([np.load(DIR / f"t2_{k}_{s}.npz")["oof"] for s in done[:n]], 0) for k in ("lgb", "xgb", "cat")]
    return out + [pipe["lr"]]


def level2(mem, fl, w_kde):
    S = np.column_stack([np.log(np.clip(m, 1e-6, 1)) for m in mem])
    P = np.zeros((len(y), 4))
    for a, b in fl:
        P[b] = LogisticRegression(C=0.5, max_iter=3000).fit(S[a], y[a]).predict_proba(S[b])
    P = (1 - w_kde) * P + w_kde * kde
    out = np.zeros(len(y), int)
    for a, b in fl:
        out[b] = apply_class_weights(P[b], optimize_class_weights(y[a], P[a])[0])
    return out


SETS = {"pipeline members": [pipe[k] for k in ("lgb", "xgb", "cat", "lr")]}
SETS.update({f"bag {n}": members(n) for n in (3, 5, 10) if n <= len(done)})
rng = np.random.default_rng(0)
boot = [rng.integers(0, len(y), len(y)) for _ in range(1000)]
splits = [("pipeline folds", folds)] + [(f"split seed {s}", list(StratifiedKFold(5, shuffle=True, random_state=s).split(kde, y))) for s in (1, 2, 3)]
summary = {}
for label, fl in splits:
    fo = np.zeros(len(y), int)
    for i, (_, b) in enumerate(fl):
        fo[b] = i
    for w_kde, tag in ((KDE_W, "v19 level 2 (0.4 stacker + 0.6 KDE NB)"), (0.0, "stacker alone")):
        lab = {k: level2(m, fl, w_kde) for k, m in SETS.items()}
        base = lab["pipeline members"]
        print(f"\n{label}, {tag}: pipeline members {mf1(y, base):.4f}", flush=True)
        for k in list(SETS)[1:]:
            d = mf1(y, lab[k]) - mf1(y, base)
            pf = [mf1(y[fo == i], lab[k][fo == i]) - mf1(y[fo == i], base[fo == i]) for i in range(5)]
            bs = [mf1(y[i], lab[k][i]) - mf1(y[i], base[i]) for i in boot]
            summary.setdefault((tag, k), []).append((d, sum(x > 0 for x in pf)))
            print(f"  {k:7s} {mf1(y, lab[k]):.4f}  delta {d:+.4f} [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}]  folds +{sum(x > 0 for x in pf)}/5 "
                  f"({' '.join(f'{x:+.4f}' for x in pf)})  labels changed {np.mean(lab[k] != base):.2%}", flush=True)
print("\nmean delta over the splits (folds better, per split):")
for (tag, k), v in summary.items():
    print(f"  {tag:40s} {k:7s} {np.mean([d for d, _ in v]):+.4f}  ({' '.join(str(n) + '/5' for _, n in v)})")
