"""Round 30, Task 1: where do the v20 stacker's errors sit? (training rows only, cached OOF predictions, CPU, 2 threads)

    python eh_r30_t1_forensics.py

The v20 level 2 is rebuilt from the cached member OOF predictions exactly as in scripts/nb_part2.py: logistic stacker on the nine
members' log-odds (OOF on the pipeline folds) and the cached tree meta-models (outputs/preds/t1_l2_trees.npz), final score =
mean of the three log-odds; cut-off = the F2-optimal one on the OOF scores (descriptive, so in-sample is fine here).
Then, on the OOF labels:
  1. error rates by sub-population: number of missing cells, account age, level, missing key columns, and quantile bins of the
     mechanical-skill, reaction-time and win-rate columns (recall / precision / false-positive rate per bin);
  2. aim x reaction and aim x win-rate grids (3 x 3 quantile bins): recall and false-positive rate per cell;
  3. are the errors predictable from the raw features? A LightGBM (5-fold CV, inside the cheaters for missed cheaters and inside the
     legitimate players for false alarms) predicts "this row is an error"; AUC near 0.5 = no structure left to exploit;
  4. how many errors are "confident" (score far from the cut-off), i.e. beyond the reach of any re-weighting of the same members.
No test data is read.
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import common  # noqa: F401  (chdir to the repo root)
from src import task1  # noqa: E402
from src.config import DATA_DIRS  # noqa: E402
from src.metrics import optimize_threshold  # noqa: E402
from src.modeling import make_folds  # noqa: E402
from src.utils import load_preds  # noqa: E402

t0 = time.time()
tr = pd.read_csv(DATA_DIRS[1] / "train.csv")
tr = tr.dropna(subset=[task1.TARGET]).reset_index(drop=True)          # as task1.load(), without reading test.csv
y = tr[task1.TARGET].astype(int).values
folds = make_folds(y)
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
KEYS = ["lgb", "xgb", "cat", "et", "lr", "rf24", "lgb24", "xgb24", "cat24"]
S = np.column_stack([logit(load_preds(f"t1_{k}")["oof"]) for k in KEYS])
lr_oof = np.zeros(len(y))
for a, b in folds:
    lr_oof[b] = LogisticRegression(C=1.0, max_iter=2000).fit(S[a], y[a]).predict_proba(S[b])[:, 1]
score = expit((logit(lr_oof) + 2 * load_preds("t1_l2_trees")["oof"]) / 3)
thr, f2_best = optimize_threshold(y, score, 2.0)
pred = (score >= thr).astype(int)
tp, fp, fn = ((y == 1) & (pred == 1)).sum(), ((y == 0) & (pred == 1)).sum(), ((y == 1) & (pred == 0)).sum()
print(f"{len(y)} labelled training rows, cheaters {y.mean():.3f}; v20 level 2 OOF F2 {f2_best:.4f} at cut-off {thr:.3f}")
print(f"  flagged {pred.mean():.3f}; recall {tp / (tp + fn):.4f}; precision {tp / (tp + fp):.4f}; missed cheaters {fn}, false alarms {fp}\n")

raw = [c for c in tr.columns if c not in ("player_id", "id", task1.TARGET)]
X = tr[raw]
nan_n = X.isna().sum(1).values


def table(name, groups):
    rows = []
    for g, m in groups:
        m = np.asarray(m, bool)
        if m.sum() < 200:
            continue
        yy, pp = y[m], pred[m]
        tp_, fp_, fn_ = ((yy == 1) & (pp == 1)).sum(), ((yy == 0) & (pp == 1)).sum(), ((yy == 1) & (pp == 0)).sum()
        rows.append([g, m.sum(), yy.mean(), tp_ / max(tp_ + fn_, 1), tp_ / max(tp_ + fp_, 1), fp_ / max((yy == 0).sum(), 1),
                     5 * tp_ / max(5 * tp_ + 4 * fn_ + fp_, 1), fn_ / fn, fp_ / fp])
    df = pd.DataFrame(rows, columns=[name, "rows", "cheater rate", "recall", "precision", "FP rate", "F2 in group",
                                     "share of misses", "share of false alarms"]).set_index(name)
    print(df.round(3).to_string(), "\n")


table("missing cells", [(f"{a}" if a == b else f"{a}-{b}", (nan_n >= a) & (nan_n <= b)) for a, b in [(0, 0), (1, 1), (2, 2), (3, 4), (5, 7), (8, 99)]])
for col in ["account_age_days", "level", "accuracy_score", "headshot_percentage", "reaction_time_ms", "win_rate", "kill_death_ratio",
            "aiming_smoothness", "reports_received"]:
    if col not in X:
        continue
    v = X[col]
    qs = np.nanquantile(v, [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(v.fillna(-1e18), qs) if v.notna().all() else np.where(v.isna(), -1, np.digitize(v.fillna(0), qs))
    table(col, [("missing", v.isna())] + [(f"q{k + 1}/5", (bins == k) & v.notna()) for k in range(5)])

# key columns missing or not
for col in ["account_age_days", "avg_session_length_min", "headshot_percentage", "reaction_time_ms", "accuracy_score"]:
    if col in X:
        table(f"{col} missing", [("yes", X[col].isna()), ("no", X[col].notna())])


def grid(cx, cy):
    vx, vy = X[cx], X[cy]
    ok = vx.notna() & vy.notna()
    qx, qy = np.nanquantile(vx, [1 / 3, 2 / 3]), np.nanquantile(vy, [1 / 3, 2 / 3])
    bx, by = np.digitize(vx.fillna(0), qx), np.digitize(vy.fillna(0), qy)
    print(f"{cx} (rows, low -> high) x {cy} (columns): cheater rate | recall | FP rate | rows")
    for i in range(3):
        cells = []
        for j in range(3):
            m = ok & (bx == i) & (by == j)
            yy, pp = y[m], pred[m]
            rec = ((yy == 1) & (pp == 1)).sum() / max((yy == 1).sum(), 1)
            fpr = ((yy == 0) & (pp == 1)).sum() / max((yy == 0).sum(), 1)
            cells.append(f"{yy.mean():.2f} | {rec:.3f} | {fpr:.3f} | {m.sum():5d}")
        print("   " + "    ".join(cells))
    print()


grid("accuracy_score", "reaction_time_ms")
grid("accuracy_score", "win_rate")
grid("headshot_percentage", "kill_death_ratio")

# 3. are the errors predictable from the raw features?
P = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.8, n_jobs=2, verbose=-1)
Xf = X.copy()
Xf["n_missing"] = nan_n
Xf["score"] = score
for label, inside, err in [("missed cheaters among cheaters", y == 1, pred == 0), ("false alarms among legitimate players", y == 0, pred == 1)]:
    Xi, ei = Xf[inside].reset_index(drop=True), err[inside].astype(int)
    for use_score in (False, True):
        cols = [c for c in Xi.columns if use_score or c != "score"]
        o = np.zeros(len(ei))
        imp = np.zeros(len(cols))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(Xi, ei):
            m = lgb.LGBMClassifier(**P).fit(Xi.iloc[a][cols], ei[a])
            o[b] = m.predict_proba(Xi.iloc[b][cols])[:, 1]
            imp += m.booster_.feature_importance("gain")
        top = ", ".join(f"{cols[k]}" for k in np.argsort(-imp)[:6])
        print(f"{label} ({ei.sum()} of {len(ei)}): predictable from the features{' + the stacker score' if use_score else ''}: "
              f"AUC {roc_auc_score(ei, o):.3f}; top gain: {top}")
print()

# 4. distance of the errors from the cut-off (log-odds)
lo = logit(score) - logit(np.array([thr]))[0]
for label, m in [("missed cheaters", (y == 1) & (pred == 0)), ("false alarms", (y == 0) & (pred == 1))]:
    d = np.abs(lo[m])
    print(f"{label}: |log-odds - cut-off| quantiles 25/50/75/90%: " + " / ".join(f"{q:.2f}" for q in np.quantile(d, [.25, .5, .75, .9]))
          + f"; within 0.5 of the cut-off {np.mean(d < 0.5):.1%}, beyond 2 {np.mean(d > 2):.1%}")
print(f"[{time.time() - t0:.0f}s]")
