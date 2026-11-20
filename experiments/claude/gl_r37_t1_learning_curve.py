"""Round 37, Task 1: learning curve and calibration check (training rows only, CPU).

    python gl_r37_t1_learning_curve.py

Question (user, 2026-11-17): is Task 1 limited by the data (flat curve, errors explained by the model's own uncertainty) or by the
learner (a curve still rising would call for more capacity)?
1. Learning curve: for each pipeline fold the labelled training rows are subsampled (stratified, seed fixed) to 12.5 / 25 / 50 /
   75 / 100%; a LightGBM with the pipeline's tuned parameters (task1_lgb, learning rate halved as in the pipeline's final fits) on
   the 49 pipeline features is fitted on the subsample (early stopping on 10% of it, never on the held-out fold) and scored on the
   full held-out fold. Metrics on the pooled out-of-fold predictions: AUC, log-loss, F2 with the pipeline's cut-off rule (each
   fold's cut-off from the other four folds). A power law score(n) = a - b n^-c is fitted to extrapolate.
2. Calibration of the v22 level 2 (the pipeline's final probabilities, out of fold on the pipeline folds; cache of
   r33_t1_common.base_probs): reliability by decile of p, and the F2 the model's own probabilities say is achievable: if p were
   the true cheating probability, the expected F2 of flagging p >= t is 5 sum_{p>=t} p / (4 sum p + #{p >= t}), maximised over t.
   Observed F2 close to this "self-consistent" F2 means the remaining errors are the uncertainty the model already reports.
"""
import json
import os
import time

NT = int(os.environ.get("R37_THREADS", 4))
os.environ.setdefault("OMP_NUM_THREADS", str(NT))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402
from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402

import r33_t1_common as c  # noqa: E402
from r33_t1_common import f2, fold_of, labels_from, y  # noqa: E402
from src import task1  # noqa: E402

t0 = time.time()
log = lambda m: print(f"{m} [{time.time() - t0:.0f}s]", flush=True)
raw = c.tr.drop(columns=[col for col in ("player_id", "id", task1.TARGET) if col in c.tr.columns])
X = task1.add_features(c.tr, ref=raw)
folds = c.folds
fo = fold_of(folds)
P = json.load(open("outputs/params/task1_lgb.json"))
P = dict(P, learning_rate=P["learning_rate"] / 2, n_estimators=10000, subsample_freq=1, n_jobs=NT, verbose=-1, random_state=0)
FRACS = (0.125, 0.25, 0.5, 0.75, 1.0)
rng = np.random.default_rng(0)
OOF = {fr: np.zeros(len(y)) for fr in FRACS}
iters = {fr: [] for fr in FRACS}
for i, (a, b) in enumerate(folds):
    for fr in FRACS:
        sub = a.copy() if fr == 1 else np.concatenate([rng.choice(a[y[a] == k], int(round(fr * np.sum(y[a] == k))), replace=False) for k in (0, 1)])
        es = rng.random(len(sub)) < 0.1
        m = lgb.LGBMClassifier(**P).fit(X.iloc[sub[~es]], y[sub[~es]], eval_set=[(X.iloc[sub[es]], y[sub[es]])],
                                        callbacks=[lgb.early_stopping(200, verbose=False)])
        OOF[fr][b] = m.predict_proba(X.iloc[b])[:, 1]
        iters[fr].append(m.best_iteration_)
    log(f"fold {i} done")
rows = []
for fr in FRACS:
    n = int(round(fr * np.mean([len(a) for a, _ in folds])))
    rows.append(dict(fraction=fr, training_rows=n, iterations=int(np.median(iters[fr])), AUC=roc_auc_score(y, OOF[fr]),
                     logloss=log_loss(y, OOF[fr]), F2_nested_cut=f2(y, labels_from(OOF[fr], fo))))
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.round(5).to_string(index=False))
law = lambda n, a_, b_, c_: a_ - b_ * n ** (-c_)
for col in ("AUC", "F2_nested_cut"):
    try:
        (a_, b_, c_), _ = curve_fit(law, df.training_rows.values, df[col].values, p0=(df[col].max() + 0.01, 1.0, 0.5), maxfev=20000)
        n1 = df.training_rows.values[-1]
        print(f"{col}: power law a - b n^-c, a = {a_:.4f} (asymptote), c = {c_:.2f}; at {n1:,} rows {law(n1, a_, b_, c_):.4f}, at twice the rows "
              f"{law(2 * n1, a_, b_, c_):.4f} ({law(2 * n1, a_, b_, c_) - law(n1, a_, b_, c_):+.4f})")
    except RuntimeError as e:
        print(f"{col}: power-law fit failed ({e})")

# 2. calibration of the pipeline's level 2
p = c.base_probs()[0]
print(f"\nv22 level 2 (pipeline folds): AUC {roc_auc_score(y, p):.5f}, log-loss {log_loss(y, p):.5f}, F2 (pipeline cut rule) "
      f"{f2(y, labels_from(p, fo)):.5f}; prevalence {y.mean():.4f}")
q = pd.qcut(p, 10, labels=False, duplicates="drop")
rel = pd.DataFrame({"p": p, "y": y, "bin": q}).groupby("bin").agg(rows=("y", "size"), mean_p=("p", "mean"), observed=("y", "mean"))
print("reliability by decile of p:\n" + rel.round(4).to_string())
ece = np.sum(rel.rows * np.abs(rel.mean_p - rel.observed)) / len(p)
o = np.argsort(-p)
cum_p = np.cumsum(p[o])
k = np.arange(1, len(p) + 1)
f_self = 5 * cum_p / (4 * p.sum() + k)
f_obs = 5 * np.cumsum(y[o]) / (4 * y.sum() + k)
ks, ko = int(np.argmax(f_self)), int(np.argmax(f_obs))
print(f"expected calibration error {ece:.4f}; share of rows with 0.1 < p < 0.9: {np.mean((p > 0.1) & (p < 0.9)):.3f}; rows with p >= 0.95: "
      f"{np.mean(p >= 0.95):.3f} (cheater rate {y[p >= 0.95].mean():.4f}), p <= 0.05: {np.mean(p <= 0.05):.3f} (cheater rate {y[p <= 0.05].mean():.4f})")
print(f"self-consistent F2 (if p were the true probability): {f_self[ks]:.4f} at the top {ks + 1:,} rows (cut p {p[o][ks]:.3f}); observed F2 at "
      f"the best single cut {f_obs[ko]:.4f} at the top {ko + 1:,} rows (cut p {p[o][ko]:.3f})")
log("done")
