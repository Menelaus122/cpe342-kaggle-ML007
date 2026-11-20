"""Round 37, Task 2: learning curve (training rows only, CPU). Would more training rows raise the score?

    python gj_r37_t2_learning_curve.py

Question (user, 2026-11-17, after a competitor reported a higher Task 2 score without naive Bayes): is Task 2 limited by the amount of
data (more rows would help) or by the noise in the labels (the curve is flat near 0.757)?
For each pipeline fold the training rows are subsampled (stratified, seed fixed) to 12.5 / 25 / 50 / 75 / 100% and two models are
fitted on the subsample only, then scored on the full held-out fold:
  LightGBM  the pipeline's Task 2 features (task2.build, label-free composites) and tuned parameters (learning rate halved as in the
            pipeline's final fits), early stopping on 10% of the subsample (never on the held-out fold);
  KDE NB    the pipeline's kernel naive Bayes (task2.kde_nb_predict) fitted on the subsample;
  blend     0.4 x LightGBM + 0.6 x KDE NB.
Metrics on the pooled out-of-fold predictions: macro-F1 at argmax (no class multipliers, so the curve is not confounded by their
tuning) and the log-loss. A power law score(n) = a - b n^-c is fitted to the macro-F1 curve to extrapolate to twice the rows.
"""
import json
import os
import time

NT = int(os.environ.get("R37_THREADS", 6))
os.environ.setdefault("OMP_NUM_THREADS", str(NT))
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402
from sklearn.metrics import f1_score, log_loss  # noqa: E402

import common  # noqa: F401,E402
from src import task2  # noqa: E402
from src.config import DATA_DIRS  # noqa: E402
from src.modeling import make_folds  # noqa: E402

t0 = time.time()
log = lambda m: print(f"{m} [{time.time() - t0:.0f}s]", flush=True)
tr = pd.read_csv(DATA_DIRS[2] / "train.csv")
X, y, _ = task2.build(tr, tr.head(5))
X = X.drop(columns=[c for c in (task2.TARGET, task2.ID_COL, "id") if c in X.columns])
folds = make_folds(y)
P = json.load(open("outputs/params/task2_lgb.json"))
P = dict(P, learning_rate=P["learning_rate"] / 2, objective="multiclass", num_class=4, n_estimators=5000, subsample_freq=1, n_jobs=NT,
         verbose=-1, random_state=0)
FRACS = (0.125, 0.25, 0.5, 0.75, 1.0)
rng = np.random.default_rng(0)
OOF = {(m, fr): np.zeros((len(y), 4)) for m in ("lgb", "nb") for fr in FRACS}
iters = {fr: [] for fr in FRACS}
for i, (a, b) in enumerate(folds):
    for fr in FRACS:
        if fr < 1:
            sub = np.concatenate([rng.choice(a[y[a] == k], int(round(fr * np.sum(y[a] == k))), replace=False) for k in range(4)])
        else:
            sub = a.copy()
        es = rng.random(len(sub)) < 0.1
        fit, val = sub[~es], sub[es]
        m = lgb.LGBMClassifier(**P).fit(X.iloc[fit], y[fit], eval_set=[(X.iloc[val], y[val])],
                                        callbacks=[lgb.early_stopping(100, verbose=False)])
        OOF[("lgb", fr)][b] = m.predict_proba(X.iloc[b])
        iters[fr].append(m.best_iteration_)
        OOF[("nb", fr)][b] = task2.kde_nb_predict(tr, y, np.sort(sub), [tr.iloc[b]])[0]
    log(f"fold {i} done")
mf1 = lambda P_: f1_score(y, P_.argmax(1), average="macro")
rows = []
for fr in FRACS:
    n = int(round(fr * np.mean([len(a) for a, _ in folds])))
    L, N = OOF[("lgb", fr)], OOF[("nb", fr)]
    B = 0.4 * L + 0.6 * N
    rows.append(dict(fraction=fr, training_rows=n, lgb_iters=int(np.median(iters[fr])), lgb_mF1=mf1(L), nb_mF1=mf1(N), blend_mF1=mf1(B),
                     lgb_logloss=log_loss(y, L), nb_logloss=log_loss(y, N), blend_logloss=log_loss(y, B)))
df = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(df.round(4).to_string(index=False))
law = lambda n, a_, b_, c_: a_ - b_ * n ** (-c_)
for col in ("lgb_mF1", "nb_mF1", "blend_mF1"):
    try:
        (a_, b_, c_), _ = curve_fit(law, df.training_rows.values, df[col].values, p0=(0.76, 1.0, 0.5), maxfev=20000)
        n1 = df.training_rows.values[-1]
        print(f"{col}: power-law fit a - b n^-c with a = {a_:.4f} (the asymptote), c = {c_:.2f}; at {n1:,} rows {law(n1, a_, b_, c_):.4f}, "
              f"at twice the rows {law(2 * n1, a_, b_, c_):.4f} ({law(2 * n1, a_, b_, c_) - law(n1, a_, b_, c_):+.4f}), "
              f"with unlimited rows {a_:.4f}")
    except RuntimeError as e:
        print(f"{col}: power-law fit failed ({e})")
log("done")
