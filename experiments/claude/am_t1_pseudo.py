"""Task 1: pseudo-labelling of the 13,184 UNLABELLED TRAINING rows (no test rows), strictly within folds:
for each fold the teacher is trained on that fold's labelled training rows only, labels the unlabelled rows it is
confident about, and the student is trained on labelled + pseudo-labelled rows. Scored on the fold's held-out
labelled rows: AUC and the stack F2 when the student replaces LightGBM."""
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import common  # noqa
from common import load_t1, T1_MODELS, tuned, _logit
from src import task1
from src.config import DATA_DIRS
from src.utils import load_preds
from src.modeling import fit_predict
from src.metrics import optimize_threshold, fbeta
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

tr, te, X, y, Xt, folds, n_unl = load_t1()
full = pd.read_csv(DATA_DIRS[1] / "train.csv")
unl = full[full[task1.TARGET].isna()].reset_index(drop=True)
Xu = task1.add_features(unl, ref=tr.drop(columns=["player_id", task1.TARGET]))
print(f"labelled {len(X):,}, unlabelled training rows {len(Xu):,}")
spw = (y == 0).sum() / (y == 1).sum()
P1 = tuned("task1_lgb", {"learning_rate": 0.05, "scale_pos_weight": spw})
base = {n: load_preds(n) for n in T1_MODELS}
stud = {c: np.zeros(len(y)) for c in ("hard 0.95/0.05", "soft all")}
for f, (a, v) in enumerate(folds):
    _, (pu, pv0), _ = fit_predict("lgb", "binary", P1, X.iloc[a], y[a], X.iloc[v], y[v], [Xu, X.iloc[v]])
    for c in stud:
        if c.startswith("hard"):
            keep = (pu >= 0.95) | (pu <= 0.05)
            Xa = pd.concat([X.iloc[a], Xu[keep]]); ya = np.r_[y[a], (pu[keep] >= 0.5).astype(int)]; w = None
        else:   # soft: each unlabelled row twice, as 1 with weight p and as 0 with weight 1 - p
            Xa = pd.concat([X.iloc[a], Xu, Xu]); ya = np.r_[y[a], np.ones(len(Xu), int), np.zeros(len(Xu), int)]
            w = np.r_[np.ones(len(a)), pu, 1 - pu]
        _, (pv,), _ = fit_predict("lgb", "binary", P1, Xa, ya, X.iloc[v], y[v], [X.iloc[v]], sample_weight=w)
        stud[c][v] = pv
    print(f"fold {f}: teacher AUC {roc_auc_score(y[v], pv0):.4f} | " + " | ".join(f"{c} {roc_auc_score(y[v], stud[c][v]):.4f}" for c in stud), flush=True)


def stack_f2(oofs):
    S = np.column_stack([_logit(o) for o in oofs]); o = np.zeros(len(y))
    for a, b in folds:
        o[b] = LogisticRegression(max_iter=3000).fit(S[a], y[a]).predict_proba(S[b])[:, 1]
    pred = np.zeros(len(y), int)
    for a, b in folds:
        t, _ = optimize_threshold(y[a], o[a], 2.0, average="binary"); pred[b] = o[b] >= t
    return fbeta(y, pred, 2, "binary")
print(f"stack nested F2: baseline {stack_f2([base[n]['oof'] for n in T1_MODELS]):.4f}")
for c, o in stud.items():
    print(f"  student ({c}) replacing LightGBM: {stack_f2([o] + [base[n]['oof'] for n in T1_MODELS[1:]]):.4f}; "
          f"added as a member: {stack_f2([base[n]['oof'] for n in T1_MODELS] + [o]):.4f}")
