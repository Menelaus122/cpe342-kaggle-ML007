"""Does player_id carry target signal in Tasks 1, 3 and 5 as it does in Task 2?"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
import common  # noqa: F401,E402  (cwd = project root)
from lightgbm import LGBMRegressor  # noqa: E402

DIRS = {1: "01_binary_classification", 3: "03_regression"}
TGT = {1: "is_cheater", 3: "spending_30d"}


def id_signal(name, idn, t, binary):
    o = np.argsort(idn); ts = t[o]
    rho, p = stats.spearmanr(idn, t)
    lag1 = stats.spearmanr(ts[1:], ts[:-1])[0]
    dec = pd.Series(t).groupby(pd.qcut(idn, 10, labels=False)).mean().values
    # id-only model, 5-fold OOF
    X = idn.reshape(-1, 1).astype(float); oof = np.zeros(len(t))
    for a, b in KFold(5, shuffle=True, random_state=0).split(X):
        oof[b] = LGBMRegressor(n_estimators=300, learning_rate=0.03, min_child_samples=200, verbose=-1).fit(X[a], t[a]).predict(X[b])
    fit = roc_auc_score(t, oof) if binary else stats.spearmanr(t, oof)[0]
    print(f"{name}: n={len(t):,}, ids {idn.min()}..{idn.max()}; Spearman(id, target) {rho:+.4f} (p={p:.2f}); "
          f"lag-1 corr in id order {lag1:+.4f}; id-only OOF {'AUC' if binary else 'Spearman'} {fit:.4f}")
    print(f"   target mean per id decile: {np.round(dec, 3)}")


for k in (1, 3):
    tr = pd.read_csv(f"{DIRS[k]}/train.csv")
    tr = tr[tr[TGT[k]].notna()]
    id_signal(f"Task {k}", tr.player_id.str[1:].astype(int).values, tr[TGT[k]].astype(float).values, k == 1)

te5 = pd.read_csv("05_anomaly_detection/test.csv")
f5 = pd.read_csv("outputs/submission_parts/task5.csv")["task5"].values  # row i of test.csv -> ANS{i}
assert len(f5) == len(te5)
id_signal("Task 5 (our 6 families as target)", te5.player_id.str[1:].astype(int).values, f5.astype(float), True)
