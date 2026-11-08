"""Task 3: does imputing the missing segment help? Ratio-target hurdle (best member) with and without it."""
import warnings
import numpy as np
warnings.filterwarnings("ignore")
import common  # noqa
from common import t3_blend_score, OUT
from src import task3
from src.metrics import nmae
from src.modeling import make_folds, fit_predict
from t3_fill import fill
from t3_segment import impute_segment

tr, te = task3.load()
y = tr[task3.TARGET].values.astype(float)
folds = make_folds(y, kind="regression")
trf, tef = fill(tr)[0], fill(te)[0]
X0, Xt0 = task3.add_features(trf), task3.add_features(tef)
X1, Xt1, acc = impute_segment(X0, Xt0)
print(f"imputer hold-out accuracy on known rows: {acc:.4f}; imputed {int(X1.segment_imputed.sum())} train / "
      f"{int(Xt1.segment_imputed.sum())} test rows; imputed train distribution "
      f"{np.bincount(X1.segment[X1.segment_imputed == 1].astype(int)).tolist()}")
z, pos = np.log1p(y), y > 0
b, bt = np.log1p(trf.historical_spending.values), np.log1p(tef.historical_spending.values)
okb = np.isfinite(b)
pool = np.load(OUT / "t3_ratio_members.npz")
names = list(pool["names"])
P = pool["P_oof"]
gate = np.load(OUT / "t3_identity_pool_all.npz")["gate_oof"]
miss = X1.segment_imputed.values == 1


def ratio_hurdle(X):
    p, mu = np.zeros(len(y)), np.zeros(len(y))
    for a, v in folds:
        _, (pv,), _ = fit_predict("lgb", "binary", {"learning_rate": 0.05}, X.iloc[a], pos[a].astype(int), X.iloc[v],
                                  pos[v].astype(int), [X.iloc[v]])
        ap, vp = a[pos[a] & okb[a]], v[pos[v] & okb[v]]
        _, (rv,), _ = fit_predict("lgb", "regression", {"objective": "regression_l1", "learning_rate": 0.05},
                                  X.iloc[ap], z[ap] - b[ap], X.iloc[vp], z[vp] - b[vp], [X.iloc[v]])
        p[v], mu[v] = pv, np.where(okb[v], b[v] + rv, 0)
    resid = z[pos] - mu[pos]
    sigma0 = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    tm = task3.tune_mixture(y, p, mu, sigma0)
    mm = task3.mixture_median(np.clip(p + tm["shift"], 0, 1), mu, tm["sigma"])
    return p, mu, mm


from sklearn.metrics import roc_auc_score
for tag, X in (("without imputation", X0), ("with imputed segment", X1)):
    p, mu, mm = ratio_hurdle(X)
    print(f"{tag}: stage-1 AUC {roc_auc_score(pos, p):.4f} (rows missing segment {roc_auc_score(pos[miss], p[miss]):.4f}); "
          f"stage-2 MAE {np.mean(np.abs(z[pos] - mu[pos])):.4f}; mixture-median alone {nmae(y, mm):.5f}; "
          f"on rows missing segment: mean |err| {np.mean(np.abs(y - mm)[miss]):.0f}", flush=True)
