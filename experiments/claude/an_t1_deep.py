"""Task 1 deep-dive: (1) duplicates, (2) per-feature AUC vs KS (shape differences AUC cannot see), overall and in
the ambiguous middle, (3) do cheaters form sub-types?"""
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import common  # noqa
from common import load_t1, T1_MODELS, _logit
from src.utils import load_preds
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

tr, te, X, y, Xt, folds, _ = load_t1()
raw = [c for c in tr.columns if c not in ("player_id", "is_cheater")]
# (1) duplicates
key = lambda d: pd.util.hash_pandas_object(d[raw].round(6), index=False)
htr, hte = key(tr), key(te)
print(f"duplicate feature rows within labelled train: {int(htr.duplicated().sum())}; test rows identical to a train row: "
      f"{int(hte.isin(set(htr)).sum())}; unique values per column (min/median): "
      f"{tr[raw].nunique().min()} / {int(tr[raw].nunique().median())}")
# (2) AUC vs KS, overall and in the middle
S = np.column_stack([_logit(load_preds(n)["oof"]) for n in T1_MODELS])
p = np.zeros(len(y))
for a, b in folds:
    p[b] = LogisticRegression(max_iter=3000).fit(S[a], y[a]).predict_proba(S[b])[:, 1]
mid = (p > 0.05) & (p < 0.95)
rows = []
for c in raw:
    v = tr[c].values; ok = ~np.isnan(v)
    r = {"feature": c, "AUC": roc_auc_score(y[ok], v[ok]), "KS": ks_2samp(v[ok & (y == 1)], v[ok & (y == 0)]).statistic}
    om = ok & mid
    r["AUC mid"] = roc_auc_score(y[om], v[om]); r["KS mid"] = ks_2samp(v[om & (y == 1)], v[om & (y == 0)]).statistic
    r["sd ratio mid (cheat/clean)"] = np.nanstd(v[om & (y == 1)]) / np.nanstd(v[om & (y == 0)])
    r["n unique"] = tr[c].nunique()
    rows.append(r)
R = pd.DataFrame(rows)
R["|AUC-0.5|"] = (R.AUC - 0.5).abs()
print(f"\nmiddle group: {mid.sum():,} rows, cheater share {y[mid].mean():.3f}")
print(R.sort_values("KS mid", ascending=False).round(3).to_string(index=False))
# (3) sub-types among cheaters: 2-component Gaussian mixture on quantile-normal features of cheaters
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import QuantileTransformer
sig = R[R["KS"] > 0.03].feature.tolist()
Q = QuantileTransformer(output_distribution="normal", random_state=0).fit_transform(tr[sig])
Qc = np.nan_to_num(Q[y == 1])
for k in (2, 3):
    g = GaussianMixture(k, covariance_type="diag", random_state=0).fit(Qc)
    lab = g.predict(Qc)
    print(f"\ncheater sub-types, GMM k={k}: sizes {np.bincount(lab).tolist()}; BIC {g.bic(Qc):.0f}")
    prof = pd.DataFrame(g.means_, columns=sig).T.round(2)
    print(prof[(prof.abs() > 0.3).any(axis=1)].to_string())
