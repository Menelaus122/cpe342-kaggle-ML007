"""Impute the missing Task 3 `segment` (5.8%) from the other features. The segment is a deterministic rule of
the features (LightGBM recovers it with accuracy 1.000), so a classifier trained on TRAIN rows with a known segment
fills it in. No target information and no test rows are used for fitting."""
import numpy as np
from src.modeling import fit_predict


def impute_segment(X, Xt, seed_folds=None):
    """X, Xt: feature frames (task3.add_features output, after t3_fill.fill). Returns copies with `segment`
    filled and a `segment_imputed` flag, plus the CV accuracy of the imputer on known rows."""
    X, Xt = X.copy(), Xt.copy()
    feats = [c for c in X.columns if c != "segment"]
    k = X.segment.notna().values
    s = X.segment.values
    idx = np.where(k)[0]
    rng = np.random.default_rng(0)
    perm = rng.permutation(idx)
    va, trn = perm[: len(perm) // 5], perm[len(perm) // 5:]
    _, (pv, pm, pt), _ = fit_predict("lgb", "multiclass", {"learning_rate": 0.1}, X.iloc[trn][feats], s[trn].astype(int),
                                     X.iloc[va][feats], s[va].astype(int),
                                     [X.iloc[va][feats], X.loc[~k, feats], Xt.loc[Xt.segment.isna(), feats]])
    acc = float((pv.argmax(1) == s[va]).mean())
    X["segment_imputed"] = (~k).astype(int)
    Xt["segment_imputed"] = Xt.segment.isna().astype(int)
    X.loc[~k, "segment"] = pm.argmax(1).astype(float)
    Xt.loc[Xt.segment.isna(), "segment"] = pt.argmax(1).astype(float)
    return X, Xt, acc
