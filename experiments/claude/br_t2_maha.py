"""Round 12, Task 2 (features only = the no-player_id fallback; v11's Task 2 is the id block and is not affected).
Brief items not tested before:
  M1  LightGBM + in-fold class-conditional Mahalanobis distances (4 classes, Ledoit-Wolf covariances) as a 5th stack member
  M2  class multipliers shrunk toward 1:  w = 0.7 * w_opt + 0.3   (nested)
Baseline B0 = the notebook's 4-member stack + nested class multipliers (round 10: 0.7536). Rule: CI > 0, >= 4/5 folds, std <= 1.1x.
"""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import QuantileTransformer

warnings.filterwarnings("ignore")
import common  # noqa: F401,E402
from common import load_t2, T2_MODELS, P2_lgb  # noqa: E402
from src.utils import load_preds  # noqa: E402
from src.metrics import macro_f1, optimize_class_weights, apply_class_weights  # noqa: E402
from src.modeling import fit_predict  # noqa: E402

T0 = time.time()
log = lambda *a: print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
rng = np.random.default_rng(0)
tr2, te2, X2, y2, Xt2, folds2 = load_t2()
fold2 = np.zeros(len(y2), int)
for k, (a, b) in enumerate(folds2):
    fold2[b] = k
m2 = {n: load_preds(n)["oof"] for n in T2_MODELS}
num_cols = [c for c in X2.columns if str(X2[c].dtype) != "category"]  # distances on the 56 numeric columns; categoricals go to LightGBM as they are
Xv = X2[num_cols].values.astype(float)
nan_cols = np.where(np.isnan(Xv).any(0))[0]


def stack2(oofs):
    S = np.column_stack([np.log(np.clip(o, 1e-6, 1)) for o in oofs])
    o = np.zeros((len(y2), 4))
    for a, b in folds2:
        o[b] = LogisticRegression(C=0.5, max_iter=3000).fit(S[a], y2[a]).predict_proba(S[b])
    return o


def mult_nested(P, shrink=0.0):
    pred = np.zeros(len(y2), int)
    for a, b in folds2:
        w, _ = optimize_class_weights(y2[a], P[a])
        w = (1 - shrink) * np.asarray(w) + shrink
        pred[b] = apply_class_weights(P[b], w)
    return pred


pf = lambda pred: np.array([macro_f1(y2[fold2 == k], pred[fold2 == k]) for k in range(5)])


def boot(pn, po, B=300):
    d = []
    for _ in range(B):
        i = rng.integers(0, len(y2), len(y2))
        d.append(macro_f1(y2[i], pn[i]) - macro_f1(y2[i], po[i]))
    return np.percentile(d, [2.5, 97.5])


S0 = stack2([m2[n] for n in T2_MODELS])
q0 = mult_nested(S0)
g0 = pf(q0)
log(f"B0: nested macro-F1 {macro_f1(y2, q0):.4f}, folds {np.round(g0, 4)}, std {g0.std():.4f}")
rows = []


def judge(name, pred):
    s, g = macro_f1(y2, pred), pf(pred)
    lo, hi = boot(pred, q0)
    better = int((g > g0).sum())
    keep = lo > 0 and better >= 4 and g.std() <= g0.std() * 1.1
    rows.append([name, s, s - macro_f1(y2, q0), f"[{lo:+.4f}, {hi:+.4f}]", f"{better}/5", g.std(), "KEEP" if keep else "discard"])
    log(f"{name}: {s:.4f} ({s - macro_f1(y2, q0):+.4f}) CI {rows[-1][3]}, {better}/5, std {g.std():.4f} -> {rows[-1][-1]}")


judge("M2 multipliers shrunk 30% toward 1", mult_nested(S0, shrink=0.3))

o = np.zeros((len(y2), 4))
for a, b in folds2:
    med = np.nanmedian(Xv[a], 0)
    fill = lambda A: np.where(np.isnan(A), med, A)
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000, subsample=200_000, random_state=0).fit(fill(Xv[a]))
    zf = lambda A: np.hstack([qt.transform(fill(A)), np.isnan(A[:, nan_cols]).astype(float)])
    za, zb = zf(Xv[a]), zf(Xv[b])
    st = []
    for c in range(4):
        lw = LedoitWolf().fit(za[y2[a] == c])
        st.append((lw.location_, lw.precision_))
    dist = lambda Zm: np.column_stack([np.sqrt(np.einsum("ij,jk,ik->i", Zm - mu, P_, Zm - mu)) for mu, P_ in st])
    feat = lambda Zm, idx: pd.concat([X2.iloc[idx].reset_index(drop=True), pd.DataFrame(dist(Zm), columns=[f"maha{c}" for c in range(4)])], axis=1)
    _, (pv,), _ = fit_predict("lgb", "multiclass", P2_lgb, feat(za, a), y2[a], feat(zb, b), y2[b], [feat(zb, b)])
    o[b] = pv
    log("  Mahalanobis fold done")
log(f"  member LightGBM + Mahalanobis alone: {macro_f1(y2, mult_nested(o)):.4f} (plain LightGBM {macro_f1(y2, mult_nested(m2['t2_lgb'])):.4f})")
judge("M1 + LightGBM with Mahalanobis distances (5th member)", mult_nested(stack2([m2[n] for n in T2_MODELS] + [o])))

pd.set_option("display.width", 250)
print("\n" + pd.DataFrame(rows, columns=["Variant", "macro-F1", "delta", "95% CI", "folds better", "fold std", "Decision"]).to_string(index=False))
log("done")
