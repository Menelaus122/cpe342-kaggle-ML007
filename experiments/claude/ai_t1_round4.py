"""Task 1 round 4: error analysis + three hypothesis-driven experiments, strict validation.

Metric: positive-class F2 (slides 23/25). Honest evaluation = NESTED threshold: for each of the notebook's 5
folds, tau is chosen on the other 4 folds' OOF stack probabilities and applied to this fold.
Significance: paired bootstrap over rows (B = 1000) of F2(candidate) - F2(baseline) on the nested binary
predictions; a change is kept only if the 95% interval lies above 0.

Baseline = the notebook stack (cached OOF of LightGBM, XGBoost, CatBoost, ExtraTrees, LogReg -> logistic stacker).
E1 feature engineering : in-fold k-nearest-neighbour target features (cheater share among the 25 / 100 nearest
                         labelled players in a rank-gauss skill space), nested so no validation label leaks.
E2 model diversity     : 4-seed bagged LightGBM + a recall-weighted LightGBM (scale_pos_weight x 3) as extra members.
E3 post-processing     : missingness-aware stacker (n_missing buckets as meta-features) and per-missingness thresholds.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score, confusion_matrix

warnings.filterwarnings("ignore")
import common  # noqa: E402
from common import load_t1, T1_MODELS, OUT, _logit, tuned  # noqa: E402
from src.utils import load_preds  # noqa: E402
from src.metrics import optimize_threshold, fbeta  # noqa: E402
from src.modeling import fit_predict  # noqa: E402

rng = np.random.default_rng(0)
tr, te, X, y, Xt, folds, _ = load_t1()
spw = (y == 0).sum() / (y == 1).sum()
P1_lgb = tuned("task1_lgb", {"learning_rate": 0.05, "scale_pos_weight": spw})
base = {n: load_preds(n) for n in T1_MODELS}
f2 = lambda yt, yp: fbeta(yt, yp, 2, "binary")
nm = X["n_missing"].values
nm_bucket = np.minimum(nm, 6) // 2                      # 0-1, 2-3, 4-5, 6+


def stack(oofs, extra=None, C_=1.0):
    S = np.column_stack([_logit(o) for o in oofs] + ([] if extra is None else [extra]))
    o = np.zeros(len(y))
    for a, b in folds:
        o[b] = LogisticRegression(C=C_, max_iter=3000).fit(S[a], y[a]).predict_proba(S[b])[:, 1]
    return o


def nested(p, groups=None):
    """Binary OOF predictions with tau chosen on the other folds (optionally one tau per group)."""
    pred = np.zeros(len(y), int)
    for a, b in folds:
        if groups is None:
            t, _ = optimize_threshold(y[a], p[a], 2.0, average="binary")
            pred[b] = p[b] >= t
        else:
            for g in np.unique(groups):
                aa, bb = a[groups[a] == g], b[groups[b] == g]
                t, _ = optimize_threshold(y[aa], p[aa], 2.0, average="binary")
                pred[bb] = p[bb] >= t
    return pred


def boot(pred_new, pred_old, B=1000):
    """Paired bootstrap of the F2 difference; multinomial row weights, F2 from weighted counts."""
    yb = y.astype(bool)
    a_tp, a_fp, a_fn = yb & (pred_new == 1), ~yb & (pred_new == 1), yb & (pred_new == 0)
    b_tp, b_fp, b_fn = yb & (pred_old == 1), ~yb & (pred_old == 1), yb & (pred_old == 0)
    M = np.column_stack([a_tp, a_fp, a_fn, b_tp, b_fp, b_fn]).astype(float)
    d = np.empty(B)
    for i in range(B):
        w = rng.poisson(1.0, len(y))                    # Poisson bootstrap (= multinomial for large n)
        c = w @ M
        fa = 5 * c[0] / (5 * c[0] + 4 * c[2] + c[1])
        fb = 5 * c[3] / (5 * c[3] + 4 * c[5] + c[4])
        d[i] = fa - fb
    return np.percentile(d, [2.5, 97.5])


rows = []


def report(name, desc, p, pred, ref_pred=None):
    s = f2(y, pred)
    if ref_pred is None:
        rows.append([name, desc, roc_auc_score(y, p), s, 0.0, "-", "baseline"])
    else:
        lo, hi = boot(pred, ref_pred)
        keep = "KEEP" if lo > 0 else "discard"
        rows.append([name, desc, roc_auc_score(y, p), s, s - f2(y, ref_pred), f"[{lo:+.4f}, {hi:+.4f}]", keep])
    print(f"{name}: AUC {rows[-1][2]:.4f}, nested F2 {s:.4f}", flush=True)


# ------------------------------------------------------------------ baseline + error analysis
p0 = stack([base[n]["oof"] for n in T1_MODELS])
pred0 = nested(p0)
report("B0", "notebook stack (5 members), nested tau", p0, pred0)
t_all, s_all = optimize_threshold(y, p0, 2.0, average="binary")
print(f"  (non-nested OOF optimum, as quoted before: F2 {s_all:.4f} at tau {t_all:.3f})")
tn, fp, fn, tp = confusion_matrix(y, pred0).ravel()
print(f"\nconfusion matrix at nested tau: TN {tn:,}  FP {fp:,}  FN {fn:,}  TP {tp:,}  "
      f"(recall {tp / (tp + fn):.3f}, precision {tp / (tp + fp):.3f})")
print("errors by missing values per row:")
df = pd.DataFrame({"nm": np.minimum(nm, 6), "y": y, "pred": pred0, "p": p0})
g = df.groupby("nm").apply(lambda d: pd.Series({"rows": len(d), "cheater share": d.y.mean(),
                                                "AUC": roc_auc_score(d.y, d.p), "F2": f2(d.y, d.pred),
                                                "recall": d.pred[d.y == 1].mean(), "FP rate": d.pred[d.y == 0].mean()}))
print(g.round(3).to_string())
print("\ncalibration by stacker-probability decile (mean p vs actual cheater share):")
dec = pd.qcut(p0, 10, labels=False, duplicates="drop")
print(pd.DataFrame({"p": p0, "y": y, "d": dec}).groupby("d").agg(mean_p=("p", "mean"), actual=("y", "mean")).round(3).T.to_string())
raw = [c for c in tr.columns if c not in ("player_id", "is_cheater")]
Z = (tr[raw] - tr[raw].mean()) / tr[raw].std()
FN, TP, TN = (y == 1) & (pred0 == 0), (y == 1) & (pred0 == 1), (y == 0) & (pred0 == 0)
prof = pd.DataFrame({"missed cheaters (FN)": Z[FN].median(), "caught cheaters (TP)": Z[TP].median(),
                     "clean (TN)": Z[TN].median()})
prof["FN - TN"] = prof.iloc[:, 0] - prof.iloc[:, 2]
print("\nworst failure mode, missed cheaters: median z-score profile (top |FN - TN| features):")
print(prof.reindex(prof["FN - TN"].abs().sort_values(ascending=False).index).head(10).round(3).to_string())
print(f"missed cheaters with p < 0.05: {int((FN & (p0 < 0.05)).sum()):,} of {int(FN.sum()):,} (indistinguishable from clean)")

# ------------------------------------------------------------------ E1: nested kNN target features
skill = [c for c in raw if abs(roc_auc_score(y[~tr[c].isna()], tr[c].dropna()) - 0.5) > 0.02]
print(f"\nE1: kNN space = {len(skill)} raw features with univariate signal", flush=True)
K = (25, 100)


def knn_feats(fit_idx, query_frames, qt):
    A = np.nan_to_num(qt.transform(tr[skill].iloc[fit_idx].values))
    nn = NearestNeighbors(n_neighbors=max(K) + 1).fit(A)
    yy = y[fit_idx]
    outs = []
    for Q, same in query_frames:
        _, ind = nn.kneighbors(np.nan_to_num(qt.transform(Q)))
        if same:
            ind = ind[:, 1:]                              # drop the point itself
        outs.append(np.column_stack([yy[ind[:, :k]].mean(1) for k in K]))
    return outs


lgb_oof_knn = np.zeros(len(y))
for f, (a, b) in enumerate(folds):
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=0).fit(tr[skill].iloc[a].values)
    # training rows: inner 4-fold kNN features from labels inside `a` only
    Fa = np.zeros((len(a), len(K)))
    inner = np.array_split(rng.permutation(len(a)), 4)
    for ii in inner:
        rest = np.setdiff1d(np.arange(len(a)), ii)
        Fa[ii] = knn_feats(a[rest], [(tr[skill].iloc[a[ii]].values, False)], qt)[0]
    Fb = knn_feats(a, [(tr[skill].iloc[b].values, False)], qt)[0]
    Xa = X.iloc[a].assign(knn25=Fa[:, 0], knn100=Fa[:, 1])
    Xb = X.iloc[b].assign(knn25=Fb[:, 0], knn100=Fb[:, 1])
    _, (pb,), _ = fit_predict("lgb", "binary", P1_lgb, Xa, y[a], Xb, y[b], [Xb])
    lgb_oof_knn[b] = pb
    print(f"  fold {f}: LightGBM+kNN AUC {roc_auc_score(y[b], pb):.4f} (LightGBM {roc_auc_score(y[b], base['t1_lgb']['oof'][b]):.4f})", flush=True)
p1 = stack([lgb_oof_knn] + [base[n]["oof"] for n in T1_MODELS[1:]])
report("E1", "kNN target features (in-fold, nested) in LightGBM", p1, nested(p1), pred0)
p1b = stack([lgb_oof_knn] + [base[n]["oof"] for n in T1_MODELS])
report("E1b", "LightGBM+kNN as an extra stack member", p1b, nested(p1b), pred0)

# ------------------------------------------------------------------ E2: model diversity
print("\nE2: seed bagging + recall-weighted LightGBM", flush=True)
bag, rw = np.zeros(len(y)), np.zeros(len(y))
for f, (a, b) in enumerate(folds):
    for s in range(4):
        prm = {**P1_lgb, "random_state": 100 + s, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.6}
        _, (pb,), _ = fit_predict("lgb", "binary", prm, X.iloc[a], y[a], X.iloc[b], y[b], [X.iloc[b]])
        bag[b] += pb / 4
    prm = {**P1_lgb, "scale_pos_weight": 3 * P1_lgb.get("scale_pos_weight", 1.0)}
    _, (pb,), _ = fit_predict("lgb", "binary", prm, X.iloc[a], y[a], X.iloc[b], y[b], [X.iloc[b]])
    rw[b] = pb
    print(f"  fold {f}: bag AUC {roc_auc_score(y[b], bag[b]):.4f}, recall-weighted AUC {roc_auc_score(y[b], rw[b]):.4f}", flush=True)
np.savez(OUT / "t1_round4_members.npz", knn=lgb_oof_knn, bag=bag, rw=rw)
p2 = stack([base[n]["oof"] for n in T1_MODELS] + [bag])
report("E2a", "+ 4-seed bagged LightGBM member", p2, nested(p2), pred0)
p2b = stack([base[n]["oof"] for n in T1_MODELS] + [bag, rw])
report("E2b", "+ bagged + recall-weighted LightGBM members", p2b, nested(p2b), pred0)

# ------------------------------------------------------------------ E3: post-processing
print("\nE3: missingness-aware stacking / thresholds", flush=True)
oh = np.eye(4)[nm_bucket][:, 1:]                         # buckets as meta-features
S = np.column_stack([_logit(base[n]["oof"]) for n in T1_MODELS])
S3 = np.column_stack([S, oh, S.mean(1, keepdims=True) * oh])
p3 = np.zeros(len(y))
for a, b in folds:
    p3[b] = LogisticRegression(C=1.0, max_iter=3000).fit(S3[a], y[a]).predict_proba(S3[b])[:, 1]
report("E3a", "stacker + missing-count buckets (and x mean logit)", p3, nested(p3), pred0)
report("E3b", "one tau per missing-count bucket (nested)", p0, nested(p0, nm_bucket), pred0)

res = pd.DataFrame(rows, columns=["Experiment", "Changes", "OOF AUC", "OOF F2 (nested tau)", "Delta vs B0",
                                  "95% CI of delta (bootstrap)", "Decision"])
print("\n" + res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
res.to_csv(OUT / "t1_round4_results.csv", index=False)
