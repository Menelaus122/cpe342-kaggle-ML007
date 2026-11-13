"""Phase 2 (2026-11-10), Tasks 1 and 2: the brief's generalisation checks with fold-stability metrics.

Acceptance rule (brief + ours): better on >= 4 of 5 folds, fold std not larger, paired-bootstrap 95% CI above 0, and for Task 1
also not worse under a TEST-LIKE missingness pattern (the audit found train/test differ only in which cells are missing, AUC 0.90).
Test-like evaluation (evaluation only, nothing is fitted on test rows): validation rows get extra random masking in the columns that
are missing more often in test, and rows are reweighted for the columns that are missing less often in test (missingness is MCAR).

Task 1 (positive-class F2; B0 = notebook stack, nested tau):
  A1 missing-value indicators + native NaN   A2 median imputation instead of native NaN
  A3 missingness-robust training (each training row duplicated with 10% of its raw cells masked at random; generic, not test-tuned)
  A4 cut-off = median of the per-fold optimal cut-offs   A5 isotonic calibration of each member before stacking
  A6 LightGBM averaged over seeds 42 / 2026 / 777
Task 2 (macro-F1, features only = the clean pipeline; B0 = stack + class multipliers fitted on the other folds):
  B1 multipliers = median of per-fold multipliers   B2 calibration report   B3 LightGBM averaged over 3 seeds
"""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

warnings.filterwarnings("ignore")
import common  # noqa: E402
from common import load_t1, load_t2, T1_MODELS, T2_MODELS, OUT, _logit, tuned, stack_multi, P2_lgb  # noqa: E402
from src import task1  # noqa: E402
from src.utils import load_preds  # noqa: E402
from src.metrics import optimize_threshold, fbeta, macro_f1, optimize_class_weights, apply_class_weights  # noqa: E402
from src.modeling import fit_predict  # noqa: E402

T0 = time.time()
log = lambda *a: print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
rng = np.random.default_rng(0)
SEEDS = (42, 2026, 777)

# =============================================================================== Task 1
tr, te, X, y, Xt, folds, _ = load_t1()
raw = [c for c in tr.columns if c not in ("player_id", "is_cheater")]
spw = (y == 0).sum() / (y == 1).sum()
P1 = tuned("task1_lgb", {"learning_rate": 0.05, "scale_pos_weight": spw})
base = {n: load_preds(n)["oof"] for n in T1_MODELS}
f2 = lambda yt, yp: fbeta(yt, yp, 2, "binary")
fold_id = np.zeros(len(y), int)
for k, (a, b) in enumerate(folds):
    fold_id[b] = k

# test-like evaluation: extra masking + weights (evaluation only)
q_tr, q_te = tr[raw].isna().mean(), te[raw].isna().mean()
more = [c for c in raw if q_te[c] > q_tr[c]]
less = [c for c in raw if q_te[c] < q_tr[c]]
mrng = np.random.default_rng(123)
raw_tl = tr[raw].copy()
for c in more:
    obs = raw_tl[c].notna().values
    extra = obs & (mrng.random(len(raw_tl)) < (q_te[c] - q_tr[c]) / (1 - q_tr[c]))
    raw_tl.loc[extra, c] = np.nan
X_tl = task1.add_features(raw_tl, ref=tr[raw])
w_tl = np.ones(len(y))
for c in less:
    m = tr[c].isna().values
    w_tl *= np.where(m, q_te[c] / q_tr[c], (1 - q_te[c]) / (1 - q_tr[c]))
log(f"test-like evaluation: {len(more)} columns masked further, {len(less)} reweighted; effective n {w_tl.sum() ** 2 / (w_tl ** 2).sum():,.0f}")


def wf2(pred, w):
    tp = w @ ((y == 1) & (pred == 1)); fp = w @ ((y == 0) & (pred == 1)); fn = w @ ((y == 1) & (pred == 0))
    return 5 * tp / (5 * tp + 4 * fn + fp)


def nested_tau(p, w=None, rule="pooled"):
    pred = np.zeros(len(y), int)
    for k, (a, b) in enumerate(folds):
        if rule == "pooled":
            t, _ = optimize_threshold(y[a], p[a], 2.0, average="binary")
        else:                                            # median of the optimal cut-offs of the other folds, one by one
            t = np.median([optimize_threshold(y[bb], p[bb], 2.0, average="binary")[0] for j, (_, bb) in enumerate(folds) if j != k])
        pred[b] = p[b] >= t
    return pred


def stack(oofs):
    S = np.column_stack([_logit(o) for o in oofs])
    o = np.zeros(len(y))
    for a, b in folds:
        o[b] = LogisticRegression(max_iter=3000).fit(S[a], y[a]).predict_proba(S[b])[:, 1]
    return o


def per_fold(pred, scorer):
    return np.array([scorer(y[fold_id == k], pred[fold_id == k]) for k in range(5)])


def boot_f2(pn, po, w=None, B=1000):
    w = np.ones(len(y)) if w is None else w
    d = []
    for _ in range(B):
        pw = w * rng.poisson(1.0, len(y))
        d.append(wf2(pn, pw) - wf2(po, pw))
    return np.percentile(d, [2.5, 97.5])


def lgb_cv(Xp, Xv=None, seeds=(42,), augment=False):
    """OOF on validation rows as given and (optionally) on their test-like version."""
    oof, oof_tl = np.zeros(len(y)), np.zeros(len(y))
    for a, b in folds:
        for s in seeds:
            Xa, ya = Xp.iloc[a], y[a]
            if augment:
                ra = tr[raw].iloc[a].copy()
                mask = np.random.default_rng(s + 1).random(ra.shape) < 0.10
                ra = ra.mask(mask)
                Xa = pd.concat([Xa, task1.add_features(ra, ref=tr[raw])[Xp.columns]], ignore_index=True)
                ya = np.r_[ya, ya]
            m, (pv, pt), _ = fit_predict("lgb", "binary", {**P1, "random_state": s}, Xa, ya, Xp.iloc[b], y[b],
                                         [Xp.iloc[b], (Xv if Xv is not None else Xp).iloc[b]])
            oof[b] += pv / len(seeds)
            oof_tl[b] += pt / len(seeds)
    return oof, oof_tl


rows1 = []
p0 = stack([base[n] for n in T1_MODELS])
pred0 = nested_tau(p0)
f0 = per_fold(pred0, f2)
log(f"T1 B0: nested F2 {f2(y, pred0):.4f}, folds {np.round(f0, 4)}, std {f0.std():.4f}")


def add_row1(name, desc, p, pred, ref_pred=pred0, ref_folds=f0, tl=None):
    s, fo = f2(y, pred), per_fold(pred, f2)
    lo, hi = boot_f2(pred, ref_pred)
    better = int((fo > ref_folds).sum())
    tl_txt = "-" if tl is None else f"{tl:+.4f}"
    keep = lo > 0 and better >= 4 and fo.std() <= ref_folds.std() * 1.1 and (tl is None or tl >= 0)
    rows1.append([name, desc, roc_auc_score(y, p), s, s - f2(y, ref_pred), f"[{lo:+.4f}, {hi:+.4f}]", f"{better}/5",
                  fo.std(), tl_txt, "KEEP" if keep else "discard"])
    log(f"T1 {name}: F2 {s:.4f} ({s - f2(y, ref_pred):+.4f}), folds better {better}/5, std {fo.std():.4f}, test-like member delta {tl_txt}")


# control LightGBM (same procedure as the variants) + its test-like member score
ctrl, ctrl_tl = lgb_cv(X, X_tl)
pc = stack([ctrl if n == "t1_lgb" else base[n] for n in T1_MODELS])
pred_c, fc = nested_tau(pc), None
fc = per_fold(pred_c, f2)
member_tl = lambda o_tl: wf2(nested_tau(o_tl), w_tl)
c_tl = member_tl(ctrl_tl)
log(f"T1 control LightGBM refit: stack F2 {f2(y, pred_c):.4f}; member test-like F2 {c_tl:.4f} (plain {f2(y, nested_tau(ctrl)):.4f})")

na_cols = [f"na_{c}" for c in raw]
XA1 = pd.concat([X, tr[raw].isna().astype(int).set_axis(na_cols, axis=1)], axis=1)
XA1_tl = pd.concat([X_tl, raw_tl.isna().astype(int).set_axis(na_cols, axis=1)], axis=1)
o, o_tl = lgb_cv(XA1, XA1_tl)
ps = stack([o if n == "t1_lgb" else base[n] for n in T1_MODELS])
add_row1("A1", "missing-value indicators + native NaN (LightGBM swapped in)", ps, nested_tau(ps), pred_c, fc, member_tl(o_tl) - c_tl)

med = tr[raw].median()
XA2 = task1.add_features(tr[raw].fillna(med), ref=tr[raw]); XA2["n_missing"] = X["n_missing"].values
XA2_tl = task1.add_features(raw_tl.fillna(med), ref=tr[raw]); XA2_tl["n_missing"] = X_tl["n_missing"].values
o, o_tl = lgb_cv(XA2, XA2_tl)
ps = stack([o if n == "t1_lgb" else base[n] for n in T1_MODELS])
add_row1("A2", "median imputation instead of native NaN", ps, nested_tau(ps), pred_c, fc, member_tl(o_tl) - c_tl)

o, o_tl = lgb_cv(X, X_tl, augment=True)
ps = stack([o if n == "t1_lgb" else base[n] for n in T1_MODELS])
add_row1("A3", "missingness-robust training (rows duplicated with 10% random masking)", ps, nested_tau(ps), pred_c, fc, member_tl(o_tl) - c_tl)

add_row1("A4", "cut-off = median of per-fold optimal cut-offs", p0, nested_tau(p0, rule="median"))

cal = {}
for n in T1_MODELS:
    c_ = np.zeros(len(y))
    for a, b in folds:
        c_[b] = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(base[n][a], y[a]).predict(base[n][b])
    cal[n] = c_
ps = stack([cal[n] for n in T1_MODELS])
add_row1("A5", "isotonic calibration of every member before stacking", ps, nested_tau(ps))
prob_bins = pd.qcut(p0, 10, labels=False, duplicates="drop")
ece = np.sum([np.mean(prob_bins == k) * abs(p0[prob_bins == k].mean() - y[prob_bins == k].mean()) for k in np.unique(prob_bins)])
log(f"T1 stacker calibration: Brier {brier_score_loss(y, p0):.4f}, ECE (10 bins) {ece:.4f}")

o, o_tl = lgb_cv(X, X_tl, seeds=SEEDS)
ps = stack([o if n == "t1_lgb" else base[n] for n in T1_MODELS])
add_row1("A6", "LightGBM averaged over seeds 42 / 2026 / 777", ps, nested_tau(ps), pred_c, fc, member_tl(o_tl) - c_tl)

# =============================================================================== Task 2 (features only)
tr2, te2, X2, y2, Xt2, folds2 = load_t2()
fold2 = np.zeros(len(y2), int)
for k, (a, b) in enumerate(folds2):
    fold2[b] = k
m2 = {n: load_preds(n) for n in T2_MODELS}


def stack2(oofs):
    S = np.column_stack([np.log(np.clip(o, 1e-6, 1)) for o in oofs])
    o = np.zeros((len(y2), 4))
    for a, b in folds2:
        o[b] = LogisticRegression(C=0.5, max_iter=3000).fit(S[a], y2[a]).predict_proba(S[b])
    return o


def mult_nested(P, rule="pooled"):
    pred = np.zeros(len(y2), int)
    for k, (a, b) in enumerate(folds2):
        if rule == "pooled":
            w, _ = optimize_class_weights(y2[a], P[a])
        else:
            w = np.median([optimize_class_weights(y2[bb], P[bb])[0] for j, (_, bb) in enumerate(folds2) if j != k], axis=0)
        pred[b] = apply_class_weights(P[b], w)
    return pred


pf2 = lambda pred: np.array([macro_f1(y2[fold2 == k], pred[fold2 == k]) for k in range(5)])


def boot_mf1(pn, po, B=300):
    d = []
    for _ in range(B):
        i = rng.integers(0, len(y2), len(y2))
        d.append(macro_f1(y2[i], pn[i]) - macro_f1(y2[i], po[i]))
    return np.percentile(d, [2.5, 97.5])


rows2 = []
S0 = stack2([m2[n]["oof"] for n in T2_MODELS])
q0 = mult_nested(S0)
g0 = pf2(q0)
log(f"T2 B0 (features only): nested macro-F1 {macro_f1(y2, q0):.4f}, folds {np.round(g0, 4)}, std {g0.std():.4f}")


def add_row2(name, desc, pred, ref=q0, gref=g0):
    s, g = macro_f1(y2, pred), pf2(pred)
    lo, hi = boot_mf1(pred, ref)
    better = int((g > gref).sum())
    keep = lo > 0 and better >= 4 and g.std() <= gref.std() * 1.1
    rows2.append([name, desc, s, s - macro_f1(y2, ref), f"[{lo:+.4f}, {hi:+.4f}]", f"{better}/5", g.std(), "KEEP" if keep else "discard"])
    log(f"T2 {name}: {s:.4f} ({s - macro_f1(y2, ref):+.4f}), folds better {better}/5, std {g.std():.4f}")


add_row2("B1", "class multipliers = median of per-fold optima", mult_nested(S0, rule="median"))
log(f"T2 B2 calibration of the stacker: log-loss {log_loss(y2, S0):.4f}, multiclass Brier {np.mean(np.sum((S0 - np.eye(4)[y2]) ** 2, 1)):.4f}")
oo = {}
for s in SEEDS:
    o2 = np.zeros((len(y2), 4))
    for a, b in folds2:
        _, (pv,), _ = fit_predict("lgb", "multiclass", {**P2_lgb, "random_state": s}, X2.iloc[a], y2[a], X2.iloc[b], y2[b], [X2.iloc[b]])
        o2[b] = pv
    oo[s] = o2
    log(f"  T2 LightGBM seed {s} done")
ctrl2 = stack2([oo[42] if n == "t2_lgb" else m2[n]["oof"] for n in T2_MODELS])
qc = mult_nested(ctrl2)
avg = stack2([np.mean([oo[s] for s in SEEDS], 0) if n == "t2_lgb" else m2[n]["oof"] for n in T2_MODELS])
add_row2("B3", "LightGBM averaged over 3 seeds (vs the same LightGBM, 1 seed)", mult_nested(avg), qc, pf2(qc))

pd.set_option("display.width", 250)
print("\n" + pd.DataFrame(rows1, columns=["Exp", "Change", "AUC", "OOF F2", "delta", "95% CI", "folds better", "fold std",
                                         "test-like member delta", "Decision"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\n" + pd.DataFrame(rows2, columns=["Exp", "Change", "macro-F1", "delta", "95% CI", "folds better", "fold std",
                                         "Decision"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
log("done")
