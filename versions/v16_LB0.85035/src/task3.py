"""Task 3 - Monthly Spending Prediction (zero-inflated regression, metric: NMAE).

NMAE = 1 / (1 + sum|y - yhat| / sum y) is monotone in the total absolute error, so the
Bayes-optimal point forecast is the conditional MEDIAN of spending_30d, not its mean.
Consequences used below:
  * L1 models trained on log1p(y) estimate the median directly (medians commute with the
    monotone log transform, so expm1 needs no smearing correction for this metric).
  * In a two-stage hurdle model with p = P(y > 0) and a log-normal positive part
    log1p(y) | y>0 ~ N(mu, sigma^2), the mixture median is
        0                                      if p <= 0.5
        expm1(mu + sigma * Phi^-1(1 - 0.5/p))  otherwise.
  * Duan's smearing (mean retransformation) is implemented for the expected-value hurdle
    p * E[y | y>0] as a comparison point, which is the right choice for MSE, not MAE.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from .config import DATA_DIRS, ID_COL, TARGETS
from .metrics import nmae
from .modeling import fit_predict

TARGET = TARGETS[3]


def load():
    d = DATA_DIRS[3]
    return pd.read_csv(d / "train.csv"), pd.read_csv(d / "test.csv")


def add_features(df):
    X = df.drop(columns=[c for c in (ID_COL, TARGET) if c in df.columns]).copy()
    raw = list(X.columns)
    eps = 1e-3
    X["log_hist"] = np.log1p(X.historical_spending)
    X["log_prev"] = np.log1p(X.prev_month_spending)
    X["prev_over_hist"] = X.prev_month_spending / (X.historical_spending + 1)
    X["hist_per_streak"] = X.historical_spending / (X.daily_login_streak + 1)
    X["hist_per_age"] = X.historical_spending / (X.account_age_days + 1) * 30
    X["hist_per_txn"] = X.historical_spending / (X.total_transactions + 1)
    X["txn_value_x_freq"] = X.avg_transaction_value * X.purchase_frequency
    X["log_txn_value_x_freq"] = np.log1p(X.txn_value_x_freq)
    X["event_x_session"] = X.event_participation_rate * X.avg_session_length
    X["weekly_minutes"] = X.sessions_per_week * X.avg_session_length
    X["recency_ratio"] = X.days_since_last_purchase / (X.days_since_last_login + 1)
    X["purchase_recency_x_freq"] = X.purchase_frequency / (X.days_since_last_purchase + 1)
    X["discount_share"] = X.purchases_on_discount / (X.total_transactions + 1)
    X["vip_x_premium"] = X.vip_status * (1 + X.is_premium_member)
    X["prev_x_segment"] = X.log_prev * (X.segment + 1)
    X["hist_x_event"] = X.log_hist * X.event_participation_rate
    X["prev_sq"] = X.log_prev ** 2
    X["prev_x_hist"] = X.log_prev * X.log_hist
    X["monetary_mean_log"] = X[["log_hist", "log_prev"]].mean(axis=1)
    X["n_missing"] = df[raw].isna().sum(axis=1)
    return X


def build(train, test):
    return add_features(train), train[TARGET].values.astype(float), add_features(test)


# --------------------------------------------------------------------- data repair (no target information)
RATIO = 0.3498  # median prev_month_spending / historical_spending (the ratio always lies in 0.30-0.40)


def fill_identities(df):
    """Fill missing spending-history values from exact identities between columns of the same row:
    historical_spending = total_transactions * avg_transaction_value (relative error 2e-16),
    purchases_on_discount = total_transactions * discount_rate_used, and prev / hist in 0.30-0.40
    (the median ratio is used only as a last resort). Each column is ~5.8% missing at random, so most gaps
    are recovered exactly. Uses no target and no other rows. Returns (filled copy, values filled per column)."""
    d = df.copy()
    H, T, A, P = d.historical_spending, d.total_transactions, d.avg_transaction_value, d.prev_month_spending
    POD, DR = d.purchases_on_discount, d.discount_rate_used
    H2 = H.fillna((T * A).where(T > 0)).fillna(P / RATIO)
    T2 = T.fillna((H / A).where(A > 0).round()).fillna((POD / DR).where(DR > 0).round())
    A2 = A.fillna((H2 / T2).where(T2 > 0))
    P2 = P.fillna(H2 * RATIO)
    POD2 = POD.fillna(T2 * DR)
    DR2 = DR.fillna((POD / T2).where(T2 > 0))
    stats = {c: int(new.notna().sum() - old.notna().sum()) for c, old, new in
             [("historical_spending", H, H2), ("total_transactions", T, T2), ("avg_transaction_value", A, A2),
              ("prev_month_spending", P, P2), ("purchases_on_discount", POD, POD2), ("discount_rate_used", DR, DR2)]}
    d["historical_spending"], d["total_transactions"], d["avg_transaction_value"] = H2, T2, A2
    d["prev_month_spending"], d["purchases_on_discount"], d["discount_rate_used"] = P2, POD2, DR2
    return d, stats


def impute_segment(X, X_test):
    """`segment` (5.8% missing) is a deterministic rule of the other features (event_participation_rate cut-offs
    plus historical_spending), so a classifier trained on TRAIN rows with a known segment fills it in.
    No target and no test rows are used for fitting. Returns copies with a `segment_imputed` flag + hold-out accuracy."""
    X, X_test = X.copy(), X_test.copy()
    feats = [c for c in X.columns if c != "segment"]
    k = X.segment.notna().values
    s = X.segment.values
    perm = np.random.default_rng(0).permutation(np.where(k)[0])
    va, trn = perm[: len(perm) // 5], perm[len(perm) // 5:]
    _, (pv, pm, pt), _ = fit_predict("lgb", "multiclass", {"learning_rate": 0.1}, X.iloc[trn][feats], s[trn].astype(int),
                                     X.iloc[va][feats], s[va].astype(int),
                                     [X.iloc[va][feats], X.loc[~k, feats], X_test.loc[X_test.segment.isna(), feats]])
    acc = float((pv.argmax(1) == s[va]).mean())
    X["segment_imputed"] = (~k).astype(int)
    X_test["segment_imputed"] = X_test.segment.isna().astype(int)
    X.loc[~k, "segment"] = pm.argmax(1).astype(float)
    X_test.loc[X_test.segment.isna(), "segment"] = pt.argmax(1).astype(float)
    return X, X_test, acc


def build_repaired(train, test):
    """Feature matrices on repaired data: identity fill -> add_features -> segment imputation."""
    trf, fill_tr = fill_identities(train)
    tef, fill_te = fill_identities(test)
    X, X_test = add_features(trf), add_features(tef)
    X, X_test, acc = impute_segment(X, X_test)
    return X, train[TARGET].values.astype(float), X_test, dict(fill_train=fill_tr, fill_test=fill_te, segment_accuracy=acc)


# --------------------------------------------------------------------- ratio-target hurdle stage 2
def _ratio_reg(kind, params):
    return {**({"objective": "regression_l1"} if kind == "lgb" else {}), **(params or {})}


def ratio_stage2_cv(kind, params, X, y, X_test, folds, base_col, mu_fallback, mu_fallback_test, cat_cols=None):
    """Hurdle stage 2 on the RATIO target log1p(y) - log1p(base): spending ~ base x segment factor x noise
    (within-segment MAD of log(y / historical_spending) 0.205 vs 0.44-0.53 for log y itself).
    Fitted on positive rows with a known base inside each fold. Rows without a base keep the plain hurdle's mu.
    Returns mu_oof, mu_test (5-fold average) and `test` = the raw ratio predictions (for the Train+Val blend)."""
    y = np.asarray(y)
    z, pos = np.log1p(y), y > 0
    b, bt = np.log1p(X[base_col].values), np.log1p(X_test[base_col].values)
    okb, okbt = np.isfinite(b), np.isfinite(bt)
    mu, ratio_t = np.array(mu_fallback, float).copy(), np.zeros(len(X_test))
    for a, v in folds:
        ap, vp, vb = a[pos[a] & okb[a]], v[pos[v] & okb[v]], v[okb[v]]
        _, (rv, rt), _ = fit_predict(kind, "regression", _ratio_reg(kind, params), X.iloc[ap], z[ap] - b[ap],
                                     X.iloc[vp], z[vp] - b[vp], [X.iloc[vb], X_test], cat_cols=cat_cols)
        mu[vb] = b[vb] + rv
        ratio_t += rt / len(folds)
    mu_t = np.array(mu_fallback_test, float).copy()
    mu_t[okbt] = bt[okbt] + ratio_t[okbt]
    return dict(mu_oof=mu, mu_test=mu_t, test=ratio_t)


def ratio_stage2_full(kind, params, X, y, X_test, folds, base_col, cat_cols=None):
    """Train+Val refit of ratio_stage2_cv (round count from fold 0, scaled; see modeling.refit_full)."""
    from .modeling import refit_full
    y = np.asarray(y)
    b = np.log1p(X[base_col].values)
    idx = np.where((y > 0) & np.isfinite(b))[0]
    remap = {o: n for n, o in enumerate(idx)}
    f0 = [(np.array([remap[i] for i in folds[0][0] if i in remap]), np.array([remap[i] for i in folds[0][1] if i in remap]))]
    r = refit_full(kind, "regression", _ratio_reg(kind, params), X.iloc[idx], np.log1p(y[idx]) - b[idx], X_test, f0,
                   cat_cols=cat_cols, name=f"ratio[{base_col}] {kind}")
    return dict(test=r["test"], n_iter=r["n_iter"])


def readouts(y, p, mu, sigma0=None):
    """Tune the three hurdle read-outs on OOF NMAE: mixture median (sigma, gate shift), hard gate (tau), smearing."""
    pos = np.asarray(y) > 0
    resid = np.log1p(y[pos]) - mu[pos]
    sigma0 = 1.4826 * np.median(np.abs(resid - np.median(resid))) if sigma0 is None else sigma0
    tm = tune_mixture(y, p, mu, sigma0)
    th = tune_hard_hurdle(y, p, mu)
    return dict(sigma=tm["sigma"], shift=tm["shift"], tau=th["tau"], smear=float(np.mean(np.exp(resid))))


def apply_readout(kind, p, mu, ro):
    if kind == "mixture-median":
        return mixture_median(np.clip(p + ro["shift"], 0, 1), mu, ro["sigma"])
    if kind == "hard gate":
        return hard_hurdle(p, mu, ro["tau"])
    return expected_value(p, mu, ro["smear"])


# --------------------------------------------------------------------- hurdle model
def hurdle_cv(kind, X, y, X_test, folds, clf_params=None, reg_params=None, cat_cols=None, verbose=True):
    """Stage 1: P(y>0) on all rows. Stage 2: median of log1p(y) trained on positive rows only.

    Both stages are fitted strictly inside each CV fold; stage-2 predictions are produced for
    every validation/test row (the gate decides how much they count)."""
    y = np.asarray(y)
    z = np.log1p(y)
    n, nt = len(X), len(X_test)
    p_oof, mu_oof = np.zeros(n), np.zeros(n)
    p_te, mu_te = np.zeros(nt), np.zeros(nt)
    reg = {**({"objective": "regression_l1"} if kind == "lgb" else {}), **(reg_params or {})}
    for f, (tr, va) in enumerate(folds):
        _, (pv, pt), _ = fit_predict(kind, "binary", clf_params or {}, X.iloc[tr], (y[tr] > 0).astype(int),
                                     X.iloc[va], (y[va] > 0).astype(int), [X.iloc[va], X_test], cat_cols=cat_cols)
        trp, vap = tr[y[tr] > 0], va[y[va] > 0]
        _, (mv, mt), _ = fit_predict(kind, "regression", reg, X.iloc[trp], z[trp], X.iloc[vap], z[vap],
                                     [X.iloc[va], X_test], cat_cols=cat_cols)
        p_oof[va], mu_oof[va] = pv, mv
        p_te += pt / len(folds)
        mu_te += mt / len(folds)
        if verbose:
            print(f"  [hurdle-{kind}] fold {f}", end="\r")
    pos = y > 0
    resid = z[pos] - mu_oof[pos]
    sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))  # robust (MAD) scale
    smear = np.mean(np.exp(resid))                                # Duan (1983) smearing factor
    if verbose:
        print(f"  [hurdle-{kind}] done: sigma={sigma:.3f} smear={smear:.3f}      ")
    return dict(p_oof=p_oof, mu_oof=mu_oof, p_test=p_te, mu_test=mu_te, sigma=sigma, smear=smear)


def hurdle_full(kind, X, y, X_test, folds, clf_params=None, reg_params=None, cat_cols=None):
    """Final Train+Val refit of both hurdle stages (round counts from fold 0, see refit_full)."""
    from .modeling import refit_full
    y = np.asarray(y)
    clf = refit_full(kind, "binary", clf_params or {}, X, (y > 0).astype(int), X_test, folds, cat_cols=cat_cols,
                     name="hurdle stage 1")
    pos = np.where(y > 0)[0]
    pos_set = set(pos.tolist())
    folds_pos = [(np.array([i for i in tr if i in pos_set]), np.array([i for i in va if i in pos_set])) for tr, va in folds[:1]]
    remap = {old: new for new, old in enumerate(pos)}
    folds_pos = [(np.array([remap[i] for i in tr]), np.array([remap[i] for i in va])) for tr, va in folds_pos]
    reg = {**({"objective": "regression_l1"} if kind == "lgb" else {}), **(reg_params or {})}
    stage2 = refit_full(kind, "regression", reg, X.iloc[pos], np.log1p(y[pos]), X_test, folds_pos,
                        cat_cols=cat_cols, name="hurdle stage 2")
    return dict(p_test=clf["test"], mu_test=stage2["test"])


def mixture_median(p, mu, sigma):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    q = np.clip(1 - 0.5 / p, 1e-6, None)
    out = np.expm1(mu + sigma * norm.ppf(q))
    return np.where(p > 0.5, np.clip(out, 0, None), 0.0)


def hard_hurdle(p, mu, tau):
    return np.where(p > tau, np.expm1(mu), 0.0)


def expected_value(p, mu, smear):
    return p * (np.exp(mu) * smear - 1).clip(0)


def tune_mixture(y, p, mu, sigma0):
    """Fine-tune (sigma, gate shift) of the mixture median directly on OOF NMAE."""
    def loss(v):
        s, shift = v
        return -nmae(y, mixture_median(np.clip(p + shift, 0, 1), mu, abs(s)))
    res = minimize(loss, [sigma0, 0.0], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-7})
    return dict(sigma=abs(res.x[0]), shift=res.x[1], score=-res.fun)


def tune_hard_hurdle(y, p, mu):
    grid = np.linspace(0.2, 0.8, 121)
    scores = [nmae(y, hard_hurdle(p, mu, t)) for t in grid]
    i = int(np.argmax(scores))
    return dict(tau=float(grid[i]), score=float(scores[i]))


# ------------------------------------------------------------------------ blending
def l1_blend_weights(y, preds):
    """Non-negative blend weights (summing to free scale) minimising total absolute error."""
    P = np.column_stack(preds)
    k = P.shape[1]

    def loss(w):
        return np.abs(y - P @ np.abs(w)).sum()

    res = minimize(loss, np.full(k, 1.0 / k), method="Powell", options={"xtol": 1e-4, "ftol": 1e-9, "maxiter": 20000})
    return np.abs(res.x)


def l1_blend_weights_exact(y, preds):
    """The same problem solved exactly. Non-negative least-absolute-deviation blending is a linear program:
        min sum(u + v)  s.t.  P w + u - v = y,  w, u, v >= 0,
    so HiGHS returns the global optimum (deterministic, no dependence on the start point or column order).
    On the Task 3 pool it beats the Powell search by +0.00007 NMAE out-of-fold (95% CI [+0.00001, +0.00014])."""
    import scipy.sparse as sp
    from scipy.optimize import linprog
    P = np.column_stack(preds)
    n, k = P.shape
    s = P.max(0) + 1e-9                                   # column scaling for numerical stability
    A = sp.hstack([sp.csr_matrix(P / s), sp.eye(n), -sp.eye(n)], format="csr")
    res = linprog(np.r_[np.zeros(k), np.ones(2 * n)], A_eq=A, b_eq=np.asarray(y, float), bounds=(0, None), method="highs")
    if res.status != 0:
        raise RuntimeError(f"L1 blend LP failed: {res.message}")
    return res.x[:k] / s
