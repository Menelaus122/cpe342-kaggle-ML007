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
