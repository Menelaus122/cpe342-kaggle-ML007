"""Unified out-of-fold training engine for LightGBM / XGBoost / CatBoost (+ sklearn models).

Every model is trained K times on K-1 folds; the held-out fold yields leakage-free OOF
predictions (used for metric estimation, threshold tuning and stacking) while the test
prediction is the average of the K fold models.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import KFold, StratifiedKFold

from .config import N_FOLDS, SEED


# ---------------------------------------------------------------------------- folds
def make_folds(y, n_folds=N_FOLDS, seed=SEED, kind="stratified"):
    """kind: 'stratified' (classes), 'regression' (stratify on zero / target quantile bins) or 'plain'."""
    y = np.asarray(y)
    idx = np.zeros(len(y))
    if kind == "stratified":
        return list(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(idx, y))
    if kind == "regression":
        bins = np.zeros(len(y), dtype=int)
        pos = y > 0
        bins[pos] = 1 + pd.qcut(y[pos], 9, labels=False, duplicates="drop")
        return list(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(idx, bins))
    return list(KFold(n_folds, shuffle=True, random_state=seed).split(idx))


# ------------------------------------------------------------------------- defaults
LGB_BASE = dict(n_estimators=10000, learning_rate=0.03, num_leaves=63, min_child_samples=40,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                random_state=SEED, n_jobs=-1, verbose=-1)
XGB_BASE = dict(n_estimators=10000, learning_rate=0.03, max_depth=7, min_child_weight=5,
                subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, tree_method="hist",
                random_state=SEED, n_jobs=-1, enable_categorical=True, max_cat_to_onehot=1)
CAT_BASE = dict(iterations=10000, learning_rate=0.05, depth=7, l2_leaf_reg=3.0,
                random_seed=SEED, thread_count=-1, verbose=0, allow_writing_files=False)

_LGB_OBJ = {"binary": "binary", "multiclass": "multiclass", "regression": "regression_l1"}
_XGB_OBJ = {"binary": "binary:logistic", "multiclass": "multi:softprob", "regression": "reg:absoluteerror"}
_CAT_OBJ = {"binary": "Logloss", "multiclass": "MultiClass", "regression": "MAE"}


def _cat_frame(X, cat_cols):
    """CatBoost wants categorical columns as strings without NaN."""
    if not cat_cols:
        return X
    X = X.copy()
    for c in cat_cols:
        # fill BEFORE casting: pandas >= 3 keeps NaN as NaN under astype(str)
        X[c] = X[c].astype(object).where(X[c].notna(), "NA").map(str).astype(object)
    return X


# --------------------------------------------------------------------- single fit
def fit_predict(kind, task, params, Xtr, ytr, Xva, yva, X_pred_list, cat_cols=None,
                early_stopping=200, sample_weight=None):
    """Fit one model with early stopping on (Xva, yva); return (model, [preds...], best_iter).

    Classification preds are probabilities ((n,) for binary, (n, k) for multiclass);
    regression preds are raw model outputs.
    """
    if kind == "lgb":
        p = {**LGB_BASE, "objective": _LGB_OBJ[task], **params}
        Model = lgb.LGBMRegressor if task == "regression" else lgb.LGBMClassifier
        model = Model(**p)
        model.fit(Xtr, ytr, sample_weight=sample_weight, eval_set=[(Xva, yva)],
                  callbacks=[lgb.early_stopping(early_stopping, verbose=False)])
        best = model.best_iteration_
    elif kind == "xgb":
        p = {**XGB_BASE, "objective": _XGB_OBJ[task], "early_stopping_rounds": early_stopping, **params}
        Model = xgb.XGBRegressor if task == "regression" else xgb.XGBClassifier
        model = Model(**p)
        model.fit(Xtr, ytr, sample_weight=sample_weight, eval_set=[(Xva, yva)], verbose=False)
        best = model.best_iteration
    elif kind == "cat":
        p = {**CAT_BASE, "loss_function": _CAT_OBJ[task], "early_stopping_rounds": early_stopping, **params}
        Model = CatBoostRegressor if task == "regression" else CatBoostClassifier
        model = Model(**p)
        Xtr, Xva = _cat_frame(Xtr, cat_cols), _cat_frame(Xva, cat_cols)
        X_pred_list = [_cat_frame(X, cat_cols) for X in X_pred_list]
        model.fit(Pool(Xtr, ytr, cat_features=cat_cols, weight=sample_weight),
                  eval_set=Pool(Xva, yva, cat_features=cat_cols), use_best_model=True)
        best = model.get_best_iteration()
    else:
        raise ValueError(kind)

    preds = []
    for X in X_pred_list:
        if task == "regression":
            preds.append(model.predict(X))
        else:
            pr = model.predict_proba(X)
            preds.append(pr[:, 1] if task == "binary" else pr)
    return model, preds, best


# ---------------------------------------------------------------------- CV driver
def run_cv(kind, task, params, X, y, X_test, folds, cat_cols=None, early_stopping=200,
           weight_fn=None, n_classes=None, verbose=True, name=None, fold_fn=None):
    """Out-of-fold training. Returns dict(oof, test, best_iters, importance, models).

    fold_fn(X_tr, y_tr, [X_va, X_test]) -> (X_tr, [X_va, X_test]) lets supervised encoders
    (e.g. target encoding) be fitted inside each fold on training rows only."""
    n, nt = len(X), len(X_test)
    shape = (n, n_classes) if task == "multiclass" else (n,)
    oof = np.zeros(shape)
    test = np.zeros((nt, n_classes) if task == "multiclass" else (nt,))
    best_iters, models, imps = [], [], []
    y = np.asarray(y)
    for f, (tr, va) in enumerate(folds):
        sw = weight_fn(y[tr]) if weight_fn is not None else None
        X_tr, X_va, X_te = X.iloc[tr], X.iloc[va], X_test
        if fold_fn is not None:
            X_tr, (X_va, X_te) = fold_fn(X_tr, y[tr], [X_va, X_te])
        model, (p_va, p_te), best = fit_predict(
            kind, task, params, X_tr, y[tr], X_va, y[va], [X_va, X_te],
            cat_cols=cat_cols, early_stopping=early_stopping, sample_weight=sw)
        oof[va] = p_va
        test += p_te / len(folds)
        best_iters.append(best)
        models.append(model)
        imps.append(_importance(kind, model, X_tr.columns))
        if verbose:
            print(f"  [{name or kind}] fold {f} best_iter={best}", end="\r")
    if verbose:
        print(f"  [{name or kind}] done, mean best_iter={np.mean(best_iters):.0f}        ")
    imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
    return dict(oof=oof, test=test, best_iters=best_iters, importance=imp, models=models)


def _importance(kind, model, cols):
    if kind == "lgb":
        v = model.booster_.feature_importance("gain")
    elif kind == "xgb":
        sc = model.get_booster().get_score(importance_type="gain")
        v = [sc.get(c, 0.0) for c in cols]
    elif kind == "cat":
        v = model.get_feature_importance()
    else:
        v = np.zeros(len(cols))
    s = pd.Series(np.asarray(v, dtype=float), index=cols)
    return s / (s.sum() + 1e-12)


def fit_fixed(kind, task, params, X, y, n_iter, X_pred, cat_cols=None):
    """Fit with a fixed number of boosting rounds and NO hold-out (final Train+Val model)."""
    if kind == "lgb":
        model = (lgb.LGBMRegressor if task == "regression" else lgb.LGBMClassifier)(
            **{**LGB_BASE, "objective": _LGB_OBJ[task], **params, "n_estimators": n_iter})
        model.fit(X, y)
    elif kind == "xgb":
        model = (xgb.XGBRegressor if task == "regression" else xgb.XGBClassifier)(
            **{**XGB_BASE, "objective": _XGB_OBJ[task], **params, "n_estimators": n_iter})
        model.fit(X, y, verbose=False)
    elif kind == "cat":
        model = (CatBoostRegressor if task == "regression" else CatBoostClassifier)(
            **{**CAT_BASE, "loss_function": _CAT_OBJ[task], **params, "iterations": n_iter})
        X, X_pred = _cat_frame(X, cat_cols), _cat_frame(X_pred, cat_cols)
        model.fit(Pool(X, y, cat_features=cat_cols))
    else:
        raise ValueError(kind)
    if task == "regression":
        return model.predict(X_pred)
    pr = model.predict_proba(X_pred)
    return pr[:, 1] if task == "binary" else pr


def refit_full(kind, task, params, X, y, X_test, folds, cat_cols=None, scale=1.15, early_stopping=200, verbose=True, name=None):
    """'Combine Train+Val' final model: the round count is found by early stopping on fold 0
    (80% of the data), scaled by `scale` for the larger training set, then the model is refitted on
    100% of the training rows. Returns dict(test=..., n_iter=...)."""
    y = np.asarray(y)
    tr, va = folds[0]
    _, _, best = fit_predict(kind, task, params, X.iloc[tr], y[tr], X.iloc[va], y[va], [X.iloc[va[:1]]],
                             cat_cols=cat_cols, early_stopping=early_stopping)
    n_iter = max(50, int(round((best + 1) * scale)))
    test = fit_fixed(kind, task, params, X, y, n_iter, X_test, cat_cols=cat_cols)
    if verbose:
        print(f"  [{name or kind} full refit] rounds={n_iter} (fold-0 best {best + 1} x {scale})")
    return dict(test=test, n_iter=n_iter)


def refit_full_sklearn(make_model, task, X, y, X_test):
    m = make_model().fit(X, np.asarray(y))
    if task == "regression":
        return dict(test=m.predict(X_test))
    p = m.predict_proba(X_test)
    return dict(test=p[:, 1] if task == "binary" else p)


def run_cv_sklearn(make_model, task, X, y, X_test, folds, n_classes=None, name="sk", verbose=True):
    """OOF loop for any sklearn-style estimator (used for the non-GBDT stack members)."""
    n, nt = len(X), len(X_test)
    oof = np.zeros((n, n_classes) if task == "multiclass" else (n,))
    test = np.zeros((nt, n_classes) if task == "multiclass" else (nt,))
    y = np.asarray(y)
    for f, (tr, va) in enumerate(folds):
        m = make_model()
        m.fit(X[tr], y[tr])
        if task == "regression":
            oof[va] = m.predict(X[va]); test += m.predict(X_test) / len(folds)
        else:
            pv, pt = m.predict_proba(X[va]), m.predict_proba(X_test)
            if task == "binary":
                pv, pt = pv[:, 1], pt[:, 1]
            oof[va] = pv; test += pt / len(folds)
        if verbose:
            print(f"  [{name}] fold {f}", end="\r")
    if verbose:
        print(f"  [{name}] done            ")
    return dict(oof=oof, test=test)
