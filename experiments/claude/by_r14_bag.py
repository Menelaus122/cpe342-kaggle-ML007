"""Round 14, step 4: 10-seed bagging of the exact winning members (features, hyper-parameters and folds unchanged; only the model seed varies).

    python by_r14_bag.py cv t1        # 5-fold OOF + fold-model test predictions for every member and seed
    python by_r14_bag.py full t1      # Train+Val refits, run only for tasks that pass the acceptance rule in bz_r14_eval.py

Results are saved per member and seed in outputs/r14_bag/, so an interrupted run resumes where it stopped.
Data preparation follows the notebook (reduce_mem_usage on the training frames only) and the seed-42 fits are compared with the pipeline cache.

Task 1: 8 of the 9 stack members (LogReg is deterministic, cached). Task 2 (clean pipeline only; the v12 column is the id block): LightGBM, XGBoost, CatBoost.
Task 3: hurdle stages 1 and 2 and the four ratio stage-2 models that carry 99.9% of the blend weight; zero-weight members stay cached.
"""
import sys
import time
import warnings

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

warnings.filterwarnings("ignore")
import common  # noqa: F401,E402
from common import OUT, tuned  # noqa: E402
from src import task1, task2, task3  # noqa: E402
from src.modeling import fit_fixed, fit_predict, make_folds  # noqa: E402
from src.utils import load_preds, reduce_mem_usage  # noqa: E402

SEEDS = (42, 100, 2026, 777, 999, 1234, 5555, 8888, 31415, 27182)
DIR = OUT / "r14_bag"
DIR.mkdir(exist_ok=True)
T0 = time.time()
log = lambda *a: print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)
SEED_KW = {"lgb": "random_state", "xgb": "random_state", "cat": "random_seed"}


def full_iters(best):
    """Same round count as modeling.refit_full: fold-0 best iteration x 1.15."""
    return max(50, int(round((best + 1) * 1.15)))


def cv_gbdt(kind, task, params, A, y, At, folds, Ae=None, cat_cols=None, k=None):
    shape = (len(y), k) if k else (len(y),)
    oof, tl = np.zeros(shape), np.zeros(shape)
    test = np.zeros((len(At), k) if k else (len(At),))
    best = []
    for a, b in folds:
        lst = [A.iloc[b], At] + ([Ae.iloc[b]] if Ae is not None else [])
        _, pr, bi = fit_predict(kind, task, params, A.iloc[a], y[a], A.iloc[b], y[b], lst, cat_cols=cat_cols)
        oof[b] = pr[0]
        test += pr[1] / len(folds)
        if Ae is not None:
            tl[b] = pr[2]
        best.append(bi)
    return dict(oof=oof, test=test, tl=tl, best=np.array(best))


def cv_sk(make, A, y, At, folds, Ae):
    oof, tl, test = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(At))
    for a, b in folds:
        m = make().fit(A[a], y[a])
        oof[b], tl[b] = m.predict_proba(A[b])[:, 1], m.predict_proba(Ae[b])[:, 1]
        test += m.predict_proba(At)[:, 1] / len(folds)
    return dict(oof=oof, test=test, tl=tl)


def check(name, new, key, field="oof"):
    old = load_preds(key)[field]
    print(f"    seed-42 check {name}: max |new - cached| {np.abs(new - old).max():.5g}, corr {np.corrcoef(new.ravel(), old.ravel())[0, 1]:.6f}", flush=True)


def run(task, mem, seed, fn):
    p = DIR / f"{task}_{mem}_{seed}.npz"
    if p.exists():
        return np.load(p)
    t0 = time.time()
    r = fn()
    np.savez(p, **r)
    log(f"{task} {mem} seed {seed}: {time.time() - t0:.0f}s")
    return np.load(p)


# ------------------------------------------------------------------------------- task 1
def setup_t1():
    tr, te, _ = task1.load()
    reduce_mem_usage(tr, verbose=False)
    X, y, Xt = task1.build(tr, te)
    folds = make_folds(y)
    raw = [c for c in tr.columns if c not in ("player_id", "is_cheater")]
    q_tr, q_te = tr[raw].isna().mean(), te[raw].isna().mean()
    mrng = np.random.default_rng(123)
    raw_tl = tr[raw].copy()
    for c in [c for c in raw if q_te[c] > q_tr[c]]:
        raw_tl.loc[raw_tl[c].notna().values & (mrng.random(len(raw_tl)) < (q_te[c] - q_tr[c]) / (1 - q_tr[c])), c] = np.nan
    X_tl = task1.add_features(raw_tl, ref=tr[raw])[X.columns]
    R, Rt = task1.build_c24(tr, te)
    med = tr[task1.C24].median()
    R_tl = raw_tl[task1.C24].fillna(med).reset_index(drop=True)
    spw = (y == 0).sum() / (y == 1).sum()
    P = {"lgb": tuned("task1_lgb", {"learning_rate": 0.05, "scale_pos_weight": spw}),
         "xgb": {"learning_rate": 0.03, "max_depth": 6, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.6, "scale_pos_weight": spw},
         "cat": tuned("task1_cat", {"learning_rate": 0.06, "depth": 6, "scale_pos_weight": spw})}
    P24 = {"lgb": {"learning_rate": 0.05, "num_leaves": 31}, "xgb": {"learning_rate": 0.05, "max_depth": 6}, "cat": {"learning_rate": 0.08, "depth": 6}}
    return X, y, Xt, X_tl, R, Rt, R_tl, folds, P, P24


def make_et(s):
    return lambda: ExtraTreesClassifier(n_estimators=400, min_samples_leaf=4, max_features=0.4, class_weight="balanced", n_jobs=-1, random_state=s)


def make_rf(s):
    return lambda: RandomForestClassifier(n_estimators=300, max_depth=10, max_features=0.7, n_jobs=-1, random_state=s)


def cv_t1():
    X, y, Xt, X_tl, R, Rt, R_tl, folds, P, P24 = setup_t1()
    for s in SEEDS:
        for k in ("lgb", "xgb", "cat"):
            r = run("t1", k, s, lambda: cv_gbdt(k, "binary", {**P[k], SEED_KW[k]: s}, X, y, Xt, folds, X_tl))
            if s == 42:
                check(f"t1_{k}", r["oof"], f"t1_{k}")
        r = run("t1", "et", s, lambda: cv_sk(make_et(s), X.values, y, Xt.values, folds, X_tl.values))
        if s == 42:
            check("t1_et", r["oof"], "t1_et")
        run("t1", "rf24", s, lambda: cv_sk(make_rf(s), R.values, y, Rt.values, folds, R_tl.values))
        for k in ("lgb", "xgb", "cat"):
            run("t1", f"{k}24", s, lambda: cv_gbdt(k, "binary", {**P24[k], SEED_KW[k]: s}, R, y, Rt, folds, R_tl))
        log(f"T1 seed {s} complete")


def full_t1():
    X, y, Xt, X_tl, R, Rt, R_tl, folds, P, P24 = setup_t1()
    for s in SEEDS:
        for k in ("lgb", "xgb", "cat"):
            b = np.load(DIR / f"t1_{k}_{s}.npz")["best"][0]
            run("t1full", k, s, lambda: dict(test=fit_fixed(k, "binary", {**P[k], SEED_KW[k]: s}, X, y, full_iters(b), Xt)))
            b = np.load(DIR / f"t1_{k}24_{s}.npz")["best"][0]
            run("t1full", f"{k}24", s, lambda: dict(test=fit_fixed(k, "binary", {**P24[k], SEED_KW[k]: s}, R, y, full_iters(b), Rt)))
        run("t1full", "et", s, lambda: dict(test=make_et(s)().fit(X.values, y).predict_proba(Xt.values)[:, 1]))
        run("t1full", "rf24", s, lambda: dict(test=make_rf(s)().fit(R.values, y).predict_proba(Rt.values)[:, 1]))
        log(f"T1 full refits seed {s} complete")


# ------------------------------------------------------------------------------- task 2
def setup_t2():
    tr, te = task2.load()
    reduce_mem_usage(tr, verbose=False)
    X, y, Xt = task2.build(tr, te)
    cat = [c for c in X.columns if str(X[c].dtype) == "category"]
    P = {"lgb": tuned("task2_lgb", {"learning_rate": 0.05}), "cat": tuned("task2_cat", {"learning_rate": 0.08, "depth": 6}),
         "xgb": {"learning_rate": 0.04, "max_depth": 5, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.5}}
    return X, y, Xt, cat, make_folds(y), P


def cv_t2():
    X, y, Xt, cat, folds, P = setup_t2()
    for s in SEEDS:
        for k in ("lgb", "xgb", "cat"):
            cc = cat if k == "cat" else None
            r = run("t2", k, s, lambda: cv_gbdt(k, "multiclass", {**P[k], SEED_KW[k]: s}, X, y, Xt, folds, cat_cols=cc, k=4))
            if s == 42:
                check(f"t2_{k}", r["oof"], f"t2_{k}")
        log(f"T2 seed {s} complete")


def full_t2():
    X, y, Xt, cat, folds, P = setup_t2()
    for s in SEEDS:
        for k in ("lgb", "xgb", "cat"):
            b = np.load(DIR / f"t2_{k}_{s}.npz")["best"][0]
            run("t2full", k, s, lambda: dict(test=fit_fixed(k, "multiclass", {**P[k], SEED_KW[k]: s}, X, y, full_iters(b), Xt, cat_cols=cat if k == "cat" else None)))
        log(f"T2 full refits seed {s} complete")


# ------------------------------------------------------------------------------- task 3
BAG3 = {"ratio_hist_lgb_tuned": ("lgb", "tuned", "historical_spending", "t3r_ratio_hist_lgb_tuned"),
        "ratio_prev_lgb_tuned": ("lgb", "tuned", "prev_month_spending", "t3r_ratio_prev_lgb_tuned"),
        "ratio_hist_xgb": ("xgb", {"learning_rate": 0.05}, "historical_spending", "t3r_ratio_hist_xgb"),
        "ratio_hist_cat": ("cat", {"learning_rate": 0.08, "depth": 8}, "historical_spending", "t3r_ratio_hist_cat")}


def setup_t3():
    tr, te = task3.load()
    reduce_mem_usage(tr, verbose=False)
    X, y, Xt, _ = task3.build_repaired(tr, te)
    folds = make_folds(y, kind="regression")
    S1 = tuned("task3_stage1", {"learning_rate": 0.05})
    R2 = tuned("task3_ratio_stage2", {"learning_rate": 0.05})
    return X, y, Xt, folds, S1, R2


def cv_t3():
    X, y, Xt, folds, S1, R2 = setup_t3()
    for s in SEEDS:
        h = run("t3", "hurdle", s, lambda: {k: v for k, v in task3.hurdle_cv("lgb", X, y, Xt, folds, clf_params={**S1, "random_state": s},
                                                                             reg_params={"learning_rate": 0.05, "random_state": s}, verbose=False).items()})
        if s == 42:
            check("t3r_hurdle p", h["p_oof"], "t3r_hurdle", "p_oof")
            check("t3r_hurdle mu", h["mu_oof"], "t3r_hurdle", "mu_oof")
        for nm, (kind, prm, bcol, key) in BAG3.items():
            prm = R2 if isinstance(prm, str) else prm
            r = run("t3", nm, s, lambda: task3.ratio_stage2_cv(kind, {**prm, SEED_KW[kind]: s}, X, y, Xt, folds, bcol, h["mu_oof"], h["mu_test"]))
            if s == 42:
                check(key, r["mu_oof"], key, "mu_oof")
        log(f"T3 seed {s} complete")


def full_t3():
    X, y, Xt, folds, S1, R2 = setup_t3()
    for s in SEEDS:
        run("t3full", "hurdle", s, lambda: {k: v for k, v in task3.hurdle_full("lgb", X, y, Xt, folds, {**S1, "random_state": s}, {"learning_rate": 0.05, "random_state": s}).items()})
        for nm, (kind, prm, bcol, _) in BAG3.items():
            prm = R2 if isinstance(prm, str) else prm
            run("t3full", nm, s, lambda: dict(test=task3.ratio_stage2_full(kind, {**prm, SEED_KW[kind]: s}, X, y, Xt, folds, bcol)["test"]))
        log(f"T3 full refits seed {s} complete")


if __name__ == "__main__":
    mode, task = sys.argv[1], sys.argv[2]
    {"cvt1": cv_t1, "cvt2": cv_t2, "cvt3": cv_t3, "fullt1": full_t1, "fullt2": full_t2, "fullt3": full_t3}[mode + task]()
    log("done", mode, task)
