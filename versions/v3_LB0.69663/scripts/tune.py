"""Optuna (TPE Bayesian optimisation) tuning for Tasks 1-3, scored with each task's own metric.

Studies persist to outputs/optuna.db so the notebook can resume/plot them; the best params are
also written to outputs/params/task{t}_{model}.json.

    python -u scripts/tune.py --task 1 --model lgb --trials 40
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import OUT, SEED  # noqa: E402
from src.metrics import macro_f1, nmae, optimize_threshold  # noqa: E402
from src.modeling import make_folds, run_cv  # noqa: E402
from src.utils import save_params  # noqa: E402

warnings.filterwarnings("ignore")
STORAGE = f"sqlite:///{(OUT / 'optuna.db').as_posix()}"


def space(trial, model, task):
    if model == "lgb":
        p = dict(learning_rate=trial.suggest_float("learning_rate", 0.03, 0.12, log=True),
                 num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
                 min_child_samples=trial.suggest_int("min_child_samples", 10, 400, log=True),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
                 subsample=trial.suggest_float("subsample", 0.5, 1.0),
                 reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                 reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
                 min_split_gain=trial.suggest_float("min_split_gain", 0.0, 0.5),
                 n_jobs=12)
        if task == "binary":
            p["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 1.0, 4.0)
    elif model == "xgb":
        p = dict(learning_rate=trial.suggest_float("learning_rate", 0.03, 0.12, log=True),
                 max_depth=trial.suggest_int("max_depth", 3, 10),
                 min_child_weight=trial.suggest_float("min_child_weight", 1, 100, log=True),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
                 subsample=trial.suggest_float("subsample", 0.5, 1.0),
                 reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                 reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
                 gamma=trial.suggest_float("gamma", 0.0, 2.0), n_jobs=12)
        if task == "binary":
            p["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 1.0, 4.0)
    else:  # catboost
        p = dict(learning_rate=trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
                 depth=trial.suggest_int("depth", 4, 9),
                 l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.5, 30, log=True),
                 random_strength=trial.suggest_float("random_strength", 0.1, 5, log=True),
                 bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 2.0),
                 thread_count=12)
        if task == "binary":
            p["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 1.0, 4.0)
    return p


def get_task(t):
    if t == 1:
        from src import task1
        tr, te, _ = task1.load()
        X, y, Xt = task1.build(tr, te)
        return X, y, Xt, "binary", make_folds(y, n_folds=3), None, None
    if t == 2:
        from src import task2
        tr, te = task2.load()
        X, y, Xt = task2.build(tr, te)
        cats = [c for c in X.columns if str(X[c].dtype) == "category"]
        # in-fold target encoding was neutral in the ablation, so tune the final (native-categorical) setup
        return X, y, Xt, "multiclass", make_folds(y, n_folds=3), cats, None
    from src import task3
    tr, te = task3.load()
    X, y, Xt = task3.build(tr, te)
    return X, y, Xt, "regression", make_folds(y, n_folds=3, kind="regression"), None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--model", default="lgb", choices=["lgb", "xgb", "cat"])
    ap.add_argument("--trials", type=int, default=40)
    args = ap.parse_args()

    X, y, Xt, task, folds, cats, fold_fn = get_task(args.task)
    X_small = Xt.iloc[:10]  # test predictions are irrelevant while tuning

    def objective(trial):
        params = space(trial, args.model, task)
        if task == "regression":  # one-stage conditional-median (L1) model on the raw THB target
            r = run_cv(args.model, task, params, X, y, X_small, folds, cat_cols=cats,
                       early_stopping=100, verbose=False)
            score = nmae(y, np.clip(r["oof"], 0, None))
        elif task == "binary":
            r = run_cv(args.model, task, params, X, y, X_small, folds, early_stopping=100, verbose=False)
            score = optimize_threshold(y, r["oof"], 2.0)[1]
        else:
            r = run_cv(args.model, task, params, X, y, X_small, folds, cat_cols=cats, n_classes=4,
                       early_stopping=100, verbose=False, fold_fn=fold_fn)
            score = macro_f1(y, r["oof"].argmax(1))
        trial.set_user_attr("best_iter", float(np.mean(r["best_iters"])))
        return score

    study = optuna.create_study(study_name=f"task{args.task}_{args.model}", storage=STORAGE,
                                direction="maximize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True))
    remaining = args.trials - len([t for t in study.trials if t.state.name == "COMPLETE"])
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False,
                       callbacks=[lambda s, t: print(f"trial {t.number}: {t.value:.5f} (best {s.best_value:.5f})", flush=True)])
    best = dict(study.best_params)
    print("BEST", study.best_value, best, flush=True)
    save_params(f"task{args.task}_{args.model}", best)


if __name__ == "__main__":
    main()
