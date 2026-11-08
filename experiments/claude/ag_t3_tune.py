"""Task 3 tuning round (Optuna TPE, same search space style as scripts/tune.py, same 5 folds as the notebook).

Study A  ratio stage 2: LightGBM L1 on r = log1p(y) - log1p(historical_spending), positive rows with a known base.
         Objective: mean |error| of log1p(y) on the validation folds' positive rows.
Study B  stage 1 (shared by every hurdle read-out and the zero gate): LightGBM P(y > 0). Objective: OOF log-loss.
Features: identity fill + imputed segment (as v7). Studies live in outputs/t3_tuning.db (resumable);
best params are written to outputs/t3_tuned_params.json.
"""
import json
import sys
import warnings

import numpy as np
import optuna
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
import common  # noqa: E402
from common import OUT  # noqa: E402
from src import task3  # noqa: E402
from src.modeling import make_folds, fit_predict  # noqa: E402
from t3_fill import fill  # noqa: E402
from t3_segment import impute_segment  # noqa: E402

N_A, N_B = (int(a) for a in (sys.argv[1:3] if len(sys.argv) > 2 else (40, 30)))
tr, te = task3.load()
y = tr[task3.TARGET].values.astype(float)
z, pos = np.log1p(y), y > 0
folds = make_folds(y, kind="regression")
trf, tef = fill(tr)[0], fill(te)[0]
X, _, _ = impute_segment(task3.add_features(trf), task3.add_features(tef))
b = np.log1p(trf.historical_spending.values)
okb = np.isfinite(b)
storage = f"sqlite:///{(OUT / 't3_tuning.db').as_posix()}"


def space(trial):
    return dict(learning_rate=trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
                num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
                min_child_samples=trial.suggest_int("min_child_samples", 10, 400, log=True),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
                subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
                reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
                min_split_gain=trial.suggest_float("min_split_gain", 0.0, 0.5))


def eval_ratio(p):
    p = {"objective": "regression_l1", **p}
    err = []
    for a, v in folds:
        ap, vp = a[pos[a] & okb[a]], v[pos[v] & okb[v]]
        _, (rv,), it = fit_predict("lgb", "regression", p, X.iloc[ap], z[ap] - b[ap], X.iloc[vp], z[vp] - b[vp], [X.iloc[vp]])
        err.append(np.mean(np.abs(z[vp] - b[vp] - rv)))
    return float(np.mean(err))


def eval_stage1(p):
    o = np.zeros(len(y))
    for a, v in folds:
        _, (pv,), _ = fit_predict("lgb", "binary", p, X.iloc[a], pos[a].astype(int), X.iloc[v], pos[v].astype(int), [X.iloc[v]])
        o[v] = pv
    return float(log_loss(pos, np.clip(o, 1e-6, 1 - 1e-6)))


out = {}
for name, ev, n in [("t3_ratio_stage2", eval_ratio, N_A), ("t3_stage1", eval_stage1, N_B)]:
    obj = lambda trial, ev=ev: ev(space(trial))
    st = optuna.create_study(study_name=name, storage=storage, direction="minimize", load_if_exists=True,
                             sampler=optuna.samplers.TPESampler(seed=0))
    todo = n - len([t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if todo > 0:
        st.optimize(obj, n_trials=todo)
    base_val = ev({"learning_rate": 0.05})                # the v7 setting (LGB_BASE + lr 0.05)
    print(f"{name}: v7 setting {base_val:.5f} -> best {st.best_value:.5f} after {len(st.trials)} trials; "
          f"best params { {k: round(v, 4) if isinstance(v, float) else v for k, v in st.best_params.items()} }", flush=True)
    out[name] = {**st.best_params, "subsample_freq": 1}
(OUT / "t3_tuned_params.json").write_text(json.dumps(out, indent=1))
