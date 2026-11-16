"""Round 30 (Jackpot loop), Task 1: teacher-student self-training on the 13,184 unlabelled TRAINING rows (no test data, CPU).

    python ew_r30_t1_teacher_student.py

train.csv has 13,184 rows whose is_cheater is missing; the pipeline drops them. Semi-supervised self-training (Yarowsky 1995; Lee 2013
"pseudo-label"; Xie et al. 2020 noisy student) uses them inside each training fold only:
  teacher = the pipeline's LightGBM member (tuned parameters, early stopping as in run_cv) fitted on the 4 training folds;
  pseudo-labels on the unlabelled rows: 1 if teacher p > 0.92, 0 if p < 0.03 (the rest unused);
  student = the same LightGBM fitted on the 4 training folds + the pseudo-labelled rows, predicting the held-out fold (OOF).
The held-out fold never touches the teacher, the pseudo-labels or the student's fit (except early stopping, as for every member).
The student enters the v22 level 2 (LogReg + LightGBM + CatBoost on the members' log-odds) as a tenth member; nested on the pipeline
folds and three other stacker splits, against the v22 nine members. Also reported: the student alone vs the teacher alone (OOF F2).
Acceptance (user, 2026-11-16): pooled delta > 0, >= 3/5 folds on each split, no fold below -0.0005.
"""
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import common  # noqa: F401  (chdir to the repo root)
from src import task1  # noqa: E402
from src.config import DATA_DIRS  # noqa: E402
from src.metrics import f2, optimize_threshold  # noqa: E402
from src.modeling import fit_predict, make_folds  # noqa: E402
from src.utils import load_params, load_preds, reduce_mem_usage  # noqa: E402

warnings.filterwarnings("ignore")
t0 = time.time()
raw = pd.read_csv(DATA_DIRS[1] / "train.csv")
lab_rows = raw[task1.TARGET].notna().values
L_df, U_df = raw[lab_rows].reset_index(drop=True), raw[~lab_rows].reset_index(drop=True)
reduce_mem_usage(L_df, verbose=False)
y = L_df[task1.TARGET].astype(int).values
feat_cols = [c for c in L_df.columns if c not in (task1.ID_COL, task1.TARGET)]
X = task1.add_features(L_df, ref=L_df[feat_cols])
XU = task1.add_features(U_df, ref=L_df[feat_cols])[X.columns]
folds = make_folds(y)
P = load_params("task1_lgb")
P["learning_rate"] = P["learning_rate"] / 2                        # as tuned() in the notebook
P["n_jobs"] = 4
print(f"labelled {len(y)}, unlabelled {len(XU)}; LightGBM member parameters {P} [{time.time() - t0:.0f}s]", flush=True)

teach, stud = np.zeros(len(y)), np.zeros(len(y))
for i, (a, b) in enumerate(folds):
    _, (pt, pu), _ = fit_predict("lgb", "binary", P, X.iloc[a], y[a], X.iloc[b], y[b], [X.iloc[b], XU])
    teach[b] = pt
    pos, neg = pu > 0.92, pu < 0.03
    Xs = pd.concat([X.iloc[a], XU[pos | neg]], ignore_index=True)
    ys = np.r_[y[a], pos[pos | neg].astype(int)]
    _, (ps,), _ = fit_predict("lgb", "binary", P, Xs, ys, X.iloc[b], y[b], [X.iloc[b]])
    stud[b] = ps
    print(f"fold {i}: pseudo-labels {pos.sum()} cheaters + {neg.sum()} legitimate of {len(XU)}; teacher F2 "
          f"{f2(y[b], (pt >= optimize_threshold(y[b], pt, 2.0)[0]).astype(int)):.4f}, student "
          f"{f2(y[b], (ps >= optimize_threshold(y[b], ps, 2.0)[0]).astype(int)):.4f} (in-fold cut-offs) [{time.time() - t0:.0f}s]", flush=True)
np.savez("experiments/claude/outputs/r30/t1_student_oof.npz", teacher=teach, student=stud)
print(f"OOF corr(teacher, pipeline LightGBM) {np.corrcoef(teach, load_preds('t1_lgb')['oof'])[0, 1]:.4f}; corr(student, teacher) "
      f"{np.corrcoef(teach, stud)[0, 1]:.4f}")

logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
KEYS = ["lgb", "xgb", "cat", "et", "lr", "rf24", "lgb24", "xgb24", "cat24"]
base_cols = [logit(load_preds("t1_lr")["oof"]) if k == "lr" else logit(load_preds(f"t1_bag10_{k}")["oof"]) for k in KEYS]
LGBP = dict(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            n_jobs=4, verbose=-1)
MAKE = [lambda: [LogisticRegression(C=1.0, max_iter=2000)],
        lambda: [lgb.LGBMClassifier(**LGBP, random_state=s) for s in (0, 1, 2)],
        lambda: [CatBoostClassifier(depth=4, iterations=400, learning_rate=0.05, random_seed=s, verbose=0, thread_count=4,
                                    allow_writing_files=False) for s in (0, 1, 2)]]


def level2(cols, fl):
    S = np.column_stack(cols)
    p = np.zeros(len(y))
    for a, b in fl:
        p[b] = expit(np.mean([np.mean([logit(m.fit(S[a], y[a]).predict_proba(S[b])[:, 1]) for m in mk()], 0) for mk in MAKE], 0))
    fo = np.zeros(len(y), int)
    for i, (_, b) in enumerate(fl):
        fo[b] = i
    lab = np.zeros(len(y), int)
    for i in range(5):
        t = optimize_threshold(y[fo != i], p[fo != i], 2.0)[0]
        lab[fo == i] = p[fo == i] >= t
    return lab, fo


splits = [("pipeline folds", folds)] + [(f"split seed {s}", list(StratifiedKFold(5, shuffle=True, random_state=s).split(base_cols[0], y))) for s in (1, 2, 3)]
rows = []
for label, fl in splits:
    base, fo = level2(base_cols, fl)
    cand, _ = level2(base_cols + [logit(stud)], fl)
    pf = [f2(y[fo == i], cand[fo == i]) - f2(y[fo == i], base[fo == i]) for i in range(5)]
    d = f2(y, cand) - f2(y, base)
    rows.append((d, sum(x > 0 for x in pf), min(pf)))
    print(f"{label}: v22 {f2(y, base):.5f}, + student member {f2(y, cand):.5f}, delta {d:+.5f}; folds +{sum(x > 0 for x in pf)}/5 "
          f"({' '.join(f'{x:+.5f}' for x in pf)}) [{time.time() - t0:.0f}s]", flush=True)
ok = np.mean([r[0] for r in rows]) > 0 and all(r[1] >= 3 and r[2] >= -0.0005 for r in rows)
print(f"\npooled mean delta {np.mean([r[0] for r in rows]):+.5f}; folds better {' '.join(str(r[1]) + '/5' for r in rows)}; "
      f"worst fold {' '.join(f'{r[2]:+.5f}' for r in rows)} -> {'PASS' if ok else 'fail'} under the 2026-11-16 rule")
