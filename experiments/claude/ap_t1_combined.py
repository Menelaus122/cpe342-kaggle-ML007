"""Task 1 round 6b: do the small positives add up? Reduced feature set (ao_t1_round6 F1) for all three GBDTs,
3-seed bagging for LightGBM, both feature views in the stack, and per-missingness thresholds. Same validation."""
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import common  # noqa
from common import load_t1, T1_MODELS, _logit, tuned, OUT
from src.utils import load_preds
from src.modeling import fit_predict
from src.metrics import optimize_threshold, fbeta
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)
tr, te, X, y, Xt, folds, _ = load_t1()
spw = (y == 0).sum() / (y == 1).sum()
P1_lgb = tuned("task1_lgb", {"learning_rate": 0.05, "scale_pos_weight": spw})
P1_cat = tuned("task1_cat", {"learning_rate": 0.06, "depth": 6, "scale_pos_weight": spw})
P1_xgb = {"learning_rate": 0.03, "max_depth": 6, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.6, "scale_pos_weight": spw}
base = {n: load_preds(n) for n in T1_MODELS}
f2 = lambda yt, yp: fbeta(yt, yp, 2, "binary")
DROP = ['device_changes_count', 'input_consistency_score', 'avg_session_length_min', 'sessions_per_day', 'night_play_ratio',
        'weapon_switch_speed', 'movement_pattern_score', 'communication_rate', 'team_play_score', 'buy_decision_score',
        'map_knowledge', 'clutch_success_rate', 'first_blood_rate', 'survival_time_avg', 'damage_per_round',
        'utility_usage_rate', 'dmg_per_survival', 'daily_playtime']
keep = [c for c in X.columns if c not in DROP]
nmb = np.minimum(X["n_missing"].values, 6) // 2


def stack(oofs):
    S = np.column_stack([_logit(o) for o in oofs]); o = np.zeros(len(y))
    for a, b in folds:
        o[b] = LogisticRegression(max_iter=3000).fit(S[a], y[a]).predict_proba(S[b])[:, 1]
    return o


def nested(p, groups=None):
    pred = np.zeros(len(y), int)
    for a, b in folds:
        if groups is None:
            t, _ = optimize_threshold(y[a], p[a], 2.0, average="binary"); pred[b] = p[b] >= t
        else:
            for g in np.unique(groups):
                aa, bb = a[groups[a] == g], b[groups[b] == g]
                t, _ = optimize_threshold(y[aa], p[aa], 2.0, average="binary"); pred[bb] = p[bb] >= t
    return pred


def boot(pn, po, B=2000):
    yb = y.astype(bool)
    M = np.column_stack([yb & (pn == 1), ~yb & (pn == 1), yb & (pn == 0), yb & (po == 1), ~yb & (po == 1), yb & (po == 0)]).astype(float)
    d = np.empty(B)
    for i in range(B):
        c = rng.poisson(1.0, len(y)) @ M
        d[i] = 5 * c[0] / (5 * c[0] + 4 * c[2] + c[1]) - 5 * c[3] / (5 * c[3] + 4 * c[5] + c[4])
    return np.percentile(d, [2.5, 97.5])


red = {}
lg = np.zeros(len(y)); xg = np.zeros(len(y)); ct = np.zeros(len(y))
for f, (a, b) in enumerate(folds):
    Xa, Xb = X.iloc[a][keep], X.iloc[b][keep]
    for s in range(3):
        _, (pb,), _ = fit_predict("lgb", "binary", {**P1_lgb, "random_state": 7 + s}, Xa, y[a], Xb, y[b], [Xb]); lg[b] += pb / 3
    _, (pb,), _ = fit_predict("xgb", "binary", P1_xgb, Xa, y[a], Xb, y[b], [Xb]); xg[b] = pb
    _, (pb,), _ = fit_predict("cat", "binary", P1_cat, Xa, y[a], Xb, y[b], [Xb]); ct[b] = pb
    print(f"fold {f}: LGB-bag {roc_auc_score(y[b], lg[b]):.4f} (orig {roc_auc_score(y[b], base['t1_lgb']['oof'][b]):.4f}), "
          f"XGB {roc_auc_score(y[b], xg[b]):.4f} (orig {roc_auc_score(y[b], base['t1_xgb']['oof'][b]):.4f}), "
          f"Cat {roc_auc_score(y[b], ct[b]):.4f} (orig {roc_auc_score(y[b], base['t1_cat']['oof'][b]):.4f})", flush=True)
np.savez(OUT / "t1_round6b_members.npz", lgb=lg, xgb=xg, cat=ct)
p0 = stack([base[n]["oof"] for n in T1_MODELS]); pred0 = nested(p0)
orig = [base[n]["oof"] for n in T1_MODELS]
cands = {"C1 reduced GBDTs (LGB 3-seed bag, XGB, Cat) + ET + LR": [lg, xg, ct, orig[3], orig[4]],
         "C2 both views: 5 original + 3 reduced members": orig + [lg, xg, ct]}
rows = [["B0", "notebook stack", roc_auc_score(y, p0), f2(y, pred0), 0.0, "-", "baseline"]]
for name, oofs in cands.items():
    p = stack(oofs)
    for tag, pr in [("global tau", nested(p)), ("tau per missing-count bucket", nested(p, nmb))]:
        lo, hi = boot(pr, pred0)
        rows.append([name.split()[0], f"{name[3:]}, {tag}", roc_auc_score(y, p), f2(y, pr), f2(y, pr) - f2(y, pred0),
                     f"[{lo:+.4f}, {hi:+.4f}]", "KEEP" if lo > 0 else "discard"])
        print(rows[-1], flush=True)
res = pd.DataFrame(rows, columns=["Experiment", "Changes", "stack AUC", "OOF F2 (nested tau)", "Delta vs B0", "95% CI", "Decision"])
print("\n" + res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
