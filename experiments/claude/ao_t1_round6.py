"""Task 1 round 6 (from the deep-dive an_t1_deep.py). Same validation as round 4: nested tau, paired bootstrap.
F1  feature selection: drop raw features with no signal even in distribution shape (KS < 0.012 in the middle
    group) and the engineered features built only from them.
F2  sub-type decomposition: cheaters in the training fold are split into 3 sub-types by a Gaussian mixture
    (fitted in-fold on cheaters only); LightGBM learns clean + 3 cheater classes; P(cheater) = 1 - P(clean).
"""
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
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import QuantileTransformer

rng = np.random.default_rng(0)
tr, te, X, y, Xt, folds, _ = load_t1()
spw = (y == 0).sum() / (y == 1).sum()
P1 = tuned("task1_lgb", {"learning_rate": 0.05, "scale_pos_weight": spw})
base = {n: load_preds(n) for n in T1_MODELS}
f2 = lambda yt, yp: fbeta(yt, yp, 2, "binary")


def stack(oofs):
    S = np.column_stack([_logit(o) for o in oofs]); o = np.zeros(len(y))
    for a, b in folds:
        o[b] = LogisticRegression(max_iter=3000).fit(S[a], y[a]).predict_proba(S[b])[:, 1]
    return o


def nested(p):
    pred = np.zeros(len(y), int)
    for a, b in folds:
        t, _ = optimize_threshold(y[a], p[a], 2.0, average="binary"); pred[b] = p[b] >= t
    return pred


def boot(pn, po, B=1000):
    yb = y.astype(bool)
    M = np.column_stack([yb & (pn == 1), ~yb & (pn == 1), yb & (pn == 0), yb & (po == 1), ~yb & (po == 1), yb & (po == 0)]).astype(float)
    d = np.empty(B)
    for i in range(B):
        c = rng.poisson(1.0, len(y)) @ M
        d[i] = 5 * c[0] / (5 * c[0] + 4 * c[2] + c[1]) - 5 * c[3] / (5 * c[3] + 4 * c[5] + c[4])
    return np.percentile(d, [2.5, 97.5])


p0 = stack([base[n]["oof"] for n in T1_MODELS]); pred0 = nested(p0)
print(f"B0 stack: AUC {roc_auc_score(y, p0):.4f}, nested F2 {f2(y, pred0):.4f}; LightGBM alone AUC {roc_auc_score(y, base['t1_lgb']['oof']):.4f}")
rows = []


def report(name, desc, member):
    print(f"{name}: member AUC {roc_auc_score(y, member):.4f}", flush=True)
    for how, oofs in [("replaces LightGBM", [member] + [base[n]["oof"] for n in T1_MODELS[1:]]),
                      ("extra member", [base[n]["oof"] for n in T1_MODELS] + [member])]:
        p = stack(oofs); pr = nested(p); lo, hi = boot(pr, pred0)
        rows.append([name, f"{desc}, {how}", roc_auc_score(y, p), f2(y, pr), f2(y, pr) - f2(y, pred0),
                     f"[{lo:+.4f}, {hi:+.4f}]", "KEEP" if lo > 0 else "discard"])
        print(f"   {how}: stack AUC {rows[-1][2]:.4f}, nested F2 {rows[-1][3]:.4f} ({rows[-1][4]:+.4f}) CI {rows[-1][5]}", flush=True)


# ---- F1 feature selection
raw = [c for c in tr.columns if c not in ("player_id", "is_cheater")]
from scipy.stats import ks_2samp
mid = (p0 > 0.05) & (p0 < 0.95)
ks_all = {c: ks_2samp(tr[c][y == 1].dropna(), tr[c][y == 0].dropna()).statistic for c in raw}
ks_mid = {c: ks_2samp(tr[c][mid & (y == 1)].dropna(), tr[c][mid & (y == 0)].dropna()).statistic for c in raw}
noise = [c for c in raw if ks_mid[c] < 0.012 and ks_all[c] < 0.012]
fe_from_noise = {"hs_x_smooth": [], "dmg_per_survival": ["damage_per_round", "survival_time_avg"],
                 "daily_playtime": ["avg_session_length_min", "sessions_per_day"], "input_x_smooth": []}
drop = noise + [f for f, srcs in fe_from_noise.items() if srcs and all(s in noise for s in srcs)]
print(f"\nF1: dropping {len(drop)} columns: {drop}")
keep = [c for c in X.columns if c not in drop]
o = np.zeros(len(y))
for a, b in folds:
    _, (pb,), _ = fit_predict("lgb", "binary", P1, X.iloc[a][keep], y[a], X.iloc[b][keep], y[b], [X.iloc[b][keep]])
    o[b] = pb
report("F1", f"LightGBM without {len(drop)} no-signal columns", o)

# ---- F2 sub-type decomposition
sig = [c for c in raw if ks_all[c] > 0.03]
o2 = np.zeros(len(y))
for f, (a, b) in enumerate(folds):
    qt = QuantileTransformer(output_distribution="normal", random_state=0).fit(tr[sig].iloc[a])
    ca = a[y[a] == 1]
    gm = GaussianMixture(3, covariance_type="diag", random_state=0).fit(np.nan_to_num(qt.transform(tr[sig].iloc[ca])))
    lab = np.zeros(len(a), int)
    lab[y[a] == 1] = 1 + gm.predict(np.nan_to_num(qt.transform(tr[sig].iloc[ca])))
    # validation labels for early stopping only need the binary task: use the same mixture on validation cheaters
    cb = b[y[b] == 1]
    labb = np.zeros(len(b), int); labb[y[b] == 1] = 1 + gm.predict(np.nan_to_num(qt.transform(tr[sig].iloc[cb])))
    prm = {k: v for k, v in P1.items() if k != "scale_pos_weight"}
    _, (pb,), _ = fit_predict("lgb", "multiclass", prm, X.iloc[a], lab, X.iloc[b], labb, [X.iloc[b]])
    o2[b] = 1 - pb[:, 0]
    print(f"  fold {f}: sub-type sizes {np.bincount(lab)[1:].tolist()}, AUC {roc_auc_score(y[b], o2[b]):.4f}", flush=True)
report("F2", "4-class LightGBM (clean + 3 in-fold cheater sub-types)", o2)
np.savez(OUT / "t1_round6_members.npz", f1=o, f2=o2)
res = pd.DataFrame(rows, columns=["Experiment", "Changes", "stack AUC", "OOF F2 (nested tau)", "Delta vs B0", "95% CI", "Decision"])
print("\n" + res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
