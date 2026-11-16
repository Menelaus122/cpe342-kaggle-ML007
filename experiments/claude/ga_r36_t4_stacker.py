"""Round 36, Task 4: a logit-space meta-learner over the members and a class-multiplier search (CPU; training images only).

    python ga_r36_t4_stacker.py [extra member tag ...]      e.g. dinov2_r2 once its views exist for all five folds

v23 combines the eight members with fixed weights (DINOv2 x8, the five HUD-dropout members x3, ConvNeXt-T / DeiT-III x1) chosen
nested on the hard views, then argmax p * pi^-0.75. Candidates, all nested (fitted on four folds' out-of-fold member predictions,
applied to the fifth), all followed by the same rule and scored like fc_r31_t4_v23_eval.py (balanced macro-F1 at a uniform class
mix; views full, zoom70, HUD hidden, gamma 3.5, 75% grid; real dark / low-contrast frames on the full view):
  1a  ridge logit stacker: multinomial logistic regression (L2, C = 0.1 and 1) on the members' log-probabilities (5 per member),
      fitted on the full view only;
  1b  the same fitted on the five views pooled (the hard views are what the leaderboard rewards);
  1c  a low-capacity version: one non-negative weight per member on the log-probabilities (weighted geometric mean, weights and
      one temperature by the log-loss, L-BFGS-B), fitted on the five views pooled;
  3   class multipliers on top of the v23 rule by the soft macro-F1 search of Task 2 (balanced row weights), fitted on the full
      view / on the five views pooled.
Rules: the hard-view rule of 2026-11-16 (hard-view mean >= +0.0015 and full >= -0.0005), the generic rule on the hard-view mean
(pooled > 0, >= 3/5 folds, no fold below -0.0005) and the v24 check (zoom70, HUD hidden and dark / low-contrast all > 0 with
full >= -0.0005).
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402

V20 = ["cnxt", "deit3", "cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud", "dinov2_v2"]
HUDM = ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
V23_W = {t: (8.0 if t == "dinov2_v2" else 3.0 if t in HUDM else 1.0) for t in V20}
ALL5 = ["full", "zoom70", "centre_grey", "dark_g35", "grid75"]
HARD = ALL5[1:]
V19, V30 = Path("experiments/claude/outputs/r19_views"), Path("experiments/claude/outputs/r30_views")
EXTRA = [t for t in sys.argv[1:] if all((V30 / f"{t}_fold{f}.npz").exists() for f in range(5))]
MEMBERS = V20 + EXTRA
train, _ = task4.load_frames()
y = train.label.values
folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
fo = np.zeros(len(y), int)
for f, va in enumerate(folds):
    fo[va] = f
P = {t: {v: np.zeros((len(y), 5)) for v in ALL5} for t in MEMBERS}
for t in MEMBERS:
    for f, va in enumerate(folds):
        z30 = np.load(V30 / f"{t}_fold{f}.npz")
        z19 = np.load(V19 / f"{t}_fold{f}.npz") if t in V20 else z30
        for v in ALL5:
            P[t][v][va] = (z30 if v in z30.files else z19)[v]
st = np.load("experiments/claude/outputs/r30/t4_errors.npz")
dark = (st["contrast"] <= np.quantile(st["contrast"], 0.1)) | (st["brightness"] <= np.quantile(st["brightness"], 0.1))
pi = np.bincount(y, minlength=5) / len(y)
wb = (0.2 / pi)[y]
bal = lambda pred, m: f1_score(y[m], pred[m], average="macro", sample_weight=wb[m])
SCORE = ALL5 + ["dark/low-contrast"]
VIEW_OF = {**{v: v for v in ALL5}, "dark/low-contrast": "full"}
allm = np.ones(len(y), bool)
softmax = lambda Z: np.exp(Z - logsumexp(Z, axis=1, keepdims=True))
rule = lambda p: (p * pi ** -0.75).argmax(1)
LOGP = {v: np.column_stack([np.log(np.clip(P[t][v], 1e-6, 1)) for t in MEMBERS]) for v in ALL5}       # rows x (5 * members)
ens = {v: sum(w * P[t][v] for t, w in V23_W.items()) / sum(V23_W.values()) for v in ALL5}
base = {v: rule(ens[v]) for v in ALL5}
print(f"{len(y)} training images; members {MEMBERS}; v23 members (uniform mix, rule): "
      + ", ".join(f"{k} {bal(base[VIEW_OF[k]], dark if k == 'dark/low-contrast' else allm):.4f}" for k in SCORE), flush=True)


def report(name, lab):
    d = {k: (lambda m: bal(lab[VIEW_OF[k]], m) - bal(base[VIEW_OF[k]], m))(dark if k == "dark/low-contrast" else allm) for k in SCORE}
    hard = np.mean([d[v] for v in HARD])
    pf = [np.mean([bal(lab[v], fo == f) - bal(base[v], fo == f) for v in HARD]) for f in range(5)]
    ok_user = hard >= 0.0015 and d["full"] >= -0.0005
    ok_gen = hard > 0 and sum(p > 0 for p in pf) >= 3 and min(pf) >= -0.0005
    ok_v24 = d["zoom70"] > 0 and d["centre_grey"] > 0 and d["dark/low-contrast"] > 0 and d["full"] >= -0.0005
    print(f"  {name}\n    hard-view mean {hard:+.4f}, folds +{sum(p > 0 for p in pf)}/5 ({' '.join(f'{p:+.4f}' for p in pf)}); "
          + ", ".join(f"{k} {d[k]:+.4f}" for k in SCORE)
          + f"\n    -> hard-view rule {'PASS' if ok_user else 'fail'}, generic rule {'PASS' if ok_gen else 'fail'}, v24 check {'PASS' if ok_v24 else 'fail'}",
          flush=True)


def nested(fit, views_fit):
    """fit(X, yy) -> predict function on a log-probability matrix; fitted on the other folds' rows of views_fit."""
    lab = {v: np.zeros(len(y), int) for v in ALL5}
    for f in range(5):
        a = np.where(fo != f)[0]
        pred = fit(np.vstack([LOGP[v][a] for v in views_fit]), np.tile(y[a], len(views_fit)))
        for v in ALL5:
            lab[v][fo == f] = rule(pred(LOGP[v][fo == f]))
    return lab


def ridge(C):
    def fit(X, yy):
        mu, sd = X.mean(0), X.std(0) + 1e-9
        m = LogisticRegression(C=C, max_iter=2000).fit((X - mu) / sd, yy)
        return lambda Z: m.predict_proba((Z - mu) / sd)
    return fit


def geo(X, yy):
    """One non-negative weight per member on its log-probabilities and one temperature (log-loss, L-BFGS-B)."""
    M = X.shape[1] // 5
    X3 = X.reshape(len(X), M, 5)
    idx = np.arange(len(yy))

    def nll(w):
        Z = (X3 * w[:M, None]).sum(1)
        return -float(np.mean((Z - logsumexp(Z, axis=1, keepdims=True))[idx, yy]))
    w0 = np.array([V23_W.get(t, 8.0) for t in MEMBERS])
    res = minimize(nll, w0 / w0.sum() * 2.0, method="L-BFGS-B", bounds=[(0, 20)] * M)
    geo.last = res.x
    return lambda Z: softmax((Z.reshape(len(Z), M, 5) * res.x[:M, None]).sum(1))


for C in (0.1, 1.0):
    report(f"1a ridge logit stacker, C = {C:g}, fitted on the full view", nested(ridge(C), ["full"]))
for C in (0.1, 1.0):
    report(f"1b ridge logit stacker, C = {C:g}, fitted on the five views pooled", nested(ridge(C), ALL5))
report("1c one weight per member on the log-probabilities, fitted on the five views pooled", nested(geo, ALL5))
print("    weights of the last fold (share): " + ", ".join(f"{t} {w / geo.last.sum():.3f}" for t, w in zip(MEMBERS, geo.last))
      + f"; v23 shares: " + ", ".join(f"{t} {w / sum(V23_W.values()):.3f}" for t, w in V23_W.items()))


def soft_mult(Lg, yy, w, temp=0.1):
    Y = np.eye(5)[yy] * w[:, None]
    ny = Y.sum(0)

    def loss(v4):
        q = softmax((Lg + np.r_[0.0, v4][None, :]) / temp)
        return -float(np.mean(2 * (Y * q).sum(0) / ((q * w[:, None]).sum(0) + ny)))
    return np.r_[0.0, minimize(loss, np.zeros(4), method="L-BFGS-B").x]


for name, views_fit in (("3 class multipliers (soft macro-F1) on top of the v23 rule, fitted on the full view", ["full"]),
                        ("3 class multipliers (soft macro-F1) on top of the v23 rule, fitted on the five views pooled", ALL5)):
    lab = {v: np.zeros(len(y), int) for v in ALL5}
    vs = []
    for f in range(5):
        a = np.where(fo != f)[0]
        Lg = np.vstack([np.log(np.clip(ens[v][a] * pi ** -0.75, 1e-12, None)) for v in views_fit])
        vv = soft_mult(Lg, np.tile(y[a], len(views_fit)), np.tile(wb[a], len(views_fit)))
        vs.append(vv)
        for v in ALL5:
            lab[v][fo == f] = (np.log(np.clip(ens[v][fo == f] * pi ** -0.75, 1e-12, None)) + vv[None, :]).argmax(1)
    report(name, lab)
    print("    log-multipliers per fold: " + "; ".join(" ".join(f"{x:+.2f}" for x in vv[1:]) for vv in vs))
print("done")
