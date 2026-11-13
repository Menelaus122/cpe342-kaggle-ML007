"""Round 23, Task 4: class prior of the decision rule, and calibration of the backbones before averaging. Training images only
(out-of-fold probabilities of the six v15 backbones on full frames and the 70% zoom, outputs/r19_views).

    python cy_r23_t4_prior_temp.py

A. Prior. Training has 39% 7-Eleven Knights; a model trained on it leans towards that class when unsure. Macro-F1 depends on the
   class mix of the evaluated set, so every rule is scored twice: on the OOF rows as they are (training mix) and with each row
   weighted by 0.2 / pi(class) (a balanced 20%-per-class set). Rules: argmax(p) and argmax(p * pi^-alpha), alpha = 0.25 / 0.5 / 0.75 / 1
   (alpha = 1: balanced prior). No test label or test image is used; the number of test labels each rule would change is reported
   (label-free) so the stake is visible.
B. Calibration. Per backbone, a temperature fitted (nested, on full frames) to the log-probabilities; then mean of calibrated
   probabilities, mean of log-probabilities (logit averaging), and the plain mean (v15), on full frames and the zoom.
"""
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import PRED_DIR, T4_CLASS_NAMES  # noqa: E402

V19 = Path("experiments/claude/outputs/r19_views")
SIX = ["cnxt", "deit3", "cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud"]
VIEWS = ["full", "zoom70"]
train, test = task4.load_frames()
y = train.label.values
folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
P = {(t, v): np.zeros((len(y), 5)) for t in SIX for v in VIEWS}
for t in SIX:
    for f, va in enumerate(folds):
        z = np.load(V19 / f"{t}_fold{f}.npz")
        for v in VIEWS:
            P[t, v][va] = z[v]
pi = np.bincount(y, minlength=5) / len(y)
w_bal = (0.2 / pi)[y]
mf1 = lambda pred, sw=None: f1_score(y, pred, average="macro", sample_weight=sw)
avg = {v: np.mean([P[t, v] for t in SIX], 0) for v in VIEWS}
print("training prior:", dict(zip(T4_CLASS_NAMES, np.round(pi, 3))))

print("\n== A. decision prior, macro-F1 (training mix | balanced 20% mix)")
FULL_W = 0.5
tst = np.mean([(1 - FULL_W) * np.mean([np.load(PRED_DIR / f"t4_{t}_fold{f}.npz")["test_probs"] for f in range(5)], 0)
               + FULL_W * np.load(PRED_DIR / f"t4_{t}_full.npz")["test_probs"] for t in SIX], 0)
for a in [0, 0.25, 0.5, 0.75, 1.0]:
    line = f"  alpha {a:4.2f}: "
    for v in VIEWS:
        pred = (avg[v] * pi ** -a).argmax(1)
        line += f"{v} {mf1(pred):.4f} | {mf1(pred, w_bal):.4f}    "
    tp = (tst * pi ** -a).argmax(1)
    line += f"test labels changed vs alpha 0: {(tp != tst.argmax(1)).sum():5d}; predicted test shares {np.round(np.bincount(tp, minlength=5) / len(tp), 3)}"
    print(line)
print("  (rows: rule; within each view: score if the evaluated set has the training mix | if it is balanced)")

print("\n== B. temperatures and averaging")
logp = {k: np.log(np.clip(p, 1e-12, 1)) for k, p in P.items()}


def nll(z, rows):
    z = z[rows] - z[rows].max(1, keepdims=True)
    return -(z[np.arange(len(rows)), y[rows]] - np.log(np.exp(z).sum(1))).mean()


cal = {k: np.zeros_like(p) for k, p in P.items()}
for t in SIX:
    ts = []
    for f, va in enumerate(folds):
        tr = np.setdiff1d(np.arange(len(y)), va)
        T = minimize_scalar(lambda T: nll(logp[t, "full"] / T, tr), bounds=(0.2, 5), method="bounded").x
        ts.append(T)
        for v in VIEWS:
            z = logp[t, v][va] / T
            e = np.exp(z - z.max(1, keepdims=True))
            cal[t, v][va] = e / e.sum(1, keepdims=True)
    mx = P[t, "full"].max(1)
    print(f"  {t:12s} temperature {np.mean(ts):.2f} (folds {min(ts):.2f}-{max(ts):.2f}); mean max-prob {mx.mean():.3f}, "
          f"accuracy {(P[t, 'full'].argmax(1) == y).mean():.4f}; NLL {nll(logp[t, 'full'], np.arange(len(y))):.4f}")
for name, fn in [("plain mean of probabilities (v15)", lambda v: avg[v]),
                 ("mean of temperature-calibrated probabilities", lambda v: np.mean([cal[t, v] for t in SIX], 0)),
                 ("mean of log-probabilities (logit averaging)", lambda v: np.mean([logp[t, v] for t in SIX], 0))]:
    print(f"  {name:46s} " + "  ".join(f"{v} {mf1(fn(v).argmax(1)):.4f}" for v in VIEWS))
