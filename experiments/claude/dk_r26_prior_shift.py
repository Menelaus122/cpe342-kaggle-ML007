"""Round 26: is there a class-prior shift between train and test in Tasks 1 and 2 (as in Task 4), and the Task 4 exponent table.

    python dk_r26_prior_shift.py

Diagnostics only: nothing here changes a submission. For each task: training class shares, mean out-of-fold and mean test
probability, predicted label shares, and the EM estimate of the test class shares (Saerens et al., 2002: fixed point of
q <- mean_i [p_i q / pi] / sum). EM is also run on each held-out fold (true answer: the training shares) to see its error.
EM assumes p(x | class) is the same in train and test; where the inputs themselves shift (Task 4) it is biased.
Task 2: macro-F1 of decision rules at the training mix and on a balanced 25% mix (rows weighted 0.25 / pi).
Task 4: argmax p * pi^-alpha for alpha in {0, 0.5, 0.75, 0.85, 1, 1.15}: OOF macro-F1 at the training mix and at a uniform mix on three
views, and the predicted test shares.
"""
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from common import load_t1, stack_binary, t2_baseline  # noqa: E402
from src import task2, task4  # noqa: E402
from src.config import PRED_DIR, T4_CLASS_NAMES  # noqa: E402
from src.metrics import apply_class_weights, optimize_class_weights, optimize_threshold  # noqa: E402
from src.modeling import make_folds  # noqa: E402
from src.utils import load_preds  # noqa: E402


def em_prior(P, pi, iters=500):
    q = pi.copy()
    for _ in range(iters):
        w = P * (q / pi)
        w /= w.sum(1, keepdims=True)
        q2 = w.mean(0)
        if np.abs(q2 - q).max() < 1e-7:
            break
        q = q2
    return q


r3 = lambda a: np.round(a, 3)

# ================================================================ Task 2
train2, _ = task2.load()
y2 = train2[task2.TARGET].values
folds2 = make_folds(y2)
_, so2, st2, _ = t2_baseline(y2, folds2)
pi2 = np.bincount(y2, minlength=4) / len(y2)
print("== Task 2 (Casual / Grinder / Social / Whale)")
print("  training shares              ", r3(pi2))
print("  mean probability  OOF | test ", r3(so2.mean(0)), "|", r3(st2.mean(0)))
W2 = optimize_class_weights(y2, so2)[0]
print("  predicted shares  OOF | test ", r3(np.bincount(apply_class_weights(so2, W2), minlength=4) / len(y2)), "|",
      r3(np.bincount(apply_class_weights(st2, W2), minlength=4) / len(st2)))
print("  EM estimate of the test shares", r3(em_prior(st2, pi2)), "| EM on each held-out fold (truth = training shares): max |error|",
      f"{max(np.abs(em_prior(so2[va], pi2) - np.bincount(y2[va], minlength=4) / len(va)).max() for _, va in folds2):.4f}")
wb2 = (0.25 / pi2)[y2]
nest_mult, nest_bal = np.zeros(len(y2), int), np.zeros(len(y2), int)
for tr, va in folds2:
    nest_mult[va] = apply_class_weights(so2[va], optimize_class_weights(y2[tr], so2[tr])[0])
    # multipliers fitted for the balanced mix: coordinate ascent on the weighted macro-F1
    w = pi2 ** -1.0
    best = f1_score(y2[tr], (so2[tr] * w).argmax(1), average="macro", sample_weight=wb2[tr])
    for _ in range(4):
        for c in range(4):
            for s in (0.1, -0.1, 0.05, -0.05, 0.02, -0.02):
                w2 = w.copy()
                w2[c] *= np.exp(s)
                v = f1_score(y2[tr], (so2[tr] * w2).argmax(1), average="macro", sample_weight=wb2[tr])
                if v > best + 1e-7:
                    w, best = w2, v
    nest_bal[va] = (so2[va] * w).argmax(1)
sc = lambda p: (f1_score(y2, p, average="macro"), f1_score(y2, p, average="macro", sample_weight=wb2),
                f1_score(y2, p, average=None, sample_weight=wb2))
print("  rule                               macro-F1 training mix | balanced mix | per-class F1 at the balanced mix")
for name, p in [("argmax", so2.argmax(1)), ("pipeline multipliers (nested)", nest_mult), ("p * pi^-0.5", (so2 * pi2 ** -0.5).argmax(1)),
                ("p * pi^-0.75", (so2 * pi2 ** -0.75).argmax(1)), ("p * pi^-1", (so2 / pi2).argmax(1)), ("multipliers fitted for balance (nested)", nest_bal)]:
    a, b, c = sc(p)
    print(f"  {name:40s} {a:.4f} | {b:.4f} | {r3(c)}")

# ================================================================ Task 1
_, _, _, y1, _, folds1, _ = load_t1()
M1 = [load_preds(k) for k in ["t1_lgb", "t1_xgb", "t1_cat", "t1_et", "t1_lr", "t1_rf24", "t1_lgb24", "t1_xgb24", "t1_cat24"]]
so1, st1 = stack_binary([m["oof"] for m in M1], [m["test"] for m in M1], y1, folds1)
pi1 = np.array([1 - y1.mean(), y1.mean()])
thr, f2v = optimize_threshold(y1, so1, 2.0)
P1o, P1t = np.column_stack([1 - so1, so1]), np.column_stack([1 - st1, st1])
print("\n== Task 1 (legitimate / cheater)")
print(f"  training cheater share {pi1[1]:.4f} | mean P(cheater) OOF {so1.mean():.4f}, test {st1.mean():.4f} | flagged at the cut {thr:.3f}: OOF {np.mean(so1 >= thr):.4f}, test {np.mean(st1 >= thr):.4f}")
print(f"  EM estimate of the test cheater share {em_prior(P1t, pi1)[1]:.4f} | EM on each held-out fold: max |error| "
      f"{max(abs(em_prior(P1o[va], pi1)[1] - y1[va].mean()) for _, va in folds1):.4f}")
print(f"  (round 23: test rows have fewer missing cells, 8.0% vs 12.0%, which moves the flag rate; OOF F2 {f2v:.4f})")

# ================================================================ Task 4
V19 = Path("experiments/claude/outputs/r19_views")
TAGS = ["cnxt", "deit3", "cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
VIEWS = ["full", "zoom70", "centre_grey"]
train4, _ = task4.load_frames()
y4 = train4.label.values
f4 = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
avg = {v: np.zeros((len(y4), 5)) for v in VIEWS}
for t in TAGS:
    for f, va in enumerate(f4):
        z = np.load(V19 / f"{t}_fold{f}.npz")
        for v in VIEWS:
            avg[v][va] += z[v] / len(TAGS)
tst = np.mean([0.5 * np.mean([np.load(PRED_DIR / f"t4_{t}_fold{f}.npz")["test_probs"] for f in range(5)], 0)
               + 0.5 * np.load(PRED_DIR / f"t4_{t}_full.npz")["test_probs"] for t in TAGS], 0)
pi4 = np.bincount(y4, minlength=5) / len(y4)
wb4 = (0.2 / pi4)[y4]
print("\n== Task 4", T4_CLASS_NAMES)
print("  training shares", r3(pi4), "| mean probability OOF", r3(avg["full"].mean(0)), "| test", r3(tst.mean(0)))
print("  alpha | OOF macro-F1 training mix: full / zoom70 / centre | uniform mix: full / zoom70 / centre (mean) | predicted test shares")
for a in [0.0, 0.5, 0.75, 0.85, 1.0, 1.15]:
    pr = {v: (avg[v] * pi4 ** -a).argmax(1) for v in VIEWS}
    tm = [f1_score(y4, pr[v], average="macro") for v in VIEWS]
    um = [f1_score(y4, pr[v], average="macro", sample_weight=wb4) for v in VIEWS]
    print(f"  {a:4.2f}  | {tm[0]:.4f} / {tm[1]:.4f} / {tm[2]:.4f} | {um[0]:.4f} / {um[1]:.4f} / {um[2]:.4f} ({np.mean(um):.4f}) | "
          f"{r3(np.bincount((tst * pi4 ** -a).argmax(1), minlength=5) / len(tst))}")
# calibration: the ensemble is under-confident (label smoothing / Mixup), so p understates the evidence
lp = np.log(np.clip(avg["full"], 1e-9, 1))
nll = lambda T: -np.mean((lp / T)[np.arange(len(y4)), y4] - np.log(np.exp(lp / T).sum(1)))
T = minimize_scalar(nll, bounds=(0.2, 3), method="bounded").x
cal = lambda P: (lambda q: q / q.sum(1, keepdims=True))(np.clip(P, 1e-9, 1) ** (1 / T))
print(f"  ensemble temperature on OOF {T:.2f} (< 1 = under-confident): the calibrated posterior is p^(1/T), so dividing it by pi equals "
      f"argmax p * pi^-T = alpha {T:.2f} on in-distribution frames")
print("  EM estimate of the test shares: raw probabilities", r3(em_prior(tst, pi4)), "| calibrated", r3(em_prior(cal(tst), pi4)))
print("  EM on each held-out fold, max |error|: raw", f"{max(np.abs(em_prior(avg['full'][va], pi4) - np.bincount(y4[va], minlength=5) / len(va)).max() for va in f4):.4f}",
      "| calibrated", f"{max(np.abs(em_prior(cal(avg['full'][va]), pi4) - np.bincount(y4[va], minlength=5) / len(va)).max() for va in f4):.4f}",
      "| on the HUD-hidden view (inputs shifted, truth unchanged): raw",
      f"{max(np.abs(em_prior(avg['centre_grey'][va], pi4) - np.bincount(y4[va], minlength=5) / len(va)).max() for va in f4):.4f}")
