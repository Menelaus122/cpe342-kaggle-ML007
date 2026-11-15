"""Round 25, Task 4: how to combine the seven members under the balanced-prior rule (training images only, OOF views).

    python dj_r25_t4_rule_variants.py

Scored with rows weighted to a balanced 20%-per-class mix (0.2 / pi), on full frames, the 70% zoom and HUD-hidden frames:
  mean-p     argmax (mean_m p_m) * pi^-alpha                      (the v17 rule at alpha 0.75)
  mean-logp  argmax mean_m log p_m - alpha log pi                 (logit averaging)
  adj-mean   argmax mean_m normalise(p_m * pi^-alpha)             (each member prior-adjusted and renormalised, then averaged)
"""
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402

V19 = Path("experiments/claude/outputs/r19_views")
TAGS = ["cnxt", "deit3", "cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
VIEWS = ["full", "zoom70", "centre_grey"]
train, _ = task4.load_frames()
y = train.label.values
folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
P = {(t, v): np.zeros((len(y), 5)) for t in TAGS for v in VIEWS}
for t in TAGS:
    for f, va in enumerate(folds):
        z = np.load(V19 / f"{t}_fold{f}.npz")
        for v in VIEWS:
            P[t, v][va] = z[v]
pi = np.bincount(y, minlength=5) / len(y)
wb = (0.2 / pi)[y]
bal = lambda pred: f1_score(y, pred, average="macro", sample_weight=wb)
norm = lambda q: q / q.sum(1, keepdims=True)
print("balanced-mix macro-F1 (full / zoom70 / centre_grey / mean)")
for a in [0.5, 0.75, 1.0]:
    rules = {"mean-p": lambda v: np.mean([P[t, v] for t in TAGS], 0) * pi ** -a,
             "mean-logp": lambda v: np.mean([np.log(np.clip(P[t, v], 1e-6, 1)) for t in TAGS], 0) - a * np.log(pi),
             "adj-mean": lambda v: np.mean([norm(P[t, v] * pi ** -a) for t in TAGS], 0)}
    for name, fn in rules.items():
        s = [bal(fn(v).argmax(1)) for v in VIEWS]
        print(f"  alpha {a:.2f} {name:10s} " + " / ".join(f"{x:.4f}" for x in s) + f" / {np.mean(s):.4f}")
