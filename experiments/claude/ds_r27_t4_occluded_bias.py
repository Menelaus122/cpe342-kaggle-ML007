"""Round 27, Task 4: does the seven-member ensemble lean towards particular classes when the HUD is missing?

    python ds_r27_t4_occluded_bias.py

Under the v17 rule (argmax p * pi^-0.75) the predicted test shares are 0.148 / 0.261 / 0.150 / 0.248 / 0.193. The true test mix is
unknown, so those shares alone do not show a bias. What can be measured on training images: with every class weighted to 20%
(rows weighted 0.2 / pi), on the 70% zoom and the HUD-hidden views, which classes are over-predicted (share of predictions above
20%, precision below recall)?
Then a correction estimated from training images only: q = normalise(p * pi^-0.75); pbar_k = balanced mean of q_k on the HUD-poor
views (zoom70 and centre_grey) of the training folds; rule argmax q * (pbar / 0.2)^-beta. beta = 0 is the v17 rule. pbar and beta
are both nested (4 folds -> 5th); beta from {0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3} by the balanced macro-F1 averaged over the three views.
Also: free class multipliers fitted on the HUD-poor views of the training folds (coordinate ascent on the same objective), nested.
The correction is applied to every frame, because on test we do not know which frames lack the HUD; the cost on full frames is
part of the objective.
"""
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import PRED_DIR, T4_CLASS_NAMES  # noqa: E402

V19 = Path("experiments/claude/outputs/r19_views")
TAGS = ["cnxt", "deit3", "cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
VIEWS = ["full", "zoom70", "centre_grey"]
train, _ = task4.load_frames()
y = train.label.values
folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
avg = {v: np.zeros((len(y), 5)) for v in VIEWS}
for t in TAGS:
    for f, va in enumerate(folds):
        z = np.load(V19 / f"{t}_fold{f}.npz")
        for v in VIEWS:
            avg[v][va] += z[v] / len(TAGS)
pi = np.bincount(y, minlength=5) / len(y)
wb = (0.2 / pi)[y]
norm = lambda q: q / q.sum(1, keepdims=True)
Q = {v: norm(avg[v] * pi ** -0.75) for v in VIEWS}
bal = lambda idx, pred: f1_score(y[idx], pred, average="macro", sample_weight=wb[idx])
allr = np.arange(len(y))
r3 = lambda a: np.round(a, 3)
print("classes:", T4_CLASS_NAMES)
print("\n== v17 rule, classes weighted to 20% each: share of predictions, precision, recall per class")
for v in VIEWS:
    pred = Q[v].argmax(1)
    share = np.bincount(pred, weights=wb, minlength=5) / wb.sum()
    pr, rc, _, _ = precision_recall_fscore_support(y, pred, sample_weight=wb, labels=range(5), zero_division=0)
    print(f"  {v:12s} macro-F1 {bal(allr, pred):.4f} | predicted shares {r3(share)} | precision {r3(pr)} | recall {r3(rc)}")
    if v != "full":
        cm = confusion_matrix(y, pred, sample_weight=wb, labels=range(5))
        cm = cm / cm.sum(1, keepdims=True)
        print("    row = true class, column = predicted (row shares):")
        for k in range(5):
            print(f"      {T4_CLASS_NAMES[k]:>10s} " + " ".join(f"{x:.3f}" for x in cm[k]))
    print(f"    balanced mean of q: {r3(np.average(Q[v], axis=0, weights=wb))}")

BETAS = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3]
obj = lambda idx, c: np.mean([bal(idx, (Q[v][idx] * c).argmax(1)) for v in VIEWS])
pbar_of = lambda idx: np.mean([np.average(Q[v][idx], axis=0, weights=wb[idx]) for v in ("zoom70", "centre_grey")], 0)
print("\n== correction (pbar / 0.2)^-beta, pbar from all training images (not nested; the nested score follows):")
pb = pbar_of(allr)
print("  pbar on the HUD-poor views:", r3(pb))
for b in BETAS:
    c = (pb / 0.2) ** -b
    s = [bal(allr, (Q[v] * c).argmax(1)) for v in VIEWS]
    print(f"  beta {b:4.2f}: full {s[0]:.4f} zoom70 {s[1]:.4f} centre {s[2]:.4f} mean {np.mean(s):.4f}")


def fit_mult(idx):
    """Free class multipliers on the three views of rows idx (coordinate ascent on the balanced macro-F1 averaged over views)."""
    c = np.ones(5)
    best = obj(idx, c)
    for _ in range(3):
        for k in range(5):
            for s in (0.2, -0.2, 0.1, -0.1, 0.05, -0.05):
                c2 = c.copy()
                c2[k] *= np.exp(s)
                val = obj(idx, c2)
                if val > best + 1e-6:
                    c, best = c2, val
    return c


pred = {"v17 rule": {v: Q[v].argmax(1) for v in VIEWS}, "pbar^-beta (nested)": {v: np.zeros(len(y), int) for v in VIEWS},
        "free multipliers (nested)": {v: np.zeros(len(y), int) for v in VIEWS}}
chosen = []
for va in folds:
    tr = np.setdiff1d(allr, va)
    pb = pbar_of(tr)
    b = BETAS[int(np.argmax([obj(tr, (pb / 0.2) ** -b) for b in BETAS]))]
    cm = fit_mult(tr)
    chosen.append((b, r3(cm)))
    for v in VIEWS:
        pred["pbar^-beta (nested)"][v][va] = (Q[v][va] * (pb / 0.2) ** -b).argmax(1)
        pred["free multipliers (nested)"][v][va] = (Q[v][va] * cm).argmax(1)
print("\n== nested (pbar, beta and multipliers from 4 folds -> 5th); balanced macro-F1")
print("  chosen per fold:", "; ".join(f"beta {b}, multipliers {c}" for b, c in chosen))
rng = np.random.default_rng(0)
boot = [rng.integers(0, len(y), len(y)) for _ in range(500)]
base = pred["v17 rule"]
for name, p in pred.items():
    s = [bal(allr, p[v]) for v in VIEWS]
    line = f"  {name:28s} full {s[0]:.4f} zoom70 {s[1]:.4f} centre {s[2]:.4f} mean {np.mean(s):.4f}"
    if name != "v17 rule":
        d = np.mean(s) - np.mean([bal(allr, base[v]) for v in VIEWS])
        bs = [np.mean([bal(i, p[v][i]) - bal(i, base[v][i]) for v in VIEWS]) for i in boot]
        pf = [np.mean([bal(va, p[v][va]) - bal(va, base[v][va]) for v in VIEWS]) for va in folds]
        tm = np.mean([f1_score(y, p[v], average="macro") - f1_score(y, base[v], average="macro") for v in VIEWS])
        line += (f" | delta {d:+.4f} [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}] folds +{sum(x > 0 for x in pf)}/5 "
                 f"({' '.join(f'{x:+.4f}' for x in pf)}) | at the training mix {tm:+.4f}")
    print(line)
