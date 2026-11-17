"""Round 39, Task 4: does the horizontal flip carry information? (training images only; CPU; no new inference)

    python gr_r39_t4_flip_check.py

Question from a suggestion list (2026-11-17 evening): "no h-flip by default, since the HUD / minimap position may identify the game".
Our members are trained with h-flip augmentation and predict with h-flip TTA. Round 29 stored every fold model's out-of-fold
probabilities on the full frame separately for the original and the flipped image (outputs/r29_tta/<tag>_fold<f>.npz). Scored here:
original only, flipped only, and their mean (the pipeline), for each v23 member and for the v29 ensemble (v23 member weights, then
0.65 x members + 0.35 x the frozen-feature probe, whose features already average the image and its flip), v17 rule, balanced class
mix (rows weighted 0.2 / pi), on all training images and on the real dark / low-contrast frames. Information only (no decision rule):
if the flipped image alone were much worse than the original, the models would read left-right layout; if the mean beats the
original, flip TTA helps.
"""
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402

V20 = ["cnxt", "deit3", "cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud", "dinov2_v2"]
HUDM = ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
W = {t: (8.0 if t == "dinov2_v2" else 3.0 if t in HUDM else 1.0) for t in V20}
R29 = Path("experiments/claude/outputs/r29_tta")
train, _ = task4.load_frames()
y = train.label.values
pi = np.bincount(y, minlength=5) / len(y)
wb = (0.2 / pi)[y]
rule = lambda p: (p * pi ** -0.75).argmax(1)
st = np.load("experiments/claude/outputs/r30/t4_errors.npz")
dark = (st["contrast"] <= np.quantile(st["contrast"], 0.1)) | (st["brightness"] <= np.quantile(st["brightness"], 0.1))
folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
fo = np.zeros(len(y), int)
for f, va in enumerate(folds):
    fo[va] = f
bal = lambda pred, m: f1_score(y[m], pred[m], average="macro", sample_weight=wb[m])
allm = np.ones(len(y), bool)

P = {t: {k: np.zeros((len(y), 5)) for k in ("orig", "flip")} for t in V20}
for t in V20:
    for f, va in enumerate(folds):
        z = np.load(R29 / f"{t}_fold{f}.npz")
        for k in ("orig", "flip"):
            P[t][k][va] = z[k]
    mean = 0.5 * (P[t]["orig"] + P[t]["flip"])
    ref = np.load(PRED_DIR / f"t4_{t}_fold0.npz")["val_probs"]
    # same rows; bf16 inference noise only (mean |diff| ~1e-4, argmax agreement >= 0.9997)
    assert np.abs(mean[folds[0]] - ref).mean() < 1e-3 and (mean[folds[0]].argmax(1) == ref.argmax(1)).mean() > 0.999, \
        f"{t}: r29 cache does not match the pipeline's fold 0"
    P[t]["mean"] = mean

probe = np.zeros((len(y), 5))
for f, va in enumerate(folds):
    probe[va] = np.load(PRED_DIR / f"t4_probe_fold{f}.npz")["val_probs"]

print(f"{'member / ensemble':22s} {'TTA':5s} {'all':>7s} {'dark':>7s}   by fold (all)")
for t in V20 + ["v29 ensemble"]:
    for k in ("orig", "flip", "mean"):
        if t == "v29 ensemble":
            mem = sum(W[m] * P[m][k] for m in V20) / sum(W.values())
            p = 0.65 * mem + 0.35 * probe
        else:
            p = P[t][k]
        pred = rule(p)
        print(f"{t:22s} {k:5s} {bal(pred, allm):.4f} {bal(pred, dark):.4f}   " + " ".join(f"{bal(pred, fo == f):.4f}" for f in range(5)),
              flush=True)
