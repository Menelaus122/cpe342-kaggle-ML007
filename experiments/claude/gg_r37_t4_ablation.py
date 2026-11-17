"""Round 37, Task 4: how much do the big ensemble and the big backbones add over small models? (CPU; training images only)

    python gg_r37_t4_ablation.py

Out-of-fold views of the v23 members (full, zoom70, HUD hidden, gamma 3.5, 75% grid; real dark / low-contrast frames on the full
view), every combination under the v17 rule (argmax p * pi^-0.75), balanced macro-F1 at a uniform class mix as in every Task 4
rule since round 30. Rows: each member alone (with its parameter count), small subsets, v23, and v23 without each member.
"""
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402

V20 = ["cnxt", "deit3", "cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud", "dinov2_v2"]
PARAMS_M = {"cnxt": 28, "deit3": 22, "cnxt_hud": 28, "cnxt22k_hud": 28, "effv2_hud": 21, "deit3_hud": 22, "cnxtS_hud": 50, "dinov2_v2": 86}
HUDM = ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
V23_W = {t: (8.0 if t == "dinov2_v2" else 3.0 if t in HUDM else 1.0) for t in V20}
ALL5 = ["full", "zoom70", "centre_grey", "dark_g35", "grid75"]
V19, V30 = Path("experiments/claude/outputs/r19_views"), Path("experiments/claude/outputs/r30_views")
train, _ = task4.load_frames()
y = train.label.values
folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
P = {t: {v: np.zeros((len(y), 5)) for v in ALL5} for t in V20}
for t in V20:
    for f, va in enumerate(folds):
        z30, z19 = np.load(V30 / f"{t}_fold{f}.npz"), np.load(V19 / f"{t}_fold{f}.npz")
        for v in ALL5:
            P[t][v][va] = (z30 if v in z30.files else z19)[v]
st = np.load("experiments/claude/outputs/r30/t4_errors.npz")
dark = (st["contrast"] <= np.quantile(st["contrast"], 0.1)) | (st["brightness"] <= np.quantile(st["brightness"], 0.1))
pi = np.bincount(y, minlength=5) / len(y)
wb = (0.2 / pi)[y]
bal = lambda pred, m: f1_score(y[m], pred[m], average="macro", sample_weight=wb[m])
allm = np.ones(len(y), bool)


def row(name, wmap):
    s = sum(wmap.values())
    lab = {v: ((sum(w * P[t][v] for t, w in wmap.items()) / s) * pi ** -0.75).argmax(1) for v in ALL5}
    sc = [bal(lab[v], allm) for v in ALL5] + [bal(lab["full"], dark)]
    hard = np.mean(sc[1:5])
    print(f"| {name} | {sum(PARAMS_M[t] for t in wmap)} | " + " | ".join(f"{x:.4f}" for x in sc) + f" | {hard:.4f} |", flush=True)


print("| model / ensemble | params (M) | full | zoom70 | HUD hidden | gamma 3.5 | 75% grid | dark frames | hard-view mean |")
print("|---|---|---|---|---|---|---|---|---|")
for t in V20:
    row(f"{t} alone", {t: 1.0})
row("cnxt + dinov2_v2 (equal)", {"cnxt": 1.0, "dinov2_v2": 1.0})
row("cnxt_hud + dinov2_v2 (1 : 2)", {"cnxt_hud": 1.0, "dinov2_v2": 2.0})
row("five HUD members + dinov2_v2 (v23 weights)", {t: V23_W[t] for t in HUDM + ["dinov2_v2"]})
row("v23 (eight members)", V23_W)
for t in V20:
    row(f"v23 without {t}", {k: w for k, w in V23_W.items() if k != t})
