"""Round 30, Task 4: member weights of the v20 average chosen nested on the hard-view criterion (CPU; training images only).

    python er_r30_t4_weights_hard.py          (needs outputs/r30_views from em_r30_t4_hardviews.py for the gamma / grid views)

v20 weights: DINOv2 x3, the seven others x1 (chosen in round 28 on full / zoom70 / HUD hidden). On gamma 3.5 and the 75% grid the
fine-tuned DINOv2 led the other members by 2-5 points (round 28), so the hard-view criterion of 2026-11-16 may prefer more weight on
it, or on the HUD-dropout members. Grid: DINOv2 weight in {2, 3, 4, 5, 6} x weight of the five HUD-dropout members in {1, 1.5, 2}
(first run; it chose the corner (6, 2) in every fold, so the grid was widened to {3, 4, 5, 6, 8, 10} x {1, 1.5, 2, 3})
(the two plain members stay at 1). Chosen on 4 folds by the hard-view mean (zoom70, HUD hidden, gamma 3.5, 75% grid; v17 rule,
uniform class mix, members only), scored on the 5th. Acceptance (user, 2026-11-16, Task 4): hard-view mean >= +0.0015 and full
>= -0.0005; the generic rule (pooled > 0, >= 3/5 folds, no fold below -0.0005) is reported too. The 10% probe is added afterwards
for the three views it has, as in the pipeline.
"""
import itertools
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402

V20 = ["cnxt", "deit3", "cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud", "dinov2_v2"]
HUDM = ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
ALL5 = ["full", "zoom70", "centre_grey", "dark_g35", "grid75"]
HARD = ALL5[1:]
V19, V30 = Path("experiments/claude/outputs/r19_views"), Path("experiments/claude/outputs/r30_views")
train, _ = task4.load_frames()
y = train.label.values
folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
fo = np.zeros(len(y), int)
for f, va in enumerate(folds):
    fo[va] = f
P = {t: {v: np.zeros((len(y), 5)) for v in ALL5} for t in V20}
for t in V20:
    for f, va in enumerate(folds):
        z19, z30 = np.load(V19 / f"{t}_fold{f}.npz"), np.load(V30 / f"{t}_fold{f}.npz")
        for v in ALL5:
            P[t][v][va] = (z30 if v in z30.files else z19)[v]
zp = np.load("experiments/claude/outputs/r27/t4_probe_views.npz")
pi = np.bincount(y, minlength=5) / len(y)
wb = (0.2 / pi)[y]
bal = lambda pred, m: f1_score(y[m], pred[m], average="macro", sample_weight=wb[m])


def labels(wd, wh, probe=False):
    w = {t: (wd if t == "dinov2_v2" else wh if t in HUDM else 1.0) for t in V20}
    s = sum(w.values())
    out = {}
    for v in ALL5:
        p = sum(w[t] * P[t][v] for t in V20) / s
        if probe and v in ("full", "zoom70", "centre_grey"):
            p = 0.9 * p + 0.1 * zp[f"probe-aug_{v}"]
        out[v] = (p * pi ** -0.75).argmax(1)
    return out


GRID = list(itertools.product((3.0, 4.0, 5.0, 6.0, 8.0, 10.0), (1.0, 1.5, 2.0, 3.0)))
base = labels(3.0, 1.0)
lab = {g: labels(*g) for g in GRID}
fold_hard = {g: [np.mean([bal(lab[g][v], fo == f) - bal(base[v], fo == f) for v in HARD]) for f in range(5)] for g in GRID}
print("in-sample hard-view mean delta against v20 (DINOv2 weight x HUD-member weight):")
for g in GRID:
    full = bal(lab[g]["full"], fo >= 0) - bal(base["full"], fo >= 0)
    print(f"  DINOv2 {g[0]:.0f}, HUD members {g[1]:.1f}: hard {np.mean([bal(lab[g][v], fo >= 0) - bal(base[v], fo >= 0) for v in HARD]):+.4f}, full {full:+.4f}")
nested = {v: np.zeros(len(y), int) for v in ALL5}
chosen = []
for f in range(5):
    g = max(GRID, key=lambda g_: np.mean([fold_hard[g_][k] for k in range(5) if k != f]))
    chosen.append(g)
    for v in ALL5:
        nested[v][fo == f] = lab[g][v][fo == f]
allm = fo >= 0
per = {v: bal(nested[v], allm) - bal(base[v], allm) for v in ALL5}
hard = np.mean([per[v] for v in HARD])
pf = [np.mean([bal(nested[v], fo == f) - bal(base[v], fo == f) for v in HARD]) for f in range(5)]
ok_user = hard >= 0.0015 and per["full"] >= -0.0005
ok_gen = hard > 0 and sum(p > 0 for p in pf) >= 3 and min(pf) >= -0.0005
print(f"\nnested (chosen {chosen}): hard mean {hard:+.4f}, full {per['full']:+.4f} | " + " ".join(f"{v} {per[v]:+.4f}" for v in HARD)
      + f" | folds (hard) +{sum(p > 0 for p in pf)}/5 ({' '.join(f'{p:+.4f}' for p in pf)}) -> hard-view rule {'PASS' if ok_user else 'fail'}, "
        f"generic rule {'PASS' if ok_gen else 'fail'}")
mode = max(set(chosen), key=chosen.count)
lp, bp = labels(*mode, probe=True), labels(3.0, 1.0, probe=True)
print(f"with the 10% probe, the most chosen weights {mode}: " + " ".join(f"{v} {bal(lp[v], allm) - bal(bp[v], allm):+.4f}" for v in ("full", "zoom70", "centre_grey")))
