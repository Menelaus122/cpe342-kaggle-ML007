"""Round 40, Task 4: dinov3L_lr8e5 (fold 0 only) against v30 on fold 0 (CPU; training images only).

    python hd_r40_t4_dinov3_fold0_v30.py [TAG]

User brief of 2026-11-18 afternoon: DINOv3-L with lr 8e-5 / layer decay 0.90 / 18 epochs, fold 0 on Colab; its notebook gate (alone >=
0.985 and the v24 check against the v23 members) is followed by a comparison with v30 on fold 0, here, to decide whether folds 1-4
and the full refit are worth their compute. Variants (fixed before this script was run; the v30 pipeline otherwise unchanged:
members, brightness / contrast gate with thresholds from folds 1-4, 0.65 x members + 0.35 x the seven-backbone MLP probe, v17 rule):
  R  the new member in place of DINOv2-L (x8, x2 on gated frames)
  A  the new member next to DINOv2-L, at the same x8 / x2
Fold-0 reading of the user's rule: zoom70, HUD hidden and the real dark / low-contrast frames all > 0, full >= -0.0005 (the >= 3/5
folds part needs all folds). This informs the decision to train folds 1-4; the member is judged on all five folds afterwards.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402

CAND = sys.argv[1] if len(sys.argv) > 1 else "dinov3L_lr8e5"
L2 = "dinov2L_hud70"
PLAIN, HUDM = ["cnxt", "deit3"], ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
V20 = PLAIN + HUDM + ["dinov2_v2"]
WM = {**{t: 1.0 for t in PLAIN}, **{t: 3.0 for t in HUDM}, "dinov2_v2": 8.0}
VIEWS = ["full", "zoom70", "centre_grey"]
V19, V30 = Path("experiments/claude/outputs/r19_views"), Path("experiments/claude/outputs/r30_views")

train, _ = task4.load_frames()
y = train.label.values
folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
fo = np.zeros(len(y), int)
for f, va in enumerate(folds):
    fo[va] = f
va = folds[0]
assert np.array_equal(np.load(PRED_DIR / f"t4_{CAND}_fold0.npz")["val_idx"], va)
P = {}
for t in V20 + [L2, CAND]:
    z30 = np.load(V30 / f"{t}_fold0.npz")
    z19 = np.load(V19 / f"{t}_fold0.npz") if t in V20 else z30
    P[t] = {v: (z30 if v in z30.files else z19)[v] for v in VIEWS}
PR = {v: np.load("experiments/claude/outputs/r39/p7_probe_views.npz")[v][va] for v in VIEWS}
ST = dict(np.load("experiments/claude/outputs/r39/t4_view_stats.npz"))
tr = fo != 0
tb, tc = np.quantile(ST["brightness_full"][tr], 0.10), np.quantile(ST["contrast_full"][tr], 0.10)
GATE = {v: (ST[f"brightness_{v}"][va] <= tb) | (ST[f"contrast_{v}"][va] <= tc) for v in VIEWS}
st = np.load("experiments/claude/outputs/r30/t4_errors.npz")
dark = ((st["contrast"] <= np.quantile(st["contrast"], 0.1)) | (st["brightness"] <= np.quantile(st["brightness"], 0.1)))[va]
yv = y[va]
pi = np.bincount(y, minlength=5) / len(y)
wb = (0.2 / pi)[yv]
bal = lambda pred, m: f1_score(yv[m], pred[m], average="macro", sample_weight=wb[m])
rule = lambda p: (p * pi ** -0.75).argmax(1)
allm = np.ones(len(va), bool)
SC = ["full", "zoom70", "centre_grey", "dark/low-contrast"]
VIEW_OF = {"full": "full", "zoom70": "zoom70", "centre_grey": "centre_grey", "dark/low-contrast": "full"}
msk = lambda k: dark if k == "dark/low-contrast" else allm


def pipeline(gated):
    """gated: {tag: (normal weight, weight on gated frames)} on top of the v23 members."""
    out = {}
    for v in VIEWS:
        num = sum(w * P[t][v] for t, w in WM.items())
        den = np.full(len(va), sum(WM.values()))
        for t, (wn, wd) in gated.items():
            w = np.where(GATE[v], wd, wn)
            num = num + w[:, None] * P[t][v]
            den = den + w
        out[v] = rule(0.65 * num / den[:, None] + 0.35 * PR[v])
    return out


ref = pipeline({L2: (8.0, 2.0)})
print(f"fold 0: {len(va)} images, {dark.sum()} real dark / low-contrast, gated (full view) {GATE['full'].sum()}")
print("alone (balanced mix, rule): " + " | ".join(
    f"{t} " + ", ".join(f"{k} {bal(rule(P[t][VIEW_OF[k]]), msk(k)):.4f}" for k in SC) for t in (L2, CAND)))
print("v30 on fold 0: " + ", ".join(f"{k} {bal(ref[VIEW_OF[k]], msk(k)):.4f}" for k in SC))
for name, g in (("R  new member in place of DINOv2-L (x8 / x2 gated)", {CAND: (8.0, 2.0)}),
                ("A  new member next to DINOv2-L (both x8 / x2 gated)", {L2: (8.0, 2.0), CAND: (8.0, 2.0)})):
    lab = pipeline(g)
    d = {k: bal(lab[VIEW_OF[k]], msk(k)) - bal(ref[VIEW_OF[k]], msk(k)) for k in SC}
    ok = d["zoom70"] > 0 and d["centre_grey"] > 0 and d["dark/low-contrast"] > 0 and d["full"] >= -0.0005
    print(f"  {name}\n    " + ", ".join(f"{k} {d[k]:+.4f}" for k in SC)
          + f" | labels changed on full frames {(lab['full'] != ref['full']).sum()}\n    -> fold-0 rule {'PASS' if ok else 'fail'}")
