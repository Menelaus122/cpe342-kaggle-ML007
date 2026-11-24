"""Round 42, Task 4: dinov3Hp_384 (fine-tuned DINOv3 ViT-H+/16 at 192 x 384, layer decay 0.93, fold 0 only) against v37 on fold 0 (CPU;
training images only).

    python ht_r42_t4_dinov3h384_fold0.py

Written 2026-11-20, before the Colab run (colab/t4_colab_dinov3h384.ipynb; notebook gate: alone >= 0.9960 on fold 0 - dinov3Hp_144's
fold-0 score - and the v24 check against the v23 members). Reference v37: older seven x0.5 of v36 (ConvNeXt-T / DeiT-III 0.25, HUD
members 0.75), DINOv2-B 8, DINOv2-L 8 / 2 gated, DINOv3-L-144 10, DINOv3-L-384 20, C-RADIOv3-L 10 / 2 gated, ViT-H+-144 40,
0.70 x members + 0.30 x P7, v17 rule, balanced class mix; gate = v30's Q10 brightness / contrast gate with thresholds from folds 1-4.
Variants, fixed now (user's brief):
  R    dinov3Hp_384 in place of dinov3Hp_144 at x40
  H    both ViT-H+ members at x20 each (the x40 shared between the two resolutions)
  A20  dinov3Hp_384 added at x20 (dinov3Hp_144 stays x40)
  A40  dinov3Hp_384 added at x40 (dinov3Hp_144 stays x40)
Fold-0 reading of the user's rule: zoom70, HUD hidden and the real dark / low-contrast frames all > 0, full >= -0.0005. If at least
one variant meets it, folds 1-4 are worth their compute; the member is then judged on all five folds by a script written before
folds 1-4 (as hn_r42 / hr_r42).
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cand", default="dinov3Hp_384")
args = ap.parse_args()
CAND, L2, D3, D384, RAD, HP = args.cand, "dinov2L_hud70", "dinov3L_lr8e5", "dinov3L_384", "cradio3L_384", "dinov3Hp_144"
PLAIN, HUDM = ["cnxt", "deit3"], ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
V20 = PLAIN + HUDM + ["dinov2_v2"]
WM = {**{t: 0.25 for t in PLAIN}, **{t: 0.75 for t in HUDM}, "dinov2_v2": 8.0, D3: 10.0, D384: 20.0}
PW = 0.30
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
for t in V20 + [L2, D3, D384, RAD, HP, CAND]:
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
    out = {}
    for v in VIEWS:
        num = sum(w * P[t][v] for t, w in WM.items())
        den = np.full(len(va), sum(WM.values()))
        for t, (wn, wd) in gated.items():
            w = np.where(GATE[v], wd, wn)
            num, den = num + w[:, None] * P[t][v], den + w
        out[v] = rule((1 - PW) * num / den[:, None] + PW * PR[v])
    return out


base = {L2: (8.0, 2.0), RAD: (10.0, 2.0)}
ref = pipeline({**base, HP: (40.0, 40.0)})
print("reference v37 on fold 0: " + ", ".join(f"{k} {bal(ref[VIEW_OF[k]], msk(k)):.4f}" for k in SC))
print("alone (balanced mix, rule): " + " | ".join(
    f"{t} " + ", ".join(f"{k} {bal(rule(P[t][VIEW_OF[k]]), msk(k)):.4f}" for k in SC) for t in (HP, CAND)))
any_ok = False
for name, g in ((f"R    {CAND} in place of {HP} (x40)", {**base, CAND: (40.0, 40.0)}),
                (f"H    {HP} x20 + {CAND} x20", {**base, HP: (20.0, 20.0), CAND: (20.0, 20.0)}),
                (f"A20  {CAND} added at x20", {**base, HP: (40.0, 40.0), CAND: (20.0, 20.0)}),
                (f"A40  {CAND} added at x40", {**base, HP: (40.0, 40.0), CAND: (40.0, 40.0)})):
    lab = pipeline(g)
    d = {k: bal(lab[VIEW_OF[k]], msk(k)) - bal(ref[VIEW_OF[k]], msk(k)) for k in SC}
    ok = d["zoom70"] > 0 and d["centre_grey"] > 0 and d["dark/low-contrast"] > 0 and d["full"] >= -0.0005
    any_ok |= ok
    print(f"  {name}\n    " + ", ".join(f"{k} {d[k]:+.4f}" for k in SC)
          + f" | labels changed on full frames {(lab['full'] != ref['full']).sum()}\n    -> fold-0 rule {'PASS' if ok else 'fail'}")
print(f"\n-> {'at least one variant passes: folds 1-4 worth running' if any_ok else 'no variant passes: stop'}")
