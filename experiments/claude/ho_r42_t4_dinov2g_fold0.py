"""Round 42, Task 4: dinov2g_126 (fine-tuned DINOv2-g/14 at 126 x 238, fold 0 only) against the pipeline in use on fold 0 (CPU;
training images only).

    python ho_r42_t4_dinov2g_fold0.py [--ref v36|v37]

Written 2026-11-20, before the Colab run (colab/t4_colab_dinov2g.ipynb; notebook gate: alone >= 0.9925 on fold 0 and the v24 check
against the v23 members). Reference = the pipeline in use when fold 0 comes back: v36 (older seven x0.5 / x1.5, DINOv2-B 8, DINOv2-L
8 / 2 gated, DINOv3-L-144 20, DINOv3-L-384 20, C-RADIOv3-L 20 / 2 gated, ViT-H+ 20, probe 0.35) or, if the round-42 re-weighting is
adopted as v37, its setting (ViT-H+ 40, DINOv3-L-144 10, DINOv3-L-384 20, C-RADIOv3-L 10 / 2 gated, older x0.25 / x0.75, probe 0.30).
Variants, fixed now (the gated variant declared up front, as for ViT-H+):
  R  dinov2g_126 in place of dinov2L_hud70 (the same family and 9 x 17 grid), with DINOv2-L's weights: x8 normal / x2 gated
  H  dinov2g_126 added at x10 on every frame
  A  dinov2g_126 added at x20 on every frame
  G  dinov2g_126 added at x20 on normal frames, x2 on gated frames (v30's gate unchanged)
Fold-0 reading of the user's rule: zoom70, HUD hidden and the real dark / low-contrast frames all > 0, full >= -0.0005. If at least
one variant meets it, folds 1-4 are worth their compute; the member is then judged on all five folds by a script written before
folds 1-4 (as hn_r42).
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ref", choices=["v36", "v37"], default="v36")
ap.add_argument("--cand", default="dinov2g_126")
args = ap.parse_args()
CAND, L2, D3, D384, RAD, HP = args.cand, "dinov2L_hud70", "dinov3L_lr8e5", "dinov3L_384", "cradio3L_384", "dinov3Hp_144"
PLAIN, HUDM = ["cnxt", "deit3"], ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
V20 = PLAIN + HUDM + ["dinov2_v2"]
H_W, D144_W, D384_W, RAD_W, OLD, PW = {"v36": (20.0, 20.0, 20.0, 20.0, 1.0, 0.35), "v37": (40.0, 10.0, 20.0, 10.0, 0.5, 0.30)}[args.ref]
WM = {**{t: 0.5 * OLD for t in PLAIN}, **{t: 1.5 * OLD for t in HUDM}, "dinov2_v2": 8.0, D3: D144_W, D384: D384_W, HP: H_W}
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


base = {L2: (8.0, 2.0), RAD: (RAD_W, 2.0)}
ref = pipeline(base)
print(f"reference {args.ref} on fold 0: " + ", ".join(f"{k} {bal(ref[VIEW_OF[k]], msk(k)):.4f}" for k in SC))
print("alone (balanced mix, rule): " + " | ".join(
    f"{t} " + ", ".join(f"{k} {bal(rule(P[t][VIEW_OF[k]]), msk(k)):.4f}" for k in SC) for t in (L2, HP, CAND)))
any_ok = False
for name, g in ((f"R  {CAND} in place of {L2} (x8 / x2 gated)", {RAD: (RAD_W, 2.0), CAND: (8.0, 2.0)}),
                (f"H  {CAND} added at x10", {**base, CAND: (10.0, 10.0)}),
                (f"A  {CAND} added at x20", {**base, CAND: (20.0, 20.0)}),
                (f"G  {CAND} added at x20, x2 on gated frames", {**base, CAND: (20.0, 2.0)})):
    lab = pipeline(g)
    d = {k: bal(lab[VIEW_OF[k]], msk(k)) - bal(ref[VIEW_OF[k]], msk(k)) for k in SC}
    ok = d["zoom70"] > 0 and d["centre_grey"] > 0 and d["dark/low-contrast"] > 0 and d["full"] >= -0.0005
    any_ok |= ok
    print(f"  {name}\n    " + ", ".join(f"{k} {d[k]:+.4f}" for k in SC)
          + f" | labels changed on full frames {(lab['full'] != ref['full']).sum()}\n    -> fold-0 rule {'PASS' if ok else 'fail'}")
print(f"\n-> {'at least one variant passes: folds 1-4 worth running' if any_ok else 'no variant passes: stop'}")
