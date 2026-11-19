"""Round 42 part B, Task 4: cradio3L_384 (fine-tuned C-RADIOv3-L at 192 x 384, fold 0 only) against the current pipeline on fold 0
(CPU; training images only).

    python hj_r42_t4_radio_fold0.py [--probe p7|p8] [--pw 0.35]

Written 2026-11-19, before the Colab run (colab/t4_colab_radio3.ipynb; notebook gate: alone >= 0.9925 on fold 0 and the v24 check
against the v23 members). Reference = v33 by default: the older seven members x0.5 (ConvNeXt-T / DeiT-III 0.5, HUD members 1.5),
DINOv2-B 8, DINOv2-L 8 / 2 on gated frames, both DINOv3-L members x20 on every frame, 0.65 x members + 0.35 x the seven-backbone MLP
probe (P7), v17 rule, balanced class mix. If round 42 part A changes the probe (P8 = P7 + frozen C-RADIOv3-L, ib_r42), pass
--probe p8 --pw <its weight> so the reference is the pipeline in use when fold 0 comes back. Variants, fixed now:
  R  cradio3L_384 in place of the 192 x 384 DINOv3-L member (x20 on every frame)
  H  cradio3L_384 added at x10 on every frame
  A  cradio3L_384 added at x20 on every frame
Fold-0 reading of the user's rule: zoom70, HUD hidden and the real dark / low-contrast frames all > 0, full >= -0.0005. If at least
one variant meets it, folds 1-4 are worth their compute; the member is then judged on all five folds by a script written before
folds 1-4 (as hi_r41).
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--probe", choices=["p7", "p8"], default="p7")
ap.add_argument("--pw", type=float, default=0.35)
ap.add_argument("--cand", default="cradio3L_384")
args = ap.parse_args()
CAND, L2, D3, D384 = args.cand, "dinov2L_hud70", "dinov3L_lr8e5", "dinov3L_384"
PLAIN, HUDM = ["cnxt", "deit3"], ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
V20 = PLAIN + HUDM + ["dinov2_v2"]
WM = {**{t: 0.5 for t in PLAIN}, **{t: 1.5 for t in HUDM}, "dinov2_v2": 8.0, D3: 20.0}
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
for t in V20 + [L2, D3, D384, CAND]:
    z30 = np.load(V30 / f"{t}_fold0.npz")
    z19 = np.load(V19 / f"{t}_fold0.npz") if t in V20 else z30
    P[t] = {v: (z30 if v in z30.files else z19)[v] for v in VIEWS}
if args.probe == "p7":
    PR = {v: np.load("experiments/claude/outputs/r39/p7_probe_views.npz")[v][va] for v in VIEWS}
else:
    PR = {v: np.load("experiments/claude/outputs/r42/ib_probe_views.npz")[f"P8|{v}"][va] for v in VIEWS}
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
    """gated: {tag: (weight on normal frames, weight on gated frames)} on top of the fixed members."""
    out = {}
    for v in VIEWS:
        num = sum(w * P[t][v] for t, w in WM.items())
        den = np.full(len(va), sum(WM.values()))
        for t, (wn, wd) in gated.items():
            w = np.where(GATE[v], wd, wn)
            num, den = num + w[:, None] * P[t][v], den + w
        out[v] = rule((1 - args.pw) * num / den[:, None] + args.pw * PR[v])
    return out


ref = pipeline({L2: (8.0, 2.0), D384: (20.0, 20.0)})
print(f"reference (v33 members + {args.pw:.2f} x {args.probe.upper()}) on fold 0: "
      + ", ".join(f"{k} {bal(ref[VIEW_OF[k]], msk(k)):.4f}" for k in SC))
print("alone (balanced mix, rule): " + " | ".join(
    f"{t} " + ", ".join(f"{k} {bal(rule(P[t][VIEW_OF[k]]), msk(k)):.4f}" for k in SC) for t in (D3, D384, CAND)))
any_ok = False
for name, g in ((f"R  {CAND} in place of {D384} (x20)", {L2: (8.0, 2.0), CAND: (20.0, 20.0)}),
                (f"H  {CAND} added at x10", {L2: (8.0, 2.0), D384: (20.0, 20.0), CAND: (10.0, 10.0)}),
                (f"A  {CAND} added at x20", {L2: (8.0, 2.0), D384: (20.0, 20.0), CAND: (20.0, 20.0)})):
    lab = pipeline(g)
    d = {k: bal(lab[VIEW_OF[k]], msk(k)) - bal(ref[VIEW_OF[k]], msk(k)) for k in SC}
    ok = d["zoom70"] > 0 and d["centre_grey"] > 0 and d["dark/low-contrast"] > 0 and d["full"] >= -0.0005
    any_ok |= ok
    print(f"  {name}\n    " + ", ".join(f"{k} {d[k]:+.4f}" for k in SC)
          + f" | labels changed on full frames {(lab['full'] != ref['full']).sum()}\n    -> fold-0 rule {'PASS' if ok else 'fail'}")
print(f"\n-> {'at least one variant passes: folds 1-4 worth running' if any_ok else 'no variant passes: stop'}")
