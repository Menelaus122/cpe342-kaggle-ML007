"""Round 41 step 2, Task 4: dinov3L_384 (DINOv3-L at 192 x 384, fold 0 only) against the current pipeline on fold 0 (CPU; training
images only).

    python hh_r41_t4_dinov3_384_fold0.py [--ref v31|v32]

Written 2026-11-18 evening, before the Colab run (colab/t4_colab_dinov3c.ipynb; notebook gate: alone >= 0.9925 on fold 0 and the
v24 check against the v23 members). Reference = the pipeline in use when fold 0 comes back: v31 (DINOv3-L x8, x2 on gated frames,
older members x1 / x3, probe 0.35) or v32 if adopted (hg_r41's setting: DINOv3-L x20 on every frame, older members x0.5 / x1.5,
probe 0.35); DINOv2-B x8 and DINOv2-L x8 / x2 gated in both. Variants, fixed now:
  R   dinov3L_384 in place of dinov3L_lr8e5, with the reference's DINOv3 weights
  H   both, each at half the reference's DINOv3 weights (the DINOv3 slot shared by two resolutions)
  A   both, each at the reference's full DINOv3 weights
Fold-0 reading of the user's rule: zoom70, HUD hidden and the real dark / low-contrast frames all > 0, full >= -0.0005. If at least
one variant meets it, folds 1-4 are worth their compute; the member is then judged on all five folds by a script written before
folds 1-4 (as he_r40).
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ref", choices=["v31", "v32"], default="v32")
ap.add_argument("--cand", default="dinov3L_384")
args = ap.parse_args()
CAND, D3, L2 = args.cand, "dinov3L_lr8e5", "dinov2L_hud70"
PLAIN, HUDM = ["cnxt", "deit3"], ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
V20 = PLAIN + HUDM + ["dinov2_v2"]
OLD, D3W = {"v31": (1.0, (8.0, 2.0)), "v32": (0.5, (20.0, 20.0))}[args.ref]
WM = {**{t: 1.0 * OLD for t in PLAIN}, **{t: 3.0 * OLD for t in HUDM}, "dinov2_v2": 8.0}
PW = 0.35
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
for t in V20 + [L2, D3, CAND]:
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
    """gated: {tag: (weight on normal frames, weight on gated frames)} on top of the fixed members."""
    out = {}
    for v in VIEWS:
        num = sum(w * P[t][v] for t, w in WM.items())
        den = np.full(len(va), sum(WM.values()))
        for t, (wn, wd) in gated.items():
            w = np.where(GATE[v], wd, wn)
            num, den = num + w[:, None] * P[t][v], den + w
        out[v] = rule((1 - PW) * num / den[:, None] + PW * PR[v])
    return out


half = (D3W[0] / 2, D3W[1] / 2)
ref = pipeline({L2: (8.0, 2.0), D3: D3W})
print(f"reference {args.ref} on fold 0: " + ", ".join(f"{k} {bal(ref[VIEW_OF[k]], msk(k)):.4f}" for k in SC))
print("alone (balanced mix, rule): " + " | ".join(
    f"{t} " + ", ".join(f"{k} {bal(rule(P[t][VIEW_OF[k]]), msk(k)):.4f}" for k in SC) for t in (D3, CAND)))
any_ok = False
for name, g in ((f"R  {CAND} in place of {D3}", {L2: (8.0, 2.0), CAND: D3W}),
                (f"H  both at half the DINOv3 weights {half}", {L2: (8.0, 2.0), D3: half, CAND: half}),
                (f"A  both at the full DINOv3 weights {D3W}", {L2: (8.0, 2.0), D3: D3W, CAND: D3W})):
    lab = pipeline(g)
    d = {k: bal(lab[VIEW_OF[k]], msk(k)) - bal(ref[VIEW_OF[k]], msk(k)) for k in SC}
    ok = d["zoom70"] > 0 and d["centre_grey"] > 0 and d["dark/low-contrast"] > 0 and d["full"] >= -0.0005
    any_ok |= ok
    print(f"  {name}\n    " + ", ".join(f"{k} {d[k]:+.4f}" for k in SC)
          + f" | labels changed on full frames {(lab['full'] != ref['full']).sum()}\n    -> fold-0 rule {'PASS' if ok else 'fail'}")
print(f"\n-> {'at least one variant passes: folds 1-4 worth running' if any_ok else 'no variant passes: stop'}")
