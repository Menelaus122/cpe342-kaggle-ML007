"""Round 42 part B, Task 4: cradio3L_384 with v30's brightness / contrast gate, fold 0 against v33 (CPU; training images only).

    python hk_r42_t4_radio_gated_fold0.py

User brief of 2026-11-19, AFTER hj_r42_t4_radio_fold0.py had failed on fold 0 (R / H / A gain on full / zoom70 / HUD hidden, lose on the
real dark frames). POST HOC: these variants were proposed after fold 0 was seen, so fold 0 is no longer an unbiased test of them; a
pass here only says folds 1-4 are worth their compute, and the member must then be judged on folds 1-4 alone (fold 0 excluded) by a
script written before folds 1-4.
Gate = v30's mechanism unchanged (no new threshold): a frame is gated when its grey-level mean or SD is at or below the 10% quantile
of the training images (here the other four folds, as every evaluation since gu_r39), computed per view from the view itself.
Variants (all four fixed by the user's brief): C-RADIOv3-L added to v33 with weight wn on normal frames and wd on gated frames,
wn in {10, 20} x wd in {2, 0}. Reference v33 and scoring exactly as hj_r42 (full, zoom70, HUD hidden, real dark / low-contrast frames;
balanced class mix, v17 rule). Reading (user's): a variant meeting zoom70, HUD hidden and dark all > 0 and full >= -0.0005 -> consider
folds 1-4 + full refit.
"""
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402

CAND, L2, D3, D384 = "cradio3L_384", "dinov2L_hud70", "dinov3L_lr8e5", "dinov3L_384"
PLAIN, HUDM = ["cnxt", "deit3"], ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
V20 = PLAIN + HUDM + ["dinov2_v2"]
WM = {**{t: 0.5 for t in PLAIN}, **{t: 1.5 for t in HUDM}, "dinov2_v2": 8.0, D3: 20.0}
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
P = {}
for t in V20 + [L2, D3, D384, CAND]:
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


ref = pipeline({L2: (8.0, 2.0), D384: (20.0, 20.0)})
print("reference v33 on fold 0: " + ", ".join(f"{k} {bal(ref[VIEW_OF[k]], msk(k)):.4f}" for k in SC))
print(f"gate (v30's Q10, thresholds from folds 1-4): {GATE['full'].mean():.1%} of fold-0 full frames gated; "
      f"real dark / low-contrast frames {dark.sum()}, of which gated {(GATE['full'] & dark).sum()}")
print(f"{CAND} alone on gated / not gated full frames: "
      f"{bal(rule(P[CAND]['full']), GATE['full']):.4f} / {bal(rule(P[CAND]['full']), ~GATE['full']):.4f} "
      f"(dinov3L_384 {bal(rule(P[D384]['full']), GATE['full']):.4f} / {bal(rule(P[D384]['full']), ~GATE['full']):.4f})")
any_ok = False
for wn in (10.0, 20.0):
    for wd in (2.0, 0.0):
        lab = pipeline({L2: (8.0, 2.0), D384: (20.0, 20.0), CAND: (wn, wd)})
        d = {k: bal(lab[VIEW_OF[k]], msk(k)) - bal(ref[VIEW_OF[k]], msk(k)) for k in SC}
        ok = d["zoom70"] > 0 and d["centre_grey"] > 0 and d["dark/low-contrast"] > 0 and d["full"] >= -0.0005
        any_ok |= ok
        print(f"  G  {CAND} x{wn:g} on normal frames, x{wd:g} on gated frames\n    " + ", ".join(f"{k} {d[k]:+.4f}" for k in SC)
              + f" | labels changed on full frames {(lab['full'] != ref['full']).sum()} (dark frames {((lab['full'] != ref['full']) & dark).sum()})"
              + f"\n    -> fold-0 rule {'PASS' if ok else 'fail'}")
print(f"\n-> {'at least one gated variant passes on fold 0 (post hoc): folds 1-4 would be the real test' if any_ok else 'no gated variant passes: stop'}")
