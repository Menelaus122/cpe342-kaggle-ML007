"""Round 40, Task 4, step 3: a second seed of the v30 DINOv2-L member (Colab), averaged with the first, against v30
(training images only; CPU).

    python gx_r40_t4_dinov2L_seed2_eval.py

Written 2026-11-18 ~02:40, before the seed-2 run started. dinov2L_hud70_s2 = the recipe of v30's dinov2L_hud70 (DINOv2 ViT-L/14,
126 x 238, lr 3e-5, layer decay 0.85, HUD dropout 0.70, 15 epochs) with --train-seed-offset 1000 (same folds).
Reference = v30 (v23 member weights + DINOv2-L at 8, or 2 on images whose grey mean / SD is at or below the 10% quantile of the other
four folds' training frames; 0.65 x members + 0.35 x the seven-backbone MLP probe; v17 rule; balanced class mix; views full / zoom70 /
HUD hidden gated on their own statistics; real dark / low-contrast frames = lowest decile, full view).
Candidate (one, fixed now): the DINOv2-L member's probabilities = the mean of the two seeds on every view, same gate and weights.
Information: seed 2 in place of seed 1.
Acceptance (user's rule against v30): zoom70, HUD hidden and the real dark / low-contrast frames all > 0, full >= -0.0005,
zoom70 / HUD-hidden mean better in >= 3/5 folds. A single pre-declared candidate, so no nested choice.
"""
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402

S1, S2 = "dinov2L_hud70", "dinov2L_hud70_s2"
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
P = {t: {v: np.zeros((len(y), 5)) for v in VIEWS} for t in V20 + [S1, S2]}
for t in V20 + [S1, S2]:
    for f, va in enumerate(folds):
        z30 = np.load(V30 / f"{t}_fold{f}.npz")
        z19 = np.load(V19 / f"{t}_fold{f}.npz") if t in V20 else z30
        for v in VIEWS:
            P[t][v][va] = (z30 if v in z30.files else z19)[v]
for f, va in enumerate(folds):
    z = np.load(PRED_DIR / f"t4_{S2}_fold{f}.npz")
    assert (z["val_idx"] == va).all() and (z["val_probs"].argmax(1) == P[S2]["full"][va].argmax(1)).mean() > 0.995
PR = dict(np.load("experiments/claude/outputs/r39/p7_probe_views.npz"))
ST = dict(np.load("experiments/claude/outputs/r39/t4_view_stats.npz"))
GATE = {v: np.zeros(len(y), bool) for v in VIEWS}
for f in range(5):
    tr, ho = fo != f, fo == f
    tb, tc = np.quantile(ST["brightness_full"][tr], 0.10), np.quantile(ST["contrast_full"][tr], 0.10)
    for v in VIEWS:
        GATE[v][ho] = (ST[f"brightness_{v}"][ho] <= tb) | (ST[f"contrast_{v}"][ho] <= tc)
st = np.load("experiments/claude/outputs/r30/t4_errors.npz")
dark = (st["contrast"] <= np.quantile(st["contrast"], 0.1)) | (st["brightness"] <= np.quantile(st["brightness"], 0.1))
pi = np.bincount(y, minlength=5) / len(y)
wb = (0.2 / pi)[y]
bal = lambda pred, m: f1_score(y[m], pred[m], average="macro", sample_weight=wb[m])
rule = lambda p: (p * pi ** -0.75).argmax(1)
allm = np.ones(len(y), bool)
SC = ["full", "zoom70", "centre_grey", "dark/low-contrast"]
VIEW_OF = {"full": "full", "zoom70": "zoom70", "centre_grey": "centre_grey", "dark/low-contrast": "full"}
msk = lambda k, m=allm: m & dark if k == "dark/low-contrast" else m


def labels(newp):
    out = {}
    for v in VIEWS:
        wn = np.where(GATE[v], 2.0, 8.0)
        mem = (sum(w * P[t][v] for t, w in WM.items()) + wn[:, None] * newp[v]) / (sum(WM.values()) + wn)[:, None]
        out[v] = rule(0.65 * mem + 0.35 * PR[v])
    return out


ref = labels(P[S1])
print("v30 reference: " + ", ".join(f"{k} {bal(ref[VIEW_OF[k]], msk(k)):.4f}" for k in SC))
for t in (S1, S2):
    print(f"{t} alone: " + ", ".join(f"{k} {bal(rule(P[t][VIEW_OF[k]]), msk(k)):.4f}" for k in SC))


def report(name, lab):
    d = {k: bal(lab[VIEW_OF[k]], msk(k)) - bal(ref[VIEW_OF[k]], msk(k)) for k in SC}
    pf = [np.mean([bal(lab[v], fo == f) - bal(ref[v], fo == f) for v in ("zoom70", "centre_grey")]) for f in range(5)]
    pdk = [bal(lab["full"], (fo == f) & dark) - bal(ref["full"], (fo == f) & dark) for f in range(5)]
    ok = d["zoom70"] > 0 and d["centre_grey"] > 0 and d["dark/low-contrast"] > 0 and d["full"] >= -0.0005 and sum(p > 0 for p in pf) >= 3
    print(f"  {name}\n    " + ", ".join(f"{k} {d[k]:+.4f}" for k in SC)
          + f"\n    zoom / HUD by fold {' '.join(f'{p:+.4f}' for p in pf)} | dark by fold {' '.join(f'{p:+.4f}' for p in pdk)}"
          + f"\n    -> {'PASS' if ok else 'fail'}", flush=True)
    return ok


print("\nCandidate against v30:")
ok = report("seed mean (seeds 1 + 2), same gate and weights", labels({v: 0.5 * (P[S1][v] + P[S2][v]) for v in VIEWS}))
print("Information:")
report("seed 2 in place of seed 1", labels(P[S2]))
print(f"\n-> the seed mean {'PASSES' if ok else 'fails'}")
