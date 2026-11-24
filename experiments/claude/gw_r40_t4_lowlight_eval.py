"""Round 40, Task 4, step 2: fold-0 test of the low-light + channel-shuffle augmentation on ConvNeXt-T with HUD dropout
(training images only; CPU after the fold-0 run).

    python gw_r40_t4_lowlight_eval.py [tag]            (default cnxt_hud_lowlight)

Written while the fold-0 model was training (2026-11-18 ~02:00), before any of its predictions were seen.
Candidate: cnxt_hud recipe (ConvNeXt-T in12k, 160 x 320, lr 2e-4, 10 epochs, HUD dropout 0.5) + --p-lowlight 0.15 (brightness x0.05-0.30
or gamma 2-4, then Gaussian noise sd 0.05-0.15) + --p-chshuffle 0.30 (random RGB order), scripts/train_task4.py, fold 0 only.
Views (full, zoom70, HUD hidden, gamma 3.5, 75% grid; h-flip TTA) from em_r30_t4_hardviews.py.
Two comparisons on fold 0 (balanced class mix, v17 rule; real dark / low-contrast frames = lowest decile of contrast or brightness):
  member: the candidate alone against cnxt_hud alone (the recipe test that decides step 3)
  ensemble: v30 with cnxt_hud replaced by the candidate (weight 3; gate and probe as v30) against v30 (information; a member change
            would need all five folds)
Recipe PASS (fixed now): member-level deltas on fold 0 with real dark frames > 0, zoom70 > 0, HUD hidden > 0 and full >= -0.0005.
Gamma 3.5 is printed but not counted: the low-light augmentation includes gamma 2-4, so that view is no longer out of distribution.
Caveat: one seed on one fold (about 830 dark frames), so effects below about 0.002 are within seed noise.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "cnxt_hud_lowlight"
NEW, OLD = "dinov2L_hud70", "cnxt_hud"
PLAIN, HUDM = ["cnxt", "deit3"], ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
V20 = PLAIN + HUDM + ["dinov2_v2"]
ALL5, VIEWS = ["full", "zoom70", "centre_grey", "dark_g35", "grid75"], ["full", "zoom70", "centre_grey"]
V19, V30 = Path("experiments/claude/outputs/r19_views"), Path("experiments/claude/outputs/r30_views")
train, _ = task4.load_frames()
y = train.label.values
folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
va = folds[0]
tr_rows = np.setdiff1d(np.arange(len(y)), va)


def views(t):
    z30 = np.load(V30 / f"{t}_fold0.npz")
    z19 = np.load(V19 / f"{t}_fold0.npz") if (V19 / f"{t}_fold0.npz").exists() else z30
    return {v: (z30 if v in z30.files else z19)[v] for v in ALL5}


P = {t: views(t) for t in V20 + [NEW, TAG]}
z = np.load(PRED_DIR / f"t4_{TAG}_fold0.npz")
assert (z["val_idx"] == va).all() and (z["val_probs"].argmax(1) == P[TAG]["full"].argmax(1)).mean() > 0.995
PR = {v: a[va] for v, a in np.load("experiments/claude/outputs/r39/p7_probe_views.npz").items()}
ST = dict(np.load("experiments/claude/outputs/r39/t4_view_stats.npz"))
tb, tc = np.quantile(ST["brightness_full"][tr_rows], 0.10), np.quantile(ST["contrast_full"][tr_rows], 0.10)
GATE = {v: (ST[f"brightness_{v}"][va] <= tb) | (ST[f"contrast_{v}"][va] <= tc) for v in VIEWS}
st = np.load("experiments/claude/outputs/r30/t4_errors.npz")
dark = ((st["contrast"] <= np.quantile(st["contrast"], 0.1)) | (st["brightness"] <= np.quantile(st["brightness"], 0.1)))[va]
yv = y[va]
pi = np.bincount(y, minlength=5) / len(y)
wb = (0.2 / pi)[yv]
bal = lambda pred, m: f1_score(yv[m], pred[m], average="macro", sample_weight=wb[m])
rule = lambda p: (p * pi ** -0.75).argmax(1)
allm = np.ones(len(va), bool)
SC = ["full", "zoom70", "centre_grey", "dark/low-contrast", "dark_g35", "grid75"]
VIEW_OF = {"full": "full", "zoom70": "zoom70", "centre_grey": "centre_grey", "dark/low-contrast": "full", "dark_g35": "dark_g35", "grid75": "grid75"}
msk = lambda k: dark if k == "dark/low-contrast" else allm


def score(lab):
    return {k: bal(lab[VIEW_OF[k]], msk(k)) for k in SC}


def show(name, a, b):
    d = {k: a[k] - b[k] for k in SC}
    ok = d["dark/low-contrast"] > 0 and d["zoom70"] > 0 and d["centre_grey"] > 0 and d["full"] >= -0.0005
    print(f"  {name}: " + ", ".join(f"{k} {a[k]:.4f} ({d[k]:+.4f})" for k in SC) + f"\n    -> {'PASS' if ok else 'fail'} (gamma not counted)")
    return ok


alone = lambda t: score({v: rule(P[t][v]) for v in ALL5})
print(f"fold 0, {len(va)} images, {dark.sum()} dark / low-contrast")
print(f"  {OLD} alone: " + ", ".join(f"{k} {v:.4f}" for k, v in alone(OLD).items()))
ok_member = show(f"{TAG} alone vs {OLD}", alone(TAG), alone(OLD))


def ens(member):
    wm = {**{t: 1.0 for t in PLAIN}, **{t: 3.0 for t in HUDM}, "dinov2_v2": 8.0}
    if member != OLD:
        wm[member] = wm.pop(OLD)
    out = {}
    for v in ALL5:
        s_rest = sum(w * P[t][v] for t, w in wm.items())
        wn = np.where(GATE[v], 2.0, 8.0) if v in VIEWS else np.full(len(va), 8.0)
        mem = (s_rest + wn[:, None] * P[NEW][v]) / (sum(wm.values()) + wn)[:, None]
        out[v] = rule(0.65 * mem + 0.35 * PR[v]) if v in VIEWS else rule(mem)
    return score(out)


print("ensemble (information):")
show(f"v30 with {OLD} -> {TAG}", ens(TAG), ens(OLD))
print(f"\n-> recipe {'PASSES' if ok_member else 'fails'} the fold-0 member test (step 3: DINOv2-L seed 2 "
      f"{'with' if ok_member else 'without'} the new augmentation)")
