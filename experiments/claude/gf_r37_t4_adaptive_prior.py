"""Round 37, Task 4, queue 3 (CPU, light): a confidence-adaptive prior exponent on the v23 rule (training images only).

    python gf_r37_t4_adaptive_prior.py

v17-v26 decide argmax p * pi^-alpha with alpha = 0.75 for every image (pi = training class shares; logit adjustment, Menon et al.
2021). Hypothesis (user brief of 2026-11-17 ~01:20): confident images need little prior correction and ambiguous ones more:
    alpha(x) = a_lo if max p(x) >= t, else a_hi.
Grid (the brief): a_lo = 0.50, a_hi in {0.85, 0.90, 0.95, 1.00}, t = 0.80; plus a_lo in {0.50, 0.75} x a_hi in {0.75 ... 1.00} x
t in {0.6, 0.7, 0.8, 0.9} chosen nested (on four folds by the hard-view mean at a uniform class mix, scored on the fifth).
Scored on the v23 members' out-of-fold views (full, zoom70, HUD hidden, gamma 3.5, 75% grid; real dark / low-contrast frames on the
full view) at the uniform class mix (balanced weights, as every Task 4 rule since round 30) and at the training mix (unweighted),
because the user asked for both. Rules: hard-view rule (uniform mix), v24 check (zoom70, HUD hidden, dark frames > 0, full >=
-0.0005), and the training-mix full view >= -0.0005.
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
V23_W = {t: (8.0 if t == "dinov2_v2" else 3.0 if t in HUDM else 1.0) for t in V20}
ALL5 = ["full", "zoom70", "centre_grey", "dark_g35", "grid75"]
HARD = ALL5[1:]
V19, V30 = Path("experiments/claude/outputs/r19_views"), Path("experiments/claude/outputs/r30_views")
train, _ = task4.load_frames()
y = train.label.values
folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
fo = np.zeros(len(y), int)
for f, va in enumerate(folds):
    fo[va] = f
ens = {v: np.zeros((len(y), 5)) for v in ALL5}
for t, w in V23_W.items():
    for f, va in enumerate(folds):
        z30, z19 = np.load(V30 / f"{t}_fold{f}.npz"), np.load(V19 / f"{t}_fold{f}.npz")
        for v in ALL5:
            ens[v][va] += w * (z30 if v in z30.files else z19)[v] / sum(V23_W.values())
st = np.load("experiments/claude/outputs/r30/t4_errors.npz")
dark = (st["contrast"] <= np.quantile(st["contrast"], 0.1)) | (st["brightness"] <= np.quantile(st["brightness"], 0.1))
pi = np.bincount(y, minlength=5) / len(y)
wb = (0.2 / pi)[y]
allm = np.ones(len(y), bool)
f1u = lambda pred, m: f1_score(y[m], pred[m], average="macro", sample_weight=wb[m])     # uniform class mix
f1t = lambda pred, m: f1_score(y[m], pred[m], average="macro")                          # training mix
SCORE = ALL5 + ["dark/low-contrast"]
VIEW_OF = {**{v: v for v in ALL5}, "dark/low-contrast": "full"}
conf = {v: ens[v].max(1) for v in ALL5}


def labels(a_lo, a_hi, t):
    return {v: (ens[v] * pi[None, :] ** -np.where(conf[v] >= t, a_lo, a_hi)[:, None]).argmax(1) for v in ALL5}


base = labels(0.75, 0.75, 0.0)
print(f"{len(y)} training images; share of images with max p >= 0.8 per view: " + ", ".join(f"{v} {np.mean(conf[v] >= 0.8):.3f}" for v in ALL5)
      + f"; dark frames {np.mean(conf['full'][dark] >= 0.8):.3f}")
print("v23 rule (alpha 0.75), uniform mix: " + ", ".join(f"{k} {f1u(base[VIEW_OF[k]], dark if k == 'dark/low-contrast' else allm):.4f}" for k in SCORE)
      + " | training mix: " + ", ".join(f"{k} {f1t(base[VIEW_OF[k]], dark if k == 'dark/low-contrast' else allm):.4f}" for k in SCORE))


def report(name, lab):
    du = {k: (lambda m: f1u(lab[VIEW_OF[k]], m) - f1u(base[VIEW_OF[k]], m))(dark if k == "dark/low-contrast" else allm) for k in SCORE}
    dt = {k: (lambda m: f1t(lab[VIEW_OF[k]], m) - f1t(base[VIEW_OF[k]], m))(dark if k == "dark/low-contrast" else allm) for k in SCORE}
    hard = np.mean([du[v] for v in HARD])
    pf = [np.mean([f1u(lab[v], fo == f) - f1u(base[v], fo == f) for v in HARD]) for f in range(5)]
    ok_user = hard >= 0.0015 and du["full"] >= -0.0005
    ok_v24 = du["zoom70"] > 0 and du["centre_grey"] > 0 and du["dark/low-contrast"] > 0 and du["full"] >= -0.0005
    ok_train = dt["full"] >= -0.0005
    print(f"  {name}\n    uniform mix: hard-view mean {hard:+.4f}, folds +{sum(p > 0 for p in pf)}/5; " + ", ".join(f"{k} {du[k]:+.4f}" for k in SCORE)
          + "\n    training mix: " + ", ".join(f"{k} {dt[k]:+.4f}" for k in SCORE)
          + f"\n    -> hard-view rule {'PASS' if ok_user else 'fail'}, v24 check {'PASS' if ok_v24 else 'fail'}, training-mix full "
          f"{'PASS' if ok_train else 'fail'}", flush=True)


for a_hi in (0.85, 0.90, 0.95, 1.00):
    report(f"brief: alpha 0.50 if max p >= 0.80, else {a_hi:.2f}", labels(0.50, a_hi, 0.80))
report("control: alpha 0.50 for every image", labels(0.50, 0.50, 0.0))
report("control: alpha 1.00 for every image", labels(1.00, 1.00, 0.0))
GRID = list(itertools.product((0.50, 0.75), (0.75, 0.85, 0.90, 0.95, 1.00), (0.6, 0.7, 0.8, 0.9)))
LABS = {g: labels(*g) for g in GRID}
nested = {v: base[v].copy() for v in ALL5}
picks = []
for f in range(5):
    m = fo != f
    sc = {g: np.mean([f1u(LABS[g][v], m) - f1u(base[v], m) for v in HARD]) for g in GRID
          if f1u(LABS[g]["full"], m) - f1u(base["full"], m) >= -0.0005}
    g = max(sc, key=sc.get)
    picks.append(g)
    for v in ALL5:
        nested[v][fo == f] = LABS[g][v][fo == f]
report("nested choice of (a_lo, a_hi, t)", nested)
print(f"  nested picks: {picks}")
