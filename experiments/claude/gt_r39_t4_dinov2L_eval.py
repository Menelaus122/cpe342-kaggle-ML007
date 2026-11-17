"""Round 39 step 3, Task 4: the Colab member dinov2L_hud70 on all five folds against v29 (training images only; CPU).

    python gt_r39_t4_dinov2L_eval.py [tag]          (default tag dinov2L_hud70)

Written 2026-11-17 ~22:15, after the fold-0 gate passed on Colab (fold 0 alone against the v23 members: added at weight 8, full +0.0016,
zoom70 +0.0029, HUD hidden +0.0012, real dark frames +0.0006) and BEFORE folds 1-4 were seen.
Member: DINOv2 ViT-L/14 at 126 x 238, lr 3e-5, layer decay 0.85, HUD dropout 0.70, 15 epochs (Colab A100). Its five views per fold come
from experiments/claude/outputs/r30_views/<tag>_fold<f>.npz (em_r30_t4_hardviews.py on Colab), the full-frame OOF from
outputs/preds/t4_<tag>_fold<f>.npz (checked against the views).
Reference = v29: the eight v23 members (DINOv2-B x8, HUD members x3, others x1) + 0.35 x the seven-backbone MLP probe
(experiments/claude/outputs/r39/p7_probe_views.npz, written by gs_r39_t4_tta_rotgray.py), v17 rule, balanced class mix; real dark /
low-contrast frames = lowest decile of contrast or brightness (full view).
Variants (fixed now; the probe stays at 0.35 of the final mix):
  A2 / A4 / A8   the member added at weight 2 / 4 / 8 (dinov2_v2 stays at 8)
  R8             the member in place of dinov2_v2 (weight 8)
  N              nested over the four (per held-out fold, the best mean of the zoom70 / HUD-hidden / dark deltas on the other four
                 folds among those with full >= -0.0005 there; none positive: the fold keeps v29)
Acceptance (user's rule against v29, applied to N): zoom70, HUD hidden and the real dark / low-contrast frames all > 0,
full >= -0.0005, zoom70 / HUD-hidden mean better in >= 3/5 folds. If N passes, the pipeline variant is the one with the best criterion
on all five folds. Gamma 3.5 and the 75% grid (members only, no probe views) are printed for information.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "dinov2L_hud70"
V20 = ["cnxt", "deit3", "cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud", "dinov2_v2"]
HUDM = ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
W23 = {t: (8.0 if t == "dinov2_v2" else 3.0 if t in HUDM else 1.0) for t in V20}
ALL5 = ["full", "zoom70", "centre_grey", "dark_g35", "grid75"]
PV = ["full", "zoom70", "centre_grey"]                       # views with probe probabilities
V19, V30 = Path("experiments/claude/outputs/r19_views"), Path("experiments/claude/outputs/r30_views")
PW = 0.35

train, _ = task4.load_frames()
y = train.label.values
folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
fo = np.zeros(len(y), int)
for f, va in enumerate(folds):
    fo[va] = f
P = {t: {v: np.zeros((len(y), 5)) for v in ALL5} for t in V20 + [TAG]}
for t in V20 + [TAG]:
    for f, va in enumerate(folds):
        z30 = np.load(V30 / f"{t}_fold{f}.npz")
        z19 = np.load(V19 / f"{t}_fold{f}.npz") if t in V20 else z30
        for v in ALL5:
            P[t][v][va] = (z30 if v in z30.files else z19)[v]
    if t == TAG:
        for f, va in enumerate(folds):
            z = np.load(PRED_DIR / f"t4_{TAG}_fold{f}.npz")
            assert (z["val_idx"] == va).all(), f"fold {f}: validation rows differ from the pipeline's folds"
            agree = (z["val_probs"].argmax(1) == P[TAG]["full"][va].argmax(1)).mean()
            assert agree > 0.995, f"fold {f}: OOF and full view disagree ({agree:.4f})"
PR = dict(np.load("experiments/claude/outputs/r39/p7_probe_views.npz"))

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


def ens(wmap):
    s = sum(wmap.values())
    out = {}
    for v in ALL5:
        mem = sum(w * P[t][v] for t, w in wmap.items()) / s
        out[v] = rule((1 - PW) * mem + PW * PR[v]) if v in PV else rule(mem)
    return out


ref = ens(W23)
print("v29 reference: " + ", ".join(f"{k} {bal(ref[VIEW_OF[k]], msk(k)):.4f}" for k in SC)
      + f" | members only: dark_g35 {bal(ref['dark_g35'], allm):.4f}, grid75 {bal(ref['grid75'], allm):.4f}", flush=True)
full_alone = rule(P[TAG]["full"])
print(f"{TAG} alone: " + ", ".join(f"{k} {bal(rule(P[TAG][VIEW_OF[k]]), msk(k)):.4f}" for k in SC)
      + " | by fold (full) " + " ".join(f"{bal(full_alone, fo == f):.4f}" for f in range(5)))
print("dinov2_v2 alone: " + ", ".join(f"{k} {bal(rule(P['dinov2_v2'][VIEW_OF[k]]), msk(k)):.4f}" for k in SC), flush=True)


def deltas(lab, m=allm):
    return {k: bal(lab[VIEW_OF[k]], msk(k, m)) - bal(ref[VIEW_OF[k]], msk(k, m)) for k in SC}


def crit(lab, m):
    d = deltas(lab, m)
    return np.mean([d["zoom70"], d["centre_grey"], d["dark/low-contrast"]]) if d["full"] >= -0.0005 else -np.inf


def report(name, lab):
    d = deltas(lab)
    pf = [np.mean([bal(lab[v], fo == f) - bal(ref[v], fo == f) for v in ("zoom70", "centre_grey")]) for f in range(5)]
    pdk = [bal(lab["full"], (fo == f) & dark) - bal(ref["full"], (fo == f) & dark) for f in range(5)]
    pfull = [bal(lab["full"], fo == f) - bal(ref["full"], fo == f) for f in range(5)]
    ok = d["zoom70"] > 0 and d["centre_grey"] > 0 and d["dark/low-contrast"] > 0 and d["full"] >= -0.0005 and sum(p > 0 for p in pf) >= 3
    print(f"  {name}\n    " + ", ".join(f"{k} {d[k]:+.4f}" for k in SC)
          + f" | gamma {bal(lab['dark_g35'], allm) - bal(ref['dark_g35'], allm):+.4f}, grid {bal(lab['grid75'], allm) - bal(ref['grid75'], allm):+.4f}"
          + f"\n    zoom / HUD by fold {' '.join(f'{p:+.4f}' for p in pf)} | dark by fold {' '.join(f'{p:+.4f}' for p in pdk)}"
          + f" | full by fold {' '.join(f'{p:+.4f}' for p in pfull)}\n    -> {'PASS' if ok else 'fail'}", flush=True)
    return ok


C = {f"A{w:g}": ens({**W23, TAG: float(w)}) for w in (2, 4, 8)}
C["R8"] = ens({**{t: w for t, w in W23.items() if t != "dinov2_v2"}, TAG: 8.0})
print("\nVariants against v29:")
for s, lab in C.items():
    report(s, lab)
outN, picks = {v: ref[v].copy() for v in ALL5}, []
for f in range(5):
    sc = {s: crit(lab, fo != f) for s, lab in C.items()}
    s = max(sc, key=sc.get) if max(sc.values()) > 0 else None
    picks.append(s)
    if s is not None:
        for v in ALL5:
            outN[v][fo == f] = C[s][v][fo == f]
okN = report(f"N nested over A2 / A4 / A8 / R8; picks {picks}", outN)
best = max(C, key=lambda s: crit(C[s], allm))
print(f"\n-> N {'PASSES' if okN else 'fails'}; the variant with the best criterion on all five folds: {best} (pipeline only if N passes)")
