"""Round 39, Task 4, step 2b: the v29 MLP probe with frozen features of two more backbones (OpenCLIP ViT-H/14 LAION-2B and Swin-L IN-22k)
(training images only; features extracted on Colab by gm_r38_t4_feat_extract.py, checked against the laptop GPU on 128 images;
the laptop GPU fits the probes).

    python gq_r39_t4_probe9.py

Reference = v29: v23 members + 0.35 x the raw MLP probe on the seven v28 backbones (7,040 features, z-scored; settings unchanged).
Probes: the same MLP (scripts/train_task4_probe.fit_mlp, seeds 0, 1, 2), fitted on the full / zoom70 / HUD-hidden rows of the training
folds, out of fold on the members' folds, on
  P9  the seven + ViT-H/14 + Swin-L   (9,856 features)
  P8H the seven + ViT-H/14            (8,320)
  P8S the seven + Swin-L              (8,576)
Candidates (fixed before the run; blending (1 - w) x members + w x probe, v17 rule after it):
  A  P9,  w in {0.30, 0.35, 0.40, 0.45}
  B  P8H, w in {0.30, 0.35, 0.40, 0.45}
  C  P8S, w in {0.30, 0.35, 0.40, 0.45}
  W  the v29 probe at w in {0.40, 0.45}  (the weight alone, since the grid now goes past 0.35)
  N  nested over all 14 settings together.
Nested = per held-out fold, the setting with the best mean of the zoom70 / HUD-hidden / dark-frame deltas against v29 on the other four
folds among those with full >= -0.0005 there (none positive: the fold keeps v29).
Acceptance (user's rule, against v29, applied to N): zoom70, HUD hidden and the real dark / low-contrast frames all > 0, full >= -0.0005,
zoom70 / HUD-hidden mean better in >= 3/5 folds. If N passes, the pipeline setting is the one with the best criterion on all five folds.
A, B, C, W and the probes alone (also ViT-H and Swin-L alone, as diagnostics) are reported for information.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)

sys.path.insert(0, "scripts")
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402
import train_task4_probe as P0  # noqa: E402  (BACKBONES of v28 / v29, fit_mlp)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FIT = ["full", "zoom70", "centre_grey"]
WS = (0.30, 0.35, 0.40, 0.45)
NEW = ["vitH14_clip_laion2b_224", "swinL_in22k_224"]
t0 = time.time()
log = lambda m: print(f"{m} [{(time.time() - t0) / 60:.1f} min]", flush=True)

train, test = task4.load_frames()
y = train.label.values.astype(np.int64)
B = {}                                                              # per backbone: {view: (n, d) float16}
for tag in [b[2] for b in P0.BACKBONES] + NEW:
    z, zv = np.load(PRED_DIR / f"t4_feat_{tag}.npz"), np.load(PRED_DIR / f"t4_feat_views_{tag}.npz")
    B[tag] = {"full": z["train"], "zoom70": zv["zoom70"], "centre_grey": zv["centre_grey"]}
    assert len(z["train"]) == len(y) and ("full" not in zv.files or np.array_equal(z["train"], zv["full"]))
SEVEN = [b[2] for b in P0.BACKBONES]
log(f"{len(B)} backbones loaded; device {DEV}")

folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
fo = np.zeros(len(y), int)
for f, va in enumerate(folds):
    fo[va] = f


def oof(tags):
    X = {v: np.hstack([B[t][v].astype(np.float32) for t in tags]) for v in FIT}
    out = {v: np.zeros((len(y), 5)) for v in FIT}
    for f, va in enumerate(folds):
        tr = np.where(fo != f)[0]
        mu, sd = X["full"][tr].mean(0), X["full"][tr].std(0) + 1e-6
        p = lambda Q: ((Q - mu) / sd).astype(np.float32)
        A, Y = np.vstack([p(X[v][tr]) for v in FIT]), np.tile(y[tr], len(FIT))
        preds = [P0.fit_mlp(A, Y, s, device=DEV) for s in P0.MLP_SEEDS]
        del A
        torch.cuda.empty_cache()
        for v in FIT:
            out[v][va] = np.mean([q(p(X[v][va])) for q in preds], 0)
        del preds
    return out


# members (v23 weights), dark frames, scoring (as gp_r39_t4_probe_heads.py)
V20 = ["cnxt", "deit3", "cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud", "dinov2_v2"]
HUDM = ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
V23_W = {t: (8.0 if t == "dinov2_v2" else 3.0 if t in HUDM else 1.0) for t in V20}
V19, V30 = Path("experiments/claude/outputs/r19_views"), Path("experiments/claude/outputs/r30_views")
M = {v: np.zeros((len(y), 5)) for v in FIT}
for t, w in V23_W.items():
    for f, va in enumerate(folds):
        z30, z19 = np.load(V30 / f"{t}_fold{f}.npz"), np.load(V19 / f"{t}_fold{f}.npz")
        for v in FIT:
            M[v][va] += w * (z30 if v in z30.files else z19)[v] / sum(V23_W.values())
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
mix = lambda P, w: {v: rule((1 - w) * M[v] + w * P[v]) for v in FIT}
alone = lambda P: ", ".join(f"{k} {bal(rule(P[VIEW_OF[k]]), msk(k)):.4f}" for k in SC)

PR = {}
for name, tags in (("P7 (v29)", SEVEN), ("P9", SEVEN + NEW), ("P8H", SEVEN + NEW[:1]), ("P8S", SEVEN + NEW[1:]),
                   ("ViT-H alone", NEW[:1]), ("Swin-L alone", NEW[1:])):
    PR[name] = oof(tags)
    log(f"probe alone, {name} ({sum(B[t]['full'].shape[1] for t in tags)} features): {alone(PR[name])}")
ref = mix(PR["P7 (v29)"], 0.35)
print("v29 reference (members + 35% P7): " + ", ".join(f"{k} {bal(ref[VIEW_OF[k]], msk(k)):.4f}" for k in SC), flush=True)


def deltas(lab, m=allm):
    return {k: bal(lab[VIEW_OF[k]], msk(k, m)) - bal(ref[VIEW_OF[k]], msk(k, m)) for k in SC}


def crit(lab, m):
    d = deltas(lab, m)
    return np.mean([d["zoom70"], d["centre_grey"], d["dark/low-contrast"]]) if d["full"] >= -0.0005 else -np.inf


def nested(cands):
    out, picks = {v: ref[v].copy() for v in FIT}, []
    for f in range(5):
        sc = {s: crit(lab, fo != f) for s, lab in cands.items()}
        s = max(sc, key=sc.get) if sc and max(sc.values()) > 0 else None
        picks.append(s)
        if s is not None:
            for v in FIT:
                out[v][fo == f] = cands[s][v][fo == f]
    return out, picks


def report(name, lab):
    d = deltas(lab)
    pf = [np.mean([bal(lab[v], fo == f) - bal(ref[v], fo == f) for v in ("zoom70", "centre_grey")]) for f in range(5)]
    pdk = [bal(lab["full"], (fo == f) & dark) - bal(ref["full"], (fo == f) & dark) for f in range(5)]
    ok = d["zoom70"] > 0 and d["centre_grey"] > 0 and d["dark/low-contrast"] > 0 and d["full"] >= -0.0005 and sum(p > 0 for p in pf) >= 3
    print(f"  {name}\n    " + ", ".join(f"{k} {d[k]:+.4f}" for k in SC) + f" | zoom / HUD by fold {' '.join(f'{p:+.4f}' for p in pf)} | "
          f"dark by fold {' '.join(f'{p:+.4f}' for p in pdk)}\n    -> {'PASS' if ok else 'fail'}", flush=True)
    return ok


C = {}
for w in WS:
    C[("A P9", w)] = mix(PR["P9"], w)
    C[("B P8H", w)] = mix(PR["P8H"], w)
    C[("C P8S", w)] = mix(PR["P8S"], w)
for w in (0.40, 0.45):
    C[("W P7", w)] = mix(PR["P7 (v29)"], w)
print(f"\n{len(C)} settings. Candidates against v29:")
for r in ("A P9", "B P8H", "C P8S", "W P7"):
    lab, picks = nested({k: v for k, v in C.items() if k[0] == r})
    report(f"{r}, w nested; picks {[p[1] if p else None for p in picks]}", lab)
labN, picksN = nested(C)
okN = report(f"N nested over all {len(C)} settings; picks {picksN}", labN)
best = max(C, key=lambda k: crit(C[k], allm))
print(f"\n-> N {'PASSES' if okN else 'fails'}; the setting with the best criterion on all five folds: {best} "
      f"(in-sample, for the pipeline only if N passes)")
print("\nIn-sample grid against v29 (information only):")
for k, lab in C.items():
    d = deltas(lab)
    print(f"  {k}: " + ", ".join(f"{q} {d[q]:+.4f}" for q in SC))
log("done")
