"""Round 39, Task 4: a brightness / contrast gate between two member weightings (training images only).

    python gu_r39_t4_gated.py

User's idea (2026-11-18 ~01:20): measure each image first and use the members that are good on that kind of image. Motivation: the
Colab member dinov2L_hud70 (DINOv2-L, HUD dropout 0.70) lifts zoom70 / HUD hidden in 5/5 folds at every weight, but on the real dark /
low-contrast frames its effect swings by fold and is negative at high weights (gt_r39_t4_dinov2L_eval.py). Gate: an image is "dark"
if its brightness (mean grey level, 0.299 R + 0.587 G + 0.114 B, on the native 230 x 120 frame) or its contrast (SD of the grey
levels) is at or below the q-quantile of the training frames; measured on the image the model sees (each view: full, 70% zoom,
HUD-hidden), so at test time it is a per-image computation like TTA. Thresholds per held-out fold from the other four folds' full
frames (label-free); for the pipeline, from all training frames.
Ensemble (v29 otherwise: v23 member weights, 0.65 x members + 0.35 x the seven-backbone MLP probe, v17 rule, balanced class mix):
dark-gated images use member weights W_dark, the others W_bright.
Settings (fixed before the run): q in {0.10, 0.20} x W_dark in {v29 (no new member), A2 (new member at weight 2)} x W_bright in
{A4, A8 (new member at weight 4 / 8, dinov2_v2 stays at 8), R8 (new member in place of dinov2_v2)}: 12 settings.
Caution: at q = 0.10 the gate on the full view nearly coincides with the evaluation's dark subset (lowest decile of contrast or
brightness), so the dark-frame delta there is mostly W_dark's own; the zoom70 / HUD-hidden deltas are not affected by that.
Selection N (fixed before the run; corrects the mismatch found in gt_r39): per held-out fold, among the settings that meet the rule on
the other four folds (zoom70, HUD hidden, dark all > 0, full >= -0.0005), the one with the best mean of the three deltas; none: the
fold keeps v29.
Acceptance (user's rule against v29, on N): zoom70, HUD hidden and the real dark / low-contrast frames all > 0, full >= -0.0005,
zoom70 / HUD-hidden mean better in >= 3/5 folds. If N passes, the pipeline setting is the one meeting the rule in-sample with the best
criterion on all five folds. Ungated A2 / A4 / A8 / R8 are printed for reference.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402

NEW = "dinov2L_hud70"
V20 = ["cnxt", "deit3", "cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud", "dinov2_v2"]
HUDM = ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
W23 = {t: (8.0 if t == "dinov2_v2" else 3.0 if t in HUDM else 1.0) for t in V20}
WMAP = {"v29": W23, "A2": {**W23, NEW: 2.0}, "A4": {**W23, NEW: 4.0}, "A8": {**W23, NEW: 8.0},
        "R8": {**{t: w for t, w in W23.items() if t != "dinov2_v2"}, NEW: 8.0}}
VIEWS = ["full", "zoom70", "centre_grey"]
V19, V30 = Path("experiments/claude/outputs/r19_views"), Path("experiments/claude/outputs/r30_views")
PW = 0.35

train, _ = task4.load_frames()
y = train.label.values
folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
fo = np.zeros(len(y), int)
for f, va in enumerate(folds):
    fo[va] = f

# ---- per-view brightness / contrast of the training images (native frame; the view applied as in em_r30_t4_hardviews.py)
cache = Path("experiments/claude/outputs/r39/t4_view_stats.npz")
if cache.exists():
    ST = dict(np.load(cache))
else:
    def zoom(x, f):
        th = torch.zeros(x.shape[0], 2, 3, device=x.device)
        th[:, 0, 0], th[:, 1, 1] = f, f
        return F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), mode="bilinear", padding_mode="border", align_corners=False)

    def centre_grey(x):
        H, W_ = x.shape[2:]
        m = torch.zeros(1, 1, H, W_, device=x.device)
        m[..., H // 5:4 * H // 5, W_ // 5:4 * W_ // 5] = 1
        return x * m + 0.5 * (1 - m)

    FN = {"full": lambda x: x, "zoom70": lambda x: zoom(x, 0.7), "centre_grey": centre_grey}
    arr = task4.load_uint8(list(train.path))
    ST = {f"{s}_{v}": np.zeros(len(y), np.float32) for s in ("brightness", "contrast") for v in VIEWS}
    lum = torch.tensor([0.299, 0.587, 0.114], device="cuda").view(1, 3, 1, 1)
    with torch.no_grad():
        for i in range(0, len(y), 1024):
            x0 = torch.from_numpy(arr[i:i + 1024]).cuda().permute(0, 3, 1, 2).float() / 255
            for v in VIEWS:
                g = (FN[v](x0) * lum).sum(1)
                ST[f"brightness_{v}"][i:i + 1024] = g.mean((1, 2)).cpu().numpy()
                ST[f"contrast_{v}"][i:i + 1024] = g.std((1, 2), unbiased=False).cpu().numpy()
    np.savez(cache, **ST)
    del arr
er = np.load("experiments/claude/outputs/r30/t4_errors.npz")
print(f"full-view stats vs t4_errors.npz: brightness max |diff| {np.abs(ST['brightness_full'] - er['brightness']).max():.5f}, "
      f"contrast max |diff| {np.abs(ST['contrast_full'] - er['contrast']).max():.5f}", flush=True)

# ---- member / probe probabilities
P = {t: {v: np.zeros((len(y), 5)) for v in VIEWS} for t in V20 + [NEW]}
for t in V20 + [NEW]:
    for f, va in enumerate(folds):
        z30 = np.load(V30 / f"{t}_fold{f}.npz")
        z19 = np.load(V19 / f"{t}_fold{f}.npz") if t in V20 else z30
        for v in VIEWS:
            P[t][v][va] = (z30 if v in z30.files else z19)[v]
PR = dict(np.load("experiments/claude/outputs/r39/p7_probe_views.npz"))
MEM = {k: {v: sum(w * P[t][v] for t, w in wm.items()) / sum(wm.values()) for v in VIEWS} for k, wm in WMAP.items()}

dark = (er["contrast"] <= np.quantile(er["contrast"], 0.1)) | (er["brightness"] <= np.quantile(er["brightness"], 0.1))
pi = np.bincount(y, minlength=5) / len(y)
wb = (0.2 / pi)[y]
bal = lambda pred, m: f1_score(y[m], pred[m], average="macro", sample_weight=wb[m])
rule = lambda p: (p * pi ** -0.75).argmax(1)
allm = np.ones(len(y), bool)
SC = ["full", "zoom70", "centre_grey", "dark/low-contrast"]
VIEW_OF = {"full": "full", "zoom70": "zoom70", "centre_grey": "centre_grey", "dark/low-contrast": "full"}
msk = lambda k, m=allm: m & dark if k == "dark/low-contrast" else m


def gate(q, rows_for_thr):
    tb = np.quantile(ST["brightness_full"][rows_for_thr], q)
    tc = np.quantile(ST["contrast_full"][rows_for_thr], q)
    return {v: (ST[f"brightness_{v}"] <= tb) | (ST[f"contrast_{v}"] <= tc) for v in VIEWS}


# gates: thresholds from the other four folds for each held-out fold (nested-clean), assembled per row
G = {}
for q in (0.10, 0.20):
    G[q] = {v: np.zeros(len(y), bool) for v in VIEWS}
    for f in range(5):
        g = gate(q, fo != f)
        for v in VIEWS:
            G[q][v][fo == f] = g[v][fo == f]
    print(f"q {q:.2f}: gated share full {G[q]['full'].mean():.3f}, zoom70 {G[q]['zoom70'].mean():.3f}, HUD hidden "
          f"{G[q]['centre_grey'].mean():.3f}; of the evaluation's dark frames gated on full {G[q]['full'][dark].mean():.3f}", flush=True)


def labels(q, wd, wbr):
    out = {}
    for v in VIEWS:
        mem = MEM[wbr][v] if q is None else np.where(G[q][v][:, None], MEM[wd][v], MEM[wbr][v])
        out[v] = rule((1 - PW) * mem + PW * PR[v])
    return out


ref = labels(None, None, "v29")
print("v29 reference: " + ", ".join(f"{k} {bal(ref[VIEW_OF[k]], msk(k)):.4f}" for k in SC), flush=True)


def deltas(lab, m=allm):
    return {k: bal(lab[VIEW_OF[k]], msk(k, m)) - bal(ref[VIEW_OF[k]], msk(k, m)) for k in SC}


def meets(d):
    return d["zoom70"] > 0 and d["centre_grey"] > 0 and d["dark/low-contrast"] > 0 and d["full"] >= -0.0005


def crit(d):
    return np.mean([d["zoom70"], d["centre_grey"], d["dark/low-contrast"]])


def report(name, lab):
    d = deltas(lab)
    pf = [np.mean([bal(lab[v], fo == f) - bal(ref[v], fo == f) for v in ("zoom70", "centre_grey")]) for f in range(5)]
    pdk = [bal(lab["full"], (fo == f) & dark) - bal(ref["full"], (fo == f) & dark) for f in range(5)]
    pfl = [bal(lab["full"], fo == f) - bal(ref["full"], fo == f) for f in range(5)]
    ok = meets(d) and sum(p > 0 for p in pf) >= 3
    print(f"  {name}\n    " + ", ".join(f"{k} {d[k]:+.4f}" for k in SC)
          + f"\n    zoom / HUD by fold {' '.join(f'{p:+.4f}' for p in pf)} | dark by fold {' '.join(f'{p:+.4f}' for p in pdk)}"
          + f" | full by fold {' '.join(f'{p:+.4f}' for p in pfl)}\n    -> {'PASS' if ok else 'fail'}", flush=True)
    return ok


print("\nUngated, for reference:")
for k in ("A2", "A4", "A8", "R8"):
    report(k, labels(None, None, k))
C = {(q, wd, wbr): labels(q, wd, wbr) for q in (0.10, 0.20) for wd in ("v29", "A2") for wbr in ("A4", "A8", "R8")}
print(f"\n{len(C)} gated settings (q, W_dark, W_bright), in-sample against v29:")
for k, lab in C.items():
    d = deltas(lab)
    print(f"  {k}: " + ", ".join(f"{q} {d[q]:+.4f}" for q in SC) + f"  {'meets the rule' if meets(d) else ''}", flush=True)
outN, picks = {v: ref[v].copy() for v in VIEWS}, []
for f in range(5):
    inner = {k: deltas(lab, fo != f) for k, lab in C.items()}
    okk = {k: crit(d) for k, d in inner.items() if meets(d)}
    s = max(okk, key=okk.get) if okk else None
    picks.append(s)
    if s is not None:
        for v in VIEWS:
            outN[v][fo == f] = C[s][v][fo == f]
print("\nCandidate:")
okN = report(f"N nested (rule-consistent); picks {picks}", outN)
ins = {k: crit(deltas(lab)) for k, lab in C.items() if meets(deltas(lab))}
best = max(ins, key=ins.get) if ins else None
print(f"\n-> N {'PASSES' if okN else 'fails'}; pipeline setting (meets the rule in-sample, best criterion): {best} (only if N passes)")
