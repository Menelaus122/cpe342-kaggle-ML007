"""Round 30, Task 4: what are the v20 ensemble's remaining errors on held-out TRAINING images? (CPU; no test image is read)

    python ek_r30_t4_errors.py

v20 = weighted mean of the eight fine-tuned members (DINOv2 x3) -> 0.9 * mean + 0.1 * frozen-feature probe -> argmax p * pi^-0.75.
Out-of-fold probabilities of every member under the full, 70%-zoom and HUD-hidden views (outputs/r19_views, written in rounds
19-28) and the probe's views (outputs/r27/t4_probe_views.npz).
  1. errors per view: count, confused class pairs, per-class recall, overlap of the errors between views;
  2. what the misclassified full frames look like, measured on the training images: brightness, contrast, saturation, edge density,
     black rotation corners, high-frequency noise, share of near-uniform pixels; each as an AUC for "this frame is an error" and as
     the error rate in its lowest / highest decile; plus the members' agreement and the ensemble's confidence;
  3. a contact sheet of 40 misclassified full frames (true -> predicted) for visual inspection:
     outputs/r30/t4_errors_sheet.png.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import PRED_DIR, T4_CLASS_NAMES  # noqa: E402

t0 = time.time()
V19 = Path("experiments/claude/outputs/r19_views")
OUT = Path("experiments/claude/outputs/r30")
OUT.mkdir(parents=True, exist_ok=True)
MEM = ["cnxt", "deit3", "cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud", "dinov2_v2"]
W = {"dinov2_v2": 3.0}
VIEWS = ["full", "zoom70", "centre_grey"]
train, _ = task4.load_frames()
y = train.label.values
folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
P = {t: {v: np.zeros((len(y), 5)) for v in VIEWS} for t in MEM}
for t in MEM:
    for f, va in enumerate(folds):
        z = np.load(V19 / f"{t}_fold{f}.npz")
        for v in VIEWS:
            P[t][v][va] = z[v]
zp = np.load("experiments/claude/outputs/r27/t4_probe_views.npz")
pi = np.bincount(y, minlength=5) / len(y)
ens = {v: 0.9 * sum(W.get(t, 1.0) * P[t][v] for t in MEM) / sum(W.get(t, 1.0) for t in MEM) + 0.1 * zp[f"probe-aug_{v}"] for v in VIEWS}
lab = {v: (ens[v] * pi ** -0.75).argmax(1) for v in VIEWS}
short = ["7-11K", "FiveN", "FreeF", "PUBE", "Roblock"]

print(f"{len(y)} training images out of fold; class shares " + ", ".join(f"{s} {p:.3f}" for s, p in zip(short, pi)))
for v in VIEWS:
    e = lab[v] != y
    pairs = pd.Series([f"{short[a]} -> {short[b]}" for a, b in zip(y[e], lab[v][e])]).value_counts()
    rec = [np.mean(lab[v][y == c] == c) for c in range(5)]
    print(f"\n{v}: errors {e.sum()} ({e.mean():.2%}); recall " + ", ".join(f"{s} {r:.4f}" for s, r in zip(short, rec)))
    print("  most confused (true -> predicted): " + ", ".join(f"{k} {n}" for k, n in pairs.head(8).items()))
ef, ez, eh = (lab[v] != y for v in VIEWS)
print(f"\nerror overlap: full errors also wrong on zoom70 {np.mean(ez[ef]):.2f}, on HUD hidden {np.mean(eh[ef]):.2f}; "
      f"wrong on all three views {np.sum(ef & ez & eh)}; zoom70-only errors {np.sum(ez & ~ef)}, HUD-hidden-only errors {np.sum(eh & ~ef)}")

# 2. image statistics of the full frames (training images)
arr = task4.load_uint8(list(train.path))
print(f"\ndecoded {arr.shape} [{time.time() - t0:.0f}s]")
stats = {k: np.zeros(len(y), np.float32) for k in ["brightness", "contrast", "saturation", "edge density", "black corners",
                                                     "high-freq noise", "near-uniform share"]}
for i in range(0, len(y), 2048):
    x = arr[i:i + 2048].astype(np.float32) / 255
    g = x @ np.array([0.299, 0.587, 0.114], np.float32)
    stats["brightness"][i:i + 2048] = g.mean((1, 2))
    stats["contrast"][i:i + 2048] = g.std((1, 2))
    mx, mn = x.max(3), x.min(3)
    stats["saturation"][i:i + 2048] = ((mx - mn) / np.maximum(mx, 1e-3)).mean((1, 2))
    gx, gy = np.abs(np.diff(g, axis=2))[:, :-1, :], np.abs(np.diff(g, axis=1))[:, :, :-1]
    mag = gx + gy
    stats["edge density"][i:i + 2048] = (mag > 0.1).mean((1, 2))
    c = np.concatenate([g[:, :12, :12].reshape(len(g), -1), g[:, :12, -12:].reshape(len(g), -1),
                        g[:, -12:, :12].reshape(len(g), -1), g[:, -12:, -12:].reshape(len(g), -1)], 1)
    stats["black corners"][i:i + 2048] = (c < 0.03).mean(1)
    blur = (g[:, 1:-1, 1:-1] * 4 - g[:, :-2, 1:-1] - g[:, 2:, 1:-1] - g[:, 1:-1, :-2] - g[:, 1:-1, 2:]) / 4
    stats["high-freq noise"][i:i + 2048] = np.median(np.abs(blur).reshape(len(g), -1), 1)
    stats["near-uniform share"][i:i + 2048] = (mag < 0.01).mean((1, 2))
del arr
conf = ens["full"].max(1)
mem_lab = np.stack([P[t]["full"].argmax(1) for t in MEM])
agree = (mem_lab == lab["full"][None, :]).mean(0)
stats["ensemble confidence"] = conf
stats["member agreement"] = agree
print("full-frame errors vs image statistics (AUC for 'error'; error rate in the lowest / highest decile; overall "
      f"{ef.mean():.4f}):")
for k, v in stats.items():
    q1, q9 = np.quantile(v, [0.1, 0.9])
    print(f"  {k:20s} AUC {roc_auc_score(ef, v):.3f}  error rate lowest decile {ef[v <= q1].mean():.4f}, highest {ef[v >= q9].mean():.4f}"
          f"  (median: errors {np.median(v[ef]):.3f}, correct {np.median(v[~ef]):.3f})")
for c in range(5):
    m = y == c
    print(f"  {short[c]:8s} errors {ef[m].sum():4d}: median brightness {np.median(stats['brightness'][m & ef]):.3f} vs "
          f"{np.median(stats['brightness'][m & ~ef]):.3f}, edge density {np.median(stats['edge density'][m & ef]):.3f} vs "
          f"{np.median(stats['edge density'][m & ~ef]):.3f}, near-uniform {np.median(stats['near-uniform share'][m & ef]):.3f} vs "
          f"{np.median(stats['near-uniform share'][m & ~ef]):.3f}")
np.savez(OUT / "t4_errors.npz", err_full=ef, err_zoom=ez, err_hud=eh, pred_full=lab["full"], **{k.replace(" ", "_"): v for k, v in stats.items()})

# 3. contact sheet of misclassified full frames (training images)
idx = np.where(ef)[0]
rng = np.random.default_rng(0)
pick = rng.choice(idx, min(40, len(idx)), replace=False)
sheet = Image.new("RGB", (8 * 236, 5 * 140), "white")
draw = ImageDraw.Draw(sheet)
for k, i in enumerate(pick):
    im = Image.open(train.path.values[i]).convert("RGB").resize((230, 120))
    x0, y0 = (k % 8) * 236, (k // 8) * 140
    sheet.paste(im, (x0, y0))
    draw.text((x0 + 2, y0 + 121), f"{short[y[i]]} -> {short[lab['full'][i]]} ({conf[i]:.2f})", fill="black")
sheet.save(OUT / "t4_errors_sheet.png")
print(f"\ncontact sheet: {OUT / 't4_errors_sheet.png'} [{time.time() - t0:.0f}s]")
