"""Round 19c: the six-backbone ensemble vs the round-17 pair when the HUD is missing, on all 31,550 training images (out of fold).

    python co_r19_t4_views_allfolds.py            (predictions already written are reused)

Stratified OOF cannot tell the two ensembles apart (cn_r19_t4_hud_ensemble.py: +0.0002, CI includes 0), because training and
validation frames share the same HUD. The test images the models are unsure about often have none, so this scores both ensembles
on the views used in cm_r19_t4_hud_eval.py, now for every fold: each fold model predicts only its own validation images.
  full         the image as is
  bottom_grey  bottom 20% greyed
  centre_grey  outer ring of a 5x5 grid greyed (centre 3x3 visible)
  zoom70       central 70% enlarged to full size (no grey area; the most natural HUD-free view)
Grey is 0.5, not the training fill. h-flip TTA as in the pipeline. Writes outputs/r19_views/<tag>_fold<f>.npz. No test image is used.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import MODEL_DIR, PRED_DIR, T4_CLASS_NAMES  # noqa: E402

OUT = Path("experiments/claude/outputs/r19_views")
OUT.mkdir(parents=True, exist_ok=True)
ARCH = {"cnxt": ("convnext_tiny.in12k_ft_in1k", (160, 320)), "deit3": ("deit3_small_patch16_384.fb_in22k_ft_in1k", (192, 384)),
        "cnxt22k": ("convnext_tiny.fb_in22k", (160, 320)), "effv2": ("tf_efficientnetv2_s.in21k_ft_in1k", (160, 320))}
PAIR = ["cnxt", "deit3"]
SIX = PAIR + ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud"]
train, _ = task4.load_frames()
y = train.label.values


def zoom(x, f):
    th = torch.zeros(x.shape[0], 2, 3, device=x.device)
    th[:, 0, 0], th[:, 1, 1] = f, f
    return F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), mode="bilinear", padding_mode="border", align_corners=False)


def bottom_grey(x):
    x = x.clone()
    x[..., int(x.shape[2] * 0.8):, :] = 0.5
    return x


def centre_grey(x):
    H, W = x.shape[2:]
    m = torch.zeros(1, 1, H, W, device=x.device)
    m[..., H // 5:4 * H // 5, W // 5:4 * W // 5] = 1
    return x * m + 0.5 * (1 - m)


VIEWS = {"full": lambda x: x, "bottom_grey": bottom_grey, "centre_grey": centre_grey, "zoom70": lambda x: zoom(x, 0.7)}


@torch.no_grad()
def predict(model, arr, size):
    out = {v: [] for v in VIEWS}
    for i in range(0, len(arr), 128):
        x0 = task4.to_gpu_batch(arr[i:i + 128], size)
        for v, f in VIEWS.items():
            x = f(x0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = 0.5 * (model(task4.normalize(x).to(memory_format=torch.channels_last)).float().softmax(-1)
                           + model(task4.normalize(x.flip(-1)).to(memory_format=torch.channels_last)).float().softmax(-1))
            out[v].append(p.cpu())
    return {v: torch.cat(o).numpy() for v, o in out.items()}


folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
for f, va in enumerate(folds):
    todo = [t for t in SIX if not (OUT / f"{t}_fold{f}.npz").exists()]
    if not todo:
        continue
    arr = task4.load_uint8(list(train.path.values[va]))
    for t in todo:
        assert (np.load(PRED_DIR / f"t4_{t}_fold{f}.npz")["val_idx"] == va).all(), f"{t} fold {f}: different folds"
        name, size = ARCH[t.replace("_hud", "")]
        m = task4.create_model(name, pretrained=False, img_size=size if "deit3" in t else None).cuda().to(memory_format=torch.channels_last)
        m.load_state_dict(torch.load(MODEL_DIR / f"t4_{t}_fold{f}.pt", map_location="cuda"))
        m.eval()
        np.savez(OUT / f"{t}_fold{f}.npz", val_idx=va, **predict(m, arr, size))
        print(f"fold {f} {t} done", flush=True)
        del m
        torch.cuda.empty_cache()
    del arr

P = {(t, v): np.zeros((len(y), 5)) for t in SIX for v in VIEWS}
for t in SIX:
    for f, va in enumerate(folds):
        z = np.load(OUT / f"{t}_fold{f}.npz")
        for v in VIEWS:
            P[t, v][va] = z[v]
pube = T4_CLASS_NAMES.index("PUBE Mobile")
avg = lambda ts, v: np.mean([P[t, v] for t in ts], 0)
print(f"\nout-of-fold, all {len(y)} training images (accuracy / macro-F1 / PUBE recall):")
print(f"{'model':14s}" + "".join(f" | {v:^24s}" for v in VIEWS))
for r in SIX + ["pair", "six"]:
    line = f"{r:14s}"
    for v in VIEWS:
        p = avg(PAIR if r == "pair" else SIX if r == "six" else [r], v)
        c = p.argmax(1)
        line += f" | {(c == y).mean():.4f} {f1_score(y, c, average='macro'):.4f} {(c[y == pube] == pube).mean():.3f}"
    print(line)
rng = np.random.default_rng(0)
print("\nsix - pair, paired bootstrap over rows (2000x), 95% CI; per-fold accuracy deltas:")
for v in VIEWS:
    c2, c6 = avg(PAIR, v).argmax(1), avg(SIX, v).argmax(1)
    d = (c6 == y).astype(float) - (c2 == y)
    bs_a, bs_f = [], []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        bs_a.append(d[i].mean())
        bs_f.append(f1_score(y[i], c6[i], average="macro") - f1_score(y[i], c2[i], average="macro"))
    df1 = f1_score(y, c6, average="macro") - f1_score(y, c2, average="macro")
    pf = " ".join(f"{d[va].mean():+.4f}" for va in folds)
    print(f"  {v:12s} acc {d.mean():+.4f} [{np.quantile(bs_a, .025):+.4f}, {np.quantile(bs_a, .975):+.4f}]"
          f"  macro-F1 {df1:+.4f} [{np.quantile(bs_f, .025):+.4f}, {np.quantile(bs_f, .975):+.4f}]  folds: {pf}")
