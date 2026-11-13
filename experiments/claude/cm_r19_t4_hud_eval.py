"""Round 19: does HUD dropout make Task 4 read the scene, without losing anything on normal frames? Fold-0 validation images only.

    python cm_r19_t4_hud_eval.py [tag_a tag_b ...]      (default: cnxt cnxt_hud)

Each tag's fold-0 checkpoint (scripts/train_task4.py, same stratified folds) predicts the 6,310 fold-0 validation images under:
  full            the image as is
  bottom_grey     bottom 20% greyed (the strip the baseline relies on most)
  centre_grey     the outer ring of a 5x5 grid greyed (only the centre 3x3 cells visible)
  bottom_cut      bottom 20% cut off and the rest resized to full size (no grey area)
  zoom70          central 70% enlarged to full size (no grey area)
Grey is 0.5, not the mean colour / noise used in training, so the HUD model does not get its own fill back. Every view uses the
pipeline's h-flip TTA. Also scored: the tags averaged, and the "dual view" 0.75 * P(view) + 0.25 * P(zoom70 of that view).
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import MODEL_DIR, PRED_DIR, T4_CLASS_NAMES  # noqa: E402

ARCH = {"cnxt": ("convnext_tiny.in12k_ft_in1k", (160, 320)), "deit3": ("deit3_small_patch16_384.fb_in22k_ft_in1k", (192, 384)),
        "cnxt22k": ("convnext_tiny.fb_in22k", (160, 320)), "effv2": ("tf_efficientnetv2_s.in21k_ft_in1k", (160, 320))}
tags = sys.argv[1:] or ["cnxt", "cnxt_hud"]
arch = lambda t: ARCH[t.replace("_hud", "")]
train, _ = task4.load_frames()
y = train.label.values
va = np.load(PRED_DIR / "t4_cnxt_fold0.npz")["val_idx"]
yv = y[va]
arr = task4.load_uint8(list(train.path.values[va]))


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


def bottom_cut(x):
    H = x.shape[2]
    return F.interpolate(x[..., :int(H * 0.8), :], size=x.shape[2:], mode="bilinear", align_corners=False)


VIEWS = {"full": lambda x: x, "bottom_grey": bottom_grey, "centre_grey": centre_grey, "bottom_cut": bottom_cut, "zoom70": lambda x: zoom(x, 0.7)}


@torch.no_grad()
def predict(model, size, view, then_zoom=False):
    out = []
    for i in range(0, len(arr), 128):
        x = view(task4.to_gpu_batch(arr[i:i + 128], size))
        if then_zoom:
            x = zoom(x, 0.7)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = 0.5 * (model(task4.normalize(x).to(memory_format=torch.channels_last)).float().softmax(-1)
                       + model(task4.normalize(x.flip(-1)).to(memory_format=torch.channels_last)).float().softmax(-1))
        out.append(p.cpu())
    return torch.cat(out).numpy()


P = {}
for t in tags:
    name, size = arch(t)
    m = task4.create_model(name, pretrained=False, img_size=size if "deit3" in t else None).cuda().to(memory_format=torch.channels_last)
    m.load_state_dict(torch.load(MODEL_DIR / f"t4_{t}_fold0.pt", map_location="cuda"))
    m.eval()
    for v, f in VIEWS.items():
        P[t, v] = predict(m, size, f)
        if v != "zoom70":
            P[t, v + "+zoom"] = predict(m, size, f, then_zoom=True)
    del m
    torch.cuda.empty_cache()

pube = T4_CLASS_NAMES.index("PUBE Mobile")
score = lambda p: ((p.argmax(1) == yv).mean(), f1_score(yv, p.argmax(1), average="macro"), (p[yv == pube].argmax(1) == pube).mean())
rows = list(tags) + (["+".join(tags)] if len(tags) > 1 else [])
print(f"fold-0 validation images: {len(va)}")
print(f"{'model':22s} {'view':12s} | {'acc':>7s} {'macroF1':>8s} {'PUBE':>6s} | dual view 0.75*view + 0.25*zoom: {'acc':>7s} {'macroF1':>8s}")
for r in rows:
    parts = r.split("+") if "+" in r else [r]
    for v in VIEWS:
        p = np.mean([P[t, v] for t in parts], 0)
        a, f, pu = score(p)
        line = f"{r:22s} {v:12s} | {a:7.4f} {f:8.4f} {pu:6.3f}"
        if v != "zoom70":
            d = 0.75 * p + 0.25 * np.mean([P[t, v + "+zoom"] for t in parts], 0)
            da, df, _ = score(d)
            line += f" | {'':33s} {da:7.4f} {df:8.4f}"
        print(line)
if len(tags) >= 2:
    a, b = tags[0], tags[1]
    rng = np.random.default_rng(0)
    print(f"\npaired accuracy difference {b} - {a} (95% bootstrap CI):")
    for v in VIEWS:
        d = (P[b, v].argmax(1) == yv).astype(float) - (P[a, v].argmax(1) == yv)
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        print(f"  {v:12s} {d.mean():+.4f} [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}]")
