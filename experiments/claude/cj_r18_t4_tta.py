"""Round 18, Task 4: test-time views and photometric "restoration", scored on validation images only.

    python cj_r18_t4_tta.py

The pipeline's fold-0 checkpoints (ConvNeXt-T, DeiT-III-S) predict their own stratified fold-0 validation images (6,310, never trained
on) under several views. Each view is scored alone and in combinations, against the pipeline's view set (original + h-flip).
Also scored on the hardest 20% of those images (darkest, lowest-contrast) to see whether restoration helps where the brief expects it.
No test image is used.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import MODEL_DIR, PRED_DIR  # noqa: E402

DEV = "cuda"
train, _ = task4.load_frames()
y = train.label.values
BACKBONES = {"cnxt": ("convnext_tiny.in12k_ft_in1k", (160, 320)), "deit3": ("deit3_small_patch16_384.fb_in22k_ft_in1k", (192, 384))}


def affine(x, angle=0.0, scale=1.0):
    B, _, H, W = x.shape
    a = math.radians(angle)
    M = torch.tensor([[math.cos(a) / scale, -math.sin(a) / scale * H / W, 0.0], [math.sin(a) / scale * W / H, math.cos(a) / scale, 0.0]],
                     device=x.device).expand(B, 2, 3)
    return F.grid_sample(x, F.affine_grid(M, x.shape, align_corners=False), padding_mode="zeros", align_corners=False)


def autocontrast(x):
    lo = torch.quantile(x.flatten(2), 0.01, dim=2)[..., None, None]
    hi = torch.quantile(x.flatten(2), 0.99, dim=2)[..., None, None]
    return ((x - lo) / (hi - lo).clamp_min(1e-3)).clamp(0, 1)


def gamma_dark(x, thr=0.3, g=0.6):
    lum = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).mean((1, 2))
    return torch.where((lum < thr)[:, None, None, None], x.clamp_min(1e-6) ** g, x)


VIEWS = {
    "orig": lambda x: x,
    "flip": lambda x: x.flip(-1),
    "autocontrast": autocontrast,
    "gamma_dark": gamma_dark,
    "rot+10": lambda x: affine(x, 10),
    "rot-10": lambda x: affine(x, -10),
    "zoom_out_0.85": lambda x: affine(x, 0, 0.85),
    "zoom_in_1.15": lambda x: affine(x, 0, 1.15),
}


@torch.no_grad()
def view_probs(model, arr, size):
    out = {k: [] for k in VIEWS}
    for i in range(0, len(arr), 128):
        x = task4.to_gpu_batch(arr[i:i + 128], size, DEV)
        for k, f in VIEWS.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out[k].append(model(task4.normalize(f(x)).to(memory_format=torch.channels_last)).float().softmax(-1).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


z = np.load(PRED_DIR / "t4_cnxt_fold0.npz")
va = z["val_idx"]
arr = task4.load_uint8(list(train.path.values[va]))
yv = y[va]
xf = arr.astype(np.float32) / 255
lum = (0.299 * xf[..., 0] + 0.587 * xf[..., 1] + 0.114 * xf[..., 2])
hard = (lum.mean((1, 2)) < np.quantile(lum.mean((1, 2)), 0.1)) | (lum.std((1, 2)) < np.quantile(lum.std((1, 2)), 0.1))
print(f"fold-0 validation images: {len(va)}; dark or low-contrast subset: {hard.sum()}")

P = {}
for tag, (name, size) in BACKBONES.items():
    m = task4.create_model(name, pretrained=False, img_size=size if tag == "deit3" else None).to(DEV).to(memory_format=torch.channels_last)
    m.load_state_dict(torch.load(MODEL_DIR / f"t4_{tag}_fold0.pt", map_location=DEV))
    m.eval()
    P[tag] = view_probs(m, arr, size)
    del m
    torch.cuda.empty_cache()

score = lambda p, mask=None: (f1_score(yv if mask is None else yv[mask], (p if mask is None else p[mask]).argmax(1), average="macro"),
                              ((p.argmax(1) == yv) if mask is None else (p[mask].argmax(1) == yv[mask])).mean())
combos = {"pipeline: orig+flip": ["orig", "flip"], "orig only": ["orig"]}
combos.update({f"orig+flip+{k}": ["orig", "flip", k] for k in VIEWS if k not in ("orig", "flip")})
combos["orig+flip+autocontrast+gamma"] = ["orig", "flip", "autocontrast", "gamma_dark"]
combos["orig+flip+rot+-10"] = ["orig", "flip", "rot+10", "rot-10"]
combos["all 8 views"] = list(VIEWS)
print(f"{'views':34s} | {'cnxt F1':>8s} {'acc':>7s} | {'deit3 F1':>8s} | {'ens F1':>8s} {'acc':>7s} | {'ens F1 hard':>11s} {'acc hard':>8s}")
for nm, vs in combos.items():
    pc = np.mean([P["cnxt"][v] for v in vs], 0)
    pd_ = np.mean([P["deit3"][v] for v in vs], 0)
    pe = (pc + pd_) / 2
    fc, ac = score(pc)
    fd, _ = score(pd_)
    fe, ae = score(pe)
    fh, ah = score(pe, hard)
    print(f"{nm:34s} | {fc:8.4f} {ac:7.4f} | {fd:8.4f} | {fe:8.4f} {ae:7.4f} | {fh:11.4f} {ah:8.4f}")
for v in VIEWS:
    print(f"  single view {v:14s}: ensemble acc {score((P['cnxt'][v] + P['deit3'][v]) / 2)[1]:.4f}")
