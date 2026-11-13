"""Round 20: second test-time view for Task 4 (central 75% zoomed to full input size), out of fold, all 31,546 training images.

    python cp_r20_t4_z75_views.py            (predictions already written are reused)

For each of the six backbones' fold models and each evaluation view of co_r19_t4_views_allfolds.py (full, bottom_grey, centre_grey,
zoom70), predicts zoom75(view) on the fold's own validation images with h-flip TTA. Together with outputs/r19_views this gives
P(view) and P(zoom75 of view), so cq_r20_t4_blend.py can score "multi-crop" inference 0.7 * P(view) + 0.3 * P(zoom75) etc.
Writes outputs/r20_views/<tag>_fold<f>.npz (keys <view>_z75). No test image is used.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import MODEL_DIR, PRED_DIR  # noqa: E402

OUT = Path("experiments/claude/outputs/r20_views")
OUT.mkdir(parents=True, exist_ok=True)
ARCH = {"cnxt": ("convnext_tiny.in12k_ft_in1k", (160, 320)), "deit3": ("deit3_small_patch16_384.fb_in22k_ft_in1k", (192, 384)),
        "cnxt22k": ("convnext_tiny.fb_in22k", (160, 320)), "effv2": ("tf_efficientnetv2_s.in21k_ft_in1k", (160, 320))}
SIX = ["cnxt", "deit3", "cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud"]
train, _ = task4.load_frames()


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
            x = zoom(f(x0), 0.75)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = 0.5 * (model(task4.normalize(x).to(memory_format=torch.channels_last)).float().softmax(-1)
                           + model(task4.normalize(x.flip(-1)).to(memory_format=torch.channels_last)).float().softmax(-1))
            out[v].append(p.cpu())
    return {v + "_z75": torch.cat(o).numpy() for v, o in out.items()}


for f in range(5):
    va = np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"]
    todo = [t for t in SIX if not (OUT / f"{t}_fold{f}.npz").exists()]
    if not todo:
        continue
    arr = task4.load_uint8(list(train.path.values[va]))
    for t in todo:
        name, size = ARCH[t.replace("_hud", "")]
        m = task4.create_model(name, pretrained=False, img_size=size if "deit3" in t else None).cuda().to(memory_format=torch.channels_last)
        m.load_state_dict(torch.load(MODEL_DIR / f"t4_{t}_fold{f}.pt", map_location="cuda"))
        m.eval()
        np.savez(OUT / f"{t}_fold{f}.npz", val_idx=va, **predict(m, arr, size))
        print(f"fold {f} {t} done", flush=True)
        del m
        torch.cuda.empty_cache()
    del arr
