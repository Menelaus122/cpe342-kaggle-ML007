"""Round 27, Task 4: frozen foundation features of the TRAINING images under the two HUD-poor views (70% zoom, HUD hidden).

    python dq_r27_t4_probe_views.py

Round 11 extracted frozen CLIP / DINOv2 / SigLIP 2 features of full frames (outputs/preds/t4_feat_*.npz). To ask whether a probe on
those features helps where the fine-tuned ensemble is unsure, the same features are needed for the views that imitate the test
frames (dr_r27_t4_probe_fusion.py). Training images only; the views are the ones of rounds 19-26 (cv_r22_t4_eval.py).
Writes experiments/claude/outputs/r27/t4_featviews_<tag>.npz with arrays zoom70 and centre_grey (n_train x 768, float16).
"""
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F

import common  # noqa: F401
from src import task4

OUT = Path("experiments/claude/outputs/r27")
OUT.mkdir(parents=True, exist_ok=True)
MODELS = [("vit_base_patch16_clip_224.openai", 224, "basep16_clip_224"),
          ("vit_base_patch14_reg4_dinov2.lvd142m", 224, "basep14_reg4_dinov2"),
          ("vit_base_patch16_siglip_256.v2_webli", 256, "basep16_siglip_256_256")]


def zoom(x, f):
    th = torch.zeros(x.shape[0], 2, 3, device=x.device)
    th[:, 0, 0], th[:, 1, 1] = f, f
    return F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), mode="bilinear", padding_mode="border", align_corners=False)


def centre_grey(x):
    H, W = x.shape[2:]
    m = torch.zeros(1, 1, H, W, device=x.device)
    m[..., H // 5:4 * H // 5, W // 5:4 * W // 5] = 1
    return x * m + 0.5 * (1 - m)


VIEWS = {"zoom70": lambda x: zoom(x, 0.7), "centre_grey": centre_grey}
train, _ = task4.load_frames()
paths = train.path.values
for name, side, tag in MODELS:
    out = OUT / f"t4_featviews_{tag}.npz"
    if out.exists():
        continue
    t0 = time.time()
    kw = {"img_size": side} if "dinov2" in name else {}
    model = timm.create_model(name, pretrained=True, num_classes=0, **kw).cuda().eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(cfg["mean"], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device="cuda").view(1, 3, 1, 1)
    feats = {v: [] for v in VIEWS}
    with torch.no_grad():
        for i in range(0, len(paths), 4096):
            arr = task4.load_uint8(list(paths[i:i + 4096]))
            for j in range(0, len(arr), 128):
                x0 = torch.from_numpy(arr[j:j + 128]).cuda().permute(0, 3, 1, 2).float().div_(255)
                for v, fn in VIEWS.items():
                    x = (F.interpolate(fn(x0), size=(side, side), mode="bicubic", align_corners=False).clamp_(0, 1) - mean) / std
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        f = model(x).float() + model(torch.flip(x, dims=[3])).float()
                    feats[v].append((f / 2).cpu().numpy().astype(np.float16))
            print(f"  {tag} {min(i + 4096, len(paths))}/{len(paths)}", flush=True)
    np.savez(out, **{v: np.concatenate(f) for v, f in feats.items()})
    print(f"saved {out.name} in {(time.time() - t0) / 60:.1f} min", flush=True)
    del model
    torch.cuda.empty_cache()
