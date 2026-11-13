"""Task 4, round 11: frozen features from large pre-trained image models (allowed: pre-trained models in Task 4).

Why: our fine-tuned ConvNeXt/DeiT reach 0.99 out-of-fold but are much less sure of themselves on the test images
(label-free estimates of test accuracy 0.87-0.91). Frozen web-scale features (CLIP, SigLIP, DINOv2) are known to lose
less accuracy under distribution shift than fine-tuned ImageNet models (Radford 2021; Taori 2020; Kumar 2022).
This script only extracts features (pooled embedding, average of the image and its horizontal flip) for train and test;
nothing is fitted on test. Usage: python bh_t4_foundation_feats.py <timm model name> [<side>]
"""
import sys
import time

import numpy as np
import timm
import torch
import torch.nn.functional as F

import common  # noqa: F401
from src import config as C
from src import task4

name = sys.argv[1]
side = int(sys.argv[2]) if len(sys.argv) > 2 else 224
tag = name.split(".")[0].replace("vit_", "").replace("_patch", "p") + ("" if side == 224 else f"_{side}")
out = C.PRED_DIR / f"t4_feat_{tag}.npz"

kw = {"img_size": side} if "dinov2" in name else {}
model = timm.create_model(name, pretrained=True, num_classes=0, **kw).cuda().eval()
cfg = timm.data.resolve_data_config({}, model=model)
mean = torch.tensor(cfg["mean"], device="cuda").view(1, 3, 1, 1)
std = torch.tensor(cfg["std"], device="cuda").view(1, 3, 1, 1)
print(name, "input", cfg["input_size"], "-> using", side, "mean", cfg["mean"], flush=True)


@torch.no_grad()
def embed(paths, bs=128, chunk=4096):
    feats = []
    for i in range(0, len(paths), chunk):
        arr = task4.load_uint8(list(paths[i:i + chunk]))
        for j in range(0, len(arr), bs):
            x = torch.from_numpy(arr[j:j + bs]).cuda().permute(0, 3, 1, 2).float().div_(255)
            x = (F.interpolate(x, size=(side, side), mode="bicubic", align_corners=False).clamp_(0, 1) - mean) / std
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = model(x).float() + model(torch.flip(x, dims=[3])).float()
            feats.append((f / 2).cpu().numpy().astype(np.float16))
        print(f"  {i + len(arr)}/{len(paths)}", flush=True)
    return np.concatenate(feats)


train, test = task4.load_frames()
t0 = time.time()
ftr = embed(train.path.values)
fte = embed(test.path.values)
np.savez(out, train=ftr, test=fte)
print(f"saved {out.name}: train {ftr.shape}, test {fte.shape}, {(time.time() - t0) / 60:.1f} min", flush=True)
