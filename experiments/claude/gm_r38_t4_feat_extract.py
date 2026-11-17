"""Round 38, Task 4, step 1a: frozen features of three more foundation backbones (GPU inference only; no training).

    python gm_r38_t4_feat_extract.py [--smoke] [--only TAG]

User brief of 2026-11-17 (after v27 = 0.86177): a "super-probe" on frozen features of large backbones from families other than
DINOv2. Each backbone stays frozen; training images are embedded under the five scoring views (full, zoom70, HUD hidden, gamma 3.5,
75% grid; applied at the native 230 x 120 frame, as gi_r37_t4_probeL.py), test images under the full frame only (inference).
Pooled embedding (num_classes = 0), mean of the image and its h-flip, resized to a square input as the existing probe
(scripts/train_task4_probe.embed). Caches in the existing probe's layout, so scripts/train_task4_probe.py can read them:
  outputs/preds/t4_feat_views_<tag>.npz   five training views
  outputs/preds/t4_feat_<tag>.npz         train (full), test (full), test_file_name
Input sizes: the frames are 230 x 120, so inputs of 384 / 448 only upsample (no new pixels) at 3-7x the cost of 224 on the laptop
GPU (DINOv2-L at 224 took 53 min). SigLIP-SO400M uses its native 224 checkpoint (SigLIP 2), EVA-02-L the 448 fine-tuned weights
with the position embedding resampled to 224, ConvNeXt-L CLIP its native 320.
Round 39 step 2 adds OpenCLIP ViT-H/14 (LAION-2B) and Swin-L (IN-22k, ft IN-1k), both at their native 224.
"""
import argparse
import sys
import time

import numpy as np
import torch

import common  # noqa: F401  (chdir to the repo root)

sys.path.insert(0, "scripts")
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402
import train_task4_probe as P0  # noqa: E402  (embed, VIEWS)

BACKBONES = [  # (timm name, square side, cache tag, create_model kwargs)
    ("vit_so400m_patch14_siglip_224.v2_webli", 224, "so400m14_siglip2_224", {}),
    ("eva02_large_patch14_448.mim_m38m_ft_in22k_in1k", 224, "eva02L14_m38m_224", {"img_size": 224}),
    ("convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_320", 320, "cnxtL_clip_320", {}),
    # round 39 step 2 (user brief after v28): two more families
    ("vit_huge_patch14_clip_224.laion2b", 224, "vitH14_clip_laion2b_224", {}),
    ("swin_large_patch4_window7_224.ms_in22k_ft_in1k", 224, "swinL_in22k_224", {}),
]
ALL5 = ["full", "zoom70", "centre_grey", "dark_g35", "grid75"]


def grid75(x):
    H, W = x.shape[2:]
    d = H // 3
    vis = int(round(d * 0.5))
    rr = torch.arange(H, device=x.device).view(H, 1)
    cc = torch.arange(W, device=x.device).view(1, W)
    m = (((rr + 7) % d) < vis) & (((cc + 11) % d) < vis)
    return torch.where(m, x, x.mean((2, 3), keepdim=True).expand_as(x))


P0.VIEWS.update({"dark_g35": lambda x: x.clamp_min(1e-6) ** 3.5, "grid75": grid75})


def load(name, kw):
    import timm
    model = timm.create_model(name, pretrained=True, num_classes=0, **kw).cuda().eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(cfg["mean"], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device="cuda").view(1, 3, 1, 1)
    return model, mean, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="load each backbone, embed 512 training images, report dim / speed / VRAM")
    ap.add_argument("--only", default=None, help="one cache tag")
    args = ap.parse_args()
    train, test = task4.load_frames()
    for name, side, tag, kw in BACKBONES:
        if args.only and tag != args.only:
            continue
        f_full, f_views = PRED_DIR / f"t4_feat_{tag}.npz", PRED_DIR / f"t4_feat_views_{tag}.npz"
        if not args.smoke and f_full.exists() and f_views.exists():
            print(f"{tag}: cached", flush=True)
            continue
        t0 = time.time()
        model, mean, std = load(name, kw)
        n_par = sum(p.numel() for p in model.parameters()) / 1e6
        if args.smoke:
            torch.cuda.reset_peak_memory_stats()
            P0.embed(model, mean, std, side, train.path.values[:128], ALL5)          # warm-up
            t1 = time.time()
            z = P0.embed(model, mean, std, side, train.path.values[:512], ALL5)
            rate = 512 * len(ALL5) / (time.time() - t1)
            est = (len(train) * len(ALL5) + len(test)) / rate / 60
            print(f"{tag}: {name}, {n_par:.0f}M parameters, input {side}, feature dim {z['full'].shape[1]}, {rate:.0f} images x views / s "
                  f"(h-flip included), peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, finite {np.isfinite(z['full']).all()}; "
                  f"estimated {est:.0f} min for train (5 views) + test", flush=True)
        else:
            if not f_views.exists():
                np.savez(f_views, **P0.embed(model, mean, std, side, train.path.values, ALL5))
                print(f"{tag}: training features (5 views) in {(time.time() - t0) / 60:.1f} min", flush=True)
            if not f_full.exists():
                np.savez(f_full, train=np.load(f_views)["full"], test=P0.embed(model, mean, std, side, test.path.values, ["full"])["full"],
                         test_file_name=np.array(test.file_name.tolist(), dtype="U"))
                print(f"{tag}: test features in {(time.time() - t0) / 60:.1f} min (total)", flush=True)
        del model
        torch.cuda.empty_cache()
    print("done", flush=True)


if __name__ == "__main__":
    main()
