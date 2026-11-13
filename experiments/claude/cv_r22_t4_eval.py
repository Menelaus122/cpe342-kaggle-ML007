"""Round 22: fold-0 test of a GridMask-trained ConvNeXt and a fine-tuned vision-language backbone for Task 4. Training images only.

    python cv_r22_t4_eval.py [cand ...]          (default: cnxt_grid siglip2_hud)

Candidates (scripts/train_task4.py --folds 0, the pipeline's stratified fold 0):
  cnxt_grid    ConvNeXt-T in12k, HUD dropout 0.5 + heavy occlusion (GridMask / one block hiding 65-85%) 0.2
               (reference: cnxt_hud, the same recipe without GridMask)
  siglip2_hud  SigLIP 2 ViT-B/16 (webli) at 160x320 with its own input normalisation, lr 2e-5, layer decay 0.75, HUD dropout 0.5,
               gradient checkpointing (CutMix/Mixup p = 0.8 as in every recipe)
Low-light augmentation and Mosaic were dropped before any run (the notebook's section 7.4: test confidence is lower in every
brightness / contrast bin, and round 18's gamma views gave nothing; CutMix/Mixup is already in the recipe).
Views of the 6,310 fold-0 validation images (h-flip TTA):
  full, zoom70, centre_grey             as in rounds 19-20
  real_hard                             the full frames of the darkest 10% / lowest-contrast 10% REAL images (as in cj_r18_t4_tta.py)
  dark_g35, dark_k15                    synthetic darkening (gamma 3.5, gain 0.15): no candidate trains on it, so a fair robustness check
  grid75                                a fixed 75%-hidden grid: CIRCULAR for cnxt_grid, reported but not used in the decision
Pass rule (fixed before the runs): adding the candidate to the current ensemble improves the objective (mean macro-F1 over full and
zoom70) with a paired-bootstrap 95% CI above 0, loses at most 0.0005 on full frames, and does not lower real_hard.
The held-out-cluster (content shift) comparison of the same recipes is in ch_r18_t4_shift.py (recipes cnxt_hud, cnxt_grid, siglip2_ld).
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import MODEL_DIR, PRED_DIR  # noqa: E402

CANDS = sys.argv[1:] or ["cnxt_grid", "siglip2_hud"]
ARCH = {"cnxt_hud": ("convnext_tiny.in12k_ft_in1k", (160, 320)), "cnxt_grid": ("convnext_tiny.in12k_ft_in1k", (160, 320)),
        "siglip2_hud": ("vit_base_patch16_siglip_224.v2_webli", (160, 320)), "cnxtS_hud": ("convnext_small.in12k_ft_in1k", (160, 320)),
        "cnxt": ("convnext_tiny.in12k_ft_in1k", (160, 320)), "deit3": ("deit3_small_patch16_384.fb_in22k_ft_in1k", (192, 384)),
        "cnxt22k_hud": ("convnext_tiny.fb_in22k", (160, 320)), "effv2_hud": ("tf_efficientnetv2_s.in21k_ft_in1k", (160, 320)),
        "deit3_hud": ("deit3_small_patch16_384.fb_in22k_ft_in1k", (192, 384))}
ENS = ["cnxt", "deit3", "cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
OUT = Path("experiments/claude/outputs/r22_views")
OUT.mkdir(parents=True, exist_ok=True)
OBJ = ["full", "zoom70"]


def zoom(x, f):
    th = torch.zeros(x.shape[0], 2, 3, device=x.device)
    th[:, 0, 0], th[:, 1, 1] = f, f
    return F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), mode="bilinear", padding_mode="border", align_corners=False)


def centre_grey(x):
    H, W = x.shape[2:]
    m = torch.zeros(1, 1, H, W, device=x.device)
    m[..., H // 5:4 * H // 5, W // 5:4 * W // 5] = 1
    return x * m + 0.5 * (1 - m)


def grid75(x):
    H, W = x.shape[2:]
    d = H // 3
    vis = int(round(d * 0.5))                                          # 0.5^2 = 25% visible per cell
    rr = torch.arange(H, device=x.device).view(H, 1)
    cc = torch.arange(W, device=x.device).view(1, W)
    m = (((rr + 7) % d) < vis) & (((cc + 11) % d) < vis)
    return torch.where(m, x, x.mean((2, 3), keepdim=True).expand_as(x))


VIEWS = {"full": lambda x: x, "zoom70": lambda x: zoom(x, 0.7), "centre_grey": centre_grey,
         "dark_g35": lambda x: x.clamp_min(1e-6) ** 3.5, "dark_k15": lambda x: x * 0.15, "grid75": grid75}


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


train, _ = task4.load_frames()
y = train.label.values
va = np.load(PRED_DIR / "t4_cnxt_fold0.npz")["val_idx"]
yv = y[va]
arr = task4.load_uint8(list(train.path.values[va]))
xf = arr.astype(np.float32) / 255
lum = 0.299 * xf[..., 0] + 0.587 * xf[..., 1] + 0.114 * xf[..., 2]
hard = (lum.mean((1, 2)) < np.quantile(lum.mean((1, 2)), 0.1)) | (lum.std((1, 2)) < np.quantile(lum.std((1, 2)), 0.1))
del xf, lum
tags = [t for t in dict.fromkeys(ENS + ["cnxt_hud"] + CANDS) if (MODEL_DIR / f"t4_{t}_fold0.pt").exists()]
for t in tags:
    if (OUT / f"{t}_fold0.npz").exists():
        continue
    name, size = ARCH[t]
    m = task4.create_model(name, pretrained=False, img_size=size if name.split("_")[0] in ("vit", "deit3", "eva02") else None)
    m = m.cuda().to(memory_format=torch.channels_last)
    m.load_state_dict(torch.load(MODEL_DIR / f"t4_{t}_fold0.pt", map_location="cuda"))
    m.eval()
    np.savez(OUT / f"{t}_fold0.npz", **predict(m, arr, size))
    print(f"{t} predicted", flush=True)
    del m
    torch.cuda.empty_cache()

P = {t: dict(np.load(OUT / f"{t}_fold0.npz")) for t in tags}
for t in P:
    P[t]["real_hard"] = P[t]["full"]
mf1 = lambda yy, p: f1_score(yy, p.argmax(1), average="macro")
ens = [t for t in ENS if t in P]
avg = lambda ts, v: np.mean([P[t][v] for t in ts], 0)
cols = list(VIEWS) + ["real_hard"]
score = lambda ts, v: mf1(yv[hard], avg(ts, v)[hard]) if v == "real_hard" else mf1(yv, avg(ts, v))
print(f"fold-0 validation images {len(va)}; real dark/low-contrast subset {hard.sum()}; current ensemble = {'+'.join(ens)}")
print(f"{'model':30s} | " + " | ".join(f"{v:>11s}" for v in cols) + " | objective")
rows = {t: [t] for t in ["cnxt_hud"] + CANDS if t in P}
rows["ensemble"] = ens
for c in CANDS:
    if c in P:
        rows[f"ensemble + {c}"] = ens + [c]
for r, ts in rows.items():
    s = {v: score(ts, v) for v in cols}
    print(f"{r:30s} | " + " | ".join(f"{s[v]:11.4f}" for v in cols) + f" | {np.mean([s[v] for v in OBJ]):.4f}")
rng = np.random.default_rng(0)
boot = [rng.integers(0, len(va), len(va)) for _ in range(2000)]
print("\nensemble + candidate minus ensemble (paired bootstrap 95% CI):")
for c in [c for c in CANDS if c in P]:
    a = {v: avg(ens, v).argmax(1) for v in VIEWS}
    b = {v: avg(ens + [c], v).argmax(1) for v in VIEWS}
    f = lambda pred, i: f1_score(yv[i], pred[i], average="macro")
    d_obj = [np.mean([f(b[v], i) - f(a[v], i) for v in OBJ]) for i in boot]
    pt_obj = np.mean([score(ens + [c], v) - score(ens, v) for v in OBJ])
    pt_full = score(ens + [c], "full") - score(ens, "full")
    pt_hard = score(ens + [c], "real_hard") - score(ens, "real_hard")
    ok = np.quantile(d_obj, .025) > 0 and pt_full >= -0.0005 and pt_hard >= 0
    print(f"  {c:12s} objective {pt_obj:+.4f} [{np.quantile(d_obj, .025):+.4f}, {np.quantile(d_obj, .975):+.4f}]  full {pt_full:+.4f}  "
          f"real_hard {pt_hard:+.4f}  -> {'PASS' if ok else 'fail'}")

# Label-free diagnostic on test images (fold-0 models' test probabilities; reported only, never used to decide: being confident on
# a test image is not the same as being right on it).
ent = lambda p: -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)
TP = {t: np.load(PRED_DIR / f"t4_{t}_fold0.npz")["test_probs"] for t in tags if (PRED_DIR / f"t4_{t}_fold0.npz").exists()}
if all(t in TP for t in ens):
    pe = np.mean([TP[t] for t in ens], 0)
    unsure = pe.max(1) < 0.6
    print(f"\ntest images where the current ensemble's fold-0 models are unsure (max prob < 0.6): {unsure.sum()} of {len(pe)}")
    for t in ["cnxt_hud"] + CANDS:
        if t in TP:
            agree = (TP[t].argmax(1) == pe.argmax(1))
            print(f"  {t:12s} entropy on them {ent(TP[t][unsure]).mean():.3f} (elsewhere {ent(TP[t][~unsure]).mean():.3f}); "
                  f"agrees with the ensemble on {agree[unsure].mean():.3f} of them, {agree[~unsure].mean():.3f} elsewhere")
