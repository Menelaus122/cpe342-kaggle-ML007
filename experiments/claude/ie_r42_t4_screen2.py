"""Round 42, Task 4: second frozen-probe screen for the remaining Colab compute (laptop GPU, 0 CU; frozen features only, no
training; training images only).

    python ie_r42_t4_screen2.py

User brief of 2026-11-20 (240 CU left, about 18-20 A100 hours; ViT-H+ fold 0 training on Colab): is DINOv2-g/14 worth being the
second candidate, against a higher-resolution DINOv3-L? Candidates (timm 1.0.30; both ImageNet mean / std in their pretrained config):
  vit_giant_patch14_reg4_dinov2.lvd142m  @ 126 x 238  (DINOv2-g/14 + registers, 1,135 M, 40 blocks, 1536-d class token; the 9 x 17
                                                        grid of v30's DINOv2-L and v31's DINOv3-L @ 144 x 272)
  vit_large_patch16_dinov3.lvd1689m      @ 224 x 448  (DINOv3-L/16, 14 x 28 = 392 tokens; its 'avg' pooling)
Probe exactly as hb_r40 / ia_r42 / id_r42 (pooled embedding of the frozen backbone, the model's own mean / std, bicubic resize, mean of
image + h-flip, training images, full frame; logistic probe C 0.03 out of fold on the members' folds; balanced class mix + v17 rule on
full frames and the real dark / low-contrast frames). DINOv2-g runs with fp32 weights + bf16 autocast, falling back to bf16 weights,
and is skipped if the 8 GB GPU cannot run it. References from caches: DINOv3-L @ 144 x 272 (hb_r40: 0.9363 / 0.8716), @ 192 x 384
(ia_r42: 0.9400 / 0.8831), DINOv2-L @ 126 x 238 (hb_r40: 0.9050 / 0.8318), ViT-H+ DINOv3 @ 144 x 272 (id_r42: 0.9486 / 0.8927).
Reading fixed before the run (as id_r42): a candidate is a clear win if, against DINOv3-L at its nearest geometry (144 x 272 for
DINOv2-g, the same 9 x 17 grid; 192 x 384 for DINOv3-L @ 224 x 448), dark >= +0.005 and full >= -0.003. Between two clear wins, the
higher dark score is the better second candidate. Known caveat: a frozen lead is a weak predictor after fine-tuning (C-RADIOv3-L).
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)

sys.path.insert(0, "scripts")
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402
import train_task4_probe as P0  # noqa: E402
import timm  # noqa: E402

CANDS = [("DINOv2-g @ 126x238", "vit_giant_patch14_reg4_dinov2.lvd142m", (126, 238), "DINOv3-L @ 144x272"),
         ("DINOv3-L @ 224x448", "vit_large_patch16_dinov3.lvd1689m", (224, 448), "DINOv3-L @ 192x384")]
OUT = Path("experiments/claude/outputs/r42")
BS = 64
t0 = time.time()
log = lambda m: print(f"{m} [{(time.time() - t0) / 60:.1f} min]", flush=True)


def embed(name, hw, paths, weights_dtype=None):
    model = timm.create_model(name, pretrained=True, num_classes=0, img_size=hw).cuda().eval()
    if weights_dtype is not None:
        model = model.to(weights_dtype)
    cfg = timm.data.resolve_model_data_config(model)
    mean = torch.tensor(cfg["mean"], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device="cuda").view(1, 3, 1, 1)
    out = []
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for i in range(0, len(paths), 4096):
            arr = task4.load_uint8(list(paths[i:i + 4096]))
            for j in range(0, len(arr), BS):
                x0 = torch.from_numpy(arr[j:j + BS]).cuda().permute(0, 3, 1, 2).float().div_(255)
                x = (F.interpolate(x0, size=hw, mode="bicubic", align_corners=False).clamp_(0, 1) - mean) / std
                if weights_dtype is not None:
                    x = x.to(weights_dtype)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    f = model(x).float() + model(torch.flip(x, dims=[3])).float()
                out.append((f / 2).cpu().numpy().astype(np.float16))
    peak = torch.cuda.max_memory_allocated() / 2**30
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out), peak


train, _ = task4.load_frames()
y = train.label.values.astype(np.int64)
FE = {"DINOv3-L @ 144x272": np.load("experiments/claude/outputs/r40/hb_feat_DINOv3-L_144x272.npy").astype(np.float32),
      "DINOv3-L @ 192x384": np.load(OUT / "ia_feat_DINOv3-L_192x384.npy").astype(np.float32),
      "DINOv2-L @ 126x238 (info)": np.load("experiments/claude/outputs/r40/hb_feat_DINOv2-L_126x238.npy").astype(np.float32),
      "ViT-H+ DINOv3 @ 144x272 (info)": np.load(OUT / "id_feat_vit_huge_plus_patch16_dinov3_144x272.npy").astype(np.float32)}
for lab, name, hw, _ in CANDS:
    p = OUT / f"ie_feat_{name.split('.')[0]}_{hw[0]}x{hw[1]}.npy"
    if not p.exists():
        t1 = time.time()
        try:
            feats, peak = embed(name, hw, train.path.values)
            how = "fp32 weights + bf16 autocast"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log(f"{lab}: out of GPU memory with fp32 weights -> bf16 weights")
            try:
                feats, peak = embed(name, hw, train.path.values, weights_dtype=torch.bfloat16)
                how = "bf16 weights"
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                log(f"{lab}: does not fit the 8 GB GPU even with bf16 weights -> skipped")
                continue
        np.save(p, feats)
        log(f"{lab}: features in {(time.time() - t1) / 60:.1f} min ({how}; peak GPU memory {peak:.1f} GB)")
    FE[lab] = np.load(p).astype(np.float32)

folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
fo = np.zeros(len(y), int)
for f, va in enumerate(folds):
    fo[va] = f
st = np.load("experiments/claude/outputs/r30/t4_errors.npz")
dark = (st["contrast"] <= np.quantile(st["contrast"], 0.1)) | (st["brightness"] <= np.quantile(st["brightness"], 0.1))
pi = np.bincount(y, minlength=5) / len(y)
wb = (0.2 / pi)[y]
bal = lambda pred, m: f1_score(y[m], pred[m], average="macro", sample_weight=wb[m])
rule = lambda p: (p * pi ** -0.75).argmax(1)
allm = np.ones(len(y), bool)
torch.set_num_threads(8)


def oof(X):
    out = np.zeros((len(y), 5))
    for f, va in enumerate(folds):
        tr = np.where(fo != f)[0]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        W, b = P0.fit_probe((X[tr] - mu) / sd, y[tr])
        out[va] = (torch.from_numpy((X[va] - mu) / sd) @ W + b).softmax(1).numpy()
    return out


print("\nfrozen linear probe, out of fold (balanced mix, v17 rule; full frames | real dark / low-contrast frames):", flush=True)
R = {}
for lab, X in FE.items():
    p = oof(X)
    R[lab] = (bal(rule(p), allm), bal(rule(p), dark), f1_score(y, p.argmax(1), average="macro"))
    print(f"  {lab:34s} {X.shape[1]:5d}-d: full {R[lab][0]:.4f} | dark {R[lab][1]:.4f} | plain macro-F1 {R[lab][2]:.4f}", flush=True)
print()
wins = {}
for lab, name, hw, ref in CANDS:
    if lab not in R:
        print(f"{lab}: skipped (GPU memory)")
        continue
    df, dd = R[lab][0] - R[ref][0], R[lab][1] - R[ref][1]
    ok = dd >= 0.005 and df >= -0.003
    if ok:
        wins[lab] = R[lab][1]
    vh = R["ViT-H+ DINOv3 @ 144x272 (info)"]
    print(f"{lab} minus {ref}: full {df:+.4f}, dark {dd:+.4f} -> {'CLEAR WIN' if ok else 'no clear win'}"
          f" | vs ViT-H+ (info): full {R[lab][0] - vh[0]:+.4f}, dark {R[lab][1] - vh[1]:+.4f}")
print(f"\n-> second candidate by the rule: {max(wins, key=wins.get) if wins else 'none'}")
log("done")
