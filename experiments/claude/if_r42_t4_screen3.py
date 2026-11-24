"""Round 42, Task 4: third frozen-probe screen - model families not tried yet (laptop GPU, 0 CU; frozen features only, no training;
training images only).

    python if_r42_t4_screen3.py

User brief of 2026-11-20 (while DINOv2-g folds 1-4 + full refit run on Colab). Loadability checked before this script:
  aimv2_large_patch14_224.apple_pt        timm 1.0.30   AIMv2-L/14 (Apple; CLIP mean / std; 'avg' pooling; 309 M)
  vit_pe_core_large_patch14_336.fb        timm 1.0.30   Perception Encoder Core L/14 (Meta; mean / std 0.5; attention pooling; 316 M)
  vit_pe_spatial_large_patch14_448.fb     timm 1.0.30   Perception Encoder Spatial L/14 (Meta; mean / std 0.5; 'avg' pooling; 303 M)
  nvidia/C-RADIOv3-H                      Hugging Face, trust_remote_code at the pinned revision d7fd0e2b (its code was diffed against
                                          the C-RADIOv3-L code read on 2026-11-19: an older snapshot of the same repo - registry entries,
                                          import fallbacks, video mode, align_corners; no network / subprocess / exec), [0, 1] input,
                                          summary = 3 class tokens x 1280
Geometry: the 12 x 24 token grid of DINOv3-L @ 192 x 384 - 168 x 336 for the three patch-14 models, 192 x 384 for C-RADIOv3-H.
Probe exactly as hb_r40 / ia_r42 / id_r42 / ie_r42 (frozen pooled embedding, the model's own normalisation, bicubic resize, mean of
image + h-flip, training images, full frame; logistic probe C 0.03 out of fold on the members' folds; balanced class mix + v17 rule
on full frames and the real dark / low-contrast frames). References from caches: DINOv3-L @ 192 x 384 (0.9400 / 0.8831), and as
information ViT-H+ @ 144 x 272 (0.9486 / 0.8927), DINOv2-g @ 126 x 238 (0.9454 / 0.8944), C-RADIOv3-L summary @ 192 x 384 (0.9543 /
0.9027).
Reading fixed before the run: a candidate is a clear win if, against DINOv3-L @ 192 x 384, dark >= +0.005 and full >= -0.003.
C-RADIOv3-H is also set against C-RADIOv3-L (information): its weakness was on the dark frames AFTER fine-tuning (frozen, RADIO-L led
there), so a frozen probe cannot show whether H fixes it - only whether H's frozen features are stronger than L's.
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

TIMM = [("AIMv2-L @ 168x336", "aimv2_large_patch14_224.apple_pt"),
        ("PE-Core-L @ 168x336", "vit_pe_core_large_patch14_336.fb"),
        ("PE-Spatial-L @ 168x336", "vit_pe_spatial_large_patch14_448.fb")]
HW14, HW16 = (168, 336), (192, 384)
RADIO_H, RADIO_H_REV = "nvidia/C-RADIOv3-H", "d7fd0e2b0a1761f1af150582e06c41e9a99b0bf8"
OUT = Path("experiments/claude/outputs/r42")
t0 = time.time()
log = lambda m: print(f"{m} [{(time.time() - t0) / 60:.1f} min]", flush=True)


def batches(paths, hw, bs):
    for i in range(0, len(paths), 4096):
        arr = task4.load_uint8(list(paths[i:i + 4096]))
        for j in range(0, len(arr), bs):
            x0 = torch.from_numpy(arr[j:j + bs]).cuda().permute(0, 3, 1, 2).float().div_(255)
            yield F.interpolate(x0, size=hw, mode="bicubic", align_corners=False).clamp_(0, 1)


def embed_timm(name, paths, weights_dtype=None, bs=64):
    model = timm.create_model(name, pretrained=True, num_classes=0, img_size=HW14).cuda().eval()
    if weights_dtype is not None:
        model = model.to(weights_dtype)
    cfg = timm.data.resolve_model_data_config(model)
    mean = torch.tensor(cfg["mean"], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device="cuda").view(1, 3, 1, 1)
    out = []
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for x in batches(paths, HW14, bs):
            x = (x - mean) / std
            if weights_dtype is not None:
                x = x.to(weights_dtype)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = model(x).float() + model(torch.flip(x, dims=[3])).float()
            out.append((f / 2).cpu().numpy().astype(np.float16))
    peak = torch.cuda.max_memory_allocated() / 2**30
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out), peak


def embed_radio_h(paths, bs=32):
    from transformers import AutoModel
    m = AutoModel.from_pretrained(RADIO_H, trust_remote_code=True, revision=RADIO_H_REV).cuda().eval()
    ic = m.input_conditioner
    log(f"C-RADIOv3-H: {sum(p.numel() for p in m.parameters()) / 1e6:.0f} M, conditioner mean {ic.norm_mean.flatten().tolist()}, "
        f"summary_idxs {m.radio_model.summary_idxs.tolist()}")
    S, M = [], []
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for x in batches(paths, HW16, bs):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s1, f1 = m(x)
                s2, f2 = m(torch.flip(x, dims=[3]))
            S.append(((s1.float() + s2.float()) / 2).cpu().numpy().astype(np.float16))
            M.append(((f1.float().mean(1) + f2.float().mean(1)) / 2).cpu().numpy().astype(np.float16))
    peak = torch.cuda.max_memory_allocated() / 2**30
    del m
    torch.cuda.empty_cache()
    return np.concatenate(S), np.concatenate(M), peak


train, _ = task4.load_frames()
y = train.label.values.astype(np.int64)
FE = {"DINOv3-L @ 192x384": np.load(OUT / "ia_feat_DINOv3-L_192x384.npy").astype(np.float32),
      "ViT-H+ DINOv3 @ 144x272 (info)": np.load(OUT / "id_feat_vit_huge_plus_patch16_dinov3_144x272.npy").astype(np.float32),
      "DINOv2-g @ 126x238 (info)": np.load(OUT / "ie_feat_vit_giant_patch14_reg4_dinov2_126x238.npy").astype(np.float32),
      "C-RADIOv3-L summary @ 192x384 (info)": np.load(OUT / "ia_feat_C-RADIOv3-L_summary_192x384.npy").astype(np.float32)}
for lab, name in TIMM:
    p = OUT / f"if_feat_{name.split('.')[0]}_168x336.npy"
    if not p.exists():
        t1 = time.time()
        try:
            feats, peak = embed_timm(name, train.path.values)
            how = "fp32 weights + bf16 autocast"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            feats, peak = embed_timm(name, train.path.values, weights_dtype=torch.bfloat16, bs=32)
            how = "bf16 weights"
        np.save(p, feats)
        log(f"{lab}: features in {(time.time() - t1) / 60:.1f} min ({how}; peak GPU memory {peak:.1f} GB)")
    FE[lab] = np.load(p).astype(np.float32)
ps, pm = OUT / "if_feat_C-RADIOv3-H_summary_192x384.npy", OUT / "if_feat_C-RADIOv3-H_spatialmean_192x384.npy"
try:
    if not ps.exists():
        t1 = time.time()
        s, mm, peak = embed_radio_h(train.path.values)
        np.save(ps, s)
        np.save(pm, mm)
        log(f"C-RADIOv3-H: features in {(time.time() - t1) / 60:.1f} min (peak GPU memory {peak:.1f} GB)")
    FE["C-RADIOv3-H summary @ 192x384"] = np.load(ps).astype(np.float32)
    FE["C-RADIOv3-H spatial mean (info)"] = np.load(pm).astype(np.float32)
except Exception as e:  # a dependency or memory failure skips this model, as the user asked
    log(f"C-RADIOv3-H skipped: {type(e).__name__}: {str(e)[:200]}")

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
    print(f"  {lab:38s} {X.shape[1]:5d}-d: full {R[lab][0]:.4f} | dark {R[lab][1]:.4f} | plain macro-F1 {R[lab][2]:.4f}", flush=True)
print()
ref = R["DINOv3-L @ 192x384"]
for lab in [l for l, _ in TIMM] + ["C-RADIOv3-H summary @ 192x384"]:
    if lab not in R:
        print(f"{lab}: skipped")
        continue
    df, dd = R[lab][0] - ref[0], R[lab][1] - ref[1]
    ok = dd >= 0.005 and df >= -0.003
    vh = R["ViT-H+ DINOv3 @ 144x272 (info)"]
    print(f"{lab} minus DINOv3-L @ 192x384: full {df:+.4f}, dark {dd:+.4f} -> {'CLEAR WIN' if ok else 'no clear win'}"
          f" | vs ViT-H+ (info): full {R[lab][0] - vh[0]:+.4f}, dark {R[lab][1] - vh[1]:+.4f}")
if "C-RADIOv3-H summary @ 192x384" in R:
    rl, rh = R["C-RADIOv3-L summary @ 192x384 (info)"], R["C-RADIOv3-H summary @ 192x384"]
    print(f"\nInformation: C-RADIOv3-H minus C-RADIOv3-L (frozen summaries): full {rh[0] - rl[0]:+.4f}, dark {rh[1] - rl[1]:+.4f}")
log("done")
