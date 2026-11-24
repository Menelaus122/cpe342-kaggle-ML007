"""Round 42, Task 4: frozen NVIDIA C-RADIO backbones against frozen DINOv3-L (laptop GPU; frozen features only, no training;
training images only).

    python ia_r42_t4_radio_frozen.py

User brief of 2026-11-19 (after v33): check NVIDIA RADIO (C-RADIOv3-L / C-RADIOv2-L) and MambaVision-L on the laptop (no Colab CU).
Part A, configuration. timm 1.0.30 lists no radio / mambavision model (only MambaOut, a different model). The Hugging Face repos load
  with transformers AutoModel + trust_remote_code: the repo code was read before this script was written (no subprocess / exec /
  network calls on the path used; open_clip / xformers optional) and the revision is pinned to that code. Checked here: the input
  normalisation (RADIO takes [0, 1] and normalises inside with its InputConditioner), patch size / min_resolution_step (192 x 384 must
  be a multiple; 12 x 24 grid), the outputs (summary = the class tokens of the teachers with use_summary, concatenated; spatial
  features 288 x 1024).
  MambaVision-L (nvidia/MambaVision-L-21K): imports mamba_ssm unconditionally (CUDA extension, source only on PyPI, no nvcc on the
  laptop) -> not run here.
Part B, frozen linear probe, exactly as hb_r40_t4_dinov3_check.py: embedding of the frozen backbone, mean of the image and its h-flip,
  training images, full frame only, bicubic to 192 x 384, bf16 autocast; standardised with the training rows' mean / SD; the
  pipeline's logistic probe (scripts/train_task4_probe.fit_probe, C 0.03, not tuned); out of fold on the members' folds; balanced
  class mix + v17 rule on full frames and the real dark / low-contrast frames. Embeddings:
    C-RADIOv3-L summary (decisive), C-RADIOv2-L summary;
    information: each RADIO's mean of the spatial features;
    reference at the same geometry: DINOv3-L @ 192 x 384 (timm, its 'avg' pooling); and DINOv3-L @ 144 x 272 from hb_r40's cache
    (expected 0.9363 / 0.8716 again).
Reading fixed before the run: a RADIO summary counts as "as strong as DINOv3-L or stronger" if, against DINOv3-L @ 192 x 384,
  full >= -0.003 and dark >= -0.003; then a Colab fold-0 notebook is worth preparing (the user decides). Otherwise stop.
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
from transformers import AutoModel  # noqa: E402

HW = (192, 384)
REV = {"nvidia/C-RADIOv3-L": "9d0413465e8a91e67bbf2c1ad342815478d1b906", "nvidia/C-RADIOv2-L": "e359f6a85c27eaf61075a7c7d68495414cccf3a0"}
V3 = "vit_large_patch16_dinov3.lvd1689m"
OUT = Path("experiments/claude/outputs/r42")
OUT.mkdir(parents=True, exist_ok=True)
BS = 64


def load_radio(repo):
    return AutoModel.from_pretrained(repo, trust_remote_code=True, revision=REV[repo]).cuda().eval()


# ---------------------------------------------------------------- part A: configuration
print(f"timm {timm.__version__}: radio models {timm.list_models('*radio*')}, mambavision models {timm.list_models('*mambavision*')}",
      flush=True)
for repo in REV:
    m = load_radio(repo)
    ic = m.input_conditioner
    print(f"\n{repo} @ {REV[repo][:8]}: {type(m).__name__} / {type(m.model).__name__}, params {sum(p.numel() for p in m.parameters()) / 1e6:.0f} M")
    print(f"  input: [0, 1] RGB, normalised inside: mean {ic.norm_mean.flatten().tolist()} std {ic.norm_std.flatten().tolist()}")
    print(f"  patch {m.patch_size}, min_resolution_step {m.min_resolution_step}, preferred {m.preferred_resolution}, "
          f"max {m.max_resolution}, summary tokens {m.num_summary_tokens}, summary_idxs "
          f"{None if m.radio_model.summary_idxs is None else m.radio_model.summary_idxs.tolist()}")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        s, f = m(torch.rand(2, 3, *HW, device="cuda"))
        _, fm = m.radio_model(torch.rand(2, 3, *HW, device="cuda"), feature_fmt="NCHW")
    print(f"  at {HW[0]} x {HW[1]}: summary {tuple(s.shape)}, spatial {tuple(f.shape)} (NLC), {tuple(fm.shape)} (NCHW: grid "
          f"{fm.shape[2]} x {fm.shape[3]}); nearest supported {m.get_nearest_supported_resolution(*HW)}", flush=True)
    del m
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- part B: frozen linear probe, full frames
def batches(paths):
    for i in range(0, len(paths), 4096):
        arr = task4.load_uint8(list(paths[i:i + 4096]))
        for j in range(0, len(arr), BS):
            x0 = torch.from_numpy(arr[j:j + BS]).cuda().permute(0, 3, 1, 2).float().div_(255)
            yield F.interpolate(x0, size=HW, mode="bicubic", align_corners=False).clamp_(0, 1)


def embed_radio(repo, paths):
    m = load_radio(repo)
    S, M = [], []
    with torch.no_grad():
        for x in batches(paths):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s1, f1 = m(x)
                s2, f2 = m(torch.flip(x, dims=[3]))
            S.append(((s1.float() + s2.float()) / 2).cpu().numpy().astype(np.float16))
            M.append(((f1.float().mean(1) + f2.float().mean(1)) / 2).cpu().numpy().astype(np.float16))
    del m
    torch.cuda.empty_cache()
    return np.concatenate(S), np.concatenate(M)


def embed_timm(name, paths):
    model = timm.create_model(name, pretrained=True, num_classes=0, img_size=HW).cuda().eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(cfg["mean"], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device="cuda").view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for x in batches(paths):
            x = (x - mean) / std
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = model(x).float() + model(torch.flip(x, dims=[3])).float()
            out.append((f / 2).cpu().numpy().astype(np.float16))
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out)


train, _ = task4.load_frames()
y = train.label.values.astype(np.int64)
FE = {}
for repo in REV:
    tag = repo.split("/")[1]
    ps, pm = OUT / f"ia_feat_{tag}_summary_{HW[0]}x{HW[1]}.npy", OUT / f"ia_feat_{tag}_spatialmean_{HW[0]}x{HW[1]}.npy"
    if not ps.exists():
        t0 = time.time()
        s, mm = embed_radio(repo, train.path.values)
        np.save(ps, s)
        np.save(pm, mm)
        print(f"{tag}: features in {(time.time() - t0) / 60:.1f} min", flush=True)
    FE[f"{tag} summary"] = np.load(ps).astype(np.float32)
    FE[f"{tag} spatial mean (info)"] = np.load(pm).astype(np.float32)
p3 = OUT / f"ia_feat_DINOv3-L_{HW[0]}x{HW[1]}.npy"
if not p3.exists():
    t0 = time.time()
    np.save(p3, embed_timm(V3, train.path.values))
    print(f"DINOv3-L @ {HW[0]}x{HW[1]}: features in {(time.time() - t0) / 60:.1f} min", flush=True)
FE["DINOv3-L @ 192x384"] = np.load(p3).astype(np.float32)
FE["DINOv3-L @ 144x272 (hb_r40)"] = np.load("experiments/claude/outputs/r40/hb_feat_DINOv3-L_144x272.npy").astype(np.float32)

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
ref = R["DINOv3-L @ 192x384"]
print()
for repo in REV:
    lab = f"{repo.split('/')[1]} summary"
    df, dd = R[lab][0] - ref[0], R[lab][1] - ref[1]
    ok = df >= -0.003 and dd >= -0.003
    print(f"{lab} minus DINOv3-L @ 192x384: full {df:+.4f}, dark {dd:+.4f} -> "
          f"{'as strong or stronger: a Colab fold-0 notebook is worth preparing' if ok else 'weaker: stop'}")
