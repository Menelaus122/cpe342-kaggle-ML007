"""Round 40, Task 4: is the DINOv3 fine-tune set up correctly? (laptop GPU; frozen features only, no training; training images only)

    python hb_r40_t4_dinov3_check.py

User brief of 2026-11-18 (after dinov3L_hud70 fold 0 = 0.9794, far below DINOv2-L's 0.9885, and slow to converge: epoch 4 0.904
against 0.965): check the input against what the model was pre-trained with before blaming the learning rate.
Part A, configuration, for the models exactly as src/task4.create_model builds them for training (num_classes 5, img_size):
  the pre-trained mean / std against the pipeline's normalisation (ImageNet), global pooling, prefix tokens (class + registers),
  patch grid (must be 9 x 17 at 144 x 272), the classifier's input; and the RoPE grid: the fixed-size model (img_size 144 x 272) must
  give the same features as timm's dynamic-size path (dynamic_img_size=True) on the same 144 x 272 batch.
Part B, frozen linear probe (decisive): pooled embedding of the frozen pre-trained backbone (num_classes 0), mean of the image and
  its h-flip, training images, full frame only; standardised with the training rows' mean / SD, the pipeline's logistic probe
  (scripts/train_task4_probe.fit_probe, C 0.03, not tuned), out of fold on the members' folds; scored as gi_r37 (balanced class mix,
  v17 rule) on full frames and the real dark / low-contrast frames. Backbones x input:
    DINOv2-L @ 224 x 224   (cached by gi_r37; also refitted with gi_r37's three-view recipe as a check against its 0.9084)
    DINOv3-L @ 224 x 224   (bicubic, as the existing probes)
    DINOv2-L @ 126 x 238   (the fine-tune geometry of v30's member)
    DINOv3-L @ 144 x 272   (the fine-tune geometry of dinov3L_hud70)
Reading fixed before the run: if a Part A item is wrong, or DINOv3-L is more than 0.01 below DINOv2-L at the matched geometry
(full frames), the set-up is suspect and the fold-0 result cannot decide; otherwise the set-up is right and the gap is the
fine-tuning recipe (or the model).
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

V3, V2 = "vit_large_patch16_dinov3.lvd1689m", "vit_large_patch14_reg4_dinov2.lvd142m"
OUT = Path("experiments/claude/outputs/r40")
import timm  # noqa: E402

# ---------------------------------------------------------------- part A: configuration
print("timm", timm.__version__, "| pipeline mean / std", task4.MEAN, task4.STD, flush=True)
for name, hw in ((V3, (144, 272)), (V2, (126, 238))):
    m = task4.create_model(name, img_size=hw)
    cfg = timm.data.resolve_model_data_config(m)
    tok = m.forward_features(torch.zeros(1, 3, *hw))
    print(f"\n{name} as trained (img_size {hw}): class {type(m).__name__}, in NATIVE_NORM {name in task4.NATIVE_NORM}")
    print(f"  resolve_model_data_config: {cfg}")
    print(f"  mean / std equal to the pipeline's: {np.allclose(cfg['mean'], task4.MEAN, atol=1e-3) and np.allclose(cfg['std'], task4.STD, atol=1e-3)}")
    print(f"  global_pool {m.global_pool!r}, num_prefix_tokens {m.num_prefix_tokens}, patch grid {tuple(m.patch_embed.grid_size)}, "
          f"patch size {tuple(m.patch_embed.patch_size)}, tokens {tuple(tok.shape)}, fc_norm {type(getattr(m, 'fc_norm', None)).__name__}, "
          f"head {m.get_classifier()}", flush=True)
    for k in ("rope", "pos_embed"):
        a = getattr(m, k, None)
        if a is not None:
            desc = tuple(a.shape) if torch.is_tensor(a) else {kk: v for kk, v in vars(a).items() if not kk.startswith("_") and not torch.is_tensor(v)}
            print(f"  {k}: {desc}")
    del m
torch.manual_seed(0)
x = torch.rand(4, 3, 144, 272).cuda()
xn = task4.normalize(x)
mf = timm.create_model(V3, pretrained=True, num_classes=0, img_size=(144, 272)).cuda().eval()
md = timm.create_model(V3, pretrained=True, num_classes=0, dynamic_img_size=True).cuda().eval()
with torch.no_grad():
    a, b = mf.forward_features(xn), md.forward_features(xn)
print(f"\nRoPE grid: fixed 144 x 272 vs dynamic-size path on the same batch: tokens {tuple(a.shape)} / {tuple(b.shape)}, "
      f"max |diff| {(a - b).abs().max().item():.2e} (fp32)", flush=True)
del mf, md, a, b
torch.cuda.empty_cache()


# ---------------------------------------------------------------- part B: frozen linear probe, full frames
def embed(name, hw, paths):
    model = timm.create_model(name, pretrained=True, num_classes=0, img_size=hw).cuda().eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(cfg["mean"], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device="cuda").view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), 4096):
            arr = task4.load_uint8(list(paths[i:i + 4096]))
            for j in range(0, len(arr), 128):
                x0 = torch.from_numpy(arr[j:j + 128]).cuda().permute(0, 3, 1, 2).float().div_(255)
                x = (F.interpolate(x0, size=hw, mode="bicubic", align_corners=False).clamp_(0, 1) - mean) / std
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    f = model(x).float() + model(torch.flip(x, dims=[3])).float()
                out.append((f / 2).cpu().numpy().astype(np.float16))
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out)


train, _ = task4.load_frames()
y = train.label.values.astype(np.int64)
FE = {}
zl = np.load(PRED_DIR / "t4_feat_views_large14_reg4_dinov2.npz")
FE["DINOv2-L @ 224x224"] = zl["full"].astype(np.float32)
for lab, name, hw in (("DINOv3-L @ 224x224", V3, (224, 224)), ("DINOv2-L @ 126x238", V2, (126, 238)), ("DINOv3-L @ 144x272", V3, (144, 272))):
    p = OUT / f"hb_feat_{lab.split()[0]}_{hw[0]}x{hw[1]}.npy"
    if not p.exists():
        t0 = time.time()
        np.save(p, embed(name, hw, train.path.values))
        print(f"{lab}: features in {(time.time() - t0) / 60:.1f} min", flush=True)
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


def oof(Xs, fit_views):
    """Xs: {view: features}; fitted on fit_views of the training folds, predicted on the held-out fold's full frames."""
    out = np.zeros((len(y), 5))
    for f, va in enumerate(folds):
        tr = np.where(fo != f)[0]
        mu, sd = Xs["full"][tr].mean(0), Xs["full"][tr].std(0) + 1e-6
        W, b = P0.fit_probe(np.vstack([(Xs[v][tr] - mu) / sd for v in fit_views]), np.tile(y[tr], len(fit_views)))
        out[va] = (torch.from_numpy((Xs["full"][va] - mu) / sd) @ W + b).softmax(1).numpy()
    return out


print("\nfrozen linear probe, out of fold (balanced mix, v17 rule; full frames | real dark / low-contrast frames):", flush=True)
chk = oof({v: zl[v].astype(np.float32) for v in ("full", "zoom70", "centre_grey")}, ["full", "zoom70", "centre_grey"])
print(f"  check: DINOv2-L @ 224x224 with gi_r37's three-view fit: full {bal(rule(chk), allm):.4f} (gi_r37: 0.9084), "
      f"dark {bal(rule(chk), dark):.4f} (0.8335)", flush=True)
R = {}
for lab, X in FE.items():
    p = oof({"full": X}, ["full"])
    R[lab] = (bal(rule(p), allm), bal(rule(p), dark), f1_score(y, p.argmax(1), average="macro"))
    print(f"  {lab:20s} {X.shape[1]}-d: full {R[lab][0]:.4f} | dark {R[lab][1]:.4f} | plain macro-F1 (training mix, argmax) {R[lab][2]:.4f}",
          flush=True)
d224 = R["DINOv3-L @ 224x224"][0] - R["DINOv2-L @ 224x224"][0]
dft = R["DINOv3-L @ 144x272"][0] - R["DINOv2-L @ 126x238"][0]
print(f"\nDINOv3-L minus DINOv2-L, full frames: at 224 x 224 {d224:+.4f}; at the fine-tune geometry {dft:+.4f}")
print("-> set-up suspect (frozen DINOv3 more than 0.01 below at the fine-tune geometry)" if dft < -0.01
      else "-> frozen DINOv3 is within 0.01 of DINOv2-L (or better): the set-up is not the cause; the gap is the fine-tune recipe / model")
