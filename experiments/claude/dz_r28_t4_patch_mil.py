"""Round 28, Task 4: a multiple-instance head on the patch tokens of a FROZEN backbone, fold 0 (training images only).

    python dz_r28_t4_patch_mil.py dinov2            # or: clip, siglip, cnxtS

The frozen-feature probe of v19 pools every image into one vector (CLS / mean token) before the classifier, so a small region that
identifies the game (a HUD element, a character, a weapon model) is averaged with everything else. Here the backbone stays frozen
and a small head scores every patch token on its own:
    h_i = MLP(LayerNorm(token_i));  class evidence e_i = W h_i (5 numbers per patch);  attention a_ic = softmax_i(v_c . tanh(U h_i))
    image logit_c = sum_i a_ic * e_ic        (gated attention pooling per class, Ilse et al. 2018)
Training: the pipeline's GPU augmentation with HUD dropout 0.5, CutMix / Mixup (p 0.5), label smoothing 0.1, 30% of the tokens
dropped at random; AdamW 1e-3, cosine, 12 epochs; EMA of the head. Only the head is trained (about 0.5M parameters); the backbone
is run without gradients. Fold 0 of the pipeline's folds; the validation images are scored under the full, 70%-zoom and HUD-hidden
views with h-flip TTA. Test images are not touched in this script.
Reported: the head alone on the three views (uniform class mix, v17 rule), next to the pooled probe of v19 on the same rows, and
the fusion with the seven fine-tuned members: v19 = 0.9 ens + 0.1 probe against mixes that add / swap in the MIL head.
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torchvision.transforms import v2

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402

NAME = sys.argv[1] if len(sys.argv) > 1 else "dinov2"
BACKBONES = {"dinov2": ("vit_base_patch14_reg4_dinov2.lvd142m", (196, 392)),
             "clip": ("vit_base_patch16_clip_224.openai", (192, 384)),
             "siglip": ("vit_base_patch16_siglip_256.v2_webli", (192, 384)),
             "cnxtS": ("convnext_small.in12k_ft_in1k", (192, 384))}
MODEL, SIZE = BACKBONES[NAME]
EPOCHS, BS, LR = 12, 128, 1e-3
OUT = Path("experiments/claude/outputs/r28")
OUT.mkdir(parents=True, exist_ok=True)
V19 = Path("experiments/claude/outputs/r19_views")
R27 = Path("experiments/claude/outputs/r27")
TAGS = ["cnxt", "deit3", "cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
VIEW_NAMES = ["full", "zoom70", "centre_grey"]
torch.manual_seed(0)
rng = np.random.default_rng(0)
dev = "cuda"

train, _ = task4.load_frames()
y = train.label.values
va = np.load(PRED_DIR / "t4_cnxt_fold0.npz")["val_idx"]
tr = np.setdiff1d(np.arange(len(y)), va)
arr = task4.load_uint8(list(train.path.values))
y_gpu = torch.as_tensor(y, device=dev)

kw = {"img_size": SIZE} if MODEL.startswith("vit") else {}
backbone = timm.create_model(MODEL, pretrained=True, num_classes=0, **kw).to(dev).eval()
for p in backbone.parameters():
    p.requires_grad_(False)
cfg = timm.data.resolve_data_config({}, model=backbone)
mean = torch.tensor(cfg["mean"], device=dev).view(1, 3, 1, 1)
std = torch.tensor(cfg["std"], device=dev).view(1, 3, 1, 1)
n_prefix = getattr(backbone, "num_prefix_tokens", 0)


@torch.no_grad()
def tokens(x):
    """x: float [0, 1] NCHW at SIZE -> patch tokens (B, N, D), float32."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        f = backbone.forward_features((x - mean) / std)
    f = f.flatten(2).transpose(1, 2) if f.ndim == 4 else f[:, n_prefix:]
    return f.float()


class MILHead(nn.Module):
    def __init__(self, d, h=384, n_cls=5, p=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(p))
        self.cls = nn.Linear(h, n_cls)
        self.att = nn.Sequential(nn.Linear(h, 128), nn.Tanh(), nn.Linear(128, n_cls))

    def forward(self, t):
        h = self.proj(self.norm(t))
        return (self.att(h).softmax(1) * self.cls(h)).sum(1)


def zoom(x, f):
    th = torch.zeros(x.shape[0], 2, 3, device=x.device)
    th[:, 0, 0], th[:, 1, 1] = f, f
    return F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), mode="bilinear", padding_mode="border", align_corners=False)


def centre_grey(x):
    H, W = x.shape[2:]
    m = torch.zeros(1, 1, H, W, device=x.device)
    m[..., H // 5:4 * H // 5, W // 5:4 * W // 5] = 1
    return x * m + 0.5 * (1 - m)


VIEWS = {"full": lambda x: x, "zoom70": lambda x: zoom(x, 0.7), "centre_grey": centre_grey}


@torch.no_grad()
def predict(head, rows, views=("full",)):
    head.eval()
    out = {v: [] for v in views}
    for i in range(0, len(rows), 256):
        x0 = task4.to_gpu_batch(arr[rows[i:i + 256]], SIZE, dev)
        for v in views:
            x = VIEWS[v](x0)
            p = 0.5 * (head(tokens(x)).softmax(-1) + head(tokens(x.flip(-1))).softmax(-1))
            out[v].append(p.cpu().numpy())
    return {v: np.concatenate(o) for v, o in out.items()}


d = tokens(task4.to_gpu_batch(arr[:2], SIZE, dev)).shape
print(f"{NAME}: {MODEL} at {SIZE}, frozen; {d[1]} patch tokens x {d[2]} dims; fold 0: train {len(tr)}, val {len(va)}", flush=True)
head = MILHead(d[2]).to(dev)
ema = MILHead(d[2]).to(dev)
ema.load_state_dict(head.state_dict())
opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.05)
steps_ep = len(tr) // BS
total, warm = EPOCHS * steps_ep, steps_ep
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
aug = task4.GPUAugment(p_hud=0.5)
mixer = v2.RandomChoice([v2.MixUp(alpha=0.4, num_classes=5), v2.CutMix(alpha=1.0, num_classes=5)])
for ep in range(EPOCHS):
    head.train()
    t0, tot = time.time(), 0.0
    perm = rng.permutation(tr)
    for b in range(steps_ep):
        idx = np.sort(perm[b * BS:(b + 1) * BS])
        x = aug(task4.to_gpu_batch(arr[idx], SIZE, dev))
        yb = y_gpu[torch.as_tensor(idx, device=dev)]
        if rng.random() < 0.5:
            x, yt = mixer(x, yb)
        else:
            yt = F.one_hot(yb, 5).float()
        yt = yt * 0.9 + 0.02
        t = tokens(x)
        keep = torch.rand(t.shape[0], t.shape[1], device=dev).argsort(1)[:, :int(0.7 * t.shape[1])]       # token dropout
        t = torch.gather(t, 1, keep.unsqueeze(-1).expand(-1, -1, t.shape[2]))
        loss = -(yt * head(t).log_softmax(-1)).sum(1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), head.parameters()):
                pe.mul_(0.998).add_(pm.detach(), alpha=0.002)
        tot += loss.item()
    pv = predict(ema, va)["full"]
    print(f"  ep {ep:2d} loss {tot / steps_ep:.4f} val macro-F1 (ema, full frames) {f1_score(y[va], pv.argmax(1), average='macro'):.4f} [{time.time() - t0:.0f}s]", flush=True)

P = predict(ema, va, VIEW_NAMES)
np.savez(OUT / f"t4_mil_{NAME}_fold0.npz", val_idx=va, **P)
torch.save(ema.state_dict(), OUT / f"t4_mil_{NAME}_fold0.pt")

# ---------------------------------------------------------------- evaluation on the fold-0 validation images
yv = y[va]
pi = np.bincount(y, minlength=5) / len(y)
wb = (0.2 / pi)[yv]
rule = lambda p: (p * pi ** -0.75).argmax(1)
bal = lambda pred, i=slice(None): f1_score(yv[i], pred[i], average="macro", sample_weight=wb[i])
ens = {v: np.mean([np.load(V19 / f"{t}_fold0.npz")[v] for t in TAGS], 0) for v in VIEW_NAMES}
zp = np.load(R27 / "t4_probe_views.npz")
probe = {v: zp[f"probe-aug_{v}"][va].astype(float) for v in VIEW_NAMES}
print("\nfold-0 validation images, uniform class mix, v17 rule: full / zoom70 / HUD hidden / mean")
for name, Q in (("seven fine-tuned members", ens), ("pooled probe (v19 member)", probe), (f"patch MIL head ({NAME})", P)):
    s = [bal(rule(Q[v])) for v in VIEW_NAMES]
    print(f"  {name:28s} " + " / ".join(f"{x:.4f}" for x in s) + f" / {np.mean(s):.4f}")
print("  agreement of the argmax, MIL vs probe: " + ", ".join(f"{v} {np.mean(P[v].argmax(1) == probe[v].argmax(1)):.3f}" for v in VIEW_NAMES)
      + "; MIL vs ensemble: " + ", ".join(f"{v} {np.mean(P[v].argmax(1) == ens[v].argmax(1)):.3f}" for v in VIEW_NAMES))
boot = [np.random.default_rng(1).integers(0, len(va), len(va)) for _ in range(1)]
rb = np.random.default_rng(1)
boot = [rb.integers(0, len(va), len(va)) for _ in range(1000)]
base = {v: rule(0.9 * ens[v] + 0.1 * probe[v]) for v in VIEW_NAMES}
print(f"\nfusion on fold 0 (weights fixed, not tuned): delta of the three-view mean against v19 = 0.9 ens + 0.1 probe ({np.mean([bal(base[v]) for v in VIEW_NAMES]):.4f})")
for we, wp, wm in ((0.9, 0.0, 0.1), (0.85, 0.05, 0.10), (0.8, 0.1, 0.1), (0.8, 0.0, 0.2), (0.7, 0.1, 0.2), (0.6, 0.1, 0.3)):
    pred = {v: rule(we * ens[v] + wp * probe[v] + wm * P[v]) for v in VIEW_NAMES}
    per = [bal(pred[v]) - bal(base[v]) for v in VIEW_NAMES]
    bs = [np.mean([bal(pred[v], i) - bal(base[v], i) for v in VIEW_NAMES]) for i in boot]
    print(f"  {we:.2f} ens + {wp:.2f} probe + {wm:.2f} MIL: {np.mean(per):+.4f} [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}] "
          f"(full {per[0]:+.4f}, zoom70 {per[1]:+.4f}, HUD hidden {per[2]:+.4f}); labels changed {np.mean([np.mean(pred[v] != base[v]) for v in VIEW_NAMES]):.2%}")
