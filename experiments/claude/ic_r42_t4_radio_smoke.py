"""Round 42 part B, Task 4: smoke test of the C-RADIOv3-L fine-tune path (src/task4.RadioClassifier) on the laptop GPU before any Colab
run (training images only; nothing is kept).

    python ic_r42_t4_radio_smoke.py [--bs 4]

Checks, each printed with its numbers:
  1. input: the wrapper receives ImageNet-normalised tensors (as every pipeline path feeds) and maps them back to [0, 1]; its summary
     must equal RADIO's own summary on the [0, 1] image (fp32, max |diff| ~ 0) -> no double normalisation; head Linear(3072 -> 5).
  2. layer decay 0.90 through timm.optim.param_groups_layer_decay (as train_fold): every parameter accounted for; the CPE patch
     generator in the bottom group, blocks 0-23 one group each with scale 0.90^(top - i), final norm + head on top.
  3. gradient checkpointing: with set_grad_checkpointing(True) RADIO's CPE forward wraps each of the 24 blocks in
     torch.utils.checkpoint.checkpoint (counted per training step: 24 with, 0 without) and the peak GPU memory of a batch of 8
     drops; every trainable parameter gets a finite gradient. (Module forward hooks are not a usable counter: they do not fire again
     in torch's recomputation; a plain timm ViT shows the same.)
  4. train_fold itself for two optimiser steps (forward + backward + clip + AdamW + EMA) on real training frames at 192 x 384 with
     the Colab recipe (lr 8e-5, layer decay 0.90, warm-up, p_hud 0.70, Mixup / CutMix, bf16, grad-ckpt), then its validation pass:
     finite loss, EMA weights moved.
  5. the checkpoint round trip used by the views / test scripts: state_dict -> create_model(pretrained=False) -> same outputs.
"""
import argparse
import sys
import time

import numpy as np
import torch

import common  # noqa: F401  (chdir to the repo root)

sys.path.insert(0, "scripts")
from src import task4  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--bs", type=int, default=4)
args = ap.parse_args()
NAME, HW = "c-radio_v3-l", (192, 384)
t0 = time.time()
log = lambda m: print(f"{m} [{time.time() - t0:.0f}s, GPU peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB]", flush=True)

# ---- 1. input normalisation and head
m = task4.create_model(NAME, img_size=HW).cuda().eval()
x01 = torch.rand(2, 3, *HW, device="cuda")
with torch.no_grad():
    s_wrap = m.forward_summary(task4.normalize(x01))
    s_own, _ = m.radio.radio_model(x01)
    logits = m(task4.normalize(x01))
print(f"1. head {m.head}; RADIO conditioner mean {m.radio.radio_model.input_conditioner.norm_mean.flatten().tolist()}")
print(f"   summary via the wrapper (ImageNet-normalised input) vs RADIO on [0, 1]: shape {tuple(s_wrap.shape)}, "
      f"max |diff| {(s_wrap - s_own).abs().max().item():.2e}; logits {tuple(logits.shape)}", flush=True)
assert (s_wrap - s_own).abs().max().item() < 1e-4 and s_wrap.shape[1] == 3072

# ---- 2. layer decay groups
from timm.optim import param_groups_layer_decay  # noqa: E402
groups = param_groups_layer_decay(m, weight_decay=0.05, layer_decay=0.90)
n_named = sum(1 for _, p in m.named_parameters() if p.requires_grad)
n_grouped = sum(len(g["params"]) for g in groups)
scale_of = {}
for g in groups:
    for p in g["params"]:
        scale_of[id(p)] = g["lr_scale"]
by_scale = {}
for n, p in m.named_parameters():
    if not p.requires_grad:      # timm's unused reg_token, frozen in RadioClassifier
        continue
    key =("block %s" % n.split(".")[4]) if ".blocks." in n else n.rsplit(".", 1)[0]
    by_scale.setdefault(round(scale_of[id(p)], 6), set()).add(key)
print(f"2. layer decay 0.90: {len(groups)} groups, {n_grouped} grouped of {n_named} trainable parameters")
for s in sorted(by_scale):
    names = sorted(by_scale[s])
    print(f"   lr scale {s:.4f}: {', '.join(names[:4])}{' ...' if len(names) > 4 else ''} ({len(names)} modules)")
blk_scales = [scale_of[id(m.radio.radio_model.model.blocks[i].attn.qkv.weight)] for i in range(24)]
top = max(scale_of.values())
assert n_grouped == n_named
assert all(abs(blk_scales[i] / blk_scales[i + 1] - 0.90) < 1e-6 for i in range(23)), blk_scales
assert scale_of[id(m.head.weight)] == top and scale_of[id(next(m.radio.radio_model.model.patch_generator.parameters()))] < blk_scales[0]
print(f"   blocks 0 / 12 / 23: {blk_scales[0]:.4f} / {blk_scales[12]:.4f} / {blk_scales[23]:.4f}; head {scale_of[id(m.head.weight)]:.4f}; "
      f"patch generator {scale_of[id(next(m.radio.radio_model.model.patch_generator.parameters()))]:.4f}", flush=True)

# ---- 3. gradient checkpointing
import torch.utils.checkpoint as tuc  # noqa: E402
_orig_ckpt, n_ckpt = tuc.checkpoint, [0]


def _counting_ckpt(fn, *a, **k):
    n_ckpt[0] += 1
    return _orig_ckpt(fn, *a, **k)


tuc.checkpoint = _counting_ckpt
m.train()
peak = {}
for ckpt in (False, True):
    m.set_grad_checkpointing(ckpt)
    m.zero_grad(set_to_none=True)
    n_ckpt[0] = 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = m(task4.normalize(torch.rand(8, 3, *HW, device="cuda"))).float().logsumexp(1).mean()
    loss.backward()
    peak[ckpt] = torch.cuda.max_memory_allocated() / 2**30
    no_grad = [n for n, p in m.named_parameters() if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    print(f"3. grad-ckpt {ckpt}: checkpointed segments per step {n_ckpt[0]} (blocks 24); parameters without a finite gradient: "
          f"{len(no_grad)} {no_grad[:3]}; peak GPU memory, batch 8: {peak[ckpt]:.2f} GB", flush=True)
    assert n_ckpt[0] == (24 if ckpt else 0) and not no_grad
tuc.checkpoint = _orig_ckpt
assert peak[True] < peak[False]
m.set_grad_checkpointing(False)
del m, loss
torch.cuda.empty_cache()

# ---- 4. train_fold: two optimiser steps + EMA + validation
train, _ = task4.load_frames()
rng = np.random.default_rng(0)
rows = np.sort(rng.choice(len(train), 2 * args.bs + 16, replace=False))
arr = task4.load_uint8(list(train.path.values[rows]))
labels = train.label.values[rows]
init = task4.create_model(NAME, img_size=HW)
w0 = {k: v.detach().clone() for k, v in init.state_dict().items() if k.startswith("radio.radio_model.model.blocks.23.mlp")}
del init
torch.cuda.reset_peak_memory_stats()
ema, hist = task4.train_fold(NAME, arr, labels, np.arange(2 * args.bs), np.arange(2 * args.bs, len(rows)), img_size=HW, epochs=1,
                             bs=args.bs, lr=8e-5, layer_decay=0.90, grad_ckpt=True, warmup_epochs=1, aug_kw={"p_hud": 0.70},
                             log=lambda s: print("   " + s, flush=True))
moved = max((ema.state_dict()[k].float() - w0[k].cuda().float()).abs().max().item() for k in w0)
log(f"4. train_fold, 2 steps of batch {args.bs} at {HW[0]} x {HW[1]}: loss {hist[0]['loss']:.4f} (finite {np.isfinite(hist[0]['loss'])}), "
    f"val macro-F1 on 16 frames {hist[0]['val_f1']:.3f}; EMA block-23 MLP weights moved by up to {moved:.2e}")
assert np.isfinite(hist[0]["loss"]) and moved > 0

# ---- 5. checkpoint round trip (pretrained=False path of the views / test scripts)
sd = {k: v.detach().cpu() for k, v in ema.state_dict().items()}
m2 = task4.create_model(NAME, pretrained=False, img_size=HW)
missing, unexpected = m2.load_state_dict(sd, strict=True), None
m2 = m2.cuda().eval()
xb = task4.normalize(torch.rand(4, 3, *HW, device="cuda"))
with torch.no_grad():
    d = (ema.eval()(xb) - m2(xb)).abs().max().item()
log(f"5. state_dict -> create_model(pretrained=False) -> load_state_dict(strict=True): max |logit diff| {d:.2e}")
assert d < 1e-4
print("\nSMOKE TEST PASSED")
