"""Task 4 - Game Title Detection (5-class image classification, metric: macro-F1).

Pre-trained timm backbones are fine-tuned with aspect-ratio preserving inputs and corruption-
mimicking augmentation (the screenshots arrive rotated, colour-inverted, hue-shifted and noisy):
affine + RandAugment + per-sample colour/hue jitter, inversion, noise, blur, random erasing,
then Mixup/CutMix with label smoothing, cosine LR with warm-up, weight EMA and h-flip TTA.

Engineering note: images are decoded once into a uint8 array and *all* augmentation runs
batched on the GPU. On this Windows laptop, per-image CPU augmentation in DataLoader workers
starved the GPU (utilisation 0-100% oscillating) and competed with the tabular tasks.
"""
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

from .config import DATA_DIRS, T4_CLASSES

DATA = DATA_DIRS[4]
CLASS_TO_IDX = {c: i for i, c in enumerate(T4_CLASSES)}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
NATIVE_HW = (120, 230)


def load_frames():
    """Train/test frames with `label` (datacard code), `class` (text) and `path`.

    Release of 2026-11-12 (current): columns id (PLY00001.. / ANS00001..), file_name, label (int code); row i of test.csv is
    ANS{i}. The first release (2026-11-03, kept in _release_2026-11-03/) had id = file stem and class = text, and its test
    rows were in a different order. The images are byte-identical in both releases.
    """
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    if "file_name" in train.columns:
        train["label"] = train["label"].astype(int)
        train["class"] = [T4_CLASSES[k] for k in train.label]
    else:
        train["label"] = train["class"].map(CLASS_TO_IDX).astype(int)
        train["file_name"] = train.id.astype(str) + ".jpg"
        test["file_name"] = test.id.astype(str) + ".jpg"
    train["path"] = [str(DATA / "train" / f) for f in train.file_name]
    test["path"] = [str(DATA / "test" / f) for f in test.file_name]
    return train, test


def load_uint8(paths, n_threads=8):
    """Decode JPEGs into an (N, H, W, 3) uint8 array (PIL releases the GIL, so threads scale)."""
    arr = np.empty((len(paths), *NATIVE_HW, 3), dtype=np.uint8)

    def _load(i):
        im = Image.open(paths[i]).convert("RGB")
        if im.size != (NATIVE_HW[1], NATIVE_HW[0]):
            im = im.resize((NATIVE_HW[1], NATIVE_HW[0]), Image.BILINEAR)
        arr[i] = np.asarray(im)

    with ThreadPoolExecutor(n_threads) as ex:
        list(ex.map(_load, range(len(paths))))
    return arr


def to_gpu_batch(arr_uint8, img_size, device="cuda"):
    """uint8 NHWC numpy -> float NCHW [0,1] on GPU, resized to img_size (aspect ~2:1 kept)."""
    x = torch.from_numpy(arr_uint8).to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
    return F.interpolate(x, size=img_size, mode="bilinear", align_corners=False)


def normalize(x):
    m = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


# ---------------------------------------------------------------- GPU augmentation
class GPUAugment:
    """Per-sample randomised augmentation for a float [0,1] NCHW batch on the GPU."""

    def __init__(self, p_affine=0.6, max_rot=25, scale=(0.85, 1.15), max_shift=0.08, max_shear=8,
                 randaug_chunk=8, p_jitter=0.5, p_invert=0.15, p_gray=0.05, p_noise=0.4,
                 max_noise=0.10, p_blur=0.2, p_erase=0.2, p_hud=0.0, hud_zoom=(0.60, 0.75),
                 p_grid=0.0, grid_cover=(0.65, 0.85)):
        self.__dict__.update(locals())
        del self.__dict__["self"]
        self.randaug = v2.RandAugment(num_ops=2, magnitude=9)

    @staticmethod
    def _bern(p, n, dev):
        return torch.rand(n, device=dev) < p

    def _affine(self, x):
        B, _, H, W = x.shape
        dev = x.device
        on = self._bern(self.p_affine, B, dev)
        u = lambda: torch.rand(B, device=dev) * 2 - 1
        ang = u() * math.radians(self.max_rot)
        sc = self.scale[0] + torch.rand(B, device=dev) * (self.scale[1] - self.scale[0])
        sh = u() * math.radians(self.max_shear)
        # forward map A (pixel space) = R(ang) @ Shear @ diag(sc); grid_sample needs the inverse
        cos, sin, tsh = torch.cos(ang), torch.sin(ang), torch.tan(sh)
        A = torch.zeros(B, 2, 2, device=dev)
        A[:, 0, 0], A[:, 0, 1] = cos * sc, (cos * tsh - sin) * sc
        A[:, 1, 0], A[:, 1, 1] = sin * sc, (sin * tsh + cos) * sc
        Ainv = torch.linalg.inv(A)
        D = torch.tensor([[W / 2, 0.0], [0.0, H / 2]], device=dev)
        Dinv = torch.linalg.inv(D)
        M = Dinv @ Ainv @ D                                  # aspect-correct in normalised coords
        t = torch.stack([u(), u()], 1) * 2 * self.max_shift
        theta = torch.cat([M, -(M @ t.unsqueeze(-1))], dim=2)
        eye = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]], device=dev).expand(B, 2, 3)
        theta = torch.where(on[:, None, None], theta, eye)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    def _randaugment(self, x):
        xu = (x * 255).round_().clamp_(0, 255).to(torch.uint8)
        xu = torch.cat([self.randaug(c) for c in xu.split(self.randaug_chunk)])  # new ops per chunk
        return xu.float().div_(255)

    def _jitter(self, x):
        B, dev = x.shape[0], x.device
        on = self._bern(self.p_jitter, B, dev)[:, None, None, None]
        r = lambda lo, hi: (lo + torch.rand(B, device=dev) * (hi - lo))[:, None, None, None]
        y = x * r(0.7, 1.3)                                                     # brightness
        gray = (0.299 * y[:, 0] + 0.587 * y[:, 1] + 0.114 * y[:, 2]).unsqueeze(1)
        y = (y - gray.mean((2, 3), keepdim=True)) * r(0.7, 1.3) + gray.mean((2, 3), keepdim=True)  # contrast
        gray = (0.299 * y[:, 0] + 0.587 * y[:, 1] + 0.114 * y[:, 2]).unsqueeze(1)
        y = gray + (y - gray) * r(0.6, 1.4)                                     # saturation
        # hue: rotate the chroma plane in YIQ space by a per-sample angle in [-pi, pi]
        T = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]], device=dev)
        Ti = torch.linalg.inv(T)
        h = (torch.rand(B, device=dev) * 2 - 1) * math.pi
        Rm = torch.zeros(B, 3, 3, device=dev)
        Rm[:, 0, 0] = 1
        Rm[:, 1, 1], Rm[:, 1, 2], Rm[:, 2, 1], Rm[:, 2, 2] = torch.cos(h), -torch.sin(h), torch.sin(h), torch.cos(h)
        Mx = Ti @ Rm @ T
        y = torch.einsum("bij,bjhw->bihw", Mx, y)
        return torch.where(on, y.clamp(0, 1), x)

    def _hud(self, x):
        """HUD dropout (round 19): hide the screen regions that carry the HUD so the model also learns the scene itself.
        Each selected image gets one of: 0 bottom strip (15-25% of the height), 1 top strip (12-18%) + the four corners,
        2 all four borders (only the central 60-70% stays), 3 zoom into the central 60-75% (no fill, looks natural).
        Hidden areas are filled with the image's mean colour or with uniform noise (50/50)."""
        B, _, H, W = x.shape
        dev = x.device
        on = self._bern(self.p_hud, B, dev)
        if not on.any():
            return x
        mode = torch.randint(0, 4, (B,), device=dev)
        u = lambda lo, hi: (lo + torch.rand(B, device=dev) * (hi - lo)).view(B, 1, 1)
        f = u(*self.hud_zoom).view(B)
        theta = torch.zeros(B, 2, 3, device=dev)
        theta[:, 0, 0], theta[:, 1, 1] = f, f
        zoomed = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), mode="bilinear", padding_mode="border",
                               align_corners=False)
        r = torch.linspace(0, 1, H, device=dev).view(1, H, 1)
        c = torch.linspace(0, 1, W, device=dev).view(1, 1, W)
        bottom = r > 1 - u(0.15, 0.25)
        cw, chh = u(0.15, 0.25), u(0.20, 0.30)
        top_corners = (r < u(0.12, 0.18)) | (((c < cw) | (c > 1 - cw)) & ((r < chh) | (r > 1 - chh)))
        lo = (1 - u(0.60, 0.70)) / 2
        borders = (r < lo) | (r > 1 - lo) | (c < lo) | (c > 1 - lo)
        m = torch.where((mode == 0).view(B, 1, 1), bottom, torch.where((mode == 1).view(B, 1, 1), top_corners, borders))
        fill = torch.where(self._bern(0.5, B, dev)[:, None, None, None], x.mean((2, 3), keepdim=True).expand_as(x), torch.rand_like(x))
        out = torch.where((mode == 3)[:, None, None, None], zoomed, torch.where(m.unsqueeze(1), fill, x))
        return torch.where(on[:, None, None, None], out, x)

    def _grid(self, x):
        """Heavy occlusion / near-blank frame (round 22): 65-85% of the image hidden (grid_cover), either by a GridMask (random
        period 1/6-1/2 of the height, one visible window per cell) or by one large block; fill = mean colour or black (50/50)."""
        B, _, H, W = x.shape
        dev = x.device
        on = self._bern(self.p_grid, B, dev)
        if not on.any():
            return x
        cover = (self.grid_cover[0] + torch.rand(B, device=dev) * (self.grid_cover[1] - self.grid_cover[0])).view(B, 1, 1)
        vis = (1 - cover).sqrt()                                         # visible side fraction per cell
        d = (H / 6 + torch.rand(B, device=dev) * (H / 2 - H / 6)).view(B, 1, 1)
        rr = torch.arange(H, device=dev).view(1, H, 1).float()
        cc = torch.arange(W, device=dev).view(1, 1, W).float()
        o_r, o_c = torch.rand(B, 1, 1, device=dev) * d, torch.rand(B, 1, 1, device=dev) * d
        grid_vis = (((rr + o_r) % d) < vis * d) & (((cc + o_c) % d) < vis * d)
        bh, bw = vis * H, vis * W                                        # one visible window, the rest hidden
        y0, x0 = torch.rand(B, 1, 1, device=dev) * (H - bh), torch.rand(B, 1, 1, device=dev) * (W - bw)
        block_vis = (rr >= y0) & (rr < y0 + bh) & (cc >= x0) & (cc < x0 + bw)
        visible = torch.where(self._bern(0.5, B, dev).view(B, 1, 1), grid_vis, block_vis).unsqueeze(1)
        fill = torch.where(self._bern(0.5, B, dev)[:, None, None, None], x.mean((2, 3), keepdim=True).expand_as(x), torch.zeros_like(x))
        return torch.where(on[:, None, None, None] & ~visible, fill, x)

    def __call__(self, x):
        B, dev = x.shape[0], x.device
        flip = self._bern(0.5, B, dev)[:, None, None, None]            # no vertical flips (HUD layout)
        x = torch.where(flip, x.flip(-1), x)
        if self.p_hud > 0:
            x = self._hud(x)                                            # before the corruptions, like a HUD-less frame would get them
        if self.p_grid > 0:
            x = self._grid(x)                                           # near-blank frame, also before the corruptions
        x = self._affine(x)
        x = self._randaugment(x)
        x = self._jitter(x)
        inv = self._bern(self.p_invert, B, dev)[:, None, None, None]
        x = torch.where(inv, 1 - x, x)
        g = self._bern(self.p_gray, B, dev)[:, None, None, None]
        gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1).expand_as(x)
        x = torch.where(g, gray, x)
        nz = self._bern(self.p_noise, B, dev)[:, None, None, None]
        sig = (torch.rand(B, device=dev) * self.max_noise)[:, None, None, None]
        x = torch.where(nz, (x + torch.randn_like(x) * sig).clamp(0, 1), x)
        bl = self._bern(self.p_blur, B, dev)
        if bl.any():
            x[bl] = v2.functional.gaussian_blur(x[bl], kernel_size=[5, 5], sigma=[float(0.1 + 1.4 * torch.rand(1))])
        x = normalize(x)
        return self._erase(x)

    def _erase(self, x):
        B, _, H, W = x.shape
        dev = x.device
        on = self._bern(self.p_erase, B, dev)
        area = (0.02 + torch.rand(B, device=dev) * 0.13) * H * W
        logr = torch.empty(B, device=dev).uniform_(math.log(0.3), math.log(3.3))
        h = (area / torch.exp(logr)).sqrt().clamp(1, H)
        w = (area * torch.exp(logr)).sqrt().clamp(1, W)
        y0 = torch.rand(B, device=dev) * (H - h)
        x0 = torch.rand(B, device=dev) * (W - w)
        yy = torch.arange(H, device=dev)[None, :, None]
        xx = torch.arange(W, device=dev)[None, None, :]
        m = (yy >= y0[:, None, None]) & (yy < (y0 + h)[:, None, None]) & (xx >= x0[:, None, None]) & (xx < (x0 + w)[:, None, None])
        m = (m & on[:, None, None]).unsqueeze(1)
        return torch.where(m, torch.randn_like(x), x)


# -------------------------------------------------------------------------- model
# Backbones pre-trained with a non-ImageNet input normalisation that should see their own (round 22). The pipeline feeds
# ImageNet-normalised tensors everywhere; for these models a forward pre-hook maps them to the model's own mean/std.
# Older members (incl. EfficientNetV2 and EVA02, fine-tuned on ImageNet-normalised input) are deliberately left as trained.
NATIVE_NORM = {"vit_base_patch16_siglip_224.v2_webli", "vit_base_patch16_siglip_224.webli",
               "eva02_base_patch16_clip_224.merged2b", "vit_base_patch16_clip_224.openai"}


def create_model(name, num_classes=5, pretrained=True, drop_path=0.1, img_size=None):
    import timm
    kw = {}
    if img_size is not None and name.split("_")[0] in ("vit", "deit3", "deit", "eva02"):
        kw["img_size"] = tuple(img_size)        # ViTs: resample position embeddings to the 2:1 token grid
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes, drop_path_rate=drop_path, **kw)
    if name in NATIVE_NORM:
        cfg = model.pretrained_cfg
        a = torch.tensor(STD) / torch.tensor(cfg["std"])
        b = (torch.tensor(MEAN) - torch.tensor(cfg["mean"])) / torch.tensor(cfg["std"])

        def renorm(module, args):
            x = args[0]
            return (x * a.to(x.device, x.dtype).view(1, 3, 1, 1) + b.to(x.device, x.dtype).view(1, 3, 1, 1),) + tuple(args[1:])
        model.register_forward_pre_hook(renorm)
    return model


def soft_ce(logits, target):
    return torch.sum(-target * F.log_softmax(logits, dim=-1), dim=-1).mean()


@torch.no_grad()
def predict_array(model, arr, img_size, bs=128, tta=True, device="cuda"):
    """Softmax probabilities for a uint8 NHWC array; TTA = average with the horizontal flip."""
    model.eval()
    out = []
    for i in range(0, len(arr), bs):
        x = normalize(to_gpu_batch(arr[i:i + bs], img_size, device)).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(x).float().softmax(-1)
            if tta:
                p = 0.5 * (p + model(torch.flip(x, dims=[3])).float().softmax(-1))
        out.append(p.cpu())
    return torch.cat(out).numpy()


def predict_paths(model, paths, img_size, bs=128, tta=True, chunk=4096):
    """Chunked decode + predict so the 25.9k test images never sit in RAM at once."""
    outs = []
    for i in range(0, len(paths), chunk):
        outs.append(predict_array(model, load_uint8(list(paths[i:i + chunk])), img_size, bs=bs, tta=tta))
    return np.concatenate(outs)


def train_fold(model_name, arr, labels, tr_idx, va_idx, img_size=(160, 320), epochs=10, bs=64,
               lr=2e-4, wd=0.05, label_smoothing=0.1, mix_prob=0.8, seed=42, log=print, ckpt_path=None, aug_kw=None,
               class_balanced=False, layer_decay=None, grad_ckpt=False):
    """Train one fold on the shared preloaded uint8 array `arr` (rows tr_idx); returns
    (model carrying the final EMA weights, history). With an empty `va_idx` it trains on all given
    rows without per-epoch validation (final Train+Val refit). class_balanced=True draws each epoch's rows with
    probability 1/(class count), so every class is seen equally often (uniform prior, matching macro-F1).
    layer_decay (e.g. 0.75) gives each block lr * decay^(depth from the top), so pre-trained ViT layers move less.
    grad_ckpt=True recomputes block activations in the backward pass (same result, less GPU memory; round 21, ConvNeXt-Small)."""
    from sklearn.metrics import f1_score
    from timm.utils import ModelEmaV3

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    labels = np.asarray(labels)
    tr_idx = np.asarray(tr_idx)
    va_idx = np.asarray(va_idx if va_idx is not None else [], dtype=int)
    has_val = len(va_idx) > 0
    va_arr, va_labels = (arr[va_idx], labels[va_idx]) if has_val else (None, None)
    y_gpu = torch.as_tensor(labels, device=device)

    model = create_model(model_name, img_size=img_size).to(device).to(memory_format=torch.channels_last)
    ema = ModelEmaV3(model, decay=0.999)
    if grad_ckpt:   # after the EMA copy, so the returned EMA model runs inference without it
        model.set_grad_checkpointing(True)
    if layer_decay is None:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        from timm.optim import param_groups_layer_decay
        groups = param_groups_layer_decay(model, weight_decay=wd, layer_decay=layer_decay)
        for g in groups:                        # torch schedulers ignore timm's lr_scale, so bake it into lr
            g["lr"] = lr * g.pop("lr_scale", 1.0)
        opt = torch.optim.AdamW(groups, lr=lr, weight_decay=wd)
    n_steps_ep = len(tr_idx) // bs
    steps, warm = epochs * n_steps_ep, n_steps_ep
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    aug = GPUAugment(**(aug_kw or {}))
    cutmix_or_mixup = v2.RandomChoice([v2.MixUp(alpha=0.4, num_classes=5), v2.CutMix(alpha=1.0, num_classes=5)])

    history = []
    for ep in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        if class_balanced:
            w = 1.0 / np.bincount(labels[tr_idx], minlength=5)[labels[tr_idx]]
            perm = rng.choice(tr_idx, size=len(tr_idx), replace=True, p=w / w.sum())
        else:
            perm = rng.permutation(tr_idx)
        for b in range(n_steps_ep):
            idx = np.sort(perm[b * bs:(b + 1) * bs])
            x = aug(to_gpu_batch(arr[idx], img_size, device))
            y = y_gpu[torch.as_tensor(idx, device=device)]
            if rng.random() < mix_prob:
                x, yt = cutmix_or_mixup(x, y)
            else:
                yt = F.one_hot(y, 5).float()
            yt = yt * (1 - label_smoothing) + label_smoothing / 5
            x = x.to(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = soft_ce(model(x).float(), yt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            ema.update(model)
            tot += loss.item() * len(idx)
            n += len(idx)
        if has_val:
            pv = predict_array(ema.module, va_arr, img_size, tta=False)
            f1 = f1_score(va_labels, pv.argmax(1), average="macro")
            acc = (pv.argmax(1) == np.asarray(va_labels)).mean()
        else:
            f1 = acc = float("nan")
        history.append(dict(epoch=ep, loss=tot / n, val_f1=f1, val_acc=acc, secs=time.time() - t0))
        log(f"  ep {ep:2d} loss {tot / n:.4f} val_f1(ema) {f1:.4f} acc {acc:.4f} [{time.time() - t0:.0f}s]")

    # Keep the final EMA weights (no epoch selection on the validation fold -> unbiased OOF).
    if ckpt_path is not None:
        torch.save(ema.module.state_dict(), ckpt_path)
    return ema.module, history


# ------------------------------------------------------------------------ Grad-CAM
class GradCAM:
    """Minimal Grad-CAM (Selvaraju et al., 2017) on a chosen feature layer."""

    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.acts = output
        output.register_hook(lambda g: setattr(self, "grads", g))   # tensor hook: robust to in-place ops

    def __call__(self, x, cls=None):
        self.model.eval()
        logits = self.model(x)
        cls = logits.argmax(1) if cls is None else cls
        self.model.zero_grad()
        logits.gather(1, cls.view(-1, 1)).sum().backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        return cam.detach().cpu().numpy(), logits.softmax(-1).detach().cpu().numpy()
