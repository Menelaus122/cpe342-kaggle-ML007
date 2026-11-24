"""Generate self-contained Round 43 Colab notebook for Top-12-block LoRA fine-tuning of DINOv3-7B @ 144x272:
    python colab/build_r43_lora_nb.py
    python colab/build_r43_lora_nb.py --smoke-local

Default run generates `colab/t4_colab_dinov3_7b_lora.ipynb` (Fold-0 Gate first).
`--smoke-local` also executes the exact notebook core code (`LoRALinear`, `TopKLoRADinoV3`, `TrainableEMA`,
`GPUAugment`, `train_lora_fold`, checkpoint save/reload parity, and 5-view eval) on the local GPU using
`vit_small_patch16_dinov3.lvd1689m` plus verifies 7B LoRA injection on `meta` device.
"""
import argparse
import ast
import math
import time
from pathlib import Path

import nbformat as nbf

SHARED_SETUP_HELPER = r'''
DRIVE_CANDIDATES = [
    "/content/drive/MyDrive/cpe342_colab",
    "/content/drive/MyDrive/cpe342-colab",
]
'''

SHARED_ERROR_HANDLER = r'''
try:
    _ip = get_ipython()
except NameError:
    _ip = None
if AUTO_DISCONNECT and _ip is not None:
    def _end_runtime_on_error(shell, etype, evalue, tb, tb_offset=None):
        shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)
        import time as _t
        print(f"\nerror -> ending runtime in 60 s [{_t.strftime('%H:%M')} UTC]", flush=True)
        _t.sleep(60)
        from google.colab import runtime
        runtime.unassign()
    _ip.set_custom_exc((Exception,), _end_runtime_on_error)
'''

SHARED_DATA_UNPACK = r'''
import os
import shutil
import zipfile
from pathlib import Path
from google.colab import drive

drive.mount("/content/drive")

D = None
for cand in DRIVE_CANDIDATES:
    p = Path(cand)
    if (p / "t4_data.zip").exists() or (p / "out").exists():
        D = p
        break
if D is None:
    my_drive = Path("/content/drive/MyDrive")
    for p in sorted(my_drive.glob("*cpe342*")):
        if (p / "t4_data.zip").exists() or (p / "out").exists():
            D = p
            break
assert D is not None and (D / "t4_data.zip").exists(), f"t4_data.zip not found under {DRIVE_CANDIDATES}"
print("Using Drive folder:", D)

R = Path("/content/cpe342")
DATA = R / "04_image_classification/_download/task4_image_classification"
if not (DATA / "test.csv").exists() or len(os.listdir(DATA / "train")) < 31546 or len(os.listdir(DATA / "test")) < 25889:
    R.mkdir(parents=True, exist_ok=True)
    shutil.copy(D / "t4_data.zip", "/content/t4_data.zip")
    zipfile.ZipFile("/content/t4_data.zip").extractall(R)
    os.remove("/content/t4_data.zip")
print("Unpacked images -> train:", len(os.listdir(DATA / "train")), "| test:", len(os.listdir(DATA / "test")))
'''

CORE_MODULE_CODE = r'''
# ---- Self-contained Top-12-block LoRA + Task 4 GPUAugment + TrainableEMA + 5-View Eval -----------------------------------
import copy
import gc
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as tuc
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from timm.layers import DropPath
from torchvision.transforms import v2

NATIVE_HW = (120, 230)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
T4_CLASSES = ["7-11k", "fiveN", "freefried", "pube", "roblock"]
LORA_TARGET_LEAF_MODULES = ("attn.qkv", "attn.proj", "mlp.fc1_g", "mlp.fc1_x", "mlp.fc2")


def load_uint8(paths, n_threads=8):
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
    x = torch.from_numpy(arr_uint8).to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
    return F.interpolate(x, size=img_size, mode="bilinear", align_corners=False)


def normalize(x):
    m = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


def soft_ce(logits, target):
    return torch.sum(-target * F.log_softmax(logits, dim=-1), dim=-1).mean()


class GPUAugment:
    """Verbatim per-sample Task 4 GPU augmentation matching src/task4.py."""

    def __init__(self, p_affine=0.6, max_rot=25, scale=(0.85, 1.15), max_shift=0.08, max_shear=8,
                 randaug_chunk=8, p_jitter=0.5, p_invert=0.15, p_gray=0.05, p_noise=0.4,
                 max_noise=0.10, p_blur=0.2, p_erase=0.2, p_hud=0.0, hud_zoom=(0.60, 0.75)):
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
        cos, sin, tsh = torch.cos(ang), torch.sin(ang), torch.tan(sh)
        A = torch.zeros(B, 2, 2, device=dev)
        A[:, 0, 0], A[:, 0, 1] = cos * sc, (cos * tsh - sin) * sc
        A[:, 1, 0], A[:, 1, 1] = sin * sc, (sin * tsh + cos) * sc
        Ainv = torch.linalg.inv(A)
        D = torch.tensor([[W / 2, 0.0], [0.0, H / 2]], device=dev)
        Dinv = torch.linalg.inv(D)
        M = Dinv @ Ainv @ D
        t = torch.stack([u(), u()], 1) * 2 * self.max_shift
        theta = torch.cat([M, -(M @ t.unsqueeze(-1))], dim=2)
        eye = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]], device=dev).expand(B, 2, 3)
        theta = torch.where(on[:, None, None], theta, eye)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    def _randaugment(self, x):
        xu = (x * 255).round_().clamp_(0, 255).to(torch.uint8)
        xu = torch.cat([self.randaug(c) for c in xu.split(self.randaug_chunk)])
        return xu.float().div_(255)

    def _jitter(self, x):
        B, dev = x.shape[0], x.device
        on = self._bern(self.p_jitter, B, dev)[:, None, None, None]
        r = lambda lo, hi: (lo + torch.rand(B, device=dev) * (hi - lo))[:, None, None, None]
        y = x * r(0.7, 1.3)
        gray = (0.299 * y[:, 0] + 0.587 * y[:, 1] + 0.114 * y[:, 2]).unsqueeze(1)
        y = (y - gray.mean((2, 3), keepdim=True)) * r(0.7, 1.3) + gray.mean((2, 3), keepdim=True)
        gray = (0.299 * y[:, 0] + 0.587 * y[:, 1] + 0.114 * y[:, 2]).unsqueeze(1)
        y = gray + (y - gray) * r(0.6, 1.4)
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
        zoomed = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), mode="bilinear",
                               padding_mode="border", align_corners=False)
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

    def __call__(self, x):
        B, dev = x.shape[0], x.device
        flip = self._bern(0.5, B, dev)[:, None, None, None]
        x = torch.where(flip, x.flip(-1), x)
        if self.p_hud > 0:
            x = self._hud(x)
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


class LoRALinear(nn.Module):
    """Frozen bfloat16 base Linear + trainable float32 low-rank adapter (r, alpha)."""

    def __init__(self, base_linear: nn.Linear, r: int = 16, alpha: float = 32.0, dropout: float = 0.05):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = int(r)
        self.scaling = float(alpha) / float(r)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        dev = base_linear.weight.device
        self.lora_A = nn.Parameter(torch.empty(self.r, base_linear.in_features, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base_linear.out_features, self.r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_dtype = self.base.weight.dtype
        xb = x if x.dtype == w_dtype else x.to(w_dtype)
        base_out = self.base(xb)
        dx = self.drop(xb) if self.training else xb
        a = self.lora_A.to(dx.dtype)
        b = self.lora_B.to(dx.dtype)
        lora_out = F.linear(F.linear(dx, a), b) * self.scaling
        return base_out + lora_out


class TopKLoRADinoV3(nn.Module):
    """Freeze bottom [0:freeze_until] EvaBlocks under torch.no_grad(); adapt top [freeze_until:] blocks via LoRA."""

    def __init__(self, base: nn.Module, freeze_until: int = 28, lora_r: int = 16,
                 lora_alpha: float = 32.0, lora_dropout: float = 0.05, drop_path: float = 0.1,
                 grad_ckpt: bool = True):
        super().__init__()
        self.base = base
        self.depth = len(base.blocks)
        assert 0 <= freeze_until < self.depth, f"Invalid freeze_until={freeze_until} for depth={self.depth}"
        self.freeze_until = int(freeze_until)
        self.grad_ckpt = bool(grad_ckpt)

        # Freeze entire base first
        for p in self.base.parameters():
            p.requires_grad_(False)

        n_top = self.depth - self.freeze_until
        dp_rates = torch.linspace(0.0, drop_path, n_top).tolist() if drop_path > 0 else [0.0] * n_top
        self.n_lora = 0

        for idx_rel, i in enumerate(range(self.freeze_until, self.depth)):
            blk = self.base.blocks[i]
            # Trainable fp32 norms inside top blocks
            for nm in ("norm1", "norm2"):
                mod = getattr(blk, nm)
                mod.to(torch.float32)
                for p in mod.parameters():
                    p.requires_grad_(True)
            # Linear DropPath across adapted top blocks
            p_dp = dp_rates[idx_rel]
            blk.drop_path1 = DropPath(p_dp) if p_dp > 0 else nn.Identity()
            blk.drop_path2 = DropPath(p_dp) if p_dp > 0 else nn.Identity()
            # Inject LoRA on attention & MLP/SwiGLU projections
            blk.attn.qkv = LoRALinear(blk.attn.qkv, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
            blk.attn.proj = LoRALinear(blk.attn.proj, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
            self.n_lora += 2
            for mlp_attr in ("fc1", "fc1_g", "fc1_x", "fc2"):
                sub = getattr(blk.mlp, mlp_attr, None)
                if isinstance(sub, nn.Linear):
                    setattr(blk.mlp, mlp_attr, LoRALinear(sub, r=lora_r, alpha=lora_alpha, dropout=lora_dropout))
                    self.n_lora += 1

        # Trainable fp32 final norm and classifier head
        self.base.norm.to(torch.float32)
        for p in self.base.norm.parameters():
            p.requires_grad_(True)
        if getattr(self.base, "fc_norm", None) is not None and not isinstance(self.base.fc_norm, nn.Identity):
            self.base.fc_norm.to(torch.float32)
            for p in self.base.fc_norm.parameters():
                p.requires_grad_(True)
        self.base.head.to(torch.float32)
        for p in self.base.head.parameters():
            p.requires_grad_(True)

    def forward_features(self, x: torch.Tensor, attn_mask=None, is_causal: bool = False) -> torch.Tensor:
        use_amp = (x.device.type == "cuda")
        with torch.amp.autocast(x.device.type, dtype=torch.bfloat16, enabled=use_amp):
            with torch.no_grad():
                w_dtype = self.base.patch_embed.proj.weight.dtype
                if not use_amp and x.dtype != w_dtype:
                    x = x.to(w_dtype)
                x = self.base.patch_embed(x)
                x, rot_pos_embed = self.base._pos_embed(x)
                x = self.base.norm_pre(x)
                rope_mixed = getattr(self.base, "rope_mixed", False)
                for i in range(self.freeze_until):
                    rope_i = rot_pos_embed[i] if (rope_mixed and rot_pos_embed is not None) else rot_pos_embed
                    x = self.base.blocks[i](x, rope=rope_i, attn_mask=attn_mask, is_causal=is_causal)

            if self.training:
                x = x.detach().requires_grad_(True)

            rope_mixed = getattr(self.base, "rope_mixed", False)
            for i in range(self.freeze_until, self.depth):
                blk = self.base.blocks[i]
                rope_i = rot_pos_embed[i] if (rope_mixed and rot_pos_embed is not None) else rot_pos_embed
                if self.grad_ckpt and self.training:
                    def _run_blk(h, _b=blk, _r=rope_i):
                        return _b(h, rope=_r, attn_mask=attn_mask, is_causal=is_causal)
                    x = tuc.checkpoint(_run_blk, x, use_reentrant=False)
                else:
                    x = blk(x, rope=rope_i, attn_mask=attn_mask, is_causal=is_causal)

            x = self.base.norm(x)
            return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use_amp = (x.device.type == "cuda")
        with torch.amp.autocast(x.device.type, dtype=torch.bfloat16, enabled=use_amp):
            feat = self.forward_features(x)
            out = self.base.forward_head(feat)
        return out.float()

    def trainable_state_dict(self):
        trainable_names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if k in trainable_names}

    def load_trainable_state_dict(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        assert not unexpected, f"Unexpected keys in LoRA checkpoint: {unexpected}"
        trainable_names = {n for n, p in self.named_parameters() if p.requires_grad}
        missing_trainable = [k for k in missing if k in trainable_names]
        assert not missing_trainable, f"Missing trainable keys in LoRA checkpoint: {missing_trainable}"


class TrainableEMA:
    """Lightweight EMA tracking ONLY requires_grad=True parameters (48 MB vs 12.5 GiB full copy)."""

    def __init__(self, model: nn.Module, decay: float = 0.998):
        self.decay = float(decay)
        self.updates = 0
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.updates += 1
        d = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        one_minus_d = 1.0 - d
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(d).add_(p.detach(), alpha=one_minus_d)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(self.shadow[n])


def load_base_dinov3_bf16(model_name: str, hw: tuple, pretrained: bool = True, device: str = "cuda"):
    """Load large DINOv3 model into bfloat16 safely on <= 32 GiB CPU RAM runtimes."""
    try:
        mem_total_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except Exception:
        mem_total_gb = 64.0
    if mem_total_gb >= 38.0:
        base = timm.create_model(model_name, pretrained=pretrained, num_classes=5, img_size=hw)
        base = base.to(torch.bfloat16)
    else:
        torch.set_default_dtype(torch.bfloat16)
        try:
            base = timm.create_model(model_name, pretrained=pretrained, num_classes=5, img_size=hw)
        finally:
            torch.set_default_dtype(torch.float32)
    return base.to(device)


def create_lora_model(model_name: str, hw: tuple, freeze_until: int = 28, lora_r: int = 16,
                      lora_alpha: float = 32.0, lora_dropout: float = 0.05, drop_path: float = 0.1,
                      grad_ckpt: bool = True, pretrained: bool = True, device: str = "cuda"):
    base = load_base_dinov3_bf16(model_name, hw, pretrained=pretrained, device=device)
    model = TopKLoRADinoV3(
        base,
        freeze_until=freeze_until,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        drop_path=drop_path,
        grad_ckpt=grad_ckpt,
    )
    return model


def _view_zoom(x, factor=0.7):
    th = torch.zeros(x.shape[0], 2, 3, device=x.device)
    th[:, 0, 0], th[:, 1, 1] = factor, factor
    return F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), mode="bilinear",
                         padding_mode="border", align_corners=False)


def _view_centre_grey(x):
    H, W = x.shape[2:]
    m = torch.zeros(1, 1, H, W, device=x.device)
    m[..., H // 5:4 * H // 5, W // 5:4 * W // 5] = 1.0
    return x * m + 0.5 * (1.0 - m)


def _view_grid75(x):
    H, W = x.shape[2:]
    d = H // 3
    vis = int(round(d * 0.5))
    rr = torch.arange(H, device=x.device).view(H, 1)
    cc = torch.arange(W, device=x.device).view(1, W)
    m = (((rr + 7) % d) < vis) & (((cc + 11) % d) < vis)
    return torch.where(m, x, x.mean((2, 3), keepdim=True).expand_as(x))


HARD_VIEW_FNS = {
    "full": lambda x: x,
    "zoom70": lambda x: _view_zoom(x, 0.7),
    "centre_grey": _view_centre_grey,
    "dark_g35": lambda x: x.clamp_min(1e-6) ** 3.5,
    "grid75": _view_grid75,
}


@torch.no_grad()
def predict_array(model, arr, hw, bs=64, tta=True, device="cuda"):
    model.eval()
    out = []
    for i in range(0, len(arr), bs):
        x = normalize(to_gpu_batch(arr[i:i + bs], hw, device))
        p = model(x).float().softmax(-1)
        if tta:
            p = 0.5 * (p + model(torch.flip(x, dims=[3])).float().softmax(-1))
        out.append(p.cpu())
    return torch.cat(out).numpy()


def predict_paths(model, paths, hw, bs=64, tta=True, chunk=4096, device="cuda"):
    outs = []
    for i in range(0, len(paths), chunk):
        outs.append(predict_array(model, load_uint8(list(paths[i:i + chunk])), hw, bs=bs, tta=tta, device=device))
    return np.concatenate(outs)


@torch.no_grad()
def predict_hard_views(model, arr, hw, bs=64, device="cuda"):
    model.eval()
    res = {v: [] for v in HARD_VIEW_FNS}
    for i in range(0, len(arr), bs):
        x0 = to_gpu_batch(arr[i:i + bs], hw, device)
        for v, fn in HARD_VIEW_FNS.items():
            xv = fn(x0)
            p = 0.5 * (model(normalize(xv)).float().softmax(-1) + model(normalize(xv.flip(-1))).float().softmax(-1))
            res[v].append(p.cpu().numpy())
    return {v: np.concatenate(r) for v, r in res.items()}


def make_param_groups(model, wd=0.05):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "norm" not in n:
            decay.append(p)
        else:
            no_decay.append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def train_lora_fold(model, arr, labels, tr_idx, va_idx, hw=(144, 272), epochs=12, warmup_epochs=1,
                    bs=64, accum=1, lr=2e-4, wd=0.05, label_smoothing=0.1, mix_prob=0.8,
                    p_hud=0.70, ema_decay=0.998, seed=42, eval_epochs=None, device="cuda", log=print):
    assert bs % accum == 0, f"bs={bs} must be divisible by accum={accum}"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    labels = np.asarray(labels)
    tr_idx = np.asarray(tr_idx)
    va_idx = np.asarray(va_idx if va_idx is not None else [], dtype=int)
    has_val = len(va_idx) > 0
    if eval_epochs is None:
        eval_epochs = {epochs // 2 - 1, epochs - 1} if has_val else set()
    va_arr, va_labels = (arr[va_idx], labels[va_idx]) if has_val else (None, None)
    y_dev = torch.as_tensor(labels, device=device)

    ema = TrainableEMA(model, decay=ema_decay)
    opt = torch.optim.AdamW(make_param_groups(model, wd=wd), lr=lr)
    n_steps_ep = len(tr_idx) // bs
    total_steps = max(1, epochs * n_steps_ep)
    warm_steps = max(1, warmup_epochs * n_steps_ep)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (s + 1) / warm_steps if s < warm_steps else 0.5 * (1.0 + math.cos(math.pi * (s - warm_steps) / max(1, total_steps - warm_steps))),
    )
    aug = GPUAugment(p_hud=p_hud)
    cutmix_or_mixup = v2.RandomChoice([v2.MixUp(alpha=0.4, num_classes=5), v2.CutMix(alpha=1.0, num_classes=5)])
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    history = []
    for ep in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        perm = rng.permutation(tr_idx)
        for b in range(n_steps_ep):
            idx = np.sort(perm[b * bs:(b + 1) * bs])
            opt.zero_grad(set_to_none=True)
            for part in np.array_split(idx, accum):
                x = aug(to_gpu_batch(arr[part], hw, device))
                yb = y_dev[torch.as_tensor(part, device=device)]
                if rng.random() < mix_prob:
                    x, yt = cutmix_or_mixup(x, yb)
                else:
                    yt = F.one_hot(yb, 5).float()
                yt = yt * (1.0 - label_smoothing) + label_smoothing / 5.0
                loss = soft_ce(model(x), yt)
                (loss / accum).backward()
                tot += loss.item() * len(part)
            nn.utils.clip_grad_norm_(trainable_params, 2.0)
            opt.step()
            sched.step()
            ema.update(model)
            n += len(idx)

        f1 = acc = float("nan")
        if has_val and ep in eval_epochs:
            cur_state = model.trainable_state_dict()
            ema.apply_to(model)
            pv = predict_array(model, va_arr, hw, bs=bs, tta=False, device=device)
            f1 = float(f1_score(va_labels, pv.argmax(1), average="macro"))
            acc = float((pv.argmax(1) == va_labels).mean())
            if ep != epochs - 1:
                model.load_trainable_state_dict(cur_state)
        elif not has_val:
            pass
        history.append(dict(epoch=ep, loss=tot / max(1, n), val_f1=f1, val_acc=acc, secs=time.time() - t0))
        val_str = f"val_f1(ema) {f1:.4f} acc {acc:.4f}" if math.isfinite(f1) else "val skipped"
        log(f"  ep {ep:2d} loss {tot / max(1, n):.4f} {val_str} [{time.time() - t0:.0f}s]")

    # Ensure final model carries final EMA trainable weights
    ema.apply_to(model)
    return model, history
'''


def build_nb() -> Path:
    title = r"""
# CPE342 Task 4 (Round 43): Top-12-Block LoRA Fine-Tuning of `DINOv3-7B` @ 144×272 (Fold-0 Gate)

**Self-contained notebook** (needs only `t4_data.zip` on Drive — no `cpe342_code.zip` needed).

### Architecture & Design
- **Backbone:** `vit_7b_patch16_dinov3.lvd1689m` (`6,716 M` params, `depth=40`, `embed_dim=4096`, RoPE, 5 prefix tokens).
- **Geometry:** Native 2:1 **`144×272`** (`9×17` patches — `192×384` failed frozen dark gate `-0.0048`).
- **Top-12-Block LoRA (`blocks[28:40]`, `freeze_until = 28`):**
  - Blocks `0..27` (`70%` of model) are completely frozen inside `torch.no_grad()` → **0 backward pass, ~2× faster & ~50% less VRAM**.
  - Blocks `28..39` inject `LoRALinear(r=16, alpha=32, dropout=0.05)` into `attn.qkv`, `attn.proj`, `mlp.fc1_g`, `mlp.fc1_x`, `mlp.fc2` (`60` LoRA modules) + trainable `fp32` `norm1/norm2` + final `norm` + `head` (`~12.02 M` trainable params = **`0.18%`**, checkpoint **`~48 MB`**).
  - Lightweight `TrainableEMA` tracks only the `12.02 M` trainable params (`~48 MB` vs `12.5 GiB`).
- **Recipe (Fixed Before Run — Zero Test Leakage):**
  - `GPUAugment(p_hud=0.70)` + `MixUp(0.4)/CutMix(1.0) p=0.8` + `label_smoothing=0.1`
  - `bs=64` (`accum=1` on A100 `>= 30 GiB`, auto `accum=2` on L4 `24 GiB`), `lr=2e-4`, `wd=0.05` on 2D weights, `EPOCHS=12` (`1` warmup + `11` cosine), gradient checkpointing on blocks `28..39`.
- **Fold-0 Gate (`FOLDS = [0]`, `DO_FULL = False`):**
  - Alone gate: `val_f1 (TTA) >= 0.9960` (benchmark: `dinov3Hp_384` `0.9963`).
  - Saves `out/models/t4_dinov3_7b_lora12_fold0_lora.pt`, `out/preds/t4_dinov3_7b_lora12_fold0.npz`, and `out/r30_views/dinov3_7b_lora12_fold0.npz` → evaluate on laptop vs `v41` with `experiments/claude/ik_r43_t4_7b_lora_fold0.py`.
  - Only if Fold 0 **PASSES**, change to `FOLDS = [0, 1, 2, 3, 4]`, `DO_FULL = True` and re-run *Run all* (`fold 0` skipped automatically from Drive).

**Expected runtime:** A100 40/80 GB `~1.8–2.0 h` (Fold 0) | L4 24 GB `~4.5–5.5 h` (Fold 0). Cell 5 runs a live 4-step timing & VRAM benchmark first!
"""

    c1 = f"""
# ---- Settings -------------------------------------------------------------------------------------------------------------
TAG = "dinov3_7b_lora12"
MODEL = "vit_7b_patch16_dinov3.lvd1689m"
HW = (144, 272)
FREEZE_UNTIL = 28           # freeze blocks 0..27; adapt top 12 blocks (28..39)
LORA_R = 16
LORA_ALPHA = 32.0
LORA_DROPOUT = 0.05
DROP_PATH = 0.10
EPOCHS = 12
WARMUP_EPOCHS = 1
BS = 64
ACCUM = "auto"              # auto -> 1 on A100 (>=30 GiB VRAM), 2 on L4 (24 GiB)
LR = 2e-4
WD = 0.05
P_HUD = 0.70
EMA_DECAY = 0.998
FOLDS = [0]                 # Fold-0 Gate first; after PASS set [0, 1, 2, 3, 4] & DO_FULL = True
DO_FULL = False
MIN_ALONE = 0.9960          # Fold-0 gate alone floor (ViT-H+ 144 was 0.9960, ViT-H+ 384 was 0.9963)
AUTO_DISCONNECT = True      # disconnect runtime 60 s after completion or on error
{SHARED_SETUP_HELPER.strip()}
{SHARED_ERROR_HANDLER.strip()}
""".strip()

    c2 = r"""
# ---- Packages, GPU, ACCUM & Disk Check ------------------------------------------------------------------------------------
!pip -q install timm==1.0.30
import os, shutil, subprocess
import numpy, pandas, sklearn, timm, torch, torchvision

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
assert torch.cuda.is_available(), "CUDA required"
cap = torch.cuda.get_device_capability(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"GPU: {torch.cuda.get_device_name(0)} | cc={cap} | VRAM={vram_gb:.1f} GiB | timm={timm.__version__}")
assert cap[0] >= 8, "GPU lacks native bf16 (T4?): switch to A100 or L4"
assert vram_gb >= 20.0, f"Need >= 20 GiB VRAM for DINOv3-7B LoRA; got {vram_gb:.1f} GiB"
if ACCUM == "auto":
    ACCUM = 2 if vram_gb < 30.0 else 1
print(f"Effective batch: BS={BS} with ACCUM={ACCUM} (microbatch={BS // ACCUM})")

free_gb = shutil.disk_usage("/content").free / 2**30
print(f"/content disk free: {free_gb:.1f} GiB")
assert free_gb >= 32.0, f"Low /content disk ({free_gb:.1f} GiB): need >= 32 GiB"
"""

    c3 = f"""
# ---- Mount Drive & Unpack Task 4 Data -------------------------------------------------------------------------------------
{SHARED_DATA_UNPACK.strip()}
OUT_PREDS = D / "out" / "preds"
OUT_MODELS = D / "out" / "models"
OUT_VIEWS = D / "out" / "r30_views"
OUT_LOGS = D / "out" / "logs"
for d in (OUT_PREDS, OUT_MODELS, OUT_VIEWS, OUT_LOGS):
    d.mkdir(parents=True, exist_ok=True)
print("Drive out dirs ready under:", D / "out")
"""

    c4 = CORE_MODULE_CODE.strip()

    c5 = r"""
# ---- VRAM Smoke Test, Zero-Parity Check & Live 4-Step ETA Benchmark -------------------------------------------------------
train_df = pd.read_csv(DATA / "train.csv")
test_df = pd.read_csv(DATA / "test.csv")
train_df["label"] = train_df["label"].astype(int)
train_df["path"] = [str(DATA / "train" / f) for f in train_df["file_name"]]
test_df["path"] = [str(DATA / "test" / f) for f in test_df["file_name"]]

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
t_init = time.time()
smoke_model = create_lora_model(
    MODEL, HW, freeze_until=FREEZE_UNTIL, lora_r=LORA_R, lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT, drop_path=DROP_PATH, grad_ckpt=True, pretrained=True, device="cuda"
)
n_tot = sum(p.numel() for p in smoke_model.parameters())
n_tr = sum(p.numel() for p in smoke_model.parameters() if p.requires_grad)
print(f"Loaded DINOv3-7B LoRA ({smoke_model.n_lora} LoRALinear layers) in {(time.time()-t_init)/60:.1f}m | "
      f"total={n_tot/1e6:.1f}M, trainable={n_tr/1e6:.2f}M ({100*n_tr/n_tot:.3f}%)")
assert smoke_model.n_lora == 60, f"Expected 60 LoRA modules across top 12 blocks, got {smoke_model.n_lora}"

# 1. Gradient flow & VRAM peak on 1 full BS/ACCUM step + EMA update
smoke_model.train()
smoke_ema = TrainableEMA(smoke_model, decay=EMA_DECAY)
smoke_opt = torch.optim.AdamW(make_param_groups(smoke_model, wd=WD), lr=LR)
smoke_opt.zero_grad(set_to_none=True)
for _ in range(ACCUM):
    xb = normalize(torch.rand(BS // ACCUM, 3, *HW, device="cuda"))
    yt = F.one_hot(torch.randint(0, 5, (len(xb),), device="cuda"), 5).float()
    loss = soft_ce(smoke_model(xb), yt)
    (loss / ACCUM).backward()

# Verify frozen bottom blocks have NO gradients and top blocks / head have finite non-zero gradients
for nm, p in smoke_model.named_parameters():
    if not p.requires_grad:
        assert p.grad is None, f"Frozen parameter {nm} unexpectedly received gradient!"
assert smoke_model.base.blocks[0].attn.qkv.weight.grad is None
assert smoke_model.base.blocks[FREEZE_UNTIL - 1].mlp.fc2.weight.grad is None
g_a28 = smoke_model.base.blocks[FREEZE_UNTIL].attn.qkv.lora_A.grad
g_b39 = smoke_model.base.blocks[39].mlp.fc2.lora_B.grad
g_hd = smoke_model.base.head.weight.grad
assert g_a28 is not None and torch.isfinite(g_a28).all() and g_b39 is not None and g_hd is not None
assert g_b39.abs().max().item() > 0 and g_hd.abs().max().item() > 0, "Zero gradient in top LoRA / head!"

nn.utils.clip_grad_norm_([p for p in smoke_model.parameters() if p.requires_grad], 2.0)
smoke_opt.step()
smoke_ema.update(smoke_model)
peak_gb = torch.cuda.max_memory_allocated() / 2**30
total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"VRAM Smoke PASS: peak={peak_gb:.2f} / {total_gb:.1f} GiB | trainable state ~{4*n_tr/2**20:.1f} MB")
assert peak_gb < 0.92 * total_gb, f"VRAM too tight ({peak_gb:.1f}/{total_gb:.1f} GiB): raise ACCUM!"

# 2. Live 256-image timing benchmark (4 optimizer steps) for accurate epoch ETA
bench_arr = load_uint8(list(train_df.path.values[:256]))
torch.cuda.synchronize()
t_b = time.time()
train_lora_fold(
    smoke_model, bench_arr, train_df.label.values[:256], np.arange(256), [], hw=HW,
    epochs=1, warmup_epochs=1, bs=BS, accum=ACCUM, lr=LR, wd=WD, p_hud=P_HUD, ema_decay=EMA_DECAY,
    seed=0, device="cuda", log=lambda s: None
)
torch.cuda.synchronize()
sec_per_img = (time.time() - t_b) / 256.0
ep_min = sec_per_img * (len(train_df) * 0.8) / 60.0
fold_h = ep_min * EPOCHS / 60.0
print(f"Timing Benchmark: {sec_per_img*1000:.1f} ms/img -> est ~{ep_min:.1f} min/epoch, ~{fold_h:.2f} h per fold ({EPOCHS} ep)")
del smoke_model, smoke_ema, smoke_opt, bench_arr, xb, yt, loss
gc.collect()
torch.cuda.empty_cache()
"""

    c6 = r"""
# ---- Train Top-12 LoRA Folds (Fold-0 Gate first, resume-safe, atomic Drive saves) ------------------------------------------
y = train_df["label"].values.astype(np.int64)
test_names = np.array(test_df["file_name"].tolist(), dtype="U")
folds = [va for _, va in StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y)]
all_idx = np.arange(len(y))

print("Decoding all 31,546 training images once into RAM (~2.6 GB uint8)...", flush=True)
arr = load_uint8(list(train_df["path"]))
print(f"Decoded {arr.shape} ({arr.nbytes / 1e9:.2f} GB)", flush=True)

t_all = time.time()
for f in FOLDS:
    va = folds[f]
    tr = np.setdiff1d(all_idx, va, assume_unique=True)
    pred_out = OUT_PREDS / f"t4_{TAG}_fold{f}.npz"
    view_out = OUT_VIEWS / f"{TAG}_fold{f}.npz"
    ckpt_out = OUT_MODELS / f"t4_{TAG}_fold{f}_lora.pt"
    log_out = OUT_LOGS / f"t4_{TAG}_fold{f}_history.json"

    if pred_out.exists() and view_out.exists():
        zp, zv = np.load(pred_out), np.load(view_out)
        if np.array_equal(zp["val_idx"], va) and np.array_equal(zp["test_file_name"], test_names) and "dark_g35" in zv.files:
            f1_prev = f1_score(y[va], zp["val_probs"].argmax(1), average="macro")
            print(f"fold {f}: cached on Drive (TTA macro-F1={f1_prev:.4f}) -> skipping")
            continue

    t_f = time.time()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = create_lora_model(
        MODEL, HW, freeze_until=FREEZE_UNTIL, lora_r=LORA_R, lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT, drop_path=DROP_PATH, grad_ckpt=True, pretrained=True, device="cuda"
    )
    print(f"fold {f}: training {len(tr)} rows, val {len(va)} rows (EPOCHS={EPOCHS}, BS={BS}, ACCUM={ACCUM})...", flush=True)
    _, hist = train_lora_fold(
        model, arr, y, tr, va, hw=HW, epochs=EPOCHS, warmup_epochs=WARMUP_EPOCHS,
        bs=BS, accum=ACCUM, lr=LR, wd=WD, p_hud=P_HUD, ema_decay=EMA_DECAY, seed=42 + f, device="cuda"
    )

    # Save small (~48 MB) EMA LoRA weights + predictions + 5 hard views atomically via local /content staging
    tmp_ckpt = Path(f"/content/tmp_{ckpt_out.name}")
    torch.save(model.trainable_state_dict(), tmp_ckpt)
    shutil.copy2(tmp_ckpt, ckpt_out.with_suffix(".tmp.pt"))
    os.replace(ckpt_out.with_suffix(".tmp.pt"), ckpt_out)
    tmp_ckpt.unlink(missing_ok=True)

    vp = predict_array(model, arr[va], HW, bs=BS, tta=True, device="cuda")
    views = predict_hard_views(model, arr[va], HW, bs=BS, device="cuda")
    parity_diff = float(np.abs(vp - views["full"]).mean())
    assert parity_diff < 1e-3, f"val_probs vs r30 full view mismatch: {parity_diff}"
    tp = predict_paths(model, test_df["path"].values, HW, bs=BS, tta=True, device="cuda")

    tmp_p = Path(f"/content/tmp_{pred_out.name}")
    np.savez_compressed(tmp_p, val_idx=va, val_probs=vp.astype(np.float32),
                        test_probs=tp.astype(np.float32), test_file_name=test_names)
    shutil.copy2(tmp_p, pred_out.with_suffix(".tmp.npz"))
    os.replace(pred_out.with_suffix(".tmp.npz"), pred_out)
    tmp_p.unlink(missing_ok=True)

    tmp_v = Path(f"/content/tmp_{view_out.name}")
    np.savez_compressed(tmp_v, **{k: v.astype(np.float32) for k, v in views.items()})
    shutil.copy2(tmp_v, view_out.with_suffix(".tmp.npz"))
    os.replace(view_out.with_suffix(".tmp.npz"), view_out)
    tmp_v.unlink(missing_ok=True)

    log_out.write_text(json.dumps(hist, indent=2), encoding="utf-8")
    tta_f1 = float(f1_score(y[va], vp.argmax(1), average="macro"))
    tta_acc = float((vp.argmax(1) == y[va]).mean())
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"fold {f}: DONE in {(time.time()-t_f)/60:.1f}m | TTA macro-F1={tta_f1:.4f} acc={tta_acc:.4f} | "
          f"peak VRAM={peak:.2f} GiB | parity={parity_diff:.1e}", flush=True)
    if f == 0:
        gate_ok = (MIN_ALONE is None) or (tta_f1 >= MIN_ALONE)
        print(f"  -> Fold-0 MIN_ALONE gate ({MIN_ALONE}): {'PASS' if gate_ok else 'FAIL'} ({tta_f1:.4f})", flush=True)

    del model, vp, tp, views
    gc.collect()
    torch.cuda.empty_cache()

if DO_FULL:
    full_pred = OUT_PREDS / f"t4_{TAG}_full.npz"
    full_ckpt = OUT_MODELS / f"t4_{TAG}_full_lora.pt"
    full_log = OUT_LOGS / f"t4_{TAG}_full_history.json"
    if full_pred.exists():
        print(f"full refit: cached -> skipping")
    else:
        t_r = time.time()
        model = create_lora_model(
            MODEL, HW, freeze_until=FREEZE_UNTIL, lora_r=LORA_R, lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT, drop_path=DROP_PATH, grad_ckpt=True, pretrained=True, device="cuda"
        )
        print(f"full refit: training all {len(arr)} rows ({EPOCHS} epochs)...", flush=True)
        _, hist = train_lora_fold(
            model, arr, y, all_idx, [], hw=HW, epochs=EPOCHS, warmup_epochs=WARMUP_EPOCHS,
            bs=BS, accum=ACCUM, lr=LR, wd=WD, p_hud=P_HUD, ema_decay=EMA_DECAY, seed=142, device="cuda"
        )
        tmp_ckpt = Path(f"/content/tmp_{full_ckpt.name}")
        torch.save(model.trainable_state_dict(), tmp_ckpt)
        shutil.copy2(tmp_ckpt, full_ckpt.with_suffix(".tmp.pt"))
        os.replace(full_ckpt.with_suffix(".tmp.pt"), full_ckpt)
        tmp_ckpt.unlink(missing_ok=True)

        tp_full = predict_paths(model, test_df["path"].values, HW, bs=BS, tta=True, device="cuda")
        tmp_p = Path(f"/content/tmp_{full_pred.name}")
        np.savez_compressed(tmp_p, test_probs=tp_full.astype(np.float32), test_file_name=test_names)
        shutil.copy2(tmp_p, full_pred.with_suffix(".tmp.npz"))
        os.replace(full_pred.with_suffix(".tmp.npz"), full_pred)
        tmp_p.unlink(missing_ok=True)

        full_log.write_text(json.dumps(hist, indent=2), encoding="utf-8")
        print(f"full refit: DONE in {(time.time()-t_r)/60:.1f}m", flush=True)
        del model, tp_full
        gc.collect()
        torch.cuda.empty_cache()

print(f"All requested runs finished in {(time.time()-t_all)/3600:.2f} h")
"""

    c7 = r"""
# ---- Saved Files on Drive & Auto-Disconnect -------------------------------------------------------------------------------
for sub in ("preds", "r30_views", "models", "logs"):
    for f in sorted((D / "out" / sub).glob(f"*{TAG}*")):
        print(f"  out/{sub}/{f.name:40s} {f.stat().st_size / 2**20:6.2f} MB")

if AUTO_DISCONNECT:
    import time
    print(f"done -> ending runtime in 60 s [{time.strftime('%H:%M')} UTC]", flush=True)
    time.sleep(60)
    from google.colab import runtime
    runtime.unassign()
"""

    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell(title.strip("\n")))
    for src in (c1, c2, c3, c4, c5, c6, c7):
        nb.cells.append(nbf.v4.new_code_cell(src.strip("\n")))
    out = Path("colab/t4_colab_dinov3_7b_lora.ipynb")
    nbf.write(nb, out)
    return out


def run_local_smoke(nb_path: Path):
    import json
    import tempfile
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import timm

    # 1. Verify AST of every code cell in generated notebook
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            src = "\n".join(l for l in "".join(cell["source"]).splitlines() if not l.lstrip().startswith("!"))
            ast.parse(src)
    print(f"AST PASS: {nb_path} ({len(nb['cells'])} cells)")

    # 2. Verify exact 7B LoRA injection on meta device (depth=40, freeze_until=28 -> 60 LoRA modules, ~12.02M trainable)
    with torch.device("meta"):
        m7b = timm.create_model("vit_7b_patch16_dinov3.lvd1689m", pretrained=False, num_classes=5, img_size=(144, 272))
    ns = {}
    exec(CORE_MODULE_CODE, ns)
    lora7b = ns["TopKLoRADinoV3"](m7b, freeze_until=28, lora_r=16, lora_alpha=32.0, lora_dropout=0.05, drop_path=0.1, grad_ckpt=True)
    n_tot_7b = sum(p.numel() for p in lora7b.parameters())
    n_tr_7b = sum(p.numel() for p in lora7b.parameters() if p.requires_grad)
    assert lora7b.n_lora == 60, f"Expected 60 LoRA modules on 7B, got {lora7b.n_lora}"
    print(f"7B Meta PASS: n_lora={lora7b.n_lora}, total={n_tot_7b:,}, trainable={n_tr_7b:,} ({100*n_tr_7b/n_tot_7b:.3f}%)")
    del m7b, lora7b

    # 3. Live GPU execution smoke test using exact notebook core code on RTX 4070 with vit_small_patch16_dinov3
    assert torch.cuda.is_available(), "CUDA required for --smoke-local"
    hw = (144, 272)
    base_ref = timm.create_model("vit_small_patch16_dinov3.lvd1689m", pretrained=False, num_classes=5, img_size=hw).to(torch.bfloat16).cuda().eval()
    base_cp = timm.create_model("vit_small_patch16_dinov3.lvd1689m", pretrained=False, num_classes=5, img_size=hw).to(torch.bfloat16).cuda().eval()
    base_cp.load_state_dict(base_ref.state_dict(), strict=True)

    lora_m = ns["TopKLoRADinoV3"](base_cp, freeze_until=8, lora_r=8, lora_alpha=16.0, lora_dropout=0.05, drop_path=0.1, grad_ckpt=True).eval()
    x_chk = ns["normalize"](torch.rand(2, 3, *hw, device="cuda"))
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            feat_ref = base_ref.forward_features(x_chk).float()
        feat_lora = lora_m.forward_features(x_chk).float()
        init_diff = float((feat_ref - feat_lora).abs().max().item())
    assert init_diff < 5e-3, f"Init feature parity diff too large: {init_diff}"
    print(f"GPU Smoke Init Feature Parity PASS: max|diff|={init_diff:.2e} (top norm fp32 vs bf16)")

    # Verify gradients, TrainableEMA, 2-epoch mini training, checkpoint reload, and 5-view output
    u8 = np.random.randint(0, 256, (32, 120, 230, 3), dtype=np.uint8)
    y_s = np.arange(32) % 5
    lora_m, hist = ns["train_lora_fold"](
        lora_m, u8, y_s, np.arange(24), np.arange(24, 32), hw=hw,
        epochs=2, warmup_epochs=1, bs=8, accum=2, lr=2e-4, p_hud=0.70, seed=0,
        eval_epochs={1}, device="cuda", log=lambda m: print("  ", m)
    )
    assert len(hist) == 2 and all(math.isfinite(h["loss"]) for h in hist)
    # After training, lora_B should be non-zero
    assert lora_m.base.blocks[11].mlp.fc2.lora_B.abs().max().item() > 0, "Top LoRA_B stayed zero after training!"
    # Check frozen layers never received grad
    assert lora_m.base.blocks[0].attn.qkv.weight.grad is None
    assert lora_m.base.blocks[7].mlp.fc2.weight.grad is None

    # Save & reload LoRA trainable state onto a fresh base -> 0.0 bit parity
    vp = ns["predict_array"](lora_m, u8[24:], hw, bs=8, tta=True, device="cuda")
    hv = ns["predict_hard_views"](lora_m, u8[24:], hw, bs=8, device="cuda")
    assert vp.shape == (8, 5) and np.isfinite(vp).all()
    assert abs(vp - hv["full"]).mean() < 1e-5
    assert set(hv.keys()) == {"full", "zoom70", "centre_grey", "dark_g35", "grid75"}

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "test_lora.pt"
        torch.save(lora_m.trainable_state_dict(), p)
        loaded_sd = torch.load(p, map_location="cuda")

    base_fresh = timm.create_model("vit_small_patch16_dinov3.lvd1689m", pretrained=False, num_classes=5, img_size=hw).to(torch.bfloat16).cuda()
    base_fresh.load_state_dict(base_ref.state_dict(), strict=True)
    lora_reloaded = ns["TopKLoRADinoV3"](base_fresh, freeze_until=8, lora_r=8, lora_alpha=16.0, lora_dropout=0.05, drop_path=0.1, grad_ckpt=True).eval()
    lora_reloaded.load_trainable_state_dict(loaded_sd)
    vp_re = ns["predict_array"](lora_reloaded, u8[24:], hw, bs=8, tta=True, device="cuda")
    reload_diff = float(np.abs(vp - vp_re).max())
    assert reload_diff == 0.0, f"Reload mismatch: {reload_diff}"
    print(f"GPU Smoke Train + Save/Reload + 5-View PASS | reload max|diff|={reload_diff:.1e} | peak VRAM={torch.cuda.max_memory_allocated()/2**30:.2f} GiB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-local", action="store_true", help="Run AST, 7B meta check, and local RTX 4070 GPU smoke test")
    args = ap.parse_args()
    out = build_nb()
    print(f"Generated {out}")
    if args.smoke_local:
        run_local_smoke(out)


if __name__ == "__main__":
    main()
