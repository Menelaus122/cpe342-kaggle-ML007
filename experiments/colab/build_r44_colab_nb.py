"""Generate the Round 44 Colab notebook:
  colab/t4_colab_dinov3_7b_384_feat.ipynb   (DINOv3-7B @ 192x384 frozen 4-block multi-layer features + Linear Screen)

Stage 1 (default, STAGE2 = False): extract only the train `full` view at 192x384 (12x24 patches), then run the id_r42
linear screen against DINOv3-7B @ 144x272 with the same code, folds and dark mask, plus the 144+384 concatenation.
The go/no-go rule for stage 2 is fixed in this file before any 192x384 result exists (see GATE below).

Stage 2 (STAGE2 = True, only if stage 1 says GO): the same notebook extracts zoom70 / centre_grey / dark_lc and the
test full view, skipping the cached train_full.npy.

Test images are never read in stage 1.

Run from repo root:
    python colab/build_r44_colab_nb.py
"""
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_r43_colab_nbs import (SHARED_DATA_UNPACK, SHARED_ERROR_HANDLER,  # noqa: E402
                                 SHARED_SETUP_HELPER, make_nb)

TITLE = r"""
# CPE342 Task 4 (Round 44, stage 1): DINOv3-7B @ 192×384 frozen features + linear screen against 7B @ 144×272

**Self-contained notebook** (needs only `t4_data.zip` on Drive; reads `out/preds/dinov3_7b_144x272/` for the reference
screen if it is still there).

### Stage 1 (default: `STAGE2 = False`)
- **Backbone:** `vit_7b_patch16_dinov3.lvd1689m`, bf16, frozen, h-flip average; same 4-block feature layout as round 43
  (`mid_patch, last_patch, last_cls, last_std` = `16,384-d`), only the input size changes: **192×384** (`12×24` patches,
  288 tokens vs 153 at 144×272).
- **Extracts only the train `full` view** (31,546 images) -> `out/preds/dinov3_7b_192x384/train_full.npy` (1.03 GB) and
  `train_full_pooled.npy` (`last_patch`, 258 MB). **Test images are not read.**
- **Linear screen** (id_r42 protocol: 5-fold L2 logistic regression `C=0.03` on standardised `last_patch`, balanced macro-F1
  with the `pi^-0.75` rule, Q10 brightness / contrast dark mask), three rows in the same run:
  - `R` = 7B @ 144×272 (re-run from Drive; falls back to the logged `0.9737 / 0.9378` if the file is gone)
  - `A` = 7B @ 192×384 alone
  - `B` = 144×272 ⊕ 192×384 concatenated (8,192-d; skipped if the 144×272 file is gone)
- **Gate, fixed before any 192×384 result:** **GO to stage 2** if `A` or `B` beats `R` by **≥ +0.005 on dark** with
  **full ≥ −0.003** (the id_r42 rule, now against 7B @ 144×272 instead of ViT-H+). Otherwise stop: the 7B family is already
  in `P_7b` and a smaller screen gain is unlikely to survive the nested check inside v41.
- Logs the measured throughput and the projected stage-2 time / CU (A100 ≈ 6 CU/h as measured on 2026-11-21).

### Stage 2 (`STAGE2 = True`, only after a GO)
Extracts `zoom70`, `centre_grey`, `dark_lc` (train) and `test_full` into the same folder; the cached `train_full.npy`
is skipped. Then the probe and the nested judge run on the laptop.

**Colab runtime:** **A100** (40 / 80 GB). L4 works (BS drops to 32) but is slower. Not a T4 (no bf16).
"""

C1_SETTINGS = r'''
# ---- Settings -------------------------------------------------------------------------------------------------------------
MODEL = "vit_7b_patch16_dinov3.lvd1689m"
HW = (192, 384)           # 12 x 24 patches of 16 px (native 2:1, same geometry as dinov3Hp_384)
TAG = "dinov3_7b_192x384"
REF_TAG = "dinov3_7b_144x272"
BS = 64                   # drops to 32 automatically below 38 GiB VRAM
STAGE2 = False            # False = stage 1 (train full + screen); True = remaining 3 train views + test (after a GO only)
RUN_LINEAR_SCREEN = not STAGE2
AUTO_DISCONNECT = True    # end runtime 60 s after completion or on error to save Colab CU
CU_PER_HOUR_A100 = 6.0    # measured 2026-11-21 (240 -> 190.17 CU over 8.3 h)
'''.strip() + "\n" + SHARED_SETUP_HELPER.strip() + "\n" + SHARED_ERROR_HANDLER.strip()

C2_ENV = r'''
# ---- Packages, GPU & Disk check -------------------------------------------------------------------------------------------
!pip -q install timm==1.0.30
import os, shutil
import numpy, pandas, sklearn, timm, torch, torchvision

print("torch", torch.__version__, "| timm", timm.__version__)
assert timm.__version__ == "1.0.30" or "vit_7b_patch16_dinov3" in " ".join(timm.list_models("*dinov3*")), "DINOv3-7B missing"
assert torch.cuda.is_available(), "CUDA required"
cap = torch.cuda.get_device_capability(0)
gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"GPU: {gpu_name} | cc={cap} | VRAM={vram_gb:.1f} GiB")
assert cap[0] >= 8, "GPU lacks native bf16 (T4?): switch runtime to A100 or L4"
assert vram_gb >= 20.0, f"Need >= 20 GiB VRAM for DINOv3-7B bf16 (~13.5 GiB weights + activations); got {vram_gb:.1f} GiB"
if vram_gb < 38.0:
    BS = min(BS, 32)
print("Batch size:", BS)

free_gb = shutil.disk_usage("/content").free / 2**30
print(f"/content disk free: {free_gb:.1f} GiB")
assert free_gb >= 32.0, f"Low /content disk space ({free_gb:.1f} GiB): DINOv3-7B weights + data need >= 32 GiB"
'''

C3_DATA = r'''
# ---- Mount Drive & unpack task4 images ------------------------------------------------------------------------------------
'''.strip() + "\n" + SHARED_DATA_UNPACK.strip() + "\n" + r'''
OUT_DIR = D / "out" / "preds" / TAG
REF_DIR = D / "out" / "preds" / REF_TAG
LOG_DIR = D / "out" / "logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
print("Output dir on Drive:", OUT_DIR)
print("Reference (144x272) dir on Drive:", REF_DIR, "| exists:", REF_DIR.exists())
'''.strip()

C4_MODEL = r'''
# ---- Load DINOv3-7B -> bfloat16 -> CUDA + pooling parity sanity check (identical to round 43 except HW) --------------------
import gc, time, json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
import timm

NATIVE_HW = (120, 230)

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

def zoom(x, factor=0.7):
    th = torch.zeros(x.shape[0], 2, 3, device=x.device)
    th[:, 0, 0], th[:, 1, 1] = factor, factor
    return F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False),
                         mode="bilinear", padding_mode="border", align_corners=False)

def centre_grey(x):
    h, w = x.shape[2:]
    m = torch.zeros(1, 1, h, w, device=x.device)
    m[..., h // 5:4 * h // 5, w // 5:4 * w // 5] = 1.0
    return x * m + 0.5 * (1.0 - m)

def dark_lc(x):
    xg = x.clamp_min(1e-6) ** 1.8
    m = xg.mean(dim=(2, 3), keepdim=True)
    return (m + 0.55 * (xg - m)).clamp(0.0, 1.0)

VIEW_FNS = {
    "full": lambda x: x,
    "zoom70": lambda x: zoom(x, 0.7),
    "centre_grey": centre_grey,
    "dark_lc": dark_lc,
}

mem_total_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
print(f"System CPU RAM: {mem_total_gb:.1f} GiB")
t_load = time.time()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

if mem_total_gb >= 38.0:
    model = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=HW)
    model = model.to(torch.bfloat16)
else:
    torch.set_default_dtype(torch.bfloat16)   # low-CPU-RAM fallback
    try:
        model = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=HW)
    finally:
        torch.set_default_dtype(torch.float32)

gc.collect()
model = model.cuda().eval()
cfg = timm.data.resolve_model_data_config(model)
MEAN = torch.tensor(cfg["mean"], device="cuda").view(1, 3, 1, 1)
STD = torch.tensor(cfg["std"], device="cuda").view(1, 3, 1, 1)
print(f"Loaded {MODEL} to CUDA bf16 in {(time.time()-t_load)/60:.1f}m | embed_dim={model.embed_dim} | "
      f"allocated={torch.cuda.memory_allocated()/2**30:.2f} GiB")

@torch.no_grad()
def extract_dinov3_batch(x_01):
    x = ((x_01 - MEAN) / STD).to(torch.bfloat16)
    def _fwd(z):
        _, inters = model.forward_intermediates(z, indices=4, return_prefix_tokens=True, norm=True, output_fmt="NLC")
        mid_patch = torch.stack([inters[k][0].float().mean(1) for k in (0, 1, 2)], dim=0).mean(0)
        last_patch = inters[3][0].float().mean(1)
        last_cls = inters[3][1][:, 0].float()
        last_std = inters[3][0].float().std(1, unbiased=False)
        return torch.cat([mid_patch, last_patch, last_cls, last_std], dim=1)
    return 0.5 * (_fwd(x) + _fwd(torch.flip(x, dims=[3])))

with torch.no_grad():
    x_test = torch.rand(BS, 3, *HW, device="cuda")
    feat_ml = extract_dinov3_batch(x_test)
    assert feat_ml.shape == (BS, 4 * model.embed_dim) and torch.isfinite(feat_ml).all()
    _, inters_chk = model.forward_intermediates(((x_test[:2] - MEAN) / STD).to(torch.bfloat16), indices=1,
                                                return_prefix_tokens=True, norm=True, output_fmt="NLC")
    n_patch = inters_chk[0][0].shape[1]
    assert n_patch == (HW[0] // 16) * (HW[1] // 16), f"expected {(HW[0]//16)*(HW[1]//16)} patch tokens, got {n_patch}"
    x_small = ((x_test[:4] - MEAN) / STD).to(torch.bfloat16)
    f_direct = 0.5 * (model(x_small).float() + model(torch.flip(x_small, dims=[3])).float())
    f_slice = extract_dinov3_batch(x_test[:4])[:, model.embed_dim:2 * model.embed_dim]
    parity_diff = float((f_direct - f_slice).abs().max().item())
    assert parity_diff < 2e-2, f"Pooling parity diff too large: {parity_diff}"

peak_gb = torch.cuda.max_memory_allocated() / 2**30
total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"Sanity PASS: batch={tuple(feat_ml.shape)} | patch tokens={n_patch} | last_patch vs model(x) max|diff|={parity_diff:.2e} | "
      f"peak VRAM={peak_gb:.2f} / {total_gb:.1f} GiB")
assert peak_gb < 0.90 * total_gb, f"VRAM too tight ({peak_gb:.1f} / {total_gb:.1f} GiB): reduce BS"
del x_test, x_small, feat_ml, f_direct, f_slice, inters_chk
torch.cuda.empty_cache()
'''

C5_EXTRACT = r'''
# ---- Resume-safe per-view extraction (write local -> verify -> copy atomically to Drive) ----------------------------------
train_df = pd.read_csv(DATA / "train.csv")
train_paths = [str(DATA / "train" / f) for f in train_df["file_name"]]
DIM = 4 * model.embed_dim  # 16,384
TRAIN_VIEWS = ["full"] + (["zoom70", "centre_grey", "dark_lc"] if STAGE2 else [])
TIMINGS = {}

def valid_npy(path: Path, expected_shape: tuple) -> bool:
    if not path.exists():
        return False
    try:
        arr = np.load(path, mmap_mode="r")
        ok = (arr.shape == expected_shape) and (arr.dtype == np.float16)
        del arr
        return ok
    except Exception:
        return False

def extract_view_npy(paths, view_name: str, out_drive_path: Path):
    expected_shape = (len(paths), DIM)
    if valid_npy(out_drive_path, expected_shape):
        print(f"[{view_name}] valid cache on Drive ({out_drive_path.name}, {expected_shape}) -> skipping")
        return
    local_tmp = Path(f"/content/tmp_{out_drive_path.name}")
    drive_tmp = out_drive_path.with_suffix(".tmp.npy")
    t0 = time.time()
    buf = np.empty(expected_shape, dtype=np.float16)
    with torch.no_grad():
        for i in range(0, len(paths), 4096):
            chunk = load_uint8(paths[i:i + 4096])
            for j in range(0, len(chunk), BS):
                x0 = torch.from_numpy(chunk[j:j + BS]).cuda().permute(0, 3, 1, 2).float().div_(255)
                xv = F.interpolate(VIEW_FNS[view_name](x0), size=HW, mode="bicubic", align_corners=False).clamp_(0, 1)
                f = extract_dinov3_batch(xv)
                assert torch.isfinite(f).all(), f"Non-finite features in {view_name} rows {i+j}:{i+j+len(x0)}"
                buf[i + j:i + j + len(x0)] = f.cpu().numpy().astype(np.float16)
            del chunk
            done = min(i + 4096, len(paths))
            el = (time.time() - t0) / 60
            print(f"  [{view_name}] {done}/{len(paths)} ({el:.1f}m elapsed, ~{el * (len(paths) - done) / done:.1f}m left)", flush=True)
    t_gpu = time.time() - t0
    TIMINGS[out_drive_path.name] = {"n": len(paths), "seconds": round(t_gpu, 1), "img_per_s": round(len(paths) / t_gpu, 2)}

    np.save(local_tmp, buf)
    if view_name == "full" and len(paths) == len(train_paths):
        pooled_local = Path("/content/tmp_train_full_pooled.npy")
        pooled_drive = OUT_DIR / "train_full_pooled.npy"
        np.save(pooled_local, buf[:, model.embed_dim:2 * model.embed_dim].copy())
        shutil.copy2(pooled_local, pooled_drive.with_suffix(".tmp.npy"))
        os.replace(pooled_drive.with_suffix(".tmp.npy"), pooled_drive)
        pooled_local.unlink(missing_ok=True)
        print(f"  [train_full_pooled] saved {pooled_drive.name} ({pooled_drive.stat().st_size / 2**20:.1f} MB)")
    del buf
    gc.collect()
    shutil.copy2(local_tmp, drive_tmp)
    os.replace(drive_tmp, out_drive_path)
    local_tmp.unlink(missing_ok=True)
    print(f"[{view_name}] DONE -> {out_drive_path.name} ({out_drive_path.stat().st_size / 2**20:.1f} MB) in {t_gpu/60:.1f}m "
          f"({len(paths) / t_gpu:.1f} img/s incl. image loading)")

for v in TRAIN_VIEWS:
    extract_view_npy(train_paths, v, OUT_DIR / f"train_{v}.npy")

pooled_drive = OUT_DIR / "train_full_pooled.npy"
if not valid_npy(pooled_drive, (len(train_paths), model.embed_dim)):
    full_mm = np.load(OUT_DIR / "train_full.npy", mmap_mode="r")
    pooled = np.array(full_mm[:, model.embed_dim:2 * model.embed_dim], dtype=np.float16)
    del full_mm
    np.save(pooled_drive.with_suffix(".tmp.npy"), pooled)
    os.replace(pooled_drive.with_suffix(".tmp.npy"), pooled_drive)
    del pooled
    print(f"[train_full_pooled] materialized ({pooled_drive.stat().st_size / 2**20:.1f} MB)")

if STAGE2:
    # Test images are read only here, in stage 2 (after the stage-1 GO on training images alone)
    test_df = pd.read_csv(DATA / "test.csv")
    test_paths = [str(DATA / "test" / f) for f in test_df["file_name"]]
    extract_view_npy(test_paths, "full", OUT_DIR / "test_full.npy")
    np.save(OUT_DIR / "test_file_name.npy", np.array(test_df["file_name"].tolist(), dtype="U"))

meta_path = OUT_DIR / "meta.json"
meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
meta.update({
    "model": MODEL, "hw": list(HW), "dim": DIM, "subblock_dims": [model.embed_dim] * 4,
    "subblock_names": ["mid_patch", "last_patch", "last_cls", "last_std"],
    "train_views": sorted(set(meta.get("train_views", [])) | set(TRAIN_VIEWS)),
    "extract_test": bool(meta.get("extract_test", False) or STAGE2),
    "n_train": len(train_paths), "gpu": gpu_name, "batch_size": BS,
})
meta.setdefault("timings", {}).update(TIMINGS)
meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

# Projected stage-2 cost from the measured throughput (3 train views + test = 3 * 31,546 + 25,889 images)
if "train_full.npy" in TIMINGS:
    ips = TIMINGS["train_full.npy"]["img_per_s"]
    n2 = 3 * len(train_paths) + 25889
    h2 = n2 / ips / 3600 + 0.25   # + ~15 min for model load and Drive copies
    STAGE2_PROJ = f"projected stage 2: {n2:,} images at {ips:.1f} img/s -> ~{h2:.1f} h on {gpu_name} (~{h2 * CU_PER_HOUR_A100:.0f} CU at A100 rate)"
else:
    STAGE2_PROJ = "projected stage 2: n/a (train_full.npy was cached, no throughput measured in this run)"
print(STAGE2_PROJ)

del model
gc.collect()
torch.cuda.empty_cache()
print("Feature caches saved to Drive.")
'''

C6_SCREEN = r'''
# ---- id_r42 linear screen on training images only: R (7B@144) vs A (7B@192x384) vs B (concat) + the fixed gate ----------
if not RUN_LINEAR_SCREEN:
    print("RUN_LINEAR_SCREEN = False -> skipped")
else:
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold

    torch.set_num_threads(min(8, os.cpu_count() or 4))
    y = train_df["label"].astype(np.int64).values
    folds = [va for _, va in StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y)]
    fo = np.zeros(len(y), dtype=int)
    for f, va in enumerate(folds):
        fo[va] = f

    grey = np.empty((len(train_paths), 2), dtype=np.float32)
    w_rgb = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    for i in range(0, len(train_paths), 4096):
        g = (load_uint8(train_paths[i:i + 4096]).astype(np.float32) / 255.0) @ w_rgb
        grey[i:i + len(g), 0] = g.mean(axis=(1, 2))
        grey[i:i + len(g), 1] = g.std(axis=(1, 2))
    dark = (grey[:, 0] <= np.quantile(grey[:, 0], 0.10)) | (grey[:, 1] <= np.quantile(grey[:, 1], 0.10))
    print(f"dark / low-contrast training frames: {dark.sum()} of {len(y)}")

    def fit_probe_lbfgs(A_np, Y_np, c_probe=0.03, iters=300):
        A_t, Y_t = torch.from_numpy(A_np), torch.from_numpy(Y_np)
        W = torch.zeros(A_t.shape[1], 5, requires_grad=True)
        b = torch.zeros(5, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, history_size=20,
                                line_search_fn="strong_wolfe", tolerance_grad=1e-6)
        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(A_t @ W + b, Y_t, reduction="sum") + 0.5 / c_probe * (W ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        return W.detach(), b.detach()

    pi = np.bincount(y, minlength=5) / len(y)
    wb = (0.2 / pi)[y]
    bal = lambda pred, m: f1_score(y[m], pred[m], average="macro", sample_weight=wb[m])
    rule = lambda p: (p * pi ** -0.75).argmax(1)
    allm = np.ones(len(y), dtype=bool)

    def screen(X, name):
        t0 = time.time()
        oof = np.zeros((len(y), 5), dtype=np.float64)
        for f, va in enumerate(folds):
            tr = np.where(fo != f)[0]
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            W, b = fit_probe_lbfgs((X[tr] - mu) / sd, y[tr])
            oof[va] = (torch.from_numpy((X[va] - mu) / sd) @ W + b).softmax(1).numpy()
        pr = rule(oof)
        res = {"full": bal(pr, allm), "dark": bal(pr, dark), "plain": f1_score(y, oof.argmax(1), average="macro"),
               "err_full": int((pr != y).sum()), "err_dark": int((pr[dark] != y[dark]).sum())}
        print(f"  {name:34s}: full {res['full']:.4f} ({res['err_full']} err) | dark {res['dark']:.4f} ({res['err_dark']} err) | "
              f"plain {res['plain']:.4f}  [{time.time()-t0:.0f}s]", flush=True)
        return res

    X384 = np.load(OUT_DIR / "train_full_pooled.npy").astype(np.float32)
    X144 = None
    if (REF_DIR / "train_full_pooled.npy").exists():
        X144 = np.load(REF_DIR / "train_full_pooled.npy").astype(np.float32)
    elif (REF_DIR / "train_full.npy").exists():
        X144 = np.array(np.load(REF_DIR / "train_full.npy", mmap_mode="r")[:, 4096:8192], dtype=np.float32)
    if X144 is not None:
        assert X144.shape == X384.shape, (X144.shape, X384.shape)

    t_scr = time.time()
    rows = {}
    if X144 is not None:
        rows["R"] = screen(X144, "R  7B @ 144x272 (4096-d, re-run)")
    else:
        rows["R"] = {"full": 0.9737, "dark": 0.9378, "plain": 0.9732, "err_full": -1, "err_dark": -1}
        print("  R  7B @ 144x272: file gone from Drive -> logged numbers 0.9737 / 0.9378 (round 43); B skipped")
    rows["A"] = screen(X384, "A  7B @ 192x384 (4096-d)")
    if X144 is not None:
        rows["B"] = screen(np.hstack([X144, X384]), "B  144x272 + 192x384 (8192-d)")
        del X144
    del X384

    # Gate fixed in colab/build_r44_colab_nb.py before any 192x384 result: id_r42 rule against R
    GATE_DARK, GATE_FULL = 0.005, -0.003
    lines = [f"DINOv3-7B @ 192x384 linear screen ({time.strftime('%Y-%m-%d %H:%M UTC')}, {time.time()-t_scr:.0f}s, "
             f"{int(dark.sum())} dark frames):"]
    for k, nm in [("R", "7B @ 144x272"), ("A", "7B @ 192x384"), ("B", "144x272 + 192x384")]:
        if k in rows:
            r = rows[k]
            lines.append(f"  {k} {nm:20s}: full {r['full']:.4f} ({r['err_full']} err) | dark {r['dark']:.4f} ({r['err_dark']} err) | plain {r['plain']:.4f}")
    go = []
    for k in ("A", "B"):
        if k in rows:
            dfu, dda = rows[k]["full"] - rows["R"]["full"], rows[k]["dark"] - rows["R"]["dark"]
            ok = dda >= GATE_DARK and dfu >= GATE_FULL
            go.append(ok)
            lines.append(f"  {k} minus R: full {dfu:+.4f}, dark {dda:+.4f} -> {'passes' if ok else 'fails'} "
                         f"(dark >= +{GATE_DARK:.3f} and full >= {GATE_FULL:+.3f})")
    verdict = "GO to stage 2 (set STAGE2 = True and re-run)" if any(go) else "STOP: no >= +0.005 dark gain over 7B @ 144x272"
    lines += [f"  gate: {verdict}", f"  {STAGE2_PROJ}"]
    report_txt = "\n".join(lines)
    print("\n" + report_txt)
    (LOG_DIR / f"{TAG}_screen.log").write_text(report_txt + "\n", encoding="utf-8")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["screen"] = {k: v for k, v in rows.items()}
    meta["gate"] = verdict
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
'''

C7_SUMMARY = r'''
# ---- Summary of files on Drive & auto-disconnect --------------------------------------------------------------------------
print("Saved files in", OUT_DIR)
for p in sorted(OUT_DIR.iterdir()):
    print(f"  {p.name:28s} {p.stat().st_size / 2**20:8.1f} MB")
scr_log = LOG_DIR / f"{TAG}_screen.log"
if scr_log.exists():
    print("Screen log:", scr_log)
    print(scr_log.read_text(encoding="utf-8"))

if AUTO_DISCONNECT:
    import time
    print(f"done -> ending runtime in 60 s [{time.strftime('%H:%M')} UTC]", flush=True)
    time.sleep(60)
    from google.colab import runtime
    runtime.unassign()
'''


def build() -> Path:
    nb = make_nb(TITLE, [C1_SETTINGS, C2_ENV, C3_DATA, C4_MODEL, C5_EXTRACT, C6_SCREEN, C7_SUMMARY])
    out = Path(__file__).resolve().parent / "t4_colab_dinov3_7b_384_feat.ipynb"
    nbf.write(nb, out)
    return out


if __name__ == "__main__":
    print("Generated", build())
