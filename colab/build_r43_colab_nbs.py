"""Generate self-contained Round 43 Colab notebooks:
  1. colab/t4_colab_dinov3_7b_feat.ipynb   (DINOv3-7B @ 144x272 frozen 4-block multi-layer features + Linear Screen)
  2. colab/t4_colab_multires_tta.ipynb     (Zero-Crop RoPE Multi-Res TTA for dinov3Hp_384 @ 160x304, 192x384, 224x432)

Run from repo root:
    python colab/build_r43_colab_nbs.py
"""
from pathlib import Path
import nbformat as nbf


def make_nb(title_md: str, code_cells: list[str]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell(title_md.strip("\n")))
    for src in code_cells:
        nb.cells.append(nbf.v4.new_code_cell(src.strip("\n")))
    return nb


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


def build_dinov3_7b_nb() -> Path:
    title = r"""
# CPE342 Task 4 (Round 43): DINOv3-7B (`vit_7b_patch16_dinov3.lvd1689m`) @ 144×272 Frozen Multi-Layer Features + Linear Screen

**Self-contained notebook** (needs only `t4_data.zip` on Drive — no `cpe342_code.zip` re-upload required).

### What this extracts & checks
- **Backbone:** `vit_7b_patch16_dinov3.lvd1689m` (6,716 M params, `embed_dim=4096`, `depth=40`, `RoPE`, 5 prefix tokens, ungated in `timm==1.0.30`).
- **Geometry:** Native 2:1 `144×272` (`9×17` patches of `16 px`), ImageNet mean/std, bicubic resize, h-flip average.
- **Multi-Layer Concat (`16,384-d` = `4 × 4096`):** matches `ig_r43_t4_probe_v2.py` exactly (`subblock_dims = [4096, 4096, 4096, 4096]`):
  1. `mid_patch`: mean of normalized patch tokens over blocks `-4, -3, -2`
  2. `last_patch`: normalized last-block mean patch tokens (`== model(x)` avg pool)
  3. `last_cls`: last-block CLS token
  4. `last_std`: last-block patch-token spatial std (`unbiased=False`)
- **Views saved atomically to Drive (`out/preds/dinov3_7b_144x272/`):**
  - `train_full.npy` (`1.03 GB`), `train_full_pooled.npy` (`258 MB` = `last_patch` only, fast download for local Linear Screen)
  - `train_zoom70.npy` (`1.03 GB`), `train_centre_grey.npy` (`1.03 GB`), and `train_dark_lc.npy` (`1.03 GB` if `INCLUDE_DARK_LC = True`)
  - `test_full.npy` (`848 MB` if `EXTRACT_TEST = True`, inference-only cache — never used in the Linear Screen), `test_file_name.npy`, `meta.json`
- **Built-in Linear Screen (`id_r42` protocol, zero test leakage):** evaluates 5-fold L2 Logistic Regression (`C=0.03`) on `train_full_pooled` against `ViT-H+ @ 144×272` (`full 0.9486 | dark 0.8927`) and saves `out/logs/dinov3_7b_144x272_screen.log`.

**Colab runtime:** select **A100 (40 GB / 80 GB)** or **L4 High-RAM**. Do **not** use a T4 (no native bf16).
"""

    c1 = f"""
# ---- Settings -------------------------------------------------------------------------------------------------------------
MODEL = "vit_7b_patch16_dinov3.lvd1689m"
HW = (144, 272)
BS = 64
INCLUDE_DARK_LC = True    # True -> extract 4 train views [full, zoom70, centre_grey, dark_lc]; False -> 3 views
EXTRACT_TEST = True       # True -> extract test full view (inference-only cache; never read by Linear Screen)
RUN_LINEAR_SCREEN = True  # True -> run exact 5-fold id_r42 Linear Screen on train_full_pooled
AUTO_DISCONNECT = True    # end runtime 60 s after completion or on error to save Colab CU
{SHARED_SETUP_HELPER.strip()}
{SHARED_ERROR_HANDLER.strip()}
""".strip()

    c2 = r"""
# ---- Packages, GPU & Disk check -------------------------------------------------------------------------------------------
!pip -q install timm==1.0.30
import os, shutil, subprocess
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

disk = shutil.disk_usage("/content")
free_gb = disk.free / 2**30
print(f"/content disk free: {free_gb:.1f} GiB")
assert free_gb >= 32.0, f"Low /content disk space ({free_gb:.1f} GiB): DINOv3-7B weights + data need >= 32 GiB"
"""

    c3 = f"""
# ---- Mount Drive & unpack task4 images ------------------------------------------------------------------------------------
{SHARED_DATA_UNPACK.strip()}
OUT_DIR = D / "out" / "preds" / "dinov3_7b_144x272"
LOG_DIR = D / "out" / "logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
print("Output dir on Drive:", OUT_DIR)
""".strip()

    c4 = r"""
# ---- Load DINOv3-7B safely -> bfloat16 -> CUDA + Pooling Parity Sanity Check ---------------------------------------------
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

# Check CPU RAM before instantiating 6.7B model
mem_total_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
print(f"System CPU RAM: {mem_total_gb:.1f} GiB")
t_load = time.time()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

if mem_total_gb >= 38.0:
    model = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=HW)
    model = model.to(torch.bfloat16)
else:
    # Low-CPU-RAM fallback: instantiate in bfloat16 directly, then restore fp32 default dtype
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=HW)
    finally:
        torch.set_default_dtype(torch.float32)

gc.collect()
model = model.cuda().eval()
cfg = timm.data.resolve_model_data_config(model)
MEAN = torch.tensor(cfg["mean"], device="cuda").view(1, 3, 1, 1)
STD = torch.tensor(cfg["std"], device="cuda").view(1, 3, 1, 1)
print(f"Loaded {MODEL} to CUDA bf16 in {(time.time()-t_load)/60:.1f}m | embed_dim={model.embed_dim} | allocated={torch.cuda.memory_allocated()/2**30:.2f} GiB")

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

# Sanity check: shape, finiteness, VRAM, and exact pooling parity vs model(x)
with torch.no_grad():
    x_test = torch.rand(BS, 3, *HW, device="cuda")
    feat_ml = extract_dinov3_batch(x_test)
    assert feat_ml.shape == (BS, 4 * model.embed_dim) and torch.isfinite(feat_ml).all()

    x_small = ((x_test[:4] - MEAN) / STD).to(torch.bfloat16)
    f_direct = 0.5 * (model(x_small).float() + model(torch.flip(x_small, dims=[3])).float())
    f_slice = extract_dinov3_batch(x_test[:4])[:, model.embed_dim:2 * model.embed_dim]
    parity_diff = float((f_direct - f_slice).abs().max().item())
    assert parity_diff < 2e-2, f"Pooling parity diff too large: {parity_diff}"

peak_gb = torch.cuda.max_memory_allocated() / 2**30
total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"Sanity PASS: batch={tuple(feat_ml.shape)} | last_patch vs model(x) max|diff|={parity_diff:.2e} | peak VRAM={peak_gb:.2f} / {total_gb:.1f} GiB")
assert peak_gb < 0.90 * total_gb, f"VRAM too tight ({peak_gb:.1f} / {total_gb:.1f} GiB): reduce BS"
del x_test, x_small, feat_ml, f_direct, f_slice
torch.cuda.empty_cache()
"""

    c5 = r"""
# ---- Resume-safe per-view extraction (write local -> verify -> copy atomically to Drive) ----------------------------------
train_df = pd.read_csv(DATA / "train.csv")
test_df = pd.read_csv(DATA / "test.csv")
train_paths = [str(DATA / "train" / f) for f in train_df["file_name"]]
test_paths = [str(DATA / "test" / f) for f in test_df["file_name"]]
DIM = 4 * model.embed_dim  # 16,384

TRAIN_VIEWS = ["full", "zoom70", "centre_grey"] + (["dark_lc"] if INCLUDE_DARK_LC else [])

def valid_npy(path: Path, expected_shape: tuple[int, int]) -> bool:
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
            eta = el * (len(paths) - done) / done
            print(f"  [{view_name}] {done}/{len(paths)} ({el:.1f}m elapsed, ~{eta:.1f}m left)", flush=True)

    np.save(local_tmp, buf)
    # Also save last_patch (dims 4096:8192) separately for train_full so local Linear Screen download is only ~258 MB
    if view_name == "full" and len(paths) == len(train_paths):
        pooled = buf[:, model.embed_dim:2 * model.embed_dim].copy()
        pooled_local = Path("/content/tmp_train_full_pooled.npy")
        pooled_drive = OUT_DIR / "train_full_pooled.npy"
        np.save(pooled_local, pooled)
        shutil.copy2(pooled_local, pooled_drive.with_suffix(".tmp.npy"))
        os.replace(pooled_drive.with_suffix(".tmp.npy"), pooled_drive)
        pooled_local.unlink(missing_ok=True)
        del pooled
        print(f"  [train_full_pooled] saved {pooled_drive.name} ({pooled_drive.stat().st_size / 2**20:.1f} MB)")

    del buf
    gc.collect()
    shutil.copy2(local_tmp, drive_tmp)
    os.replace(drive_tmp, out_drive_path)
    local_tmp.unlink(missing_ok=True)
    print(f"[{view_name}] DONE -> {out_drive_path.name} ({out_drive_path.stat().st_size / 2**20:.1f} MB) in {(time.time()-t0)/60:.1f}m")

for v in TRAIN_VIEWS:
    extract_view_npy(train_paths, v, OUT_DIR / f"train_{v}.npy")

# Ensure train_full_pooled.npy exists even if train_full.npy was pre-cached on Drive
pooled_drive = OUT_DIR / "train_full_pooled.npy"
if not valid_npy(pooled_drive, (len(train_paths), model.embed_dim)):
    full_mm = np.load(OUT_DIR / "train_full.npy", mmap_mode="r")
    pooled = np.array(full_mm[:, model.embed_dim:2 * model.embed_dim], dtype=np.float16)
    del full_mm
    np.save(pooled_drive.with_suffix(".tmp.npy"), pooled)
    os.replace(pooled_drive.with_suffix(".tmp.npy"), pooled_drive)
    del pooled
    print(f"[train_full_pooled] materialized ({pooled_drive.stat().st_size / 2**20:.1f} MB)")

if EXTRACT_TEST:
    extract_view_npy(test_paths, "full", OUT_DIR / "test_full.npy")
    np.save(OUT_DIR / "test_file_name.npy", np.array(test_df["file_name"].tolist(), dtype="U"))

meta = {
    "model": MODEL,
    "hw": list(HW),
    "dim": DIM,
    "subblock_dims": [model.embed_dim] * 4,
    "subblock_names": ["mid_patch", "last_patch", "last_cls", "last_std"],
    "train_views": TRAIN_VIEWS,
    "extract_test": EXTRACT_TEST,
    "n_train": len(train_paths),
    "n_test": len(test_paths),
}
(OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

# Free GPU before CPU Linear Screen
del model
gc.collect()
torch.cuda.empty_cache()
print("All feature caches saved cleanly to Drive.")
"""

    c6 = r"""
# ---- Built-in id_r42 Linear Screen (training train_full_pooled only; zero test data) ---------------------------------------
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

    # Compute exact Q10 brightness/contrast dark mask from native 120x230 training images (0.0 diff vs r30/t4_errors.npz)
    grey = np.empty((len(train_paths), 2), dtype=np.float32)
    w_rgb = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    for i in range(0, len(train_paths), 4096):
        g = (load_uint8(train_paths[i:i + 4096]).astype(np.float32) / 255.0) @ w_rgb
        grey[i:i + len(g), 0] = g.mean(axis=(1, 2))
        grey[i:i + len(g), 1] = g.std(axis=(1, 2))
    dark = (grey[:, 0] <= np.quantile(grey[:, 0], 0.10)) | (grey[:, 1] <= np.quantile(grey[:, 1], 0.10))

    def fit_probe_lbfgs(A_np, Y_np, c_probe=0.03, iters=300):
        A_t = torch.from_numpy(A_np)
        Y_t = torch.from_numpy(Y_np)
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

    X = np.load(OUT_DIR / "train_full_pooled.npy").astype(np.float32)
    oof_probs = np.zeros((len(y), 5), dtype=np.float64)
    t_scr = time.time()
    for f, va in enumerate(folds):
        tr = np.where(fo != f)[0]
        mu = X[tr].mean(0)
        sd = X[tr].std(0) + 1e-6
        W, b = fit_probe_lbfgs((X[tr] - mu) / sd, y[tr])
        oof_probs[va] = (torch.from_numpy((X[va] - mu) / sd) @ W + b).softmax(1).numpy()
        print(f"  screen fold {f}: val acc={np.mean(oof_probs[va].argmax(1) == y[va]):.4f}", flush=True)

    pi = np.bincount(y, minlength=5) / len(y)
    wb = (0.2 / pi)[y]
    bal = lambda pred, m: f1_score(y[m], pred[m], average="macro", sample_weight=wb[m])
    rule = lambda p: (p * pi ** -0.75).argmax(1)
    allm = np.ones(len(y), dtype=bool)

    pred_r = rule(oof_probs)
    sc_full = bal(pred_r, allm)
    sc_dark = bal(pred_r, dark)
    sc_plain = f1_score(y, oof_probs.argmax(1), average="macro")

    # Reference numbers recorded in experiments/claude/outputs/r42/id_dark_screen.log
    refs = {
        "ViT-H+ DINOv3 @ 144x272 (1280-d)": (0.9486, 0.8927),
        "DINOv3-L @ 144x272 (1024-d)":      (0.9363, 0.8716),
        "C-RADIOv3-L summary @ 192x384":    (0.9543, 0.9027),
    }
    lines = [
        f"DINOv3-7B @ 144x272 linear probe screen ({time.strftime('%Y-%m-%d %H:%M UTC')}, {(time.time()-t_scr):.0f}s):",
        f"  DINOv3-7B @ 144x272 (4096-d)     : full {sc_full:.4f} | dark {sc_dark:.4f} | plain macro-F1 {sc_plain:.4f}",
    ]
    for rname, (rf, rd) in refs.items():
        lines.append(f"  minus {rname:32s}: full {sc_full - rf:+.4f}, dark {sc_dark - rd:+.4f}")
    df_h, dd_h = sc_full - refs["ViT-H+ DINOv3 @ 144x272 (1280-d)"][0], sc_dark - refs["ViT-H+ DINOv3 @ 144x272 (1280-d)"][1]
    verdict = "CLEAR WIN over ViT-H+" if (dd_h >= 0.005 and df_h >= -0.003) else "no clear >=+0.005 dark win (still check in P_v2)"
    lines.append(f"  id_r42 rule vs ViT-H+: {verdict}")
    report_txt = "\n".join(lines)
    print("\n" + report_txt)
    (LOG_DIR / "dinov3_7b_144x272_screen.log").write_text(report_txt + "\n", encoding="utf-8")
"""

    c7 = r"""
# ---- Summary of files on Drive & auto-disconnect --------------------------------------------------------------------------
print("Saved files in", OUT_DIR)
for p in sorted(OUT_DIR.iterdir()):
    print(f"  {p.name:28s} {p.stat().st_size / 2**20:8.1f} MB")
scr_log = LOG_DIR / "dinov3_7b_144x272_screen.log"
if scr_log.exists():
    print("Screen log:", scr_log)

if AUTO_DISCONNECT:
    import time
    print(f"done -> ending runtime in 60 s [{time.strftime('%H:%M')} UTC]", flush=True)
    time.sleep(60)
    from google.colab import runtime
    runtime.unassign()
"""

    nb = make_nb(title, [c1, c2, c3, c4, c5, c6, c7])
    out = Path("colab/t4_colab_dinov3_7b_feat.ipynb")
    nbf.write(nb, out)
    return out


def build_multires_tta_nb() -> Path:
    title = r"""
# CPE342 Task 4 (Round 43): Zero-Crop RoPE Multi-Resolution TTA (`dinov3Hp_384` @ `160×304`, `192×384`, `224×432`)

**Self-contained notebook** (uses the restored `t4_dinov3Hp_384_fold0..4.pt` and `t4_dinov3Hp_384_full.pt` checkpoints on Drive + `t4_data.zip` — no `cpe342_code.zip` re-upload required).

### Protocol
- Loads each fold's EMA checkpoint with `timm.create_model("vit_huge_plus_patch16_dinov3.lvd1689m", num_classes=5, dynamic_img_size=True)`.
- Predicts OOF validation images (strictly training folds) under `full`, `zoom70`, `centre_grey` at three native 2:1 RoPE grids:
  - `160×304` (`10×19` patches)
  - `192×384` (`12×24` patches — baseline, checked for `mean|diff| < 2e-3`, `argmax agree >= 99.8%`, `max|diff| < 0.08` vs cached `r30_views`)
  - `224×432` (`14×27` patches)
- Saves tiny outputs (< 5 MB total) to `Drive/out/preds/`:
  - `t4_mres_dinov3Hp_384_fold{0..4}.npz` (used on laptop for strict Nested 5-fold `--eval` against `v38`)
  - `t4_mres_dinov3Hp_384_test.npz` (if `EXTRACT_TEST = True`: inference-only cache for `fold0..4` + `full`, never read by `--eval`)
"""

    c1 = f"""
# ---- Settings -------------------------------------------------------------------------------------------------------------
MODEL = "vit_huge_plus_patch16_dinov3.lvd1689m"
TAG = "dinov3Hp_384"
BASE_HW = (192, 384)
RESOLUTIONS = [(160, 304), (192, 384), (224, 432)]
RES_KEYS = [f"{{h}}x{{w}}" for h, w in RESOLUTIONS]
EVAL_VIEWS = ["full", "zoom70", "centre_grey"]
FOLDS = [0, 1, 2, 3, 4]
BS = 64
EXTRACT_TEST = True       # True -> also score test.csv across fold0..4.pt + full.pt (inference-only cache)
AUTO_DISCONNECT = True    # end runtime 60 s after completion or on error to save Colab CU
{SHARED_SETUP_HELPER.strip()}
{SHARED_ERROR_HANDLER.strip()}
""".strip()

    c2 = r"""
# ---- Packages & GPU check -------------------------------------------------------------------------------------------------
!pip -q install timm==1.0.30
import numpy, pandas, sklearn, timm, torch, torchvision

print("torch", torch.__version__, "| timm", timm.__version__)
assert torch.cuda.is_available(), "CUDA required"
cap = torch.cuda.get_device_capability(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"GPU: {torch.cuda.get_device_name(0)} | cc={cap} | VRAM={vram_gb:.1f} GiB")
assert cap[0] >= 8, "GPU lacks native bf16 (T4?): switch runtime to A100 or L4"
"""

    c3 = f"""
# ---- Mount Drive, unpack data, verify all restored .pt checkpoints --------------------------------------------------------
{SHARED_DATA_UNPACK.strip()}
OUT_PREDS = D / "out" / "preds"
OUT_MODELS = D / "out" / "models"
OUT_R30 = D / "out" / "r30_views"
OUT_PREDS.mkdir(parents=True, exist_ok=True)

needed_ckpts = [OUT_MODELS / f"t4_{{TAG}}_fold{{f}}.pt" for f in FOLDS]
if EXTRACT_TEST:
    needed_ckpts.append(OUT_MODELS / f"t4_{{TAG}}_full.pt")
missing = [p.name for p in needed_ckpts if not p.exists()]
assert not missing, f"Missing checkpoints in {{OUT_MODELS}}: {{missing}}"
print("Verified checkpoints in Drive models/:", [p.name for p in needed_ckpts])
""".strip()

    c4 = r"""
# ---- Run Multi-Res TTA (OOF folds + optional test full+5fold) -------------------------------------------------------------
import gc, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn.functional as F
import timm

NATIVE_HW = (120, 230)
MEAN = torch.tensor((0.485, 0.456, 0.406), device="cuda").view(1, 3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225), device="cuda").view(1, 3, 1, 1)

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

def to_gpu_batch(arr_u8, hw):
    x = torch.from_numpy(arr_u8).cuda(non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
    return F.interpolate(x, size=hw, mode="bilinear", align_corners=False)

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

def load_dinov3_dynamic(ckpt_path: Path):
    model = timm.create_model(MODEL, pretrained=False, num_classes=5, dynamic_img_size=True)
    state = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(state, strict=True)
    return model.to(device="cuda", memory_format=torch.channels_last).eval()

@torch.no_grad()
def predict_multires_views(model, paths, views):
    out = {f"{v}_{rk}": [] for v in views for rk in RES_KEYS}
    for i in range(0, len(paths), 4096):
        arr = load_uint8(paths[i:i + 4096])
        for j in range(0, len(arr), BS):
            batch_u8 = arr[j:j + BS]
            x_base = to_gpu_batch(batch_u8, BASE_HW) if any(v != "full" for v in views) else None
            view_192 = {}
            for v in views:
                if v == "zoom70":
                    view_192[v] = zoom(x_base, 0.7)
                elif v == "centre_grey":
                    view_192[v] = centre_grey(x_base)
            for hw, rk in zip(RESOLUTIONS, RES_KEYS):
                for v in views:
                    if v == "full":
                        xv = to_gpu_batch(batch_u8, hw)
                    else:
                        x192 = view_192[v]
                        xv = x192 if hw == BASE_HW else F.interpolate(x192, size=hw, mode="bilinear", align_corners=False)
                    xn = ((xv - MEAN) / STD).to(memory_format=torch.channels_last)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        p = 0.5 * (model(xn).float().softmax(-1) + model(xn.flip(-1)).float().softmax(-1))
                    out[f"{v}_{rk}"].append(p.cpu().numpy().astype(np.float32))
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}

train_df = pd.read_csv(DATA / "train.csv")
test_df = pd.read_csv(DATA / "test.csv")
train_paths = [str(DATA / "train" / f) for f in train_df["file_name"]]
test_paths = [str(DATA / "test" / f) for f in test_df["file_name"]]
y = train_df["label"].astype(np.int64).values
fold_idx = [va for _, va in StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y)]
req_keys = [f"{v}_{rk}" for v in EVAL_VIEWS for rk in RES_KEYS]

# 1. Extract OOF folds (resume-safe)
t_all = time.time()
for f in FOLDS:
    va = fold_idx[f]
    out_path = OUT_PREDS / f"t4_mres_{TAG}_fold{f}.npz"
    if out_path.exists():
        with np.load(out_path) as z:
            if all(k in z.files and z[k].shape == (len(va), 5) for k in req_keys):
                print(f"fold {f}: valid cache on Drive -> skipping")
                continue
    t1 = time.time()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = load_dinov3_dynamic(OUT_MODELS / f"t4_{TAG}_fold{f}.pt")
    preds = predict_multires_views(model, [train_paths[idx] for idx in va], EVAL_VIEWS)
    peak = torch.cuda.max_memory_allocated() / 2**30

    parity_msg = ""
    r30_path = OUT_R30 / f"{TAG}_fold{f}.npz"
    if r30_path.exists():
        r30_full = np.load(r30_path)["full"]
        abs_diff = np.abs(preds["full_192x384"] - r30_full)
        mean_diff = float(abs_diff.mean())
        max_diff = float(abs_diff.max())
        agree = float(np.mean(preds["full_192x384"].argmax(1) == r30_full.argmax(1)))
        assert mean_diff < 2e-3 and agree >= 0.998 and max_diff < 0.08, (
            f"fold {f} 192x384 parity failed: mean={mean_diff:.2e}, max={max_diff:.4f}, agree={agree:.4f}"
        )
        parity_msg = f" | 192x384 parity mean={mean_diff:.2e}, max={max_diff:.4f}, agree={agree:.4f}"

    tmp = Path(f"/content/tmp_{out_path.name}")
    np.savez(tmp, val_idx=va, **preds)
    shutil.copy2(tmp, out_path.with_suffix(".tmp.npz"))
    os.replace(out_path.with_suffix(".tmp.npz"), out_path)
    tmp.unlink(missing_ok=True)
    print(f"fold {f}: saved {out_path.name} in {(time.time()-t1)/60:.1f}m | peak={peak:.2f} GiB{parity_msg}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()

# 2. Extract test Multi-Res TTA (inference-only cache)
if EXTRACT_TEST:
    test_out = OUT_PREDS / f"t4_mres_{TAG}_test.npz"
    if test_out.exists():
        print(f"test cache {test_out.name} already exists on Drive -> skipping")
    else:
        fold_means = {f"fold_mean_{rk}": np.zeros((len(test_paths), 5), dtype=np.float64) for rk in RES_KEYS}
        for f in range(5):
            t1 = time.time()
            model = load_dinov3_dynamic(OUT_MODELS / f"t4_{TAG}_fold{f}.pt")
            preds = predict_multires_views(model, test_paths, ["full"])
            for rk in RES_KEYS:
                fold_means[f"fold_mean_{rk}"] += preds[f"full_{rk}"] / 5.0
            print(f"test fold {f}: done in {(time.time()-t1)/60:.1f}m", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()

        t1 = time.time()
        model = load_dinov3_dynamic(OUT_MODELS / f"t4_{TAG}_full.pt")
        full_preds = predict_multires_views(model, test_paths, ["full"])
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"test full refit: done in {(time.time()-t1)/60:.1f}m", flush=True)

        tmp = Path(f"/content/tmp_{test_out.name}")
        np.savez(
            tmp,
            **{k: v.astype(np.float32) for k, v in fold_means.items()},
            **{f"full_{rk}": full_preds[f"full_{rk}"].astype(np.float32) for rk in RES_KEYS},
            test_file_name=np.array(test_df["file_name"].tolist(), dtype="U"),
        )
        shutil.copy2(tmp, test_out.with_suffix(".tmp.npz"))
        os.replace(test_out.with_suffix(".tmp.npz"), test_out)
        tmp.unlink(missing_ok=True)
        print(f"saved {test_out.name} ({test_out.stat().st_size / 2**20:.2f} MB)")

print(f"Finished Multi-Res TTA in {(time.time()-t_all)/60:.1f}m")
"""

    c5 = r"""
# ---- Summary of tiny files to pull back + auto-disconnect -----------------------------------------------------------------
for p in sorted(OUT_PREDS.glob(f"t4_mres_{TAG}*.npz")):
    print(f"  {p.name:36s} {p.stat().st_size / 2**20:6.2f} MB")

if AUTO_DISCONNECT:
    import time
    print(f"done -> ending runtime in 60 s [{time.strftime('%H:%M')} UTC]", flush=True)
    time.sleep(60)
    from google.colab import runtime
    runtime.unassign()
"""

    nb = make_nb(title, [c1, c2, c3, c4, c5])
    out = Path("colab/t4_colab_multires_tta.ipynb")
    nbf.write(nb, out)
    return out


if __name__ == "__main__":
    p1 = build_dinov3_7b_nb()
    p2 = build_multires_tta_nb()
    print(f"Generated {p1} and {p2}")
