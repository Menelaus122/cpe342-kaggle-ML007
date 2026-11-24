"""Round 44, Stage 1 (0 Colab CU): Non-DINOv3 giant vision backbone linear screen on laptop RTX 4070.

    python experiments/claude/im_r44_t4_non_dinov3_screen.py --smoke
    python experiments/claude/im_r44_t4_non_dinov3_screen.py

Strictly Zero Test Leakage:
  Reads ONLY training CSV (`04_image_classification/_download/task4_image_classification/train.csv`)
  and training images. `test.csv` and `test/` images are NEVER read.

Protocol (parity with id_r42 / if_r42 / Round 44):
  - Native 2:1 geometry (~12x24 patch grid):
      * C-RADIOv3-H @ 192x384 (652M, HF nvidia/C-RADIOv3-H pinned rev, summary 3840-d + spatial_mean 1280-d)
      * TIPSv2-Giant @ 168x336 (1,134.8M, timm vit_giant_patch14_reg1_tipsv2.webli, 1536-d)
      * SigLIP2-GiantOpt @ 192x384 (1,163.2M, timm vit_giantopt_patch16_siglip_384.v2_webli, 1536-d)
      * MetaCLIP2-Gigantic @ 168x336 (1,842.8M, timm vit_gigantic_patch14_clip_378.metaclip2_worldwide, 1664-d)
  - Frozen features: mean of image and horizontal flip (h-flip), saved as float16 in `experiments/claude/outputs/r44/`.
  - 5-fold L2 Logistic Regression (C=0.03, CPU L-BFGS) using exact pipeline folds + v17 rule + Q10 dark/low-contrast mask.
  - Reference `R`: DINOv3-7B @ 144x272 `last_patch` (4096-d, `0.9737-0.9739 full / 0.9378-0.9381 dark`).
  - Gate (pre-declared): Stage-2A GO if any `B_i = 7B + X_i` (or `A_i`) beats `R` by dark >= +0.005 AND full >= -0.003.
    Stage-2B Info Flag (Laptop LoRA FT candidate) if any `A_i` alone reaches dark >= 0.9050 AND full >= 0.9550.
"""
import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from src import task4  # noqa: E402
from src.config import PRED_DIR, T4_CLASSES  # noqa: E402
import train_task4_probe as P0  # noqa: E402

OUT_DIR = Path("experiments/claude/outputs/r44")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "im_non_dinov3_screen.log"
JSON_PATH = OUT_DIR / "im_non_dinov3_screen.json"

RADIO_H_REPO = "nvidia/C-RADIOv3-H"
RADIO_H_REV = "d7fd0e2b0a1761f1af150582e06c41e9a99b0bf8"

GATE_DARK = 0.005
GATE_FULL = -0.003
LORA_FLAG_DARK = 0.9050
LORA_FLAG_FULL = 0.9550
CHUNK_SIZE = 2048

# Pre-declared candidates: (label, kind, model_id, hw, tag, dim, batch_size)
CANDIDATES = [
    ("C-RADIOv3-H @ 192x384", "radio", RADIO_H_REPO, (192, 384), "cradio3H_summary_192x384", 3840, 32),
    ("TIPSv2-Giant @ 168x336", "timm", "vit_giant_patch14_reg1_tipsv2.webli", (168, 336), "tipsv2_giant_168x336", 1536, 32),
    ("SigLIP2-GiantOpt @ 192x384", "timm", "vit_giantopt_patch16_siglip_384.v2_webli", (192, 384), "siglip2_giantopt_192x384", 1536, 32),
    ("MetaCLIP2-Gigantic @ 168x336", "timm", "vit_gigantic_patch14_clip_378.metaclip2_worldwide", (168, 336), "metaclip2_gigantic_168x336", 1664, 32),
]
RADIO_SPATIAL_TAG = "cradio3H_spatialmean_192x384"
RADIO_SPATIAL_DIM = 1280


def load_train_only():
    """Load training metadata and image paths without reading test.csv or test images."""
    df = pd.read_csv(task4.DATA / "train.csv")
    if "file_name" in df.columns:
        df["label"] = df["label"].astype(int)
        df["class"] = [T4_CLASSES[k] for k in df.label]
    else:
        df["label"] = df["class"].map(task4.CLASS_TO_IDX).astype(int)
        df["file_name"] = df.id.astype(str) + ".jpg"
    df["path"] = [str(task4.DATA / "train" / f) for f in df.file_name]
    return df


def save_npy_atomic(path: Path, arr: np.ndarray):
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def valid_npy(path: Path, expected_shape, expected_dtype=np.float16) -> bool:
    if not path.exists():
        return False
    try:
        mm = np.load(path, mmap_mode="r")
        ok = (mm.shape == expected_shape) and (mm.dtype == expected_dtype)
        base = getattr(mm, "_mmap", None)
        if base is not None:
            base.close()
        del mm
        return ok
    except Exception:
        return False


def clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def load_timm_bf16(name: str, hw: tuple[int, int]):
    """Instantiate a large timm backbone directly in bfloat16 to respect ~4.9 GB free CPU RAM."""
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = timm.create_model(name, pretrained=True, num_classes=0, img_size=hw)
    except RuntimeError as e:
        # Fallback if any timm pos_embed cubic interpolation on CPU lacks bfloat16 support
        torch.set_default_dtype(torch.float32)
        del e
        gc.collect()
        model = timm.create_model(name, pretrained=True, num_classes=0, img_size=hw).to(dtype=torch.bfloat16)
    finally:
        torch.set_default_dtype(torch.float32)
    return model.cuda().eval()


def load_radio_h():
    from transformers import AutoModel
    return AutoModel.from_pretrained(RADIO_H_REPO, trust_remote_code=True, revision=RADIO_H_REV).cuda().eval()


@torch.no_grad()
def run_timm_extraction(model, paths, hw, dim, bs, collect_pair_stats=False, log_fn=print):
    cfg = timm.data.resolve_model_data_config(model)
    mean = torch.tensor(cfg["mean"], device="cuda", dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device="cuda", dtype=torch.float32).view(1, 3, 1, 1)

    n = len(paths)
    out = np.empty((n, dim), dtype=np.float16)
    orig_buf = np.empty((n, dim), dtype=np.float32) if collect_pair_stats else None
    flip_buf = np.empty((n, dim), dtype=np.float32) if collect_pair_stats else None

    torch.cuda.reset_peak_memory_stats()
    t_ext = time.time()
    t_steady = None
    steady_imgs = 0
    done = 0

    for i in range(0, n, CHUNK_SIZE):
        chunk = task4.load_uint8(list(paths[i:i + CHUNK_SIZE]))
        for j in range(0, len(chunk), bs):
            x0 = torch.from_numpy(chunk[j:j + bs]).cuda().permute(0, 3, 1, 2).float().div_(255.0)
            xv = F.interpolate(x0, size=hw, mode="bicubic", align_corners=False).clamp_(0.0, 1.0)
            x = ((xv - mean) / std).to(torch.bfloat16)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f1 = model(x).float()
                f2 = model(torch.flip(x, dims=[3])).float()
            avg = 0.5 * (f1 + f2)
            assert torch.isfinite(avg).all(), f"Non-finite features at slice {i + j}:{i + j + len(chunk[j:j + bs])}"
            sl = slice(i + j, i + j + len(avg))
            out[sl] = avg.cpu().numpy().astype(np.float16)
            if collect_pair_stats:
                orig_buf[sl] = f1.cpu().numpy()
                flip_buf[sl] = f2.cpu().numpy()
            if t_steady is None and (i + j) > 0:
                t_steady = time.time()
            elif t_steady is not None:
                steady_imgs += len(avg)
            done += len(avg)
        del chunk
        if (i + CHUNK_SIZE) % 4096 == 0 or done == n:
            el = (time.time() - t_ext) / 60.0
            ips = done / max(time.time() - t_ext, 1e-6)
            eta = el * (n - done) / max(done, 1)
            peak_gb = torch.cuda.max_memory_allocated() / 2**30
            if n > 256:
                log_fn(f"    {done}/{n} ({el:.1f}m elapsed, {ips:.1f} img/s, ~{eta:.1f}m left, peak VRAM {peak_gb:.2f} GiB)")

    torch.cuda.synchronize()
    total_s = time.time() - t_ext
    steady_ips = steady_imgs / max(time.time() - t_steady, 1e-6) if t_steady is not None and steady_imgs > 0 else n / max(total_s, 1e-6)
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    return {"primary": out, "orig": orig_buf, "flip": flip_buf, "peak_gb": peak_gb,
            "total_s": total_s, "img_s": n / max(total_s, 1e-6), "steady_img_s": steady_ips}


@torch.no_grad()
def run_radio_extraction(model, paths, hw, bs, collect_pair_stats=False, log_fn=print):
    n = len(paths)
    out_sum = np.empty((n, 3840), dtype=np.float16)
    out_spa = np.empty((n, RADIO_SPATIAL_DIM), dtype=np.float16)
    orig_buf = np.empty((n, 3840), dtype=np.float32) if collect_pair_stats else None
    flip_buf = np.empty((n, 3840), dtype=np.float32) if collect_pair_stats else None

    torch.cuda.reset_peak_memory_stats()
    t_ext = time.time()
    t_steady = None
    steady_imgs = 0
    done = 0

    for i in range(0, n, CHUNK_SIZE):
        chunk = task4.load_uint8(list(paths[i:i + CHUNK_SIZE]))
        for j in range(0, len(chunk), bs):
            x0 = torch.from_numpy(chunk[j:j + bs]).cuda().permute(0, 3, 1, 2).float().div_(255.0)
            xv = F.interpolate(x0, size=hw, mode="bicubic", align_corners=False).clamp_(0.0, 1.0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s1, f1 = model(xv)
                s2, f2 = model(torch.flip(xv, dims=[3]))
            s_avg = 0.5 * (s1.float() + s2.float())
            m_avg = 0.5 * (f1.float().mean(1) + f2.float().mean(1))
            assert torch.isfinite(s_avg).all() and torch.isfinite(m_avg).all(), f"Non-finite RADIO features at {i + j}"
            sl = slice(i + j, i + j + len(s_avg))
            out_sum[sl] = s_avg.cpu().numpy().astype(np.float16)
            out_spa[sl] = m_avg.cpu().numpy().astype(np.float16)
            if collect_pair_stats:
                orig_buf[sl] = s1.float().cpu().numpy()
                flip_buf[sl] = s2.float().cpu().numpy()
            if t_steady is None and (i + j) > 0:
                t_steady = time.time()
            elif t_steady is not None:
                steady_imgs += len(s_avg)
            done += len(s_avg)
        del chunk
        if (i + CHUNK_SIZE) % 4096 == 0 or done == n:
            el = (time.time() - t_ext) / 60.0
            ips = done / max(time.time() - t_ext, 1e-6)
            eta = el * (n - done) / max(done, 1)
            peak_gb = torch.cuda.max_memory_allocated() / 2**30
            if n > 256:
                log_fn(f"    {done}/{n} ({el:.1f}m elapsed, {ips:.1f} img/s, ~{eta:.1f}m left, peak VRAM {peak_gb:.2f} GiB)")

    torch.cuda.synchronize()
    total_s = time.time() - t_ext
    steady_ips = steady_imgs / max(time.time() - t_steady, 1e-6) if t_steady is not None and steady_imgs > 0 else n / max(total_s, 1e-6)
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    return {"primary": out_sum, "spatial": out_spa, "orig": orig_buf, "flip": flip_buf,
            "peak_gb": peak_gb, "total_s": total_s, "img_s": n / max(total_s, 1e-6), "steady_img_s": steady_ips}


def extract_candidate_with_oom_fallback(label, kind, model_id, hw, dim, bs, paths, collect_pair_stats=False, log_fn=print):
    clear_cuda()
    t_load = time.time()
    model = load_radio_h() if kind == "radio" else load_timm_bf16(model_id, hw)
    load_s = time.time() - t_load
    used_bs = bs
    try:
        res = (run_radio_extraction(model, paths, hw, used_bs, collect_pair_stats, log_fn)
               if kind == "radio" else
               run_timm_extraction(model, paths, hw, dim, used_bs, collect_pair_stats, log_fn))
    except torch.cuda.OutOfMemoryError:
        log_fn(f"  [OOM] {label} at bs={used_bs} -> retrying with bs=16")
        torch.cuda.empty_cache()
        used_bs = 16
        res = (run_radio_extraction(model, paths, hw, used_bs, collect_pair_stats, log_fn)
               if kind == "radio" else
               run_timm_extraction(model, paths, hw, dim, used_bs, collect_pair_stats, log_fn))
    finally:
        del model
        clear_cuda()
    res["load_s"] = load_s
    res["bs"] = used_bs
    return res


def check_pair_diagnostics(arr_avg, arr_orig, arr_flip, expected_dim):
    f32 = arr_avg.astype(np.float32)
    assert f32.ndim == 2 and f32.shape[1] == expected_dim, f"Bad shape {f32.shape}, expected dim {expected_dim}"
    assert np.isfinite(f32).all(), "Non-finite entries found"
    std_val = float(f32.std())
    assert std_val > 1e-3, f"Degenerate features std={std_val}"

    # Pairwise cosine across distinct images
    norms = np.linalg.norm(f32, axis=1, keepdims=True) + 1e-8
    u = f32 / norms
    cos_mat = u @ u.T
    mask = ~np.eye(len(u), dtype=bool)
    mean_pair_cos = float(cos_mat[mask].mean())
    max_pair_cos = float(cos_mat[mask].max())
    assert mean_pair_cos < 0.99, f"Features collapsed across images (mean_pair_cos={mean_pair_cos:.4f})"

    # Original vs h-flip cosine similarity
    o_n = arr_orig / (np.linalg.norm(arr_orig, axis=1, keepdims=True) + 1e-8)
    f_n = arr_flip / (np.linalg.norm(arr_flip, axis=1, keepdims=True) + 1e-8)
    hflip_cos = float((o_n * f_n).sum(axis=1).mean())
    assert hflip_cos > 0.5, f"Low orig/hflip cosine similarity ({hflip_cos:.4f})"
    return {"std": std_val, "mean_pair_cos": mean_pair_cos, "max_pair_cos": max_pair_cos, "hflip_cos": hflip_cos}


def run_smoke(train_paths):
    n_smoke = 64
    sub_paths = train_paths[:n_smoke]
    free_gb = shutil.disk_usage(ROOT).free / 2**30
    print(f"=== SMOKE TEST (n={n_smoke} train images, free disk C:={free_gb:.1f} GiB) ===", flush=True)
    assert free_gb >= 20.0, f"Low free disk space ({free_gb:.1f} GiB)"

    ok_count = 0
    for label, kind, model_id, hw, tag, dim, bs in CANDIDATES:
        print(f"\n[smoke] {label} ({model_id}, hw={hw}, init bs={bs})", flush=True)
        try:
            res = extract_candidate_with_oom_fallback(label, kind, model_id, hw, dim, bs, sub_paths, collect_pair_stats=True)
            diag = check_pair_diagnostics(res["primary"], res["orig"], res["flip"], dim)
            assert res["peak_gb"] < 7.2, f"Peak VRAM {res['peak_gb']:.2f} GiB >= 7.2 GiB"
            extra = ""
            if kind == "radio":
                sp = res["spatial"].astype(np.float32)
                assert sp.shape == (n_smoke, RADIO_SPATIAL_DIM) and np.isfinite(sp).all() and sp.std() > 1e-3
                extra = f", spatial={sp.shape} std={sp.std():.4f}"
            proj_min = len(train_paths) / res["steady_img_s"] / 60.0
            print(f"  PASS | bs={res['bs']} | load={res['load_s']:.1f}s | peak_VRAM={res['peak_gb']:.2f} GiB | "
                  f"dim={dim} std={diag['std']:.4f} | pair_cos(mean={diag['mean_pair_cos']:.4f}, max={diag['max_pair_cos']:.4f}) | "
                  f"hflip_cos={diag['hflip_cos']:.4f}{extra} | steady={res['steady_img_s']:.1f} img/s (~{proj_min:.1f} min for {len(train_paths)})",
                  flush=True)
            ok_count += 1
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}", flush=True)
        finally:
            clear_cuda()

    print(f"\nSmoke summary: {ok_count}/{len(CANDIDATES)} candidates passed.", flush=True)
    if ok_count < len(CANDIDATES) - 1:
        sys.exit(2)


def run_screen_cv(X: np.ndarray, y: np.ndarray, fo: np.ndarray, folds, bal, rule, allm, dark, label: str, log_fn=print):
    """5-fold L2 Logistic Regression (C=0.03) matching id_r42 / if_r42 / Round 44 protocol."""
    t0 = time.time()
    oof = np.zeros((len(y), 5), dtype=np.float64)
    n_classes_x = X.shape[1]
    for f, va in enumerate(folds):
        tr = np.where(fo != f)[0]
        A_tr = np.array(X[tr], dtype=np.float32, copy=True)
        mu = A_tr.mean(0)
        sd = A_tr.std(0) + 1e-6
        A_tr -= mu
        A_tr /= sd
        W, b = P0.fit_probe(A_tr, y[tr])
        del A_tr
        A_va = np.array(X[va], dtype=np.float32, copy=True)
        A_va -= mu
        A_va /= sd
        oof[va] = (torch.from_numpy(A_va) @ W + b).softmax(1).numpy()
        del A_va, mu, sd, W, b
    pr = rule(oof)
    res = {
        "dim": int(n_classes_x),
        "full": float(bal(pr, allm)),
        "dark": float(bal(pr, dark)),
        "plain": float(f1_score(y, oof.argmax(1), average="macro")),
        "err_full": int((pr != y).sum()),
        "err_dark": int((pr[dark] != y[dark]).sum()),
        "seconds": round(time.time() - t0, 1),
    }
    log_fn(f"  {label:40s} {res['dim']:5d}-d: full {res['full']:.4f} ({res['err_full']:4d} err) | "
           f"dark {res['dark']:.4f} ({res['err_dark']:3d} err) | plain {res['plain']:.4f}  [{res['seconds']:.0f}s]")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Run 64-image GPU smoke test on all 4 models without writing full caches.")
    args = ap.parse_args()

    train_df = load_train_only()
    train_paths = train_df["path"].values

    if args.smoke:
        run_smoke(train_paths)
        return

    t_start = time.time()
    log_lines = []

    def log(msg):
        line = f"[{(time.time() - t_start) / 60:5.1f}m] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    n_train = len(train_paths)
    log(f"Round 44 Stage 1 Non-DINOv3 Giant Screen | n_train={n_train} (0 Colab CU, Zero Test Leakage)")

    # Phase 1: Feature extraction (one model at a time, atomic float16 caches)
    extracted = {}
    meta_timings = {}
    for label, kind, model_id, hw, tag, dim, bs in CANDIDATES:
        out_path = OUT_DIR / f"im_feat_{tag}.npy"
        spa_path = OUT_DIR / f"im_feat_{RADIO_SPATIAL_TAG}.npy"
        need_primary = not valid_npy(out_path, (n_train, dim))
        need_spa = (kind == "radio") and not valid_npy(spa_path, (n_train, RADIO_SPATIAL_DIM))
        if not need_primary and not need_spa:
            log(f"[cache hit] {label} -> {out_path.name}")
            extracted[label] = {"path": out_path, "dim": dim, "tag": tag}
            if kind == "radio":
                extracted["C-RADIOv3-H spatial_mean @ 192x384 (info)"] = {"path": spa_path, "dim": RADIO_SPATIAL_DIM, "tag": RADIO_SPATIAL_TAG, "info_only": True}
            continue

        log(f"[extract] {label} ({model_id}, hw={hw}, init bs={bs})")
        try:
            res = extract_candidate_with_oom_fallback(label, kind, model_id, hw, dim, bs, train_paths, collect_pair_stats=False, log_fn=log)
            save_npy_atomic(out_path, res["primary"])
            if kind == "radio":
                save_npy_atomic(spa_path, res["spatial"])
                extracted["C-RADIOv3-H spatial_mean @ 192x384 (info)"] = {"path": spa_path, "dim": RADIO_SPATIAL_DIM, "tag": RADIO_SPATIAL_TAG, "info_only": True}
            extracted[label] = {"path": out_path, "dim": dim, "tag": tag}
            meta_timings[tag] = {
                "bs": res["bs"], "load_s": round(res["load_s"], 1),
                "extract_min": round(res["total_s"] / 60.0, 2),
                "img_per_s": round(res["img_s"], 2),
                "peak_vram_gb": round(res["peak_gb"], 2),
            }
            log(f"  DONE {label}: {res['total_s'] / 60:.1f}m ({res['img_s']:.1f} img/s), bs={res['bs']}, "
                f"peak VRAM={res['peak_gb']:.2f} GiB -> {out_path.name} ({out_path.stat().st_size / 2**20:.1f} MB)")
            del res
            clear_cuda()
        except Exception as e:
            log(f"  [FAILED] {label}: {type(e).__name__}: {e}")
            clear_cuda()

    assert len(extracted) > 0, "No candidate features extracted!"

    # Phase 2: Sequential RAM-safe 5-fold Linear Screen
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    y = train_df["label"].values.astype(np.int64)
    folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
    fo = np.zeros(n_train, dtype=int)
    for f, va in enumerate(folds):
        fo[va] = f

    st = np.load("experiments/claude/outputs/r30/t4_errors.npz")
    dark = (st["contrast"] <= np.quantile(st["contrast"], 0.10)) | (st["brightness"] <= np.quantile(st["brightness"], 0.10))
    pi = np.bincount(y, minlength=5) / len(y)
    wb = (0.2 / pi)[y]
    bal = lambda pred, m: f1_score(y[m], pred[m], average="macro", sample_weight=wb[m])
    rule = lambda p: (p * pi ** -0.75).argmax(1)
    allm = np.ones(n_train, dtype=bool)
    log(f"Dark/low-contrast training frames: {int(dark.sum())} / {n_train}")

    X7b_mm = np.load("outputs/preds/dinov3_7b_144x272/train_full.npy", mmap_mode="r")
    X7b = np.array(X7b_mm[:, 4096:8192], dtype=np.float32)
    base_mm = getattr(X7b_mm, "_mmap", None)
    if base_mm is not None:
        base_mm.close()
    del X7b_mm
    gc.collect()

    log("\n--- 5-fold Linear Screen (C=0.03, balanced macro-F1, v17 rule) ---")
    results = {}
    results["R: 7B @ 144x272"] = run_screen_cv(X7b, y, fo, folds, bal, rule, allm, dark, "R: 7B @ 144x272 (ref)", log)
    ref = results["R: 7B @ 144x272"]

    primary_entries = []
    for label, info in extracted.items():
        Xi = np.load(info["path"]).astype(np.float32)
        results[f"A: {label}"] = run_screen_cv(Xi, y, fo, folds, bal, rule, allm, dark, f"A: {label}", log)
        if not info.get("info_only", False):
            primary_entries.append((label, info))
            Bi = np.hstack([X7b, Xi])
            del Xi
            gc.collect()
            results[f"B: 7B + {label}"] = run_screen_cv(Bi, y, fo, folds, bal, rule, allm, dark, f"B: 7B + {label}", log)
            del Bi
        else:
            del Xi
        gc.collect()

    if len(primary_entries) >= 2:
        Xs = [X7b] + [np.load(info["path"]).astype(np.float32) for _, info in primary_entries]
        B_all = np.hstack(Xs)
        del Xs
        gc.collect()
        results["B_all: 7B + all primaries"] = run_screen_cv(
            B_all, y, fo, folds, bal, rule, allm, dark, f"B_all: 7B + {len(primary_entries)} primaries", log
        )
        del B_all
        gc.collect()

    del X7b
    gc.collect()

    # Phase 3: Pre-declared Gate evaluation
    log("\n--- Pre-declared Gate Verdict (Stage-2A GO: dark >= +0.005 AND full >= -0.003 vs R) ---")
    vh_full, vh_dark = 0.9486, 0.8927  # logged ViT-H+ @ 144x272 reference from id_r42
    go_list = []
    lora_flags = []
    for key, r in results.items():
        if key == "R: 7B @ 144x272":
            continue
        dfu, dda = r["full"] - ref["full"], r["dark"] - ref["dark"]
        if key.startswith("A:"):
            vh_msg = f" | vs ViT-H+ (info): full {r['full'] - vh_full:+.4f}, dark {r['dark'] - vh_dark:+.4f}"
            l_flag = (r["dark"] >= LORA_FLAG_DARK) and (r["full"] >= LORA_FLAG_FULL)
            if l_flag and "(info)" not in key:
                lora_flags.append(key)
            tag_str = " [Stage-2B LoRA FLAG]" if (l_flag and "(info)" not in key) else ""
        else:
            vh_msg = ""
            tag_str = ""
        is_info = ("(info)" in key) or key.startswith("B_all:")
        ok = (dda >= GATE_DARK) and (dfu >= GATE_FULL)
        if ok and not is_info:
            go_list.append(key)
        status = "PASS (GO)" if ok else "fail"
        log(f"  {key:44s}: full {dfu:+.4f}, dark {dda:+.4f} -> {status}{tag_str}{vh_msg}")

    verdict = f"GO Stage 2A ({', '.join(go_list)})" if go_list else "STOP: no pre-declared candidate met dark >= +0.005 & full >= -0.003 vs R"
    log(f"\nFINAL VERDICT: {verdict}")
    if lora_flags:
        log(f"Stage-2B Info Flags (A_i dark >= {LORA_FLAG_DARK:.3f} & full >= {LORA_FLAG_FULL:.3f}): {', '.join(lora_flags)}")
    log(f"Total elapsed: {(time.time() - t_start) / 60:.1f} min")

    LOG_PATH.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    JSON_PATH.write_text(json.dumps({
        "timings": meta_timings,
        "results": results,
        "go_candidates": go_list,
        "lora_flags": lora_flags,
        "verdict": verdict,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {LOG_PATH} and {JSON_PATH}", flush=True)


if __name__ == "__main__":
    main()
