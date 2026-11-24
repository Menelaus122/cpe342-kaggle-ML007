"""Round 43, Task 4: Train production DINOv3-7B frozen-feature probe (P_7b, 16,384-d) on GPU.

    python scripts/train_task4_probe_7b.py --smoke
    python scripts/train_task4_probe_7b.py

Reads pre-extracted DINOv3-7B @ 144x272 multi-layer features (`outputs/preds/dinov3_7b_144x272/`):
  - train views: train_full.npy, train_zoom70.npy, train_centre_grey.npy, train_dark_lc.npy
  - test view:   test_full.npy + test_file_name.npy (inference-only, verified against test.csv order)

Normalization per image:
  per-subblock L2-norm ([4096, 4096, 4096, 4096], eps=1e-6) -> concat -> per-backbone L2-norm (eps=1e-6).
Standardized per fold with `full` training rows' channel mean and global scalar SD.
Writes standard Task 4 caches compatible with `nb_part2.py`:
  - `outputs/preds/t4_probe7b_fold{0..4}.npz` (val_idx, val_probs, test_probs, test_file_name)
  - `outputs/preds/t4_probe7b_full.npz` (test_probs, test_file_name)
"""
import argparse
import gc
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import task4  # noqa: E402
from src.config import OUT, PRED_DIR  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FEAT_DIR = PRED_DIR / "dinov3_7b_144x272"
TRAIN_VIEWS = ["full", "zoom70", "centre_grey", "dark_lc"]
SUBBLOCK_DIMS = [4096, 4096, 4096, 4096]
DIM = 16384
SEEDS = (0, 1, 2)
EPOCHS = 30
BS = 512
OOF_PARITY_PATH = Path("experiments/claude/outputs/r42/ii_probe_v3_views.npz")
LOG_PATH = OUT / "logs" / "t4_probe7b_train.log"


def make_logger(log_path, write_log):
    t0 = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "w", encoding="utf-8", buffering=1) if write_log else None

    def log(msg):
        line = f"[{(time.time() - t0) / 60:5.1f}m] {msg}"
        print(line, flush=True)
        if f is not None:
            f.write(line + "\n")
            f.flush()

    return log, f


def close_mmap_obj(mm):
    if mm is None:
        return
    try:
        mm.flush()
    except Exception:
        pass
    base = getattr(mm, "_mmap", None)
    if base is not None:
        base.close()


def normalize_backbone_view(arr_f16, subdims=SUBBLOCK_DIMS):
    """Per-subblock L2-norm -> concat -> per-backbone L2-norm, returned in float16."""
    x = arr_f16.astype(np.float32)
    off = 0
    parts = []
    for d in subdims:
        sub = x[:, off:off + d]
        nrm = np.linalg.norm(sub, axis=1, keepdims=True) + 1e-6
        parts.append(sub / nrm)
        off += d
    assert off == x.shape[1]
    cat = np.hstack(parts)
    bnrm = np.linalg.norm(cat, axis=1, keepdims=True) + 1e-6
    return (cat / bnrm).astype(np.float16)


def verify_feature_files(n_train, n_test, test_names):
    for v in TRAIN_VIEWS:
        p = FEAT_DIR / f"train_{v}.npy"
        assert p.exists(), f"Missing {p}"
        mm = np.load(p, mmap_mode="r")
        try:
            assert mm.shape == (n_train, DIM) and mm.dtype == np.float16, f"Bad shape/dtype in {p}: {mm.shape}, {mm.dtype}"
        finally:
            close_mmap_obj(mm)
            del mm
    tfn = FEAT_DIR / "test_file_name.npy"
    tp = FEAT_DIR / "test_full.npy"
    assert tfn.exists() and tp.exists(), f"Missing test caches under {FEAT_DIR}"
    cached_names = np.load(tfn)
    assert np.array_equal(cached_names, test_names), "test_file_name.npy does not match test.csv order"
    mm = np.load(tp, mmap_mode="r")
    try:
        assert mm.shape == (n_test, DIM) and mm.dtype == np.float16, f"Bad shape/dtype in {tp}: {mm.shape}, {mm.dtype}"
    finally:
        close_mmap_obj(mm)
        del mm


def build_l2_memmaps(n_train_rows, n_test_rows, prefix, log=print):
    """Normalize train views + test full view into temporary float16 memmaps."""
    t0 = time.time()
    specs = [(f"train_{v}", FEAT_DIR / f"train_{v}.npy", n_train_rows) for v in TRAIN_VIEWS] + [
        ("test_full", FEAT_DIR / "test_full.npy", n_test_rows)
    ]
    mm_paths = {}
    chunk = 4096
    for key, src_path, n_rows in specs:
        out_path = FEAT_DIR / f"{prefix}_{key}.npy"
        if out_path.exists():
            out_path.unlink()
        src = np.load(src_path, mmap_mode="r")
        dst = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(n_rows, DIM))
        try:
            for i in range(0, n_rows, chunk):
                hi = min(i + chunk, n_rows)
                normed = normalize_backbone_view(src[i:hi])
                assert np.isfinite(normed).all(), f"Non-finite L2 features in {key} rows {i}:{hi}"
                dst[i:hi] = normed
        finally:
            close_mmap_obj(src)
            close_mmap_obj(dst)
            del src, dst
        gc.collect()
        mm_paths[key] = out_path
    log(f"Built normalized L2 memmaps ({n_train_rows} train x 4 + {n_test_rows} test) in {time.time() - t0:.1f}s")
    return mm_paths


def cleanup_memmaps(mm_paths):
    gc.collect()
    for p in mm_paths.values():
        if p.exists():
            p.unlink()


def compute_fold_stats(A_tr_f16, n_tr):
    """Chunked float64 channel mean & global scalar SD over the `full` training view (rows 0:n_tr)."""
    sum_f64 = np.zeros(DIM, dtype=np.float64)
    chunk = 2048
    for i in range(0, n_tr, chunk):
        sum_f64 += A_tr_f16[i:min(i + chunk, n_tr)].astype(np.float64).sum(axis=0)
    mu = (sum_f64 / n_tr).astype(np.float32)

    sq_sum = 0.0
    mu64 = mu.astype(np.float64)
    for i in range(0, n_tr, chunk):
        diff = A_tr_f16[i:min(i + chunk, n_tr)].astype(np.float64) - mu64
        sq_sum += float((diff * diff).sum())
    sd = float(math.sqrt(sq_sum / (n_tr * DIM))) + 1e-6
    return mu, sd


def fit_mlp_p7b(A_tr_f16, Y_tr, mu, sd, epochs=EPOCHS, seeds=SEEDS, p_mix=0.8, mix_alpha=0.4, label_smoothing=0.1):
    """Stage float16 training tensor on GPU once (CPU fallback on OOM), train 3 seeds matching P_7b architecture."""
    n_rows = len(Y_tr)
    mu_gpu = torch.from_numpy(mu).to(DEV, dtype=torch.float32)
    inv_sd = float(1.0 / sd)

    on_gpu = False
    try:
        A_t = torch.from_numpy(A_tr_f16).to(DEV)
        Y_t = torch.from_numpy(Y_tr).to(DEV)
        on_gpu = True
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        A_t = torch.from_numpy(A_tr_f16)
        Y_t = torch.from_numpy(Y_tr)

    spe = math.ceil(n_rows / BS)
    total_steps = epochs * spe
    nets = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        g_cpu = torch.Generator(device="cpu").manual_seed(seed)
        g_dev = torch.Generator(device=DEV).manual_seed(seed + 100)
        torch.manual_seed(seed)

        net = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(DIM, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 5),
        ).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda s: (s + 1) / spe if s < spe else 0.5 * (1.0 + math.cos(math.pi * (s - spe) / max(1, total_steps - spe))),
        )

        for _ in range(epochs):
            net.train()
            perm = torch.randperm(n_rows, generator=g_cpu)
            if on_gpu:
                perm = perm.to(DEV)
            for i in range(spe):
                idx = perm[i * BS:(i + 1) * BS]
                xb = A_t[idx].to(DEV, dtype=torch.float32)
                xb.sub_(mu_gpu).mul_(inv_sd)
                yb = Y_t[idx].to(DEV)
                yt = F.one_hot(yb, 5).float()
                if rng.random() < p_mix and len(xb) > 1:
                    lam = torch.from_numpy(rng.beta(mix_alpha, mix_alpha, size=(len(xb), 1)).astype(np.float32)).to(DEV)
                    perm_m = torch.randperm(len(xb), generator=g_dev, device=DEV)
                    xb = lam * xb + (1.0 - lam) * xb[perm_m]
                    yt = lam * yt + (1.0 - lam) * yt[perm_m]
                yt = yt * (1.0 - label_smoothing) + label_smoothing / 5.0
                loss = (-yt * F.log_softmax(net(xb), dim=-1)).sum(-1).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()

        net.eval()
        nets.append(net)
        del opt, sched

    del A_t, Y_t, mu_gpu
    return nets, on_gpu


@torch.no_grad()
def predict_rows(nets, mm_path, row_idx, mu, sd, bs=1024):
    mu_gpu = torch.from_numpy(mu).to(DEV, dtype=torch.float32)
    inv_sd = float(1.0 / sd)
    n = len(row_idx)
    probs = np.zeros((n, 5), dtype=np.float64)
    mm = np.load(mm_path, mmap_mode="r")
    try:
        for i in range(0, n, bs):
            sl = row_idx[i:i + bs]
            qb = torch.from_numpy(np.array(mm[sl], dtype=np.float16)).to(DEV, dtype=torch.float32)
            qb.sub_(mu_gpu).mul_(inv_sd)
            probs[i:i + len(sl)] = (sum(net(qb).softmax(-1) for net in nets) / len(nets)).cpu().numpy()
    finally:
        close_mmap_obj(mm)
        del mm, mu_gpu
    return probs


def build_train_matrix(mm_paths, tr_idx):
    n_tr = len(tr_idx)
    A_tr = np.empty((n_tr * len(TRAIN_VIEWS), DIM), dtype=np.float16)
    for i, v in enumerate(TRAIN_VIEWS):
        mm = np.load(mm_paths[f"train_{v}"], mmap_mode="r")
        dst0 = i * n_tr
        try:
            for k in range(0, n_tr, 4096):
                idx_k = tr_idx[k:k + 4096]
                A_tr[dst0 + k:dst0 + k + len(idx_k)] = mm[idx_k]
        finally:
            close_mmap_obj(mm)
            del mm
    return A_tr


def train_subset(mm_paths, y, tr_idx, va_idx, test_len, epochs=EPOCHS, seeds=SEEDS):
    A_tr = build_train_matrix(mm_paths, tr_idx)
    mu, sd = compute_fold_stats(A_tr, len(tr_idx))
    Y_tr = np.tile(y[tr_idx], len(TRAIN_VIEWS))

    if DEV == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    nets, on_gpu = fit_mlp_p7b(A_tr, Y_tr, mu, sd, epochs=epochs, seeds=seeds)
    peak_vram = (torch.cuda.max_memory_allocated() / 2**30) if DEV == "cuda" else 0.0
    del A_tr, Y_tr
    gc.collect()

    vp = predict_rows(nets, mm_paths["train_full"], va_idx, mu, sd) if len(va_idx) > 0 else None
    tp = predict_rows(nets, mm_paths["test_full"], np.arange(test_len), mu, sd)

    del nets
    gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return vp, tp, on_gpu, peak_vram


def run_smoke(y, test_names, log):
    n_s = 256
    log(f"=== SMOKE MODE ({n_s} train & {n_s} test rows, dev={DEV}) ===")
    verify_feature_files(len(y), len(test_names), test_names)
    log(f"Verified source features & test_file_name parity (train={len(y)}, test={len(test_names)}, dim={DIM})")

    mm_paths = build_l2_memmaps(n_s, n_s, prefix="_tmp_p7b_smoke", log=log)
    try:
        tr_idx = np.arange(0, 128)
        va_idx = np.arange(128, n_s)
        vp, tp, on_gpu, peak = train_subset(mm_paths, y, tr_idx, va_idx, test_len=n_s, epochs=2, seeds=(0,))
        assert vp.shape == (len(va_idx), 5) and np.isfinite(vp).all()
        assert tp.shape == (n_s, 5) and np.isfinite(tp).all()
        assert np.allclose(tp.sum(1), 1.0, atol=1e-4)
        log(f"Smoke train+predict OK | on_gpu={on_gpu} | peak VRAM={peak:.2f} GiB")
    finally:
        cleanup_memmaps(mm_paths)

    leftover = list(FEAT_DIR.glob("_tmp_p7b_smoke_*.npy"))
    assert not leftover, f"Leftover temp memmaps: {leftover}"
    log("SMOKE PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Fast 256-row end-to-end verification")
    args = ap.parse_args()

    torch.set_num_threads(8)
    log, log_f = make_logger(LOG_PATH, write_log=not args.smoke)
    try:
        train, test = task4.load_frames()
        y = train.label.values.astype(np.int64)
        test_names = np.array(test.file_name.tolist(), dtype="U")

        if args.smoke:
            run_smoke(y, test_names, log)
            return

        fold_outs = [PRED_DIR / f"t4_probe7b_fold{f}.npz" for f in range(5)]
        full_out = PRED_DIR / "t4_probe7b_full.npz"
        if all(p.exists() for p in fold_outs) and full_out.exists():
            log("All t4_probe7b_fold{0..4}.npz & t4_probe7b_full.npz already exist -> nothing to do")
            return

        log(f"Starting production P_7b training (dev={DEV}, train={len(y)}, test={len(test_names)})")
        verify_feature_files(len(y), len(test_names), test_names)
        oof_ref = np.load(OOF_PARITY_PATH)["p7b_full"] if OOF_PARITY_PATH.exists() else None

        mm_paths = build_l2_memmaps(len(y), len(test_names), prefix="_tmp_p7b_l2", log=log)
        try:
            for f in range(5):
                out_f = PRED_DIR / f"t4_probe7b_fold{f}.npz"
                va = np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"]
                if out_f.exists():
                    z = np.load(out_f)
                    if np.array_equal(z["test_file_name"], test_names) and np.array_equal(z["val_idx"], va):
                        log(f"fold {f}: cache {out_f.name} valid -> skipping")
                        continue
                t_f = time.time()
                tr = np.setdiff1d(np.arange(len(y)), va, assume_unique=True)
                vp, tp, on_gpu, peak = train_subset(mm_paths, y, tr, va, len(test_names), epochs=EPOCHS, seeds=SEEDS)
                assert np.isfinite(vp).all() and np.isfinite(tp).all()
                assert np.allclose(vp.sum(1), 1.0, atol=1e-4) and np.allclose(tp.sum(1), 1.0, atol=1e-4)

                parity_msg = ""
                if oof_ref is not None:
                    ref_va = oof_ref[va]
                    diff = np.abs(vp - ref_va)
                    md, mxd = float(diff.mean()), float(diff.max())
                    agr = float(np.mean(vp.argmax(1) == ref_va.argmax(1)))
                    assert md < 5e-4 and agr >= 0.999, f"fold {f} parity failed: mean={md:.2e}, agree={agr:.4f}"
                    parity_msg = f" | OOF parity mean={md:.2e}, max={mxd:.2e}, agree={agr:.4f}"

                np.savez_compressed(
                    out_f,
                    val_idx=va,
                    val_probs=vp.astype(np.float32),
                    test_probs=tp.astype(np.float32),
                    test_file_name=test_names,
                )
                acc = float(np.mean(vp.argmax(1) == y[va]))
                log(f"fold {f}: saved {out_f.name} in {time.time() - t_f:.1f}s | on_gpu={on_gpu} | peak={peak:.2f} GiB | val_acc={acc:.4f}{parity_msg}")

            if not full_out.exists():
                t_r = time.time()
                tr_all = np.arange(len(y))
                _, tp_full, on_gpu, peak = train_subset(mm_paths, y, tr_all, np.array([], dtype=int), len(test_names), epochs=EPOCHS, seeds=SEEDS)
                assert np.isfinite(tp_full).all() and np.allclose(tp_full.sum(1), 1.0, atol=1e-4)
                np.savez_compressed(
                    full_out,
                    test_probs=tp_full.astype(np.float32),
                    test_file_name=test_names,
                )
                log(f"full refit: saved {full_out.name} in {time.time() - t_r:.1f}s | on_gpu={on_gpu} | peak={peak:.2f} GiB")
        finally:
            cleanup_memmaps(mm_paths)
            log("Cleaned up temporary P_7b L2 memmaps.")
        log("done")
    finally:
        if log_f is not None:
            log_f.close()


if __name__ == "__main__":
    main()
