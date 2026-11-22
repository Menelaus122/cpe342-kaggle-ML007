"""Round 43, Task 4: Gen-2+3 frozen probe P_7b (16,384-d) & P_v3 (31,744-d, 4 backbones) on laptop GPU.

    python experiments/claude/ii_r43_t4_probe_v3.py --smoke
    python experiments/claude/ii_r43_t4_probe_v3.py
    python experiments/claude/ii_r43_t4_probe_v3.py --eval-only

Strictly zero test leakage: only training images / training feature caches are read and evaluated OOF.
Four native-2:1 backbones (total 31,744 dims = 4 x 4 subblocks):
  1. dinov3Hp_144x272  (vit_huge_plus_patch16_dinov3.lvd1689m @ 144x272): 5,120-d  [1280 x 4]
  2. dinov3L_192x384   (vit_large_patch16_dinov3.lvd1689m     @ 192x384): 4,096-d  [1024 x 4]
  3. cradio3L_192x384  (nvidia/C-RADIOv3-L                     @ 192x384): 6,144-d  [3072, 1024 x 3]
  4. dinov3_7b_144x272 (vit_7b_patch16_dinov3.lvd1689m        @ 144x272): 16,384-d [4096 x 4]

RAM-safe execution (~10.5 GiB free RAM budget):
  - Normalizes (per-subblock L2 -> per-backbone L2) in 4096-row chunks into 4 temporary SSD memmaps.
  - Trains one fold at a time (4 views x len(tr) in float16 on CPU, streaming 512-row batches to GPU).
  - Unlinks temporary memmaps cleanly on exit (including explicit Windows mmap handle close).

Pre-declared 12-setting candidate grid vs v38 (v17 rule p * pi^-0.75, balanced class mix):
  - P_v3 @ {0.15, 0.20, 0.25, 0.30, 0.35}
  - P_7b @ {0.15, 0.20, 0.25, 0.30}
  - (P_v2+P_7b)/2 @ {0.20, 0.25, 0.30}
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
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score, log_loss

import common  # noqa: F401  (chdir to repo root)
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_VIEWS = ["full", "zoom70", "centre_grey", "dark_lc"]
EVAL_VIEWS = ["full", "zoom70", "centre_grey"]
OUT_DIR = Path("experiments/claude/outputs/r42")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "ii_probe_v3.log"
VIEWS_OUT_PATH = OUT_DIR / "ii_probe_v3_views.npz"
PV2_VIEWS_PATH = OUT_DIR / "ig_probe_v2_views.npz"

SPECS = [
    dict(
        tag="dinov3Hp_144x272",
        kind="npz",
        path=PRED_DIR / "t4_feat_ml_dinov3Hp_144x272_train.npz",
        dim=5120,
        subblock_dims=[1280, 1280, 1280, 1280],
        col_start=0,
        col_end=5120,
    ),
    dict(
        tag="dinov3L_192x384",
        kind="npz",
        path=PRED_DIR / "t4_feat_ml_dinov3L_192x384_train.npz",
        dim=4096,
        subblock_dims=[1024, 1024, 1024, 1024],
        col_start=5120,
        col_end=9216,
    ),
    dict(
        tag="cradio3L_192x384",
        kind="npz",
        path=PRED_DIR / "t4_feat_ml_cradio3L_192x384_train.npz",
        dim=6144,
        subblock_dims=[3072, 1024, 1024, 1024],
        col_start=9216,
        col_end=15360,
    ),
    dict(
        tag="dinov3_7b_144x272",
        kind="npy_dir",
        path=PRED_DIR / "dinov3_7b_144x272",
        dim=16384,
        subblock_dims=[4096, 4096, 4096, 4096],
        col_start=15360,
        col_end=31744,
    ),
]
TOTAL_DIM = sum(s["dim"] for s in SPECS)  # 31,744
SLICE_P7B = slice(15360, 31744)
SLICE_PV3 = slice(0, 31744)


def make_logger(log_path, write_log):
    t0 = time.time()
    f = open(log_path, "w", encoding="utf-8", buffering=1) if write_log else None

    def log(msg):
        line = f"[{(time.time() - t0) / 60:5.1f}m] {msg}"
        print(line, flush=True)
        if f is not None:
            f.write(line + "\n")
            f.flush()

    return log, f


def close_mmap_obj(mm):
    """Explicitly flush and close a numpy memmap handle so Windows unlink succeeds."""
    if mm is None:
        return
    try:
        mm.flush()
    except Exception:
        pass
    base = getattr(mm, "_mmap", None)
    if base is not None:
        base.close()


def normalize_backbone_view(arr_f16, subdims):
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


def verify_source_caches(n_expected_rows):
    """Verify all 4 backbone training feature caches exist with exact full shape and float16 dtype."""
    for spec in SPECS:
        tag, kind, path, dim = spec["tag"], spec["kind"], spec["path"], spec["dim"]
        if kind == "npz":
            assert path.exists(), f"Missing {tag} cache: {path}"
            with np.load(path) as z:
                for v in TRAIN_VIEWS:
                    assert v in z.files, f"Missing view '{v}' in {path}"
        else:
            for v in TRAIN_VIEWS:
                vp = path / f"train_{v}.npy"
                assert vp.exists(), f"Missing DINOv3-7B cache: {vp}"
                mm = np.load(vp, mmap_mode="r")
                try:
                    assert mm.shape == (n_expected_rows, dim), (
                        f"Unexpected shape in {vp}: {mm.shape}, expected {(n_expected_rows, dim)}"
                    )
                    assert mm.dtype == np.float16, f"Unexpected dtype in {vp}: {mm.dtype}"
                finally:
                    close_mmap_obj(mm)
                    del mm


def build_l2_memmaps(n_rows, n_expected_rows, prefix, log=print):
    """Normalize all 4 backbones in 4096-row chunks into 4 temporary SSD-backed float16 memmaps."""
    t0 = time.time()
    mm_paths = {v: OUT_DIR / f"{prefix}_{v}.npy" for v in TRAIN_VIEWS}
    for v, p in mm_paths.items():
        if p.exists():
            p.unlink()
        mm = np.lib.format.open_memmap(p, mode="w+", dtype=np.float16, shape=(n_rows, TOTAL_DIM))
        close_mmap_obj(mm)
        del mm
    gc.collect()

    chunk = 4096
    for spec in SPECS:
        tag, kind, path, dim, subdims = spec["tag"], spec["kind"], spec["path"], spec["dim"], spec["subblock_dims"]
        c0, c1 = spec["col_start"], spec["col_end"]
        if kind == "npz":
            with np.load(path) as z:
                for v in TRAIN_VIEWS:
                    raw = z[v][:n_rows]
                    assert raw.shape == (n_rows, dim)
                    mm = np.load(mm_paths[v], mmap_mode="r+")
                    try:
                        for i in range(0, n_rows, chunk):
                            normed = normalize_backbone_view(raw[i:i + chunk], subdims)
                            assert np.isfinite(normed).all(), f"Non-finite L2 features in {tag}/{v} rows {i}:{i+len(normed)}"
                            mm[i:i + len(normed), c0:c1] = normed
                    finally:
                        close_mmap_obj(mm)
                        del mm
                    del raw
                    gc.collect()
        else:
            for v in TRAIN_VIEWS:
                src = np.load(path / f"train_{v}.npy", mmap_mode="r")
                dst = np.load(mm_paths[v], mmap_mode="r+")
                try:
                    assert src.shape == (n_expected_rows, dim)
                    for i in range(0, n_rows, chunk):
                        hi = min(i + chunk, n_rows)
                        normed = normalize_backbone_view(src[i:hi], subdims)
                        assert np.isfinite(normed).all(), f"Non-finite L2 features in {tag}/{v} rows {i}:{hi}"
                        dst[i:hi, c0:c1] = normed
                finally:
                    close_mmap_obj(src)
                    close_mmap_obj(dst)
                    del src, dst
                gc.collect()
        log(f"  normalized & wrote {tag} ({dim}-d) -> memmaps [{time.time() - t0:.1f}s]")

    log(f"Built 4 L2 memmaps ({n_rows}x{TOTAL_DIM} float16 each) in {time.time() - t0:.1f}s")
    return mm_paths


def cleanup_memmaps(mm_paths):
    gc.collect()
    for p in mm_paths.values():
        if p.exists():
            p.unlink()


def compute_fold_stats(A_tr_f16, n_tr):
    """Chunked float64 mean & global scalar SD over the `full` training view (rows 0:n_tr)."""
    col_dim = A_tr_f16.shape[1]
    sum_f64 = np.zeros(col_dim, dtype=np.float64)
    chunk = 2048
    for i in range(0, n_tr, chunk):
        sum_f64 += A_tr_f16[i:min(i + chunk, n_tr)].astype(np.float64).sum(axis=0)
    mu = (sum_f64 / n_tr).astype(np.float32)

    sq_sum = 0.0
    mu64 = mu.astype(np.float64)
    for i in range(0, n_tr, chunk):
        diff = A_tr_f16[i:min(i + chunk, n_tr)].astype(np.float64) - mu64
        sq_sum += float((diff * diff).sum())
    sd = float(math.sqrt(sq_sum / (n_tr * col_dim))) + 1e-6
    return mu, sd


def fit_mlp_streaming(A_tr_f16, Y_tr, mu, sd, seed, epochs=30, bs=512, p_mix=0.8, mix_alpha=0.4, label_smoothing=0.1):
    """Train Gen-2 MLP streaming 512-row float16 CPU batches -> GPU float32 standardized."""
    rng = np.random.default_rng(seed)
    g_cpu = torch.Generator(device="cpu").manual_seed(seed)
    g_dev = torch.Generator(device=DEV).manual_seed(seed + 100)
    torch.manual_seed(seed)

    dim = A_tr_f16.shape[1]
    n_rows = len(Y_tr)
    A_cpu = torch.from_numpy(A_tr_f16)  # zero-copy view on CPU numpy array
    Y_cpu = torch.from_numpy(Y_tr)
    mu_gpu = torch.from_numpy(mu).to(DEV, dtype=torch.float32)
    inv_sd = float(1.0 / sd)

    net = nn.Sequential(
        nn.Dropout(0.25),
        nn.Linear(dim, 1024),
        nn.LayerNorm(1024),
        nn.GELU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 5),
    ).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
    spe = math.ceil(n_rows / bs)
    total_steps = epochs * spe
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (s + 1) / spe if s < spe else 0.5 * (1.0 + math.cos(math.pi * (s - spe) / max(1, total_steps - spe))),
    )

    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n_rows, generator=g_cpu)
        for i in range(spe):
            idx = perm[i * bs:(i + 1) * bs]
            xb = A_cpu[idx].to(DEV, dtype=torch.float32)
            xb.sub_(mu_gpu).mul_(inv_sd)
            yb = Y_cpu[idx].to(DEV)
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
    del A_cpu, Y_cpu, opt, sched, g_cpu, g_dev, mu_gpu
    return net


@torch.no_grad()
def predict_from_memmap(nets, mm_path, va_idx, col_start, col_end, mu, sd, bs=1024):
    """Predict averaged softmax probabilities over seed nets reading val rows directly from memmap."""
    mu_gpu = torch.from_numpy(mu).to(DEV, dtype=torch.float32)
    inv_sd = float(1.0 / sd)
    n_va = len(va_idx)
    probs = np.zeros((n_va, 5), dtype=np.float64)
    mm = np.load(mm_path, mmap_mode="r")
    try:
        for i in range(0, n_va, bs):
            idx = va_idx[i:i + bs]
            qb = torch.from_numpy(np.array(mm[idx, col_start:col_end], dtype=np.float16)).to(DEV, dtype=torch.float32)
            qb.sub_(mu_gpu).mul_(inv_sd)
            p_batch = sum(net(qb).softmax(-1) for net in nets) / len(nets)
            probs[i:i + len(idx)] = p_batch.cpu().numpy()
    finally:
        close_mmap_obj(mm)
        del mm, mu_gpu
    return probs


def train_oof_from_memmaps(mm_paths, col_slice, y, folds, fo, tag, n_rows, epochs=30, seeds=(0, 1, 2), log=print):
    c0, c1 = col_slice.start, col_slice.stop
    col_dim = c1 - c0
    pv = {v: np.zeros((n_rows, 5), dtype=np.float64) for v in EVAL_VIEWS}

    for f, va in enumerate(folds):
        t_f = time.time()
        all_rows = np.concatenate(folds)
        # Support both full 5-fold (fo over 0..N-1) and smoke subset folds
        tr = np.setdiff1d(all_rows, va, assume_unique=True) if len(all_rows) != n_rows else np.where(fo != f)[0]
        n_tr = len(tr)

        A_tr = np.empty((n_tr * len(TRAIN_VIEWS), col_dim), dtype=np.float16)
        for i, v in enumerate(TRAIN_VIEWS):
            mm = np.load(mm_paths[v], mmap_mode="r")
            dst0 = i * n_tr
            try:
                for k in range(0, n_tr, 4096):
                    idx_k = tr[k:k + 4096]
                    A_tr[dst0 + k:dst0 + k + len(idx_k)] = mm[idx_k, c0:c1]
            finally:
                close_mmap_obj(mm)
                del mm

        mu, sd = compute_fold_stats(A_tr, n_tr)
        Y_tr = np.tile(y[tr], len(TRAIN_VIEWS))

        torch.cuda.reset_peak_memory_stats() if DEV == "cuda" else None
        nets = [fit_mlp_streaming(A_tr, Y_tr, mu, sd, seed=s, epochs=epochs) for s in seeds]
        peak_vram = (torch.cuda.max_memory_allocated() / 2**30) if DEV == "cuda" else 0.0

        del A_tr, Y_tr
        gc.collect()

        for v in EVAL_VIEWS:
            pv[v][va] = predict_from_memmap(nets, mm_paths[v], va, c0, c1, mu, sd)

        del nets
        gc.collect()
        if DEV == "cuda":
            torch.cuda.empty_cache()

        acc_f = float(np.mean(pv["full"][va].argmax(1) == y[va]))
        log(f"  [{tag}] fold {f}: {len(seeds)} seeds x {epochs} ep in {time.time() - t_f:.1f}s | peak VRAM {peak_vram:.2f} GiB | full val acc {acc_f:.4f}")
    return pv


def calibration_stats(y_true, probs, pred_rule):
    p = np.clip(probs, 1e-7, 1.0)
    lp = np.log(p)
    res = minimize_scalar(
        lambda T: log_loss(y_true, np.exp(lp / T) / np.exp(lp / T).sum(1, keepdims=True)),
        bounds=(0.1, 3.0),
        method="bounded",
    )
    return dict(
        nll=log_loss(y_true, p),
        opt_T=res.x,
        cal_nll=res.fun,
        med_maxP=float(np.median(p.max(1))),
        err_cnt=int((pred_rule != y_true).sum()),
    )


def load_v38_reference(y, folds, fo):
    N = len(y)
    L2, D3, D384, RAD, HP, HP384 = "dinov2L_hud70", "dinov3L_lr8e5", "dinov3L_384", "cradio3L_384", "dinov3Hp_144", "dinov3Hp_384"
    PLAIN, HUDM = ["cnxt", "deit3"], ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
    V20 = PLAIN + HUDM + ["dinov2_v2"]
    ALL_V38 = V20 + [L2, D3, D384, RAD, HP, HP384]
    V19, V30 = Path("experiments/claude/outputs/r19_views"), Path("experiments/claude/outputs/r30_views")

    P = {t: {v: np.zeros((N, 5), dtype=np.float64) for v in EVAL_VIEWS} for t in ALL_V38}
    for t in ALL_V38:
        for f, va in enumerate(folds):
            z30 = np.load(V30 / f"{t}_fold{f}.npz")
            z19 = np.load(V19 / f"{t}_fold{f}.npz") if t in V20 else z30
            for v in EVAL_VIEWS:
                P[t][v][va] = (z30 if v in z30.files else z19)[v]

    PR = dict(np.load("experiments/claude/outputs/r39/p7_probe_views.npz"))
    PV2 = dict(np.load(PV2_VIEWS_PATH))
    ST = dict(np.load("experiments/claude/outputs/r39/t4_view_stats.npz"))
    st30 = np.load("experiments/claude/outputs/r30/t4_errors.npz")

    GATE = {v: np.zeros(N, bool) for v in EVAL_VIEWS}
    for f in range(5):
        tr, ho = fo != f, fo == f
        tb, tc = np.quantile(ST["brightness_full"][tr], 0.10), np.quantile(ST["contrast_full"][tr], 0.10)
        for v in EVAL_VIEWS:
            GATE[v][ho] = (ST[f"brightness_{v}"][ho] <= tb) | (ST[f"contrast_{v}"][ho] <= tc)

    dark = (st30["contrast"] <= np.quantile(st30["contrast"], 0.1)) | (st30["brightness"] <= np.quantile(st30["brightness"], 0.1))
    W_NORM = {**{t: 0.25 for t in PLAIN}, **{t: 0.75 for t in HUDM}, "dinov2_v2": 8.0,
              L2: 8.0, D3: 10.0, D384: 20.0, RAD: 10.0, HP: 40.0, HP384: 40.0}
    W_DARK = {**W_NORM, L2: 2.0, RAD: 2.0}

    FT_v38 = {}
    for v in EVAL_VIEWS:
        num_n = sum(W_NORM[t] * P[t][v] for t in ALL_V38)
        den_n = sum(W_NORM.values())
        num_d = sum(W_DARK[t] * P[t][v] for t in ALL_V38)
        den_d = sum(W_DARK.values())
        FT_v38[v] = np.where(GATE[v][:, None], num_d / den_d, num_n / den_n)

    return PR, PV2, FT_v38, P[HP384], dark


def evaluate_v3(P7B, PV3, y, folds, fo, log=print):
    PR, PV2, FT_v38, HP384_P, dark = load_v38_reference(y, folds, fo)
    N = len(y)
    pi = np.bincount(y, minlength=5) / N
    wb = (0.2 / pi)[y]
    bal = lambda pred, m: f1_score(y[m], pred[m], average="macro", sample_weight=wb[m])
    rule = lambda p: (p * pi ** -0.75).argmax(1)

    allm = np.ones(N, bool)
    SC = ["full", "zoom70", "centre_grey", "dark/low-contrast"]
    VIEW_OF = {"full": "full", "zoom70": "zoom70", "centre_grey": "centre_grey", "dark/low-contrast": "full"}
    msk = lambda k, m=allm: m & dark if k == "dark/low-contrast" else m

    ref = {v: rule(0.70 * FT_v38[v] + 0.30 * PR[v]) for v in EVAL_VIEWS}
    ref_scores = {k: bal(ref[VIEW_OF[k]], msk(k)) for k in SC}
    log("v38 reference: " + ", ".join(f"{k} {ref_scores[k]:.4f}" for k in SC))
    for k, exp_v in {"full": 0.9956, "zoom70": 0.9890, "centre_grey": 0.9821, "dark/low-contrast": 0.9890}.items():
        assert abs(ref_scores[k] - exp_v) <= 1.5e-4, f"v38 mismatch on {k}: got {ref_scores[k]:.5f}, expected {exp_v:.4f}"

    ref_pv2 = {v: rule(0.80 * FT_v38[v] + 0.20 * PV2[v]) for v in EVAL_VIEWS}
    pv2_d = {k: bal(ref_pv2[VIEW_OF[k]], msk(k)) - ref_scores[k] for k in SC}
    log("v38 + P_v2 @ 0.20 baseline vs v38: " + ", ".join(f"{k} {pv2_d[k]:+.4f}" for k in SC))

    # 1. Probe-alone diagnostics
    P_AVG = {v: 0.5 * PV2[v] + 0.5 * P7B[v] for v in EVAL_VIEWS}
    log("\n--- Probe alone diagnostics (v17 rule, balanced mix) ---")
    for nm, pd_map in [
        ("P7 (v38)", PR),
        ("P_v2 (3bb 15360d)", PV2),
        ("P_7b (7B 16384d)", P7B),
        ("P_v3 (4bb 31744d)", PV3),
        ("P_v2+P_7b (50/50)", P_AVG),
    ]:
        scores_str = ", ".join(f"{k} {bal(rule(pd_map[VIEW_OF[k]]), msk(k)):.4f}({(rule(pd_map[VIEW_OF[k]])[msk(k)] != y[msk(k)]).sum()})" for k in SC)
        cs = calibration_stats(y, pd_map["full"], rule(pd_map["full"]))
        log(f"  {nm:19s}: {scores_str} | full NLL={cs['nll']:.4f} optT={cs['opt_T']:.3f} calNLL={cs['cal_nll']:.4f} med_maxP={cs['med_maxP']:.3f}")

    # 2. Pre-declared 12-setting candidate grid vs v38
    C = {}
    for pw in (0.15, 0.20, 0.25, 0.30, 0.35):
        C[f"P_v3 @ {pw:.2f}"] = {v: rule((1.0 - pw) * FT_v38[v] + pw * PV3[v]) for v in EVAL_VIEWS}
    for pw in (0.15, 0.20, 0.25, 0.30):
        C[f"P_7b @ {pw:.2f}"] = {v: rule((1.0 - pw) * FT_v38[v] + pw * P7B[v]) for v in EVAL_VIEWS}
    for pw in (0.20, 0.25, 0.30):
        C[f"P_v2+P_7b @ {pw:.2f}"] = {v: rule((1.0 - pw) * FT_v38[v] + pw * P_AVG[v]) for v in EVAL_VIEWS}

    deltas = lambda lab, base=ref, m=allm: {k: bal(lab[VIEW_OF[k]], msk(k, m)) - bal(base[VIEW_OF[k]], msk(k, m)) for k in SC}
    meets = lambda d: d["zoom70"] > 0 and d["centre_grey"] > 0 and d["dark/low-contrast"] > 0 and d["full"] >= -0.0005
    crit = lambda d: float(np.mean([d["zoom70"], d["centre_grey"], d["dark/low-contrast"]]))

    def report(nm, lab, base=ref, base_name="v38", check_pass=True):
        d = deltas(lab, base=base)
        pf = [np.mean([bal(lab[v], fo == f) - bal(base[v], fo == f) for v in ("zoom70", "centre_grey")]) for f in range(5)]
        pdk = [bal(lab["full"], (fo == f) & dark) - bal(base["full"], (fo == f) & dark) for f in range(5)]
        ok = meets(d) and sum(x > 0 for x in pf) >= 3
        verdict = f"\n    -> {'PASS' if ok else 'fail'} (vs {base_name})" if check_pass else ""
        log(
            f"  {nm}\n    "
            + ", ".join(f"{k} {d[k]:+.4f}" for k in SC)
            + f"\n    zoom / HUD by fold {' '.join(f'{x:+.4f}' for x in pf)} | dark by fold {' '.join(f'{x:+.4f}' for x in pdk)}"
            + f" | labels changed (full) {(lab['full'] != base['full']).sum()}{verdict}"
        )
        return ok

    log("\n--- Pre-declared 12 candidates vs v38 (in-sample) ---")
    ins_passing = {}
    for nm, lab in C.items():
        report(nm, lab)
        d = deltas(lab)
        if meets(d):
            ins_passing[nm] = crit(d)

    # 3. Strict Nested 5-fold selection
    outN, picks = {v: ref[v].copy() for v in EVAL_VIEWS}, []
    for f in range(5):
        okk = {nm: crit(d) for nm, d in ((nm, deltas(lab, ref, fo != f)) for nm, lab in C.items()) if meets(d)}
        pick = max(okk, key=okk.get) if okk else None
        picks.append(pick)
        if pick is not None:
            for v in EVAL_VIEWS:
                outN[v][fo == f] = C[pick][v][fo == f]

    log("\n--- Nested choice per held-out fold ---")
    for f, pk in enumerate(picks):
        log(f"  fold {f}: {pk if pk else 'v38 (no candidate meets rule)'}")

    log("\nCandidate (decision vs v38):")
    okN = report("N nested over 12-setting grid", outN)
    best_ins = max(ins_passing, key=ins_passing.get) if ins_passing else None

    # 4. Head-to-head vs v38 + P_v2 @ 0.20
    log("\n--- Head-to-head comparison vs (v38 + P_v2 @ 0.20) ---")
    report("N nested (v3) vs (v38 + P_v2 @ 0.20)", outN, base=ref_pv2, base_name="v38+P_v2", check_pass=False)
    if best_ins is not None:
        report(f"best in-sample ({best_ins}) vs (v38 + P_v2 @ 0.20)", C[best_ins], base=ref_pv2, base_name="v38+P_v2", check_pass=False)

    # 5. Vote-out & error table
    hp384_lab = {v: rule(HP384_P[v]) for v in EVAL_VIEWS}
    log("\n--- Vote-out diagnostic (dinov3Hp_384 vs v38 vs P_v2 vs N nested (v3) / best) ---")
    rows_diag = [("v38", ref), ("v38 + P_v2 @ 0.20", ref_pv2), ("N nested (v3)", outN)]
    if best_ins is not None:
        rows_diag.append((f"best ({best_ins})", C[best_ins]))
    for label_nm, lab in rows_diag:
        vo_full = int(((hp384_lab["full"] == y) & (lab["full"] != y)).sum())
        res_full = int(((hp384_lab["full"] != y) & (lab["full"] == y)).sum())
        vo_dark = int(((hp384_lab["full"] == y) & (lab["full"] != y) & dark).sum())
        res_dark = int(((hp384_lab["full"] != y) & (lab["full"] == y) & dark).sum())
        err_f = int((lab["full"] != y).sum())
        err_d = int(((lab["full"] != y) & dark).sum())
        log(f"  {label_nm:24s}: full err={err_f:3d} (Hp384 vote-out={vo_full:2d}, rescued={res_full:2d}) | dark err={err_d:2d} (vote-out={vo_dark:2d}, rescued={res_dark:2d})")

    log(f"\n-> N {'PASSES' if okN else 'fails'}; best in-sample passing setting: {best_ins}")
    return okN


def run_smoke(y, folds, fo, log):
    n_smoke = 256
    log(f"=== SMOKE MODE ({n_smoke} training rows, dev={DEV}) ===")
    verify_source_caches(len(y))
    log(f"Verified all 4 backbone training caches (full N={len(y)}, total_dim={TOTAL_DIM})")

    mm_paths = build_l2_memmaps(n_smoke, len(y), prefix="_tmp_pv3_smoke_l2", log=log)
    try:
        smoke_folds = [np.arange(0, 128), np.arange(128, 256)]
        smoke_fo = np.zeros(n_smoke, int)
        smoke_fo[128:] = 1
        p7b_s = train_oof_from_memmaps(mm_paths, SLICE_P7B, y[:n_smoke], smoke_folds, smoke_fo, "smoke P_7b", n_smoke, epochs=2, seeds=(0,), log=log)
        pv3_s = train_oof_from_memmaps(mm_paths, SLICE_PV3, y[:n_smoke], smoke_folds, smoke_fo, "smoke P_v3", n_smoke, epochs=2, seeds=(0,), log=log)
        for name_m, d_m in [("P_7b", p7b_s), ("P_v3", pv3_s)]:
            for v in EVAL_VIEWS:
                assert d_m[v].shape == (n_smoke, 5) and np.isfinite(d_m[v]).all(), f"Non-finite smoke output in {name_m}/{v}"
    finally:
        cleanup_memmaps(mm_paths)

    leftover = [p for p in OUT_DIR.glob("_tmp_pv3_smoke_l2_*.npy")]
    assert not leftover, f"Temp smoke memmaps failed to unlink: {leftover}"
    log("Temp smoke memmaps cleaned up cleanly")

    # Verify v38 reference + P_v2 load
    PR, PV2, FT_v38, _, dark = load_v38_reference(y, folds, fo)
    assert all(k in PV2 and PV2[k].shape == (len(y), 5) for k in EVAL_VIEWS), "P_v2 OOF views invalid"
    pi = np.bincount(y, minlength=5) / len(y)
    wb = (0.2 / pi)[y]
    bal = lambda pred, m: f1_score(y[m], pred[m], average="macro", sample_weight=wb[m])
    rule = lambda p: (p * pi ** -0.75).argmax(1)
    ref = {v: rule(0.70 * FT_v38[v] + 0.30 * PR[v]) for v in EVAL_VIEWS}
    SC = ["full", "zoom70", "centre_grey", "dark/low-contrast"]
    VIEW_OF = {"full": "full", "zoom70": "zoom70", "centre_grey": "centre_grey", "dark/low-contrast": "full"}
    allm = np.ones(len(y), bool)
    ref_scores = {k: bal(ref[VIEW_OF[k]], dark if k == "dark/low-contrast" else allm) for k in SC}
    log("v38 reference verified: " + ", ".join(f"{k} {ref_scores[k]:.4f}" for k in SC))
    for k, exp_v in {"full": 0.9956, "zoom70": 0.9890, "centre_grey": 0.9821, "dark/low-contrast": 0.9890}.items():
        assert abs(ref_scores[k] - exp_v) <= 1.5e-4, f"v38 mismatch on {k}: {ref_scores[k]}"
    log("SMOKE PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Fast 256-row end-to-end check on memmap, P_7b, P_v3, cleanup & v38 ref")
    ap.add_argument("--eval-only", action="store_true", help="Evaluate saved ii_probe_v3_views.npz without retraining")
    args = ap.parse_args()

    torch.set_num_threads(8)
    log, log_f = make_logger(LOG_PATH, write_log=not args.smoke)
    try:
        train, _ = task4.load_frames()
        y = train.label.values.astype(np.int64)
        folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
        fo = np.zeros(len(y), int)
        for f, va in enumerate(folds):
            fo[va] = f

        if args.smoke:
            run_smoke(y, folds, fo, log)
            return

        if args.eval_only:
            assert VIEWS_OUT_PATH.exists(), f"Missing {VIEWS_OUT_PATH}"
            z = np.load(VIEWS_OUT_PATH)
            P7B = {v: z[f"p7b_{v}"] for v in EVAL_VIEWS}
            PV3 = {v: z[f"pv3_{v}"] for v in EVAL_VIEWS}
            evaluate_v3(P7B, PV3, y, folds, fo, log=log)
            return

        log(f"Starting P_7b (16384-d) & P_v3 (31744-d) on device={DEV}, N={len(y)}")
        verify_source_caches(len(y))
        mm_paths = build_l2_memmaps(len(y), len(y), prefix="_tmp_pv3_l2", log=log)
        try:
            log("Training 5-fold P_7b (DINOv3-7B 16384-d, 3 seeds x 30 ep)...")
            P7B = train_oof_from_memmaps(mm_paths, SLICE_P7B, y, folds, fo, "P_7b", len(y), epochs=30, seeds=(0, 1, 2), log=log)

            log("Training 5-fold P_v3 (4 backbones 31744-d, 3 seeds x 30 ep)...")
            PV3 = train_oof_from_memmaps(mm_paths, SLICE_PV3, y, folds, fo, "P_v3", len(y), epochs=30, seeds=(0, 1, 2), log=log)
        finally:
            cleanup_memmaps(mm_paths)
            log("Cleaned up temporary L2 memmaps.")

        save_dict = {
            **{f"p7b_{v}": P7B[v].astype(np.float32) for v in EVAL_VIEWS},
            **{f"pv3_{v}": PV3[v].astype(np.float32) for v in EVAL_VIEWS},
        }
        np.savez(VIEWS_OUT_PATH, **save_dict)
        log(f"Saved P_7b & P_v3 OOF views to {VIEWS_OUT_PATH}")

        evaluate_v3(P7B, PV3, y, folds, fo, log=log)
        log("done")
    finally:
        if log_f is not None:
            log_f.close()


if __name__ == "__main__":
    main()
