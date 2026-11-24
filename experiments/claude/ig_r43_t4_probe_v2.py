"""Round 43, Task 4: Gen-2 native-2:1 multi-layer Mixup frozen probe (P_v2) on laptop GPU (0 Colab CU; training images only).

    python experiments/claude/ig_r43_t4_probe_v2.py --smoke
    python experiments/claude/ig_r43_t4_probe_v2.py

Strictly zero test leakage: only training images are read, embedded, and evaluated out-of-fold on the 5 pipeline folds.
Three native-2:1 backbones (total 15,360 dims):
  1. vit_huge_plus_patch16_dinov3.lvd1689m @ 144x272 (bf16 weights, last 4 blocks: mid_patch, last_patch, last_cls, last_std -> 5,120-d)
  2. vit_large_patch16_dinov3.lvd1689m     @ 192x384 (bf16 weights, last 4 blocks: mid_patch, last_patch, last_cls, last_std -> 4,096-d)
  3. nvidia/C-RADIOv3-L (9d041346)         @ 192x384 (fp32 weights + bf16 autocast, summary + last 4 blocks spatial/std -> 6,144-d)

Four training views per image: full, zoom70, centre_grey, and mild dark/low-contrast (dark_lc).
Preprocessing: per-subblock L2-norm -> per-backbone L2-norm -> per-fold training full-frame mean & global scalar SD.
MLP (3 seeds): Dropout(0.25) -> Linear(D, 1024) -> LayerNorm -> GELU -> Dropout(0.5) -> Linear(1024, 5),
trained with Feature-Space Mixup (p=0.8, Beta(0.4, 0.4)) + label smoothing 0.1.

Pre-declared 8-setting candidate grid vs v38 (v17 rule p * pi^-0.75, balanced class mix):
  - P_v2 @ pw for pw in {0.15, 0.20, 0.25, 0.30}
  - P7+P_v2 (50/50) @ pw for pw in {0.20, 0.25, 0.30, 0.35}
Nested N: per held-out fold f, pick the candidate meeting (zoom70 > 0, centre_grey > 0, dark > 0, full >= -0.0005) on fo != f with
highest mean(zoom70, centre_grey, dark); fallback v38. Acceptance: N meets the same 4-slice criteria and zoom/HUD mean > 0 in >= 3/5 folds.
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

sys.path.insert(0, "scripts")
from src import task4  # noqa: E402
from src.config import PRED_DIR  # noqa: E402
import train_task4_probe as P0  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_VIEWS = ["full", "zoom70", "centre_grey", "dark_lc"]
EVAL_VIEWS = ["full", "zoom70", "centre_grey"]
OUT_DIR = Path("experiments/claude/outputs/r42")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "ig_probe_v2.log"
VIEWS_OUT_PATH = OUT_DIR / "ig_probe_v2_views.npz"

RADIO_REPO = "nvidia/C-RADIOv3-L"
RADIO_REV = "9d0413465e8a91e67bbf2c1ad342815478d1b906"

SPECS = [
    dict(
        tag="dinov3Hp_144x272",
        kind="dinov3",
        model_name="vit_huge_plus_patch16_dinov3.lvd1689m",
        hw=(144, 272),
        bs=64,
        dim=5120,
        subblock_dims=[1280, 1280, 1280, 1280],
    ),
    dict(
        tag="dinov3L_192x384",
        kind="dinov3",
        model_name="vit_large_patch16_dinov3.lvd1689m",
        hw=(192, 384),
        bs=64,
        dim=4096,
        subblock_dims=[1024, 1024, 1024, 1024],
    ),
    dict(
        tag="cradio3L_192x384",
        kind="radio",
        model_name=RADIO_REPO,
        hw=(192, 384),
        bs=64,
        dim=6144,
        subblock_dims=[3072, 1024, 1024, 1024],
    ),
]


def dark_lc(x):
    """Mild dark / low-contrast view targeting real dark/low-contrast errors without synthetic noise."""
    xg = x.clamp_min(1e-6) ** 1.8
    m = xg.mean(dim=(2, 3), keepdim=True)
    return (m + 0.55 * (xg - m)).clamp(0.0, 1.0)


VIEW_FNS = {
    "full": lambda x: x,
    "zoom70": lambda x: P0.zoom(x, 0.7),
    "centre_grey": P0.centre_grey,
    "dark_lc": dark_lc,
}


def make_logger(log_path, write_log):
    t0 = time.time()
    f = open(log_path, "w", encoding="utf-8", buffering=1) if write_log else None

    def log(msg):
        line = f"[{ (time.time() - t0) / 60:5.1f}m] {msg}"
        print(line, flush=True)
        if f is not None:
            f.write(line + "\n")
            f.flush()

    return log, f


def extract_dinov3_batch(model, mean, std, x_01):
    x = ((x_01 - mean) / std).to(torch.bfloat16)

    def _fwd(z):
        _, inters = model.forward_intermediates(z, indices=4, return_prefix_tokens=True, norm=True, output_fmt="NLC")
        mid_patch = torch.stack([inters[k][0].float().mean(1) for k in (0, 1, 2)], dim=0).mean(0)
        last_patch = inters[3][0].float().mean(1)
        last_cls = inters[3][1][:, 0].float()
        last_std = inters[3][0].float().std(1, unbiased=False)
        return torch.cat([mid_patch, last_patch, last_cls, last_std], dim=1)

    return 0.5 * (_fwd(x) + _fwd(torch.flip(x, dims=[3])))


def extract_radio_batch(model, captured, x_01, check_state):
    def _fwd(z):
        captured.clear()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            summary, spatial = model(z)
        assert len(captured) == 4, f"Expected 4 captured RADIO blocks, got {len(captured)}"
        n_sp = spatial.shape[1]
        if not check_state["checked"]:
            diff = (captured[-1][:, -n_sp:].float() - spatial.float()).abs().max().item()
            assert diff < 1e-2, f"C-RADIOv3-L spatial alignment failed: max |diff|={diff}"
            check_state["checked"] = True
        mid_spatial = torch.stack([captured[k][:, -n_sp:].float().mean(1) for k in (0, 1, 2)], dim=0).mean(0)
        last_spatial = spatial.float().mean(1)
        last_std = spatial.float().std(1, unbiased=False)
        res = torch.cat([summary.float(), mid_spatial, last_spatial, last_std], dim=1)
        captured.clear()
        return res

    return 0.5 * (_fwd(x_01) + _fwd(torch.flip(x_01, dims=[3])))


def cache_valid(path, n_expected, dim_expected):
    if not path.exists():
        return False
    try:
        with np.load(path) as z:
            for v in TRAIN_VIEWS:
                if v not in z.files or z[v].shape != (n_expected, dim_expected):
                    return False
        return True
    except Exception:
        return False


def extract_backbone(spec, paths, log, save_cache=True):
    tag, kind, name, hw, bs, dim, subdims = (
        spec["tag"], spec["kind"], spec["model_name"], spec["hw"], spec["bs"], spec["dim"], spec["subblock_dims"]
    )
    cache_path = PRED_DIR / f"t4_feat_ml_{tag}_train.npz"
    if save_cache and cache_valid(cache_path, len(paths), dim):
        log(f"{tag}: valid cache found at {cache_path.name} ({len(paths)}, {dim}) -> skipping extraction")
        return None

    t1 = time.time()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    hooks = []
    captured = []
    check_state = {"checked": False}

    if kind == "dinov3":
        import timm
        model = timm.create_model(name, pretrained=True, num_classes=0, img_size=hw).cuda().to(torch.bfloat16).eval()
        cfg = timm.data.resolve_model_data_config(model)
        mean = torch.tensor(cfg["mean"], device="cuda").view(1, 3, 1, 1)
        std = torch.tensor(cfg["std"], device="cuda").view(1, 3, 1, 1)
    else:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(name, trust_remote_code=True, revision=RADIO_REV).cuda().eval()
        vit = model.radio_model.model
        hooks = [blk.register_forward_hook(lambda mod, inp, out: captured.append(out.detach())) for blk in vit.blocks[-4:]]

    feats = {v: np.empty((len(paths), dim), dtype=np.float16) for v in TRAIN_VIEWS}
    try:
        with torch.no_grad():
            for i in range(0, len(paths), 4096):
                chunk_paths = list(paths[i:i + 4096])
                arr = task4.load_uint8(chunk_paths)
                for j in range(0, len(arr), bs):
                    x0 = torch.from_numpy(arr[j:j + bs]).cuda().permute(0, 3, 1, 2).float().div_(255)
                    row0, row1 = i + j, i + j + len(x0)
                    for v in TRAIN_VIEWS:
                        xv = F.interpolate(VIEW_FNS[v](x0), size=hw, mode="bicubic", align_corners=False).clamp_(0, 1)
                        if kind == "dinov3":
                            f = extract_dinov3_batch(model, mean, std, xv)
                        else:
                            f = extract_radio_batch(model, captured, xv, check_state)
                        assert torch.isfinite(f).all(), f"Non-finite features in {tag}/{v} rows {row0}:{row1}"
                        feats[v][row0:row1] = f.cpu().numpy().astype(np.float16)
                del arr
                if (i + 4096) < len(paths):
                    done = min(i + 4096, len(paths))
                    el = (time.time() - t1) / 60
                    eta = el * (len(paths) - done) / done
                    log(f"  {tag}: {done}/{len(paths)} images ({el:.1f}m elapsed, ~{eta:.1f}m left)")
        peak = torch.cuda.max_memory_allocated() / 2**30
        if save_cache:
            np.savez(cache_path, **feats, subblock_dims=np.array(subdims, dtype=np.int32))
        log(f"{tag}: extracted {len(paths)}x{dim} x 4 views in {(time.time() - t1) / 60:.1f} min | peak VRAM {peak:.2f} GiB")
    finally:
        for h in hooks:
            h.remove()
        del model
        captured.clear()
        gc.collect()
        torch.cuda.empty_cache()

    return feats if not save_cache else None


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


def load_normalized_features(n_rows, smoke_feats=None, log=print):
    total_dim = sum(s["dim"] for s in SPECS)
    X_l2 = {v: np.empty((n_rows, total_dim), dtype=np.float16) for v in TRAIN_VIEWS}
    col = 0
    for spec in SPECS:
        tag, dim, subdims = spec["tag"], spec["dim"], spec["subblock_dims"]
        if smoke_feats is not None:
            raw = smoke_feats[tag]
            for v in TRAIN_VIEWS:
                X_l2[v][:, col:col + dim] = normalize_backbone_view(raw[v], subdims)
        else:
            cache_path = PRED_DIR / f"t4_feat_ml_{tag}_train.npz"
            with np.load(cache_path) as z:
                for v in TRAIN_VIEWS:
                    X_l2[v][:, col:col + dim] = normalize_backbone_view(z[v], subdims)
        col += dim
        gc.collect()
    assert col == total_dim
    log(f"Loaded & L2-normalized {len(SPECS)} backbones -> {total_dim} dims x {len(TRAIN_VIEWS)} views in float16")
    return X_l2


def fit_mlp_v2(A_np, Y_np, seed, epochs=30, bs=512, p_mix=0.8, mix_alpha=0.4, label_smoothing=0.1):
    """Train Gen-2 MLP with LayerNorm + Feature-Space Mixup + soft cross-entropy."""
    rng = np.random.default_rng(seed)
    g_cpu = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)

    dim = A_np.shape[1]
    n_rows = len(Y_np)
    on_gpu = False
    try:
        A_t = torch.from_numpy(A_np).to(DEV)
        Y_t = torch.from_numpy(Y_np).to(DEV)
        on_gpu = True
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        A_t = torch.from_numpy(A_np)
        Y_t = torch.from_numpy(Y_np)

    g_dev = torch.Generator(device=DEV if on_gpu else "cpu").manual_seed(seed + 100)
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
        if on_gpu:
            perm = perm.to(DEV)
        for i in range(spe):
            idx = perm[i * bs:(i + 1) * bs]
            xb = A_t[idx].to(DEV, dtype=torch.float32)
            yb = Y_t[idx].to(DEV)
            yt = F.one_hot(yb, 5).float()
            if rng.random() < p_mix and len(xb) > 1:
                lam = torch.from_numpy(rng.beta(mix_alpha, mix_alpha, size=(len(xb), 1)).astype(np.float32)).to(DEV)
                perm_m = torch.randperm(len(xb), generator=g_dev, device=xb.device)
                xb = lam * xb + (1.0 - lam) * xb[perm_m]
                yt = lam * yt + (1.0 - lam) * yt[perm_m]
            yt = yt * (1.0 - label_smoothing) + label_smoothing / 5.0
            loss = (-yt * F.log_softmax(net(xb), dim=-1)).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

    net.eval()

    def predict(Q_np):
        outs = []
        with torch.no_grad():
            for i in range(0, len(Q_np), 1024):
                qb = torch.from_numpy(Q_np[i:i + 1024]).to(DEV, dtype=torch.float32)
                outs.append(net(qb).softmax(-1).cpu().numpy())
        return np.vstack(outs)

    del A_t, Y_t
    return predict


def train_oof_probe(X_l2, y, folds, fo, epochs=30, seeds=(0, 1, 2), log=print):
    n_rows, dim = X_l2["full"].shape
    pv2 = {v: np.zeros((n_rows, 5), dtype=np.float64) for v in EVAL_VIEWS}
    for f, va in enumerate(folds):
        t_f = time.time()
        tr = np.where(fo != f)[0]
        xf = X_l2["full"][tr].astype(np.float32)
        mu = xf.mean(0)
        sd = float(np.sqrt(((xf - mu) ** 2).mean())) + 1e-6
        del xf
        gc.collect()

        A = np.empty((len(tr) * len(TRAIN_VIEWS), dim), dtype=np.float16)
        for i, v in enumerate(TRAIN_VIEWS):
            A[i * len(tr):(i + 1) * len(tr)] = ((X_l2[v][tr].astype(np.float32) - mu) / sd).astype(np.float16)
        Y = np.tile(y[tr], len(TRAIN_VIEWS))

        preds = [fit_mlp_v2(A, Y, s, epochs=epochs) for s in seeds]
        del A, Y
        gc.collect()
        torch.cuda.empty_cache()

        for v in EVAL_VIEWS:
            Q = ((X_l2[v][va].astype(np.float32) - mu) / sd).astype(np.float32)
            pv2[v][va] = np.mean([p(Q) for p in preds], axis=0)
            del Q
        del preds
        gc.collect()
        torch.cuda.empty_cache()
        acc_f = np.mean(pv2["full"][va].argmax(1) == y[va])
        log(f"  fold {f}: P_v2 trained ({len(seeds)} seeds x {epochs} ep) in {time.time() - t_f:.1f}s | full val acc {acc_f:.4f}")
    return pv2


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


def evaluate_against_v38(PV2, y, folds, fo, log=print):
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
    ST = dict(np.load("experiments/claude/outputs/r39/t4_view_stats.npz"))
    st30 = np.load("experiments/claude/outputs/r30/t4_errors.npz")

    GATE = {v: np.zeros(N, bool) for v in EVAL_VIEWS}
    for f in range(5):
        tr, ho = fo != f, fo == f
        tb, tc = np.quantile(ST["brightness_full"][tr], 0.10), np.quantile(ST["contrast_full"][tr], 0.10)
        for v in EVAL_VIEWS:
            GATE[v][ho] = (ST[f"brightness_{v}"][ho] <= tb) | (ST[f"contrast_{v}"][ho] <= tc)

    dark = (st30["contrast"] <= np.quantile(st30["contrast"], 0.1)) | (st30["brightness"] <= np.quantile(st30["brightness"], 0.1))
    pi = np.bincount(y, minlength=5) / N
    wb = (0.2 / pi)[y]
    bal = lambda pred, m: f1_score(y[m], pred[m], average="macro", sample_weight=wb[m])
    rule = lambda p: (p * pi ** -0.75).argmax(1)

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

    allm = np.ones(N, bool)
    SC = ["full", "zoom70", "centre_grey", "dark/low-contrast"]
    VIEW_OF = {"full": "full", "zoom70": "zoom70", "centre_grey": "centre_grey", "dark/low-contrast": "full"}
    msk = lambda k, m=allm: m & dark if k == "dark/low-contrast" else m

    ref = {v: rule(0.70 * FT_v38[v] + 0.30 * PR[v]) for v in EVAL_VIEWS}
    ref_scores = {k: bal(ref[VIEW_OF[k]], msk(k)) for k in SC}
    log("v38 reference: " + ", ".join(f"{k} {ref_scores[k]:.4f}" for k in SC))
    expected_ref = {"full": 0.9956, "zoom70": 0.9890, "centre_grey": 0.9821, "dark/low-contrast": 0.9890}
    for k, exp_v in expected_ref.items():
        assert abs(ref_scores[k] - exp_v) <= 1.5e-4, f"v38 mismatch on {k}: got {ref_scores[k]:.5f}, expected {exp_v:.4f}"

    # Diagnostics: P_v2 alone vs P7 alone
    log("\n--- Probe alone diagnostics (v17 rule, balanced mix) ---")
    for nm, pd_map in [("P7 (v38)", PR), ("P_v2 (ours)", PV2), ("P7+P_v2 (50/50)", {v: 0.5 * PR[v] + 0.5 * PV2[v] for v in EVAL_VIEWS})]:
        scores_str = ", ".join(f"{k} {bal(rule(pd_map[VIEW_OF[k]]), msk(k)):.4f}({(rule(pd_map[VIEW_OF[k]])[msk(k)] != y[msk(k)]).sum()})" for k in SC)
        cs = calibration_stats(y, pd_map["full"], rule(pd_map["full"]))
        log(f"  {nm:16s}: {scores_str} | full NLL={cs['nll']:.4f} optT={cs['opt_T']:.3f} calNLL={cs['cal_nll']:.4f} med_maxP={cs['med_maxP']:.3f}")

    # Pre-declared 8-setting candidate grid
    P50 = {v: 0.5 * PR[v] + 0.5 * PV2[v] for v in EVAL_VIEWS}
    C = {}
    for pw in (0.15, 0.20, 0.25, 0.30):
        C[f"P_v2 @ {pw:.2f}"] = {v: rule((1.0 - pw) * FT_v38[v] + pw * PV2[v]) for v in EVAL_VIEWS}
    for pw in (0.20, 0.25, 0.30, 0.35):
        C[f"P7+P_v2 @ {pw:.2f}"] = {v: rule((1.0 - pw) * FT_v38[v] + pw * P50[v]) for v in EVAL_VIEWS}

    deltas = lambda lab, m=allm: {k: bal(lab[VIEW_OF[k]], msk(k, m)) - bal(ref[VIEW_OF[k]], msk(k, m)) for k in SC}
    meets = lambda d: d["zoom70"] > 0 and d["centre_grey"] > 0 and d["dark/low-contrast"] > 0 and d["full"] >= -0.0005
    crit = lambda d: float(np.mean([d["zoom70"], d["centre_grey"], d["dark/low-contrast"]]))

    def report(nm, lab):
        d = deltas(lab)
        pf = [np.mean([bal(lab[v], fo == f) - bal(ref[v], fo == f) for v in ("zoom70", "centre_grey")]) for f in range(5)]
        pdk = [bal(lab["full"], (fo == f) & dark) - bal(ref["full"], (fo == f) & dark) for f in range(5)]
        ok = meets(d) and sum(x > 0 for x in pf) >= 3
        log(
            f"  {nm}\n    "
            + ", ".join(f"{k} {d[k]:+.4f}" for k in SC)
            + f"\n    zoom / HUD by fold {' '.join(f'{x:+.4f}' for x in pf)} | dark by fold {' '.join(f'{x:+.4f}' for x in pdk)}"
            + f" | labels changed (full) {(lab['full'] != ref['full']).sum()}\n    -> {'PASS' if ok else 'fail'}"
        )
        return ok

    log("\n--- Pre-declared 8 candidates vs v38 (in-sample) ---")
    ins_passing = {}
    for nm, lab in C.items():
        report(nm, lab)
        d = deltas(lab)
        if meets(d):
            ins_passing[nm] = crit(d)

    # Nested 5-fold selection
    outN, picks = {v: ref[v].copy() for v in EVAL_VIEWS}, []
    for f in range(5):
        okk = {nm: crit(d) for nm, d in ((nm, deltas(lab, fo != f)) for nm, lab in C.items()) if meets(d)}
        pick = max(okk, key=okk.get) if okk else None
        picks.append(pick)
        if pick is not None:
            for v in EVAL_VIEWS:
                outN[v][fo == f] = C[pick][v][fo == f]

    log("\n--- Nested choice per held-out fold ---")
    for f, pk in enumerate(picks):
        log(f"  fold {f}: {pk if pk else 'v38 (no candidate meets rule)'}")

    log("\nCandidate (decision):")
    okN = report("N nested over 8-setting grid", outN)
    best_ins = max(ins_passing, key=ins_passing.get) if ins_passing else None

    # Vote-out diagnostic vs v38
    hp384_lab = {v: rule(P[HP384][v]) for v in EVAL_VIEWS}
    log("\n--- Vote-out diagnostic (dinov3Hp_384 vs v38 vs N / best) ---")
    for label_nm, lab in [("v38", ref), ("N nested", outN)] + ([(f"best in-sample ({best_ins})", C[best_ins])] if best_ins else []):
        vo_full = int(((hp384_lab["full"] == y) & (lab["full"] != y)).sum())
        res_full = int(((hp384_lab["full"] != y) & (lab["full"] == y)).sum())
        vo_dark = int(((hp384_lab["full"] == y) & (lab["full"] != y) & dark).sum())
        res_dark = int(((hp384_lab["full"] != y) & (lab["full"] == y) & dark).sum())
        err_f = int((lab["full"] != y).sum())
        err_d = int(((lab["full"] != y) & dark).sum())
        log(f"  {label_nm:26s}: full err={err_f:3d} (Hp384 vote-out={vo_full:2d}, rescued={res_full:2d}) | dark err={err_d:2d} (vote-out={vo_dark:2d}, rescued={res_dark:2d})")

    log(f"\n-> N {'PASSES' if okN else 'fails'}; best in-sample passing setting: {best_ins}")
    return okN


def run_smoke(train, y, folds, fo, log):
    n_smoke = 256
    log(f"=== SMOKE MODE ({n_smoke} training images) ===")
    smoke_paths = train.path.values[:n_smoke]
    arr0 = task4.load_uint8(list(smoke_paths))
    x0 = torch.from_numpy(arr0).cuda().permute(0, 3, 1, 2).float().div_(255)
    dl = dark_lc(x0)
    dl_g = dl[:, 0] * 0.299 + dl[:, 1] * 0.587 + dl[:, 2] * 0.114
    log(f"dark_lc view stats on {n_smoke} images: mean brightness={dl_g.mean():.3f}, mean std={dl_g.std((1, 2)).mean():.3f}")
    del arr0, x0, dl, dl_g
    torch.cuda.empty_cache()

    smoke_feats = {}
    for spec in SPECS:
        smoke_feats[spec["tag"]] = extract_backbone(spec, smoke_paths, log, save_cache=False)

    X_l2 = load_normalized_features(n_smoke, smoke_feats=smoke_feats, log=log)
    smoke_folds = [np.arange(0, 128), np.arange(128, 256)]
    smoke_fo = np.zeros(n_smoke, int)
    smoke_fo[128:] = 1
    pv2_smoke = train_oof_probe(X_l2, y[:n_smoke], smoke_folds, smoke_fo, epochs=2, seeds=(0,), log=log)
    for v in EVAL_VIEWS:
        assert np.isfinite(pv2_smoke[v]).all() and pv2_smoke[v].shape == (n_smoke, 5)
    del X_l2, smoke_feats, pv2_smoke
    gc.collect()
    torch.cuda.empty_cache()

    # Verify v38 reference reproduction on full training OOF
    L2, D3, D384, RAD, HP, HP384 = "dinov2L_hud70", "dinov3L_lr8e5", "dinov3L_384", "cradio3L_384", "dinov3Hp_144", "dinov3Hp_384"
    PLAIN, HUDM = ["cnxt", "deit3"], ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
    V20 = PLAIN + HUDM + ["dinov2_v2"]
    ALL_V38 = V20 + [L2, D3, D384, RAD, HP, HP384]
    V19, V30 = Path("experiments/claude/outputs/r19_views"), Path("experiments/claude/outputs/r30_views")
    N = len(y)
    P = {t: {v: np.zeros((N, 5), dtype=np.float64) for v in EVAL_VIEWS} for t in ALL_V38}
    for t in ALL_V38:
        for f, va in enumerate(folds):
            z30 = np.load(V30 / f"{t}_fold{f}.npz")
            z19 = np.load(V19 / f"{t}_fold{f}.npz") if t in V20 else z30
            for v in EVAL_VIEWS:
                P[t][v][va] = (z30 if v in z30.files else z19)[v]
    PR = dict(np.load("experiments/claude/outputs/r39/p7_probe_views.npz"))
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
    pi = np.bincount(y, minlength=5) / N
    wb = (0.2 / pi)[y]
    bal = lambda pred, m: f1_score(y[m], pred[m], average="macro", sample_weight=wb[m])
    rule = lambda p: (p * pi ** -0.75).argmax(1)
    ref = {}
    for v in EVAL_VIEWS:
        ft = np.where(GATE[v][:, None], sum(W_DARK[t] * P[t][v] for t in ALL_V38) / sum(W_DARK.values()),
                      sum(W_NORM[t] * P[t][v] for t in ALL_V38) / sum(W_NORM.values()))
        ref[v] = rule(0.70 * ft + 0.30 * PR[v])
    SC = ["full", "zoom70", "centre_grey", "dark/low-contrast"]
    VIEW_OF = {"full": "full", "zoom70": "zoom70", "centre_grey": "centre_grey", "dark/low-contrast": "full"}
    msk = lambda k: dark if k == "dark/low-contrast" else np.ones(N, bool)
    ref_scores = {k: bal(ref[VIEW_OF[k]], msk(k)) for k in SC}
    log("v38 reference verified: " + ", ".join(f"{k} {ref_scores[k]:.4f}" for k in SC))
    for k, exp_v in {"full": 0.9956, "zoom70": 0.9890, "centre_grey": 0.9821, "dark/low-contrast": 0.9890}.items():
        assert abs(ref_scores[k] - exp_v) <= 1.5e-4, f"v38 mismatch on {k}: {ref_scores[k]}"
    log("SMOKE PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Run fast 256-image end-to-end check without saving production caches")
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
            run_smoke(train, y, folds, fo, log)
            return

        log(f"Starting P_v2 extraction + 5-fold training + nested evaluation vs v38 (device={DEV}, N={len(y)})")
        for spec in SPECS:
            extract_backbone(spec, train.path.values, log, save_cache=True)

        X_l2 = load_normalized_features(len(y), log=log)
        log("Training 5-fold Gen-2 MLP probe P_v2 (3 seeds x 30 epochs, Feature-Space Mixup)...")
        PV2 = train_oof_probe(X_l2, y, folds, fo, epochs=30, seeds=(0, 1, 2), log=log)
        del X_l2
        gc.collect()
        torch.cuda.empty_cache()

        np.savez(VIEWS_OUT_PATH, **PV2)
        log(f"Saved P_v2 OOF views to {VIEWS_OUT_PATH}")

        evaluate_against_v38(PV2, y, folds, fo, log=log)
        log("done")
    finally:
        if log_f is not None:
            log_f.close()


if __name__ == "__main__":
    main()
