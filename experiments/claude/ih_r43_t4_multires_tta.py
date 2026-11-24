"""Round 43, Task 4: Zero-Crop RoPE Multi-Resolution TTA on DINOv3 ViT-H+/16 (dinov3Hp_384).

    python experiments/claude/ih_r43_t4_multires_tta.py --smoke
    python experiments/claude/ih_r43_t4_multires_tta.py --extract-oof
    python experiments/claude/ih_r43_t4_multires_tta.py --eval
    python experiments/claude/ih_r43_t4_multires_tta.py --extract-test

Strictly zero test leakage:
  --eval evaluates ONLY on Training OOF views against the v38 ensemble via strict Nested 5-fold selection.
  --extract-test runs inference-only predictions on test.csv (never read by --eval).
"""
import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import task4  # noqa: E402
from src.config import DATA_DIRS, MODEL_DIR, N_FOLDS, PRED_DIR  # noqa: E402

MODEL = "vit_huge_plus_patch16_dinov3.lvd1689m"
TAG = "dinov3Hp_384"
BASE_HW = (192, 384)
RESOLUTIONS = [(160, 304), (192, 384), (224, 432)]
RES_KEYS = [f"{h}x{w}" for h, w in RESOLUTIONS]
EVAL_VIEWS = ["full", "zoom70", "centre_grey"]

OUT_DIR = Path("experiments/claude/outputs/r42")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "ih_multires_tta.log"


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


def zoom(x, factor=0.7):
    th = torch.zeros(x.shape[0], 2, 3, device=x.device)
    th[:, 0, 0], th[:, 1, 1] = factor, factor
    return F.grid_sample(
        x,
        F.affine_grid(th, x.shape, align_corners=False),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


def centre_grey(x):
    h, w = x.shape[2:]
    m = torch.zeros(1, 1, h, w, device=x.device)
    m[..., h // 5:4 * h // 5, w // 5:4 * w // 5] = 1.0
    return x * m + 0.5 * (1.0 - m)


def load_frames_meta():
    train = pd.read_csv(DATA_DIRS[4] / "train.csv")
    test = pd.read_csv(DATA_DIRS[4] / "test.csv")
    train["path"] = [str(DATA_DIRS[4] / "train" / f) for f in train.file_name]
    test["path"] = [str(DATA_DIRS[4] / "test" / f) for f in test.file_name]
    return train, test


def load_dinov3_dynamic(ckpt_path, device="cuda"):
    import timm

    model = timm.create_model(MODEL, pretrained=False, num_classes=5, dynamic_img_size=True)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    return model.to(device=device, memory_format=torch.channels_last).eval()


@torch.no_grad()
def predict_multires_views(model, paths, views, bs=64, device="cuda"):
    out = {f"{v}_{rk}": [] for v in views for rk in RES_KEYS}
    for i in range(0, len(paths), 4096):
        arr = task4.load_uint8(list(paths[i:i + 4096]))
        for j in range(0, len(arr), bs):
            batch_u8 = arr[j:j + bs]
            x_base = task4.to_gpu_batch(batch_u8, BASE_HW, device) if any(v != "full" for v in views) else None
            view_192 = {}
            for v in views:
                if v == "zoom70":
                    view_192[v] = zoom(x_base, 0.7)
                elif v == "centre_grey":
                    view_192[v] = centre_grey(x_base)
            for hw, rk in zip(RESOLUTIONS, RES_KEYS):
                for v in views:
                    if v == "full":
                        xv = task4.to_gpu_batch(batch_u8, hw, device)
                    else:
                        x192 = view_192[v]
                        xv = x192 if hw == BASE_HW else F.interpolate(x192, size=hw, mode="bilinear", align_corners=False)
                    xn = task4.normalize(xv).to(memory_format=torch.channels_last)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        p = 0.5 * (
                            model(xn).float().softmax(-1)
                            + model(xn.flip(-1)).float().softmax(-1)
                        )
                    out[f"{v}_{rk}"].append(p.cpu().numpy().astype(np.float32))
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def extract_oof(folds_arg, bs, log):
    train, _ = load_frames_meta()
    fold_idx = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(N_FOLDS)]
    r30_dir = Path("experiments/claude/outputs/r30_views")
    req_keys = [f"{v}_{rk}" for v in EVAL_VIEWS for rk in RES_KEYS]

    for f in folds_arg:
        va = fold_idx[f]
        out_path = PRED_DIR / f"t4_mres_{TAG}_fold{f}.npz"
        if out_path.exists():
            with np.load(out_path) as z:
                if all(k in z.files and z[k].shape == (len(va), 5) for k in req_keys):
                    log(f"fold {f}: cache {out_path.name} valid -> skipping")
                    continue
        ckpt_path = MODEL_DIR / f"t4_{TAG}_fold{f}.pt"
        assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"

        t1 = time.time()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = load_dinov3_dynamic(ckpt_path, device="cuda")
        preds = predict_multires_views(model, train.path.values[va], EVAL_VIEWS, bs=bs, device="cuda")
        peak = torch.cuda.max_memory_allocated() / 2**30

        r30_path = r30_dir / f"{TAG}_fold{f}.npz"
        parity_msg = ""
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

        np.savez(out_path, val_idx=va, **preds)
        log(f"fold {f}: saved {out_path.name} ({len(va)} rows x {len(req_keys)} keys) in {(time.time() - t1)/60:.1f}m | peak={peak:.2f} GiB{parity_msg}")
        del model
        gc.collect()
        torch.cuda.empty_cache()


def extract_test(bs, log):
    _, test = load_frames_meta()
    names = np.array(test.file_name.tolist(), dtype="U")
    out_path = PRED_DIR / f"t4_mres_{TAG}_test.npz"
    if out_path.exists():
        log(f"test cache {out_path.name} already exists -> skipping")
        return

    fold_means = {f"fold_mean_{rk}": np.zeros((len(test), 5), dtype=np.float64) for rk in RES_KEYS}
    for f in range(N_FOLDS):
        ckpt_path = MODEL_DIR / f"t4_{TAG}_fold{f}.pt"
        assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"
        t1 = time.time()
        model = load_dinov3_dynamic(ckpt_path, device="cuda")
        preds = predict_multires_views(model, test.path.values, ["full"], bs=bs, device="cuda")
        for rk in RES_KEYS:
            fold_means[f"fold_mean_{rk}"] += preds[f"full_{rk}"] / N_FOLDS
        log(f"test fold {f}: done in {(time.time() - t1)/60:.1f}m")
        del model
        gc.collect()
        torch.cuda.empty_cache()

    full_ckpt = MODEL_DIR / f"t4_{TAG}_full.pt"
    assert full_ckpt.exists(), f"Missing full checkpoint: {full_ckpt}"
    t1 = time.time()
    model = load_dinov3_dynamic(full_ckpt, device="cuda")
    full_preds = predict_multires_views(model, test.path.values, ["full"], bs=bs, device="cuda")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log(f"test full refit: done in {(time.time() - t1)/60:.1f}m")

    save_dict = {
        **{k: v.astype(np.float32) for k, v in fold_means.items()},
        **{f"full_{rk}": full_preds[f"full_{rk}"].astype(np.float32) for rk in RES_KEYS},
        "test_file_name": names,
    }
    np.savez(out_path, **save_dict)
    log(f"saved test multi-res cache to {out_path}")


def load_v38_oof_reference():
    train = pd.read_csv(DATA_DIRS[4] / "train.csv")
    y = train.label.values.astype(np.int64)
    n_rows = len(y)
    folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(N_FOLDS)]
    fo = np.zeros(n_rows, dtype=int)
    for f, va in enumerate(folds):
        fo[va] = f

    l2, d3, d384, rad, hp, hp384 = (
        "dinov2L_hud70",
        "dinov3L_lr8e5",
        "dinov3L_384",
        "cradio3L_384",
        "dinov3Hp_144",
        "dinov3Hp_384",
    )
    plain, hudm = ["cnxt", "deit3"], ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
    v20 = plain + hudm + ["dinov2_v2"]
    all_v38 = v20 + [l2, d3, d384, rad, hp, hp384]
    v19_dir = Path("experiments/claude/outputs/r19_views")
    v30_dir = Path("experiments/claude/outputs/r30_views")

    probs = {t: {v: np.zeros((n_rows, 5), dtype=np.float64) for v in EVAL_VIEWS} for t in all_v38}
    for t in all_v38:
        for f, va in enumerate(folds):
            z30 = np.load(v30_dir / f"{t}_fold{f}.npz")
            z19 = np.load(v19_dir / f"{t}_fold{f}.npz") if t in v20 else z30
            for v in EVAL_VIEWS:
                probs[t][v][va] = (z30 if v in z30.files else z19)[v]

    pr = dict(np.load("experiments/claude/outputs/r39/p7_probe_views.npz"))
    st = dict(np.load("experiments/claude/outputs/r39/t4_view_stats.npz"))
    st30 = np.load("experiments/claude/outputs/r30/t4_errors.npz")

    gate = {v: np.zeros(n_rows, dtype=bool) for v in EVAL_VIEWS}
    for f in range(N_FOLDS):
        tr, ho = fo != f, fo == f
        tb = np.quantile(st["brightness_full"][tr], 0.10)
        tc = np.quantile(st["contrast_full"][tr], 0.10)
        for v in EVAL_VIEWS:
            gate[v][ho] = (st[f"brightness_{v}"][ho] <= tb) | (st[f"contrast_{v}"][ho] <= tc)

    dark = (st30["contrast"] <= np.quantile(st30["contrast"], 0.1)) | (st30["brightness"] <= np.quantile(st30["brightness"], 0.1))

    w_norm = {
        **{t: 0.25 for t in plain},
        **{t: 0.75 for t in hudm},
        "dinov2_v2": 8.0,
        l2: 8.0,
        d3: 10.0,
        d384: 20.0,
        rad: 10.0,
        hp: 40.0,
        hp384: 40.0,
    }
    w_dark = {**w_norm, l2: 2.0, rad: 2.0}
    return y, folds, fo, probs, pr, gate, dark, w_norm, w_dark, all_v38, hp384


def blend_scales(mres_dict, weights, mode):
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    if mode == "prob":
        return sum(wi * mres_dict[rk] for wi, rk in zip(w, RES_KEYS))
    if mode == "geo":
        lp = sum(wi * np.log(np.clip(mres_dict[rk], 1e-7, 1.0)) for wi, rk in zip(w, RES_KEYS))
        lp = lp - lp.max(axis=1, keepdims=True)
        ex = np.exp(lp)
        return ex / ex.sum(axis=1, keepdims=True)
    raise ValueError(f"Unknown blend mode: {mode}")


def eval_oof(log):
    y, folds, fo, probs, pr, gate, dark, w_norm, w_dark, all_v38, hp384 = load_v38_oof_reference()
    n_rows = len(y)
    pi = np.bincount(y, minlength=5) / n_rows
    wb = (0.2 / pi)[y]
    bal = lambda pred, m: f1_score(y[m], pred[m], average="macro", sample_weight=wb[m])
    rule = lambda p: (p * pi ** -0.75).argmax(1)

    allm = np.ones(n_rows, dtype=bool)
    sc = ["full", "zoom70", "centre_grey", "dark/low-contrast"]
    view_of = {"full": "full", "zoom70": "zoom70", "centre_grey": "centre_grey", "dark/low-contrast": "full"}
    msk = lambda k, m=allm: m & dark if k == "dark/low-contrast" else m

    def make_ensemble_labels(hp384_override=None):
        den_n = sum(w_norm.values())
        den_d = sum(w_dark.values())
        out = {}
        for v in EVAL_VIEWS:
            num_n = sum(w_norm[t] * (hp384_override[v] if (hp384_override is not None and t == hp384) else probs[t][v]) for t in all_v38)
            num_d = sum(w_dark[t] * (hp384_override[v] if (hp384_override is not None and t == hp384) else probs[t][v]) for t in all_v38)
            ft = np.where(gate[v][:, None], num_d / den_d, num_n / den_n)
            out[v] = rule(0.70 * ft + 0.30 * pr[v])
        return out

    ref = make_ensemble_labels(None)
    ref_scores = {k: bal(ref[view_of[k]], msk(k)) for k in sc}
    log("v38 reference: " + ", ".join(f"{k} {ref_scores[k]:.4f}" for k in sc))
    for k, exp_v in {"full": 0.9956, "zoom70": 0.9890, "centre_grey": 0.9821, "dark/low-contrast": 0.9890}.items():
        assert abs(ref_scores[k] - exp_v) <= 1.5e-4, f"v38 mismatch on {k}: got {ref_scores[k]:.5f}, expected {exp_v:.4f}"

    mres = {v: {rk: np.zeros((n_rows, 5), dtype=np.float64) for rk in RES_KEYS} for v in EVAL_VIEWS}
    for f, va in enumerate(folds):
        p = PRED_DIR / f"t4_mres_{TAG}_fold{f}.npz"
        assert p.exists(), f"Missing {p}: run Colab / --extract-oof first"
        with np.load(p) as z:
            if "val_idx" in z.files:
                assert (z["val_idx"] == va).all(), f"val_idx mismatch in {p.name}"
            for v in EVAL_VIEWS:
                for rk in RES_KEYS:
                    mres[v][rk][va] = z[f"{v}_{rk}"]

    cand_specs = [
        ("eq3_prob", (1 / 3, 1 / 3, 1 / 3), "prob"),
        ("ctr_prob", (0.25, 0.50, 0.25), "prob"),
        ("hi_prob", (0.20, 0.45, 0.35), "prob"),
        ("lo_prob", (0.35, 0.45, 0.20), "prob"),
        ("pair_hi", (0.0, 0.50, 0.50), "prob"),
        ("pair_lo", (0.50, 0.50, 0.0), "prob"),
        ("eq3_geo", (1 / 3, 1 / 3, 1 / 3), "geo"),
        ("ctr_geo", (0.25, 0.50, 0.25), "geo"),
    ]

    log("\n--- Single-scale & Multi-Res dinov3Hp_384 ALONE (v17 rule, balanced mix) ---")
    for label_nm, pmap in (
        [(f"orig r30_views {TAG}", probs[hp384])]
        + [(f"scale {rk}", {v: mres[v][rk] for v in EVAL_VIEWS}) for rk in RES_KEYS]
        + [(f"blend {nm}", {v: blend_scales(mres[v], wt, md) for v in EVAL_VIEWS}) for nm, wt, md in cand_specs[:2] + cand_specs[6:]]
    ):
        s_str = ", ".join(
            f"{k} {bal(rule(pmap[view_of[k]]), msk(k)):.4f}({(rule(pmap[view_of[k]])[msk(k)] != y[msk(k)]).sum()})"
            for k in sc
        )
        log(f"  {label_nm:22s}: {s_str}")

    cands = {}
    for nm, wt, md in cand_specs:
        hp_ovr = {v: blend_scales(mres[v], wt, md) for v in EVAL_VIEWS}
        cands[nm] = make_ensemble_labels(hp_ovr)

    deltas = lambda lab, m=allm: {k: bal(lab[view_of[k]], msk(k, m)) - bal(ref[view_of[k]], msk(k, m)) for k in sc}
    meets = lambda d: d["zoom70"] > 0 and d["centre_grey"] > 0 and d["dark/low-contrast"] > 0 and d["full"] >= -0.0005
    crit = lambda d: float(np.mean([d["zoom70"], d["centre_grey"], d["dark/low-contrast"]]))

    def report(nm, lab):
        d = deltas(lab)
        pf = [np.mean([bal(lab[v], fo == f) - bal(ref[v], fo == f) for v in ("zoom70", "centre_grey")]) for f in range(N_FOLDS)]
        pdk = [bal(lab["full"], (fo == f) & dark) - bal(ref["full"], (fo == f) & dark) for f in range(N_FOLDS)]
        ok = meets(d) and sum(x > 0 for x in pf) >= 3
        log(
            f"  {nm}\n    "
            + ", ".join(f"{k} {d[k]:+.4f}" for k in sc)
            + f"\n    zoom / HUD by fold {' '.join(f'{x:+.4f}' for x in pf)} | dark by fold {' '.join(f'{x:+.4f}' for x in pdk)}"
            + f" | labels changed (full) {(lab['full'] != ref['full']).sum()}\n    -> {'PASS' if ok else 'fail'}"
        )
        return ok

    log("\n--- Pre-declared 8 Multi-Res candidates vs v38 (in-sample) ---")
    ins_passing = {}
    for nm, lab in cands.items():
        report(nm, lab)
        d = deltas(lab)
        if meets(d):
            ins_passing[nm] = crit(d)

    out_n, picks = {v: ref[v].copy() for v in EVAL_VIEWS}, []
    for f in range(N_FOLDS):
        okk = {nm: crit(d) for nm, d in ((nm, deltas(lab, fo != f)) for nm, lab in cands.items()) if meets(d)}
        pick = max(okk, key=okk.get) if okk else None
        picks.append(pick)
        if pick is not None:
            for v in EVAL_VIEWS:
                out_n[v][fo == f] = cands[pick][v][fo == f]

    log("\n--- Nested choice per held-out fold ---")
    for f, pk in enumerate(picks):
        log(f"  fold {f}: {pk if pk else 'v38 (no candidate meets rule)'}")

    log("\nCandidate (decision):")
    ok_n = report("N nested over 8-setting Multi-Res TTA grid", out_n)
    best_ins = max(ins_passing, key=ins_passing.get) if ins_passing else None

    hp_orig_lab = {v: rule(probs[hp384][v]) for v in EVAL_VIEWS}
    log("\n--- Vote-out & error summary (full & dark) ---")
    for nm_l, lab in [("v38", ref), ("N nested", out_n)] + ([(f"best in-sample ({best_ins})", cands[best_ins])] if best_ins else []):
        vo_f = int(((hp_orig_lab["full"] == y) & (lab["full"] != y)).sum())
        res_f = int(((hp_orig_lab["full"] != y) & (lab["full"] == y)).sum())
        vo_d = int(((hp_orig_lab["full"] == y) & (lab["full"] != y) & dark).sum())
        res_d = int(((hp_orig_lab["full"] != y) & (lab["full"] == y) & dark).sum())
        err_f = int((lab["full"] != y).sum())
        err_d = int(((lab["full"] != y) & dark).sum())
        log(f"  {nm_l:26s}: full err={err_f:3d} (Hp384 vote-out={vo_f:2d}, rescued={res_f:2d}) | dark err={err_d:2d} (vote-out={vo_d:2d}, rescued={res_d:2d})")

    log(f"\n-> N {'PASSES' if ok_n else 'fails'}; best in-sample passing setting: {best_ins}")
    return ok_n


def run_smoke(log):
    import timm

    log("=== CPU SMOKE MODE (Zero CUDA) ===")
    m_fix = timm.create_model("vit_small_patch16_dinov3.lvd1689m", pretrained=False, num_classes=5, img_size=BASE_HW).eval()
    m_dyn = timm.create_model("vit_small_patch16_dinov3.lvd1689m", pretrained=False, num_classes=5, dynamic_img_size=True).eval()
    m_dyn.load_state_dict(m_fix.state_dict(), strict=True)

    with torch.no_grad():
        x_base = torch.randn(2, 3, *BASE_HW)
        y_fix = m_fix(x_base)
        y_dyn_base = m_dyn(x_base)
        parity_diff = float((y_fix - y_dyn_base).abs().max().item())
        assert parity_diff == 0.0, f"Dynamic RoPE mismatch at {BASE_HW}: {parity_diff}"
        log(f"dynamic_img_size=True parity vs fixed {BASE_HW}: max |diff|={parity_diff:.1e}")
        for hw in RESOLUTIONS:
            y_hw = m_dyn(torch.randn(2, 3, *hw))
            assert y_hw.shape == (2, 5) and torch.isfinite(y_hw).all(), f"Bad output at {hw}"
            log(f"  forward {hw} -> shape {tuple(y_hw.shape)}, finite=True")

    y, _, _, probs, pr, gate, dark, w_norm, w_dark, all_v38, _ = load_v38_oof_reference()
    pi = np.bincount(y, minlength=5) / len(y)
    wb = (0.2 / pi)[y]
    bal = lambda pred, m: f1_score(y[m], pred[m], average="macro", sample_weight=wb[m])
    rule = lambda p: (p * pi ** -0.75).argmax(1)

    den_n, den_d = sum(w_norm.values()), sum(w_dark.values())
    ref = {}
    for v in EVAL_VIEWS:
        ft = np.where(
            gate[v][:, None],
            sum(w_dark[t] * probs[t][v] for t in all_v38) / den_d,
            sum(w_norm[t] * probs[t][v] for t in all_v38) / den_n,
        )
        ref[v] = rule(0.70 * ft + 0.30 * pr[v])

    sc = ["full", "zoom70", "centre_grey", "dark/low-contrast"]
    view_of = {"full": "full", "zoom70": "zoom70", "centre_grey": "centre_grey", "dark/low-contrast": "full"}
    allm = np.ones(len(y), dtype=bool)
    ref_scores = {k: bal(ref[view_of[k]], dark if k == "dark/low-contrast" else allm) for k in sc}
    log("v38 reference verified: " + ", ".join(f"{k} {ref_scores[k]:.4f}" for k in sc))
    for k, exp_v in {"full": 0.9956, "zoom70": 0.9890, "centre_grey": 0.9821, "dark/low-contrast": 0.9890}.items():
        assert abs(ref_scores[k] - exp_v) <= 1.5e-4, f"v38 mismatch on {k}: {ref_scores[k]}"
    log("SMOKE PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="CPU-only check of dynamic RoPE + v38 OOF reference")
    ap.add_argument("--extract-oof", action="store_true", help="Extract 3-resolution OOF TTA probs on GPU (training val folds)")
    ap.add_argument("--extract-test", action="store_true", help="Extract 3-resolution test TTA probs on GPU (inference-only)")
    ap.add_argument("--eval", action="store_true", help="Nested 5-fold CPU evaluation against v38 (training OOF only)")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--bs", type=int, default=64)
    args = ap.parse_args()

    if not any([args.smoke, args.extract_oof, args.extract_test, args.eval]):
        ap.error("Specify at least one of --smoke, --extract-oof, --extract-test, --eval")

    log, log_f = make_logger(LOG_PATH, write_log=args.eval)
    try:
        if args.smoke:
            run_smoke(log)
        if args.extract_oof:
            extract_oof(args.folds, args.bs, log)
        if args.extract_test:
            extract_test(args.bs, log)
        if args.eval:
            eval_oof(log)
    finally:
        if log_f is not None:
            log_f.close()


if __name__ == "__main__":
    main()
