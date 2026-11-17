"""Round 30, Task 4: all-fold evaluation of a candidate member under the hard-view criterion (training images only).

    python em_r30_t4_hardviews.py cand [cand ...]

Every fold model predicts its own validation images (h-flip TTA) under five views: full, zoom70, HUD hidden (centre_grey), gamma 3.5
(dark_g35) and a fixed 75%-hidden grid (grid75), defined exactly as in cv_r22_t4_eval.py. The v20 members' full / zoom70 / HUD-hidden
views come from outputs/r19_views; their gamma / grid views and every view of a candidate are written to outputs/r30_views.
Scoring: v20 = weighted mean of the eight members (DINOv2 x3) under the v17 rule (argmax p * pi^-0.75) at a uniform class mix, members
only (the frozen-feature probe has no gamma / grid views; the probe-fused three-view score is printed as well).
Variants per candidate: added at weight 1, 2 or 3 (fixed), weight chosen nested (on 4 folds by the hard-view mean, scored on the
5th), and, if the candidate is a DINOv2 re-run, in place of dinov2_v2 at its weight.
Acceptance (user, 2026-11-16): hard-view mean (zoom70, HUD hidden, gamma 3.5, 75% grid) >= +0.0015 and full >= -0.0005; the generic
rule (pooled > 0, >= 3/5 folds, no fold below -0.0005 on the hard-view mean) is reported too.
Round 31: a candidate without a checkpoint for some folds (a fold-0 test) gets views for the folds it has; scoring is then left to
fc_r31_t4_v23_eval.py, which scores against v23 on the folds available.
Round 37: with the environment variable VIEWS_ONLY set, only the candidates' views are written (no v20 members, no scoring);
used on Colab, where the other members' files are not present.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import MODEL_DIR, PRED_DIR  # noqa: E402

ARCH = {"cnxt": ("convnext_tiny.in12k_ft_in1k", (160, 320)), "deit3": ("deit3_small_patch16_384.fb_in22k_ft_in1k", (192, 384)),
        "cnxt_hud": ("convnext_tiny.in12k_ft_in1k", (160, 320)), "cnxt22k_hud": ("convnext_tiny.fb_in22k", (160, 320)),
        "effv2_hud": ("tf_efficientnetv2_s.in21k_ft_in1k", (160, 320)), "deit3_hud": ("deit3_small_patch16_384.fb_in22k_ft_in1k", (192, 384)),
        "cnxtS_hud": ("convnext_small.in12k_ft_in1k", (160, 320)), "dinov2_v2": ("vit_base_patch14_reg4_dinov2.lvd142m", (126, 238)),
        "dinov2_r2": ("vit_base_patch14_reg4_dinov2.lvd142m", (252, 476)),
        "dinov2_sam": ("vit_base_patch14_reg4_dinov2.lvd142m", (126, 238)),
        "cnxt_mix": ("convnext_tiny.in12k_ft_in1k", (160, 320)),
        "dinov2L": ("vit_large_patch14_reg4_dinov2.lvd142m", (126, 238)),
        "dinov2_photo": ("vit_base_patch14_reg4_dinov2.lvd142m", (126, 238)),
        "cnxtB_hud": ("convnext_base.fb_in22k_ft_in1k", (160, 320)),
        "dinov2L_lpft": ("vit_large_patch14_reg4_dinov2.lvd142m", (126, 238)),  # round 38: LP-FT (Colab)
        "dinov2L_hud70": ("vit_large_patch14_reg4_dinov2.lvd142m", (126, 238))}  # round 39 step 3: HUD dropout 0.70, lr 3e-5 (Colab)
V20 = ["cnxt", "deit3", "cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud", "dinov2_v2"]
MEMBER_W = {"dinov2_v2": 3.0}
REPLACES = {"dinov2_r2": "dinov2_v2", "dinov2_sam": "dinov2_v2", "dinov2_photo": "dinov2_v2", "dinov2L": "dinov2_v2", "dinov2L_lpft": "dinov2_v2", "dinov2L_hud70": "dinov2_v2"}
CANDS = sys.argv[1:]
V19, V30 = Path("experiments/claude/outputs/r19_views"), Path("experiments/claude/outputs/r30_views")
V30.mkdir(parents=True, exist_ok=True)
ALL5 = ["full", "zoom70", "centre_grey", "dark_g35", "grid75"]
HARD = ["zoom70", "centre_grey", "dark_g35", "grid75"]


def zoom(x, f):
    th = torch.zeros(x.shape[0], 2, 3, device=x.device)
    th[:, 0, 0], th[:, 1, 1] = f, f
    return F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), mode="bilinear", padding_mode="border", align_corners=False)


def centre_grey(x):
    H, W = x.shape[2:]
    m = torch.zeros(1, 1, H, W, device=x.device)
    m[..., H // 5:4 * H // 5, W // 5:4 * W // 5] = 1
    return x * m + 0.5 * (1 - m)


def grid75(x):
    H, W = x.shape[2:]
    d = H // 3
    vis = int(round(d * 0.5))
    rr = torch.arange(H, device=x.device).view(H, 1)
    cc = torch.arange(W, device=x.device).view(1, W)
    m = (((rr + 7) % d) < vis) & (((cc + 11) % d) < vis)
    return torch.where(m, x, x.mean((2, 3), keepdim=True).expand_as(x))


FN = {"full": lambda x: x, "zoom70": lambda x: zoom(x, 0.7), "centre_grey": centre_grey,
      "dark_g35": lambda x: x.clamp_min(1e-6) ** 3.5, "grid75": grid75}
train, _ = task4.load_frames()
y = train.label.values
folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
VIEWS_ONLY = bool(__import__('os').environ.get('VIEWS_ONLY'))     # round 37 (Colab): only the candidates' views, no scoring
for tag in ([] if VIEWS_ONLY else V20) + CANDS:
    views = ["dark_g35", "grid75"] if tag in V20 else ALL5
    name, size = ARCH[tag]
    for f, va in enumerate(folds):
        out = V30 / f"{tag}_fold{f}.npz"
        if out.exists() or (tag in CANDS and not (MODEL_DIR / f"t4_{tag}_fold{f}.pt").exists()):
            continue
        m = task4.create_model(name, pretrained=False, img_size=size).cuda().to(memory_format=torch.channels_last)
        m.load_state_dict(torch.load(MODEL_DIR / f"t4_{tag}_fold{f}.pt", map_location="cuda"))
        m.eval()
        arr = task4.load_uint8(list(train.path.values[va]))
        res = {v: [] for v in views}
        with torch.no_grad():
            for i in range(0, len(arr), 128):
                x0 = task4.to_gpu_batch(arr[i:i + 128], size)
                for v in views:
                    x = FN[v](x0)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        p = 0.5 * (m(task4.normalize(x).to(memory_format=torch.channels_last)).float().softmax(-1)
                                   + m(task4.normalize(x.flip(-1)).to(memory_format=torch.channels_last)).float().softmax(-1))
                    res[v].append(p.cpu().numpy())
        np.savez(out, **{v: np.concatenate(r) for v, r in res.items()})
        print(f"{tag} fold {f}: views {views} written", flush=True)
        del m, arr
        torch.cuda.empty_cache()

if VIEWS_ONLY:
    sys.exit(0)
partial = [t for t in CANDS if any(not (V30 / f"{t}_fold{f}.npz").exists() for f in range(5))]
if partial:
    sys.exit(f"views written; {partial} lack some folds, so score them with fc_r31_t4_v23_eval.py (v23 base, partial folds)")
P = {t: {v: np.zeros((len(y), 5)) for v in ALL5} for t in V20 + CANDS}
for t in P:
    for f, va in enumerate(folds):
        z30 = np.load(V30 / f"{t}_fold{f}.npz")
        z19 = np.load(V19 / f"{t}_fold{f}.npz") if t in V20 else z30
        for v in ALL5:
            P[t][v][va] = (z30 if v in z30.files else z19)[v]
pi = np.bincount(y, minlength=5) / len(y)
wb = (0.2 / pi)[y]
fo = np.zeros(len(y), int)
for f, va in enumerate(folds):
    fo[va] = f
zp = np.load("experiments/claude/outputs/r27/t4_probe_views.npz")
bal = lambda pred, m: f1_score(y[m], pred[m], average="macro", sample_weight=wb[m])


def labels(members, wmap, probe=False):
    s = sum(wmap.get(t, 1.0) for t in members)
    out = {}
    for v in ALL5:
        p = sum(wmap.get(t, 1.0) * P[t][v] for t in members) / s
        if probe and v in ("full", "zoom70", "centre_grey"):
            p = 0.9 * p + 0.1 * zp[f"probe-aug_{v}"]
        out[v] = (p * pi ** -0.75).argmax(1)
    return out


allm = np.ones(len(y), bool)
base = labels(V20, MEMBER_W)
for v in ALL5:
    print(f"v20 members, {v}: balanced macro-F1 {bal(base[v], allm):.4f}")
rng = np.random.default_rng(0)
boots = [rng.integers(0, len(y), len(y)) for _ in range(300)]


def report(label, lab):
    per = {v: bal(lab[v], allm) - bal(base[v], allm) for v in ALL5}
    hard = np.mean([per[v] for v in HARD])
    pf = [np.mean([bal(lab[v], fo == f) - bal(base[v], fo == f) for v in HARD]) for f in range(5)]
    ci = np.quantile([np.mean([bal(lab[v], b) - bal(base[v], b) for v in HARD]) for b in boots], [0.025, 0.975])
    ok_user = hard >= 0.0015 and per["full"] >= -0.0005
    ok_gen = hard > 0 and sum(p > 0 for p in pf) >= 3 and min(pf) >= -0.0005
    print(f"  {label:44s} hard mean {hard:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]  full {per['full']:+.4f} | "
          + " ".join(f"{v} {per[v]:+.4f}" for v in HARD)
          + f" | folds (hard) +{sum(p > 0 for p in pf)}/5 ({' '.join(f'{p:+.4f}' for p in pf)})"
          + f"  -> hard-view rule {'PASS' if ok_user else 'fail'}, generic rule {'PASS' if ok_gen else 'fail'}", flush=True)
    return pf


for c in CANDS:
    print(f"\ncandidate {c}:")
    for v in ALL5:
        print(f"  alone, {v}: {bal((P[c][v] * pi ** -0.75).argmax(1), allm):.4f}  (dinov2_v2 {bal((P['dinov2_v2'][v] * pi ** -0.75).argmax(1), allm):.4f})")
    labs = {w: labels(V20 + [c], {**MEMBER_W, c: w}) for w in (1.0, 2.0, 3.0)}
    pfs = {w: report(f"added at weight {w:g}", labs[w]) for w in (1.0, 2.0, 3.0)}
    nested = {v: np.zeros(len(y), int) for v in ALL5}
    chosen = []
    for f in range(5):
        inner = [g for g in range(5) if g != f]
        w_best = max(pfs, key=lambda w: np.mean([pfs[w][g] for g in inner]))
        chosen.append(w_best)
        for v in ALL5:
            nested[v][fo == f] = labs[w_best][v][fo == f]
    report(f"weight nested (picks {chosen})", nested)
    if c in REPLACES:
        old = REPLACES[c]
        rep = [c if t == old else t for t in V20]
        report(f"in place of {old} (weight {MEMBER_W[old]:g})", labels(rep, {c: MEMBER_W[old]}))
    lp, bp = labels(V20 + [c], {**MEMBER_W, c: 1.0}, probe=True), labels(V20, MEMBER_W, probe=True)
    d3 = [bal(lp[v], allm) - bal(bp[v], allm) for v in ("full", "zoom70", "centre_grey")]
    print(f"  (with the 10% probe, three views, weight 1: {np.mean(d3):+.4f}; full {d3[0]:+.4f}, zoom70 {d3[1]:+.4f}, HUD hidden {d3[2]:+.4f})")
