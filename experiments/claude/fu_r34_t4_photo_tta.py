"""Round 34, Task 4 item 3a: photometric test-time augmentation for the DINOv2 member (training images only).

    python fu_r34_t4_photo_tta.py tag [tag ...]        tag = dinov2_v2 (the v23 member) or dinov2_r2 (the 252x476 candidate)

Round 30 forensics: the ensemble's remaining errors are concentrated in dark / low-contrast frames. Test-time augmentation is allowed
(TA ruling of 15 Nov 2026); everything here is chosen on training images (each fold model predicts its own validation images).
GPU phase (skipped for folds already written): every fold model predicts its validation images under the five scoring views (full,
zoom70, HUD hidden, gamma 3.5, 75% grid; as em_r30_t4_hardviews.py), each view passed through four photometric transforms before
the network (h-flip averaged as in the pipeline):
  id            none (the pipeline's prediction; checked against outputs/r30_views / r19_views)
  autocontrast  per image, the luminance range between its 1st and 99th percentile stretched to [0, 1] (skipped if the range < 0.05)
  autogamma     per image, x ** g with g = log(0.45) / log(mean luminance), clipped to [0.25, 1] (brightens dark frames only)
  gamma05       x ** 0.5 for every image
  hudmask       the HUD border greyed out (centre_grey), round 36
  crop85        the central 85% of the frame, resized (round 36)
Probabilities go to experiments/claude/outputs/r34_views/<tag>_fold<f>.npz (keys "<view>|<transform>").
CPU phase: v23 ensemble (member weights DINOv2 x8, HUD-dropout members x3, others x1; v17 rule; uniform class mix; members only)
with the DINOv2 member's probabilities replaced by the mean over a transform set. For dinov2_r2 the base is v23 with dinov2_r2 in
place of dinov2_v2 (weight 8).
Rule, fixed before the run: the gamma-3.5 view is NOT counted (autogamma / gamma05 invert it by construction). A transform set
passes if, pooled over the folds: the real dark / low-contrast frames (lowest decile of contrast or brightness, full view) gain
> 0 and in >= 3/5 folds; full >= -0.0005; and the mean of zoom70, HUD hidden and the 75% grid >= -0.0005.
The set is also chosen nested (on four folds by the dark / low-contrast gain among the sets that keep full >= -0.0005 there).
Round 36, entropy-gated TTA: the transform set is used only for images whose ensemble prediction on the incoming view is uncertain
(entropy of the v23 members' mean probability above the 80th / 90th / 95th percentile of the full-view entropies of the training
images); every other image keeps the plain prediction. Same rule; the (set, percentile) pair is also chosen nested.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)
from src import task4  # noqa: E402
from src.config import MODEL_DIR, PRED_DIR  # noqa: E402

V20 = ["cnxt", "deit3", "cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud", "dinov2_v2"]
HUDM = ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
V23_W = {t: (8.0 if t == "dinov2_v2" else 3.0 if t in HUDM else 1.0) for t in V20}
SIZE = {"dinov2_v2": (126, 238), "dinov2_r2": (252, 476)}
NAME = "vit_base_patch14_reg4_dinov2.lvd142m"
ALL5 = ["full", "zoom70", "centre_grey", "dark_g35", "grid75"]
TTAS = ["id", "autocontrast", "autogamma", "gamma05", "hudmask", "crop85"]
SETS = {"id + autocontrast": ["id", "autocontrast"], "id + autogamma": ["id", "autogamma"], "id + gamma05": ["id", "gamma05"],
        "id + autocontrast + autogamma": ["id", "autocontrast", "autogamma"], "all four photometric": TTAS[:4], "autogamma only": ["autogamma"],
        "id + hudmask": ["id", "hudmask"], "id + crop85": ["id", "crop85"], "all six": TTAS}
GATED = {"photometric": ["id", "autocontrast", "autogamma"], "all six": TTAS}          # round 36: sets used only on uncertain images
GATE_Q = (0.80, 0.90, 0.95)
V19, V30, V34 = (Path(f"experiments/claude/outputs/{d}") for d in ("r19_views", "r30_views", "r34_views"))
V34 = Path(__import__("os").environ.get("R34_VIEWS", V34))            # override for a dry run of the CPU phase
V34.mkdir(parents=True, exist_ok=True)
TAGS = sys.argv[1:] or ["dinov2_v2"]
train, _ = task4.load_frames()
y = train.label.values
folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]

# ----------------------------------------------------------------------------------------------------------------- GPU phase
todo = [(t, f) for t in TAGS for f in range(5) if not (V34 / f"{t}_fold{f}.npz").exists() and (MODEL_DIR / f"t4_{t}_fold{f}.pt").exists()]
if todo:
    import torch
    import torch.nn.functional as F

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

    lum = lambda x: 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]

    def autocontrast(x):
        q = torch.quantile(lum(x).flatten(1).float(), torch.tensor([0.01, 0.99], device=x.device), dim=1)        # 2 x B
        lo, hi = q[0].view(-1, 1, 1, 1), q[1].view(-1, 1, 1, 1)
        return torch.where(hi - lo >= 0.05, ((x - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1), x)

    def autogamma(x):
        m = lum(x).flatten(1).mean(1).clamp(1e-4, 0.999)
        g = (np.log(0.45) / torch.log(m)).clamp(0.25, 1.0).view(-1, 1, 1, 1)
        return x.clamp_min(1e-6) ** g

    VIEW = {"full": lambda x: x, "zoom70": lambda x: zoom(x, 0.7), "centre_grey": centre_grey,
            "dark_g35": lambda x: x.clamp_min(1e-6) ** 3.5, "grid75": grid75}
    TF = {"id": lambda x: x, "autocontrast": autocontrast, "autogamma": autogamma, "gamma05": lambda x: x.clamp_min(1e-6) ** 0.5,
          "hudmask": centre_grey, "crop85": lambda x: zoom(x, 0.85)}
    for tag, f in todo:
        size, va = SIZE[tag], folds[f]
        m = task4.create_model(NAME, pretrained=False, img_size=size).cuda().to(memory_format=torch.channels_last)
        m.load_state_dict(torch.load(MODEL_DIR / f"t4_{tag}_fold{f}.pt", map_location="cuda"))
        m.eval()
        arr = task4.load_uint8(list(train.path.values[va]))
        res = {f"{v}|{t}": [] for v in ALL5 for t in TTAS}
        bs = 128 if size[0] < 200 else 48
        with torch.no_grad():
            for i in range(0, len(arr), bs):
                x0 = task4.to_gpu_batch(arr[i:i + bs], size)
                for v in ALL5:
                    xv = VIEW[v](x0)
                    for t in TTAS:
                        x = TF[t](xv)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            p = 0.5 * (m(task4.normalize(x).to(memory_format=torch.channels_last)).float().softmax(-1)
                                       + m(task4.normalize(x.flip(-1)).to(memory_format=torch.channels_last)).float().softmax(-1))
                        res[f"{v}|{t}"].append(p.cpu().numpy())
        np.savez(V34 / f"{tag}_fold{f}.npz", **{k: np.concatenate(r) for k, r in res.items()})
        print(f"{tag} fold {f}: {len(res)} view x transform predictions written", flush=True)
        del m, arr
        torch.cuda.empty_cache()

# ----------------------------------------------------------------------------------------------------------------- CPU phase
fo = np.zeros(len(y), int)
for f, va in enumerate(folds):
    fo[va] = f
P = {t: {v: np.zeros((len(y), 5)) for v in ALL5} for t in V20}
for t in V20:
    for f, va in enumerate(folds):
        z30, z19 = np.load(V30 / f"{t}_fold{f}.npz"), np.load(V19 / f"{t}_fold{f}.npz")
        for v in ALL5:
            P[t][v][va] = (z30 if v in z30.files else z19)[v]
st = np.load("experiments/claude/outputs/r30/t4_errors.npz")
dark = (st["contrast"] <= np.quantile(st["contrast"], 0.1)) | (st["brightness"] <= np.quantile(st["brightness"], 0.1))
pi = np.bincount(y, minlength=5) / len(y)
wb = (0.2 / pi)[y]
bal = lambda pred, m: f1_score(y[m], pred[m], average="macro", sample_weight=wb[m])
SCORE = ALL5 + ["dark/low-contrast"]
VIEW_OF = {**{v: v for v in ALL5}, "dark/low-contrast": "full"}
for tag in TAGS:
    F5 = [f for f in range(5) if (V34 / f"{tag}_fold{f}.npz").exists()]
    if len(F5) < 5:
        print(f"{tag}: transform predictions for folds {F5} only; scoring needs all five")
        continue
    Q = {k: np.zeros((len(y), 5)) for k in [f"{v}|{t}" for v in ALL5 for t in TTAS]}
    for f, va in enumerate(folds):
        z = np.load(V34 / f"{tag}_fold{f}.npz")
        for k in Q:
            Q[k][va] = z[k]
    if tag in V20:
        print(f"{tag}: id transform vs the cached views, max |dp| " + ", ".join(f"{v} {np.abs(Q[f'{v}|id'] - P[tag][v]).max():.4f}" for v in ALL5))
    others = {t: w for t, w in V23_W.items() if t != "dinov2_v2"}

    def labels(ttas):
        out = {}
        for v in ALL5:
            p = (sum(w * P[t][v] for t, w in others.items()) + 8.0 * np.mean([Q[f"{v}|{t}"] for t in ttas], 0)) / (sum(others.values()) + 8.0)
            out[v] = (p * pi ** -0.75).argmax(1)
        return out

    def delta(lab, base, k, m):
        mm = m & dark if k == "dark/low-contrast" else m
        return bal(lab[VIEW_OF[k]], mm) - bal(base[VIEW_OF[k]], mm)

    base = labels(["id"])
    allm = np.ones(len(y), bool)
    print(f"\n===== {tag} ({'the v23 member' if tag in V20 else 'in place of dinov2_v2 at weight 8'}); base (id): "
          + ", ".join(f"{k} {bal(base[VIEW_OF[k]], dark if k == 'dark/low-contrast' else allm):.4f}" for k in SCORE)
          + f"; dark / low-contrast frames {dark.sum()}")
    print("member alone, argmax accuracy on the full view / on the dark frames: "
          + ", ".join(f"{t} {np.mean(Q[f'full|{t}'].argmax(1) == y):.4f} / {np.mean(Q[f'full|{t}'].argmax(1)[dark] == y[dark]):.4f}" for t in TTAS))
    labs = {n: labels(s) for n, s in SETS.items()}

    def report(name, lab):
        d = {k: delta(lab, base, k, allm) for k in SCORE}
        pdk = [delta(lab, base, "dark/low-contrast", fo == f) for f in range(5)]
        geo = np.mean([d[v] for v in ("zoom70", "centre_grey", "grid75")])
        ok = d["dark/low-contrast"] > 0 and sum(p > 0 for p in pdk) >= 3 and d["full"] >= -0.0005 and geo >= -0.0005
        print(f"  {name:40s} " + ", ".join(f"{k} {d[k]:+.4f}" for k in SCORE) + f" | dark by fold {' '.join(f'{p:+.4f}' for p in pdk)} "
              f"| zoom / HUD / grid mean {geo:+.4f} -> {'PASS' if ok else 'fail'}", flush=True)

    ens = {v: (sum(w * P[t][v] for t, w in others.items()) + 8.0 * Q[f"{v}|id"]) / (sum(others.values()) + 8.0) for v in ALL5}
    ent = {v: -(np.clip(ens[v], 1e-9, 1) * np.log(np.clip(ens[v], 1e-9, 1))).sum(1) for v in ALL5}

    def labels_gated(ttas, q):
        thr = np.quantile(ent["full"], q)
        out = {}
        for v in ALL5:
            m = np.where((ent[v] > thr)[:, None], np.mean([Q[f"{v}|{t}"] for t in ttas], 0), Q[f"{v}|id"])
            p = (sum(w * P[t][v] for t, w in others.items()) + 8.0 * m) / (sum(others.values()) + 8.0)
            out[v] = (p * pi ** -0.75).argmax(1)
        return out

    print(f"  ensemble entropy: full-view percentiles 80 / 90 / 95 = " + " / ".join(f"{np.quantile(ent['full'], q):.3f}" for q in GATE_Q)
          + "; share of images above the 90th-percentile threshold per view: " + ", ".join(f"{v} {np.mean(ent[v] > np.quantile(ent['full'], .9)):.3f}" for v in ALL5)
          + f"; dark / low-contrast frames {np.mean(ent['full'][dark] > np.quantile(ent['full'], .9)):.3f}")
    for gn, gs in GATED.items():
        for q in GATE_Q:
            labs[f"gated {gn}, entropy > p{int(q * 100)}"] = labels_gated(gs, q)
    for n, lab in labs.items():
        report(n, lab)
    nested = {v: base[v].copy() for v in ALL5}
    picks = []
    for f in range(5):
        m = fo != f
        ok = [n for n in labs if delta(labs[n], base, "full", m) >= -0.0005]
        best = max(ok, key=lambda n: delta(labs[n], base, "dark/low-contrast", m)) if ok else None
        if best is not None and delta(labs[best], base, "dark/low-contrast", m) <= 0:
            best = None
        picks.append(best or "id")
        if best:
            for v in ALL5:
                nested[v][fo == f] = labs[best][v][fo == f]
    report("nested choice of the set", nested)
    print(f"  nested picks: {picks}")
