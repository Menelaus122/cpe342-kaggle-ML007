"""Round 39, Task 4: rotation and greyscale test-time augmentation for all eight members (training images only).

    python gs_r39_t4_tta_rotgray.py

From a suggestion list (2026-11-17 evening; the user asked for this check): TTA = mean over the original, rotated +-8 degrees and a
greyscale copy. Zoom TTA was tested before (rounds 29 and 36: no gain), so only rotation and greyscale are new. TTA is allowed (TA
ruling of 2026-11-15); everything here is chosen on training images.
GPU phase (resumable; outputs/r39_tta/<tag>_fold<f>.npz, keys "<view>|<transform>"): every fold model of the eight v23 members
predicts its own validation images under the views full, zoom70 and HUD hidden (centre_grey; as em_r30_t4_hardviews.py), each view
then passed through rot+8 / rot-8 (rotation about the centre with black corners, aspect-correct, as the training affine) and grey
(luminance 0.299 / 0.587 / 0.114 in all three channels); h-flip averaged as in the pipeline. The original comes from the cached
views (outputs/r19_views), i.e. exactly the v29 inputs.
Ensemble: v23 member weights, then 0.65 x members + 0.35 x the seven-backbone MLP probe (v29; the probe is unchanged by TTA), v17 rule,
balanced class mix; real dark / low-contrast frames = lowest decile of contrast or brightness (full view).
Candidates (fixed before the run; the TTA mean replaces each member's probabilities, all members alike):
  T1 rot   mean(orig, rot+8, rot-8)
  T2 grey  mean(orig, grey)
  T3 all   mean(orig, rot+8, rot-8, grey)   (the suggestion's set without the zoom, which was tested before)
  N  nested over T1-T3 (per held-out fold, the best mean of the zoom70 / HUD-hidden / dark deltas on the other four folds among those
     with full >= -0.0005 there; none positive: the fold keeps v29).
Acceptance (user's rule against v29, applied to N): zoom70, HUD hidden and the real dark / low-contrast frames all > 0,
full >= -0.0005, zoom70 / HUD-hidden mean better in >= 3/5 folds. If N passes, the pipeline candidate is the set with the best
criterion on all five folds (test inference with that set would follow). Per-member effects are printed for information.
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

import common  # noqa: F401  (chdir to the repo root)

sys.path.insert(0, "scripts")
from src import task4  # noqa: E402
from src.config import MODEL_DIR, PRED_DIR  # noqa: E402
import train_task4_probe as P0  # noqa: E402

ARCH = {"cnxt": ("convnext_tiny.in12k_ft_in1k", (160, 320)), "deit3": ("deit3_small_patch16_384.fb_in22k_ft_in1k", (192, 384)),
        "cnxt_hud": ("convnext_tiny.in12k_ft_in1k", (160, 320)), "cnxt22k_hud": ("convnext_tiny.fb_in22k", (160, 320)),
        "effv2_hud": ("tf_efficientnetv2_s.in21k_ft_in1k", (160, 320)), "deit3_hud": ("deit3_small_patch16_384.fb_in22k_ft_in1k", (192, 384)),
        "cnxtS_hud": ("convnext_small.in12k_ft_in1k", (160, 320)), "dinov2_v2": ("vit_base_patch14_reg4_dinov2.lvd142m", (126, 238))}
V20 = list(ARCH)
HUDM = ["cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud"]
W = {t: (8.0 if t == "dinov2_v2" else 3.0 if t in HUDM else 1.0) for t in V20}
VIEWS = ["full", "zoom70", "centre_grey"]
TRANS = ["rot+8", "rot-8", "grey"]
OUT = Path("experiments/claude/outputs/r39_tta")
OUT.mkdir(parents=True, exist_ok=True)
V19 = Path("experiments/claude/outputs/r19_views")


def zoom(x, f):
    th = torch.zeros(x.shape[0], 2, 3, device=x.device)
    th[:, 0, 0], th[:, 1, 1] = f, f
    return F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), mode="bilinear", padding_mode="border", align_corners=False)


def centre_grey(x):
    H, W_ = x.shape[2:]
    m = torch.zeros(1, 1, H, W_, device=x.device)
    m[..., H // 5:4 * H // 5, W_ // 5:4 * W_ // 5] = 1
    return x * m + 0.5 * (1 - m)


def rot(x, deg):
    # rotation in pixel space about the centre, expressed in normalised coordinates of an H x W frame; black outside
    H, W_ = x.shape[2:]
    a = math.radians(deg)
    th = torch.zeros(x.shape[0], 2, 3, device=x.device)
    th[:, 0, 0], th[:, 0, 1] = math.cos(a), -math.sin(a) * H / W_
    th[:, 1, 0], th[:, 1, 1] = math.sin(a) * W_ / H, math.cos(a)
    return F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), mode="bilinear", padding_mode="zeros", align_corners=False)


def grey(x):
    g = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
    return g.unsqueeze(1).expand_as(x).contiguous()


VIEW_FN = {"full": lambda x: x, "zoom70": lambda x: zoom(x, 0.7), "centre_grey": centre_grey}
TRANS_FN = {"rot+8": lambda x: rot(x, 8), "rot-8": lambda x: rot(x, -8), "grey": grey}

train, _ = task4.load_frames()
y = train.label.values.astype(np.int64)
folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
fo = np.zeros(len(y), int)
for f, va in enumerate(folds):
    fo[va] = f

# ---- GPU phase
for tag in V20:
    name, size = ARCH[tag]
    for f, va in enumerate(folds):
        out = OUT / f"{tag}_fold{f}.npz"
        if out.exists():
            continue
        m = task4.create_model(name, pretrained=False, img_size=size).cuda().to(memory_format=torch.channels_last)
        m.load_state_dict(torch.load(MODEL_DIR / f"t4_{tag}_fold{f}.pt", map_location="cuda"))
        m.eval()
        arr = task4.load_uint8(list(train.path.values[va]))
        res = {f"{v}|{t}": [] for v in VIEWS for t in TRANS}
        with torch.no_grad():
            for i in range(0, len(arr), 128):
                x0 = task4.to_gpu_batch(arr[i:i + 128], size)
                for v in VIEWS:
                    xv = VIEW_FN[v](x0)
                    for t in TRANS:
                        x = TRANS_FN[t](xv)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            p = 0.5 * (m(task4.normalize(x).to(memory_format=torch.channels_last)).float().softmax(-1)
                                       + m(task4.normalize(x.flip(-1)).to(memory_format=torch.channels_last)).float().softmax(-1))
                        res[f"{v}|{t}"].append(p.cpu().numpy())
        np.savez(out, **{k: np.concatenate(r) for k, r in res.items()})
        print(f"{tag} fold {f}: written", flush=True)
        del m, arr
        torch.cuda.empty_cache()

# ---- member probabilities: original (cached v29 inputs) and the three transforms
P = {t: {f"{v}|{k}": np.zeros((len(y), 5)) for v in VIEWS for k in ["orig"] + TRANS} for t in V20}
for t in V20:
    for f, va in enumerate(folds):
        z19, zt = np.load(V19 / f"{t}_fold{f}.npz"), np.load(OUT / f"{t}_fold{f}.npz")
        for v in VIEWS:
            P[t][f"{v}|orig"][va] = z19[v]
            for k in TRANS:
                P[t][f"{v}|{k}"][va] = zt[f"{v}|{k}"]

# ---- the v29 probe on the three views (seven-backbone MLP, out of fold; cached after the first run)
pcache = Path("experiments/claude/outputs/r39/p7_probe_views.npz")
if pcache.exists():
    PR = dict(np.load(pcache))
else:
    tags7 = [b[2] for b in P0.BACKBONES]
    X = {}
    for v in VIEWS:
        X[v] = np.hstack([(np.load(PRED_DIR / f"t4_feat_{tg}.npz")["train"] if v == "full" else np.load(PRED_DIR / f"t4_feat_views_{tg}.npz")[v]).astype(np.float32) for tg in tags7])
    PR = {v: np.zeros((len(y), 5)) for v in VIEWS}
    for f, va in enumerate(folds):
        tr = np.where(fo != f)[0]
        mu, sd = X["full"][tr].mean(0), X["full"][tr].std(0) + 1e-6
        prep = lambda Q: ((Q - mu) / sd).astype(np.float32)
        A, Y = np.vstack([prep(X[v][tr]) for v in VIEWS]), np.tile(y[tr], len(VIEWS))
        preds = [P0.fit_mlp(A, Y, s) for s in P0.MLP_SEEDS]
        del A
        torch.cuda.empty_cache()
        for v in VIEWS:
            PR[v][va] = np.mean([q(prep(X[v][va])) for q in preds], 0)
    np.savez(pcache, **PR)
    del X

st = np.load("experiments/claude/outputs/r30/t4_errors.npz")
dark = (st["contrast"] <= np.quantile(st["contrast"], 0.1)) | (st["brightness"] <= np.quantile(st["brightness"], 0.1))
pi = np.bincount(y, minlength=5) / len(y)
wb = (0.2 / pi)[y]
bal = lambda pred, m: f1_score(y[m], pred[m], average="macro", sample_weight=wb[m])
rule = lambda p: (p * pi ** -0.75).argmax(1)
allm = np.ones(len(y), bool)
SC = ["full", "zoom70", "centre_grey", "dark/low-contrast"]
VIEW_OF = {"full": "full", "zoom70": "zoom70", "centre_grey": "centre_grey", "dark/low-contrast": "full"}
msk = lambda k, m=allm: m & dark if k == "dark/low-contrast" else m
SETS = {"orig": ["orig"], "T1 rot": ["orig", "rot+8", "rot-8"], "T2 grey": ["orig", "grey"], "T3 all": ["orig", "rot+8", "rot-8", "grey"]}


def ens(ks, members=V20):
    out = {}
    for v in VIEWS:
        mem = sum(W[t] * np.mean([P[t][f"{v}|{k}"] for k in ks], 0) for t in members) / sum(W[t] for t in members)
        out[v] = rule(0.65 * mem + 0.35 * PR[v])
    return out


ref = ens(["orig"])
print("v29 reference: " + ", ".join(f"{k} {bal(ref[VIEW_OF[k]], msk(k)):.4f}" for k in SC), flush=True)
print("\nsingle transforms alone, per member (full / zoom70 / HUD hidden, balanced macro-F1; information):")
for t in V20:
    print(f"  {t:12s} " + " | ".join(f"{k} " + "/".join(f"{bal(rule(P[t][f'{v}|{k}']), allm):.4f}" for v in VIEWS) for k in ["orig"] + TRANS), flush=True)


def deltas(lab, m=allm):
    return {k: bal(lab[VIEW_OF[k]], msk(k, m)) - bal(ref[VIEW_OF[k]], msk(k, m)) for k in SC}


def crit(lab, m):
    d = deltas(lab, m)
    return np.mean([d["zoom70"], d["centre_grey"], d["dark/low-contrast"]]) if d["full"] >= -0.0005 else -np.inf


def report(name, lab):
    d = deltas(lab)
    pf = [np.mean([bal(lab[v], fo == f) - bal(ref[v], fo == f) for v in ("zoom70", "centre_grey")]) for f in range(5)]
    pdk = [bal(lab["full"], (fo == f) & dark) - bal(ref["full"], (fo == f) & dark) for f in range(5)]
    ok = d["zoom70"] > 0 and d["centre_grey"] > 0 and d["dark/low-contrast"] > 0 and d["full"] >= -0.0005 and sum(p > 0 for p in pf) >= 3
    print(f"  {name}\n    " + ", ".join(f"{k} {d[k]:+.4f}" for k in SC) + f" | zoom / HUD by fold {' '.join(f'{p:+.4f}' for p in pf)} | "
          f"dark by fold {' '.join(f'{p:+.4f}' for p in pdk)}\n    -> {'PASS' if ok else 'fail'}", flush=True)
    return ok


C = {s: ens(ks) for s, ks in SETS.items() if s != "orig"}
print("\nCandidates against v29:")
for s, lab in C.items():
    report(s, lab)
outN, picks = {v: ref[v].copy() for v in VIEWS}, []
for f in range(5):
    sc = {s: crit(lab, fo != f) for s, lab in C.items()}
    s = max(sc, key=sc.get) if max(sc.values()) > 0 else None
    picks.append(s)
    if s is not None:
        for v in VIEWS:
            outN[v][fo == f] = C[s][v][fo == f]
okN = report(f"N nested over T1-T3; picks {picks}", outN)
best = max(C, key=lambda s: crit(C[s], allm))
print(f"\n-> N {'PASSES' if okN else 'fails'}; the set with the best criterion on all five folds: {best} (pipeline candidate only if N passes)")
print("\nPer member, TTA set on that member only (others original), against v29 (information):")
for t in V20:
    for s, ks in SETS.items():
        if s == "orig":
            continue
        out = {}
        for v in VIEWS:
            mem = sum(W[u] * (np.mean([P[u][f"{v}|{k}"] for k in ks], 0) if u == t else P[u][f"{v}|orig"]) for u in V20) / sum(W.values())
            out[v] = rule(0.65 * mem + 0.35 * PR[v])
        d = deltas(out)
        print(f"  {t:12s} {s:8s} " + ", ".join(f"{k} {d[k]:+.4f}" for k in SC), flush=True)
print("done", flush=True)
