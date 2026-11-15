"""Round 29, Task 4: 4-view multi-scale TTA on the out-of-fold predictions (training images only).

    python eg_r29_t4_tta4.py [tag ...]        (default: the nine v21 members, DINOv2 and EVA02 first)

The TAs allow test-time augmentation (2026-11-15). The pipeline averages 2 views (original + h-flip). Candidate: 4 views =
original + h-flip + central 85% zoom + its h-flip. Every fold model predicts its own validation images under the four views,
separately (written to outputs/r29_tta/<tag>_fold<f>.npz), so any weighting can be scored afterwards:
    p(w) = (1 - w) * mean(original, flip) + w * mean(zoom85, zoom85 flip);  w = 0.5 is the plain 4-view mean.
Scored on all 31,546 training images out of fold, at the TRAINING class mix (no re-weighting, no synthetic views: the TA ruling
of 2026-11-15 rules out validation designed from test data), macro-F1 under plain argmax and under the pipeline's rule
argmax p * pi^-0.75 (pi = training shares). Ensemble = v21: weighted mean of the nine members (DINOv2 x3, EVA02 x2), then
0.9 * mean + 0.1 * frozen-feature probe (the probe has no image TTA and is unchanged).
Acceptance (fixed before the run): mean gain > 0 and better in >= 4/5 folds; w = 0.5 is the candidate, w chosen nested from
{0.25, 0.5} is reported alongside.
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

ARCH = {"dinov2_v2": ("vit_base_patch14_reg4_dinov2.lvd142m", (126, 238)),
        "eva02B": ("eva02_base_patch14_448.mim_in22k_ft_in22k_in1k", (126, 238)),
        "cnxt": ("convnext_tiny.in12k_ft_in1k", (160, 320)),
        "deit3": ("deit3_small_patch16_384.fb_in22k_ft_in1k", (192, 384)),
        "cnxt_hud": ("convnext_tiny.in12k_ft_in1k", (160, 320)),
        "cnxt22k_hud": ("convnext_tiny.fb_in22k", (160, 320)),
        "effv2_hud": ("tf_efficientnetv2_s.in21k_ft_in1k", (160, 320)),
        "deit3_hud": ("deit3_small_patch16_384.fb_in22k_ft_in1k", (192, 384)),
        "cnxtS_hud": ("convnext_small.in12k_ft_in1k", (160, 320))}
V21 = ["cnxt", "deit3", "cnxt_hud", "cnxt22k_hud", "effv2_hud", "deit3_hud", "cnxtS_hud", "dinov2_v2", "eva02B"]
MEMBER_W = {"dinov2_v2": 3.0, "eva02B": 2.0}
TAGS = sys.argv[1:] or list(ARCH)
OUT = Path("experiments/claude/outputs/r29_tta")
OUT.mkdir(parents=True, exist_ok=True)
ZOOM = 0.85


def zoom(x, f):
    th = torch.zeros(x.shape[0], 2, 3, device=x.device)
    th[:, 0, 0], th[:, 1, 1] = f, f
    return F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), mode="bilinear", padding_mode="border", align_corners=False)


train, _ = task4.load_frames()
y = train.label.values
folds = [np.load(PRED_DIR / f"t4_cnxt_fold{f}.npz")["val_idx"] for f in range(5)]
for tag in TAGS:
    name, size = ARCH[tag]
    for f, va in enumerate(folds):
        out = OUT / f"{tag}_fold{f}.npz"
        if out.exists():
            continue
        z = np.load(PRED_DIR / f"t4_{tag}_fold{f}.npz")
        assert (z["val_idx"] == va).all(), f"{tag} fold {f}: rows differ from the pipeline folds"
        m = task4.create_model(name, pretrained=False, img_size=size).cuda().to(memory_format=torch.channels_last)
        m.load_state_dict(torch.load(MODEL_DIR / f"t4_{tag}_fold{f}.pt", map_location="cuda"))
        m.eval()
        arr = task4.load_uint8(list(train.path.values[va]))
        res = {k: [] for k in ("orig", "flip", "z85", "z85flip")}
        with torch.no_grad():
            for i in range(0, len(arr), 128):
                x0 = task4.to_gpu_batch(arr[i:i + 128], size)
                xz = zoom(x0, ZOOM)
                for k, x in (("orig", x0), ("flip", x0.flip(-1)), ("z85", xz), ("z85flip", xz.flip(-1))):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        res[k].append(m(task4.normalize(x).to(memory_format=torch.channels_last)).float().softmax(-1).cpu().numpy())
        res = {k: np.concatenate(v) for k, v in res.items()}
        gap = np.abs(0.5 * (res["orig"] + res["flip"]) - z["val_probs"]).max()
        np.savez(out, **res)
        print(f"{tag} fold {f}: 4 views written; max |2-view - cached val_probs| = {gap:.4f}", flush=True)
        del m, arr
        torch.cuda.empty_cache()

# ------------------------------------------------------------------ scoring (CPU)
have = [t for t in V21 if all((OUT / f"{t}_fold{f}.npz").exists() for f in range(5))]
V = {t: {k: np.zeros((len(y), 5)) for k in ("orig", "flip", "z85", "z85flip")} for t in have}
for t in have:
    for f, va in enumerate(folds):
        z = np.load(OUT / f"{t}_fold{f}.npz")
        for k in V[t]:
            V[t][k][va] = z[k]
CACHED = {t: np.zeros((len(y), 5)) for t in V21}
for t in V21:
    for f, va in enumerate(folds):
        CACHED[t][va] = np.load(PRED_DIR / f"t4_{t}_fold{f}.npz")["val_probs"]
probe = np.zeros((len(y), 5))
for f, va in enumerate(folds):
    probe[np.load(PRED_DIR / f"t4_probe_fold{f}.npz")["val_idx"]] = np.load(PRED_DIR / f"t4_probe_fold{f}.npz")["val_probs"]
pi = np.bincount(y, minlength=5) / len(y)
fold_of = np.zeros(len(y), int)
for f, va in enumerate(folds):
    fold_of[va] = f


def member(t, w):
    # baseline for a member with fresh views = its re-computed 2-view mean (bf16 re-runs differ from the cache by <= ~0.01 per
    # probability), so the comparison isolates the TTA change; members without fresh views use the cached 2-view probabilities
    if t not in V:
        return CACHED[t]
    return (1 - w) * 0.5 * (V[t]["orig"] + V[t]["flip"]) + w * 0.5 * (V[t]["z85"] + V[t]["z85flip"])


def ensemble(wmap):
    s = sum(MEMBER_W.get(t, 1.0) for t in V21)
    p = sum(MEMBER_W.get(t, 1.0) * member(t, wmap.get(t, 0.0)) for t in V21) / s
    return 0.9 * p + 0.1 * probe


lab = {"argmax": lambda p: p.argmax(1), "rule a=0.75": lambda p: (p * pi ** -0.75).argmax(1)}
mf1 = lambda yy, pp: f1_score(yy, pp, average="macro")
rng = np.random.default_rng(0)
boots = [rng.integers(0, len(y), len(y)) for _ in range(500)]


def compare(pa, pb, rule):
    a, b = lab[rule](pa), lab[rule](pb)
    per = [mf1(y[fold_of == f], b[fold_of == f]) - mf1(y[fold_of == f], a[fold_of == f]) for f in range(5)]
    d = [mf1(y[i], b[i]) - mf1(y[i], a[i]) for i in boots]
    return mf1(y, b) - mf1(y, a), np.quantile(d, [0.025, 0.975]), per, np.mean(a != b)


def show(label, pa, pb):
    for rule in lab:
        tot, ci, per, ch = compare(pa, pb, rule)
        print(f"  {label:58s} {rule:11s} {tot:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]  folds +{sum(p > 0 for p in per)}/5 "
              f"({' '.join(f'{p:+.4f}' for p in per)})  labels changed {ch:.2%}", flush=True)


print(f"\nOOF macro-F1 at the training mix, {len(y)} images; members with 4 views: {have}")
print("single members, argmax: 2 views (pipeline) -> 4 views (w = 0.5) | w = 0.25 | zoom85 pair alone")
for t in have:
    s2 = mf1(y, member(t, 0.0).argmax(1))
    s4, s25, sz = (mf1(y, member(t, w).argmax(1)) for w in (0.5, 0.25, 1.0))
    print(f"  {t:12s} {s2:.4f} (cache {mf1(y, CACHED[t].argmax(1)):.4f}) -> {s4:.4f} ({s4 - s2:+.4f}) | {s25:.4f} ({s25 - s2:+.4f}) | {sz:.4f}")

base = ensemble({})
for rule in lab:
    cached_ens = 0.9 * sum(MEMBER_W.get(t, 1.0) * CACHED[t] for t in V21) / sum(MEMBER_W.get(t, 1.0) for t in V21) + 0.1 * probe
    print(f"v21 ensemble (2 views everywhere), {rule}: {mf1(y, lab[rule](base)):.4f} (from the cached probabilities {mf1(y, lab[rule](cached_ens)):.4f})")
print("\nensemble deltas against v21 (paired bootstrap 95% CI, 500 resamples; per-fold deltas)")
vit = {t: 0.5 for t in ("dinov2_v2", "eva02B") if t in have}
allm = {t: 0.5 for t in have}
show("4 views on DINOv2 + EVA02 (w = 0.5)", base, ensemble(vit))
show("4 views on DINOv2 + EVA02 (w = 0.25)", base, ensemble({t: 0.25 for t in vit}))
if len(have) == len(V21):
    show("4 views on all nine members (w = 0.5)", base, ensemble(allm))
    show("4 views on all nine members (w = 0.25)", base, ensemble({t: 0.25 for t in have}))
    # w chosen nested: on 4 folds pick w in {0.25, 0.5} (all nine members), score the 5th
    for rule in lab:
        cand = {w: lab[rule](ensemble({t: w for t in have})) for w in (0.25, 0.5)}
        a = lab[rule](base)
        per, chosen = [], []
        for f in range(5):
            inn = fold_of != f
            w_best = max(cand, key=lambda w: mf1(y[inn], cand[w][inn]) - mf1(y[inn], a[inn]))
            chosen.append(w_best)
            per.append(mf1(y[fold_of == f], cand[w_best][fold_of == f]) - mf1(y[fold_of == f], a[fold_of == f]))
        print(f"  {'all nine, w nested from {0.25, 0.5}':58s} {rule:11s} mean {np.mean(per):+.4f}  folds +{sum(p > 0 for p in per)}/5 "
              f"({' '.join(f'{p:+.4f}' for p in per)})  chosen {chosen}")
